"""Local LLM integration via Ollama for tagging and (later) generating quiz content."""
import html
import json
import logging
import os
import re
import time

logger = logging.getLogger(__name__)

# Retry policy for transient upstream errors (e.g. Ollama cloud 5xx).
_CHAT_MAX_ATTEMPTS = 4
_CHAT_BACKOFF_BASE_SECS = 1.0


def _chat_with_retry(client, **chat_kwargs):
    """Call client.chat() with exponential-backoff retries on transient 5xx errors.

    Retries only on HTTP 5xx (server-side hiccups); 4xx errors raise immediately
    since those indicate a client-side problem (auth, bad model, malformed input).
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            return client.chat(**chat_kwargs)
        except Exception as e:
            status_code = getattr(e, "status_code", None)
            transient = status_code is not None and 500 <= status_code < 600
            if not transient or attempt >= _CHAT_MAX_ATTEMPTS:
                raise
            delay = _CHAT_BACKOFF_BASE_SECS * (2 ** (attempt - 1))
            logger.info(
                f"Transient {status_code} from LLM; retrying in {delay:.1f}s "
                f"(attempt {attempt + 1}/{_CHAT_MAX_ATTEMPTS})"
            )
            time.sleep(delay)


DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b")
DEFAULT_AUDIO_MODEL = os.environ.get("OLLAMA_AUDIO_MODEL", "gemma4:e4b")
# Used for instructor-side text tasks that never see student data (tagging quiz
# questions, generating open-ended follow-up questions). A larger cloud model
# is the better fit here since privacy constraints don't apply.
DEFAULT_TEXT_MODEL = os.environ.get("OLLAMA_TEXT_MODEL", "gemini-3-flash-preview")

# Cloud host for Ollama's hosted models. Used when OLLAMA_API_KEY is set, since
# models like gemini-3-flash-preview are not available on a local Ollama server.
OLLAMA_CLOUD_HOST = "https://ollama.com"


def _make_client(cloud=False):
    """Build an ollama.Client — cloud=True points at ollama.com with OLLAMA_API_KEY."""
    import ollama
    if cloud:
        api_key = os.environ.get("OLLAMA_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OLLAMA_API_KEY is required for cloud-hosted text models. "
                "Re-run ./configure.sh (or add `export OLLAMA_API_KEY=...` to set_env.sh) and re-source."
            )
        return ollama.Client(
            host=OLLAMA_CLOUD_HOST,
            headers={"Authorization": "Bearer " + api_key},
        )
    return ollama.Client()


_TAG_SYSTEM_PROMPT = (
    "You are a concise topic tagger for university quiz questions. "
    "Given a question, respond with 1 to 3 short concept tags that describe "
    "the core idea(s) being tested. Tags must be lowercase, comma-separated, "
    "each 1-4 words. Respond with ONLY the tags on a single line — no preamble, "
    "no numbering, no explanation, no quotes."
)

_CLASSIFY_SYSTEM_PROMPT = (
    "You are an expert university instructor deciding how to assess student understanding. "
    "Given a topic and details from an existing quiz question, decide whether the best way "
    "to assess a student's understanding of this material is:\n"
    "- \"explain\" — the student gives a verbal explanation in their own words, OR\n"
    "- \"draw\" — the student draws a diagram, figure, or visual representation by hand.\n\n"
    "Choose \"draw\" when the concept is inherently visual or spatial — for example: "
    "data structures (trees, linked lists, graphs), memory layouts, network topologies, "
    "circuit diagrams, process flows, state machines, architectural diagrams, or anything "
    "where a picture would communicate the idea more clearly than words alone.\n\n"
    "Choose \"explain\" when the concept is better communicated verbally — for example: "
    "definitions, trade-offs, comparisons, reasoning about behavior, algorithm logic, "
    "or conceptual understanding that doesn't map naturally to a picture.\n\n"
    "Respond with ONLY the single word \"explain\" or \"draw\" — nothing else."
)

_EXPLAIN_SYSTEM_PROMPT = (
    "You are an expert university instructor designing oral exam questions. "
    "Given a topic and details from an existing quiz question, create THREE distinct "
    "open-ended questions that a student would answer verbally. Each question should:\n"
    "- Begin with \"Explain\" and require the student to explain a concept or idea clearly "
    "in their own words\n"
    "- Be answerable in under 1 minute of speaking\n"
    "- Test understanding, not memorization — the student should demonstrate they grasp "
    "the underlying idea, not just recall a definition\n"
    "- Be self-contained (a student should understand what is being asked without seeing "
    "the original quiz question)\n"
    "- NOT be a yes/no question or a question that can be answered in one word\n\n"
    "The three questions must cover DIFFERENT angles, framings, or sub-aspects of the concept — "
    "do not reword the same question three times.\n\n"
    "Respond with ONLY the three questions, numbered \"1.\", \"2.\", \"3.\", one per line. "
    "No preamble, no explanation, no quotation marks."
)

_DRAW_SYSTEM_PROMPT = (
    "You are an expert university instructor designing visual assessment questions. "
    "Given a topic and details from an existing quiz question, create THREE distinct "
    "questions that ask a student to draw a diagram or figure by hand. Each question should:\n"
    "- Begin with \"Draw a diagram\" or \"Draw a figure\" and clearly describe what the "
    "student should illustrate\n"
    "- Be completable in under 2 minutes of drawing\n"
    "- Test understanding of structure, relationships, or processes — not artistic skill\n"
    "- Be self-contained (a student should understand what is being asked without seeing "
    "the original quiz question)\n"
    "- Specify what key elements or labels should appear in the diagram\n\n"
    "The three questions must cover DIFFERENT angles, framings, or sub-aspects of the concept — "
    "do not reword the same question three times.\n\n"
    "Respond with ONLY the three questions, numbered \"1.\", \"2.\", \"3.\", one per line. "
    "No preamble, no explanation, no quotation marks."
)

_RUBRIC_COMMON_SCHEMA = (
    '  "canonical_answer": "<concise 1-2 sentence model answer that represents a strong response>",\n'
    '  "model_answer": "<an 80-120 word fully-developed exemplar of an A-grade student response, '
    'written in the register a real student would use — informal but substantive — describing '
    'every concept the response should cover>",\n'
    '  "pass_criteria": ["<2-5 concrete must-haves a passing response must demonstrate>"],\n'
    '  "acceptable_alternatives": ["<equivalent framings, synonyms, or alternative correct approaches that should still count as passing>"],\n'
    '  "common_misconceptions": ["<wrong-but-plausible answers that should NOT count as passing, each with a brief reason>"],\n'
    '  "fatal_errors": ["<statements or content so incorrect they force an automatic fail regardless of other criteria>"]'
)

_RUBRIC_EXPLAIN_SYSTEM_PROMPT = (
    "You are an expert university instructor producing a structured grading rubric for an "
    "open-ended verbal-explanation follow-up question. The rubric will be consumed by a "
    "smaller AI grader and by a human instructor, so it must be concrete, specific to the "
    "question, and free of filler.\n\n"
    "Return ONLY a single JSON object with exactly the schema below — no prose before or "
    "after, no markdown code fences, no commentary.\n\n"
    "{\n" +
    _RUBRIC_COMMON_SCHEMA + "\n" +
    "}\n\n"
    "Rules:\n"
    "- Each list must have 2-5 items. Each item must be a short string (ideally under 20 words).\n"
    "- Do not repeat the same idea across lists. Pass criteria are what MUST be present; "
    "alternatives are equivalent framings; misconceptions are wrong answers; fatal errors "
    "are auto-fails.\n"
    "- canonical_answer is a tight 1-2 sentence summary. model_answer is longer (80-120 words), "
    "phrased the way a strong student would actually speak — informal but substantive — and must "
    "cover every concept the pass_criteria expect to see.\n"
    "- Focus on conceptual substance, not stylistic polish. The student is speaking, so "
    "informal language is fine — criteria should be about understanding, not phrasing.\n"
    "- All strings must be valid JSON (escape quotes, no trailing commas)."
)

_RUBRIC_DRAW_SYSTEM_PROMPT = (
    "You are an expert university instructor producing a structured grading rubric for an "
    "open-ended hand-drawn-diagram follow-up question. The rubric will be consumed by a "
    "smaller AI grader and by a human instructor, so it must be concrete, specific to the "
    "question, and free of filler.\n\n"
    "Return ONLY a single JSON object with exactly the schema below — no prose before or "
    "after, no markdown code fences, no commentary.\n\n"
    "{\n" +
    _RUBRIC_COMMON_SCHEMA + ",\n" +
    '  "required_visual_elements": ["<key shapes, labels, arrows, or structural relationships that must appear in the drawing>"]\n' +
    "}\n\n"
    "Rules:\n"
    "- Each list must have 2-5 items. Each item must be a short string (ideally under 20 words).\n"
    "- Do not repeat the same idea across lists. Pass criteria cover correctness of structure "
    "or relationships; required_visual_elements are concrete things that should literally "
    "appear in the picture (nodes, edges, labels, arrows, regions).\n"
    "- canonical_answer is a tight 1-2 sentence summary. model_answer is an 80-120 word prose "
    "description of what an A-grade student drawing would contain — every shape, label, arrow, "
    "and structural relationship the picture must show. Describe the picture, do not draw it.\n"
    "- Focus on structural correctness, not artistic quality. Hand-drawn, rough, or "
    "imperfectly-proportioned drawings are fine — criteria should be about the idea being "
    "communicated.\n"
    "- All strings must be valid JSON (escape quotes, no trailing commas)."
)

_ASSESSMENT_GUIDE_SYSTEM_PROMPT = (
    "You are an expert university instructor writing a short assessment guide for a specific "
    "open-ended follow-up question. The guide will be used by another instructor (or an AI "
    "grader) to decide whether a student's response demonstrates a reasonable understanding "
    "of the concept.\n\n"
    "Write 2–4 plain-prose sentences (no bullets, no markdown, no headings) that cover:\n"
    "- For an \"Explain\" question: the key concepts, terms, or keywords the student should "
    "mention; for a \"Draw\" question: the key elements, labels, or structural relationships "
    "that should appear in the drawing.\n"
    "- What a minimally-acceptable passing response looks like — the bar for \"pass\".\n"
    "- Common misconceptions, wrong framings, or missing pieces that should NOT count as "
    "passing.\n\n"
    "Keep it tight and practical — a grader should be able to read it in under 15 seconds. "
    "Refer to the student as \"the student\". Respond with ONLY the guide text on a single "
    "paragraph — no preamble, no label, no numbering, no quotation marks."
)


def _strip_html(text):
    """Reduce Canvas HTML question text to plain text suitable for prompting."""
    if not text:
        return ""

    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p\s*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _parse_tags(response):
    """Parse the model's single-line response into a clean list of 1-3 tags."""
    if not response:
        return []
    line = response.strip().splitlines()[0] if response.strip() else ""
    parts = [p.strip().strip("\"'").lower() for p in line.split(",")]
    seen = set()
    tags = []
    for p in parts:
        if p and p not in seen:
            seen.add(p)
            tags.append(p)
        if len(tags) == 3:
            break
    return tags


def _answer_labels(answers_json):
    """Extract short textual labels from the answers JSON blob, if any."""
    try:
        answers = json.loads(answers_json) if isinstance(answers_json, str) else (answers_json or [])
    except (ValueError, TypeError):
        return []
    labels = []
    for a in answers:
        if isinstance(a, dict):
            text = a.get("text") or a.get("html") or ""
            text = _strip_html(text)
            if text:
                labels.append(text)
    return labels


def _build_prompt(question_name, question_text, answers_json):
    """Build the user-side prompt for a single question."""
    clean_text = _strip_html(question_text)
    labels = _answer_labels(answers_json)
    parts = []
    if question_name:
        parts.append(f"Question name: {question_name}")
    if clean_text:
        parts.append(f"Question: {clean_text}")
    if labels:
        joined = " | ".join(labels[:6])
        parts.append(f"Answer choices: {joined}")
    parts.append("Tags:")
    return "\n".join(parts)


def tag_question(row, client, model):
    """Call the LLM once for one question row and return a list of tags."""
    prompt = _build_prompt(
        row.get("question_name"),
        row.get("question_text"),
        row.get("answers"),
    )
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": _TAG_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.2},
        )
        content = resp["message"]["content"]
        return _parse_tags(content)
    except Exception as e:
        logger.warning(f"LLM tagging failed for question {row.get('question_id')}: {e}")
        return []


def tag_questions(rows, model=None, progress=None):
    """Annotate each row dict in-place with a 'tags' key (comma-separated string).

    `progress` is an optional `(spin, spin_done)` callback pair from canvigator_utils;
    when provided, per-question status is shown as a single overwriting spinner line
    instead of one printed line per question.
    """
    try:
        import ollama  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "The 'ollama' package is required for --tag. Install with: pip install ollama"
        ) from e

    model = model or DEFAULT_TEXT_MODEL
    client = _make_client(cloud=True)
    spin_fn, spin_done_fn = progress if progress else (None, None)

    total = len(rows)
    print(f"Tagging {total} questions with cloud model '{model}'...")
    for i, row in enumerate(rows, start=1):
        label = row.get('question_name') or row.get('question_id')
        if spin_fn:
            spin_fn(i, f"Tagging [{i}/{total}] {label}")
        else:
            print(f"  [{i}/{total}] {label}")
        tags = tag_question(row, client, model)
        row["keywords"] = ", ".join(tags)
    if spin_done_fn:
        spin_done_fn(f"Tagged {total} questions")
    else:
        print("Tagging complete.")


def _parse_candidates(response, n=3):
    """Parse up to n candidate questions from a numbered LLM response."""
    if not response:
        return []
    candidates = []
    for line in response.strip().splitlines():
        stripped = re.sub(r'^\s*(?:\d+[\.\)]|[-*])\s+', '', line).strip().strip("\"'")
        if stripped:
            candidates.append(stripped)
        if len(candidates) == n:
            break
    return candidates


def _parse_question_mode(response):
    """Parse the classify response into 'explain' or 'draw', defaulting to 'explain'."""
    if not response:
        return "explain"
    token = response.strip().splitlines()[0].strip().lower().strip("\"'.,")
    if token == "draw":
        return "draw"
    return "explain"


def _build_classify_prompt(keywords, question_text, answers_json):
    """Build the user-side prompt for the classify (explain vs. draw) step."""
    clean_text = _strip_html(question_text)
    labels = _answer_labels(answers_json)
    parts = []
    if keywords:
        parts.append(f"Topic keywords: {keywords}")
    if clean_text:
        parts.append(f"Original question: {clean_text}")
    if labels:
        joined = " | ".join(labels[:6])
        parts.append(f"Answer choices: {joined}")
    parts.append("Best assessment mode (explain or draw):")
    return "\n".join(parts)


def _build_open_ended_prompt(keywords, question_text, answers_json, mode):
    """Build the user-side prompt for generating one open-ended question."""
    clean_text = _strip_html(question_text)
    labels = _answer_labels(answers_json)
    parts = []
    if keywords:
        parts.append(f"Topic keywords: {keywords}")
    if clean_text:
        parts.append(f"Original question: {clean_text}")
    if labels:
        joined = " | ".join(labels[:6])
        parts.append(f"Answer choices from the original: {joined}")
    if mode == "draw":
        parts.append("Visual assessment question (must start with \"Draw a diagram\" or \"Draw a figure\"):")
    else:
        parts.append("Oral explanation question (must start with \"Explain\"):")
    return "\n".join(parts)


def classify_question_mode(row, client, model):
    """Call the LLM to decide whether 'explain' or 'draw' is the better follow-up format."""
    prompt = _build_classify_prompt(
        row.get("keywords"),
        row.get("question_text"),
        row.get("answers"),
    )
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": _CLASSIFY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.1},
        )
        content = resp["message"]["content"]
        return _parse_question_mode(content)
    except Exception as e:
        logger.warning(f"LLM classify failed for question {row.get('question_id')}: {e}")
        return "explain"


def generate_open_ended_candidates(row, client, model, mode, n=3):
    """Call the LLM to generate n candidate open-ended questions of the given mode."""
    prompt = _build_open_ended_prompt(
        row.get("keywords"),
        row.get("question_text"),
        row.get("answers"),
        mode,
    )
    system_prompt = _DRAW_SYSTEM_PROMPT if mode == "draw" else _EXPLAIN_SYSTEM_PROMPT
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.7},
        )
        content = resp["message"]["content"].strip()
        return _parse_candidates(content, n=n)
    except Exception as e:
        logger.warning(f"LLM generation failed for question {row.get('question_id')}: {e}")
        return []


def _build_assessment_guide_prompt(keywords, original_question_text, answers_json, mode, open_ended_question):
    """Build the user-side prompt for generating an assessment guide."""
    clean_text = _strip_html(original_question_text)
    labels = _answer_labels(answers_json)
    parts = []
    if keywords:
        parts.append(f"Topic keywords: {keywords}")
    if clean_text:
        parts.append(f"Original quiz question: {clean_text}")
    if labels:
        joined = " | ".join(labels[:6])
        parts.append(f"Answer choices from the original: {joined}")
    parts.append(f"Question type: {'draw (hand-drawn diagram)' if mode == 'draw' else 'explain (verbal explanation)'}")
    parts.append(f"Open-ended question the student will answer: {open_ended_question}")
    parts.append("Assessment guide:")
    return "\n".join(parts)


def generate_assessment_guide(row, client, model, mode, open_ended_question):
    """Call the LLM to produce a short assessment guide for one open-ended question."""
    if not open_ended_question:
        return ""
    prompt = _build_assessment_guide_prompt(
        row.get("keywords"),
        row.get("question_text"),
        row.get("answers"),
        mode,
        open_ended_question,
    )
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": _ASSESSMENT_GUIDE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )
        content = resp["message"]["content"].strip()
        # Collapse any accidental newlines into spaces — guide should be a single paragraph.
        content = re.sub(r"\s*\n+\s*", " ", content).strip().strip('"\'')
        return content
    except Exception as e:
        logger.warning(f"LLM assessment-guide generation failed for question {row.get('question_id')}: {e}")
        return ""


def _empty_rubric(mode):
    """Return the canonical empty rubric shape for the given mode."""
    base = {
        'canonical_answer': '',
        'model_answer': '',
        'pass_criteria': [],
        'acceptable_alternatives': [],
        'common_misconceptions': [],
        'fatal_errors': [],
    }
    if mode == 'draw':
        base['required_visual_elements'] = []
    return base


def _parse_structured_rubric(response, mode):
    """Parse the JSON rubric from an LLM response, always returning a well-formed dict.

    Tolerates responses wrapped in ```json fences or with stray whitespace. On any
    parse failure returns the empty-shaped rubric so downstream callers don't need
    to defensive-check individual keys.
    """
    out = _empty_rubric(mode)
    if not response:
        return out

    text = response.strip()
    # Strip an opening fence like ```json or ```
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    # If the model emitted any prose before the JSON object, take the largest
    # {...} slice we can find.
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        logger.warning(f"Rubric JSON parse failed; raw response starts with: {response[:200]!r}")
        return out
    if not isinstance(data, dict):
        return out

    string_keys = {'canonical_answer', 'model_answer'}
    for key in out:
        val = data.get(key)
        if key in string_keys:
            out[key] = str(val).strip() if val else ''
        elif isinstance(val, list):
            out[key] = [str(v).strip() for v in val if v is not None and str(v).strip()]
    return out


def _build_structured_rubric_prompt(keywords, original_question_text, answers_json, mode, open_ended_question):
    """Build the user-side prompt for structured-rubric generation."""
    clean_text = _strip_html(original_question_text)
    labels = _answer_labels(answers_json)
    parts = []
    if keywords:
        parts.append(f"Topic keywords: {keywords}")
    if clean_text:
        parts.append(f"Original quiz question: {clean_text}")
    if labels:
        joined = " | ".join(labels[:6])
        parts.append(f"Answer choices from the original: {joined}")
    parts.append(f"Question type: {'draw (hand-drawn diagram)' if mode == 'draw' else 'explain (verbal explanation)'}")
    parts.append(f"Open-ended question the student will answer: {open_ended_question}")
    parts.append("Rubric JSON:")
    return "\n".join(parts)


def generate_structured_rubric(row, client, model, mode, open_ended_question):
    """Call the LLM to produce a structured JSON rubric for one open-ended question."""
    if not open_ended_question:
        return _empty_rubric(mode)
    prompt = _build_structured_rubric_prompt(
        row.get("keywords"),
        row.get("question_text"),
        row.get("answers"),
        mode,
        open_ended_question,
    )
    system_prompt = _RUBRIC_DRAW_SYSTEM_PROMPT if mode == 'draw' else _RUBRIC_EXPLAIN_SYSTEM_PROMPT
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )
        return _parse_structured_rubric(resp["message"]["content"], mode)
    except Exception as e:
        logger.warning(f"LLM structured-rubric generation failed for question {row.get('question_id')}: {e}")
        return _empty_rubric(mode)


_EXEMPLARS_EXPLAIN_SYSTEM_PROMPT = (
    "You are an expert university instructor producing few-shot calibration examples for an "
    "AI grader. Given a follow-up question and its rubric, write ONE realistic passing student "
    "response and ONE realistic failing student response, plus a one-sentence note on why each "
    "sits at the bar.\n\n"
    "Both responses must read like spoken student responses — informal language, hesitations, "
    "imperfect phrasing — NOT polished prose. Aim for 50-100 words each.\n\n"
    "The passing response should be a realistic minimum-bar pass: it covers the core concepts "
    "but is not perfect (think B/B+ student, not A+). The failing response should sound "
    "plausible — a student who tried but missed the point, exhibits a misconception, or hit a "
    "fatal error from the rubric — NOT obvious nonsense or a one-line shrug.\n\n"
    "Return ONLY a single JSON object with exactly this schema, no prose, no markdown fences:\n"
    "{\n"
    '  "exemplar_pass": "<50-100 word realistic spoken passing response>",\n'
    '  "exemplar_pass_note": "<one-sentence note on why this is at the passing bar>",\n'
    '  "exemplar_fail": "<50-100 word realistic spoken failing response>",\n'
    '  "exemplar_fail_note": "<one-sentence note on why this is at the failing bar>"\n'
    "}"
)

_EXEMPLARS_DRAW_SYSTEM_PROMPT = (
    "You are an expert university instructor producing few-shot calibration examples for an "
    "AI grader. Given a follow-up question and its rubric, write a prose description of ONE "
    "realistic passing student drawing and ONE realistic failing student drawing, plus a "
    "one-sentence note on why each sits at the bar.\n\n"
    "Describe the drawings — do not produce images. Each description should sound like an "
    "instructor narrating what they see on a piece of paper: shapes, labels, arrows, "
    "relationships, missing or incorrect parts. 50-100 words each.\n\n"
    "The passing description should be a realistic minimum-bar pass: structure correct, most "
    "labels present, but not flawless (think B/B+ student, not A+). The failing description "
    "should sound plausible — wrong structure, missing critical elements, or shows a "
    "misconception from the rubric — NOT a blank page.\n\n"
    "Return ONLY a single JSON object with exactly this schema, no prose, no markdown fences:\n"
    "{\n"
    '  "exemplar_pass": "<50-100 word prose description of a passing drawing>",\n'
    '  "exemplar_pass_note": "<one-sentence note on why this is at the passing bar>",\n'
    '  "exemplar_fail": "<50-100 word prose description of a failing drawing>",\n'
    '  "exemplar_fail_note": "<one-sentence note on why this is at the failing bar>"\n'
    "}"
)


_EMPTY_EXEMPLARS = {
    'exemplar_pass': '',
    'exemplar_pass_note': '',
    'exemplar_fail': '',
    'exemplar_fail_note': '',
}


def _parse_exemplars(response):
    """Parse the exemplars JSON response into the canonical 4-key dict."""
    out = dict(_EMPTY_EXEMPLARS)
    if not response:
        return out
    text = response.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        logger.warning(f"Exemplars JSON parse failed; raw response starts with: {response[:200]!r}")
        return out
    if not isinstance(data, dict):
        return out
    for key in out:
        v = data.get(key)
        out[key] = str(v).strip() if v else ''
    return out


def _build_exemplars_prompt(keywords, original_question_text, answers_json, mode, open_ended_question, rubric):
    """Build the user-side prompt for exemplar generation."""
    clean_text = _strip_html(original_question_text)
    labels = _answer_labels(answers_json)
    parts = []
    if keywords:
        parts.append(f"Topic keywords: {keywords}")
    if clean_text:
        parts.append(f"Original quiz question: {clean_text}")
    if labels:
        joined = " | ".join(labels[:6])
        parts.append(f"Answer choices from the original: {joined}")
    parts.append(f"Question type: {'draw (hand-drawn diagram)' if mode == 'draw' else 'explain (verbal explanation)'}")
    parts.append(f"Open-ended question the student will answer: {open_ended_question}")

    rubric_lines = []
    if rubric.get('canonical_answer'):
        rubric_lines.append(f"Canonical answer: {rubric['canonical_answer']}")
    if rubric.get('pass_criteria'):
        rubric_lines.append("Pass criteria: " + "; ".join(rubric['pass_criteria']))
    if rubric.get('common_misconceptions'):
        rubric_lines.append("Common misconceptions: " + "; ".join(rubric['common_misconceptions']))
    if rubric.get('fatal_errors'):
        rubric_lines.append("Fatal errors: " + "; ".join(rubric['fatal_errors']))
    if mode == 'draw' and rubric.get('required_visual_elements'):
        rubric_lines.append("Required visual elements: " + "; ".join(rubric['required_visual_elements']))
    if rubric_lines:
        parts.append("Rubric:")
        parts.extend(rubric_lines)
    parts.append("Exemplars JSON:")
    return "\n".join(parts)


def generate_exemplars(row, client, model, mode, open_ended_question, rubric):
    """Call the LLM to produce one passing + one failing exemplar response for the question."""
    if not open_ended_question:
        return dict(_EMPTY_EXEMPLARS)
    prompt = _build_exemplars_prompt(
        row.get("keywords"),
        row.get("question_text"),
        row.get("answers"),
        mode,
        open_ended_question,
        rubric or _empty_rubric(mode),
    )
    system_prompt = _EXEMPLARS_DRAW_SYSTEM_PROMPT if mode == 'draw' else _EXEMPLARS_EXPLAIN_SYSTEM_PROMPT
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.5},
        )
        return _parse_exemplars(resp["message"]["content"])
    except Exception as e:
        logger.warning(f"LLM exemplar generation failed for question {row.get('question_id')}: {e}")
        return dict(_EMPTY_EXEMPLARS)


_TRANSCRIBE_SYSTEM_PROMPT = (
    "You are a transcription assistant. Listen to the audio and produce an accurate, "
    "verbatim transcription of everything the speaker says. Output ONLY the transcription "
    "text — no preamble, no labels, no timestamps, no commentary."
)

_ASSESS_EXPLAIN_SYSTEM_PROMPT = (
    "You are a university instructor assessing a student's verbal explanation of a concept. "
    "You will be given the original quiz question, topic keywords, the follow-up question asked, "
    "a structured rubric, an instructor-written assessment guide, optionally a model answer and "
    "calibration exemplars and previously-graded examples, and the student's spoken response "
    "(as a transcript).\n\n"
    "Walk the rubric one item at a time. For EACH pass criterion answer met / partial / missing. "
    "For EACH fatal error answer absent / present. Be charitable: informal phrasing, hesitation, "
    "and incomplete-but-substantive coverage should still earn met or partial. A student who "
    "repeats the question, says \"I don't know\", or shows a fundamental misunderstanding fails.\n\n"
    "Then write 2-3 sentences of feedback for the student that references the criteria they met "
    "and missed.\n\n"
    "Return ONLY a single JSON object with this exact schema (no prose, no markdown fences):\n\n"
    "{\n"
    '  "pass_criteria_evaluations": [\n'
    '    {"criterion": "<the criterion text, copied verbatim from the rubric>", "status": "met"|"partial"|"missing"}\n'
    "  ],\n"
    '  "fatal_errors_evaluations": [\n'
    '    {"error": "<the error text, copied verbatim from the rubric>", "status": "absent"|"present"}\n'
    "  ],\n"
    '  "feedback": "<2-3 sentences of feedback for the student>"\n'
    "}\n\n"
    "Include exactly one entry per rubric item — no more, no less. If the rubric supplies an "
    "empty list, return an empty list. Do not invent criteria or errors that were not in the rubric."
)

_ASSESS_DRAW_SYSTEM_PROMPT = (
    "You are a university instructor assessing a student's hand-drawn diagram or figure. "
    "You will be given the original quiz question, topic keywords, the follow-up question asked, "
    "a structured rubric, an instructor-written assessment guide, optionally a model answer and "
    "calibration exemplars, and the student's drawing (as an image).\n\n"
    "Walk the rubric one item at a time. For EACH pass criterion answer met / partial / missing. "
    "For EACH fatal error answer absent / present. For EACH required visual element answer "
    "yes / no / unclear based on what literally appears in the drawing. Be charitable about "
    "neatness — rough hand-drawn diagrams with the right structure should pass.\n\n"
    "Then write 2-3 sentences of feedback that references which elements are present and which "
    "are missing.\n\n"
    "Return ONLY a single JSON object with this exact schema (no prose, no markdown fences):\n\n"
    "{\n"
    '  "pass_criteria_evaluations": [\n'
    '    {"criterion": "<the criterion text, copied verbatim from the rubric>", "status": "met"|"partial"|"missing"}\n'
    "  ],\n"
    '  "fatal_errors_evaluations": [\n'
    '    {"error": "<the error text, copied verbatim from the rubric>", "status": "absent"|"present"}\n'
    "  ],\n"
    '  "visual_elements_evaluations": [\n'
    '    {"element": "<the element text, copied verbatim from the rubric>", "status": "yes"|"no"|"unclear"}\n'
    "  ],\n"
    '  "feedback": "<2-3 sentences of feedback for the student>"\n'
    "}\n\n"
    "Include exactly one entry per rubric item — no more, no less. Do not invent items that were "
    "not in the rubric."
)


_VALID_PASS_STATUSES = {'met', 'partial', 'missing'}
_VALID_FATAL_STATUSES = {'absent', 'present'}
_VALID_VISUAL_STATUSES = {'yes', 'no', 'unclear'}


def _parse_per_criterion_response(response, mode):
    """Parse Gemma's JSON response into a per-criterion evaluation dict.

    Returns a dict with keys: pass_criteria_evaluations, fatal_errors_evaluations,
    visual_elements_evaluations (draw mode only), feedback. Statuses outside the
    permitted vocabulary degrade to the worst-case (missing / present / no) so a
    bogus status never silently flips a fail to a pass.
    """
    out = {
        'pass_criteria_evaluations': [],
        'fatal_errors_evaluations': [],
        'feedback': '',
    }
    if mode == 'draw':
        out['visual_elements_evaluations'] = []
    if not response:
        return out

    text = response.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s*```\s*$', '', text)
    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        text = text[brace_start:brace_end + 1]

    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        logger.warning(f"Per-criterion JSON parse failed; raw response starts with: {response[:200]!r}")
        return out
    if not isinstance(data, dict):
        return out

    def _normalize(items, item_key, status_key, valid, worst):
        result = []
        if not isinstance(items, list):
            return result
        for entry in items:
            if not isinstance(entry, dict):
                continue
            text = str(entry.get(item_key, '')).strip()
            status = str(entry.get('status', '')).strip().lower()
            if status not in valid:
                status = worst
            if text:
                result.append({item_key: text, 'status': status})
        return result

    out['pass_criteria_evaluations'] = _normalize(
        data.get('pass_criteria_evaluations'), 'criterion', 'status',
        _VALID_PASS_STATUSES, 'missing',
    )
    out['fatal_errors_evaluations'] = _normalize(
        data.get('fatal_errors_evaluations'), 'error', 'status',
        _VALID_FATAL_STATUSES, 'present',
    )
    if mode == 'draw':
        out['visual_elements_evaluations'] = _normalize(
            data.get('visual_elements_evaluations'), 'element', 'status',
            _VALID_VISUAL_STATUSES, 'no',
        )
    fb = data.get('feedback')
    out['feedback'] = str(fb).strip() if fb else ''
    return out


_AGGREGATION_THRESHOLD = 0.5


def _aggregate_pass_fail(per_criterion, mode):
    """Compute deterministic pass/fail from a parsed per-criterion evaluation dict.

    Pass requires:
    - pass-criteria score (met=1, partial=0.5, missing=0) / count >= 0.5,
      defaulting to pass when there are zero criteria.
    - No fatal error marked present.
    - For draw mode: visual-elements score (yes=1, unclear=0.5, no=0) / count >= 0.5,
      defaulting to pass when there are zero visual elements.
    """
    pass_items = per_criterion.get('pass_criteria_evaluations', [])
    if pass_items:
        pass_score = sum(
            1.0 if e['status'] == 'met' else (0.5 if e['status'] == 'partial' else 0.0)
            for e in pass_items
        ) / len(pass_items)
    else:
        pass_score = 1.0

    fatal_items = per_criterion.get('fatal_errors_evaluations', [])
    fatal_present = any(e['status'] == 'present' for e in fatal_items)

    visual_score = 1.0
    if mode == 'draw':
        visual_items = per_criterion.get('visual_elements_evaluations', [])
        if visual_items:
            visual_score = sum(
                1.0 if e['status'] == 'yes' else (0.5 if e['status'] == 'unclear' else 0.0)
                for e in visual_items
            ) / len(visual_items)

    passed = (
        pass_score >= _AGGREGATION_THRESHOLD and
        not fatal_present and
        visual_score >= _AGGREGATION_THRESHOLD
    )
    return 'pass' if passed else 'fail'


def _render_rubric_block(rubric, mode):
    """Render a structured rubric dict as a prompt-ready bullet block.

    Returns an empty string when the rubric is empty so callers can drop the
    section cleanly.
    """
    if not rubric:
        return ''
    blocks = []
    if rubric.get('canonical_answer'):
        blocks.append(f"Canonical answer (1-2 sentence summary): {rubric['canonical_answer']}")
    if rubric.get('model_answer'):
        blocks.append(
            "Model answer (an A-grade response in a real student's voice — use this as a "
            f"depth/coverage reference, the student does NOT need to phrase things this way):\n{rubric['model_answer']}"
        )
    if rubric.get('pass_criteria'):
        bullets = "\n".join(f"- {c}" for c in rubric['pass_criteria'])
        blocks.append(
            "Pass criteria (the student's response MUST demonstrate these — evaluate each one):\n"
            f"{bullets}"
        )
    if rubric.get('acceptable_alternatives'):
        bullets = "\n".join(f"- {c}" for c in rubric['acceptable_alternatives'])
        blocks.append(
            "Acceptable alternative framings (count these as equivalent to the criteria):\n"
            f"{bullets}"
        )
    if rubric.get('common_misconceptions'):
        bullets = "\n".join(f"- {c}" for c in rubric['common_misconceptions'])
        blocks.append(
            "Common misconceptions (do NOT count these as passing):\n"
            f"{bullets}"
        )
    if rubric.get('fatal_errors'):
        bullets = "\n".join(f"- {c}" for c in rubric['fatal_errors'])
        blocks.append(
            "Fatal errors (any of these forces a fail — evaluate each one):\n"
            f"{bullets}"
        )
    if mode == 'draw' and rubric.get('required_visual_elements'):
        bullets = "\n".join(f"- {c}" for c in rubric['required_visual_elements'])
        blocks.append(
            "Required visual elements — walk the drawing one element at a time, marking yes / "
            "no / unclear based on what is literally drawn:\n"
            f"{bullets}"
        )
    return "\n\n".join(blocks)


def _render_exemplars_block(exemplars):
    """Render Gemini-authored pass/fail exemplars as a prompt-ready section."""
    if not exemplars:
        return ''
    p_resp = exemplars.get('exemplar_pass') or ''
    p_note = exemplars.get('exemplar_pass_note') or ''
    f_resp = exemplars.get('exemplar_fail') or ''
    f_note = exemplars.get('exemplar_fail_note') or ''
    if not (p_resp or f_resp):
        return ''
    parts = ["Calibration exemplars (use these to anchor the bar, not as ground truth):"]
    if p_resp:
        parts.append(f'Passing example: "{p_resp}"')
        if p_note:
            parts.append(f"  (Note: {p_note})")
    if f_resp:
        parts.append(f'Failing example: "{f_resp}"')
        if f_note:
            parts.append(f"  (Note: {f_note})")
    return "\n".join(parts)


def _render_locked_examples_block(locked_examples):
    """Render previously instructor-approved assessments as in-context few-shot examples.

    Each entry should be a dict with keys: response, result, feedback. Caller is
    responsible for sampling — this function just renders.
    """
    if not locked_examples:
        return ''
    parts = ["Previously-graded examples for this question (instructor-approved verdicts):"]
    for i, ex in enumerate(locked_examples, start=1):
        resp = (ex.get('response') or '').strip()
        result = (ex.get('result') or '').strip()
        feedback = (ex.get('feedback') or '').strip()
        if not resp or not result:
            continue
        parts.append(f'Example {i} student response: "{resp}"')
        parts.append(f"Example {i} verdict: {result}")
        if feedback:
            parts.append(f"Example {i} instructor feedback: {feedback}")
    if len(parts) == 1:
        return ''
    return "\n".join(parts)


def _build_assessment_prompt(
    keywords,
    open_ended_question,
    original_question_text,
    transcript=None,
    assessment_guide=None,
    rubric=None,
    exemplars=None,
    locked_examples=None,
    mode='explain',
):
    """Build the user-side prompt for assessing a student response."""
    parts = []
    if keywords:
        parts.append(f"Topic keywords: {keywords}")
    if original_question_text:
        parts.append(f"Original quiz question: {original_question_text}")
    if open_ended_question:
        parts.append(f"Follow-up question asked: {open_ended_question}")
    if assessment_guide:
        parts.append(f"Assessment guide (instructor framing): {assessment_guide}")

    rubric_block = _render_rubric_block(rubric, mode)
    if rubric_block:
        parts.append(rubric_block)

    exemplars_block = _render_exemplars_block(exemplars)
    if exemplars_block:
        parts.append(exemplars_block)

    locked_block = _render_locked_examples_block(locked_examples)
    if locked_block:
        parts.append(locked_block)

    if transcript:
        parts.append(f"Student's response (transcript): {transcript}")
    return "\n\n".join(parts)


def transcribe_audio(audio_path, client, model):
    """Transcribe an audio file using a multimodal model (e.g. gemma4:e4b)."""
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": _TRANSCRIBE_SYSTEM_PROMPT},
                {"role": "user", "content": "Transcribe this audio recording.", "images": [audio_path]},
            ],
            options={"temperature": 0.1},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Audio transcription failed for {audio_path}: {e}")
        return ""


def _assess_explain_once(prompt, client, model):
    """Single-pass explain assessment. Returns (result, feedback, evaluations_dict)."""
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": _ASSESS_EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )
        evals = _parse_per_criterion_response(resp["message"]["content"], 'explain')
    except Exception as e:
        logger.warning(f"Explain assessment failed: {e}")
        return 'fail', f'Assessment error: {e}', _parse_per_criterion_response('', 'explain')
    return _aggregate_pass_fail(evals, 'explain'), evals.get('feedback', ''), evals


def _assess_draw_once(prompt, image_path, client, model):
    """Single-pass draw assessment. Returns (result, feedback, evaluations_dict)."""
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": _ASSESS_DRAW_SYSTEM_PROMPT},
                {"role": "user", "content": "", "images": [image_path]},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )
        evals = _parse_per_criterion_response(resp["message"]["content"], 'draw')
    except Exception as e:
        logger.warning(f"Draw assessment failed for {image_path}: {e}")
        return 'fail', f'Assessment error: {e}', _parse_per_criterion_response('', 'draw')
    return _aggregate_pass_fail(evals, 'draw'), evals.get('feedback', ''), evals


_SELF_CONSISTENCY_N = 3


def _vote_assessments(runs):
    """Majority-vote a list of (result, feedback, evals) tuples.

    Returns (result, confidence, feedback, evals) where confidence is "high"
    when all runs agree and "borderline" otherwise. The feedback and evals are
    drawn from the first run that matches the majority verdict so the per-criterion
    audit trail aligns with the reported result.
    """
    if not runs:
        return 'fail', 'borderline', 'No assessment runs completed.', {}
    n_pass = sum(1 for r, _, _ in runs if r == 'pass')
    n_fail = len(runs) - n_pass
    majority = 'pass' if n_pass >= n_fail else 'fail'
    confidence = 'high' if (n_pass == 0 or n_fail == 0) else 'borderline'
    for r, fb, ev in runs:
        if r == majority:
            return majority, confidence, fb, ev
    r, fb, ev = runs[0]
    return r, confidence, fb, ev


def assess_explain(transcript, keywords, open_ended_question, original_question_text, client, model,
                   assessment_guide=None, rubric=None, exemplars=None, locked_examples=None,
                   n_consistency=_SELF_CONSISTENCY_N):
    """Assess a verbal explanation, running N self-consistency passes.

    Returns (result, confidence, feedback, evaluations).
    """
    prompt = _build_assessment_prompt(
        keywords, open_ended_question, original_question_text,
        transcript=transcript, assessment_guide=assessment_guide,
        rubric=rubric, exemplars=exemplars, locked_examples=locked_examples,
        mode='explain',
    )
    runs = [_assess_explain_once(prompt, client, model) for _ in range(n_consistency)]
    return _vote_assessments(runs)


def assess_draw(image_path, keywords, open_ended_question, original_question_text, client, model,
                assessment_guide=None, rubric=None, exemplars=None, locked_examples=None,
                n_consistency=_SELF_CONSISTENCY_N):
    """Assess a hand-drawn diagram, running N self-consistency passes.

    Returns (result, confidence, feedback, evaluations).
    """
    prompt = _build_assessment_prompt(
        keywords, open_ended_question, original_question_text,
        assessment_guide=assessment_guide, rubric=rubric, exemplars=exemplars,
        locked_examples=locked_examples, mode='draw',
    )
    runs = [_assess_draw_once(prompt, image_path, client, model) for _ in range(n_consistency)]
    return _vote_assessments(runs)


def _parse_rubric_from_row(question_info_row):
    """Extract rubric and exemplars from an open-ended question row dict."""
    mode = (question_info_row.get('question_mode') or 'explain').strip().lower()
    rubric_raw = question_info_row.get('rubric_json') or ''
    if isinstance(rubric_raw, str) and rubric_raw.strip():
        rubric = _parse_structured_rubric(rubric_raw, 'draw' if mode == 'draw' else 'explain')
    else:
        rubric = _empty_rubric('draw' if mode == 'draw' else 'explain')
    exemplars = {
        'exemplar_pass': str(question_info_row.get('exemplar_pass') or '').strip(),
        'exemplar_pass_note': str(question_info_row.get('exemplar_pass_note') or '').strip(),
        'exemplar_fail': str(question_info_row.get('exemplar_fail') or '').strip(),
        'exemplar_fail_note': str(question_info_row.get('exemplar_fail_note') or '').strip(),
    }
    return rubric, exemplars, mode


def assess_replies(replies, question_info_row, model=None, audio_model=None,
                   locked_examples=None, n_consistency=_SELF_CONSISTENCY_N):
    """Assess a list of student reply dicts, returning a list of assessment result dicts.

    Each reply dict should have keys: student_id, student_name, question_id,
    question_mode, reply_text, attachment_path, audio_path.

    question_info_row should have: keywords, open_ended_question,
    original_question_text, assessment_guide, rubric_json, and the four
    exemplar_* columns.

    locked_examples (optional): list of dicts {response, result, feedback}
    drawn from previously instructor-approved assessments for the same question.
    Used as in-context few-shot calibration.
    """
    try:
        import ollama  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "The 'ollama' package is required for assess-replies. "
            "Install with: pip install ollama"
        ) from e

    model = model or DEFAULT_MODEL
    audio_model = audio_model or DEFAULT_AUDIO_MODEL
    client = _make_client(cloud=False)

    try:
        client.list()
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama at its configured host ({os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}). "
            f"Is the Ollama server running? Original error: {e}"
        ) from e

    keywords = question_info_row.get('keywords', '')
    oe_question = question_info_row.get('open_ended_question', '')
    orig_text = question_info_row.get('original_question_text', '')
    assessment_guide = question_info_row.get('assessment_guide', '') or ''
    rubric, exemplars, _row_mode = _parse_rubric_from_row(question_info_row)

    total = len(replies)
    print(f"Assessing {total} student replies with model '{model}' (n={n_consistency} self-consistency)...")
    results = []
    for i, reply in enumerate(replies, start=1):
        student_name = reply.get('student_name', '?')
        mode = reply.get('question_mode', 'explain')
        print(f"  [{i}/{total}] {student_name} ({mode})...", end="", flush=True)

        transcript = ''
        evaluations = {}
        if mode == 'explain':
            audio_path = reply.get('audio_path', '')
            if audio_path:
                print(" transcribing...", end="", flush=True)
                transcript = transcribe_audio(audio_path, client, audio_model)
            if not transcript:
                # Fall back to reply text if no audio or transcription failed
                transcript = _strip_html(reply.get('reply_text', ''))
            if not transcript:
                print(" no response content — skipped")
                results.append({
                    'student_id': reply['student_id'],
                    'student_name': student_name,
                    'question_id': reply['question_id'],
                    'question_mode': mode,
                    'result': 'fail',
                    'confidence': 'high',
                    'feedback': 'No response content to assess (no audio and no text).',
                    'transcript': '',
                    'criteria_evaluations': '',
                    'assessed_at': '',
                })
                continue
            print(f" assessing (x{n_consistency})...", end="", flush=True)
            result, confidence, feedback, evaluations = assess_explain(
                transcript, keywords, oe_question, orig_text, client, model,
                assessment_guide=assessment_guide, rubric=rubric, exemplars=exemplars,
                locked_examples=locked_examples, n_consistency=n_consistency,
            )
        else:
            # Draw mode
            image_path = reply.get('attachment_path', '')
            if not image_path:
                print(" no image attached — skipped")
                results.append({
                    'student_id': reply['student_id'],
                    'student_name': student_name,
                    'question_id': reply['question_id'],
                    'question_mode': mode,
                    'result': 'fail',
                    'confidence': 'high',
                    'feedback': 'No image attachment to assess.',
                    'transcript': '',
                    'criteria_evaluations': '',
                    'assessed_at': '',
                })
                continue
            print(f" assessing image (x{n_consistency})...", end="", flush=True)
            # locked_examples carry transcripts, which don't apply to drawings — drop them.
            result, confidence, feedback, evaluations = assess_draw(
                image_path, keywords, oe_question, orig_text, client, model,
                assessment_guide=assessment_guide, rubric=rubric, exemplars=exemplars,
                locked_examples=None, n_consistency=n_consistency,
            )

        from datetime import datetime, timezone
        assessed_at = datetime.now(timezone.utc).isoformat()
        print(f" {result} ({confidence})")

        results.append({
            'student_id': reply['student_id'],
            'student_name': student_name,
            'question_id': reply['question_id'],
            'question_mode': mode,
            'result': result,
            'confidence': confidence,
            'feedback': feedback,
            'transcript': transcript,
            'criteria_evaluations': json.dumps(evaluations, ensure_ascii=False) if evaluations else '',
            'assessed_at': assessed_at,
        })

    print("Assessment complete.")
    return results


def generate_open_ended_questions(rows, model=None, n=3, progress=None):
    """Classify and generate n candidate open-ended questions per input row.

    Step 1: For each question, ask the LLM whether 'explain' or 'draw' is the
    better assessment mode based on the topic and question content.
    Step 2: Generate n candidate open-ended questions using the mode-specific prompt.
    Returns a flat list of result dicts — n rows per input row (padded with empty
    candidate strings if the LLM returned fewer). Each row has selected_question=0;
    the instructor reviews the output CSV and sets one row per group to 1.

    `progress` is an optional `(spin, spin_done)` callback pair from canvigator_utils;
    when provided, per-question stage status is rendered as a single overwriting
    spinner line instead of one printed line per question.
    """
    try:
        import ollama  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "The 'ollama' package is required for generate-follow-up-questions. "
            "Install with: pip install ollama"
        ) from e

    model = model or DEFAULT_TEXT_MODEL
    client = _make_client(cloud=True)
    spin_fn, spin_done_fn = progress if progress else (None, None)

    total = len(rows)
    print(f"Generating {n} candidate open-ended questions for each of {total} questions with cloud model '{model}'...")
    results = []
    frame = 0
    for i, row in enumerate(rows, start=1):
        label = row.get('question_name') or row.get('question_id')
        if spin_fn:
            spin_fn(frame, f"[{i}/{total}] {label} — classifying")
            frame += 1
        else:
            print(f"  [{i}/{total}] {label} — classifying...", end="", flush=True)
        mode = classify_question_mode(row, client, model)
        if spin_fn:
            spin_fn(frame, f"[{i}/{total}] {label} — {mode} — generating {n} candidates")
            frame += 1
        else:
            print(f" {mode} — generating {n} candidates...", end="", flush=True)
        candidates = generate_open_ended_candidates(row, client, model, mode, n=n)
        non_empty = len([c for c in candidates if c])
        if spin_fn:
            spin_fn(frame, f"[{i}/{total}] {label} — writing {non_empty} guide(s) + rubric(s) + exemplars")
            frame += 1
        else:
            print(f" writing {non_empty} guide(s) + rubric(s) + exemplars...", end="", flush=True)
        guides = [generate_assessment_guide(row, client, model, mode, cand) for cand in candidates]
        rubrics = [generate_structured_rubric(row, client, model, mode, cand) for cand in candidates]
        exemplars = [generate_exemplars(row, client, model, mode, cand, rub) for cand, rub in zip(candidates, rubrics)]
        if not spin_fn:
            print(" done")

        if not candidates:
            logger.warning(f"No candidates generated for question {row.get('question_id')}")

        # Always emit exactly n rows per question so every group has a predictable shape.
        padded_candidates = (candidates + [''] * n)[:n]
        padded_guides = (guides + [''] * n)[:n]
        padded_rubrics = (rubrics + [_empty_rubric(mode)] * n)[:n]
        padded_exemplars = (exemplars + [dict(_EMPTY_EXEMPLARS)] * n)[:n]
        original_text = _strip_html(row.get('question_text'))
        for cand, guide, rubric, ex in zip(padded_candidates, padded_guides, padded_rubrics, padded_exemplars):
            results.append({
                'selected_question': 0,
                'question_id': row.get('question_id'),
                'position': row.get('position'),
                'question_name': row.get('question_name'),
                'keywords': row.get('keywords'),
                'question_mode': mode,
                'open_ended_question': cand,
                'assessment_guide': guide,
                'rubric_json': json.dumps(rubric, ensure_ascii=False),
                'exemplar_pass': ex.get('exemplar_pass', ''),
                'exemplar_pass_note': ex.get('exemplar_pass_note', ''),
                'exemplar_fail': ex.get('exemplar_fail', ''),
                'exemplar_fail_note': ex.get('exemplar_fail_note', ''),
                'original_question_text': original_text,
            })
    if spin_done_fn:
        spin_done_fn(f"Generated open-ended questions for {total} originals")
    else:
        print("Generation complete.")
    return results
