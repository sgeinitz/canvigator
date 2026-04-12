"""Local LLM integration via Ollama for tagging and (later) generating quiz content."""
import html
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b")

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
    "Given a topic and details from an existing quiz question, create ONE new open-ended "
    "question that a student would answer verbally. The question should:\n"
    "- Begin with \"Explain\" and require the student to explain a concept or idea clearly "
    "in their own words\n"
    "- Be answerable in under 1 minute of speaking\n"
    "- Test understanding, not memorization — the student should demonstrate they grasp "
    "the underlying idea, not just recall a definition\n"
    "- Be self-contained (a student should understand what is being asked without seeing "
    "the original quiz question)\n"
    "- NOT be a yes/no question or a question that can be answered in one word\n\n"
    "Respond with ONLY the question text — no preamble, no numbering, no explanation, "
    "no quotation marks."
)

_DRAW_SYSTEM_PROMPT = (
    "You are an expert university instructor designing visual assessment questions. "
    "Given a topic and details from an existing quiz question, create ONE new question "
    "that asks a student to draw a diagram or figure by hand. The question should:\n"
    "- Begin with \"Draw a diagram\" or \"Draw a figure\" and clearly describe what the "
    "student should illustrate\n"
    "- Be completable in under 2 minutes of drawing\n"
    "- Test understanding of structure, relationships, or processes — not artistic skill\n"
    "- Be self-contained (a student should understand what is being asked without seeing "
    "the original quiz question)\n"
    "- Specify what key elements or labels should appear in the diagram\n\n"
    "Respond with ONLY the question text — no preamble, no numbering, no explanation, "
    "no quotation marks."
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
        resp = client.chat(
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


def tag_questions(rows, model=None):
    """Annotate each row dict in-place with a 'tags' key (comma-separated string)."""
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "The 'ollama' package is required for --tag. Install with: pip install ollama"
        ) from e

    model = model or DEFAULT_MODEL
    client = ollama.Client()

    try:
        client.list()
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama at its configured host ({os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}). "
            f"Is the Ollama server running? Original error: {e}"
        ) from e

    total = len(rows)
    print(f"Tagging {total} questions with model '{model}'...")
    for i, row in enumerate(rows, start=1):
        print(f"  [{i}/{total}] {row.get('question_name') or row.get('question_id')}")
        tags = tag_question(row, client, model)
        row["keywords"] = ", ".join(tags)
    print("Tagging complete.")


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
        resp = client.chat(
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


def generate_open_ended_question(row, client, model, mode):
    """Call the LLM to generate one open-ended question of the given mode (explain or draw)."""
    prompt = _build_open_ended_prompt(
        row.get("keywords"),
        row.get("question_text"),
        row.get("answers"),
        mode,
    )
    system_prompt = _DRAW_SYSTEM_PROMPT if mode == "draw" else _EXPLAIN_SYSTEM_PROMPT
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.7},
        )
        content = resp["message"]["content"].strip()
        # Take only the first paragraph in case the model over-generates
        first_para = content.split("\n\n")[0].strip()
        return first_para
    except Exception as e:
        logger.warning(f"LLM generation failed for question {row.get('question_id')}: {e}")
        return ""


def generate_open_ended_questions(rows, model=None):
    """Classify then generate an open-ended question for each row; returns a list of result dicts.

    Step 1: For each question, ask the LLM whether 'explain' or 'draw' is the
    better assessment mode based on the topic and question content.
    Step 2: Generate the open-ended question using the mode-specific prompt.
    """
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "The 'ollama' package is required for generate-open-ended-questions. "
            "Install with: pip install ollama"
        ) from e

    model = model or DEFAULT_MODEL
    client = ollama.Client()

    try:
        client.list()
    except Exception as e:
        raise RuntimeError(
            f"Could not reach Ollama at its configured host ({os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}). "
            f"Is the Ollama server running? Original error: {e}"
        ) from e

    total = len(rows)
    print(f"Generating {total} open-ended questions with model '{model}'...")
    results = []
    for i, row in enumerate(rows, start=1):
        label = row.get('question_name') or row.get('question_id')
        print(f"  [{i}/{total}] {label} — classifying...", end="", flush=True)
        mode = classify_question_mode(row, client, model)
        print(f" {mode} — generating...", end="", flush=True)
        question = generate_open_ended_question(row, client, model, mode)
        print(" done")
        results.append({
            'question_id': row.get('question_id'),
            'position': row.get('position'),
            'question_name': row.get('question_name'),
            'keywords': row.get('keywords'),
            'question_mode': mode,
            'original_question_text': _strip_html(row.get('question_text')),
            'open_ended_question': question,
        })
    print("Generation complete.")
    return results
