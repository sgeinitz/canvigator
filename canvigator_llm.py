"""Local LLM integration via Ollama for tagging and (later) generating quiz content."""
import html
import json
import logging
import os
import re

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:31b")
DEFAULT_AUDIO_MODEL = os.environ.get("OLLAMA_AUDIO_MODEL", "gemma4:e4b")

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
        resp = client.chat(
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


_TRANSCRIBE_SYSTEM_PROMPT = (
    "You are a transcription assistant. Listen to the audio and produce an accurate, "
    "verbatim transcription of everything the speaker says. Output ONLY the transcription "
    "text — no preamble, no labels, no timestamps, no commentary."
)

_ASSESS_EXPLAIN_SYSTEM_PROMPT = (
    "You are a university instructor assessing a student's verbal explanation of a concept. "
    "Given the original quiz question, the topic keywords, and the student's spoken response "
    "(provided as a transcript), determine whether the student demonstrates a reasonable "
    "understanding of the core concept.\n\n"
    "A \"pass\" means the student demonstrates a reasonable understanding of the core concept, "
    "even if their explanation is imprecise, incomplete, or uses informal language. "
    "A \"fail\" means the student shows a fundamental misunderstanding, did not address the "
    "question, or gave a response with no substantive content.\n\n"
    "Respond in EXACTLY this format (two lines):\n"
    "Result: pass\n"
    "Feedback: Your 2-3 sentence feedback here.\n\n"
    "Or:\n"
    "Result: fail\n"
    "Feedback: Your 2-3 sentence feedback here."
)

_ASSESS_DRAW_SYSTEM_PROMPT = (
    "You are a university instructor assessing a student's hand-drawn diagram or figure. "
    "Given the original quiz question, the topic keywords, and the student's drawing "
    "(provided as an image), determine whether the drawing demonstrates a reasonable "
    "understanding of the key concepts and relationships.\n\n"
    "A \"pass\" means the drawing shows the essential structure or relationships, even if "
    "it is rough, has minor inaccuracies, or is missing non-critical labels. "
    "A \"fail\" means the drawing is fundamentally incorrect, shows the wrong structure, "
    "is missing critical elements, or does not address the question.\n\n"
    "Respond in EXACTLY this format (two lines):\n"
    "Result: pass\n"
    "Feedback: Your 2-3 sentence feedback here.\n\n"
    "Or:\n"
    "Result: fail\n"
    "Feedback: Your 2-3 sentence feedback here."
)


def _parse_assessment(response):
    r"""Parse a 'Result: pass/fail\nFeedback: ...' response into (result, feedback)."""
    if not response:
        return 'fail', 'No response from model.'

    result = 'fail'
    feedback = ''
    for line in response.strip().splitlines():
        line_stripped = line.strip()
        if line_stripped.lower().startswith('result:'):
            token = line_stripped[len('result:'):].strip().lower().strip('.,')
            result = 'pass' if token == 'pass' else 'fail'
        elif line_stripped.lower().startswith('feedback:'):
            feedback = line_stripped[len('feedback:'):].strip()

    if not feedback:
        # Fallback: treat the whole response as feedback
        feedback = response.strip()

    return result, feedback


def _build_assessment_prompt(keywords, open_ended_question, original_question_text, transcript=None):
    """Build the user-side prompt for assessing a student response."""
    parts = []
    if keywords:
        parts.append(f"Topic keywords: {keywords}")
    if original_question_text:
        parts.append(f"Original quiz question: {original_question_text}")
    if open_ended_question:
        parts.append(f"Follow-up question asked: {open_ended_question}")
    if transcript:
        parts.append(f"Student's response (transcript): {transcript}")
    return "\n".join(parts)


def transcribe_audio(audio_path, client, model):
    """Transcribe an audio file using a multimodal model (e.g. gemma4:e4b)."""
    try:
        resp = client.chat(
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


def assess_explain(transcript, keywords, open_ended_question, original_question_text, client, model):
    """Assess a student's verbal explanation using the transcript."""
    prompt = _build_assessment_prompt(keywords, open_ended_question, original_question_text, transcript=transcript)
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _ASSESS_EXPLAIN_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            options={"temperature": 0.3},
        )
        return _parse_assessment(resp["message"]["content"])
    except Exception as e:
        logger.warning(f"Explain assessment failed: {e}")
        return 'fail', f'Assessment error: {e}'


def assess_draw(image_path, keywords, open_ended_question, original_question_text, client, model):
    """Assess a student's drawing by sending the image to a multimodal model."""
    prompt = _build_assessment_prompt(keywords, open_ended_question, original_question_text)
    try:
        resp = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": _ASSESS_DRAW_SYSTEM_PROMPT},
                {"role": "user", "content": prompt, "images": [image_path]},
            ],
            options={"temperature": 0.3},
        )
        return _parse_assessment(resp["message"]["content"])
    except Exception as e:
        logger.warning(f"Draw assessment failed for {image_path}: {e}")
        return 'fail', f'Assessment error: {e}'


def assess_replies(replies, question_info_row, model=None, audio_model=None):
    """Assess a list of student reply dicts, returning a list of assessment result dicts.

    Each reply dict should have keys: student_id, student_name, question_id,
    question_mode, reply_text, attachment_path, audio_path.

    question_info_row should have: keywords, open_ended_question, original_question_text.
    """
    try:
        import ollama
    except ImportError as e:
        raise RuntimeError(
            "The 'ollama' package is required for assess-replies. "
            "Install with: pip install ollama"
        ) from e

    model = model or DEFAULT_MODEL
    audio_model = audio_model or DEFAULT_AUDIO_MODEL
    client = ollama.Client()

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

    total = len(replies)
    print(f"Assessing {total} student replies with model '{model}'...")
    results = []
    for i, reply in enumerate(replies, start=1):
        student_name = reply.get('student_name', '?')
        mode = reply.get('question_mode', 'explain')
        print(f"  [{i}/{total}] {student_name} ({mode})...", end="", flush=True)

        transcript = ''
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
                    'feedback': 'No response content to assess (no audio and no text).',
                    'transcript': '',
                    'assessed_at': '',
                })
                continue
            print(" assessing...", end="", flush=True)
            result, feedback = assess_explain(transcript, keywords, oe_question, orig_text, client, model)
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
                    'feedback': 'No image attachment to assess.',
                    'transcript': '',
                    'assessed_at': '',
                })
                continue
            transcript = ''
            print(" assessing image...", end="", flush=True)
            result, feedback = assess_draw(image_path, keywords, oe_question, orig_text, client, model)

        from datetime import datetime, timezone
        assessed_at = datetime.now(timezone.utc).isoformat()
        print(f" {result}")

        results.append({
            'student_id': reply['student_id'],
            'student_name': student_name,
            'question_id': reply['question_id'],
            'question_mode': mode,
            'result': result,
            'feedback': feedback,
            'transcript': transcript,
            'assessed_at': assessed_at,
        })

    print("Assessment complete.")
    return results


def generate_open_ended_questions(rows, model=None, n=3):
    """Classify and generate n candidate open-ended questions per input row.

    Step 1: For each question, ask the LLM whether 'explain' or 'draw' is the
    better assessment mode based on the topic and question content.
    Step 2: Generate n candidate open-ended questions using the mode-specific prompt.
    Returns a flat list of result dicts — n rows per input row (padded with empty
    candidate strings if the LLM returned fewer). Each row has selected_question=0;
    the instructor reviews the output CSV and sets one row per group to 1.
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
    print(f"Generating {n} candidate open-ended questions for each of {total} questions with model '{model}'...")
    results = []
    for i, row in enumerate(rows, start=1):
        label = row.get('question_name') or row.get('question_id')
        print(f"  [{i}/{total}] {label} — classifying...", end="", flush=True)
        mode = classify_question_mode(row, client, model)
        print(f" {mode} — generating {n} candidates...", end="", flush=True)
        candidates = generate_open_ended_candidates(row, client, model, mode, n=n)
        print(" done")

        if not candidates:
            logger.warning(f"No candidates generated for question {row.get('question_id')}")

        # Always emit exactly n rows per question so every group has a predictable shape.
        padded = (candidates + [''] * n)[:n]
        original_text = _strip_html(row.get('question_text'))
        for cand in padded:
            results.append({
                'selected_question': 0,
                'question_id': row.get('question_id'),
                'position': row.get('position'),
                'question_name': row.get('question_name'),
                'keywords': row.get('keywords'),
                'question_mode': mode,
                'open_ended_question': cand,
                'original_question_text': original_text,
            })
    print("Generation complete.")
    return results
