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
