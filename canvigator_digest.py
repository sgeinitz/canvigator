"""Pre-class cohort digest.

Synthesizes a 1-page Markdown brief on cohort gaps over a configurable window
(default 7 days) from three signal sources:

* Recent quiz misses (joined to topic tags from ``*_questions_w_tags_*.csv``).
* Follow-up reply themes (failing/borderline rows in
  ``*_followup_assessments.csv``).
* Media-recording transcripts in window
  (``assignment<id>_recordings_*.csv``).

The brief ends with 2-3 suggested in-class discussion questions targeted at the
top-ranked gaps. Local Gemma 4 handles all student-derived content; an
instructor opt-in (``--cloud-questions``) routes only the final
question-drafting step to cloud Gemini 3 with a redacted prompt that contains
no transcripts or derived themes — just tag names and miss counts.
"""
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone

import pandas as pd

from canvigator_utils import find_csvs_in_window, find_latest_csv, today_str
from canvigator_assignment import _collectUniqueTags
from canvigator_llm import (
    DEFAULT_MODEL,
    DEFAULT_TEXT_MODEL,
    OLLAMA_CLOUD_HOST,
    _chat_with_retry,
    _make_client,
    analyze_recording_tags,
    extract_recording_themes,
)

logger = logging.getLogger(__name__)


_MAX_FOLLOWUP_ROWS_PER_PROMPT = 12
_TOP_PRIORITIES = 5

_FOLLOWUP_THEMES_SYSTEM_PROMPT = (
    "You are an instructor analyzing student responses to an open-ended follow-up "
    "question to surface where the cohort is struggling. The students who replied "
    "either failed the rubric or scored borderline; their transcripts and the "
    "rubric-criterion evaluations from an automated grader are provided.\n\n"
    "Identify 2-4 distinct concept-level struggles that show up across multiple "
    "students. For each:\n"
    "- A short title (3-6 words, in bold).\n"
    "- One sentence describing the misunderstanding.\n"
    "- A list of student indices (1-based) who exhibited it.\n\n"
    "Format as Markdown bullets. No preamble, no headers, no summary — just "
    "the themed bullets. If fewer than 2 distinct themes are evident, return "
    "only what is genuinely present."
)

_DISCUSSION_LOCAL_SYSTEM_PROMPT = (
    "You are helping an instructor plan the next class session. You will receive "
    "a ranked list of cohort gaps drawn from recent quiz misses, follow-up reply "
    "themes, and media-recording themes. Draft 2-3 short discussion questions "
    "the instructor can ask in class to surface and resolve the highest-priority "
    "gaps.\n\n"
    "Each question should:\n"
    "- Be open-ended (not yes/no, not single-word).\n"
    "- Target a specific gap from the list — name the topic in the question.\n"
    "- Be answerable in 2-5 minutes of class discussion.\n"
    "- Invite student-to-student exchange, not just instructor-to-student Q&A.\n\n"
    "Return ONLY the questions as Markdown bullets — one per line, prefixed with "
    "``- ``. No preamble, no headers, no numbering."
)

_DISCUSSION_CLOUD_SYSTEM_PROMPT = (
    "You are helping an instructor plan the next class session. You will receive "
    "a ranked list of quiz topics, each with the number of students who missed "
    "questions on it recently. Draft 2-3 short open-ended discussion questions "
    "the instructor can ask in class to surface and resolve student gaps on the "
    "highest-priority topics.\n\n"
    "Each question should:\n"
    "- Be open-ended (not yes/no, not single-word).\n"
    "- Target a specific topic from the list — name the topic in the question.\n"
    "- Be answerable in 2-5 minutes of class discussion.\n"
    "- Invite student-to-student exchange, not just instructor-to-student Q&A.\n\n"
    "Return ONLY the questions as Markdown bullets — one per line, prefixed with "
    "``- ``. No preamble, no headers, no numbering."
)


# ---------------------------------------------------------------------------
# Phase A — Load (no LLM)
# ---------------------------------------------------------------------------


_QUIZ_PREFIX_RE = re.compile(r'^(.*?_\d+)_')


def _extractQuizPrefix(filename, marker):
    """Return the ``<quiz_name>_<id>_`` prefix from a filename built with ``marker``.

    For example, ``quiz1_547889_all_subs_by_question_20260424.csv`` with
    ``marker='all_subs_by_question'`` returns ``'quiz1_547889_'``.
    """
    parts = filename.split(f"_{marker}_")
    if len(parts) < 2:
        return None
    return parts[0] + "_"


def _quizMetaFromPrefix(prefix):
    """Split a ``<quiz_name>_<id>_`` prefix into (quiz_name, quiz_id) strings."""
    stripped = prefix.rstrip('_')
    m = re.match(r'^(.*)_(\d+)$', stripped)
    if not m:
        return stripped, ''
    return m.group(1), m.group(2)


def _loadQuizMisses(data_path, since_date):
    """Compute per-tag miss counts for every quiz with subs-by-question data in the window.

    For each ``*_all_subs_by_question_*.csv`` whose embedded date is in window,
    locate that quiz's latest (static) ``*_questions_w_tags_*.csv``, join on
    ``question_id``, and tally how many (student, question) rows had
    ``points < points_possible``. Tags from each missed question's ``keywords``
    cell are credited once per missed row.

    Returns a list of dicts ``{quiz_id, quiz_name, total_attempts, n_missed_per_tag}``,
    one per quiz that had usable data. Quizzes with no tagged-questions CSV are
    skipped with a warning so a single missing artifact doesn't block the digest.
    """
    sub_files = find_csvs_in_window(data_path, 'all_subs_by_question_', since_date)
    by_prefix = {}
    for f in sub_files:
        prefix = _extractQuizPrefix(f.name, 'all_subs_by_question')
        if prefix is None:
            continue
        # Keep the newest file per quiz (find_csvs_in_window already sorts ascending).
        by_prefix[prefix] = f

    out = []
    for prefix, sub_path in by_prefix.items():
        quiz_name, quiz_id = _quizMetaFromPrefix(prefix)
        tags_path = find_latest_csv(data_path, prefix + 'questions_w_tags_')
        if tags_path is None:
            logger.warning(f"No questions_w_tags CSV for {prefix}; skipping miss tally.")
            continue
        try:
            subs_df = pd.read_csv(sub_path)
            tags_df = pd.read_csv(tags_path)
        except Exception as e:
            logger.warning(f"Failed to read miss data for {prefix}: {e}")
            continue
        if 'keywords' not in tags_df.columns or 'question_id' not in tags_df.columns:
            logger.warning(f"{tags_path.name}: missing keywords/question_id; skipping.")
            continue

        kw_by_qid = {}
        for _, row in tags_df.iterrows():
            qid = row.get('question_id')
            if pd.isna(qid):
                continue
            kw_by_qid[int(qid)] = str(row.get('keywords') or '')

        n_missed_per_tag = Counter()
        for _, row in subs_df.iterrows():
            pts = row.get('points')
            pp = row.get('points_possible')
            qid = row.get('question_id')
            if pd.isna(pts) or pd.isna(pp) or pd.isna(qid):
                continue
            if float(pts) >= float(pp):
                continue
            kw = kw_by_qid.get(int(qid), '')
            for tag in _splitKeywords(kw):
                n_missed_per_tag[tag] += 1

        out.append({
            'quiz_id': quiz_id,
            'quiz_name': quiz_name,
            'total_attempts': int(subs_df['id'].nunique()) if 'id' in subs_df.columns else 0,
            'n_missed_per_tag': n_missed_per_tag,
        })
    return out


def _splitKeywords(raw):
    """Split a comma-separated keywords cell into a clean lowercase tag list."""
    if not raw or pd.isna(raw):
        return []
    out = []
    seen = set()
    for piece in str(raw).split(','):
        t = piece.strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _loadFollowupAssessments(data_path, since_date):
    """Load follow-up assessments per quiz, filtered to ``assessed_at >= since_date``.

    Returns a list of ``{quiz_id, quiz_name, rows_by_question, n_dropped_nat}``
    where ``rows_by_question`` is ``{question_id: list[dict]}`` containing only
    rows whose ``result`` is ``fail`` or whose ``confidence`` is ``borderline``
    — i.e. the rows where students still need help. Pass + high-confidence rows
    are not interesting for the digest's "where the cohort is struggling" framing.
    """
    if not os.path.isdir(data_path):
        return []
    assessments_files = sorted(
        f for f in os.listdir(data_path) if f.endswith('_followup_assessments.csv')
    )
    cutoff = pd.Timestamp(since_date.year, since_date.month, since_date.day, tz='UTC')
    out = []
    for fname in assessments_files:
        prefix = fname[:-len('followup_assessments.csv')]
        quiz_name, quiz_id = _quizMetaFromPrefix(prefix)
        path = os.path.join(data_path, fname)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            logger.warning(f"Failed to read {fname}: {e}")
            continue
        if 'assessed_at' not in df.columns:
            continue
        parsed = pd.to_datetime(df['assessed_at'], errors='coerce', utc=True, format='ISO8601')
        n_dropped = int(parsed.isna().sum())
        df = df.assign(_assessed_dt=parsed)
        df = df[df['_assessed_dt'].notna() & (df['_assessed_dt'] >= cutoff)]
        if df.empty:
            continue

        is_fail = df['result'].astype(str).str.lower() == 'fail'
        is_borderline = df.get('confidence', pd.Series(dtype=str)).astype(str).str.lower() == 'borderline'
        focus = df[is_fail | is_borderline]
        if focus.empty:
            continue

        rows_by_question = {}
        for _, row in focus.iterrows():
            qid = row.get('question_id')
            if pd.isna(qid):
                continue
            rows_by_question.setdefault(int(qid), []).append(row.to_dict())

        out.append({
            'quiz_id': quiz_id,
            'quiz_name': quiz_name,
            'rows_by_question': rows_by_question,
            'n_dropped_nat': n_dropped,
        })
    return out


def _loadRecordingsInWindow(data_path, since_date):
    """Discover media-recording CSVs in window, deduped to the newest per assignment id.

    Each entry attaches the most recent ``*_questions_w_tags_*.csv`` from the same
    course directory (the tag vocabulary used by ``analyze-media-recordings``).
    Entries with no tags CSV available are still returned but with
    ``tags_path=None``; the analyzer then falls back to per-recording theme
    extraction without tag classification.
    """
    rec_files = find_csvs_in_window(data_path, '_recordings_', since_date)
    rec_files = [f for f in rec_files if re.match(r'^assignment\d+_recordings_', f.name)]
    by_assignment = {}
    for f in rec_files:
        m = re.match(r'^assignment(\d+)_recordings_(\d{8})\.csv$', f.name)
        if not m:
            continue
        # Latest wins (find_csvs_in_window sorted ascending → last write of the day).
        by_assignment[m.group(1)] = f

    if not by_assignment:
        return []

    tags_path = find_latest_csv(data_path, '_questions_w_tags_')
    out = []
    for assignment_id, rec_path in by_assignment.items():
        try:
            rec_df = pd.read_csv(rec_path)
        except Exception as e:
            logger.warning(f"Failed to read {rec_path.name}: {e}")
            continue
        out.append({
            'assignment_id': assignment_id,
            'recordings_df': rec_df,
            'tags_path': tags_path,
        })
    return out


# ---------------------------------------------------------------------------
# Phase B — Per-quiz follow-up theme summarization (Gemma 4)
# ---------------------------------------------------------------------------


def _summarizeCriteriaEvaluations(criteria_json):
    """Render the criteria_evaluations JSON blob as a compact bullet block, or empty."""
    if not criteria_json or pd.isna(criteria_json):
        return ''
    try:
        data = json.loads(criteria_json)
    except (ValueError, TypeError):
        return ''
    lines = []
    for crit in data.get('pass_criteria_evaluations', []) or []:
        status = crit.get('status', '')
        text = crit.get('criterion', '')
        if text:
            lines.append(f"  - [{status}] {text}")
    for err in data.get('fatal_errors_evaluations', []) or []:
        status = err.get('status', '')
        text = err.get('error', '')
        if status == 'present' and text:
            lines.append(f"  - [FATAL] {text}")
    return "\n".join(lines)


def _buildFollowupThemePrompt(quiz_name, question_id, focus_rows, mode):
    """Build the user-side prompt for cohort-level follow-up theme summarization.

    ``focus_rows`` is the combined fail + borderline list (newest first); the
    builder caps it at ``_MAX_FOLLOWUP_ROWS_PER_PROMPT`` to keep Gemma's context
    bounded on long windows.
    """
    capped = list(focus_rows)[:_MAX_FOLLOWUP_ROWS_PER_PROMPT]
    parts = [
        f"Quiz: {quiz_name}",
        f"Question id: {question_id}",
        f"Mode: {mode}",
        f"Number of struggling responses: {len(capped)}",
        "",
        "Responses (newest first):",
    ]
    for i, row in enumerate(capped, start=1):
        result = row.get('result', '')
        confidence = row.get('confidence', '')
        parts.append(f"\nStudent {i} (result={result}, confidence={confidence}):")
        if mode == 'explain':
            transcript = str(row.get('transcript') or '').strip()
            if transcript:
                parts.append(f"  Transcript: {transcript}")
            feedback = str(row.get('feedback') or '').strip()
            if feedback:
                parts.append(f"  Grader feedback: {feedback}")
        else:
            feedback = str(row.get('feedback') or '').strip()
            if feedback:
                parts.append(f"  Grader feedback: {feedback}")
        crit_block = _summarizeCriteriaEvaluations(row.get('criteria_evaluations'))
        if crit_block:
            parts.append("  Criteria:")
            parts.append(crit_block)
    parts.append("")
    parts.append("Recurring concept-level struggles across these students (Markdown bullets):")
    return "\n".join(parts)


def _summarizeFollowupThemes(quiz_blocks, client, model):
    """Run one Gemma 4 call per quiz/question and return ``{(quiz_id, question_id): bullets}``.

    ``quiz_blocks`` is the output of ``_loadFollowupAssessments``. The model and
    host are logged explicitly so the operator can verify locality.
    """
    logger.info(f"Follow-up theme summarization: model={model} host={os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}")
    out = {}
    for entry in quiz_blocks:
        for question_id, rows in entry['rows_by_question'].items():
            raw_mode = rows[0].get('question_mode')
            # NaN is truthy in Python — guard explicitly so a hand-edited CSV
            # row with an empty question_mode falls back to 'explain' instead
            # of crashing on `.lower()`.
            if raw_mode is None or pd.isna(raw_mode) or not str(raw_mode).strip():
                mode = 'explain'
            else:
                mode = str(raw_mode).strip().lower()
            prompt = _buildFollowupThemePrompt(
                entry['quiz_name'], question_id, rows, mode,
            )
            try:
                resp = _chat_with_retry(
                    client,
                    model=model,
                    messages=[
                        {"role": "system", "content": _FOLLOWUP_THEMES_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.3},
                )
                bullets = resp["message"]["content"].strip()
            except Exception as e:
                logger.warning(f"Follow-up theme summarization failed for quiz {entry['quiz_id']} q{question_id}: {e}")
                bullets = f"_(theme extraction failed: {e})_"
            out[(entry['quiz_id'], int(question_id))] = bullets
    return out


# ---------------------------------------------------------------------------
# Phase C — Per-recording-assignment analysis (Gemma 4 — direct reuse)
# ---------------------------------------------------------------------------


def _analyzeOneRecordingsBatch(rec_entry, client, model):
    """Run ``analyze_recording_tags`` and ``extract_recording_themes`` for one recordings CSV."""
    rec_df = rec_entry['recordings_df'].copy()
    rec_df['transcript'] = rec_df.get('transcript', pd.Series(dtype=str)).fillna('').astype(str)
    non_empty = rec_df[rec_df['transcript'].str.strip() != '']
    if non_empty.empty:
        return {
            'assignment_id': rec_entry['assignment_id'],
            'transcripts_with_names': [],
            'tags': [],
            'tag_to_indices': {},
            'themes_md': '_(no non-empty transcripts)_',
        }

    transcripts_with_names = [
        (str(row.get('student_name') or f"id={row.get('student_id')}"), row['transcript'])
        for _, row in non_empty.iterrows()
    ]
    transcripts_only = [t for _, t in transcripts_with_names]

    tags = []
    if rec_entry.get('tags_path') is not None:
        try:
            tags_df = pd.read_csv(rec_entry['tags_path'])
            if 'keywords' in tags_df.columns:
                tags = _collectUniqueTags(tags_df['keywords'])
        except Exception as e:
            logger.warning(f"Failed to load tags for assignment {rec_entry['assignment_id']}: {e}")

    tag_to_indices = analyze_recording_tags(transcripts_only, tags, client, model) if tags else {}
    themes_md = extract_recording_themes(transcripts_only, tags, client, model)

    return {
        'assignment_id': rec_entry['assignment_id'],
        'transcripts_with_names': transcripts_with_names,
        'tags': tags,
        'tag_to_indices': tag_to_indices,
        'themes_md': themes_md,
    }


# ---------------------------------------------------------------------------
# Phase D — Discussion-question synthesis
# ---------------------------------------------------------------------------


def _buildDigestPriorities(quiz_misses, followup_themes, recording_results, top_n=_TOP_PRIORITIES):
    """Rank cohort gaps across all three signal sources, returning the top ``top_n``.

    Each priority is ``{tag, miss_count, sources, evidence_snippets}`` where
    ``sources`` is the subset of ``{"quiz", "followup", "recording"}`` that
    surfaced the tag and ``evidence_snippets`` is a small list of theme bullet
    strings used in the local discussion-question prompt (not in the cloud one).
    """
    counts = Counter()
    sources_by_tag = {}
    snippets_by_tag = {}

    for entry in quiz_misses:
        for tag, n in entry['n_missed_per_tag'].items():
            counts[tag] += n
            sources_by_tag.setdefault(tag, set()).add('quiz')

    # Follow-up themes don't carry a clean per-tag count; we credit each unique
    # quiz that surfaced a theme so a struggle that shows up across multiple
    # quizzes ranks higher than one isolated to a single question. Snippets are
    # stored verbatim only for the local prompt.
    for (quiz_id, question_id), bullets in followup_themes.items():
        if not bullets or bullets.startswith('_('):
            continue
        # Without a tag attribution, we route the snippet under a synthetic tag
        # named after the quiz so it's still rankable; the snippet text is what
        # actually feeds the local prompt.
        synthetic = f"quiz {quiz_id}: q{question_id} struggles"
        counts[synthetic] += 1
        sources_by_tag.setdefault(synthetic, set()).add('followup')
        snippets_by_tag.setdefault(synthetic, []).append(bullets)

    for entry in recording_results:
        for tag, indices in (entry.get('tag_to_indices') or {}).items():
            if not indices:
                continue
            counts[tag] += len(indices)
            sources_by_tag.setdefault(tag, set()).add('recording')
        themes_md = entry.get('themes_md') or ''
        if themes_md and not themes_md.startswith('_('):
            synthetic = f"recording themes (assignment {entry['assignment_id']})"
            counts[synthetic] += 1
            sources_by_tag.setdefault(synthetic, set()).add('recording')
            snippets_by_tag.setdefault(synthetic, []).append(themes_md)

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    out = []
    for tag, n in ranked[:top_n]:
        out.append({
            'tag': tag,
            'miss_count': int(n),
            'sources': sorted(sources_by_tag.get(tag, set())),
            'evidence_snippets': list(snippets_by_tag.get(tag, [])),
        })
    return out


def _buildDiscussionPromptLocal(priorities):
    """Build the full-fidelity (local-only) discussion-question prompt."""
    parts = ["Top cohort gaps (ranked):"]
    for i, p in enumerate(priorities, start=1):
        parts.append(
            f"\n{i}. {p['tag']} — {p['miss_count']} signal(s) "
            f"[sources: {', '.join(p['sources'])}]"
        )
        for snippet in p['evidence_snippets']:
            parts.append("   Evidence:")
            for line in snippet.splitlines():
                parts.append(f"     {line}")
    parts.append("")
    parts.append(
        "Draft 2-3 in-class discussion questions targeted at the highest-priority gaps. "
        "Return ONLY the bullet list."
    )
    return "\n".join(parts)


def _buildDiscussionPromptCloud(priorities):
    """Build the redacted (cloud-safe) discussion-question prompt.

    Contains only ``tag`` strings + integer ``miss_count`` + integer
    ``n_themes`` (the count of evidence snippets, not their content). No
    transcripts, no criteria_evaluations, no theme bullet text — those would
    be derived from student-submitted content. Unit-tested as a privacy guardrail.
    """
    parts = ["Top cohort topics (ranked by recent miss count):"]
    for i, p in enumerate(priorities, start=1):
        parts.append(
            f"{i}. {p['tag']} — {p['miss_count']} miss(es) "
            f"({len(p['evidence_snippets'])} related theme cluster(s))"
        )
    parts.append("")
    parts.append(
        "Draft 2-3 in-class discussion questions targeted at the highest-priority topics. "
        "Return ONLY the bullet list."
    )
    return "\n".join(parts)


def _suggestDiscussionQuestions(priorities, *, cloud_questions):
    """Call the appropriate model to draft the discussion-question bullets.

    Default (``cloud_questions=False``): local Gemma 4 + full-fidelity prompt.
    Opt-in (``cloud_questions=True``): cloud Gemini 3 + redacted prompt. Logs
    the chosen model + host + prompt variant so the operator can verify which
    path ran.
    """
    if not priorities:
        logger.info("Discussion-question step skipped: no priorities (all-pass cohort).")
        return "_(no significant gaps detected — cohort is on track)_"

    if cloud_questions:
        client = _make_client(cloud=True)
        model = DEFAULT_TEXT_MODEL
        host = OLLAMA_CLOUD_HOST
        system_prompt = _DISCUSSION_CLOUD_SYSTEM_PROMPT
        user_prompt = _buildDiscussionPromptCloud(priorities)
        variant = 'cloud'
    else:
        client = _make_client(cloud=False)
        model = DEFAULT_MODEL
        host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
        system_prompt = _DISCUSSION_LOCAL_SYSTEM_PROMPT
        user_prompt = _buildDiscussionPromptLocal(priorities)
        variant = 'local'

    logger.info(f"Discussion-question generation: model={model} host={host} prompt_variant={variant}")
    try:
        resp = _chat_with_retry(
            client,
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0.4},
        )
        return resp["message"]["content"].strip()
    except Exception as e:
        logger.warning(f"Discussion-question generation failed: {e}")
        return f"_(discussion-question generation failed: {e})_"


# ---------------------------------------------------------------------------
# Phase E — Render and orchestrate
# ---------------------------------------------------------------------------


def _renderQuizPerformanceSection(quiz_misses):
    """Render the quiz-performance Markdown table sorted by descending miss count."""
    lines = ["## Quiz performance", ""]
    if not quiz_misses:
        lines.append("_(no quiz miss data in this window)_")
        lines.append("")
        return lines

    aggregated = Counter()
    for entry in quiz_misses:
        for tag, n in entry['n_missed_per_tag'].items():
            aggregated[tag] += n

    if not aggregated:
        lines.append("_(no missed questions in this window)_")
        lines.append("")
        return lines

    lines.append("| Tag | Total misses | Contributing quizzes |")
    lines.append("|---|---:|---|")
    contrib_by_tag = {}
    for entry in quiz_misses:
        for tag in entry['n_missed_per_tag']:
            contrib_by_tag.setdefault(tag, []).append(entry['quiz_name'])
    for tag, n in sorted(aggregated.items(), key=lambda kv: (-kv[1], kv[0])):
        contributors = ', '.join(sorted(set(contrib_by_tag.get(tag, []))))
        lines.append(f"| {tag} | {n} | {contributors} |")
    lines.append("")
    return lines


def _renderFollowupSection(followup_themes, followup_blocks):
    """Render the follow-up reply themes section, one block per (quiz, question)."""
    lines = ["## Follow-up reply themes", ""]
    if not followup_themes:
        lines.append("_(no follow-up replies in this window)_")
        lines.append("")
        return lines

    name_by_quiz = {entry['quiz_id']: entry['quiz_name'] for entry in followup_blocks}
    for (quiz_id, question_id), bullets in followup_themes.items():
        quiz_name = name_by_quiz.get(quiz_id, quiz_id)
        lines.append(f"### {quiz_name} — question {question_id}")
        lines.append("")
        lines.append(bullets)
        lines.append("")
    return lines


def _renderRecordingSection(recording_results):
    """Render the per-assignment media-recording themes + tag-coverage section."""
    lines = ["## Media-recording themes", ""]
    if not recording_results:
        lines.append("_(no media-recording assignments in this window)_")
        lines.append("")
        return lines

    for entry in recording_results:
        lines.append(f"### Assignment {entry['assignment_id']}")
        lines.append("")
        ranked = sorted(
            (entry.get('tag_to_indices') or {}).items(),
            key=lambda kv: (-len(kv[1]), kv[0]),
        )
        nonzero = [(t, idxs) for t, idxs in ranked if idxs]
        if nonzero:
            lines.append("| Tag | Students |")
            lines.append("|---|---:|")
            for tag, idxs in nonzero:
                lines.append(f"| {tag} | {len(idxs)} |")
            lines.append("")
        lines.append(entry.get('themes_md') or '')
        lines.append("")
    return lines


def _renderDigest(course_label, since_date, today, quiz_misses, followup_themes,
                  followup_blocks, recording_results, discussion_md, days):
    """Assemble the five-section Markdown digest."""
    lines = []
    lines.append(f"# Pre-class digest — {course_label}")
    lines.append("")
    lines.append(
        f"_Generated {today.isoformat()} covering the last {days} day(s) "
        f"({since_date.isoformat()} — {today.isoformat()})._"
    )
    lines.append("")

    lines.extend(_renderQuizPerformanceSection(quiz_misses))
    lines.extend(_renderFollowupSection(followup_themes, followup_blocks))
    lines.extend(_renderRecordingSection(recording_results))

    lines.append("## Suggested in-class discussion questions")
    lines.append("")
    lines.append(discussion_md or "_(no discussion questions generated)_")
    lines.append("")

    return "\n".join(lines)


def _haveAnySignal(quiz_misses, followup_blocks, recording_results):
    """Return True if any of the three signal sources produced data in window."""
    if quiz_misses:
        return True
    if followup_blocks:
        return True
    if recording_results:
        return True
    return False


def prepClassDigest(course, days=7, cloud_questions=False):
    """Top-level orchestrator: load → analyze → render → write.

    Returns the path of the written digest, or ``None`` when the window had no
    signal (in which case nothing is written).
    """
    today = datetime.now(timezone.utc).date()
    since_date = today - timedelta(days=days)
    data_path = course.config.data_path

    print(f"\nBuilding digest for last {days} day(s) ({since_date.isoformat()} → {today.isoformat()})")

    print("  Loading quiz misses...")
    quiz_misses = _loadQuizMisses(data_path, since_date)
    print(f"    {len(quiz_misses)} quiz(zes) with miss data in window")

    print("  Loading follow-up assessments...")
    followup_blocks = _loadFollowupAssessments(data_path, since_date)
    n_followup_questions = sum(len(b['rows_by_question']) for b in followup_blocks)
    print(f"    {len(followup_blocks)} quiz(zes), {n_followup_questions} question(s) with struggling rows")

    print("  Loading media-recording transcripts...")
    rec_entries = _loadRecordingsInWindow(data_path, since_date)
    print(f"    {len(rec_entries)} recording assignment(s) in window")

    if not _haveAnySignal(quiz_misses, followup_blocks, rec_entries):
        print(f"\nNo quiz/follow-up/recording activity in the last {days} day(s) — nothing to digest.")
        return None

    needs_local = bool(followup_blocks) or bool(rec_entries) or not cloud_questions
    local_client = None
    if needs_local:
        local_client = _make_client(cloud=False)
        try:
            local_client.list()
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at its configured host "
                f"({os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}). "
                f"Is the Ollama server running? Original error: {e}"
            ) from e

    followup_themes = {}
    if followup_blocks:
        print("  Summarizing follow-up reply themes (Gemma 4)...")
        followup_themes = _summarizeFollowupThemes(followup_blocks, local_client, DEFAULT_MODEL)

    recording_results = []
    if rec_entries:
        print("  Analyzing media-recording transcripts (Gemma 4)...")
        for entry in rec_entries:
            recording_results.append(_analyzeOneRecordingsBatch(entry, local_client, DEFAULT_MODEL))

    priorities = _buildDigestPriorities(quiz_misses, followup_themes, recording_results)
    print(f"  Top {len(priorities)} priority gap(s) identified.")

    print("  Drafting in-class discussion questions...")
    discussion_md = _suggestDiscussionQuestions(priorities, cloud_questions=cloud_questions)

    course_code = getattr(course.canvas_course, 'course_code', '') or course.canvas_course.name
    report = _renderDigest(
        course_code, since_date, today, quiz_misses, followup_themes,
        followup_blocks, recording_results, discussion_md, days,
    )
    out_path = data_path / f"class_digest_{today_str()}.md"
    out_path.write_text(report)
    print(f"\nSaved digest to {out_path.name}")
    logger.info(
        f"prepClassDigest: wrote {out_path} "
        f"(days={days}, cloud_questions={cloud_questions}, priorities={len(priorities)})"
    )
    return out_path
