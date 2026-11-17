"""Follow-up questions delivered as a Canvas assignment.

The message-based chain in ``canvigator_quiz`` sends each student a personal
Canvas conversation. When every student gets the same follow-up question, an
assignment is the better vehicle: one item in the course, an ordinary
submission box, a gradebook column, and feedback as a submission comment.

Three tasks, each driven by a ``CanvigatorQuiz``:

* ``create-follow-up-assignment`` — one assignment per selected follow-up question.
* ``assess-follow-up-assignment`` — download submissions and run the local model.
* ``review-return-follow-up-assignment`` — the instructor hears/sees each
  submission and records their own verdict *before* the model's is shown, then
  reconciles the two and returns feedback with completion credit.

Assessments share the quiz's ``*_followup_assessments.csv`` with the
message-based chain; rows written here carry an ``assignment_id``.
"""
import html
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from canvigator_utils import prompt_for_index
from canvigator_assignment import CanvigatorAssignment, _promptDueAt, _promptInt, _promptYesNo
from canvigator_quiz import _composeFollowUpFeedbackMessage, _normId

logger = logging.getLogger(__name__)

# File types a 'draw' submission may be: what the local model can read directly
# (images) or after rasterizing (PDF, the usual phone-scanner output).
DRAW_EXTENSIONS = ['jpg', 'jpeg', 'png', 'pdf']

MANIFEST_COLUMNS = [
    'assignment_id', 'question_id', 'position', 'question_mode', 'keywords',
    'open_ended_question', 'points_possible', 'due_at', 'published', 'html_url', 'created_at',
]

# Columns the review task writes text into; read back from CSV they may be
# all-NaN floats, which pandas will not accept a string in.
_REVIEW_TEXT_COLUMNS = ('result', 'feedback', 'instructor_result', 'reviewed_at', 'sent_at')

_BLIND_VERDICTS = {'p': 'pass', 'f': 'fail', 'n': 'no_media'}


def _now():
    """Return the current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Assignments manifest
# ---------------------------------------------------------------------------

def _manifestPath(quiz):
    """Return the persistent (non-dated) follow-up assignments manifest path for this quiz."""
    return quiz.config.data_path / f"{quiz.config.quiz_prefix}{quiz.canvas_quiz.id}_followup_assignments.csv"


def _loadManifest(quiz):
    """Load the follow-up assignments manifest, or an empty frame when none exists yet."""
    path = _manifestPath(quiz)
    if not path.exists():
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.read_csv(path)


def _selectAssignmentRow(quiz):
    """Return the manifest row (a dict) of the follow-up assignment to work on.

    With one assignment for the quiz, that one; with several, list them in
    quiz order and ask. Raises FileNotFoundError when none has been created.
    """
    manifest = _loadManifest(quiz)
    if manifest.empty:
        raise FileNotFoundError(
            f"No follow-up assignment found for quiz '{quiz.quiz_name}'. "
            f"Run 'create-follow-up-assignment' first."
        )
    rows = manifest.sort_values('position').to_dict('records')
    if len(rows) == 1:
        return rows[0]
    print("\nThis quiz has more than one follow-up assignment:")
    for i, row in enumerate(rows, start=1):
        print(f"[ {i} ] Q{row['position']} — {row['keywords']} ({row['question_mode']})")
    idx = prompt_for_index("\nSelect the assignment (using index in '[ ]'): ", len(rows) - 1)
    return rows[idx]


# ---------------------------------------------------------------------------
# create-follow-up-assignment
# ---------------------------------------------------------------------------

def _buildAssignmentPayload(quiz_title, position, keywords, question_text, question_mode,
                            points_possible, due_at, published):
    """Return the Canvas assignment payload for one follow-up question.

    'explain' questions accept only a media recording; 'draw' questions accept
    only an uploaded image or PDF.
    """
    if question_mode == 'draw':
        submission_types = ['online_upload']
        instructions = (
            "Draw your answer on paper (or a tablet), then upload a photo or scan "
            "of your drawing (JPG, PNG, or PDF)."
        )
    else:
        submission_types = ['media_recording']
        instructions = (
            "Record a short voice response explaining your answer, using the "
            "media recorder on this assignment's submission page."
        )
    topic = f" (topic: {html.escape(str(keywords))})" if keywords else ""
    question_html = html.escape(str(question_text)).replace('\n', '<br>')
    description = (
        f"<p>This is a follow-up question to <strong>{html.escape(str(quiz_title))}</strong>, "
        f"question {position}{topic}.</p>"
        f"<p><strong>{question_html}</strong></p>"
        f"<p>{instructions}</p>"
        f"<p>You will receive full credit for a genuine attempt, along with feedback on your "
        f"answer. You may resubmit as many times as you'd like — only your most recent "
        f"submission is assessed.</p>"
    )
    payload = {
        'name': f"{quiz_title} - Q{position} Follow-Up",
        'description': description,
        'submission_types': submission_types,
        'points_possible': points_possible,
        'published': published,
    }
    if question_mode == 'draw':
        payload['allowed_extensions'] = DRAW_EXTENSIONS
    if due_at:
        payload['due_at'] = due_at
    return payload


def _printPayloadPreview(payload):
    """Print what the assignment would look like on Canvas."""
    print("\n=== DRY RUN MODE - No assignment will be created on Canvas ===\n")
    print(f"  Name:             {payload['name']}")
    print(f"  Submission types: {', '.join(payload['submission_types'])}")
    if 'allowed_extensions' in payload:
        print(f"  Allowed files:    {', '.join(payload['allowed_extensions'])}")
    print(f"  Points possible:  {payload['points_possible']}")
    print(f"  Due at:           {payload.get('due_at', '(none)')}")
    print(f"  Published:        {payload['published']}")
    print(f"  Description (HTML):\n    {payload['description']}")


def createFollowUpAssignment(quiz, dry_run=False):
    """Create a Canvas assignment for the instructor-selected open-ended follow-up question.

    Uses the row with ``selected_question == 1`` in the quiz's
    ``*_open_ended_*.csv`` (asking which when several are selected). The
    assignment goes to every student in the course and is recorded in the
    quiz's ``*_followup_assignments.csv`` so the assess and review tasks can
    find it. A question that already has an assignment is not given a second
    one. ``dry_run`` previews the assignment without creating it.
    """
    question_info = quiz._loadQuestionInfo()
    open_ended_rows = quiz._loadOpenEndedQuestions()
    if not open_ended_rows:
        print("No `selected_question=1` rows found in the open-ended CSV.")
        print("Edit the CSV to mark the follow-up question to use and re-run.")
        return

    question_id = quiz._chooseFollowUpQuestion(open_ended_rows, question_info)
    oe_row = open_ended_rows[question_id]
    info = question_info.get(question_id)
    if info is None:
        print(f"Question {question_id} from open-ended CSV not found in questions metadata — re-run 'get-quiz-questions --tag'.")
        return
    position, keywords = info['position'], info['keywords']
    question_mode = oe_row.get('question_mode')
    question_mode = 'draw' if isinstance(question_mode, str) and question_mode.strip().lower() == 'draw' else 'explain'
    question_text = oe_row['open_ended_question']

    manifest = _loadManifest(quiz)
    existing = manifest[manifest['question_id'].astype(int) == int(question_id)]
    if not existing.empty:
        print(f"\nQ{position} already has a follow-up assignment: {existing.iloc[0]['html_url']}")
        print(f"(To replace it, delete it on Canvas and remove its row from {_manifestPath(quiz).name}.)")
        return

    print(f"\nSelected follow-up: Q{position} — {keywords}")
    print(f"  Mode: {question_mode}")
    print(f"  Follow-up: {question_text}\n")

    points_possible = _promptInt("Points possible", 1)
    due_at = _promptDueAt()
    published = _promptYesNo("Publish immediately?", default=False)
    payload = _buildAssignmentPayload(
        quiz.canvas_quiz.title, position, keywords, question_text, question_mode,
        points_possible, due_at, published,
    )
    if dry_run:
        _printPayloadPreview(payload)
        return

    assignment = quiz.canvas_course.canvas_course.create_assignment(payload)
    row = {
        'assignment_id': assignment.id,
        'question_id': question_id,
        'position': position,
        'question_mode': question_mode,
        'keywords': keywords,
        'open_ended_question': question_text,
        'points_possible': points_possible,
        'due_at': due_at or '',
        'published': published,
        'html_url': getattr(assignment, 'html_url', ''),
        'created_at': _now(),
    }
    manifest = pd.concat([manifest, pd.DataFrame([row], columns=MANIFEST_COLUMNS)], ignore_index=True)
    manifest.to_csv(_manifestPath(quiz), index=False, columns=MANIFEST_COLUMNS)

    state = "published" if published else "unpublished"
    print(f"\nCreated assignment: '{assignment.name}' (id={assignment.id}, {state})")
    print(f"  Canvas URL: {row['html_url']}")
    print(f"  Recorded in: {_manifestPath(quiz).name}")
    logger.info(f"Created follow-up assignment '{assignment.name}' (id={assignment.id}) for question_id {question_id}")


# ---------------------------------------------------------------------------
# assess-follow-up-assignment
# ---------------------------------------------------------------------------

def _submittedAttempts(assignment):
    """Return the assignment's submissions that carry an actual attempt (Canvas lists every student)."""
    return [
        s for s in assignment.get_submissions()
        if getattr(s, 'attempt', None) and getattr(s, 'workflow_state', '') in ('submitted', 'graded')
    ]


def _selectSubmissionsToAssess(replies, existing_index):
    """Return the submissions that need (re)grading, printing what was skipped.

    Skipped: an attempt already assessed — whether its feedback was returned
    or is still waiting for review — unless that assessment ended in a model
    error. A newer attempt is always assessed.
    """
    to_assess = []
    n_returned = n_unchanged = 0
    for r in replies:
        existing = existing_index.get((int(r['student_id']), int(r['question_id'])))
        if existing is not None and _normId(existing.get('attempt')) == _normId(r['attempt']):
            if int(existing.get('sent_assessment', 0) or 0) == 1:
                n_returned += 1
                continue
            if str(existing.get('result', '')) != 'error':
                n_unchanged += 1
                continue
        to_assess.append(r)
    if n_returned:
        print(f"  Skipping {n_returned} submission(s) whose feedback was already returned.")
    if n_unchanged:
        print(f"  Keeping {n_unchanged} existing assessment(s) — no newer attempt since they were assessed.")
    return to_assess


def _downloadSubmissionMedia(quiz, submission, reply, replies_dir):
    """Download one submission's recording or drawing and set the reply's media paths in place."""
    msg = {'id': f"a{reply['attempt']}"}
    if reply['question_mode'] == 'draw':
        msg['attachments'] = getattr(submission, 'attachments', None) or []
    else:
        url, _ext = CanvigatorAssignment._extractAudioUrl(submission)
        if url:
            msg['media_comment'] = {'url': url}
    file_prefix = f"{quiz.config.quiz_prefix}{quiz.canvas_quiz.id}_"
    attachment_path, audio_path = quiz._downloadReplyMedia(
        msg, reply['student_id'], reply['student_name'], reply['question_id'], file_prefix, replies_dir,
    )
    reply['attachment_path'] = attachment_path or ''
    reply['audio_path'] = audio_path or ''
    reply['media_path'] = reply['attachment_path'] if reply['question_mode'] == 'draw' else reply['audio_path']


def _quizStatusByStudent(quiz, question_id, question_info):
    """Return ``{student_id: 'missed' | 'correct'}`` on the original quiz question, from fresh submissions.

    The quiz is built without its student data (so a run with nothing new to
    assess stays quick); the report is downloaded here, only once it is needed.
    """
    if quiz.quiz_df is None:
        quiz.getQuizData()
    quiz._ensureFreshSubmissions()
    subs_by_q_df = quiz.all_subs_by_question
    if subs_by_q_df is None or subs_by_q_df.empty:
        return {}
    classification = quiz._classifyStudentsByQuestionResult(question_id, subs_by_q_df, question_info)
    return {int(sid): status for sid, status in classification.items()}


def assessFollowUpAssignment(quiz, chosen_model=None, save_transcript=False):
    """Download a follow-up assignment's submissions and assess them with the local LLM.

    Takes each student's latest attempt, downloads the recording or drawing to
    ``data/<course>/replies/``, and merges the assessments into the quiz's
    persistent ``*_followup_assessments.csv``. An attempt already assessed is
    not re-graded; a newer attempt replaces an unreturned row, or is added as
    a new row when feedback on the earlier one was already returned. Each row
    records the student's status on the original quiz question at assessment
    time (``missed`` / ``correct`` / ``no_attempt``).

    Nothing is written to Canvas, and no pass/fail is printed: the instructor
    judges each submission in review-return-follow-up-assignment before seeing
    the model's verdict. ``chosen_model`` and ``save_transcript`` are as in
    assess-replies.
    """
    mrow = _selectAssignmentRow(quiz)
    question_id = int(mrow['question_id'])
    question_mode = mrow['question_mode']
    assignment_id = int(mrow['assignment_id'])

    question_info = quiz._loadQuestionInfo()
    oe_row = quiz._loadOpenEndedQuestions().get(question_id)
    if oe_row is None:
        print(f"\nNo selected open-ended question found for question_id {question_id}. "
              f"Mark it with selected_question=1 in the open-ended CSV.")
        return

    assignment = quiz.canvas_course.canvas_course.get_assignment(assignment_id)
    submissions = {int(s.user_id): s for s in _submittedAttempts(assignment)}
    print(f"\nAssignment: {assignment.name}")
    print(f"  Submissions: {len(submissions)}")
    if not submissions:
        print("No student submissions to assess.")
        return

    names = {s['id']: s.get('name', '') for s in quiz.canvas_course.students}
    replies = [{
        'student_id': sid,
        'student_name': names.get(sid) or f"id={sid}",
        'question_id': question_id,
        'question_mode': question_mode,
        'attempt': int(sub.attempt),
        'submitted_at': getattr(sub, 'submitted_at', '') or '',
    } for sid, sub in submissions.items()]

    existing_df = quiz._loadOrMigrateAssessments()
    to_assess = _selectSubmissionsToAssess(replies, quiz._indexAssessments(existing_df))
    if not to_assess:
        print("  Nothing new to assess.")
        return

    replies_dir = quiz.config.data_path / "replies"
    replies_dir.mkdir(exist_ok=True)
    print(f"  Downloading {len(to_assess)} submission(s)...")
    for reply in to_assess:
        _downloadSubmissionMedia(quiz, submissions[reply['student_id']], reply, replies_dir)

    print("\nChecking each student's result on the original quiz question...")
    quiz_status = _quizStatusByStudent(quiz, question_id, question_info)

    by_student = {r['student_id']: dict(r) for r in to_assess}  # keeps the relative media paths
    quiz._absolutizeMediaPaths(to_assess)
    locked_examples = quiz._collectLockedExamples(existing_df, question_id)
    if locked_examples:
        print(f"  Using {len(locked_examples)} instructor-approved example(s) for calibration.")

    import canvigator_llm
    results = canvigator_llm.assess_replies(
        to_assess, oe_row, model=chosen_model, locked_examples=locked_examples,
        save_transcript=save_transcript, show_verdicts=False,
    )
    for res in results:
        reply = by_student[int(res['student_id'])]
        res.update({
            'assignment_id': assignment_id,
            'attempt': reply['attempt'],
            'submitted_at': reply['submitted_at'],
            'quiz_status': quiz_status.get(int(res['student_id']), 'no_attempt'),
            'media_path': reply['media_path'],
            'sent_assessment': 0,
            'llm_result': res['result'],
            'llm_confidence': res['confidence'],
            'llm_feedback': res['feedback'],
        })

    quiz._writeAssessmentsCsv(quiz._mergeAssessments(existing_df, results))
    n_no_media = sum(1 for r in results if r['result'] == 'no_media')
    n_error = sum(1 for r in results if r['result'] == 'error')
    print(f"\n  Assessed this run: {len(results)} ({len(results) - n_no_media - n_error} graded by the model, "
          f"{n_no_media} with no usable media, {n_error} model error)")
    print("  Next: run 'review-return-follow-up-assignment' to hear/see each submission, "
          "give your own verdict, and return feedback.")
    print("  (Open the assessments CSV only after that review, so your verdicts stay independent of the model's.)")


# ---------------------------------------------------------------------------
# review-return-follow-up-assignment
# ---------------------------------------------------------------------------

def _openWithSystemViewer(path):
    """Open a file in the operating system's default application (returns immediately)."""
    if sys.platform == 'darwin':
        subprocess.run(['open', str(path)])
    elif os.name == 'nt':
        os.startfile(str(path))
    else:
        subprocess.run(['xdg-open', str(path)])


def _playAudio(path):
    """Play a recording in the foreground; Ctrl-C stops the playback, not the task."""
    if shutil.which('afplay'):
        cmd = ['afplay', str(path)]
    elif shutil.which('ffplay'):
        cmd = ['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', str(path)]
    else:
        _openWithSystemViewer(path)
        input("  The recording is open in your media player. Press Enter once you have listened to it: ")
        return
    print("  Playing the recording (Ctrl-C stops playback)...")
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print()


def _presentSubmission(media_path, question_mode, canvas_url):
    """Put the student's submission in front of the instructor: play the audio or open the drawing.

    When no file was downloaded, point at the submission on Canvas instead.
    """
    if not media_path or not Path(media_path).is_file():
        print("  No recording/image file was downloaded for this submission.")
        print(f"  Look at it on Canvas: {canvas_url}")
        input("  Press Enter once you have checked it: ")
    elif question_mode == 'draw':
        _openWithSystemViewer(media_path)
        input("  The drawing is open in your image viewer. Press Enter once you have looked at it: ")
    else:
        _playAudio(media_path)


def _editText(text):
    """Open ``text`` in the instructor's editor ($VISUAL / $EDITOR, else nano or notepad) and return the result."""
    editor = os.environ.get('VISUAL') or os.environ.get('EDITOR') or ('notepad' if os.name == 'nt' else 'nano')
    with tempfile.NamedTemporaryFile('w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write(text)
        tmp_path = f.name
    try:
        subprocess.run(shlex.split(editor) + [tmp_path])
        return Path(tmp_path).read_text(encoding='utf-8').strip()
    finally:
        os.unlink(tmp_path)


def _cell(row, col):
    """Return a CSV cell as a stripped string ('' when missing/NaN)."""
    value = row.get(col, '')
    return '' if pd.isna(value) else str(value).strip()


def _askBlindVerdict(media_path, question_mode, canvas_url):
    """Present the submission, then ask for the instructor's own verdict.

    Returns 'pass', 'fail', or 'no_media'; or 'skip' / 'quit'. The prompt is
    reached only after the submission has been presented, and nothing from the
    model has been printed yet.
    """
    _presentSubmission(media_path, question_mode, canvas_url)
    while True:
        choice = input("  Your verdict — [p]ass / [f]ail / [n]o usable submission / [r]eplay / [s]kip / [q]uit: ").strip().lower()
        if choice in _BLIND_VERDICTS:
            return _BLIND_VERDICTS[choice]
        if choice == 'r':
            _presentSubmission(media_path, question_mode, canvas_url)
        elif choice == 's':
            return 'skip'
        elif choice == 'q':
            return 'quit'


def _revealModelAssessment(row):
    """Print the transcript and the model's verdict and draft feedback."""
    transcript = _cell(row, 'transcript')
    if transcript:
        print(f"\n  Transcript:\n    {transcript}")
    llm_result = _cell(row, 'llm_result')
    llm_confidence = _cell(row, 'llm_confidence')
    print(f"\n  Local model's verdict: {llm_result or '(none)'}" + (f" ({llm_confidence})" if llm_confidence else ""))
    feedback = _cell(row, 'feedback')
    print(f"  Draft feedback:\n    {feedback or '(none)'}")


def _reconcileResult(blind, llm_result):
    """Return the final result given the instructor's blind verdict and the model's.

    The instructor is asked only when both gave pass/fail and they differ;
    otherwise the instructor's verdict stands.
    """
    if blind in ('pass', 'fail') and llm_result in ('pass', 'fail') and blind != llm_result:
        print(f"\n  You said {blind}; the model said {llm_result}.")
        choice = input(f"  Final result — [p]ass / [f]ail (Enter keeps yours: {blind}): ").strip().lower()
        return {'p': 'pass', 'f': 'fail'}.get(choice, blind)
    return blind


def _chooseFeedback(row, final_result):
    """Return the feedback text to send: the draft, or the instructor's edit of it.

    The editor opens without asking when there is no draft, or when the draft
    is still the model's own text but argues a different result than the final one.
    """
    draft = _cell(row, 'feedback')
    untouched = draft == _cell(row, 'llm_feedback')
    if not draft or (untouched and final_result != _cell(row, 'llm_result')):
        reason = "There is no draft feedback" if not draft else "The draft feedback was written for a different result"
        input(f"\n  {reason} — press Enter to write it in your editor: ")
        return _editText(draft)
    choice = input("\n  Feedback — [a]ccept the draft / [e]dit it (Enter accepts): ").strip().lower()
    return _editText(draft) if choice == 'e' else draft


def _returnToStudent(submission, student_name, body, grade, points_possible, dry_run):
    """Show the exact comment and grade, confirm, and post them. Returns True on a real post."""
    print("\n  Comment to return:")
    for line in body.splitlines():
        print(f"     {line}")
    print(f"  Grade: {grade} / {points_possible}" if grade is not None else "  Grade: none (no usable submission)")
    if dry_run:
        print("  [DRY-RUN] not posted to Canvas.")
        return False
    if input(f"  Return to {student_name}? [y/N]: ").strip().lower() != 'y':
        print(f"  Not returned to {student_name}; your review is saved.")
        return False
    kwargs = {'comment': {'text_comment': body}}
    if grade is not None:
        kwargs['submission'] = {'posted_grade': grade}
    try:
        submission.edit(**kwargs)
    except Exception as e:
        logger.warning(f"Returning feedback failed for {student_name}: {e}")
        print(f"  ERROR: {e}")
        return False
    print(f"  Returned to {student_name}.")
    return True


def _reviewOne(df, idx, submission, question_mode, points_possible, data_root, save, dry_run):
    """Run the blind-verdict → reveal → reconcile → feedback → return sequence for one row.

    Updates ``df`` in place and calls ``save()`` after each step that changes
    it. Returns 'returned', 'reviewed' (saved but not returned), 'skipped', or 'quit'.
    """
    row = df.loc[idx]
    student_name = _cell(row, 'student_name') or '?'
    media_rel = _cell(row, 'media_path')
    media_path = str(data_root / media_rel) if media_rel else ''
    canvas_url = getattr(submission, 'preview_url', '') or ''

    blind = _cell(row, 'instructor_result')
    if blind:
        print(f"  Your earlier verdict: {blind}")
        if input("  Replay the submission first? [y/N]: ").strip().lower() == 'y':
            _presentSubmission(media_path, question_mode, canvas_url)
    else:
        blind = _askBlindVerdict(media_path, question_mode, canvas_url)
        if blind in ('skip', 'quit'):
            return 'skipped' if blind == 'skip' else 'quit'
        df.at[idx, 'instructor_result'] = blind
        df.at[idx, 'reviewed_at'] = _now()
        save()

    _revealModelAssessment(row)
    final_result = _reconcileResult(blind, _cell(row, 'llm_result'))
    feedback = _chooseFeedback(row, final_result)
    df.at[idx, 'result'] = final_result
    df.at[idx, 'feedback'] = feedback
    save()
    if not feedback:
        print("  No feedback entered — nothing returned. Run the task again to finish this one.")
        return 'reviewed'

    body = _composeFollowUpFeedbackMessage(student_name, feedback, final_result)
    grade = points_possible if final_result in ('pass', 'fail') else None
    if not _returnToStudent(submission, student_name, body, grade, points_possible, dry_run):
        return 'reviewed'
    df.at[idx, 'sent_assessment'] = 1
    df.at[idx, 'sent_at'] = _now()
    save()
    return 'returned'


def _pendingReviewRows(df, assignment_id, submissions):
    """Return ``[(row_index, submission)]`` for this assignment's unreturned assessments.

    A row whose student has since submitted a newer attempt is left out (and
    named), since feedback on a superseded attempt should not be returned.
    """
    pending = []
    stale = []
    for idx, row in df.iterrows():
        if _normId(row.get('assignment_id')) != str(assignment_id) or int(row['sent_assessment']) == 1:
            continue
        sub = submissions.get(int(row['student_id']))
        if sub is None or _normId(getattr(sub, 'attempt', None)) != _normId(row.get('attempt')):
            stale.append(_cell(row, 'student_name') or '?')
            continue
        pending.append((idx, sub))
    if stale:
        print(f"  Resubmitted since the last assessment — run 'assess-follow-up-assignment' again for: {', '.join(stale)}")
    return pending


def reviewReturnFollowUpAssignment(quiz, dry_run=False):
    """Review each assessed submission and return feedback and completion credit to the student.

    For every unreturned assessment of the chosen follow-up assignment, the
    instructor first hears the recording or sees the drawing and enters their
    own verdict (saved as ``instructor_result``); only then are the
    transcript, the model's verdict, and its draft feedback shown. The
    instructor settles any disagreement, accepts or edits the feedback, and
    confirms ``[y/N]`` before it is posted as a submission comment. A real
    submission earns full points whether it passed or failed; "no usable
    submission" gets the comment and no grade.

    Progress is saved to ``*_followup_assessments.csv`` after every step, so
    quitting loses nothing. ``dry_run`` runs the whole review and saves it,
    but posts nothing to Canvas.
    """
    mrow = _selectAssignmentRow(quiz)
    assignment_id = int(mrow['assignment_id'])
    path = quiz._assessmentsPath()
    if not path.exists():
        raise FileNotFoundError(
            f"No assessments file found at {path.name}. Run 'assess-follow-up-assignment' first."
        )
    df = quiz._loadOrMigrateAssessments()
    for col in _REVIEW_TEXT_COLUMNS:
        df[col] = df[col].astype(object).where(df[col].notna(), '')

    assignment = quiz.canvas_course.canvas_course.get_assignment(assignment_id)
    points_possible = getattr(assignment, 'points_possible', None) or 0
    submissions = {int(s.user_id): s for s in _submittedAttempts(assignment)}
    print(f"\nAssignment: {assignment.name}")
    pending = _pendingReviewRows(df, assignment_id, submissions)
    if not pending:
        print("No assessed submissions are waiting for review.")
        return

    def save():
        df.to_csv(path, index=False, columns=quiz.ASSESSMENTS_COLUMNS)

    print(f"{'[DRY-RUN] ' if dry_run else ''}Submissions to review: {len(pending)}")
    print("For each one you will hear/see the submission and give your own verdict "
          "before the local model's assessment is shown.")
    counts = {'returned': 0, 'reviewed': 0, 'skipped': 0}
    data_root = quiz.config.data_path.parent
    for i, (idx, sub) in enumerate(pending, start=1):
        row = df.loc[idx]
        print(f"\n--- Submission {i}/{len(pending)}: {_cell(row, 'student_name') or '?'} "
              f"(attempt {_normId(row.get('attempt'))}, submitted {_cell(row, 'submitted_at')}) ---")
        outcome = _reviewOne(df, idx, sub, mrow['question_mode'], points_possible, data_root, save, dry_run)
        if outcome == 'quit':
            break
        counts[outcome] += 1

    print(f"\nReturned {counts['returned']}; reviewed but not returned {counts['reviewed']}; "
          f"skipped {counts['skipped']}. Saved: {path.name}")
    logger.info(f"review-return-follow-up-assignment: {counts} (dry_run={dry_run}) for assignment {assignment_id}")
