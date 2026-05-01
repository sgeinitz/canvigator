"""Media-recording check-in assignments.

Lightweight participation flow built on Canvas Assignments restricted to
``media_recording`` submissions. The instructor posts a single open-ended
prompt ("explain a topic", "what did you struggle with"), students upload
an audio recording from Canvas, and ``get-media-recordings`` fetches every
submission, downloads the audio, and transcribes it locally so the
instructor can scan a CSV instead of listening to N recordings.
"""
import logging
import os
from datetime import datetime, timezone

import pandas as pd
import requests

from canvigator_utils import today_str, selectFromList
from canvigator_llm import transcribe_audio, _make_client, DEFAULT_AUDIO_MODEL

logger = logging.getLogger(__name__)


def _isMediaRecordingAssignment(assignment):
    """Return True if the assignment accepts only media_recording submissions."""
    types = getattr(assignment, 'submission_types', None)
    return list(types or []) == ['media_recording']


def _selectMediaRecordingAssignment(course):
    """List media-recording assignments in the course and prompt the instructor to pick one."""
    candidates = [a for a in course.canvas_course.get_assignments() if _isMediaRecordingAssignment(a)]
    if not candidates:
        raise ValueError(
            "No media-recording assignments found in this course. "
            "Run 'create-media-recording-assignment' first."
        )
    return selectFromList(candidates, "assignment")


def _promptInt(prompt, default):
    """Prompt for a non-negative integer, returning ``default`` on empty input."""
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    try:
        value = int(raw)
        if value < 0:
            raise ValueError
        return value
    except ValueError:
        print(f"  Invalid integer; using default ({default}).")
        return default


def _promptYesNo(prompt, default=False):
    """Prompt for y/n; returns ``default`` on empty input."""
    suffix = "y/N" if not default else "Y/n"
    raw = input(f"{prompt} [{suffix}]: ").strip().lower()
    if not raw:
        return default
    return raw in ('y', 'yes')


def createMediaRecordingAssignment(course):
    """Interactively create a Canvas assignment that accepts only media_recording submissions.

    Mirrors the interactive style of ``CanvigatorCourse.createQuiz``. The instructor
    supplies a title, a prompt body (becomes the assignment description), point value,
    and optional due date; the assignment is created unpublished by default so the
    instructor can review before exposing it to students.
    """
    title = input("Enter assignment title: ").strip()
    if not title:
        print("Assignment title cannot be empty.")
        return

    print("\nEnter the prompt students will see (single line). Examples:")
    print("  'In 30 seconds, explain what we discussed about hash tables today.'")
    print("  'What did you struggle with this week?'")
    body = input("Prompt: ").strip()
    if not body:
        print("Prompt cannot be empty.")
        return
    description_html = f"<p>{body}</p>"

    points_possible = _promptInt("Points possible", 1)

    due_at_raw = input("Due at (ISO 8601, e.g. 2026-05-08T23:59:00; press Enter for none): ").strip()
    due_at = None
    if due_at_raw:
        try:
            datetime.fromisoformat(due_at_raw.replace('Z', '+00:00'))
            due_at = due_at_raw
        except ValueError:
            print(f"  Could not parse '{due_at_raw}' as ISO 8601 — leaving due_at unset.")

    publish_now = _promptYesNo("Publish immediately?", default=False)

    payload = {
        'name': title,
        'description': description_html,
        'submission_types': ['media_recording'],
        'points_possible': points_possible,
        'published': publish_now,
    }
    if due_at:
        payload['due_at'] = due_at

    assignment = course.canvas_course.create_assignment(payload)
    state = "published" if publish_now else "unpublished"
    print(f"\nCreated assignment: '{assignment.name}' (id={assignment.id}, {state})")
    if hasattr(assignment, 'html_url'):
        print(f"  Canvas URL: {assignment.html_url}")
    logger.info(f"Created media-recording assignment: '{assignment.name}' (id={assignment.id})")


class CanvigatorAssignment:
    """A Canvas assignment wrapper focused on media-recording check-ins."""

    def __init__(self, canvas, course, canvas_assignment, config):
        """Capture references and compute an assignment-scoped filename prefix."""
        self.canvas = canvas
        self.course = course
        self.canvas_assignment = canvas_assignment
        self.config = config
        self.assignment_prefix = f"assignment{canvas_assignment.id}_"

    def _extractAudioUrl(self, submission):
        """Return ``(url, ext)`` for an audio/video payload on a submission, or ``(None, None)``.

        Checks ``submission.media_comment`` first (the primary path for
        media_recording submissions), then falls back to scanning
        ``submission.attachments`` for any audio/video mime type.
        """
        media = getattr(submission, 'media_comment', None)
        if media:
            url = media.get('url', '')
            media_type = media.get('media_type', 'audio')
            if url:
                ext = '.m4a' if media_type == 'audio' else '.mp4'
                return url, ext

        attachments = getattr(submission, 'attachments', None) or []
        for att in attachments:
            ct = (att.get('content-type') or att.get('content_type') or '').lower()
            if not (ct.startswith('audio/') or ct.startswith('video/')):
                continue
            url = att.get('url', '')
            if not url:
                continue
            filename = att.get('filename') or att.get('display_name') or ''
            ext = os.path.splitext(filename)[1] or ('.m4a' if ct.startswith('audio/') else '.mp4')
            return url, ext
        return None, None

    def _downloadAudio(self, url, dest_path):
        """Download an audio/video URL to ``dest_path``. Returns True on success."""
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            with open(dest_path, 'wb') as f:
                f.write(resp.content)
            return True
        except Exception as e:
            logger.warning(f"Failed to download {url} -> {dest_path.name}: {e}")
            return False

    def _buildRecordingRow(self, submission, student_name, audio_path, transcript):
        """Return a dict with the columns written to the recordings CSV."""
        return {
            'student_id': submission.user_id,
            'student_name': student_name,
            'submission_id': getattr(submission, 'id', None),
            'submitted_at': getattr(submission, 'submitted_at', None),
            'audio_path': audio_path,
            'transcript': transcript,
            'transcribed_at': datetime.now(timezone.utc).isoformat(),
        }

    def _postGrade(self, submission, points_possible, dry_run):
        """Award full credit on the submission via ``Submission.edit``.

        Returns True if a real grade write succeeded, False on dry-run or failure.
        """
        student_id = getattr(submission, 'user_id', '?')
        if dry_run:
            logger.info(f"WOULD set grade {points_possible} on submission user_id={student_id}")
            print(f"     [DRY-RUN] would post grade={points_possible}")
            return False
        try:
            submission.edit(submission={'posted_grade': points_possible})
            logger.info(f"Posted grade {points_possible} on submission user_id={student_id}")
            print(f"     Posted grade={points_possible}")
            return True
        except Exception as e:
            logger.warning(f"Grade post failed for user_id={student_id}: {e}")
            print(f"     ERROR posting grade: {e}")
            return False

    def getMediaRecordings(self, grade=False, dry_run=False):
        """Fetch every submission, download audio, transcribe locally, write a CSV.

        With ``grade=True``, also posts ``points_possible`` as the grade for every
        submitter. ``dry_run=True`` skips the grade write but still downloads and
        transcribes (downloads are local-only, not Canvas mutations).
        """
        student_lookup = {s['id']: s.get('name', '') for s in self.course.students}
        points_possible = getattr(self.canvas_assignment, 'points_possible', 0) or 0

        media_dir = self.config.data_path / 'media_recordings' / f'assignment{self.canvas_assignment.id}'
        media_dir.mkdir(parents=True, exist_ok=True)

        client = _make_client(cloud=False)
        try:
            client.list()
        except Exception as e:
            raise RuntimeError(
                f"Could not reach Ollama at its configured host ({os.environ.get('OLLAMA_HOST', 'http://localhost:11434')}). "
                f"Is the Ollama server running? Original error: {e}"
            ) from e
        audio_model = DEFAULT_AUDIO_MODEL

        submissions = list(self.canvas_assignment.get_submissions())
        submitted = [s for s in submissions if getattr(s, 'workflow_state', '') in ('submitted', 'graded')]
        print(f"\nFound {len(submitted)} submitted recording(s) (of {len(submissions)} total).")

        rows = []
        n_graded = 0
        for i, sub in enumerate(submitted, start=1):
            student_id = getattr(sub, 'user_id', None)
            student_name = student_lookup.get(student_id, '?')
            url, ext = self._extractAudioUrl(sub)
            if not url:
                logger.info(f"No media payload on submission user_id={student_id}; skipping.")
                print(f"  [{i}/{len(submitted)}] {student_name} — no media payload, skipping")
                continue

            local_name = f"{student_id}_{getattr(sub, 'id', 'sub')}{ext}"
            local_path = media_dir / local_name
            print(f"  [{i}/{len(submitted)}] {student_name} — downloading...", end="", flush=True)
            if not self._downloadAudio(url, local_path):
                print(" FAILED")
                continue
            print(" transcribing...", end="", flush=True)
            transcript = transcribe_audio(str(local_path), client, audio_model)
            audio_rel = str(local_path.relative_to(self.config.data_path.parent))
            rows.append(self._buildRecordingRow(sub, student_name, audio_rel, transcript))
            chars = len(transcript)
            print(f" done ({chars} chars)")

            if grade:
                if self._postGrade(sub, points_possible, dry_run):
                    n_graded += 1

        if not rows:
            print("\nNo recordings retrieved.")
            return

        df = pd.DataFrame(rows, columns=[
            'student_id', 'student_name', 'submission_id',
            'submitted_at', 'audio_path', 'transcript', 'transcribed_at',
        ])
        csv_path = self.config.data_path / f"{self.assignment_prefix}recordings_{today_str()}.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nSaved {len(df)} transcript(s) to {csv_path.name}")
        if grade:
            verb = "Would have graded" if dry_run else "Graded"
            print(f"{verb} {n_graded} submission(s) at {points_possible} point(s) each.")
        logger.info(
            f"getMediaRecordings: wrote {len(df)} rows to {csv_path} "
            f"(grade={grade}, dry_run={dry_run}, n_graded={n_graded})"
        )
