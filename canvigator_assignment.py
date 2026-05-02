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
import subprocess
from datetime import datetime, timezone

import pandas as pd

from canvigator_utils import today_str, selectFromList
from canvigator_llm import transcribe_audio, _make_client, DEFAULT_AUDIO_MODEL

logger = logging.getLogger(__name__)

# ffmpeg needs https in its protocol whitelist when an input is a DASH/HLS
# manifest whose fragments are served over https — Canvas (Kaltura/Studio)
# media-recording playback is delivered as DASH, so without this the demuxer
# fails with "Protocol 'https' not on whitelist" when loading fragments.
_FFMPEG_PROTOCOL_WHITELIST = 'file,crypto,data,https,tls,tcp,http,httpproxy'


def _fetchAudio(url, audio_path):
    """Fetch a Canvas media URL and produce a 16 kHz mono PCM WAV at ``audio_path`` via ffmpeg.

    Handles direct media URLs, HLS playlists, and DASH manifests in one pass.
    The output is always 16 kHz mono PCM WAV — that's the format Ollama's
    media-detection routes to the audio path on Gemma multimodal models;
    AAC-in-m4a gets misclassified and rejected as ``image: unknown format``.
    Returns True on success; False when ffmpeg is missing or the fetch fails
    (the caller logs and skips).
    """
    cmd = [
        'ffmpeg', '-y', '-loglevel', 'error',
        '-protocol_whitelist', _FFMPEG_PROTOCOL_WHITELIST,
        '-i', url,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000',
        '-ac', '1',
        str(audio_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except FileNotFoundError:
        logger.warning(
            "ffmpeg not found in PATH; cannot fetch audio for "
            f"{audio_path.name}. Install ffmpeg (e.g. 'brew install ffmpeg') to enable."
        )
        return False
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors='ignore') if e.stderr else ''
        logger.warning(f"ffmpeg failed to fetch audio for {audio_path.name}: {stderr[:300]}")
        return False


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


def _promptForGrade(student_label, transcript, audio_path, points_possible):
    """Show a transcript review screen and prompt for the points to award.

    Returns the grade as a float, or ``None`` when the instructor types
    ``s``/``skip`` to record no grade. Empty input awards ``points_possible``
    (full credit). Negative or non-numeric input re-prompts.
    """
    print("\n" + "-" * 60)
    print(f"  {student_label}")
    print("-" * 60)
    if transcript:
        print(f"  Transcript:\n    {transcript}")
    else:
        print(f"  (transcription empty — listen to {audio_path})")
    while True:
        raw = input(f"  Points to award (Enter for {points_possible}, 's' to skip) [{points_possible}]: ").strip()
        if not raw:
            return float(points_possible)
        if raw.lower() in ('s', 'skip'):
            return None
        try:
            value = float(raw)
        except ValueError:
            print("  Invalid; please enter a non-negative number, 's' to skip, or Enter for full credit.")
            continue
        if value < 0:
            print("  Invalid; please enter a non-negative number, 's' to skip, or Enter for full credit.")
            continue
        return value


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

    def _buildRecordingRow(self, submission, student_name, audio_path, transcript, grade=None):
        """Return a dict with the columns written to the recordings CSV.

        ``grade`` is the points awarded (float) or ``None`` when no grade was
        recorded for this row. ``graded_at`` is set to the current UTC ISO
        timestamp when ``grade`` is not None, otherwise empty.
        """
        return {
            'student_id': submission.user_id,
            'student_name': student_name,
            'submission_id': getattr(submission, 'id', None),
            'submitted_at': getattr(submission, 'submitted_at', None),
            'audio_path': audio_path,
            'transcript': transcript,
            'transcribed_at': datetime.now(timezone.utc).isoformat(),
            'grade': '' if grade is None else grade,
            'graded_at': datetime.now(timezone.utc).isoformat() if grade is not None else '',
        }

    def _postGrade(self, submission, points, dry_run):
        """Post ``points`` on the submission via ``Submission.edit``.

        Returns True if a real grade write succeeded, False on dry-run or failure.
        """
        student_id = getattr(submission, 'user_id', '?')
        if dry_run:
            logger.info(f"WOULD set grade {points} on submission user_id={student_id}")
            print(f"     [DRY-RUN] would post grade={points}")
            return False
        try:
            submission.edit(submission={'posted_grade': points})
            logger.info(f"Posted grade {points} on submission user_id={student_id}")
            print(f"     Posted grade={points}")
            return True
        except Exception as e:
            logger.warning(f"Grade post failed for user_id={student_id}: {e}")
            print(f"     ERROR posting grade: {e}")
            return False

    def _writeRecordingsCsv(self, rows):
        """Write the recordings CSV with the canonical column order."""
        df = pd.DataFrame(rows, columns=[
            'student_id', 'student_name', 'submission_id',
            'submitted_at', 'audio_path', 'transcript', 'transcribed_at',
            'grade', 'graded_at',
        ])
        csv_path = self.config.data_path / f"{self.assignment_prefix}recordings_{today_str()}.csv"
        df.to_csv(csv_path, index=False)
        return csv_path, df

    def getMediaRecordings(self, auto_grade=False, dry_run=False):
        """Fetch every submission, transcribe locally, optionally review and grade interactively.

        For each submitter, ffmpeg pulls the media URL directly (handling DASH
        manifests, HLS playlists, and direct media URLs uniformly) and writes
        a 16 kHz mono PCM WAV to ``data/<course>/media_recordings/assignment<id>/``.
        The local audio model transcribes that file.

        Default: after each transcription, the transcript is displayed and the
        instructor is prompted for the points to award. Enter awards full credit;
        a number overrides; ``s``/``skip`` records no grade. With
        ``auto_grade=True``, the per-student display and prompt are suppressed
        and ``points_possible`` is awarded automatically. ``dry_run=True`` skips
        Canvas writes on either path; the CSV is still written.
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
        try:
            for i, sub in enumerate(submitted, start=1):
                student_id = getattr(sub, 'user_id', None)
                student_name = student_lookup.get(student_id, '?')
                url, _ext = self._extractAudioUrl(sub)
                if not url:
                    logger.info(f"No media payload on submission user_id={student_id}; skipping.")
                    print(f"  [{i}/{len(submitted)}] {student_name} — no media payload, skipping")
                    continue

                local_path = media_dir / f"{student_id}_{getattr(sub, 'id', 'sub')}.wav"
                audio_rel = str(local_path.relative_to(self.config.data_path.parent))
                print(f"  [{i}/{len(submitted)}] {student_name} — fetching audio...", end="", flush=True)
                if not _fetchAudio(url, local_path):
                    print(" FAILED")
                    rows.append(self._buildRecordingRow(sub, student_name, audio_rel, ""))
                    continue
                print(" transcribing...", end="", flush=True)
                transcript = transcribe_audio(str(local_path), client, audio_model)
                chars = len(transcript)
                print(f" done ({chars} chars)")

                if auto_grade:
                    grade_value = float(points_possible)
                else:
                    student_label = student_name if student_name and student_name != '?' else f"id={student_id}"
                    grade_value = _promptForGrade(student_label, transcript, audio_rel, points_possible)

                if grade_value is not None:
                    if self._postGrade(sub, grade_value, dry_run):
                        n_graded += 1
                rows.append(self._buildRecordingRow(sub, student_name, audio_rel, transcript, grade_value))
        except KeyboardInterrupt:
            print("\nInterrupted — flushing partial CSV before exiting.")
            if rows:
                csv_path, _ = self._writeRecordingsCsv(rows)
                print(f"Saved {len(rows)} partial row(s) to {csv_path.name}")
            raise

        if not rows:
            print("\nNo recordings retrieved.")
            return

        csv_path, df = self._writeRecordingsCsv(rows)
        print(f"\nSaved {len(df)} transcript(s) to {csv_path.name}")
        if n_graded:
            verb = "Would have graded" if dry_run else "Graded"
            print(f"{verb} {n_graded} submission(s).")
        logger.info(
            f"getMediaRecordings: wrote {len(df)} rows to {csv_path} "
            f"(auto_grade={auto_grade}, dry_run={dry_run}, n_graded={n_graded})"
        )
