# Media-recording check-ins

Three tasks form an end-to-end loop for short audio check-ins where each
student records a brief reflection (e.g. *"What was hardest on this week's
quiz?"*), the recording is downloaded and transcribed locally, the instructor
grades each submission, and the cohort's transcripts are then summarized
against the quiz topics.

| Task | Summary |
|---|---|
| [`create-media-recording-assignment`](#create-media-recording-assignment) | Create a Canvas assignment that accepts only media_recording submissions. |
| [`get-media-recordings`](#get-media-recordings) | Fetch + transcode + transcribe each submission and either prompt for a grade or auto-award `points_possible`. |
| [`analyze-media-recordings`](#analyze-media-recordings) | Two local-LLM passes against the transcripts: tag classification + theme extraction. |

Per-task `--help` is also available from the CLI:
`python canvigator.py <task> --help`.

← [Back to main README](../README.md)

## End-to-end workflow

1. **`create-media-recording-assignment`** — create the assignment on Canvas.
2. **`get-media-recordings`** — fetch and transcribe each submission, and grade.
3. **`analyze-media-recordings`** — surface tag coverage and recurring themes
   across the cohort.

Audio handling is local-only: transcripts are produced via the local
`OLLAMA_AUDIO_MODEL` (default `gemma4:e4b`) and the cohort analysis runs
against the local `OLLAMA_MODEL` (default `gemma4:31b`). Student audio never
leaves your machine.

---

## `create-media-recording-assignment`

*Create a Canvas assignment that accepts only media_recording submissions.*

Interactively creates a Canvas assignment with `submission_types=['media_recording']`.
Prompts for the title, prompt body (becomes the HTML `description`),
`points_possible` (default 1), an optional ISO `due_at`, and whether to publish
immediately (default unpublished so you can review before exposing it to
students). Logs the new assignment's ID and Canvas URL on success.

| | Files |
|---|---|
| **Input** | _(none — interactive prompts only)_ |
| **Output** | _(none — assignment is created directly on Canvas)_ |

---

## `get-media-recordings`

*Fetch + transcode + transcribe each submission and either prompt for a grade or auto-award `points_possible`.*

Pulls every media-recording submission for a chosen assignment, re-encodes the
audio to 16 kHz mono PCM WAV (via ffmpeg directly on Canvas's playback URL —
DASH manifests are handled in one pass), and transcribes each WAV locally
using `OLLAMA_AUDIO_MODEL` (default `gemma4:e4b`). Audio files are saved to
`data/<course>/media_recordings/assignment<id>/<student_id>_<submission_id>.wav`.

By default, after each transcription the transcript is displayed and the
instructor is prompted for the points to award:

- **Enter** — award `points_possible` (full credit).
- **A number** — award that value.
- **`s` / `skip`** — leave the submission ungraded.

The chosen value is posted via Canvas's `Submission.edit({'posted_grade': ...})`.

Pass `--auto-grade` to skip the per-student review and award `points_possible`
to every submitter automatically. Combine with `--dry-run` to preview the
auto-graded writes without mutating Canvas; in dry-run, transcripts are still
shown and the CSV is still written, but the grade write is suppressed.

Pressing Ctrl-C mid-loop flushes a partial CSV before re-raising, so review
progress is never lost.

| | Files |
|---|---|
| **Input** | _(Canvas — selected interactively)_ |
| **Output** | `data/<course>/assignment<id>_recordings_YYYYMMDD.csv` — columns: `student_id`, `student_name`, `submission_id`, `submitted_at`, `audio_path`, `transcript`, `transcribed_at`, `grade`, `graded_at` |
| | `data/<course>/media_recordings/assignment<id>/<student_id>_<submission_id>.wav` — one WAV per submission |
| **Canvas side-effect** | Posts `posted_grade` on each graded submission (skipped in `--dry-run` mode) |

---

## `analyze-media-recordings`

*Two local-LLM passes against the transcripts: tag classification + theme extraction.*

Loads the most recent `assignment<id>_recordings_*.csv` produced by
`get-media-recordings`, prompts the instructor to pick a `*_questions_w_tags_*.csv`
from the same course directory (so tag vocabulary stays grounded in actual
quiz content), then runs two local-LLM passes against `OLLAMA_MODEL` (default
`gemma4:31b`):

1. **Tag classification** — one call per non-empty transcript, asking which of
   the unique tags from the quiz CSV's `keywords` column the student is
   actually discussing. Paraphrase-tolerant, so *"I got confused on the linked
   list one"* maps to `linked lists` even when the literal substring isn't
   present.
2. **Theme extraction** — a single cohort-level call that surfaces 3–5
   recurring themes across all transcripts, with the tag list as background
   context.

The output is a Markdown report with three sections: a tag-grounded table
ranked by descending student count (the *"specialized word cloud"*), the
LLM-extracted themes, and a roster mapping the 1-based student indices used
in the themes section back to names.

**Before running:** `get-media-recordings` must have produced the recordings
CSV, and [`get-quiz-questions --tag`](quiz-followup-questions.md#get-quiz-questions)
must have produced a tags CSV in the same course directory.

| | Files |
|---|---|
| **Input** | `data/<course>/assignment<id>_recordings_YYYYMMDD.csv` (from `get-media-recordings`) |
| | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (from `get-quiz-questions --tag`) |
| **Output** | `data/<course>/assignment<id>_analysis_YYYYMMDD.md` — Markdown report |
