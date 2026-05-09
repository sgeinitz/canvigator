# Quiz follow-up questions

A connected set of tasks that turn a Canvas quiz into a personalized,
LLM-graded follow-up loop: tag the questions by topic, generate open-ended
follow-ups, nudge students who haven't aced the quiz, send the follow-up to
students who missed a specific question, then assess (and finally send back)
the replies.

| Task | Summary |
|---|---|
| [`get-quiz-questions`](#get-quiz-questions) | Export quiz metadata + question content; with `--tag` adds LLM-generated topic tags. |
| [`generate-follow-up-questions`](#generate-follow-up-questions) | Per original question, draft 3 candidate open-ended follow-ups + rubric + exemplars; instructor curates. |
| [`send-quiz-reminder`](#send-quiz-reminder) | Personalized Canvas reminders to students who haven't attempted or are below perfect. |
| [`send-follow-up-question`](#send-follow-up-question) | Send the curated follow-up via Canvas conversation to students who missed the corresponding question. |
| [`assess-replies`](#assess-replies) | Fetch student replies (audio/image) from Canvas and grade them with the local LLM. |
| [`send-follow-up-assessments`](#send-follow-up-assessments) | Post curated feedback back as a reply on the existing follow-up thread. |

Per-task `--help` is also available from the CLI:
`python canvigator.py <task> --help`.

← [Back to main README](../README.md)

## End-to-end workflow

Several of the newer tasks are designed to chain together into a single
end-to-end flow. Run them in this order for a given quiz:

1. `python canvigator.py --tag get-quiz-questions` — export the quiz's question content and add LLM topic tags (cloud model, requires `OLLAMA_API_KEY`).
2. `python canvigator.py generate-follow-up-questions` — generate 3 candidate open-ended follow-up questions per original question, with an assessment guide for each. **Review the output CSV and set `selected_question=1` on one row per question group before moving on.**
3. _(optional)_ `python canvigator.py send-quiz-reminder` — nudge students who haven't attempted the quiz or who scored below perfect. Imperfect-score students get a bulleted list of the topics (from the tags) they missed.
4. `python canvigator.py send-follow-up-question` — send the instructor-selected open-ended question (the first row in the CSV with `selected_question=1`) to each student who missed the corresponding original quiz question. Students reply via Canvas conversations with an audio recording ("explain") or a photo ("draw").
5. `python canvigator.py assess-replies` — pull the students' replies (and attached audio/images) back from Canvas, then run them through the local `gemma4` models (transcription + assessment) to produce pass/fail + feedback. Results are merged into a single persistent `*_followup_assessments.csv` (no date suffix); rows where `sent_assessment=1` are preserved verbatim across re-runs.
6. **Edit the `feedback` column** in `*_followup_assessments.csv` for any student where the LLM's assessment needs correction.
7. `python canvigator.py send-follow-up-assessments` — post the (curated) `feedback` text back to each student as a reply on the existing follow-up conversation thread; sets `sent_assessment=1`.

Steps 4–7 can be repeated as students keep replying: `assess-replies` picks up new messages and reassesses against the latest reply per student (skipping rows already sent), and `send-follow-up-assessments` only sends rows that haven't been sent yet.

### Data flow

```
get-quiz-questions --tag
   │
   └─→ *_questions_w_tags_*.csv  ─────────────────────────────┐
                                                               │ (read by)
generate-follow-up-questions  ◄─ reads *_questions_w_tags_*    │
   │                                                           │
   └─→ *_open_ended_*.csv  ──┐                                 │
       (instructor sets       │                                 │
        selected_question=1)  │                                 │
                              │                                 │
send-follow-up-question  ◄────┼─────────────────────────────────┘
   │  (recipients = students who missed the original question on
   │   their latest attempt; auto-refreshes submission CSVs)
   │
   └─→ *_followup_sent_*.csv  ──┐  (manifest of approved sends with
                                 │   conversation_id per row)
                                 │
assess-replies  ◄────────────────┘
   │  (fetches Canvas replies; reads *_open_ended_*.csv for
   │   the rubric, assessment guide, and exemplars)
   │
   ├─→ *_followup_replies_*.csv      (refreshed every run)
   └─→ *_followup_assessments.csv    (persistent across runs;
                                      instructor edits `feedback` column)
                                 │
send-follow-up-assessments  ◄────┘
   │
   └─→ updates *_followup_assessments.csv (sent_assessment=1, sent_at)
```

`send-quiz-reminder` is an optional side-branch off this chain — it reads the
same `*_questions_w_tags_*.csv`, auto-refreshes submission CSVs, and writes a
`*_reminder_sent_*.csv` manifest, but doesn't feed any of the downstream tasks.

---

## `get-quiz-questions`

*Export quiz metadata + question content; with `--tag` adds LLM-generated topic tags.*

Exports quiz metadata and question content to a CSV file. Skips downloading
student submission data, so this is a quick way to get a snapshot of the quiz
structure.

| | Files |
|---|---|
| **Input** | _(none — data comes from the Canvas API)_ |
| **Output (default)** | `data/<course>/<quiz>_<id>_questions_YYYYMMDD.csv` |
| **Output (with `--tag`)** | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` |

The output CSV contains columns: `quiz_id`, `assignment_id`, `question_id`,
`position`, `question_name`, `question_type`, `question_text`, `points_possible`,
`answers` (JSON-encoded). The `assignment_id` enables joining quiz data with
gradebook/assignment exports.

Pass `--tag` to add a `keywords` column (inserted before `question_text`) with
1–3 short topical tags per question, produced by an LLM via
[Ollama](https://ollama.com). The output is written to a separate file
(`*_questions_w_tags_*.csv`) so untagged and tagged exports never overwrite
each other. By default this uses the cloud-hosted text model
`gemini-3-flash-preview` at `https://ollama.com`, which requires `OLLAMA_API_KEY`
to be set — see [Ollama setup](installation.md#ollama-setup-optional). The
model can be overridden via `OLLAMA_TEXT_MODEL`.

Pass `--all` to skip the interactive quiz prompt and export every quiz in the
selected course — one CSV per quiz, using each quiz's own filename prefix.
`--all` and `--tag` can be used independently or together.

---

## `generate-follow-up-questions`

*Per original question, draft 3 candidate open-ended follow-ups + rubric + exemplars; instructor curates.*

Reads the tagged questions CSV for a selected quiz and uses a cloud-hosted LLM
(via Ollama's hosted endpoint, default `gemini-3-flash-preview`) to produce
open-ended follow-up candidates. For each original quiz question the task runs
five steps:

1. **Classify** — The LLM decides whether an oral explanation ("explain") or a
   hand-drawn diagram ("draw") would be the better way to assess student
   understanding. Inherently visual topics (e.g. data structures, memory
   layouts, process flows) are classified as "draw"; verbal topics (e.g.
   trade-offs, algorithm logic, definitions) as "explain".
2. **Generate 3 candidates** — Using the classification, the LLM generates 3
   self-contained candidate open-ended questions. "Explain" questions begin with
   "Explain..." and target a ~1 minute oral response. "Draw" questions begin
   with "Draw a diagram..." or "Draw a figure..." and target a ~2 minute
   hand-drawn response.
3. **Assessment guide** — For each candidate, the LLM writes a short
   `assessment_guide` describing the key concepts/elements a passing student
   response must include. This guide is the human-readable summary shown in
   the CSV and used as backup rubric material in `assess-replies`.
4. **Structured rubric** — For each candidate, the LLM also emits a JSON
   object in the `rubric_json` column with fields `canonical_answer`,
   `model_answer` (an 80–120-word A-grade exemplar response in a real
   student's voice), `pass_criteria`, `acceptable_alternatives`,
   `common_misconceptions`, `fatal_errors` (plus `required_visual_elements`
   for draw questions). This structured rubric is what the local Gemma grader
   uses during `assess-replies`; edit the JSON blob if you want to tighten the
   criteria. Do not open the CSV in Excel and save — Excel will mangle the
   JSON quoting.
5. **Calibration exemplars** — A separate LLM call drafts one passing and one
   failing example response (for explain mode, a transcript fragment; for
   draw mode, a prose description of the drawing) plus a one-sentence note
   explaining what makes each one pass or fail. These land in the
   `exemplar_pass`, `exemplar_pass_note`, `exemplar_fail`,
   `exemplar_fail_note` columns and are passed to the grader at assessment
   time as an "anchor the bar here" reference (separate from the live
   instructor-approved examples that `assess-replies` accumulates).

The output CSV is intended for instructor review. Three rows are written per
original question (one per candidate) with `selected_question=0`; **the
instructor must review the candidates offline and set `selected_question=1`
on exactly one row per question group** before running
`send-follow-up-question`.

**Before running:** `get-quiz-questions --tag` must have produced
`*_questions_w_tags_*.csv` for the same quiz. Requires `OLLAMA_API_KEY` to be
set — see [Ollama setup](installation.md#ollama-setup-optional).

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (from `get-quiz-questions --tag`) |
| **Output** | `data/<course>/<quiz>_<id>_open_ended_YYYYMMDD.csv` |

The output CSV contains columns: `selected_question`, `question_id`,
`position`, `question_name`, `keywords`, `question_mode` (`explain` or
`draw`), `open_ended_question`, `assessment_guide`, `rubric_json`,
`exemplar_pass`, `exemplar_pass_note`, `exemplar_fail`, `exemplar_fail_note`,
`original_question_text`.

---

## `send-quiz-reminder`

*Personalized Canvas reminders to students who haven't attempted or are below perfect.*

Sends personalized Canvas messages to students based on their quiz performance.
Students who have not yet attempted the quiz receive a reminder to make an
attempt. Students who attempted but scored below perfect receive encouragement
to retake, plus — when their most recent attempt had any missed questions — a
bulleted list of the concepts/topics those questions covered along with their
per-question score. Students with perfect scores are skipped.

When not in `--dry-run` mode, the instructor is prompted for each student
with a full message preview and a `[send/SKIP]` choice (default: skip), so
every send is reviewed before going to Canvas.

**Before running:** `get-quiz-questions --tag` must have produced
`*_questions_w_tags_*.csv` for the quiz. The reminder task will not run
`get-quiz-questions` automatically — quiz content is static, so you should
generate it once ahead of time. Submissions, on the other hand, change right
up to the moment the reminder runs, so the task automatically invokes
`getAllSubmissionsAndEvents()` on every run to pick up the latest data.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (from `get-quiz-questions --tag`) |
| **Output** | Fresh `data/<course>/<quiz>_<id>_all_submissions_YYYYMMDD.csv`, `..._all_subs_by_question_YYYYMMDD.csv`, and `..._all_subs_and_events_YYYYMMDD.csv` (written automatically via `getAllSubmissionsAndEvents()`) |
| | `data/<course>/<quiz>_<id>_reminder_sent_YYYYMMDD.csv` — audit log of approved sends (one row per student, with `conversation_id`); a `..._reminder_sent_dryrun_*.csv` is written in `--dry-run` mode |
| **Canvas side-effect** | Sends a Canvas conversation message to each student who hasn't attempted or hasn't achieved a perfect score (skipped in `--dry-run` mode) |

Example of the appended section for an imperfect-score student:

```
The questions that you missed on this most recent attempt covered the concepts/topics:
• recursion, base case, stack frames — 0.50 / 1.00 points
• big-o, sorting — 0.00 / 1.00 points
```

Use `--dry-run` to preview all messages (recipient, subject, body, and reason)
without sending anything to Canvas and without the interactive prompt.

Pass `--all` (or `-a`) to skip the interactive quiz picker and instead
iterate every published quiz with a future `due_at`. Each student's state is
aggregated across those quizzes and the task sends **one consolidated Canvas
message per student** listing every eligible quiz they still have work to do
on (no attempt, imperfect score, page-blur, or perfect-clean), with a single
course-level manifest `course_reminder_sent[_dryrun]_YYYYMMDD.csv` instead of
the per-quiz manifest. Quizzes without a `*_questions_w_tags_*.csv` are
skipped with a warning. Combine with `--dry-run` to preview the consolidated
messages without sending.

---

## `send-follow-up-question`

*Send the curated follow-up via Canvas conversation to students who missed the corresponding question.*

Reads the `*_open_ended_*.csv`, picks the first row marked
`selected_question=1` as the question to send, and sends it via a Canvas
conversation message to each student who missed the corresponding original
quiz question on their latest attempt. Each thread uses `force_new=True` so
the follow-up exchange lives in its own dedicated conversation. The
Canvas-assigned `conversation_id` is captured at send time and recorded in
the manifest so `assess-replies` can fetch each thread directly by ID.

The subject line includes a compact course code (prefix-number-CRN, with the
section dropped), the quiz name, and the specific question number — e.g.
`CSI-3300-12345 - Quiz 1 - Q3 Follow-Up`. This makes it easy to triage
conversations across multiple classes/quizzes/questions; the thread is still
tracked internally by `conversation_id`, so the subject format has no effect
on reply collection.

The wording of the response instructions depends on the `question_mode` of
the open-ended question: `explain` asks the student to record a short voice
response, while `draw` asks them to attach a photo of a hand-drawn diagram.

When not in `--dry-run` mode, the instructor is prompted for each student
with a full message preview and a `[send/SKIP]` choice (default: skip). Only
messages the instructor approves are sent and recorded in the manifest.
Submissions are auto-refreshed via `getAllSubmissionsAndEvents()` on every
run so the recipient list reflects the latest attempts.

**Before running:** `get-quiz-questions --tag` and then
`generate-follow-up-questions` must have been run for the same quiz so both
the `*_questions_w_tags_*.csv` and `*_open_ended_*.csv` are on disk.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (from `get-quiz-questions --tag`) |
| | `data/<course>/<quiz>_<id>_open_ended_YYYYMMDD.csv` (from `generate-follow-up-questions`) |
| **Output** | Fresh `*_all_submissions_*.csv`, `*_all_subs_by_question_*.csv`, and `*_all_subs_and_events_*.csv` (auto-refreshed via `getAllSubmissionsAndEvents()`) |
| | `data/<course>/<quiz>_<id>_followup_sent_YYYYMMDD.csv` — manifest of approved sends (for use by `assess-replies`) |
| **Canvas side-effect** | Sends a Canvas conversation message (in a dedicated thread) to each approved student (skipped in `--dry-run` mode) |

Use `--dry-run` to preview every candidate message without sending anything
or prompting interactively.

---

## `assess-replies`

*Fetch student replies (audio/image) from Canvas and grade them with the local LLM.*

First fetches the latest student replies from Canvas (loading the manifest
written by `send-follow-up-question`, finding matching sent conversations by
subject, downloading image attachments and audio recordings to
`data/<course>/replies/`, and writing a dated `*_followup_replies_*.csv`).
Only replies received within the configured window after the follow-up was
sent are accepted (default 5 days; override with `--reply-window-days N`).

Then loads the latest reply per student from that CSV and uses local LLMs via
Ollama to evaluate each one against the original question. Two model pipelines
are used depending on `question_mode`:

- **`explain` mode**: `OLLAMA_AUDIO_MODEL` (default `gemma4:e4b`) transcribes
  the student's audio recording, then `OLLAMA_MODEL` (default `gemma4:31b`)
  assesses the transcript against the question context.
- **`draw` mode**: `OLLAMA_MODEL` directly assesses the student's submitted
  image against the question context.

Each reply is graded with N=3 self-consistency passes and a majority vote;
`confidence=borderline` flags the rows worth instructor review before sending.
The grader is also primed with the structured rubric and exemplars from
`*_open_ended_*.csv`, plus up to 3 prior pass + 3 prior fail
`sent_assessment=1` responses for the same question (sampled from the
assessments file itself) as in-context calibration — so once you correct a few
borderline cases and send them, future re-runs anchor to your judgment.

Re-running this task is safe: rows with `sent_assessment=1` are skipped and
preserved verbatim, so previously sent feedback is never overwritten. Rows
with `sent_assessment=0` are re-assessed in place. On the first run, an
existing dated `*_followup_assessments_YYYYMMDD.csv` is migrated forward into
the non-dated file.

**Before running:** `send-follow-up-question` must have produced
`*_followup_sent_*.csv`. Requires a running Ollama server with both models
pulled.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_followup_sent_YYYYMMDD.csv` (from `send-follow-up-question`) |
| | `data/<course>/<quiz>_<id>_open_ended_YYYYMMDD.csv` (provides the rubric, assessment guide, and exemplars used by the grader) |
| **Output** | `data/<course>/<quiz>_<id>_followup_replies_YYYYMMDD.csv` |
| | `data/<course>/replies/` — downloaded image attachments and audio recordings |
| | `data/<course>/<quiz>_<id>_followup_assessments.csv` (persistent — merged across runs) |

The replies CSV contains columns: `student_id`, `student_name`, `question_id`,
`question_mode`, `conversation_id`, `message_id`, `reply_text`, `has_attachment`,
`attachment_path`, `has_audio`, `audio_path`, `replied_at`, `latest`. The
`latest` flag marks the most recent reply per student (used for assessment).

The assessments CSV contains columns: `student_id`, `student_name`, `question_id`,
`question_mode`, `conversation_id`, `result` (`pass` or `fail`), `confidence`
(`high` when all self-consistency runs agreed, `borderline` when they split),
`feedback`, `transcript` (for `explain` mode), `criteria_evaluations` (a JSON
blob with the per-criterion `met`/`partial`/`missing` ratings the grader produced),
`assessed_at`, `sent_assessment` (`0` until `send-follow-up-assessments` posts
the row, then `1`), `sent_at`.

---

## `send-follow-up-assessments`

*Post curated feedback back as a reply on the existing follow-up thread.*

Reads `*_followup_assessments.csv` and, for every row with `sent_assessment=0`
and a non-empty `feedback` value, posts the feedback text as a reply on the
existing follow-up conversation thread (`Conversation.add_message`). On
success, sets `sent_assessment=1` and stamps `sent_at`, then rewrites the file
in place. The `--dry-run` flag previews what would be sent without touching
Canvas (and leaves `sent_assessment=0`).

The expected workflow is to **edit the `feedback` column** between
`assess-replies` and `send-follow-up-assessments` for any rows where the LLM
assessment needs correction; the curated text is what the student sees.

**Before running:** `assess-replies` must have produced
`*_followup_assessments.csv`.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_followup_assessments.csv` (from `assess-replies`, edited by instructor) |
| | `data/<course>/<quiz>_<id>_followup_sent_*.csv` (fallback for `conversation_id` lookup) |
| **Output** | The same `*_followup_assessments.csv`, with `sent_assessment` and `sent_at` updated |
