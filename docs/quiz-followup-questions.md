# Quiz follow-up questions

A connected set of tasks that turn a Canvas quiz into an LLM-graded follow-up
loop: tag the questions by topic, generate open-ended follow-ups, nudge
students about the quiz, deliver the follow-up for a specific question, then
assess the students' responses and return feedback.

The first steps are the same however the follow-up is delivered. From there
it goes out in one of two ways, each a three-task chain:

- **As a Canvas message** — a personal conversation with every student who
  attempted the quiz, worded by whether they missed the question. See
  [Deliver as a Canvas message](#deliver-as-a-canvas-message).
- **As a Canvas assignment** — one assignment that asks every student the
  same question. See
  [Deliver as a Canvas assignment](#deliver-as-a-canvas-assignment).

| Task | Used by | Summary |
|---|---|---|
| [`get-quiz-questions`](#get-quiz-questions) | Both | Export quiz metadata + question content; with `--tag` adds LLM-generated topic tags. |
| [`generate-follow-up-questions`](#generate-follow-up-questions) | Both | Per original question, draft 3 candidate open-ended follow-ups + rubric + exemplars; instructor curates. |
| [`send-quiz-reminder`](#send-quiz-reminder) | Both (optional) | Personalized Canvas reminders: no attempt, imperfect score, perfect but left the quiz window, or perfect and clean. |
| [`send-follow-up-question`](#send-follow-up-question) | Message | Send the curated follow-up via Canvas conversation to every attempter, logging whether each missed the corresponding question. |
| [`assess-replies`](#assess-replies) | Message | Fetch student replies (audio/image) from Canvas and grade them with the local LLM. |
| [`send-follow-up-assessments`](#send-follow-up-assessments) | Message | Post curated feedback back as a reply on the existing follow-up thread. |
| [`create-follow-up-assignment`](#create-follow-up-assignment) | Assignment | Give every student the same curated follow-up as a Canvas assignment. |
| [`assess-follow-up-assignment`](#assess-follow-up-assignment) | Assignment | Download the assignment's submissions and grade them with the local LLM; nothing reaches students. |
| [`review-return-follow-up-assignment`](#review-return-follow-up-assignment) | Assignment | Hear/see each submission and give your own verdict before the LLM's is shown; then return feedback + completion credit. |

Per-task `--help` is also available from the CLI:
`python canvigator.py <task> --help`.

← [Back to main README](../README.md)

## End-to-end workflow

These tasks chain together into a single end-to-end flow for a given quiz:
the shared first steps, then one of the two delivery methods.

### Shared first steps

1. `python canvigator.py --tag get-quiz-questions` — export the quiz's question content and add LLM topic tags (cloud model, requires `OLLAMA_API_KEY`).
2. `python canvigator.py generate-follow-up-questions` — generate 3 candidate open-ended follow-up questions per original question, with an assessment guide for each. **Review the output CSV and set `selected_question=1` on one row per question group before moving on.**
3. _(optional)_ `python canvigator.py send-quiz-reminder` — nudge students who haven't attempted the quiz or who scored below perfect. Imperfect-score students get a bulleted list of the topics (from the tags) they missed.

Then deliver the selected follow-up question in one of two ways:

| | As a Canvas message | As a Canvas assignment |
|---|---|---|
| **Who gets it** | Every student who has attempted the quiz | Every student in the course |
| **Wording** | Per student, by whether they missed the original question | The same question for everyone |
| **Students respond** | By replying in the Canvas conversation | By submitting to the assignment (resubmissions allowed) |
| **Your review** | Edit the model's verdict and feedback in the CSV | Give your own verdict on each submission before the model's is shown |
| **Feedback returns as** | A reply on the conversation thread | A submission comment, with completion credit in the gradebook |

Use one delivery method per follow-up question. Both write to the same `*_followup_assessments.csv`, keyed by student and question, so `create-follow-up-assignment` refuses a question already sent by `send-follow-up-question`, and vice versa.

### Deliver as a Canvas message

1. `python canvigator.py send-follow-up-question` — send the instructor-selected open-ended question (the row with `selected_question=1`; if several questions are selected you are asked which one) to every student who has attempted the quiz. Students who missed the original question on their latest attempt get a "reinforce your understanding" intro; students who got it right get a "slightly more challenging" intro; the manifest records which (`status`). Students reply via Canvas conversations with an audio recording ("explain") or a photo ("draw").
2. `python canvigator.py assess-replies` — pull the students' replies (and attached audio/images) back from Canvas, then run them through the local `gemma4` models (transcription + assessment) to produce pass/fail + feedback. Results are merged into a single persistent `*_followup_assessments.csv` (no date suffix); rows where `sent_assessment=1` are preserved verbatim across re-runs.
3. **Edit the `feedback` column** (and `result`, if you disagree) in `*_followup_assessments.csv` for any student where the LLM's assessment needs correction. The model's original verdict stays in `llm_result` / `llm_confidence` / `llm_feedback`. Rows with `result=error` have empty feedback — grade those by hand.
4. `python canvigator.py send-follow-up-assessments` — post the (curated) `feedback` text back to each student as a reply on the existing follow-up conversation thread; sets `sent_assessment=1`.

These steps can be repeated as students keep attempting and replying: re-running `send-follow-up-question` reaches only attempters who haven't been sent that question yet; `assess-replies` reads every send manifest for the quiz, grades only replies it hasn't graded yet (so your unsent edits survive), and assesses a newer reply that arrives after feedback was sent as a new row; `send-follow-up-assessments` only sends rows that haven't been sent yet.

### Deliver as a Canvas assignment

1. `python canvigator.py create-follow-up-assignment` — create one assignment for the selected question (the row with `selected_question=1`; if several questions are selected you are asked which one), assigned to everyone in the course. An `explain` question accepts only a media recording; a `draw` question accepts only an uploaded JPG, PNG, or PDF. Unpublished by default — review it on Canvas, then publish. Run once per question when a quiz has several.
2. Students submit on Canvas, and may resubmit; only the latest attempt is assessed.
3. `python canvigator.py assess-follow-up-assignment` — download the submissions and grade them with the local `gemma4` models. This is the slow step and can run unattended. Nothing is written to Canvas and no verdict is printed. **Do not open `*_followup_assessments.csv` yet** — the next step asks for your verdict before showing the model's.
4. `python canvigator.py review-return-follow-up-assignment` — for each submission: the recording plays (or the drawing opens), you enter **your own** pass/fail, and only then are the transcript, the model's verdict, and its draft feedback shown. You settle any disagreement, accept or edit the feedback, and confirm before it is posted as a submission comment. A real submission earns full points whether it passed or failed.

Steps 3–4 can be repeated as submissions arrive: an attempt already assessed is not re-graded, and a resubmission is assessed (and reviewed) as a new attempt.

### Data flow

Both delivery methods start from the instructor-curated `*_open_ended_*.csv`.
Message delivery:

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
   │  (recipients = every attempter not already sent this question;
   │   status = missed/correct on their latest attempt;
   │   auto-refreshes submission CSVs)
   │
   └─→ *_followup_sent_*.csv  ──┐  (manifest of approved sends with
                                 │   conversation_id + status per row;
                                 │   same-day sends append)
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

Assignment delivery, from the same `*_open_ended_*.csv`:

```
*_open_ended_*.csv  (selected_question=1)
   │
create-follow-up-assignment
   │
   └─→ *_followup_assignments.csv  ──┐  (persistent; one row per assignment:
                                      │   assignment_id, question_id, mode)
                                      │
assess-follow-up-assignment  ◄────────┘
   │  (fetches each student's latest attempt; reads *_open_ended_*.csv for
   │   the rubric; refreshes quiz submissions to record quiz_status)
   │
   ├─→ replies/                       (downloaded recordings + drawings)
   └─→ *_followup_assessments.csv     (rows carry assignment_id + attempt)
                                      │
review-return-follow-up-assignment  ◄─┘
   │  (plays/opens each submission → your verdict → model's verdict →
   │   reconcile → feedback → [y/N])
   │
   ├─→ updates *_followup_assessments.csv (instructor_result, reviewed_at,
   │                                       result, feedback, sent_assessment)
   └─→ Canvas: submission comment + grade
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
`glm-5.3-flash:cloud` at `https://ollama.com`, which requires `OLLAMA_API_KEY`
to be set — see [Ollama setup](installation.md#ollama-setup-optional). The
model can be overridden via `OLLAMA_TEXT_MODEL`.

Pass `--all` to skip the interactive quiz prompt and export every quiz in the
selected course — one CSV per quiz, using each quiz's own filename prefix.
`--all` and `--tag` can be used independently or together.

---

## `generate-follow-up-questions`

*Per original question, draft 3 candidate open-ended follow-ups + rubric + exemplars; instructor curates.*

Reads the tagged questions CSV for a selected quiz and uses a cloud-hosted LLM
(via Ollama's hosted endpoint, default `glm-5.3-flash:cloud`) to produce
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
   hand-drawn response. The 3 are generated together as a difficulty spread —
   one relatively easy, one medium, one relatively hard compared with each
   other — and the `est_difficulty` column records which is which. If the LLM
   returns fewer than 3 candidates, `est_difficulty` is left blank for that
   group.
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
`draw`), `est_difficulty` (`easy`, `medium`, or `hard`, relative to the other
two candidates for the same question), `open_ended_question`,
`assessment_guide`, `rubric_json`,
`exemplar_pass`, `exemplar_pass_note`, `exemplar_fail`, `exemplar_fail_note`,
`original_question_text`.

---

## `send-quiz-reminder`

*Personalized Canvas reminders: no attempt, imperfect score, perfect but left the quiz window, or perfect and clean.*

Sends personalized Canvas messages to students based on their quiz performance.
Students who have not yet attempted the quiz receive a reminder to make an
attempt. Students who attempted but scored below perfect receive encouragement
to retake, plus — when their most recent attempt had any missed questions — a
bulleted list of the concepts/topics those questions covered along with their
per-question score. Students with a perfect score whose most recent attempt
had page-blur (left-the-window) events get a note listing the questions where
that happened; perfect-score students with no blur events are encouraged to
retake once more before the due date.

When not in `--dry-run` mode, the instructor is prompted for each student
with a full message preview and a `[y/N]` choice (default: skip), so
every send is reviewed before going to Canvas.

Stopping partway (Ctrl-C, or Ctrl-D at a prompt) is safe: the reminders
already sent are still written to the `*_reminder_sent_*.csv` log, Canvigator
prints "Stopped early", and it exits with code 130 instead of 0. A re-run does
not skip students who were already reminded, so check the log first. If you
stop while a message is being sent, that one may reach Canvas without being
logged; Canvigator says so, and you can check Canvas for that student.

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
| | `data/<course>/<quiz>_<id>_reminder_sent_YYYYMMDD.csv` — audit log of approved sends (one row per student, with `reason` and `conversation_id`; a second run on the same day appends); a `..._reminder_sent_dryrun_*.csv` is written in `--dry-run` mode |
| **Canvas side-effect** | Sends a Canvas conversation message to each approved enrolled student (skipped in `--dry-run` mode) |

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

*Send the curated follow-up via Canvas conversation to every attempter, logging whether each missed the corresponding question.*

Reads the `*_open_ended_*.csv` and sends the question marked
`selected_question=1` — when more than one question is selected, it lists
them in quiz order and asks which one to send, so earlier selections can stay
marked (`assess-replies` needs them to find each question's rubric). The
question goes via a Canvas conversation message to every student who has
attempted the quiz: students who missed the corresponding original question on
their latest attempt get a "reinforce your understanding" intro, and students
who answered it correctly get a "slightly more challenging follow-up" intro.
Students who already received this question's follow-up in an earlier real
send are skipped, so re-running later in the week reaches only new
attempters. Each thread uses `force_new=True` so the follow-up exchange lives
in its own dedicated conversation. The Canvas-assigned `conversation_id` is
captured at send time and recorded in the manifest so `assess-replies` can
fetch each thread directly by ID.

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
with a full message preview and a `[y/N]` choice (default: skip). Only
messages the instructor approves are sent and recorded in the manifest.
Submissions are auto-refreshed via `getAllSubmissionsAndEvents()` on every
run so the recipient list reflects the latest attempts.

Stopping partway (Ctrl-C, or Ctrl-D at a prompt) is safe: the students already
messaged are recorded in the manifest, so running the task again won't message
them twice. Canvigator prints "Stopped early" and exits with code 130 instead
of 0. If you stop while a message is being sent, that one may reach Canvas
without being recorded; Canvigator says so, and you can check Canvas for that
student before re-running.

**Before running:** `get-quiz-questions --tag` and then
`generate-follow-up-questions` must have been run for the same quiz so both
the `*_questions_w_tags_*.csv` and `*_open_ended_*.csv` are on disk.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (from `get-quiz-questions --tag`) |
| | `data/<course>/<quiz>_<id>_open_ended_YYYYMMDD.csv` (from `generate-follow-up-questions`) |
| **Output** | Fresh `*_all_submissions_*.csv`, `*_all_subs_by_question_*.csv`, and `*_all_subs_and_events_*.csv` (auto-refreshed via `getAllSubmissionsAndEvents()`) |
| | `data/<course>/<quiz>_<id>_followup_sent_YYYYMMDD.csv` — manifest of approved sends (for use by `assess-replies`), one row per student with `question_id`, `question_mode`, `status` (`missed`/`correct` at send time), `conversation_id`, `conversation_subject`, `open_ended_question` (the text sent), and `sent_at`; a second run on the same day appends |
| **Canvas side-effect** | Sends a Canvas conversation message (in a dedicated thread) to each approved student (skipped in `--dry-run` mode) |

Use `--dry-run` to preview every candidate message without sending anything
or prompting interactively.

---

## `assess-replies`

*Fetch student replies (audio/image) from Canvas and grade them with the local LLM.*

First fetches the latest student replies from Canvas (loading every real-send
manifest written by `send-follow-up-question` for the quiz, fetching each
conversation by its recorded `conversation_id`, downloading image attachments
and audio recordings to `data/<course>/replies/` — one file per reply message
— and writing a dated `*_followup_replies_*.csv`). Only replies received within
the configured window after the follow-up was sent are accepted (default 5
days; override with `--reply-window-days N`).

Then takes the latest reply per student and question from that CSV — the
newest reply that carries the media the question asks for (audio for
`explain`, an image for `draw`), so a trailing text-only "thanks!" doesn't hide
the answer — and uses local LLMs via Ollama to evaluate each one against the
original question. Two model pipelines are used depending on `question_mode`:

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

A grading run that errors, or whose output can't be parsed into per-criterion
ratings, is discarded rather than counted; if every run fails the row gets
`result=error` with empty feedback (it is never sent until you write
feedback), and the run summary names those students. A reply without a usable
recording or image gets `result=no_media`.

All student-data model calls go to the local Ollama server only: the task
refuses to run if `OLLAMA_HOST` points anywhere other than this machine, or if
`OLLAMA_MODEL` / `OLLAMA_AUDIO_MODEL` is an Ollama cloud model (`…-cloud`).

Re-running this task is safe: a reply already assessed (same `message_id`) is
not re-graded, so your edits to unsent feedback survive; rows with
`sent_assessment=1` are preserved verbatim, so previously sent feedback is
never overwritten. If a student sends a newer reply — after unsent feedback,
it replaces that row; after sent feedback (e.g. "give it another try"), it is
assessed as a new row. Rows assessed before `message_id` was recorded are
re-graded once. On the first run, an existing dated
`*_followup_assessments_YYYYMMDD.csv` is migrated forward into the non-dated
file.

**Before running:** `send-follow-up-question` must have produced
`*_followup_sent_*.csv`. Requires a running Ollama server with both models
pulled.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_followup_sent_YYYYMMDD.csv` — every real-send manifest for the quiz (from `send-follow-up-question`) |
| | `data/<course>/<quiz>_<id>_open_ended_YYYYMMDD.csv` (provides the rubric, assessment guide, and exemplars used by the grader) |
| **Output** | `data/<course>/<quiz>_<id>_followup_replies_YYYYMMDD.csv` |
| | `data/<course>/replies/` — downloaded image attachments and audio recordings |
| | `data/<course>/<quiz>_<id>_followup_assessments.csv` (persistent — merged across runs) |

The replies CSV contains columns: `student_id`, `student_name`, `question_id`,
`question_mode`, `conversation_id`, `message_id`, `reply_text`, `has_attachment`,
`attachment_path`, `has_audio`, `audio_path`, `replied_at`, `latest`. The
`latest` flag marks the reply per student and question that is assessed (the
newest one carrying the expected media, else the newest).

The assessments CSV contains columns: `student_id`, `student_name`, `question_id`,
`question_mode`, `conversation_id`, `message_id` (the reply that was assessed),
`result` (`pass`, `fail`, `no_media`, or `error`), `confidence`
(`high` when all self-consistency runs agreed, `borderline` when they split),
`feedback`, `transcript` (for `explain` mode), `criteria_evaluations` (a JSON
blob with the per-criterion `met`/`partial`/`missing` ratings the grader produced),
`assessed_at`, `sent_assessment` (`0` until `send-follow-up-assessments` posts
the row, then `1`), `sent_at`, and `llm_result` / `llm_confidence` /
`llm_feedback` — the model's original verdict, written once and never edited,
so draft-vs-final feedback and instructor–LLM agreement can be compared later.
Edit `result` / `feedback`, not the `llm_*` columns. The remaining columns
(`assignment_id`, `attempt`, `submitted_at`, `quiz_status`, `media_path`,
`instructor_result`, `reviewed_at`) are filled only by the
[assignment-based tasks](#assess-follow-up-assignment) and are empty here.

---

## `send-follow-up-assessments`

*Post curated feedback back as a reply on the existing follow-up thread.*

Reads `*_followup_assessments.csv` and, for every row with `sent_assessment=0`
and a non-empty `feedback` value, shows the full message and asks `[y/N]`
before posting the feedback text as a reply on the existing follow-up
conversation thread (`Conversation.add_message`). On success, sets
`sent_assessment=1` and stamps `sent_at`, then rewrites the file in place. The
`--dry-run` flag previews what would be sent without touching Canvas (and
leaves `sent_assessment=0`). Rows with `result=error` have empty feedback, so
they are not sent until you write feedback for them. Rows that belong to a
follow-up assignment (non-empty `assignment_id`) are skipped; those are
returned by
[`review-return-follow-up-assignment`](#review-return-follow-up-assignment).

The expected workflow is to **edit the `feedback` column** between
`assess-replies` and `send-follow-up-assessments` for any rows where the LLM
assessment needs correction; the curated text is what the student sees.

Stopping partway (Ctrl-C, or Ctrl-D at a prompt) is safe: rows already sent
are saved with `sent_assessment=1`, so running the task again won't send them
twice. Canvigator prints "Stopped early" and exits with code 130 instead of 0.
If you stop while a message is being sent, that one may reach Canvas without
being saved; Canvigator says so, and you can check Canvas for that student
before re-running. A stopped `--dry-run` preview saves no dry-run record.

**Before running:** `assess-replies` must have produced
`*_followup_assessments.csv`.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_followup_assessments.csv` (from `assess-replies`, edited by instructor) |
| | `data/<course>/<quiz>_<id>_followup_sent_*.csv` (fallback for `conversation_id` lookup) |
| **Output** | The same `*_followup_assessments.csv`, with `sent_assessment` and `sent_at` updated |

---

## `create-follow-up-assignment`

*Give every student the same curated follow-up as a Canvas assignment.*

Reads the `*_open_ended_*.csv` and creates a Canvas assignment for the question
marked `selected_question=1` — when more than one question is selected, it
lists them in quiz order and asks which one. The assignment is assigned to
everyone in the course, so there is no per-student wording and no recipient
list to maintain.

The submission type follows the `question_mode`:

- **`explain`** — `media_recording` only: students record a short voice response on the assignment page.
- **`draw`** — `online_upload` only, limited to `jpg`, `jpeg`, `png`, and `pdf`: students upload a photo or scan of a hand-drawn diagram.

The assignment is named `<quiz title> - Q<n> Follow-Up`. Its description names
the quiz, question number, and topic; states the follow-up question; gives the
mode-specific instructions; and tells students that a genuine attempt earns
full credit and that they may resubmit (only the latest submission is
assessed).

The task prompts for `points_possible` (default 1), an optional ISO `due_at`,
and whether to publish immediately (default unpublished, so you can review —
and edit — the assignment on Canvas first).

It then shows the assignment's name, settings, and description and asks
`Create this assignment on Canvas? [y/N]` before creating it.

There is one assignment per follow-up question. If the selected question
already has one, the task prints its URL and creates nothing. To replace an
assignment, delete it on Canvas and remove its row from
`*_followup_assignments.csv`. A question already sent as a Canvas message by
`send-follow-up-question` is refused, since both methods share the
assessments CSV.

**Before running:** `get-quiz-questions --tag` and then
`generate-follow-up-questions` must have been run for the same quiz.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (from `get-quiz-questions --tag`) |
| | `data/<course>/<quiz>_<id>_open_ended_YYYYMMDD.csv` (from `generate-follow-up-questions`) |
| **Output** | `data/<course>/<quiz>_<id>_followup_assignments.csv` (persistent — one row per assignment created) |
| **Canvas side-effect** | Creates one assignment (skipped in `--dry-run` mode) |

The manifest contains columns: `assignment_id`, `question_id`, `position`,
`question_mode`, `keywords`, `open_ended_question`, `points_possible`,
`due_at`, `published`, `html_url`, `created_at`.

Use `--dry-run` to print the assignment's name, settings, and description
without creating it.

---

## `assess-follow-up-assignment`

*Download the assignment's submissions and grade them with the local LLM; nothing reaches students.*

Fetches each student's latest attempt on a follow-up assignment, downloads the
recording or drawing to `data/<course>/replies/`, and grades it with the same
local pipeline as [`assess-replies`](#assess-replies) (`OLLAMA_AUDIO_MODEL`
transcribes, `OLLAMA_MODEL` assesses; N=3 self-consistency passes; the rubric,
exemplars, and previously returned examples from the same question calibrate
the grader). Results are merged into the quiz's `*_followup_assessments.csv`.
When the quiz has more than one follow-up assignment, the task asks which one.

This task is the slow part of grading and needs no attention while it runs.
It writes nothing to Canvas and **prints no pass/fail** — only how many
submissions were assessed. That is deliberate:
[`review-return-follow-up-assignment`](#review-return-follow-up-assignment)
asks for your own verdict on each submission before showing the model's, so
leave the assessments CSV closed until that review is done.

Each row also records `quiz_status` — the student's result on the original
quiz question on their latest quiz attempt at assessment time (`missed`,
`correct`, or `no_attempt`) — so analyses can separate those groups even
though everyone received the same assignment. The quiz's submission CSVs are
refreshed for this (or reused when fetched in the last 10 minutes). If the
quiz results can't be fetched, the assessments are still saved, with
`quiz_status=unknown`.

Re-running is safe: an attempt that has already been assessed is not graded
again, unless that assessment ended in `result=error` or `result=no_media`
(e.g. a failed download) and you have not yet given your verdict on it in
the review. A newer attempt
replaces a row that has not been returned yet, and is added as a new row when
feedback on the earlier attempt was already returned.

All student-data model calls go to the local Ollama server only, as in
`assess-replies`.

**Before running:** `create-follow-up-assignment` must have produced
`*_followup_assignments.csv`. Requires `ffmpeg` and a running Ollama server
with both models pulled.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_followup_assignments.csv` (from `create-follow-up-assignment`) |
| | `data/<course>/<quiz>_<id>_open_ended_YYYYMMDD.csv` (provides the rubric, assessment guide, and exemplars) |
| | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (question points, for `quiz_status`) |
| **Output** | `data/<course>/replies/` — downloaded recordings and drawings (`…_<student_id>_<firstname>_<question_id>_a<attempt>.<ext>`) |
| | `data/<course>/<quiz>_<id>_followup_assessments.csv` (persistent — merged across runs) |
| | Fresh `*_all_submissions_*.csv`, `*_all_subs_by_question_*.csv`, and `*_all_subs_and_events_*.csv` |

Rows written by this task fill these columns in addition to the ones described
under [`assess-replies`](#assess-replies): `assignment_id`, `attempt` (the
Canvas attempt number that was assessed), `submitted_at`, `quiz_status`, and
`media_path` (the downloaded file, relative to `data/`). `conversation_id` and
`message_id` are empty. `--choose-model` and `--save-transcript` work as they
do for `assess-replies`.

---

## `review-return-follow-up-assignment`

*Hear/see each submission and give your own verdict before the LLM's is shown; then return feedback + completion credit.*

Walks through every assessed submission of a follow-up assignment whose
feedback has not been returned yet. For each one, in this order:

1. **The submission.** A recording plays in the terminal (Ctrl-C stops the
   playback, not the task); a drawing opens in your image viewer. Nothing from
   the model is on screen.
2. **Your verdict.** `[p]ass`, `[f]ail`, or `[n]o usable submission` (also
   `[r]eplay`, `[s]kip`, `[q]uit`). It is saved immediately as
   `instructor_result`, with `reviewed_at`.
3. **The model's assessment.** The transcript (for `explain`), the model's
   verdict and confidence, and its draft feedback.
4. **Reconciliation.** If you and the model both gave pass/fail and differ,
   you choose the final `result` (`p` or `f`; there is no default, so a stray
   Enter can't settle it). The model's verdict stays in `llm_result`.
5. **Feedback.** Accept the draft or edit it in your editor (`$VISUAL` /
   `$EDITOR`, else `nano`). The editor opens without asking when there is no
   draft (`result=error` rows) or when the draft argues a different result
   than the final one.
6. **Return.** The exact comment and grade are shown, and `[y/N]` (default
   no) posts them: the feedback as a submission comment on the attempt you
   reviewed, and the grade.

Credit is for completion: a real submission earns the assignment's full
`points_possible` whether the final result is pass or fail. A submission you
mark as not usable gets the comment (asking the student to resubmit) and no
grade.

Because `instructor_result` is recorded before the model's verdict is shown,
comparing it with `llm_result` gives an instructor–model agreement measure
that is not anchored on the model.

Progress is written to the CSV after every step, so you can quit (`[q]` at
the verdict prompt, Ctrl-C at any other) and resume later. On a later run, a submission that already has your
verdict resumes at step 3 (it offers to replay first). A submission whose
student has resubmitted since it was assessed is held back, with a note to run
`assess-follow-up-assignment` again.

Audio playback uses `afplay` (macOS) or `ffplay` when available; otherwise the
recording opens in your default media player and the task waits for Enter. If
a submission's file was not downloaded, the task prints its Canvas URL instead.

There is no `--send-all` for this task: every submission is reviewed.

**Before running:** `assess-follow-up-assignment` must have assessed at least
one submission, on the same machine (the review plays the files it
downloaded).

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_followup_assignments.csv` |
| | `data/<course>/<quiz>_<id>_followup_assessments.csv` (from `assess-follow-up-assignment`) |
| | `data/<course>/replies/` — the downloaded recordings and drawings |
| **Output** | The same `*_followup_assessments.csv`, with `instructor_result`, `reviewed_at`, `result`, `feedback`, `sent_assessment`, and `sent_at` updated |
| **Canvas side-effect** | Posts a submission comment and a grade for each returned submission (skipped in `--dry-run` mode) |

Use `--dry-run` to run the whole review — your verdicts and feedback edits
are saved — without posting anything to Canvas.
