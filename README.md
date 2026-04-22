

![Canvigator](images/canvigator2.png) 
## Helping educators enhance and experiment with their pedagogy via the Canvas LMS

### Overview 

Canvigator is a tool for instructors to be able to both automate some common
Canvas administrative tasks and utilize proven educational techniques such as
Continuous Assessments and Collaborative Learning. This tool has been used to
carry out novel research utilizing these well-known techniques
(as see in [this paper](https://link.springer.com/chapter/10.1007/978-3-031-74627-7_1)).

More recently, Canvigator has added LLM-assisted workflows (via
[Ollama](https://ollama.com)) for topic-tagging quiz questions, generating
open-ended follow-up questions for students who missed a concept, and
assessing the free-form replies (audio "explain" or photo "draw") that
students send back through Canvas conversations.

Currently, this terminal-based tool works with the Canvas Learning Management System (LMS) using the 
[CanvasAPI](https://github.com/ucfopen/canvasapi). In addition to increased functionality, development 
plans for this project include extending it to other LMSs and creating a richer interface. Suggestions 
for other functionality/features (see Issues), as well as any feedback on the project, are welcomed.


### Installation

Note that installation and usage requires some basic knowledge on how
to use the command line. If necessary, there are many brief
tutorials/lessons available to help in this area, e.g.,
[freecodecamp.org](https://www.freecodecamp.org/news/command-line-for-beginners/).

1. **Clone the repository**:
    [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) this repository to your local system. This can either be done 
in just one place on your local system, or separately for each course that it will be used in. The latter option is our preferred method since we tend to have a separate directory for each course we teach, and prefer to keep the course data separate.

2. **Generate a Canvas API Token**:
    Naviagte to the `canvigator` directory then run configuration setup script by simply typing, `./configure.sh` at the terminal prompt.
   ```bash
   cd canvigator
   ```
   ```bash
   ./configure.sh
   ```
   You will be prompted to enter:
   - Your institution's Canvas LMS base URL (you can find this on your Canvas home page).
   - Your Canvas token, which can be created by navigating to _'Account'_, then _'Settings'_. Towards the bottom of this window in Canvas you will see a blue button, _'+ New Access Token_'. Click on this button to copy/download the token to your local system. DO NOT TO SHARE YOUR CANVAS TOKEN w/ ANYONE (e.g. do not save it in a shared directory).
   - Your Ollama API key (optional — only needed if you plan to use the cloud-hosted text model for the LLM-powered tasks below). Leave it blank if you plan to use only local Ollama models (or none at all). See the [Ollama setup](#ollama-setup-optional) section for how to generate one.

This will prompt the creation of the _data/_ and _figures/_ subdirectories.

3. **Verify Setup**:
    Once this is complete, double check that the configuration script, _set_env.sh_ has been created, that it has the correct values for your URL and token, and that the subdirectories have been created. 
    ```bash
    set_env.sh
    ```
    ```bash
    env
    ```
    This command will list all environment variables, allowing you to confirm that the necessary variables have been set correctly.

4. **Verify Python Libraries**:
    Before running the project, ensure that all required Python libraries are installed:
    1. Install the required libraries using `pip`:
    
    ```bash
    pip install -r requirements.txt 
    ```
    
    2. Alternatively, you can install each library individually if needed.

        * ![canvasapi](https://img.shields.io/badge/canvasapi-3.3.0-blue)
        * ![matplotlib](https://img.shields.io/badge/matplotlib-3.8.4-brightgreen)
        * ![numpy](https://img.shields.io/badge/numpy-1.26.4-yellow)
        * ![pandas](https://img.shields.io/badge/pandas-2.2.2-orange)
        * ![requests](https://img.shields.io/badge/requests-2.33.0-red)
        * ![scipy](https://img.shields.io/badge/scipy-1.13.1-lightgrey)
        * ![seaborn](https://img.shields.io/badge/seaborn-0.13.2-blueviolet)
        * ![ollama](https://img.shields.io/badge/ollama-0.4.7-black)

    If no errors are thrown, the libraries are successfully installed. The `ollama` client is only used by the LLM-assisted tasks described below — see [Ollama setup](#ollama-setup-optional).


### Ollama setup (optional)

Several tasks use a Large Language Model (LLM) via [Ollama](https://ollama.com) to tag questions, generate open-ended follow-ups, transcribe student audio, and assess student replies. You can skip this section if you will not be running any of these tasks (`get-quiz-questions --tag`, `generate-open-ended-questions`, `send-quiz-reminder`, `send-follow-up-question`, `assess-replies`).

Canvigator uses two kinds of models:

1. **A cloud-hosted text model** (default `gemini-3-flash-preview`, set via `OLLAMA_TEXT_MODEL`) for instructor-side text generation — tagging quiz questions and generating open-ended questions. These tasks never see student data, so a larger cloud model is a good fit.
2. **Local models** for tasks that process student input — `gemma4:31b` (default `OLLAMA_MODEL`) for assessing text/image replies, and `gemma4:e4b` (default `OLLAMA_AUDIO_MODEL`) for transcribing student audio. Keeping these local is deliberate: student submissions should not leave your machine.

**To use the cloud text model:**
1. Sign in at [ollama.com](https://ollama.com) and create an API key from your account settings.
2. Paste the key when prompted by `./configure.sh` (or add `export OLLAMA_API_KEY="..."` directly to `set_env.sh`).

**To use the local models:**
1. Install Ollama from [ollama.com/download](https://ollama.com/download) and start it (`ollama serve`, or use the Ollama desktop app).
2. Pull the models you need:
   ```bash
   ollama pull gemma4:31b   # assessment of text/image replies (assess-replies)
   ollama pull gemma4:e4b   # transcription of student audio (assess-replies, explain mode)
   ```
   Only pull the models for tasks you actually plan to run. `gemma4:31b` is a large download (~20 GB) and is only required for `assess-replies`.

All model names are overridable via env vars (`OLLAMA_TEXT_MODEL`, `OLLAMA_MODEL`, `OLLAMA_AUDIO_MODEL`), and the local host can be overridden via the standard `OLLAMA_HOST` env var.


### Usage

```bash
source set_env.sh                            # set Canvas (and optional Ollama) environment variables (once per terminal session)
python canvigator.py <task>                  # run a task (prompts for course selection)
python canvigator.py --crn <CRN> <task>      # select course by CRN (last 5 digits of course code)
python canvigator.py --dry-run <task>        # preview changes without modifying Canvas (bonus, reminder, and follow-up tasks)
python canvigator.py --tag get-quiz-questions     # add LLM-generated topic tags to the quiz questions export
python canvigator.py --reply-window-days N <task>  # set the days-after-send window for get-replies (default: 5)
```

The `--crn` option selects a course by its CRN (the last 5 digits of the Canvas
course code), bypassing the interactive course selection prompt. This is useful
for automated/scheduled runs, e.g. `python canvigator.py --crn 12345 get-activity`.

Available tasks: `assess-replies`, `award-bonus`, `award-bonus-partner-only`, `award-bonus-retake-only`, `create-pairs`, `create-quiz`, `export-anon-data`, `generate-open-ended-questions`, `get-activity`, `get-all-subs`, `get-gradebook`, `get-quiz-questions`, `get-replies`, `send-follow-up-question`, `send-quiz-reminder`

All tasks begin by prompting you to select a course. Output files are written to
`data/<course>/` and `figures/<course>/`, where `<course>` is derived from the
Canvas course code. In the file names below, `<quiz>` refers to the quiz title
(lowercased, spaces replaced with underscores), `<id>` is the Canvas quiz ID,
and `YYYYMMDD` is the current date. Figure files now follow the same pattern as
the CSV exports, with the date at the end of the filename. Event exports and
per-question histograms are only generated by `get-all-subs`.

#### LLM-assisted follow-up workflow

Several of the newer tasks are designed to chain together into a single
end-to-end flow. Run them in this order for a given quiz:

1. `python canvigator.py --tag get-quiz-questions` — export the quiz's question content and add LLM topic tags (cloud model, requires `OLLAMA_API_KEY`).
2. `python canvigator.py generate-open-ended-questions` — generate 3 candidate open-ended follow-up questions per original question, with an assessment guide for each. **Review the output CSV and set `selected_question=1` on one row per question group before moving on.**
3. _(optional)_ `python canvigator.py send-quiz-reminder` — nudge students who haven't attempted the quiz or who scored below perfect. Imperfect-score students get a bulleted list of the topics (from the tags) they missed.
4. `python canvigator.py send-follow-up-question` — send the instructor-selected open-ended question (the first row in the CSV with `selected_question=1`) to each student who missed the corresponding original quiz question. Students reply via Canvas conversations with an audio recording ("explain") or a photo ("draw").
5. `python canvigator.py get-replies` — pull the students' replies (and attached audio/images) back from Canvas into a local CSV.
6. `python canvigator.py assess-replies` — run the replies through the local `gemma4` models (transcription + assessment) to produce pass/fail + feedback for each student.

Steps 4–6 can be repeated as students keep replying: `get-replies` picks up new messages, and `assess-replies` reassesses against the latest reply per student.

---

#### `assess-replies` — Assess student follow-up replies with a local LLM

Loads the latest reply per student from the follow-up replies CSV (produced by
`get-replies`) and uses local LLMs via Ollama to evaluate each one against the
original question. Two model pipelines are used depending on `question_mode`:

- **`explain` mode**: `OLLAMA_AUDIO_MODEL` (default `gemma4:e4b`) transcribes
  the student's audio recording, then `OLLAMA_MODEL` (default `gemma4:31b`)
  assesses the transcript against the question context.
- **`draw` mode**: `OLLAMA_MODEL` directly assesses the student's submitted
  image against the question context.

Each assessment yields a pass/fail result plus 2–3 sentences of feedback.
Re-running this task after `get-replies` picks up new student replies will
re-assess against the latest response per student.

**Prerequisite**: run `get-replies` first so the `*_followup_replies_*.csv` is
on disk. Requires a running Ollama server with both models pulled.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_followup_replies_YYYYMMDD.csv` (from `get-replies`) |
| | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (for question context) |
| | Audio/image files referenced by the replies CSV (under `data/<course>/replies/`) |
| **Output** | `data/<course>/<quiz>_<id>_followup_assessments_YYYYMMDD.csv` |

The output CSV contains columns: `student_id`, `student_name`, `question_id`,
`question_mode`, `result` (`pass` or `fail`), `feedback`, `transcript` (for
`explain` mode), `assessed_at`.

---

#### `award-bonus` — Award both partner and retake bonus points

Detects partner groups and qualifying retakers, then awards both bonus types
as combined fudge points on their Canvas submissions. Each bonus defaults to
15% of quiz points; a student earning both receives 30% total.

**Partner bonus:** Students whose first attempts show a high fraction of
matching scores (default ≥ 80%) and closely timed answers (default within
5 seconds on ≥ 80% of questions) are grouped as partners. Groups larger than
3 are flagged for manual review.

**Retake bonus:** Students who retook the quiz at least 3 times with at least
1 day (24 hours) between qualifying attempts.

**Before running:** the `get-all-subs` task must have been run first for the same
quiz so that the submission CSVs exist.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_all_subs_and_events_YYYYMMDD.csv` (from `get-all-subs`) |
| | `data/<course>/<quiz>_<id>_all_subs_by_question_YYYYMMDD.csv` (from `get-all-subs`) |
| | `data/<course>/<quiz>_<id>_all_submissions_YYYYMMDD.csv` (from `get-all-subs`) |
| **Output** | `data/<course>/<quiz>_<id>_detected_partners_YYYYMMDD.csv` — detected partner groups |
| | `data/<course>/<quiz>_<id>_retake_qualified_YYYYMMDD.csv` — students qualifying for retake bonus |
| | `data/<course>/<quiz>_<id>_scores_w_bonus_YYYYMMDD.csv` — scores with partner_bonus, retake_bonus, and combined bonus columns |
| **Canvas side-effect** | Sets `fudge_points` on the student's highest-scoring attempt and leaves a submission comment describing the bonus breakdown (skipped in `--dry-run` mode) |

Use `--dry-run` to preview which students would receive bonus points without
modifying anything in Canvas. In dry-run mode the scores CSV is named
`*_scores_w_bonus_dryrun_*.csv`.

**Typical workflow:**
1. Run `python canvigator.py get-all-subs` to export submission data for all quizzes.
2. Run `python canvigator.py [--dry-run] award-bonus` and select the course, quiz, and date when prompted.
3. Review the `*_detected_partners_*.csv` and `*_retake_qualified_*.csv` to verify the results.

---

#### `award-bonus-partner-only` — Award only the partner bonus

Same as `award-bonus` but only detects and awards the partner bonus (no retake
bonus). Useful when you want to reward collaborative work without the retake
incentive.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_all_subs_and_events_YYYYMMDD.csv` (from `get-all-subs`) |
| | `data/<course>/<quiz>_<id>_all_subs_by_question_YYYYMMDD.csv` (from `get-all-subs`) |
| **Output** | `data/<course>/<quiz>_<id>_detected_partners_YYYYMMDD.csv` — detected partner groups |
| | `data/<course>/<quiz>_<id>_scores_w_bonus_YYYYMMDD.csv` — scores with bonus column |
| **Canvas side-effect** | Sets `fudge_points` on the student's highest-scoring attempt and leaves a submission comment (skipped in `--dry-run` mode) |

---

#### `award-bonus-retake-only` — Award only the retake bonus

Same as `award-bonus` but only detects and awards the retake bonus (no partner
detection). Useful when you want to incentivize retakes independently.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_all_submissions_YYYYMMDD.csv` (from `get-all-subs`) |
| **Output** | `data/<course>/<quiz>_<id>_retake_qualified_YYYYMMDD.csv` — students qualifying for retake bonus |
| | `data/<course>/<quiz>_<id>_scores_w_bonus_YYYYMMDD.csv` — scores with bonus column |
| **Canvas side-effect** | Sets `fudge_points` on the student's highest-scoring attempt and leaves a submission comment (skipped in `--dry-run` mode) |

---

#### `create-pairs` — Create student pairings from quiz scores

Uses an independent (pre-class) quiz to pair students for a collaborative
in-class quiz. Computes pairwise Euclidean distance between student
answer-score vectors and applies a greedy pairing algorithm. This task writes
the raw quiz report plus pairing-specific figures, but does not export
submission events or histograms.

**Before running:** create a `present_*.csv` file in `data/<course>/` marking
which students are physically present. See `data/present_example.csv` for the
required format (columns: `name`, `id`, `present` where 1 = present).

| | Files |
|---|---|
| **Input** | `data/<course>/present_*.csv` (user-created, selected via prompt) |
| **Output — data** | `data/<course>/<quiz>_<id>_student_analysis_YYYYMMDD.csv` — raw quiz report downloaded from Canvas |
| | `data/<course>/pairings_based_on_<quiz>_<id>_YYYYMMDD.csv` — student pairings (person3/id3 columns omitted when no triples exist) |
| **Output — figures** | `figures/<course>/<quiz>_<id>_dist_euclid_YYYYMMDD.png` — distance matrix heatmap (present students) |
| | `figures/<course>/<quiz>_<id>_compare_pairing_methods_YYYYMMDD.png` — comparison of all four pairing methods |

**Typical workflow:**
1. Mark which students are present in your `present_*.csv` file.
2. Run `python canvigator.py create-pairs` and select the course, quiz, and presence CSV when prompted.
3. Open the generated `pairings_based_on_*.csv` and share pairings with the class.

---

#### `create-quiz` — Create an unpublished placeholder quiz on Canvas

Interactively creates a new unpublished quiz on Canvas. Prompts the user for a
quiz title, then iteratively prompts for question descriptions. Each question is
added as a multiple-choice placeholder with 1 point possible. The quiz is
created with default settings: `quiz_type='assignment'`, `time_limit=30`,
`one_question_at_a_time=True`, `cant_go_back=True`, `shuffle_answers=True`.

| | Files |
|---|---|
| **Input** | _(none — interactive prompts only)_ |
| **Output** | _(none — quiz is created directly on Canvas, unpublished)_ |

---

#### `export-anon-data` — Export anonymized course data

Anonymizes all CSV files in a course data directory by replacing student IDs
with hashed anonymous IDs (SHA-256 mod 10^10) and removing identifying columns
(`name`, `sis_id`). This task works entirely with local files and does not
require Canvas API credentials. Course selection is done from existing `data/`
subdirectories (by `--crn` or interactively).

> **Note:** Anonymization alone does not make data shareable. Do not publish/share
> data without participant consent, IRB approval, and FERPA compliance.

| | Files |
|---|---|
| **Input** | All CSV files in `data/<course>/` |
| **Output** | `data/<course>/anon_mapping_YYYYMMDD.csv` — mapping of original IDs to anonymous IDs |
| | `data/<course>/anonymized/` — directory containing anonymized copies of all CSVs |
| | `data/<course>/anonymized_YYYYMMDD.zip` — zip archive of the anonymized directory |

---

#### `generate-open-ended-questions` — Generate open-ended questions from a tagged quiz

Reads the tagged questions CSV for a selected quiz and uses a cloud-hosted LLM
(via Ollama's hosted endpoint, default `gemini-3-flash-preview`) to produce
open-ended follow-up candidates. For each original quiz question the task runs
three steps:

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
   `pass_criteria`, `acceptable_alternatives`, `common_misconceptions`,
   `fatal_errors` (plus `required_visual_elements` for draw questions). This
   structured rubric is what the local Gemma grader uses during
   `assess-replies`; edit the JSON blob if you want to tighten the criteria.
   Do not open the CSV in Excel and save — Excel will mangle the JSON quoting.

The output CSV is intended for instructor review. Three rows are written per
original question (one per candidate) with `selected_question=0`; **the
instructor must review the candidates offline and set `selected_question=1`
on exactly one row per question group** before running
`send-follow-up-question`.

**Prerequisite**: run `get-quiz-questions --tag` for the same quiz first so the
`*_questions_w_tags_*.csv` is on disk. Requires `OLLAMA_API_KEY` to be set —
see [Ollama setup](#ollama-setup-optional).

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (from `get-quiz-questions --tag`) |
| **Output** | `data/<course>/<quiz>_<id>_open_ended_YYYYMMDD.csv` |

The output CSV contains columns: `selected_question`, `question_id`,
`position`, `question_name`, `keywords`, `question_mode` (`explain` or
`draw`), `open_ended_question`, `assessment_guide`, `rubric_json`,
`original_question_text`.

**Typical workflow:**
1. Run `python canvigator.py --tag get-quiz-questions` and select the quiz.
2. Run `python canvigator.py generate-open-ended-questions` and select the tagged questions CSV.
3. Open the `*_open_ended_*.csv`, review the 3 candidates per question, and set `selected_question=1` on the single row you want to use per question group (edit the question text, `question_mode`, or `assessment_guide` as desired).

---

#### `get-activity` — Export student activity

Fetches enrollment activity data and course-level summary data from Canvas,
merges them, and saves a single CSV.

| | Files |
|---|---|
| **Input** | _(none — data comes from the Canvas API)_ |
| **Output** | `data/<course>/course_activity_YYYYMMDD.csv` |

The output CSV contains columns: `name`, `id`, `page_views`, `page_views_level`,
`participations`, `participations_level`, `missing`, `late`, `on_time`,
`total_activity_mins`, `first_activity_at`, `last_activity_at`,
`active_days_total`, `active_days_last_14`, `views_last_7d`,
`messages_to_instructor`.

The `*_level` columns are Canvas-computed 0–3 buckets summarizing the raw counts.
`first_activity_at` / `last_activity_at` come from per-student page-view buckets
(course-scoped analytics) — more accurate than the enrollment-level field, which
Canvas often leaves stale. `messages_to_instructor` counts student-initiated
messages in the course conversation thread.

If the Canvas course has a `start_at` date set, additional columns
`views_wk_01`, `views_wk_02`, … are appended (one per 7-day window from the
course start, capped at 16 weeks). Useful for spotting an engagement drop
mid-semester.

If the Canvas token has admin-level permission to read user profiles / view
usage reports, four further columns are appended: `top_browser`, `top_os`,
`used_mobile_app`, `n_distinct_user_agents` — derived from per-user page-view
records for this course. Most instructor tokens lack this permission, in which
case these columns are simply omitted from the CSV.

---

#### `get-all-subs` — Export all quiz submissions and events

Iterates over every published quiz in the course and downloads detailed
submission history and events for each one. This is the only task that exports
event history and per-question histogram figures.

| | Files |
|---|---|
| **Input** | _(none — data comes from the Canvas API)_ |
| **Output (per quiz)** | `data/<course>/<quiz>_<id>_student_analysis_YYYYMMDD.csv` — raw quiz report |
| | `data/<course>/<quiz>_<id>_all_submissions_YYYYMMDD.csv` — all submission attempts with scores |
| | `data/<course>/<quiz>_<id>_all_subs_by_question_YYYYMMDD.csv` — per-question results for each attempt |
| | `data/<course>/<quiz>_<id>_all_subs_and_events_YYYYMMDD.csv` — timestamped submission events |
| **Output — figures** | `figures/<course>/<quiz>_<id>_histograms_YYYYMMDD.png` — per-question score histograms |

---

#### `get-gradebook` — Export course gradebook

Fetches all published assignments and their submissions from Canvas, joins with
enrolled student data, and exports a comprehensive gradebook CSV. One row per
student per assignment.

| | Files |
|---|---|
| **Input** | _(none — data comes from the Canvas API)_ |
| **Output** | `data/<course>/gradebook_YYYYMMDD.csv` |

The output CSV contains columns: `name`, `sortable_name`, `user_id`,
`assignment_name`, `assignment_id`, `points_possible`, `grade`, `score`.

---

#### `get-quiz-questions` — Export quiz question content

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
to be set — see [Ollama setup](#ollama-setup-optional). The model can be
overridden via `OLLAMA_TEXT_MODEL`.

Pass `--all` to skip the interactive quiz prompt and export every quiz in the
selected course — one CSV per quiz, using each quiz's own filename prefix.
`--all` and `--tag` can be used independently or together.

---

#### `get-replies` — Retrieve student replies to follow-up questions

Loads the manifest written by `send-follow-up-question`, finds the matching
sent conversations on Canvas by subject, and extracts each student's reply.
Instructor messages and system-generated messages are filtered out. Image
attachments and audio/media recordings are downloaded to
`data/<course>/replies/`.

Only replies received within the configured window after the follow-up was
sent are accepted (default 5 days; override with `--reply-window-days N`).
The `latest` flag in the output CSV marks the most recent reply per student
and is what `assess-replies` uses for evaluation.

**Prerequisite**: run `send-follow-up-question` first so the
`*_followup_sent_*.csv` manifest is on disk.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_followup_sent_YYYYMMDD.csv` (from `send-follow-up-question`) |
| **Output** | `data/<course>/<quiz>_<id>_followup_replies_YYYYMMDD.csv` |
| | `data/<course>/replies/` — downloaded image attachments and audio recordings |

The output CSV contains columns: `student_id`, `student_name`, `question_id`,
`question_mode`, `message_id`, `reply_text`, `has_attachment`,
`attachment_path`, `has_audio`, `audio_path`, `replied_at`, `latest`.

---

#### `send-follow-up-question` — Send the instructor-selected open-ended follow-up question to students

Reads the `*_open_ended_*.csv`, picks the first row marked
`selected_question=1` as the question to send, and sends it via a Canvas
conversation message to each student who missed the corresponding original
quiz question on their latest attempt. Each thread uses `force_new=True` so
the follow-up exchange lives in its own dedicated conversation. The
Canvas-assigned `conversation_id` is captured at send time and recorded in
the manifest so `get-replies` can fetch each thread directly by ID.

The wording of the response instructions depends on the `question_mode` of
the open-ended question: `explain` asks the student to record a short voice
response, while `draw` asks them to attach a photo of a hand-drawn diagram.

When not in `--dry-run` mode, the instructor is prompted for each student
with a full message preview and a `[send/SKIP]` choice (default: skip). Only
messages the instructor approves are sent and recorded in the manifest.
Submissions are auto-refreshed via `getAllSubmissionsAndEvents()` on every
run so the recipient list reflects the latest attempts.

**Prerequisite**: run `get-quiz-questions --tag` and then
`generate-open-ended-questions` for the same quiz first so both the
`*_questions_w_tags_*.csv` and `*_open_ended_*.csv` are on disk.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (from `get-quiz-questions --tag`) |
| | `data/<course>/<quiz>_<id>_open_ended_YYYYMMDD.csv` (from `generate-open-ended-questions`) |
| **Output** | Fresh `*_all_submissions_*.csv`, `*_all_subs_by_question_*.csv`, and `*_all_subs_and_events_*.csv` (auto-refreshed via `getAllSubmissionsAndEvents()`) |
| | `data/<course>/<quiz>_<id>_followup_sent_YYYYMMDD.csv` — manifest of approved sends (for use by `get-replies`) |
| **Canvas side-effect** | Sends a Canvas conversation message (in a dedicated thread) to each approved student (skipped in `--dry-run` mode) |

Use `--dry-run` to preview every candidate message without sending anything
or prompting interactively.

---

#### `send-quiz-reminder` — Send quiz reminder messages to students

Sends personalized Canvas messages to students based on their quiz performance.
Students who have not yet attempted the quiz receive a reminder to make an
attempt. Students who attempted but scored below perfect receive encouragement
to retake, plus — when their most recent attempt had any missed questions — a
bulleted list of the concepts/topics those questions covered along with their
per-question score. Students with perfect scores are skipped.

**Prerequisite**: run `get-quiz-questions --tag` for the same quiz first so the
`*_questions_w_tags_*.csv` is on disk. The reminder task will not run
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

When not in `--dry-run` mode, the instructor is prompted for each student
with a full message preview and a `[send/SKIP]` choice (default: skip), so
every send is reviewed before going to Canvas.

Use `--dry-run` to preview all messages (recipient, subject, body, and reason)
without sending anything to Canvas and without the interactive prompt.
