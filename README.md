

![Canvigator](images/canvigator2.png) 
## Helping educators enhance and experiment with their pedagogy via the Canvas LMS

### Overview 

Canvigator is a tool for instructors to be able to both automate some common
Canvas administrative tasks and utilize proven educational techniques such as
Continuous Assessments and Collaborative Learning. This tool has been used to
carry out novel research utilizing these well-known techniques
(as see in [this paper](https://link.springer.com/chapter/10.1007/978-3-031-74627-7_1)).

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

        * ![canvasapi](https://img.shields.io/badge/canvasapi-2.2.0-blue)
        * ![matplotlib](https://img.shields.io/badge/matplotlib-3.3.4-brightgreen)
        * ![numpy](https://img.shields.io/badge/numpy-1.22.0-yellow)
        * ![pandas](https://img.shields.io/badge/pandas-1.3.4-orange)
        * ![requests](https://img.shields.io/badge/requests-2.33.0-red)
        * ![scipy](https://img.shields.io/badge/scipy-1.10.0-lightgrey)
        * ![seaborn](https://img.shields.io/badge/seaborn-0.11.2-blueviolet)

    If no errors are thrown, the libraries are successfully installed.


### Usage

```bash
source set_env.sh                            # set Canvas environment variables (once per terminal session)
python canvigator.py <task>                  # run a task (prompts for course selection)
python canvigator.py --crn <CRN> <task>      # select course by CRN (last 5 digits of course code)
python canvigator.py --dry-run <task>        # preview changes without modifying Canvas (bonus, reminder, and follow-up tasks)
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

Reads the tagged questions CSV for a selected quiz and uses a local LLM (via
Ollama) in two steps to generate one open-ended follow-up question per quiz
question:

1. **Classify** — For each question, the LLM decides whether an oral
   explanation ("explain") or a hand-drawn diagram ("draw") would be the better
   way to assess student understanding. Inherently visual topics (e.g. data
   structures, memory layouts, process flows) are classified as "draw"; verbal
   topics (e.g. trade-offs, algorithm logic, definitions) as "explain".
2. **Generate** — Using the classification, the LLM generates a self-contained
   open-ended question. "Explain" questions begin with "Explain..." and target
   a ~1 minute oral response. "Draw" questions begin with "Draw a diagram..." or
   "Draw a figure..." and target a ~2 minute hand-drawn response.

The output CSV is intended for instructor review — edit the questions and
override the `question_mode` column as needed before using them with students.

**Prerequisite**: run `get-quiz-questions --tag` for the same quiz first so the
`*_questions_w_tags_*.csv` is on disk.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` (from `get-quiz-questions --tag`) |
| **Output** | `data/<course>/<quiz>_<id>_open_ended_YYYYMMDD.csv` |

The output CSV contains columns: `question_id`, `position`, `question_name`,
`keywords`, `question_mode` (`explain` or `draw`), `original_question_text`,
`open_ended_question`.

**Typical workflow:**
1. Run `python canvigator.py --tag get-quiz-questions` and select the quiz.
2. Run `python canvigator.py generate-open-ended-questions` and select the same quiz.
3. Open the `*_open_ended_*.csv`, review/edit questions, and adjust any `question_mode` values as desired.

---

#### `get-activity` — Export student activity

Fetches enrollment activity data and course-level summary data from Canvas,
merges them, and saves a single CSV.

| | Files |
|---|---|
| **Input** | _(none — data comes from the Canvas API)_ |
| **Output** | `data/<course>/course_activity_YYYYMMDD.csv` |

The output CSV contains columns: `name`, `id`, `page_views`, `missing`, `late`,
`total_activity_mins`, `last_activity_at`.

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
1–3 short topical tags per question, produced by a local LLM via
[Ollama](https://ollama.com). The output is written to a separate file
(`*_questions_w_tags_*.csv`) so untagged and tagged exports never overwrite
each other. This requires the Ollama server to be running and the model to be
pulled locally. The model defaults to `gemma4:31b` and can be overridden with
the `OLLAMA_MODEL` env var; the host can be overridden with the standard
`OLLAMA_HOST` env var.

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

#### `send-follow-up-question` — Send the most-missed open-ended follow-up question to students

Identifies the single most-missed question on the quiz (highest miss rate
across students' latest attempts), looks up its open-ended counterpart from
the `*_open_ended_*.csv`, and sends it via a Canvas conversation message to
each student who missed it. Each thread uses `force_new=True` so the
follow-up exchange lives in its own dedicated conversation, which lets
`get-replies` find the thread later by subject.

The wording of the response instructions depends on the `question_mode` of
the open-ended question: `explain` asks the student to record a short voice
response, while `draw` asks them to attach a photo of a hand-drawn diagram.

When not in `--dry-run` mode, the instructor is prompted for each student
with a full message preview and a `[send/SKIP]` choice (default: skip). Only
messages the instructor approves are sent and recorded in the manifest.
Submissions are auto-refreshed via `getAllSubmissionsAndEvents()` on every
run so the most-missed question reflects the latest attempts.

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
