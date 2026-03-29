

![Canvigator](images/canvigator2.png) 
## Canvigator: Helping educators enhance their teaching by applying tried-and-true pedagogical techniques via the Canvas LMS

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
   - Your Canvas token, which can be created by navigating to _'Account'_, then _'Settings'_. Towards the bottom of this window in Canvas you will see a blue button, _'+ New Access Token_'. Click on this button to copy/download the token to your local system. DO NOT TO SHARE YOUR CANVAS TOKEN w/ ANYONE (e.g. do not save it a shared directory). 
 
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
python canvigator.py --dry-run <task>        # preview changes without modifying Canvas (bonus and reminder tasks)
```

The `--crn` option selects a course by its CRN (the last 5 digits of the Canvas
course code), bypassing the interactive course selection prompt. This is useful
for automated/scheduled runs, e.g. `python canvigator.py --crn 12345 activity`.

Available tasks: `activity`, `pair`, `award-bonus`, `award-bonus-partner-only`, `award-bonus-retake-only`, `all-subs`, `get-quiz-questions`, `create-quiz`, `export-anon-data`, `export-gradebook`, `quiz-reminder`

All tasks begin by prompting you to select a course. Output files are written to
`data/<course>/` and `figures/<course>/`, where `<course>` is derived from the
Canvas course code. In the file names below, `<quiz>` refers to the quiz title
(lowercased, spaces replaced with underscores), `<id>` is the Canvas quiz ID,
and `YYYYMMDD` is the current date. Figure files now follow the same pattern as
the CSV exports, with the date at the end of the filename. Event exports and
per-question histograms are only generated by `all-subs`.

---

#### `activity` — Export student activity

Fetches enrollment activity data and course-level summary data from Canvas,
merges them, and saves a single CSV.

| | Files |
|---|---|
| **Input** | _(none — data comes from the Canvas API)_ |
| **Output** | `data/<course>/course_activity_YYYYMMDD.csv` |

The output CSV contains columns: `name`, `id`, `page_views`, `missing`, `late`,
`total_activity_mins`, `last_activity_at`.

---

#### `pair` — Create student pairings from quiz scores

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
| | `data/<course>/<quiz>_<id>_pairing_via_med_YYYYMMDD.csv` — student pairings (median method) |
| **Output — figures** | `figures/<course>/<quiz>_<id>_dist_euclid_YYYYMMDD.png` — distance matrix heatmap (present students) |
| | `figures/<course>/<quiz>_<id>_compare_pairing_methods_YYYYMMDD.png` — comparison of all four pairing methods |

**Typical workflow:**
1. Mark which students are present in your `present_*.csv` file.
2. Run `python canvigator.py pair` and select the course, quiz, and presence CSV when prompted.
3. Open the generated `*_pairing_via_med_*.csv` and share pairings with the class.

---

#### `award-bonus` — Award both partner and retake bonus points

Detects partner groups and qualifying retakers, then awards both bonus types
as combined fudge points on their Canvas submissions. Each bonus defaults to
15% of quiz points; a student earning both receives 30% total.

**Partner bonus:** Students whose first attempts show a high fraction of
matching scores (default ≥ 80%) and closely timed answers (default within
10 seconds on ≥ 80% of questions) are grouped as partners. Groups larger than
3 are flagged for manual review.

**Retake bonus:** Students who retook the quiz at least 3 times with at least
1 day (24 hours) between qualifying attempts.

**Before running:** the `all-subs` task must have been run first for the same
quiz so that the submission CSVs exist.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_all_subs_and_events_YYYYMMDD.csv` (from `all-subs`) |
| | `data/<course>/<quiz>_<id>_all_subs_by_question_YYYYMMDD.csv` (from `all-subs`) |
| | `data/<course>/<quiz>_<id>_all_submissions_YYYYMMDD.csv` (from `all-subs`) |
| **Output** | `data/<course>/<quiz>_<id>_detected_partners_YYYYMMDD.csv` — detected partner groups |
| | `data/<course>/<quiz>_<id>_retake_qualified_YYYYMMDD.csv` — students qualifying for retake bonus |
| | `data/<course>/<quiz>_<id>_scores_w_bonus_YYYYMMDD.csv` — scores with partner_bonus, retake_bonus, and combined bonus columns |
| **Canvas side-effect** | Sets `fudge_points` on the student's highest-scoring attempt and leaves a submission comment describing the bonus breakdown (skipped in `--dry-run` mode) |

Use `--dry-run` to preview which students would receive bonus points without
modifying anything in Canvas. In dry-run mode the scores CSV is named
`*_scores_w_bonus_dryrun_*.csv`.

**Typical workflow:**
1. Run `python canvigator.py all-subs` to export submission data for all quizzes.
2. Run `python canvigator.py [--dry-run] award-bonus` and select the course, quiz, and date when prompted.
3. Review the `*_detected_partners_*.csv` and `*_retake_qualified_*.csv` to verify the results.

---

#### `award-bonus-partner-only` — Award only the partner bonus

Same as `award-bonus` but only detects and awards the partner bonus (no retake
bonus). Useful when you want to reward collaborative work without the retake
incentive.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_all_subs_and_events_YYYYMMDD.csv` (from `all-subs`) |
| | `data/<course>/<quiz>_<id>_all_subs_by_question_YYYYMMDD.csv` (from `all-subs`) |
| **Output** | `data/<course>/<quiz>_<id>_detected_partners_YYYYMMDD.csv` — detected partner groups |
| | `data/<course>/<quiz>_<id>_scores_w_bonus_YYYYMMDD.csv` — scores with bonus column |
| **Canvas side-effect** | Sets `fudge_points` on the student's highest-scoring attempt and leaves a submission comment (skipped in `--dry-run` mode) |

---

#### `award-bonus-retake-only` — Award only the retake bonus

Same as `award-bonus` but only detects and awards the retake bonus (no partner
detection). Useful when you want to incentivize retakes independently.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_all_submissions_YYYYMMDD.csv` (from `all-subs`) |
| **Output** | `data/<course>/<quiz>_<id>_retake_qualified_YYYYMMDD.csv` — students qualifying for retake bonus |
| | `data/<course>/<quiz>_<id>_scores_w_bonus_YYYYMMDD.csv` — scores with bonus column |
| **Canvas side-effect** | Sets `fudge_points` on the student's highest-scoring attempt and leaves a submission comment (skipped in `--dry-run` mode) |

---

#### `all-subs` — Export all quiz submissions and events

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

#### `get-quiz-questions` — Export quiz question content

Exports quiz metadata and question content to a CSV file. Skips downloading
student submission data, so this is a quick way to get a snapshot of the quiz
structure.

| | Files |
|---|---|
| **Input** | _(none — data comes from the Canvas API)_ |
| **Output** | `data/<course>/<quiz>_<id>_data_and_content_YYYYMMDD.csv` |

The output CSV contains columns: `quiz_id`, `assignment_id`, `id`, `position`,
`question_name`, `question_type`, `question_text`, `points_possible`, `answers`
(JSON-encoded). The `assignment_id` enables joining quiz data with
gradebook/assignment exports.

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

| | Files |
|---|---|
| **Input** | All CSV files in `data/<course>/` |
| **Output** | `data/<course>/anon_mapping_YYYYMMDD.csv` — mapping of original IDs to anonymous IDs |
| | `data/<course>/anonymized/` — directory containing anonymized copies of all CSVs |
| | `data/<course>/anonymized_YYYYMMDD.zip` — zip archive of the anonymized directory |

---

#### `export-gradebook` — Export course gradebook

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

#### `quiz-reminder` — Send quiz reminder messages to students

Sends personalized Canvas messages to students based on their quiz performance.
Students who have not yet attempted the quiz receive a reminder to make an
attempt. Students who attempted but scored below perfect receive encouragement
to retake. Students with perfect scores are skipped.

| | Files |
|---|---|
| **Input** | _(none — uses the student_analysis report downloaded from Canvas and enrolled student list)_ |
| **Output** | _(none — messages are sent as Canvas conversations)_ |
| **Canvas side-effect** | Sends a Canvas conversation message to each student who hasn't attempted or hasn't achieved a perfect score (skipped in `--dry-run` mode) |

Use `--dry-run` to preview all messages (recipient, subject, body, and reason)
without sending anything to Canvas.
