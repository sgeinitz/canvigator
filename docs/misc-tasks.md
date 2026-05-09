# Miscellaneous tasks

Standalone tasks that don't belong to one of the multi-step workflows. Most
are simple Canvas exports (gradebook, roster, conversations); a few are
foundational utilities used by other workflows
(`get-quiz-submission-events` is the upstream of the
[bonus tasks](peer-instruction-and-bonus.md) and
[`prep-class-digest`](cross-workflow.md)) or one-shot administrative actions
(`create-quiz`, `delete-old-conversations`, `export-anon-data`).

| Task | Summary |
|---|---|
| [`create-quiz`](#create-quiz) | Interactively create an unpublished quiz on Canvas (placeholder or LLM-generated questions). |
| [`export-anon-data`](#export-anon-data) | Anonymize the local CSVs for a course (no Canvas API needed); produces a zip archive plus an ID mapping file. |
| [`get-activity`](#get-activity) | Export per-student page-view, participation, and weekly-activity data for the course. |
| [`get-quiz-submission-events`](#get-quiz-submission-events) | Export per-attempt submission and event data for a quiz; renders score, timing, and page-blur histograms. |
| [`get-conversations`](#get-conversations) | Export every Canvas conversation involving an active student in the course. |
| [`delete-old-conversations`](#delete-old-conversations) | Sweep + delete the instructor's old Canvas conversations (account-wide, not course-scoped). |
| [`get-gradebook`](#get-gradebook) | Export the full course gradebook (one row per student-assignment pair). |
| [`get-roster`](#get-roster) | Export the full course roster: students, teachers, TAs, designers, observers. |

Per-task `--help` is also available from the CLI:
`python canvigator.py <task> --help`.

← [Back to main README](../README.md)

---

## `create-quiz`

*Interactively create an unpublished quiz on Canvas (placeholder or LLM-generated questions).*

Interactively creates a new unpublished quiz on Canvas. Prompts the user for a
quiz title, then for each question prompts:

```
QN — [p]laceholder, [g]enerate w/ LLM, [e]nd quiz:
```

- **`p`** (placeholder) — asks for a one-line description and creates a
  multiple-choice placeholder with 1 point possible (and no answer choices).
- **`g`** (generate w/ LLM) — asks the user for a natural-language seed prompt
  (e.g. *"a question about combinatorics asking how many ways 11 soccer players
  can be selected from a team of 23"*) and routes it through the cloud text
  model (default `gemini-3-flash-preview`, requires `OLLAMA_API_KEY` — see
  [Ollama setup](installation.md#ollama-setup-optional)). The LLM returns a
  complete Canvas-shaped question of one of seven auto-gradable types:
  multiple-choice, multiple-answers, true/false, fill-in-multiple-blanks,
  multiple-dropdowns, matching, or calculated. The proposed question is
  rendered inline (type, name, text, answer choices with `*` marking the
  correct one, plus per-type extras like match pairs or formula variables) and
  the user is prompted `[y]/[r]/[s]` — accept, regenerate, or skip. Each
  `[r]egenerate` accumulates the rejected draft and passes the full list back
  to the LLM as anti-context (with sampling temperature bumped from 0.4 to
  0.8) so successive drafts diverge in scenario / values / question_type
  instead of recycling the same angle.
- **`e`** or empty input — finalizes the quiz with the questions added so far.

`points_possible` is forced to `1` for both placeholder and LLM-generated
questions; the instructor can refine wording, distractors, and points later in
the Canvas UI. The quiz is created with default settings:
`quiz_type='assignment'`, `time_limit=30`, `one_question_at_a_time=True`,
`cant_go_back=True`, `shuffle_answers=True`.

The cloud text model is lazy-loaded only on the first `g` choice, so a
pure-placeholder run still works without `OLLAMA_API_KEY` set.

| | Files |
|---|---|
| **Input** | _(none — interactive prompts only)_ |
| **Output** | _(none — quiz is created directly on Canvas, unpublished)_ |

---

## `export-anon-data`

*Anonymize the local CSVs for a course (no Canvas API needed); produces a zip archive plus an ID mapping file.*

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

## `get-activity`

*Export per-student page-view, participation, and weekly-activity data for the course.*

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

## `get-quiz-submission-events`

*Export per-attempt submission and event data for a quiz; renders score, timing, and page-blur histograms.*

Downloads detailed submission history and events for a user-selected quiz,
and renders per-question score histograms. Pass `--all` to iterate over every
published quiz in the course instead of prompting for one. This is the only
task that exports event history and per-question histogram figures.

| | Files |
|---|---|
| **Input** | _(none — data comes from the Canvas API)_ |
| **Output (per quiz)** | `data/<course>/<quiz>_<id>_student_analysis_YYYYMMDD.csv` — raw quiz report |
| | `data/<course>/<quiz>_<id>_all_submissions_YYYYMMDD.csv` — all submission attempts with scores |
| | `data/<course>/<quiz>_<id>_all_subs_by_question_YYYYMMDD.csv` — per-question results for each attempt |
| | `data/<course>/<quiz>_<id>_all_subs_and_events_YYYYMMDD.csv` — timestamped submission events |
| **Output — figures** | `figures/<course>/<quiz>_<id>_score_histograms_YYYYMMDD.png` — per-question score histograms |
| | `figures/<course>/<quiz>_<id>_timing_first_attempt_YYYYMMDD.png` — per-question first-attempt timing histograms (minutes) |
| | `figures/<course>/<quiz>_<id>_blurs_first_attempt_YYYYMMDD.png` — per-question first-attempt page-blur counts |

---

## `get-conversations`

*Export every Canvas conversation involving an active student in the course.*

Canvas conversations are not course-scoped, so this pulls the instructor's
full inbox and sent folders via `canvas.get_conversations()` (default scope
plus `scope='sent'`, deduped by `conversation_id`) and filters each
conversation's participants list against the active-student roster for the
selected course. Primary use case: back-fill the `conversation_id` for
follow-up sends made before the per-send manifest was added, so `assess-replies`
can still fetch those threads by ID.

| | Files |
|---|---|
| **Input** | _(none — data comes from the Canvas API)_ |
| **Output** | `data/<course>/conversations_YYYYMMDD.csv` |

The output CSV is sorted newest first and contains columns: `conversation_id`,
`subject`, `last_message_at`, `first_message_at`, `message_count`,
`workflow_state`, `student_ids` (comma-separated), `student_names`
(semicolon-separated), `n_student_participants`, `last_message`.
`first_message_at` is derived by fetching each matched conversation (one extra
API call per conversation) and taking the earliest message `created_at`; it
is also substituted into `last_message_at` when the list endpoint returns an
empty value (occasionally happens for older/archived threads), so sort order
stays meaningful.

---

## `delete-old-conversations`

*Sweep + delete the instructor's old Canvas conversations (account-wide, not course-scoped).*

Sweeps the instructor's Canvas inbox, sent folder, and archived folder for
conversations whose most recent message is older than `N` months
(approximate; one month = 30 days, default `N = 6`). Each match is shown
(last_message_at, subject, participant names) before any action is taken.
In live mode the task requires an explicit `yes` confirmation at the prompt
before any deletes run; `--dry-run` previews only.

Canvas conversations are not course-scoped, so this task skips course
selection entirely and operates on the full account inbox. Deletion uses
`Conversation.delete()` to remove the whole thread (not just individual
messages).

Override the cutoff with `--months`/`-m`, e.g.
`python canvigator.py --months 12 delete-old-conversations` keeps the
most recent year and deletes everything older.

| | Files |
|---|---|
| **Input** | _(none — operates directly against the Canvas inbox)_ |
| **Output** | _(none — matched conversations are listed in the terminal)_ |
| **Canvas side-effect** | Permanently deletes matched conversations (skipped in `--dry-run` mode); requires explicit `yes` confirmation in live mode |

---

## `get-gradebook`

*Export the full course gradebook (one row per student-assignment pair).*

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

## `get-roster`

*Export the full course roster: students, teachers, TAs, designers, observers.*

Iterates `canvas_course.get_enrollments()` with no role filter and writes a
minimal CSV of every enrolled person — students, teachers, TAs, designers, and
observers. Useful as a course-wide identity reference and as a stable input
for scripts outside Canvigator.

| | Files |
|---|---|
| **Input** | _(none — data comes from the Canvas API)_ |
| **Output** | `data/<course>/roster_YYYYMMDD.csv` |

The output CSV contains columns: `name`, `id`, `sis_id`, `enrollment_type`,
`state`. `enrollment_type` is the Canvas type string (`StudentEnrollment`,
`TeacherEnrollment`, `TaEnrollment`, `DesignerEnrollment`, `ObserverEnrollment`);
`state` is the enrollment state (`active`, `invited`, `inactive`, `completed`,
`rejected`, `deleted`).
