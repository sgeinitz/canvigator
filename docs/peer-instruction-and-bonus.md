# Peer instruction and retake incentives

Tasks for quizzes built around peer collaboration and retake bonuses.

| Task | Summary |
|---|---|
| [`create-pairs`](#create-pairs) | Pair students for a collaborative in-class quiz using a prior quiz's score vectors. |
| [`award-bonus`](#award-bonus) | Award both partner and retake bonus points as fudge points on each student's best attempt. |
| [`award-bonus-partner-only`](#award-bonus-partner-only) | Same as `award-bonus` but skips retake detection. |
| [`award-bonus-retake-only`](#award-bonus-retake-only) | Same as `award-bonus` but skips partner detection. |

Per-task `--help` is also available from the CLI:
`python canvigator.py <task> --help`.

← [Back to main README](../README.md)

## End-to-end workflow

1. _(Optional, before a collaborative in-class quiz)_ Mark attendance in
   `data/<course>/present_*.csv`, then run **`create-pairs`** to pair up
   present students using a prior quiz's score vectors.
2. After the quiz attempts have been collected, run
   [`get-quiz-submission-events`](misc-tasks.md#get-quiz-submission-events) to
   export the submission CSVs that the bonus tasks read from.
3. Run **`award-bonus`** (or `award-bonus-partner-only` /
   `award-bonus-retake-only`) to detect partners and/or retakers and award
   fudge points on each student's best attempt. Use `--dry-run` first to
   preview without writing to Canvas.

---

## `create-pairs`

*Pair students for a collaborative in-class quiz using a prior quiz's score vectors.*

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

---

## `award-bonus`

*Award both partner and retake bonus points as fudge points on each student's best attempt.*

Detects partner groups and qualifying retakers, then awards both bonus types
as combined fudge points on their Canvas submissions. Each bonus defaults to
15% of quiz points; a student earning both receives 30% total.

**Partner bonus:** Students whose first attempts show a high fraction of
matching scores (default ≥ 80%) and closely timed answers (default within
5 seconds on ≥ 80% of questions) are grouped as partners. Groups larger than
3 are flagged for manual review.

**Retake bonus:** Students who retook the quiz at least 3 times with at least
1 day (24 hours) between qualifying attempts.

**Before running:** [`get-quiz-submission-events`](misc-tasks.md#get-quiz-submission-events)
must have been run first for the same quiz so that the submission CSVs exist.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_all_submissions_YYYYMMDD.csv` (from `get-quiz-submission-events`) |
| | `data/<course>/<quiz>_<id>_all_subs_by_question_YYYYMMDD.csv` (from `get-quiz-submission-events`) |
| | `data/<course>/<quiz>_<id>_all_subs_and_events_YYYYMMDD.csv` (from `get-quiz-submission-events`) |
| **Output** | `data/<course>/<quiz>_<id>_detected_partners_YYYYMMDD.csv` — detected partner groups |
| | `data/<course>/<quiz>_<id>_retake_qualified_YYYYMMDD.csv` — students qualifying for retake bonus |
| | `data/<course>/<quiz>_<id>_scores_w_bonus_YYYYMMDD.csv` — scores with partner_bonus, retake_bonus, and combined bonus columns |
| **Canvas side-effect** | Sets `fudge_points` on the student's highest-scoring attempt and leaves a submission comment describing the bonus breakdown (skipped in `--dry-run` mode) |

Use `--dry-run` to preview which students would receive bonus points without
modifying anything in Canvas. In dry-run mode the scores CSV is named
`*_scores_w_bonus_dryrun_*.csv`.

---

## `award-bonus-partner-only`

*Same as `award-bonus` but skips retake detection.*

Same as `award-bonus` but only detects and awards the partner bonus (no retake
bonus). Useful when you want to reward collaborative work without the retake
incentive.

**Before running:** [`get-quiz-submission-events`](misc-tasks.md#get-quiz-submission-events)
must have been run first for the same quiz.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_all_submissions_YYYYMMDD.csv` (from `get-quiz-submission-events`) |
| | `data/<course>/<quiz>_<id>_all_subs_by_question_YYYYMMDD.csv` (from `get-quiz-submission-events`) |
| | `data/<course>/<quiz>_<id>_all_subs_and_events_YYYYMMDD.csv` (from `get-quiz-submission-events`) |
| **Output** | `data/<course>/<quiz>_<id>_detected_partners_YYYYMMDD.csv` — detected partner groups |
| | `data/<course>/<quiz>_<id>_scores_w_bonus_YYYYMMDD.csv` — scores with bonus column |
| **Canvas side-effect** | Sets `fudge_points` on the student's highest-scoring attempt and leaves a submission comment (skipped in `--dry-run` mode) |

---

## `award-bonus-retake-only`

*Same as `award-bonus` but skips partner detection.*

Same as `award-bonus` but only detects and awards the retake bonus (no partner
detection). Useful when you want to incentivize retakes independently.

**Before running:** [`get-quiz-submission-events`](misc-tasks.md#get-quiz-submission-events)
must have been run first for the same quiz.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_all_submissions_YYYYMMDD.csv` (from `get-quiz-submission-events`) |
| **Output** | `data/<course>/<quiz>_<id>_retake_qualified_YYYYMMDD.csv` — students qualifying for retake bonus |
| | `data/<course>/<quiz>_<id>_scores_w_bonus_YYYYMMDD.csv` — scores with bonus column |
| **Canvas side-effect** | Sets `fudge_points` on the student's highest-scoring attempt and leaves a submission comment (skipped in `--dry-run` mode) |
