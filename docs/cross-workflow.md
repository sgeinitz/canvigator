# Cross-workflow tasks

Tasks that pull signals from multiple workflows and synthesize them into a
single artifact for the instructor.

| Task | Summary |
|---|---|
| [`prep-class-digest`](#prep-class-digest) | Synthesize a 1-page Markdown brief on cohort gaps from quiz misses, follow-up reply themes, and media-recording transcripts; ends with 2–3 suggested in-class discussion questions. |

Per-task `--help` is also available from the CLI:
`python canvigator.py <task> --help`.

← [Back to main README](../README.md)

---

## `prep-class-digest`

*Synthesize a 1-page Markdown brief on cohort gaps and 2–3 in-class discussion questions.*

Synthesizes a 1-page brief from three signal sources collected over a
configurable lookback window (default 7 days, override with `--days N`):

- **Recent quiz misses** — joined to the topic tags from
  `*_questions_w_tags_*.csv` so misses are aggregated by concept rather than
  by individual question.
- **Follow-up reply themes** — failing/borderline rows in
  `*_followup_assessments.csv` (within the window) are summarized by a local
  Gemma 4 call into "where students went wrong" bullets, one block per
  quiz/question.
- **Media-recording transcripts** — every
  `assignment<id>_recordings_*.csv` in the window is run through the existing
  tag-classification + theme-extraction Gemma 4 pipeline (the same one used
  by [`analyze-media-recordings`](media-recording-checkins.md#analyze-media-recordings)).

The brief ends with **2–3 suggested in-class discussion questions** targeted
at the highest-priority gaps. By default these are drafted by the local
Gemma 4 model with a full-fidelity prompt that includes the derived theme
bullets and evidence snippets. With `--cloud-questions` (`-q`) the step
routes to cloud Gemini 3 instead, but only with a *redacted* prompt that
contains tag names + integer miss counts + integer theme-cluster counts —
no transcripts, no `criteria_evaluations`, no theme text. Student-derived
content stays local in either configuration.

Empty windows (no quiz/follow-up/recording activity in the lookback
period) short-circuit with a friendly message and write no file. Same-day
re-runs overwrite.

**Before running:** `data/<course>/` should already be populated by upstream
tasks: [`get-quiz-submission-events`](misc-tasks.md#get-quiz-submission-events)
and [`get-quiz-questions --tag`](quiz-followup-questions.md#get-quiz-questions)
for misses, [`assess-replies`](quiz-followup-questions.md#assess-replies) for
follow-up themes, [`get-media-recordings`](media-recording-checkins.md#get-media-recordings)
for transcripts.

| | Files |
|---|---|
| **Input** | `data/<course>/<quiz>_<id>_all_subs_by_question_YYYYMMDD.csv` files in window (from [`get-quiz-submission-events`](misc-tasks.md#get-quiz-submission-events)) |
| | `data/<course>/<quiz>_<id>_questions_w_tags_YYYYMMDD.csv` per quiz (from [`get-quiz-questions --tag`](quiz-followup-questions.md#get-quiz-questions)) |
| | `data/<course>/<quiz>_<id>_followup_assessments.csv` (from [`assess-replies`](quiz-followup-questions.md#assess-replies), persistent) |
| | `data/<course>/assignment<id>_recordings_YYYYMMDD.csv` files in window (from [`get-media-recordings`](media-recording-checkins.md#get-media-recordings)) |
| **Output** | `data/<course>/class_digest_YYYYMMDD.md` — Markdown report (5 sections: header, quiz performance table, follow-up reply themes, media-recording themes, suggested discussion questions) |

### Examples

```bash
python canvigator.py prep-class-digest                  # default 7-day window, Gemma-only
python canvigator.py --days 14 prep-class-digest        # widen the window
python canvigator.py --cloud-questions prep-class-digest  # opt the question step into Gemini
```

Open the generated `class_digest_*.md`, scan the priority gaps, and use the
suggested questions to seed a 5–10 minute in-class discussion.
