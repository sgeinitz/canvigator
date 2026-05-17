

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
students send back through Canvas conversations. The assessment step runs
Gemma 4 locally, so student submissions never leave the instructor's
machine — no cloud-provider data-use policies to vet, no FERPA exposure,
and peace of mind on both sides of the gradebook. The instructor verifies
and edits each assessment before any feedback reaches the student.

Currently, this terminal-based tool works with the Canvas Learning Management System (LMS) using the 
[CanvasAPI](https://github.com/ucfopen/canvasapi). In addition to increased functionality, development 
plans for this project include extending it to other LMSs and creating a richer interface. Suggestions 
for other functionality/features (see Issues), as well as any feedback on the project, are welcomed.


### Setup

First-time setup (Canvas API token, Python dependencies, and the optional
Ollama configuration for LLM-assisted tasks) is documented in
**[docs/installation.md](docs/installation.md)**.


### Usage

```bash
source set_env.sh                            # set Canvas (and optional Ollama) environment variables (once per terminal session)
python canvigator.py <task>                  # run a task (prompts for course selection)
python canvigator.py --help                  # print the grouped task list and global options
python canvigator.py <task> --help           # print a per-task cheat-sheet (prerequisites, inputs/outputs, applicable flags, examples, related tasks)
python canvigator.py --crn <CRN> <task>      # select course by CRN (last 5 digits of course code)
python canvigator.py --dry-run <task>        # preview changes without modifying Canvas (bonus, reminder, follow-up, delete-old-conversations, get-media-recordings)
python canvigator.py --tag get-quiz-questions     # add LLM-generated topic tags to the quiz questions export
python canvigator.py --reply-window-days N <task>  # set the days-after-send window for assess-replies (default: 5)
python canvigator.py --months N delete-old-conversations  # cutoff age in months for delete-old-conversations (default: 6)
python canvigator.py --auto-grade get-media-recordings  # skip per-student review and auto-grade every submission at full credit
python canvigator.py --days N prep-class-digest          # synthesize a 1-page brief on cohort gaps from the last N days (default 7)
python canvigator.py --cloud-questions prep-class-digest # opt the discussion-question step into cloud Gemini 3 with a redacted prompt (default: local Gemma 4)
```

Every flag has a single-dash short alias: `-d` (`--dry-run`), `-t` (`--tag`),
`-a` (`--all`), `-c` (`--crn`), `-m` (`--months`), `-w`
(`--reply-window-days`), `-g` (`--auto-grade`), `-n` (`--days`), `-q`
(`--cloud-questions`), `-h` (`--help`). Short and long forms are
interchangeable.

The `--crn` option selects a course by its CRN (the last 5 digits of the Canvas
course code), bypassing the interactive course selection prompt. This is useful
for automated/scheduled runs, e.g. `python canvigator.py --crn 12345 get-activity`.

All tasks begin by prompting you to select a course (unless `--crn` is
supplied). Output files are written to `data/<course>/` and
`figures/<course>/`, where `<course>` is derived from the Canvas course code.
In the file names referenced from the docs below, `<quiz>` is the quiz title
(lowercased, spaces replaced with underscores), `<id>` is the Canvas quiz ID,
and `YYYYMMDD` is the current date.


### Tasks by workflow

Per-task documentation has been split out by workflow. Each page below covers
a connected set of tasks plus the order in which they're typically run.

| Workflow | Tasks |
|---|---|
| **[Peer instruction & retake bonuses](docs/peer-instruction-and-bonus.md)** | `create-pairs`, `award-bonus`, `award-bonus-partner-only`, `award-bonus-retake-only` |
| **[Quiz follow-up questions](docs/quiz-followup-questions.md)** | `get-quiz-questions`, `generate-follow-up-questions`, `send-quiz-reminder`, `send-follow-up-question`, `assess-replies`, `send-follow-up-assessments` |
| **[Media-recording check-ins](docs/media-recording-checkins.md)** | `create-media-recording-assignment`, `get-media-recordings`, `analyze-media-recordings` |
| **[Cross-workflow](docs/cross-workflow.md)** | `prep-class-digest` |
| **[Miscellaneous tasks](docs/misc-tasks.md)** | `create-quiz`, `export-anon-data`, `get-activity`, `get-quiz-submission-events`, `get-conversations`, `delete-old-conversations`, `get-gradebook`, `get-roster` |

For a quick on-the-CLI reference, run `python canvigator.py --help` (the
grouped task list with global options) or `python canvigator.py <task> --help`
(a focused cheat-sheet for one task: prerequisites, inputs/outputs, applicable
flags, examples, and related tasks to run before/after).
