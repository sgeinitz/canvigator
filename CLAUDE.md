# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Canvigator is a Python terminal tool for educators to automate Canvas LMS tasks and implement pedagogical techniques (collaborative learning, continuous assessments). It integrates with Canvas via the `canvasapi` library.

## Setup & Running

```bash
# First-time setup
./configure.sh          # Creates set_env.sh with CANVAS_URL and CANVAS_TOKEN
pip install -r requirements.txt

# Before each session
source set_env.sh

# Run a task
python canvigator.py [--dry-run] [--crn <5-digit-CRN>] <task>
```

Available tasks: `activity`, `award-bonus`, `award-bonus-partner-only`, `award-bonus-retake-only`, `pair`, `all-subs`, `get-quiz-questions`, `create-quiz`, `export-anon-data`, `export-gradebook`, `quiz-reminder`

The `--dry-run` flag works with bonus and reminder tasks to preview without modifying Canvas.

## Linting

```bash
# Match CI exactly (GitHub Actions runs on every push):
flake8 *.py --count --select=E9,F63,F7,F82 --show-source --statistics
flake8 *.py --count --max-complexity=15 --max-line-length=160 --ignore=D100,D205,D210,D400,D401,W504,E111,E402 --statistics
```

Requires: `pip install flake8 flake8-docstrings`

## Testing

```bash
# Run all tests
python -m pytest test_canvigator.py -v

# Run a single test class
python -m pytest test_canvigator.py::TestCreateStudentPairings -v

# Run a single test
python -m pytest test_canvigator.py::TestMakeAnonId::test_deterministic -v
```

Tests cover utility functions, anonymization logic, and core algorithms (partner detection, union-find grouping, student pairing). Canvas API interactions are not tested; use `--dry-run` for manual verification of those.

## Architecture

Four Python files, task-driven design:

- **`canvigator.py`** - Entry point. Parses CLI args, initializes Canvas API connection, routes to task handlers. The `export-anon-data` task is special: it works on local files only (no Canvas API needed).

- **`canvigator_utils.py`** - Shared utilities. `CanvigatorConfig` manages data/figures directory paths. Interactive selection helpers (`selectCourse`, `selectFromList`, `prompt_for_index`). Logging setup (file: INFO, console: WARNING).

- **`canvigator_course.py`** - Course-level operations via `CanvigatorCourse` class. Handles: student activity export, gradebook export, quiz creation, bulk submission retrieval. Also contains standalone anonymization functions (`exportAnonymizedData`, `_make_anon_id`).

- **`canvigator_quiz.py`** - Quiz-level operations via `CanvigatorQuiz` class. The largest module. Handles: partner detection (score + timestamp overlap), retake detection, bonus point awarding, student pairing (Euclidean distance matrix with greedy algorithm), quiz reminders, submission/event export.

### Data flow for bonus awarding (most complex task)

```
CLI args → select course → select quiz → CanvigatorQuiz
  → detectPartners() (loads submission events, compares scores & timestamps, union-find grouping)
  → detectRetakers() (identifies students with ≥3 attempts ≥1 day apart)
  → awardBonusPoints() (sets Canvas fudge points + comments on highest-scoring attempt)
```

## Conventions

- Environment variables `CANVAS_URL` and `CANVAS_TOKEN` must be set (via `source set_env.sh`). Never committed; `set_env.sh` is in `.gitignore`.
- All output goes to `data/<course_prefix><course_number>_<CRN>/` and `figures/<course_prefix><course_number>_<CRN>/`.
- File naming: `quiz<id>_<operation>_YYYYMMDD.csv` (dates from `canvigator_utils.today_str()`).
- Every module uses `logger = logging.getLogger(__name__)`. Log file: `canvigator_YYYYMMDD.log`.
- All functions and methods require docstrings (enforced by flake8-docstrings in CI).
- 4-space indentation used throughout despite E111 being suppressed in flake8.
