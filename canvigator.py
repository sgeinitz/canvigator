#!/usr/bin/env python3
import sys

task_descriptions = {
    'assess-replies': 'Assess student follow-up replies using local LLM (requires get-replies)',
    'award-bonus': 'Award partner + retake bonus points for a quiz',
    'award-bonus-partner-only': 'Award only the partner bonus points',
    'award-bonus-retake-only': 'Award only the retake bonus points',
    'create-pairs': 'Create student pairings from quiz scores',
    'create-quiz': 'Create an unpublished placeholder quiz on Canvas',
    'export-anon-data': 'Export anonymized course data (no Canvas API needed)',
    'generate-open-ended-questions': 'Generate open-ended questions from a tagged quiz (requires get-quiz-questions --tag)',
    'get-activity': 'Export student activity data',
    'get-all-subs': 'Export all quiz submissions and events',
    'get-gradebook': 'Export course gradebook',
    'get-quiz-questions': 'Export quiz question content',
    'get-replies': 'Retrieve student replies to follow-up questions',
    'send-follow-up-question': 'Send the most-missed open-ended follow-up question to students',
    'send-quiz-reminder': 'Send quiz reminder messages to students',
}
tasks = list(task_descriptions.keys())


def print_help():
    """Print usage information with task descriptions."""
    print("Usage: canvigator.py [--dry-run] [--tag] [--crn <CRN>] <task>\n")
    print("Options:")
    print("  --dry-run      Preview changes without modifying Canvas (bonus, reminder, and follow-up tasks)")
    print("  --tag          Use a local LLM via Ollama to tag questions (get-quiz-questions only)")
    print("  --crn <CRN>    Select course by CRN (last 5 digits of course code)")
    print("  --reply-window-days N  Days to accept replies after follow-up sent (default: 5, get-replies only)\n")
    print("Tasks:")
    max_name = max(len(t) for t in tasks)
    for name, desc in task_descriptions.items():
        print(f"  {name:<{max_name}}  {desc}")
    print("\nNotes:")
    print("  generate-open-ended-questions uses a local LLM (via Ollama) in two steps:")
    print("    1. Classifies each question as 'explain' (oral) or 'draw' (visual)")
    print("    2. Generates a mode-appropriate open-ended question for instructor review")
    print("  Output CSV includes a question_mode column so the instructor can override choices.")


def _run_quiz_task(task, quiz, dry_run, tag, reply_window_days):
    """Dispatch a quiz-level task to the appropriate method."""
    if task == 'get-quiz-questions':
        quiz.getQuizQuestions(tag=tag)
    elif task == 'get-replies':
        quiz.getFollowUpReplies(reply_window_days=reply_window_days)
    elif task == 'assess-replies':
        quiz.assessFollowUpReplies()
    elif task == 'send-quiz-reminder':
        quiz.sendQuizReminders(dry_run=dry_run)
    elif task == 'send-follow-up-question':
        quiz.sendFollowUpQuestions(dry_run=dry_run)
    elif task == 'create-pairs':
        quiz.openPresentCSV()
        quiz.generateDistanceMatrix(only_present=True)
        quiz.comparePairingMethods()
        quiz.createStudentPairings(method='med', write_csv=True)
    elif task == 'award-bonus':
        quiz.detectPartners()
        quiz.detectRetakers()
        quiz.awardBonusPoints(dry_run=dry_run)
    elif task == 'award-bonus-partner-only':
        quiz.detectPartners()
        quiz.awardBonusPoints(dry_run=dry_run)
    elif task == 'award-bonus-retake-only':
        quiz.detectRetakers()
        quiz.awardBonusPoints(dry_run=dry_run)


args = sys.argv[1:]
dry_run = '--dry-run' in args
if dry_run:
    args.remove('--dry-run')

tag = '--tag' in args
if tag:
    args.remove('--tag')

crn = None
if '--crn' in args:
    crn_idx = args.index('--crn')
    if crn_idx + 1 >= len(args):
        print("Error: --crn requires a 5-digit CRN value")
        sys.exit(1)
    crn = args[crn_idx + 1]
    if not crn.isdigit() or len(crn) != 5:
        print(f"Error: CRN must be exactly 5 digits, got '{crn}'")
        sys.exit(1)
    args.pop(crn_idx)  # remove '--crn'
    args.pop(crn_idx)  # remove the CRN value

reply_window_days = 5
if '--reply-window-days' in args:
    rw_idx = args.index('--reply-window-days')
    if rw_idx + 1 >= len(args):
        print("Error: --reply-window-days requires a numeric value")
        sys.exit(1)
    try:
        reply_window_days = int(args[rw_idx + 1])
        if reply_window_days < 1:
            raise ValueError
    except ValueError:
        print(f"Error: --reply-window-days must be a positive integer, got '{args[rw_idx + 1]}'")
        sys.exit(1)
    args.pop(rw_idx)  # remove '--reply-window-days'
    args.pop(rw_idx)  # remove the value

if len(args) < 1:
    print_help()
    sys.exit(1)

task = args[0]

if task in ("help", "--help"):
    print_help()
    sys.exit(0)

if task not in tasks:
    print(f"Invalid task: '{task}'. Run with --help to see available tasks.")
    sys.exit(1)

# export-anon-data works with local files only — no Canvas API needed
if task == 'export-anon-data':
    import logging
    from pathlib import Path
    import canvigator_utils as cu
    import canvigator_course as cc

    cu.setup_logging()
    logger = logging.getLogger(__name__)

    data_base = Path.cwd() / "data"
    if not data_base.exists():
        print("No data directory found.")
        sys.exit(1)

    course_dirs = sorted([d for d in data_base.iterdir() if d.is_dir() and not d.name.startswith('.')])

    if crn:
        matches = [d for d in course_dirs if d.name.endswith(f"_{crn}")]
        if not matches:
            print(f"Error: No data directory found for CRN '{crn}'")
            sys.exit(1)
        course_data_path = matches[0]
    else:
        if not course_dirs:
            print("No course data directories found.")
            sys.exit(1)
        print("\nCourse data directories:")
        for i, d in enumerate(course_dirs, start=1):
            print(f"[ {i:2d} ] {d.name}")
        idx = cu.prompt_for_index("\nSelect course data directory: ", len(course_dirs) - 1)
        course_data_path = course_dirs[idx]

    print(f"\nSelected: {course_data_path.name}")
    cc.exportAnonymizedData(course_data_path)
    print("\n*** Done ***\n")
    sys.exit(0)

import os
import logging
import canvasapi
import canvigator_course as cc
import canvigator_quiz as cq
import canvigator_utils as cu

cu.setup_logging()
logger = logging.getLogger(__name__)

# User/instructor must first have created and downloaded a token in Canvas, and
# set up environment vars appropriately (e.g. 'source <file w/ canvas info>').
API_URL = os.environ.get("CANVAS_URL")
API_KEY = os.environ.get("CANVAS_TOKEN")
if not API_URL or not API_KEY:
    print("Error: CANVAS_URL and CANVAS_TOKEN must be set. Run 'source set_env.sh' first.")
    sys.exit(1)

canv_config = cu.CanvigatorConfig()

# Initialize a new Canvas object
canvas = canvasapi.Canvas(API_URL, API_KEY)

# Select course by CRN if provided, otherwise prompt user interactively
if crn:
    course_choice = cu.selectCourseByCRN(canvas, crn)
else:
    course_choice = cu.selectCourse(canvas)

print(f"\nSelected course: {course_choice.name}")

course = cc.CanvigatorCourse(canvas, course_choice, canv_config, verbose=False)

if task == 'get-activity':
    course.saveStudentActivity(canv_config.data_path)

elif task == 'get-all-subs':
    course.getAllQuizzesAndSubmissions()

elif task == 'create-quiz':
    course.createQuiz()

elif task == 'get-gradebook':
    course.exportGradebook(canv_config.data_path)

elif task == 'generate-open-ended-questions':
    # Driven by a pre-selected tagged-questions CSV, not a Canvas quiz
    tagged_csv = cu.selectCSVFromList(
        canv_config.data_path,
        'questions_w_tags',
        "\nSelect tagged questions CSV (using index in '[ ]'): ",
    )
    cq.generateOpenEndedQuestions(tagged_csv)

else:
    # All remaining tasks require quiz selection
    quiz_choice = cu.selectFromList(course_choice.get_quizzes(), "quiz")
    print(f"\nSelected quiz: {quiz_choice.title}")
    skip = task in ('get-quiz-questions', 'get-replies', 'assess-replies')
    quiz = cq.CanvigatorQuiz(canvas, course, quiz_choice, canv_config, verbose=False, skip_student_data=skip)
    _run_quiz_task(task, quiz, dry_run, tag, reply_window_days)

print("\n*** Done ***\n")
