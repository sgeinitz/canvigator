#!/usr/bin/env python3
import sys

tasks = ['activity', 'award-bonus', 'award-bonus-partner-only', 'award-bonus-retake-only', 'pair', 'all-subs',
         'get-quiz-questions', 'create-quiz', 'export-anon-data']

args = sys.argv[1:]
dry_run = '--dry-run' in args
if dry_run:
    args.remove('--dry-run')

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

if len(args) < 1:
    print(f"Usage: canvigator.py [--dry-run] [--crn <CRN>] {' | '.join(tasks)}")
    sys.exit(1)

task = args[0]

if task in ("help", "--help"):
    print(f"Usage: canvigator.py [--dry-run] [--crn <CRN>] {' | '.join(tasks)}")
    print("  --dry-run      Preview bonus changes without modifying Canvas")
    print("  --crn <CRN>    Select course by CRN (last 5 digits of course code)")
    sys.exit(0)

if task not in tasks:
    print(f"Invalid task: '{task}'. Valid tasks: {', '.join(tasks)}")
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
    print("\n** Done ***\n")
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
    raise Exception("'CANVAS_' environment variables not set - see installation instructions to resolve this")

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

if task == 'activity':
    course.saveStudentActivity(canv_config.data_path)

elif task == 'all-subs':
    course.getAllQuizzesAndSubmissions()

elif task == 'get-quiz-questions':
    quiz_choice = cu.selectFromList(course_choice.get_quizzes(), "quiz")
    print(f"\nSelected quiz: {quiz_choice.title}")
    quiz = cq.CanvigatorQuiz(canvas, course, quiz_choice, canv_config, verbose=False, skip_student_data=True)
    quiz.getQuizQuestions()

elif task == 'create-quiz':
    course.createQuiz()

elif task in ['pair', 'award-bonus', 'award-bonus-partner-only', 'award-bonus-retake-only']:
    # Prompt user to select a quiz
    quiz_choice = cu.selectFromList(course_choice.get_quizzes(), "quiz")
    print(f"\nSelected quiz: {quiz_choice.title}")

    # Obtain quiz data for the selected task.
    quiz = cq.CanvigatorQuiz(canvas, course, quiz_choice, canv_config, verbose=False)

    if task == 'pair':
        # Open the CSV file with student data for who is present today and recalculate distance matrix.
        quiz.openPresentCSV()
        quiz.generateDistanceMatrix(only_present=True)

        # Compare all four methods of pairings students
        quiz.comparePairingMethods()

        # Generate pairings for today using the median method
        quiz.createStudentPairings(method='med', write_csv=True)

    elif task == 'award-bonus':
        # Detect partners and retakers, then award both bonus types
        quiz.detectPartners()
        quiz.detectRetakers()
        quiz.awardBonusPoints(dry_run=dry_run)

    elif task == 'award-bonus-partner-only':
        # Detect partners and award only the partner bonus
        quiz.detectPartners()
        quiz.awardBonusPoints(dry_run=dry_run)

    elif task == 'award-bonus-retake-only':
        # Detect retakers and award only the retake bonus
        quiz.detectRetakers()
        quiz.awardBonusPoints(dry_run=dry_run)

print("\n** Done ***\n")
