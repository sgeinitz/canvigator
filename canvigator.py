#!/usr/bin/env python3
import sys

tasks = ['activity', 'auto-award-bonus', 'award-bonus', 'pair', 're-award-bonus', 'all-subs']

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

elif task in ['pair', 'auto-award-bonus', 'award-bonus', 're-award-bonus']:
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

    elif task == 'auto-award-bonus':
        # Detect partners automatically from submission timestamps and scores
        quiz.detectPartners()

        # Award bonus points to detected partners
        quiz.awardBonusPoints(dry_run=dry_run)

    elif task == 'award-bonus':
        quiz.generateDistanceMatrix(only_present=False)

        # Prompt user to find the pairings CSV file
        quiz.getPastPairingsCSV()

        # Check if paired students have distance of 0
        quiz.checkForBonusEarned()

        # Award bonus points to students who received it by setting fudge points
        quiz.awardBonusPoints(dry_run=dry_run)

    elif task == 're-award-bonus':
        quiz.generateDistanceMatrix(only_present=False)

        # Prompt user to find the pairings CSV file
        quiz.getPastBonusCSV()

        # Re-Award bonus points to students who received it by setting fudge points
        quiz.reAwardBonusPoints(dry_run=dry_run)

print("\n** Done ***\n")
