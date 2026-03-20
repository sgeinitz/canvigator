#!/usr/bin/env python3
import sys

tasks = ['activity', 'award-bonus', 'pair', 're-award-bonus', 'all-subs']

args = sys.argv[1:]
dry_run = '--dry-run' in args
if dry_run:
    args.remove('--dry-run')

if len(args) < 1:
    print(f"Usage: canvigator.py [--dry-run] {' | '.join(tasks)}")
    sys.exit(1)

task = args[0]

if task in ("help", "--help"):
    print(f"Usage: canvigator.py [--dry-run] {' | '.join(tasks)}")
    print("  --dry-run    Preview bonus changes without modifying Canvas")
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

# Prompt user to select a course
course_choice = cu.selectCourse(canvas)

print(f"\nSelected course: {course_choice.name}")

course = cc.CanvigatorCourse(canvas, course_choice, canv_config, verbose=False)

if task == 'activity':
    course.saveStudentActivity(canv_config.data_path)

elif task == 'all-subs':
    course.getAllQuizzesAndSubmissions()

elif task in ['pair', 'award-bonus', 're-award-bonus']:
    # Prompt user to select a quiz
    quiz_choice = cu.selectFromList(course_choice.get_quizzes(), "quiz")
    print(f"\nSelected quiz: {quiz_choice.title}")

    # Obtain quiz data and generate plots to visualize the data.
    quiz = cq.CanvigatorQuiz(canvas, course, quiz_choice, canv_config, verbose=False)
    quiz.generateQuestionHistograms()
    quiz.generateDistanceMatrix(only_present=False)
    quiz.getUserQuizEvents()

    if task == 'pair':
        # Open the CSV file with student data for who is present today and recalculate distance matrix.
        quiz.openPresentCSV()
        quiz.generateDistanceMatrix(only_present=True)

        # Compare all four methods of pairings students
        quiz.comparePairingMethods()

        # Generate pairings for today using the median method
        quiz.createStudentPairings(method='med', write_csv=True)

    elif task == 'award-bonus':
        # Prompt user to find the pairings CSV file
        quiz.getPastPairingsCSV()

        # Check if paired students have distance of 0
        quiz.checkForBonusEarned()

        # Award bonus points to students who received it by setting fudge points
        quiz.awardBonusPoints(dry_run=dry_run)

    elif task == 're-award-bonus':
        # Prompt user to find the pairings CSV file
        quiz.getPastBonusCSV()

        # Re-Award bonus points to students who received it by setting fudge points
        quiz.reAwardBonusPoints(dry_run=dry_run)

print("\n** Done ***\n")
