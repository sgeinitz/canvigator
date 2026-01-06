#!/usr/bin/env python3
import sys

tasks = ['activity', 'award-bonus', 'pair', 're-award-bonus', 'all-subs']
task = None

if len(sys.argv) < 2:
    print(f"Usage: canvigator {' | '.join(tasks)}")
    sys.exit(1)
else: 
    task = sys.argv[1]

if task == "help" or task == "--help":
    print(f"Usage: canvigator.py [{'|'.join(tasks)}]")
    sys.exit(1)

#elif task not in tasks:
#    # prompt the user to select a task
#    task_ind = input("Select a valid task: [0] activity, [1] award-bonus, [2] pair, or [3] re-award-bonus\n")
#    task_ind = int(re.sub(r'\D', '', task_ind))
#    if task_ind < 0 or task_ind > 3:
#        print("Invalid task selected. Exiting.")
#        sys.exit(1)
#    else:
#        task = tasks[task_ind]

import os
import re
import canvasapi
import canvigator_course as cc
import canvigator_quiz as cq
import canvigator_utils as cu

# User/instructor must first have created and downloaded a token in Canvas, and
# set up environment vars appropriately (e.g. 'source <file w/ canvas info>').
API_URL = os.environ.get("CANVAS_URL")
API_KEY = os.environ.get("CANVAS_TOKEN")
if not (API_URL or API_KEY):
    raise Exception("'CANVAS_' environment variables not set - see installation instructions to resolve this")

canv_config = cu.CanvigatorConfig()

# Initialize a new Canvas object
canvas = canvasapi.Canvas(API_URL, API_KEY)

# Prompt user to select a course
#course_choice = cu.selectFromList(canvas.get_courses(), "course")
course_choice = cu.selectCourse(canvas)


print(f"\nSelected course: {course_choice.name}")

course = cc.CanvigatorCourse(canvas, course_choice, canv_config, verbose=False)

if task == 'activity':
    course.saveStudentActivity(canv_config.data_path)

elif task  == 'all-subs':
    course.getAllQuizzesAndSubmissions()
    #quiz_choice = cu.selectFromList(course_choice.get_quizzes(), "quiz")
    #print(f"\nSelected quiz: {quiz_choice.title}")
    #quiz = cq.CanvigatorQuiz(canvas, course, quiz_choice, canv_config, verbose=False)
    #$quiz.getAllSubmissionsAndEvents()

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
        quiz.awardBonusPoints()

    elif task == 're-award-bonus':
        # Prompt user to find the pairings CSV file
        quiz.getPastBonusCSV()

        # Re-Award bonus points to students who received it by setting fudge points
        quiz.reAwardBonusPoints()

print("\n** Done ***\n")
