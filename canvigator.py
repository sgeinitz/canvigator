#!/usr/bin/env python3
import sys

tasks = ['activity', 'award-bonus', 'pair', 're-award-bonus']
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
import canvigator_course
import canvigator_quiz
import canvigator_utils as pu
#import picata_config as pc

# User/instructor must first have created and downloaded a token in Canvas, and
# set up environment vars appropriately (e.g. 'source <file w/ canvas info>').
API_URL = os.environ.get("CANVAS_URL")
API_KEY = os.environ.get("CANVAS_TOKEN")
if not (API_URL or API_KEY):
    raise Exception("'CANVAS_' environment variables not set - see installation instructions to resolve this")

canv_config = pc.CanvigatorConfig()

# Initialize a new Canvas object
canvas = canvasapi.Canvas(API_URL, API_KEY)

# Prompt user to select a course
#chosen_course = pu.selectFromList(canvas.get_courses(), "course")
chosen_course = pu.selectCourse(canvas)


print(f"\nSelected course: {chosen_course.name}")

pica_course = pu.CanvigatorCourse(chosen_course, canv_config, verbose=False)

if task == 'activity':
    pica_course.saveStudentActivity(canv_config.data_path)

elif task in ['pair', 'award-bonus', 're-award-bonus']:
    # Prompt user to select a quiz
    chosen_quiz = pu.selectFromList(chosen_course.get_quizzes(), "quiz")
    print(f"\nSelected quiz: {chosen_quiz.title}")

    # Obtain quiz data and generate plots to visualize the data.
    pica_quiz = pu.CanvigatorQuiz(canvas, chosen_quiz, canv_config, verbose=False)
    pica_quiz.generateQuestionHistograms()
    pica_quiz.generateDistanceMatrix(only_present=False)
    pica_quiz.getUserQuizEvents()

    if task == 'pair':
        # Open the CSV file with student data for who is present today and recalculate distance matrix.
        pica_quiz.openPresentCSV()
        pica_quiz.generateDistanceMatrix(only_present=True)

        # Compare all four methods of pairings students
        pica_quiz.comparePairingMethods()

        # Generate pairings for today using the median method
        pica_quiz.createStudentPairings(method='med', write_csv=True)

    elif task == 'award-bonus':
        # Prompt user to find the pairings CSV file
        pica_quiz.getPastPairingsCSV()

        # Check if paired students have distance of 0
        pica_quiz.checkForBonusEarned()

        # Award bonus points to students who received it by setting fudge points
        pica_quiz.awardBonusPoints()

    elif task == 're-award-bonus':
        # Prompt user to find the pairings CSV file
        pica_quiz.getPastBonusCSV()

        # Re-Award bonus points to students who received it by setting fudge points
        pica_quiz.reAwardBonusPoints()

print("\n** Done ***\n")
