import os
import re
import sys
import random
import requests
import time
import pandas as pd
import scipy.stats as stats
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sbn
from datetime import datetime


class CanvigatorConfig:
    """ Canvigator configuration class, mostly for file paths """

    def __init__(self):
        """ Simple file path configuration for data and figures (for both input and output). """
        self.data_path = os.getcwd() + "/data/"
        self.figures_path = os.getcwd() + "/figures/"
        self.quiz_prefix = "quiz_"

    def addCourseToDataPath(self, course_path):
        """ Create data path directory if it does not already exist. """
        if not os.path.exists(self.data_path + course_path):
            os.makedirs(self.data_path + course_path)
        self.data_path += course_path + "/"


def selectFromList(paginated_list, item_type="item"):
    """
    A general function that takes a canvasapi paginated_list object
    and lists each item in it, then prompts user to select one item.
    """
    print(f"\nOptions:")
    i = 0
    subobject_list = []
    for i, so in enumerate(paginated_list):
        print(f"[ {i:2d} ] {so}")
        subobject_list.append(so)
    str_index = input(f"\nSelect {item_type} from above using index in square brackets: ")

    if int(str_index) < 0 or int(str_index) > i:
        raise IndexError("Invalid selection")

    return subobject_list[int(str_index)]


def selectCourse(canvas):
    """
    Given a canvas instance object, display a list of courses and prompt
    the user to select a course. The selected course is returned.

    1. First prompt the user to choose between past or current courses.
    2. List the selected type of courses (past or current).
    """

    past_courses = []
    current_courses = []
    current_date = datetime.now()

    for course in canvas.get_courses():
        try:
            course_name = course.name

            # Check start and end dates
            start_date = datetime.strptime(course.start_at, '%Y-%m-%dT%H:%M:%SZ') if course.start_at else None
            end_date = datetime.strptime(course.end_at, '%Y-%m-%dT%H:%M:%SZ') if course.end_at else None

            # Separate past and current courses
            if end_date and current_date > end_date:
                past_courses.append(course)
            elif not end_date or current_date <= end_date:
                current_courses.append(course)

        except Exception as e:
            print(f"Error: {e} - exiting")
            sys.exit(1)

    # Prompt user to select between past and current courses
    print("\nSelect Course Type:")
    print("[ 0 ] Past Courses")
    print("[ 1 ] Current Courses")
    selection = input("\nSelect course type (using index in '[ ]'): ")
    selection = int(re.sub(r'\D', '', selection))

    if selection == 0:
        print("\nPast Courses:")
        for i, course in enumerate(past_courses):
            print(f"[ {i:2d} ] {course.name}")
        valid_courses = past_courses

    elif selection == 1:
        print("\nCurrent Courses:")
        for i, course in enumerate(current_courses):
            print(f"[ {i:2d} ] {course.name}")
        valid_courses = current_courses
    else:
        raise IndexError("Invalid selection format. Please select '0' for past or '1' for current.")

    str_index = input("\nSelect a course from the list above (using index in '[ ]'): ")

    # if not str_index.startswith("[") or not str_index.endswith("]"):
    #    raise ValueError("Invalid selection format. Please use '[ ]' around the index.")
    # Convert input into an integer
    # course_index = int(str_index[1:-1])

    # remove any characters that are not digits
    course_index = int(re.sub(r'\D', '', str_index))

    if course_index < 0 or course_index >= len(valid_courses):
        raise IndexError("Invalid course selection.")

    return canvas.get_course(valid_courses[course_index].id)

def sendMessage(canvas, pica_course, pairs):
    """
    Message template to be sent to student pairs in Canvas. The user should be
    able to customize this message each time the program is run, rather than it
    being hard-coded here. One option is to prompt the user to input the message
    when the program runs. Or, to prompt them to specify an input file that will
    contain the message template.
    """
    message_template = "Hello {},\n  In our upcoming class session the {} of you \
    will meet to work out a problem together. If you don't already know one another, \
    then it may be helpful to plan on meeting in a certain section of the room, or a \
    share a distinguishing feature (e.g. 'I have a red hat on today'). \n\nThere \
    is no preparation required on your part, however, if you want to see the type of \
    problem you'll see today you can refer to the quiz question shown below. Please \
    wait until class for more details. \n\nBest,\nSteve \n \nQuestion from previous quiz: {}"

    message_dict1 = {2: 'two', 3: 'three'}
    subject_str = "Today's quiz review session - " + datetime.today().strftime('%Y.%m.%d')
    question_text = "A question from the quiz will be shown here."

    # Individual messages to be sent
    for pair in pairs:
        num_students = len(pair) - 1
        recipient_canvas_ids = [str(id) for id in pair[0:-1]]
        print("recipients =", recipient_canvas_ids)
        names = [pica_course.canvas_course.get_user(id).name.split()[0] for id in pair[0:-1]]
        names.sort()
        print("names =", names)
        names_str = ", ".join(names)
        message_str = message_template.format(names_str, message_dict1[num_students], question_text)
        print("message =")
        print(message_str)

        # Create_convo returns a list, so here convo[0] is the conversation object
        convo = canvas.create_conversation(recipient_canvas_ids, message_str, subject=subject_str)

    return convo