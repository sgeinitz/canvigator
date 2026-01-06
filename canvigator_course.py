import pandas as pd
import canvigator_quiz as cq
from datetime import datetime


class CanvigatorCourse:
    """ A general class for a course and associated attributes/data. """

    def __init__(self, canvas, canvas_course, config, verbose=False):
        """ Retrieve the selected course and get list of all students. """
        self.canvas = canvas
        self.canvas_course = canvas_course
        self.config = config
        self.students = []
        self.verbose = verbose
        student = None
        for i, student in enumerate(self.canvas_course.get_enrollments()):
            # Use print(f"dir(student): {dir(student)}") to see all attributes
            if student.role == 'StudentEnrollment' and student.enrollment_state == 'active' and student.sis_user_id is not None:
                student.user['total_activity_time'] = None
                if hasattr(student, 'total_activity_time') and isinstance(student.total_activity_time, (int, float)):
                    student.user['total_activity_time'] = student.total_activity_time
                student.user['last_activity_at'] = None
                if hasattr(student, 'last_activity_at') and isinstance(student.last_activity_at, (int, float)):
                    student.user['last_activity_at'] = student.last_activity_at
                self.students.append(student.user)
        if verbose:
            print(self.students)

    def getAllQuizzesAndSubmissions(self):
        """ Get all quizzes and their submissions for the course. """
        all_quizzes = self.canvas_course.get_quizzes()

        for i, q in enumerate(all_quizzes):
            # if q is a legit quiz, with at least one submission, then get submissions
            quiz = cq.CanvigatorQuiz(self.canvas, self, q, self.config, self.verbose)
            # check that the dataframe, quiz.quiz_df has at least 2 rows (header + at least one submission)
            if quiz.published and quiz.n_students != None and quiz.n_students > 1:
                quiz.getAllSubmissionsAndEvents()

    def saveStudentActivity(self, data_path):
        """ Get student activity from two sources and save to csv files """
        student_summary_data = self.canvas_course.get_course_level_student_summary_data()
        summ_acts = pd.DataFrame(columns=['id', 'page_views', 'missing', 'late'])
        for s in student_summary_data:
            summ_acts.loc[len(summ_acts)] = [s.id, s.page_views, s.tardiness_breakdown['missing'], s.tardiness_breakdown['late']]
        summ_activity_csv = data_path + "course_activity_partA_" + datetime.today().strftime('%Y%m%d') + ".csv"
        summ_acts.to_csv(summ_activity_csv, index=False)

        acts = pd.DataFrame(columns=['name', 'id', 'total_activity_mins', 'last_activity_at'])
        for s in self.students:
            acts.loc[len(acts)] = [s['name'], s['id'], s['total_activity_time'] / 60.0, s['last_activity_at']]
        activity_csv = data_path + "course_activity_partB_" + datetime.today().strftime('%Y%m%d') + ".csv"
        acts.to_csv(activity_csv, index=False)

        # do an outer join of summ_acts and acts on 'id' column and save to csv file
        merged_acts = pd.merge(summ_acts, acts, on='id', how='outer')
        merged_acts = merged_acts[['name', 'id', 'page_views', 'missing', 'late', 'total_activity_mins', 'last_activity_at']]
        merged_acts_csv = data_path + "course_activity_both_" + datetime.today().strftime('%Y%m%d') + ".csv"
        merged_acts.to_csv(merged_acts_csv, index=False)