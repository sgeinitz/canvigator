import logging
import pandas as pd
import canvigator_quiz as cq
from canvigator_utils import today_str

logger = logging.getLogger(__name__)


class CanvigatorCourse:
    """A general class for a course and associated attributes/data."""

    def __init__(self, canvas, canvas_course, config, verbose=False):
        """Retrieve the selected course and get list of all students."""
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

        # use course_code prefix, course number, and CRN to create course_path
        tmp_course_code = str(self.canvas_course.course_code)
        course_path = tmp_course_code.split('-')[0] + tmp_course_code.split('-')[1] + "_" + tmp_course_code[-5:]
        course_path = "/" + course_path.lower()
        course_path = course_path.replace(" ", "")
        self.config.addCourseToPath(course_path)
        logger.info(f"Initialized course: {self.canvas_course.name}")

    def getAllQuizzesAndSubmissions(self):
        """Get all quizzes and their submissions for the course."""
        all_quizzes = self.canvas_course.get_quizzes()

        for i, q in enumerate(all_quizzes):
            # if q is a legit quiz, with at least one submission, then get submissions
            quiz = cq.CanvigatorQuiz(self.canvas, self, q, self.config, self.verbose)
            # check that the dataframe, quiz.quiz_df has at least 2 rows (header + at least one submission)
            if quiz.published and quiz.n_students is not None and quiz.n_students > 1:
                quiz.generateQuestionHistograms()
                quiz.getAllSubmissionsAndEvents()

    def createQuiz(self):
        """Create a new placeholder quiz with stub questions on Canvas."""
        title = input("Enter quiz title: ").strip()
        if not title:
            print("Quiz title cannot be empty.")
            return

        quiz = self.canvas_course.create_quiz({
            'title': title,
            'quiz_type': 'assignment',
            'time_limit': 30,
            'published': False,
            'allowed_attempts': 1,
            'scoring_policy': 'keep_highest',
            'one_question_at_a_time': True,
            'cant_go_back': True,
            'shuffle_answers': True
        })
        print(f"\nCreated quiz: '{quiz.title}' (id={quiz.id}, unpublished)")
        logger.info(f"Created quiz: '{quiz.title}' (id={quiz.id})")

        position = 1
        print("\nAdd placeholder questions (press Enter on an empty line to finish):")
        while True:
            description = input(f"  Q{position} description: ").strip()
            if not description:
                break
            quiz.create_question(question={
                'question_name': f"Question {position}",
                'question_text': description,
                'question_type': 'multiple_choice_question',
                'points_possible': 1,
                'position': position,
            })
            print(f"    Added Q{position}: {description}")
            logger.info(f"  Added question {position}: {description}")
            position += 1

        n_questions = position - 1
        print(f"\nQuiz '{title}' created with {n_questions} placeholder question{'s' if n_questions != 1 else ''}.")
        logger.info(f"Quiz '{title}' complete: {n_questions} questions")

    def saveStudentActivity(self, data_path):
        """Get student activity from two sources and save to csv files."""
        student_summary_data = self.canvas_course.get_course_level_student_summary_data()

        summ_acts_data = []
        for s in student_summary_data:
            summ_acts_data.append({
                'id': s.id,
                'page_views': s.page_views,
                'missing': s.tardiness_breakdown['missing'],
                'late': s.tardiness_breakdown['late']
            })
        summ_acts = pd.DataFrame(summ_acts_data, columns=['id', 'page_views', 'missing', 'late'])

        acts_data = []
        for s in self.students:
            acts_data.append({
                'name': s['name'],
                'id': s['id'],
                'total_activity_mins': s['total_activity_time'] / 60.0 if s['total_activity_time'] is not None else None,
                'last_activity_at': s['last_activity_at']
            })
        acts = pd.DataFrame(acts_data, columns=['name', 'id', 'total_activity_mins', 'last_activity_at'])

        # do an outer join of summ_acts and acts on 'id' column and save to csv file
        merged_acts = pd.merge(summ_acts, acts, on='id', how='outer')
        merged_acts = merged_acts[['name', 'id', 'page_views', 'missing', 'late', 'total_activity_mins', 'last_activity_at']]
        merged_acts_csv = data_path / f"course_activity_{today_str()}.csv"
        merged_acts.to_csv(merged_acts_csv, index=False)
        logger.info(f"Saved student activity to {merged_acts_csv}")
