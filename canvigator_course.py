import hashlib
import logging
import shutil
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

    def exportGradebook(self, data_path):
        """Export the current gradebook for all graded assignments to a CSV file.

        Fetches all published assignments and their submissions, then joins
        with student enrollment data to produce a gradebook export.
        """
        print("Fetching assignments...")
        assignments = list(self.canvas_course.get_assignments())
        published = [a for a in assignments if a.published]
        print(f"Found {len(published)} published assignments.")

        if not published:
            print("No published assignments found.")
            return

        # Build student lookup from enrolled students
        student_lookup = {}
        for s in self.students:
            student_lookup[s['id']] = {
                'name': s['name'],
                'sortable_name': s.get('sortable_name', ''),
            }

        gradebook_rows = []
        for i, assignment in enumerate(published):
            print(f"  [{i + 1}/{len(published)}] {assignment.name}")
            logger.info(f"Fetching submissions for assignment: {assignment.name} (id={assignment.id})")
            for sub in assignment.get_submissions():
                user_id = sub.user_id
                student = student_lookup.get(user_id)
                if student is None:
                    continue
                gradebook_rows.append({
                    'name': student['name'],
                    'sortable_name': student['sortable_name'],
                    'user_id': user_id,
                    'assignment_name': assignment.name,
                    'assignment_id': assignment.id,
                    'points_possible': assignment.points_possible,
                    'grade': sub.grade,
                    'score': sub.score,
                })

        gradebook_df = pd.DataFrame(gradebook_rows, columns=[
            'name', 'sortable_name', 'user_id', 'assignment_name',
            'assignment_id', 'points_possible', 'grade', 'score'
        ])

        csv_path = data_path / f"gradebook_{today_str()}.csv"
        gradebook_df.to_csv(csv_path, index=False)
        print(f"\nExported gradebook with {len(gradebook_df)} entries to {csv_path.name}")
        logger.info(f"Saved gradebook to {csv_path}")

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


def _make_anon_id(student_id):
    """Hash a student ID to a numeric identifier with at most 10 digits."""
    hash_hex = hashlib.sha256(str(student_id).encode()).hexdigest()
    return int(hash_hex, 16) % 10_000_000_000


def _addStudentId(id_to_info, sid, name):
    """Add a student ID and name to the id_to_info dict if not already present."""
    if pd.notna(sid):
        sid = int(float(sid))
        if sid not in id_to_info:
            id_to_info[sid] = {'name': name if pd.notna(name) else '', 'sis_id': ''}
    return sid


def _collectStudentIds(csv_files):
    """Scan CSV files and collect unique student IDs with associated name and sis_id."""
    id_to_info = {}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        cols = set(df.columns)

        # Standard files with name + id columns (student data)
        if 'name' in cols and 'id' in cols:
            for _, row in df.iterrows():
                sid = _addStudentId(id_to_info, row['id'], row['name'])
                if 'sis_id' in cols and pd.notna(row.get('sis_id')) and sid in id_to_info:
                    id_to_info[sid]['sis_id'] = row['sis_id']

        # Gradebook files with name + user_id columns
        if 'name' in cols and 'user_id' in cols:
            for _, row in df.iterrows():
                _addStudentId(id_to_info, row['user_id'], row['name'])

        # Pairing-style files with person1/id1, person2/id2, person3/id3
        for suffix in ['1', '2', '3']:
            id_col, name_col = f'id{suffix}', f'person{suffix}'
            if id_col in cols and name_col in cols:
                for _, row in df.iterrows():
                    if pd.notna(row[id_col]) and int(float(row[id_col])) != -1:
                        _addStudentId(id_to_info, row[id_col], row[name_col])
    return id_to_info


def _anonymizeCsvFile(csv_file, anon_dir, id_to_anon):
    """Create an anonymized copy of a single CSV file. Returns True if the file was modified."""
    def map_student_id(x):
        """Map a student ID to its anon_id, returning empty string for missing/invalid."""
        if pd.isna(x):
            return ''
        x_int = int(float(x))
        if x_int == -1:
            return ''
        return id_to_anon.get(x_int, '')

    df = pd.read_csv(csv_file)
    cols = set(df.columns)
    modified = False

    # Standard files: replace id with anon_id, drop name and sis_id
    if 'name' in cols and 'id' in cols:
        df['anon_id'] = df['id'].apply(map_student_id)
        drop_cols = [c for c in ['name', 'id', 'sis_id'] if c in cols]
        df = df.drop(columns=drop_cols)
        remaining = [c for c in df.columns if c != 'anon_id']
        df = df[['anon_id'] + remaining]
        modified = True

    # Gradebook files: replace user_id with anon_id, drop name and sortable_name
    if 'name' in cols and 'user_id' in cols:
        df['anon_id'] = df['user_id'].apply(map_student_id)
        drop_cols = [c for c in ['name', 'sortable_name', 'user_id'] if c in cols]
        df = df.drop(columns=drop_cols)
        remaining = [c for c in df.columns if c != 'anon_id']
        df = df[['anon_id'] + remaining]
        modified = True

    # Pairing-style files: person1/id1 -> anon_id1, etc.
    for suffix in ['1', '2', '3']:
        id_col, name_col, anon_col = f'id{suffix}', f'person{suffix}', f'anon_id{suffix}'
        if id_col in cols:
            df[anon_col] = df[id_col].apply(map_student_id)
            df = df.drop(columns=[id_col])
            cols = set(df.columns)
            modified = True
        if name_col in cols:
            df = df.drop(columns=[name_col])
            cols = set(df.columns)
            modified = True

    df.to_csv(anon_dir / csv_file.name, index=False)
    return modified


def exportAnonymizedData(data_path):
    """Export anonymized copies of all course data CSV files.

    Creates an anonymized/ subdirectory with all CSVs stripped of identifying
    columns (name, id, sis_id), replaced by a hashed anon_id. A mapping file
    is saved in the original data directory. The anonymized directory is also
    zipped for easy export.
    """
    from pathlib import Path
    data_path = Path(data_path)

    csv_files = sorted(f for f in data_path.glob("*.csv") if not f.name.startswith("anon_mapping_"))
    if not csv_files:
        print(f"No CSV files found in {data_path}")
        return

    anon_dir = data_path / "anonymized"
    anon_dir.mkdir(exist_ok=True)

    # Collect all unique student IDs from all CSV files
    id_to_info = _collectStudentIds(csv_files)
    if not id_to_info:
        print("No student data found in CSV files.")
        shutil.rmtree(anon_dir)
        return

    # Generate privacy-safe anon_ids (SHA-256 hash truncated to max 10 digits)
    id_to_anon = {sid: _make_anon_id(sid) for sid in id_to_info}

    # Save mapping file in original data directory
    mapping_rows = []
    for sid in sorted(id_to_info, key=lambda s: str(id_to_info[s]['name'])):
        mapping_rows.append({'name': id_to_info[sid]['name'], 'id': sid, 'sis_id': id_to_info[sid]['sis_id'], 'anon_id': id_to_anon[sid]})
    mapping_df = pd.DataFrame(mapping_rows, columns=['name', 'id', 'sis_id', 'anon_id'])
    mapping_file = data_path / f"anon_mapping_{today_str()}.csv"
    mapping_df.to_csv(mapping_file, index=False)
    print(f"Saved mapping: {mapping_file.name}")
    logger.info(f"Saved anonymization mapping to {mapping_file}")

    # Create anonymized copies of each CSV
    n_anonymized = sum(1 for f in csv_files if _anonymizeCsvFile(f, anon_dir, id_to_anon))

    # Zip the anonymized directory
    zip_path = data_path / f"anonymized_{today_str()}"
    shutil.make_archive(str(zip_path), 'zip', data_path, 'anonymized')
    print(f"Created: {zip_path.name}.zip")

    n_total = len(csv_files)
    print(f"Exported {n_total} file{'s' if n_total != 1 else ''} ({n_anonymized} anonymized) to {anon_dir.name}/")
    logger.info(f"Exported {n_total} anonymized files to {zip_path}.zip")
