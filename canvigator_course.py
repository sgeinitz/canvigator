import hashlib
import logging
import shutil
from collections import Counter
from datetime import datetime, timedelta, timezone
import pandas as pd
import canvigator_quiz as cq
from canvigator_utils import today_str

MAX_COURSE_WEEKS = 16

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
        all_quizzes = list(self.canvas_course.get_quizzes())
        print(f"\nFound {len(all_quizzes)} quiz(zes).")

        for i, q in enumerate(all_quizzes, start=1):
            print(f"\n[{i}/{len(all_quizzes)}] Processing: {q.title}")
            # if q is a legit quiz, with at least one submission, then get submissions
            quiz = cq.CanvigatorQuiz(self.canvas, self, q, self.config, self.verbose)
            # check that the dataframe, quiz.quiz_df has at least 2 rows (header + at least one submission)
            if quiz.published and quiz.n_students is not None and quiz.n_students > 1:
                quiz.generateQuestionHistograms()
                quiz.getAllSubmissionsAndEvents()
                quiz.generateFirstAttemptHistograms()
            else:
                print("  Skipping (unpublished or insufficient submissions)")

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
                    'id': user_id,
                    'assignment_name': assignment.name,
                    'assignment_id': assignment.id,
                    'points_possible': assignment.points_possible,
                    'grade': sub.grade,
                    'score': sub.score,
                })

        gradebook_df = pd.DataFrame(gradebook_rows, columns=[
            'name', 'sortable_name', 'id', 'assignment_name',
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
                'page_views_level': getattr(s, 'page_views_level', None),
                'participations': s.participations,
                'participations_level': getattr(s, 'participations_level', None),
                'missing': s.tardiness_breakdown['missing'],
                'late': s.tardiness_breakdown['late'],
                'on_time': s.tardiness_breakdown.get('on_time'),
            })
        summ_cols = ['id', 'page_views', 'page_views_level', 'participations', 'participations_level',
                     'missing', 'late', 'on_time']
        summ_acts = pd.DataFrame(summ_acts_data, columns=summ_cols)

        acts_data = []
        for s in self.students:
            acts_data.append({
                'name': s['name'],
                'id': s['id'],
                'total_activity_mins': s['total_activity_time'] / 60.0 if s['total_activity_time'] is not None else None,
            })
        acts = pd.DataFrame(acts_data, columns=['name', 'id', 'total_activity_mins'])

        part_acts, week_cols = self._collectStudentParticipation()
        device_acts = self._collectStudentDeviceInfo()

        merged_acts = pd.merge(summ_acts, acts, on='id', how='outer')
        merged_acts = pd.merge(merged_acts, part_acts, on='id', how='outer')
        out_cols = [
            'name', 'id',
            'page_views', 'page_views_level',
            'participations', 'participations_level',
            'missing', 'late', 'on_time',
            'total_activity_mins',
            'first_activity_at', 'last_activity_at',
            'active_days_total', 'active_days_last_14',
            'views_last_7d', 'messages_to_instructor',
        ] + week_cols
        if device_acts is not None:
            merged_acts = pd.merge(merged_acts, device_acts, on='id', how='outer')
            out_cols += ['top_browser', 'top_os', 'used_mobile_app', 'n_distinct_user_agents']
        merged_acts = merged_acts[out_cols]
        merged_acts_csv = data_path / f"course_activity_{today_str()}.csv"
        merged_acts.to_csv(merged_acts_csv, index=False)
        print(f"Saved student activity to {merged_acts_csv.name}")
        logger.info(f"Saved student activity to {merged_acts_csv}")

    def _collectStudentParticipation(self, now_utc=None):
        """Per-student historical activity from Canvas course-level analytics.

        Calls two course-scoped analytics endpoints per student (both accessible
        to instructor tokens, unlike the admin-only /users/:id/page_views):

          - ``get_user_in_a_course_level_participation_data`` → hourly
            page-view buckets and a list of participation events.
          - ``get_user_in_a_course_level_messaging_data`` → per-day instructor/
            student message counts.

        Derives the following columns:
          - ``first_activity_at``, ``last_activity_at`` (UTC ISO strings, from
            actual page-view buckets — supersedes the enrollment field, which
            Canvas often leaves stale).
          - ``active_days_total`` — distinct calendar days with any activity.
          - ``active_days_last_14`` — distinct active days in the last 14 days
            (recency / disengagement signal).
          - ``views_last_7d`` — total page views in the last 7 days.
          - ``messages_to_instructor`` — sum of student-initiated messages.
          - ``views_wk_01`` … ``views_wk_NN`` — page views bucketed into 7-day
            windows from ``course.start_at``. Capped at ``MAX_COURSE_WEEKS``
            (16) for typical semester length. Omitted entirely if the Canvas
            course has no ``start_at`` set.

        Failures on individual students are logged and leave that student's
        derived columns as None; the rest of the roster is still processed.
        """
        now_utc = now_utc or datetime.now(timezone.utc)

        course_start_dt = _parse_canvas_timestamp(getattr(self.canvas_course, 'start_at', None))
        if course_start_dt is None:
            logger.info("Canvas course.start_at is not set — per-week views_wk_NN columns will be omitted.")
            print("  (course start date not set — skipping per-week columns)")
            n_weeks = 0
        else:
            days_elapsed = (now_utc - course_start_dt).days
            n_weeks = max(0, min(MAX_COURSE_WEEKS, days_elapsed // 7 + 1)) if days_elapsed >= 0 else 0

        week_cols = [f"views_wk_{i:02d}" for i in range(1, n_weeks + 1)]
        base_cols = ['id', 'first_activity_at', 'last_activity_at',
                     'active_days_total', 'active_days_last_14',
                     'views_last_7d', 'messages_to_instructor']
        cutoff_7d = now_utc - timedelta(days=7)
        cutoff_14d = now_utc - timedelta(days=14)

        rows = []
        print(f"Fetching per-student participation data for {len(self.students)} students...")
        for i, s in enumerate(self.students, start=1):
            sid = s['id']
            print(f"  [{i}/{len(self.students)}] {s['name']}")
            row = {c: None for c in base_cols + week_cols}
            row['id'] = sid

            try:
                part = self.canvas_course.get_user_in_a_course_level_participation_data(sid)
            except Exception as e:
                logger.warning(f"Could not fetch participation data for id={sid}: {e}")
                part = None

            if isinstance(part, dict):
                row.update(_summarize_participation(
                    part.get('page_views') or {},
                    course_start_dt, n_weeks, week_cols,
                    cutoff_7d, cutoff_14d,
                ))

            try:
                msgs = self.canvas_course.get_user_in_a_course_level_messaging_data(sid)
            except Exception as e:
                logger.warning(f"Could not fetch messaging data for id={sid}: {e}")
                msgs = None
            row['messages_to_instructor'] = _count_student_messages(msgs)

            rows.append(row)

        return pd.DataFrame(rows, columns=base_cols + week_cols), week_cols

    def _collectStudentDeviceInfo(self):
        """Aggregate browser/OS/mobile-app usage per student from Canvas page views.

        Calls ``user.get_page_views()`` for each enrolled student, filters to
        this course's context, and returns a DataFrame with columns
        ``id, top_browser, top_os, used_mobile_app, n_distinct_user_agents``.

        Permission model (important):
            The underlying endpoints — ``GET /api/v1/users/:id`` and
            ``GET /api/v1/users/:id/page_views`` — are gated by Canvas's
            "read user profile" / "view usage reports" permissions, which are
            typically granted only to **account admins**. Instructor-level
            tokens at most institutions will get a 404 (Canvas returns 404
            rather than 403 to avoid leaking user existence).

            To avoid emitting one warning per student on every run when the
            token is locked out, we do a single-student *permission probe*
            up front. If it fails, we log one info-level message explaining
            the likely cause and **return None** so the caller can omit the
            four device columns from the output CSV entirely (cleaner than
            showing a column of blanks to instructors who don't know about
            this Canvas permission). If the probe succeeds we proceed with
            the normal per-student loop and return a DataFrame.
        """
        if not self.students:
            return None

        # Permission probe — see docstring. A failure here (404 from Canvas)
        # almost always means the token lacks admin-level access to the
        # /users/:id endpoint, so the per-student page_views calls will fail
        # identically for every student. Short-circuit instead of looping.
        probe_sid = self.students[0]['id']
        try:
            self.canvas.get_user(probe_sid)
        except Exception as e:
            logger.info(
                "Skipping browser/device detection: cannot access /api/v1/users/:id "
                f"({e}). This usually means the Canvas token lacks the admin-level "
                "permission to read user profiles / view usage reports; instructor "
                "tokens are typically restricted from this endpoint. The four "
                "device columns will be omitted from the output CSV."
            )
            print("  (skipping browser/device columns — token lacks page-view permissions)")
            return None

        course_id = self.canvas_course.id
        rows = []
        print(f"Fetching page view device data for {len(self.students)} students...")
        for i, s in enumerate(self.students, start=1):
            sid = s['id']
            print(f"  [{i}/{len(self.students)}] {s['name']}")
            browsers = Counter()
            oses = Counter()
            user_agents = set()
            used_mobile_app = False
            try:
                user = self.canvas.get_user(sid)
                for pv in user.get_page_views():
                    if getattr(pv, 'context_type', None) != 'Course':
                        continue
                    if getattr(pv, 'context_id', None) != course_id:
                        continue
                    app = getattr(pv, 'app_name', None)
                    if app and 'Canvas for' in app:
                        used_mobile_app = True
                    ua = getattr(pv, 'user_agent', None)
                    if ua:
                        user_agents.add(ua)
                        browsers[_parse_browser(ua)] += 1
                        oses[_parse_os(ua)] += 1
            except Exception as e:
                logger.warning(f"Could not fetch page views for student id={sid}: {e}")
                rows.append({'id': sid, 'top_browser': None, 'top_os': None,
                             'used_mobile_app': None, 'n_distinct_user_agents': None})
                continue
            rows.append({
                'id': sid,
                'top_browser': browsers.most_common(1)[0][0] if browsers else None,
                'top_os': oses.most_common(1)[0][0] if oses else None,
                'used_mobile_app': used_mobile_app,
                'n_distinct_user_agents': len(user_agents) if user_agents else 0,
            })
        return pd.DataFrame(rows, columns=['id', 'top_browser', 'top_os',
                                           'used_mobile_app', 'n_distinct_user_agents'])


def _parse_canvas_timestamp(ts_str):
    """Parse a Canvas ISO 8601 timestamp into a UTC-aware datetime, or None on failure."""
    if not ts_str:
        return None
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _bucket_week(timestamp_dt, course_start_dt, max_weeks=MAX_COURSE_WEEKS):
    """Return the 1-indexed week number for a timestamp relative to course start.

    Week 1 spans [course_start, course_start + 7 days); Week 2 the next 7 days,
    and so on. Returns None for timestamps before the course start or beyond
    ``max_weeks``.
    """
    if timestamp_dt is None or course_start_dt is None:
        return None
    if timestamp_dt < course_start_dt:
        return None
    days_in = (timestamp_dt - course_start_dt).days
    wk = days_in // 7 + 1
    if wk > max_weeks:
        return None
    return wk


def _summarize_participation(page_views_dict, course_start_dt, n_weeks, week_cols, cutoff_7d, cutoff_14d):
    """Reduce Canvas hourly page-view buckets into summary + per-week fields."""
    first_ts = None
    last_ts = None
    active_days = set()
    active_days_recent = set()
    views_7d = 0
    week_buckets = {col: 0 for col in week_cols}

    for ts_str, count in page_views_dict.items():
        if not count:
            continue
        ts = _parse_canvas_timestamp(ts_str)
        if ts is None:
            continue
        if first_ts is None or ts < first_ts:
            first_ts = ts
        if last_ts is None or ts > last_ts:
            last_ts = ts
        day = ts.date()
        active_days.add(day)
        if ts >= cutoff_14d:
            active_days_recent.add(day)
        if ts >= cutoff_7d:
            views_7d += count
        wk = _bucket_week(ts, course_start_dt)
        if wk is not None and wk <= n_weeks:
            week_buckets[f"views_wk_{wk:02d}"] += count

    out = {
        'first_activity_at': first_ts.isoformat() if first_ts else None,
        'last_activity_at': last_ts.isoformat() if last_ts else None,
        'active_days_total': len(active_days),
        'active_days_last_14': len(active_days_recent),
        'views_last_7d': views_7d,
    }
    out.update(week_buckets)
    return out


def _count_student_messages(msgs):
    """Sum the studentMessages field across Canvas messaging-data records."""
    if msgs is None:
        return None
    total = 0
    # Canvas typically returns a list of {date, instructorMessages, studentMessages};
    # tolerate the dict-keyed-by-date variant some wrappers produce.
    if isinstance(msgs, list):
        records = msgs
    elif isinstance(msgs, dict):
        records = msgs.values()
    else:
        return None
    for m in records:
        if isinstance(m, dict):
            total += m.get('studentMessages') or 0
    return total


def _parse_browser(user_agent):
    """Extract a coarse browser label from a User-Agent string."""
    if not user_agent:
        return 'Unknown'
    if 'Edg' in user_agent:
        return 'Edge'
    if 'OPR/' in user_agent or 'Opera' in user_agent:
        return 'Opera'
    if 'Chrome' in user_agent:
        return 'Chrome'
    if 'Firefox' in user_agent:
        return 'Firefox'
    if 'Safari' in user_agent:
        return 'Safari'
    return 'Other'


def _parse_os(user_agent):
    """Extract a coarse operating-system label from a User-Agent string."""
    if not user_agent:
        return 'Unknown'
    if 'iPhone' in user_agent or 'iPad' in user_agent or 'iOS' in user_agent:
        return 'iOS'
    if 'Android' in user_agent:
        return 'Android'
    if 'Mac OS' in user_agent or 'Macintosh' in user_agent:
        return 'macOS'
    if 'Windows' in user_agent:
        return 'Windows'
    if 'Linux' in user_agent:
        return 'Linux'
    return 'Other'


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
        drop_cols = [c for c in ['name', 'id', 'sis_id', 'sortable_name'] if c in cols]
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
