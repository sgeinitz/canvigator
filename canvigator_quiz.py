import sys
import time
import random
import logging
import csv
import requests
import pandas as pd
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sbn
import os
import re
import json
from canvigator_utils import today_str, selectCSVFromList, prompt_for_index

logger = logging.getLogger(__name__)


def _parse_points_possible_from_student_analysis(csv_path):
    """Return {question_id: points_possible} parsed from a Canvas student_analysis header.

    Canvas places a numeric points_possible cell immediately after each question-text
    cell (e.g. "10746018: When training...", "1.5"). Pandas mangles duplicate numeric
    headers (".1"/".2" suffixes), so we read the raw header with the csv module.
    """
    with open(csv_path, newline='') as f:
        raw_header = next(csv.reader(f))

    points_possible = {}
    for i, cell in enumerate(raw_header):
        m = re.match(r'^\s*(\d+)\s*:', cell or '')
        if not m or i + 1 >= len(raw_header):
            continue
        try:
            points_possible[int(m.group(1))] = float(raw_header[i + 1])
        except (TypeError, ValueError):
            continue
    return points_possible


def _render_missed_bullets(missed_rows, question_info):
    """Render the 'questions you missed' bullet section for one student's most recent attempt.

    `missed_rows` is an iterable of dicts (or a list of DataFrame row dicts) each with
    keys `question_id`, `points`, `points_possible`. `question_info` maps question_id
    (int) to a dict with at least `position` and `keywords`. Returns None if there is
    nothing to render (empty input, or no rows map to known questions).
    """
    rendered = []
    for row in missed_rows:
        qid = row.get('question_id')
        try:
            qid_int = int(qid) if qid is not None else None
        except (TypeError, ValueError):
            qid_int = None
        info = question_info.get(qid_int) if qid_int is not None else None
        if info is None:
            logger.warning(f"No question_info for question_id={qid}; skipping in missed-questions bullet")
            continue
        rendered.append({
            'position': info.get('position', 0),
            'keywords': info.get('keywords') or '',
            'points': row.get('points', 0.0),
            'points_possible': row.get('points_possible', 0.0),
        })

    if not rendered:
        return None

    rendered.sort(key=lambda r: r['position'])
    header = "\n\nThe questions that you missed on this most recent attempt covered the concepts/topics:\n"
    lines = [
        f"• Q{r['position']}: {r['keywords']} ({r['points']:.2f} / {r['points_possible']:.2f} pts)"
        for r in rendered
    ]
    return header + "\n".join(lines)


def _render_blur_bullets(blurred_question_ids, question_info):
    """Render the 'questions with window focus change' bullet section.

    `blurred_question_ids` is a set (or iterable) of question_id ints that had
    at least one page_blurred event. `question_info` maps question_id (int) to
    a dict with at least `position` and `keywords`. Returns None if there is
    nothing to render.
    """
    rendered = []
    for qid in blurred_question_ids:
        info = question_info.get(qid)
        if info is None:
            logger.warning(f"No question_info for question_id={qid}; skipping in blur bullet")
            continue
        rendered.append({
            'position': info.get('position', 0),
            'keywords': info.get('keywords') or '',
        })

    if not rendered:
        return None

    rendered.sort(key=lambda r: r['position'])
    header = "\n\nThe questions that you changed window focus on covered the concepts/topics:\n"
    lines = [
        f"• Q{r['position']}: {r['keywords']}"
        for r in rendered
    ]
    return header + "\n".join(lines)


def _extract_question_id(event):
    """Extract quiz_question_id from a Canvas question_answered event, or None."""
    edata = getattr(event, 'event_data', None)
    if isinstance(edata, list) and edata:
        first_entry = edata[0]
        if isinstance(first_entry, dict):
            return first_entry.get('quiz_question_id')
    return None


def _collect_timing_and_blurs(student_events, timing_by_q, blurs_by_q):
    """Walk one student's first-attempt events and accumulate per-question timing and blur counts."""
    prev_time = None
    blur_count = 0

    for evt in student_events:
        if prev_time is None:
            prev_time = evt['timestamp']

        if evt['event'] == 'page_blurred':
            blur_count += 1

        if evt['event'] == 'question_answered':
            qid = evt.get('question_id')
            try:
                qid_str = str(int(float(qid))) if pd.notna(qid) else None
            except (TypeError, ValueError):
                qid_str = None
            if qid_str and qid_str in timing_by_q:
                delta_mins = (evt['timestamp'] - prev_time).total_seconds() / 60.0
                if 0 <= delta_mins < 10:
                    timing_by_q[qid_str].append(delta_mins)
                blurs_by_q[qid_str].append(blur_count)

            prev_time = evt['timestamp']
            blur_count = 0


class CanvigatorQuiz:
    """A class for one quiz and associated attributes/data."""

    def __init__(self, canvas, canvas_course, canvas_quiz, config, verbose=False, skip_student_data=False):
        """Initialize quiz object by getting all quiz data from Canvas."""
        self.canvas = canvas
        self.canvas_course = canvas_course
        self.canvas_quiz = canvas_quiz
        self.published = canvas_quiz.published
        self.verbose = verbose
        self.config = config
        self.quiz_name = self.canvas_quiz.title.lower().replace(" ", "_")
        # Remove underscore between 'quiz' and number so filenames read 'quizX_' not 'quiz_X_'
        self.quiz_name = re.sub(r'^quiz_(\d)', r'quiz\1', self.quiz_name)
        self.config.modifyQuizPrefix(self.quiz_name + "_")

        self.quiz_df = None
        self.n_students = None
        self.question_stats = None
        self.dist_matrix = None
        self.question_points_possible = {}
        self.quiz_questions = []  # Can later get text for kth question using quiz_question[k].question_text

        if self.verbose:
            print(f"Quiz title: {self.canvas_quiz.title}")

        for i, quest in enumerate(canvas_quiz.get_questions()):
            self.quiz_questions.append(quest)
            if verbose:
                print(f"Question {i}: {quest}")
        self.quiz_question_ids = [str(c.id) for c in self.quiz_questions]
        if self.published and not skip_student_data:
            self.getQuizData()

    def getQuizData(self):
        """Download student_analysis csv quiz report."""
        quiz_report_request = self.canvas_quiz.create_report('student_analysis')
        request_id = quiz_report_request.progress_url.split('/')[-1]
        if self.verbose:
            print("type(quiz_report_request) = ", type(quiz_report_request))
            print("quiz_report_request.__dict__ = ", quiz_report_request.__dict__)

        frame = 0
        quiz_report_progress = self.canvas.get_progress(request_id)
        while quiz_report_progress.workflow_state != 'completed':
            pct = int(quiz_report_progress.completion)
            self._spin(frame, f"Downloading {self.quiz_name} report... {pct}%")
            frame += 1
            time.sleep(0.1)
            quiz_report_progress = self.canvas.get_progress(request_id)

        quiz_report = self.canvas_quiz.get_quiz_report(quiz_report_request)
        quiz_csv_url = quiz_report.file['url']
        quiz_csv = requests.get(quiz_csv_url)
        csv_name = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_student_analysis_{today_str()}.csv"

        with open(csv_name, 'wb') as f:
            for content in quiz_csv.iter_content(chunk_size=2**20):
                if content:
                    f.write(content)

        self._spin_done(f"Saved: {csv_name.name}")
        logger.info(f"Quiz report downloaded: {self.quiz_name}")

        self.question_points_possible = _parse_points_possible_from_student_analysis(csv_name)

        self.quiz_df = pd.read_csv(csv_name)

        # rename columns to be shorter, cleaner, and with question_id as a column
        for old_col_name in self.quiz_df.columns:
            new_col_name = old_col_name.split(':')[0].replace(' ', '_').replace('.', '_')
            self.quiz_df.rename(columns={old_col_name: new_col_name}, inplace=True)

        for i, col in enumerate(self.quiz_df.columns):
            if col in self.quiz_question_ids:
                next_col = self.quiz_df.columns[i + 1]
                next_col_new_name = col + '_score'
                self.quiz_df.rename(columns={next_col: next_col_new_name}, inplace=True)
                self.quiz_df[next_col_new_name].apply(pd.to_numeric)

        self.n_students = self.quiz_df.shape[0]
        if self.verbose:
            print(f"Number of students who took quiz: {self.n_students}")
        if self.n_students == 0:
            return

        # dictionary w/ question id as key and summary stats for each question
        self.question_stats = dict()
        for q in self.quiz_question_ids:
            score_col = q + '_score'
            self.question_stats[q] = {
                'mean': self.quiz_df[score_col].mean(),
                'var': self.quiz_df[score_col].var(),
                'n_zeros': sum(self.quiz_df[score_col] == 0.0),
                'n_ones': sum(self.quiz_df[score_col] == 1.0),
            }

        if self.verbose:
            for key, val in self.question_stats.items():
                print("key =", key, "->", val)

    def getQuizQuestions(self, tag=False):
        """Save quiz metadata and questions to a CSV file."""
        quiz_id = self.canvas_quiz.id
        assignment_id = getattr(self.canvas_quiz, 'assignment_id', None)

        fields = ['position', 'question_name', 'question_type', 'question_text', 'points_possible']
        rows = []
        for q in self.quiz_questions:
            row = {'quiz_id': quiz_id, 'assignment_id': assignment_id, 'question_id': getattr(q, 'id', None)}
            row.update({f: getattr(q, f, None) for f in fields})
            row['answers'] = json.dumps(getattr(q, 'answers', []))
            rows.append(row)

        if tag:
            import canvigator_llm
            canvigator_llm.tag_questions(rows)

        columns = ['quiz_id', 'assignment_id', 'question_id', 'position', 'question_name', 'question_type']
        if tag:
            columns.append('keywords')
        columns += ['question_text', 'points_possible', 'answers']
        df = pd.DataFrame(rows, columns=columns)
        suffix = "questions_w_tags" if tag else "questions"
        csv_name = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_{suffix}_{today_str()}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Saved {len(rows)} questions to {csv_name}")
        logger.info(f"Quiz questions saved: {csv_name}")

    def generateOpenEndedQuestions(self):
        """Generate open-ended oral exam questions from the tagged questions CSV.

        Requires that get-quiz-questions --tag has been run first. Reads the
        tagged questions CSV, sends each question to the LLM, and writes the
        results to a new CSV for instructor review.
        """
        # Reuse _loadQuestionInfo's file-finding logic but we need the full CSV rows
        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"
        tagged_pattern = file_prefix + "questions_w_tags_"

        all_files = os.listdir(self.config.data_path)
        matching_dates = []
        for f in all_files:
            m = re.search(r'(\d{8})\.csv$', f)
            if m and tagged_pattern in f:
                matching_dates.append((m.group(1), f))

        if not matching_dates:
            raise FileNotFoundError(
                f"No *_questions_w_tags_*.csv found for quiz '{self.quiz_name}'. "
                f"Run 'python canvigator.py --crn <CRN> --tag get-quiz-questions' first."
            )

        matching_dates.sort(key=lambda t: t[0])
        latest_file = self.config.data_path / matching_dates[-1][1]
        print(f"  Using tagged questions from: {latest_file.name}")

        df = pd.read_csv(latest_file)
        if 'keywords' not in df.columns:
            raise RuntimeError(
                f"The file {latest_file.name} has no 'keywords' column. "
                f"Re-run 'get-quiz-questions --tag' first."
            )

        rows = df.to_dict('records')

        import canvigator_llm
        results = canvigator_llm.generate_open_ended_questions(rows)

        out_df = pd.DataFrame(results, columns=[
            'question_id', 'position', 'question_name', 'keywords',
            'question_mode', 'original_question_text', 'open_ended_question',
        ])
        csv_name = self.config.data_path / f"{file_prefix}open_ended_{today_str()}.csv"
        out_df.to_csv(csv_name, index=False)
        print(f"Saved {len(results)} open-ended questions to {csv_name}")
        logger.info(f"Open-ended questions saved: {csv_name}")

    def _buildMissedBulletsForStudent(self, student_id, subs_by_q_df, question_info):
        """Return the missed-questions bullet section for one student, or None to skip."""
        if subs_by_q_df is None or subs_by_q_df.empty:
            return None
        student_rows = subs_by_q_df[subs_by_q_df['id'] == student_id]
        if student_rows.empty:
            return None
        latest_attempt = student_rows['attempt'].max()
        latest_rows = student_rows[student_rows['attempt'] == latest_attempt]

        missed = []
        for row in latest_rows.to_dict('records'):
            qid = row.get('question_id')
            try:
                qid_int = int(qid) if qid is not None and not pd.isna(qid) else None
            except (TypeError, ValueError):
                qid_int = None
            info = question_info.get(qid_int) if qid_int is not None else None

            # Canvas submission_data doesn't always include points_possible; prefer the
            # canonical value from the questions CSV when available.
            pp_row = row.get('points_possible')
            pp_info = info.get('points_possible') if info else None
            if pp_info is not None:
                pp = pp_info
            elif pp_row is not None and not pd.isna(pp_row):
                pp = pp_row
            else:
                continue

            points = row.get('points')
            if points is None or pd.isna(points):
                continue

            try:
                if float(points) < float(pp):
                    row['points_possible'] = pp
                    missed.append(row)
            except (TypeError, ValueError):
                continue

        if not missed:
            return None
        return _render_missed_bullets(missed, question_info)

    def _buildBlurBulletsForStudent(self, student_id, events_df, question_info):
        """Return the blur-questions bullet section for one student, or None to skip.

        Walks the student's most recent attempt events chronologically and attributes
        each page_blurred event to the next question_answered event. Returns a rendered
        bullet section listing the questions that had at least one blur.
        """
        if events_df is None or events_df.empty:
            return None
        student_events = events_df[events_df['id'] == student_id]
        if student_events.empty:
            return None
        latest_attempt = student_events['attempt'].max()
        attempt_events = student_events[student_events['attempt'] == latest_attempt].sort_values('timestamp')

        blur_pending = False
        blurred_qids = set()
        for evt in attempt_events.to_dict('records'):
            if evt['event'] == 'page_blurred':
                blur_pending = True
            elif evt['event'] == 'question_answered' and blur_pending:
                qid = evt.get('question_id')
                try:
                    qid_int = int(qid) if qid is not None and not pd.isna(qid) else None
                except (TypeError, ValueError):
                    qid_int = None
                if qid_int is not None:
                    blurred_qids.add(qid_int)
                blur_pending = False

        if not blurred_qids:
            return None
        return _render_blur_bullets(blurred_qids, question_info)

    def _loadQuestionInfo(self):
        """Load the latest *_questions_w_tags_*.csv and build a question_id -> info map.

        Raises FileNotFoundError if no tagged questions CSV exists for the quiz,
        meaning `get-quiz-questions --tag` has not been run yet.
        """
        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"
        tagged_pattern = file_prefix + "questions_w_tags_"

        all_files = os.listdir(self.config.data_path)
        matching_dates = []
        for f in all_files:
            m = re.search(r'(\d{8})\.csv$', f)
            if m and tagged_pattern in f:
                matching_dates.append((m.group(1), f))

        if not matching_dates:
            raise FileNotFoundError(
                f"No *_questions_w_tags_*.csv found for quiz '{self.quiz_name}'. "
                f"Run 'python canvigator.py --crn <CRN> --tag get-quiz-questions' first."
            )

        matching_dates.sort(key=lambda t: t[0])
        latest_file = self.config.data_path / matching_dates[-1][1]
        print(f"  Using question data from: {latest_file.name}")
        dc_df = pd.read_csv(latest_file)

        if 'keywords' not in dc_df.columns:
            raise RuntimeError(
                f"The file {latest_file.name} has no 'keywords' column. "
                f"Re-run 'get-quiz-questions --tag' first."
            )

        question_info = {}
        for idx, (_, row) in enumerate(dc_df.iterrows(), start=1):
            qid = row.get('question_id')
            if pd.isna(qid):
                continue
            pp = row.get('points_possible')
            question_info[int(qid)] = {
                'position': int(row['position']) if pd.notna(row.get('position')) else idx,
                'keywords': row.get('keywords') if pd.notna(row.get('keywords')) else '',
                'question_name': row.get('question_name') if pd.notna(row.get('question_name')) else '',
                'points_possible': float(pp) if pd.notna(pp) else None,
            }
        return question_info

    def _buildQuizScores(self):
        """Build a dict mapping student_id -> score from the student_analysis report."""
        quiz_scores = {}
        if self.quiz_df is not None and self.n_students is not None and self.n_students > 0:
            for _, row in self.quiz_df.iterrows():
                student_id = row['id']
                score = row['score']
                if pd.notna(score):
                    quiz_scores[student_id] = score
        return quiz_scores

    def sendQuizReminders(self, dry_run=False):
        """Send reminder messages to students who haven't taken the quiz, haven't achieved a perfect score, or had page blur events."""
        quiz_name = self.canvas_quiz.title
        points_possible = self.canvas_quiz.points_possible

        # Prerequisite: a *_questions_w_tags_*.csv must exist for this quiz.
        # Fail fast with a clear instruction if not.
        question_info = self._loadQuestionInfo()

        # Refresh submission data — may be minutes old, so always re-fetch.
        print("\nFetching latest submissions for missed-question details...")
        self.getAllSubmissionsAndEvents()
        subs_by_q_df = self.all_subs_by_question
        events_df = self.all_subs_and_events

        quiz_scores = self._buildQuizScores()
        enrolled = self.canvas_course.students

        no_attempt_template = (
            "You have not yet attempted {quiz_name}. Be sure to make an attempt soon "
            "to help stay on top of the content in this course. And remember, the best "
            "way to use the quiz as a learning tool is to try to answer the questions "
            "without going to outside references or AI tools. Trying to answer on your "
            "own, even if it feels like a struggle, is the best way to help learn this "
            "material. Good luck! \n\nNOTE: This an auto-generated message, please let me know if you have any questions/concerns/suggestions about it."
        )

        imperfect_template = (
            "Nice work on attempting {quiz_name}, but you don't yet have a perfect "
            "score. Be sure to try it again soon to earn a perfect score. And remember, "
            "quizzes are most effective as learning tools when you try to answer the "
            "questions on your own without using any other resources. Learning happens best "
            "when it feels challenging to recall concepts and ideas, so embrace the struggle."
            " Good luck! \n\nNOTE: This an auto-generated message, please let me know if you have any questions/concerns/suggestions about it."
        )

        blur_template = (
            "Nice work on earning a perfect score on {quiz_name}! However, it looks "
            "like you may have left the quiz window at some point during your most recent "
            "attempt. Remember, quizzes are most effective as learning tools when you try "
            "to answer the questions on your own without using any other resources. Learning "
            "happens best when it feels challenging to recall concepts and ideas, so embrace "
            "the struggle. \n\nNOTE: This an auto-generated message, please let me know if you have any questions/concerns/suggestions about it."
        )

        subject_str = f"Quiz Reminder - {quiz_name}"
        messages = []

        for student in enrolled:
            student_id = student['id']
            student_name = student['name']
            first_name = student_name.split()[0]

            if student_id not in quiz_scores:
                reminder = no_attempt_template.format(quiz_name=quiz_name)
                reason = "no attempt"
            elif quiz_scores[student_id] < points_possible:
                score = quiz_scores[student_id]
                reminder = imperfect_template.format(quiz_name=quiz_name)
                reason = f"score {score}/{points_possible}"

                bullets = self._buildMissedBulletsForStudent(student_id, subs_by_q_df, question_info)
                if bullets:
                    reminder = reminder + bullets
            else:
                # Perfect score — check for page blur events
                blur_bullets = self._buildBlurBulletsForStudent(student_id, events_df, question_info)
                if not blur_bullets:
                    continue
                reminder = blur_template.format(quiz_name=quiz_name) + blur_bullets
                reason = "page blur"

            message_str = f"Hello {first_name}, {reminder}"
            messages.append((student_id, student_name, message_str, reason))

        self._sendOrPreviewMessages(messages, subject_str, quiz_name, points_possible, dry_run)

    def _sendOrPreviewMessages(self, messages, subject_str, quiz_name, points_possible, dry_run):
        """Send or preview the collected reminder messages and print a summary."""
        if not messages:
            print("No reminders to send — all students have perfect scores with no page blur events!")
            return

        if dry_run:
            print("\n=== DRY RUN MODE - No messages will be sent on Canvas ===\n")

        print(f"Quiz: {quiz_name} ({points_possible} points possible)")
        print(f"Reminders to send: {len(messages)}\n")

        for student_id, student_name, message_str, reason in messages:
            if dry_run:
                print(f"  [DRY RUN] To: {student_name} (id: {student_id}, {reason})")
                print(f"            Subject: {subject_str}")
                print(f"            Message: {message_str}\n")
            else:
                self.canvas.create_conversation(
                    [str(student_id)], message_str, subject=subject_str, force_new=True
                )
                print(f"  Sent to: {student_name} (id: {student_id}, {reason})")

        n_no_attempt = sum(1 for _, _, _, r in messages if r == "no attempt")
        n_blur = sum(1 for _, _, _, r in messages if r == "page blur")
        n_imperfect = len(messages) - n_no_attempt - n_blur
        action = "would be sent" if dry_run else "sent"
        summary_parts = []
        if n_no_attempt:
            summary_parts.append(f"{n_no_attempt} no-attempt")
        if n_imperfect:
            summary_parts.append(f"{n_imperfect} imperfect-score")
        if n_blur:
            summary_parts.append(f"{n_blur} page-blur")
        print(f"\n{len(messages)} reminder(s) {action} ({', '.join(summary_parts)}).")
        logger.info(f"Quiz reminders {'(dry run) ' if dry_run else ''}{action}: {len(messages)} for {quiz_name}")

    def sendFollowUpQuestions(self, dry_run=False):
        """Send the most-missed open-ended follow-up question to students who missed it.

        Requires that get-quiz-questions --tag and generate-open-ended-questions
        have both been run for this quiz. Auto-refreshes submission data to pick
        up recent attempts.
        """
        quiz_name = self.canvas_quiz.title

        # Load question metadata (from *_questions_w_tags_*.csv)
        question_info = self._loadQuestionInfo()

        # Load the open-ended questions CSV
        open_ended_rows = self._loadOpenEndedQuestions()

        # Refresh submission data for up-to-date per-question scores
        print("\nFetching latest submissions for follow-up question targeting...")
        self.getAllSubmissionsAndEvents()
        subs_by_q_df = self.all_subs_by_question

        if subs_by_q_df is None or subs_by_q_df.empty:
            print("No submission data available — cannot determine missed questions.")
            return

        # Find the most-missed question
        most_missed_qid = self._findMostMissedQuestion(subs_by_q_df, question_info)
        if most_missed_qid is None:
            print("All students scored perfectly on all questions — no follow-up needed!")
            return

        info = question_info[most_missed_qid]
        position = info['position']
        keywords = info['keywords']

        # Look up the open-ended question for this question_id
        oe_row = open_ended_rows.get(most_missed_qid)
        if oe_row is None:
            print(f"No open-ended question found for question_id {most_missed_qid} (Q{position}: {keywords}).")
            print("Re-run 'generate-open-ended-questions' to regenerate.")
            return

        question_mode = oe_row.get('question_mode', 'explain')
        open_ended_text = oe_row['open_ended_question']

        print(f"\nMost-missed question: Q{position} — {keywords}")
        print(f"  Mode: {question_mode}")
        print(f"  Follow-up: {open_ended_text[:120]}{'...' if len(open_ended_text) > 120 else ''}")

        # Build the list of students who missed this question on their latest attempt
        students_who_missed = self._findStudentsWhoMissed(most_missed_qid, subs_by_q_df, question_info)

        if not students_who_missed:
            print("No students missed this question on their latest attempt — no follow-up needed!")
            return

        # Compose messages
        if question_mode == 'draw':
            instructions = (
                "Please draw your answer on paper (or a tablet) and reply to this "
                "message with a photo of your drawing attached using the attachment "
                "button (the paperclip icon)."
            )
        else:
            instructions = (
                "Please reply to this message with a voice recording explaining "
                "your answer. You can use the microphone button in the Canvas "
                "message editor to record your response."
            )

        subject_str = f"Follow-Up Question - {quiz_name} - Q{position}"
        enrolled = self.canvas_course.students
        enrolled_map = {s['id']: s['name'] for s in enrolled}

        messages = []
        for student_id in students_who_missed:
            student_name = enrolled_map.get(student_id)
            if student_name is None:
                continue
            first_name = student_name.split()[0]
            message_str = (
                f"Hello {first_name}, based on your recent attempt on {quiz_name}, "
                f"here is a follow-up question to help reinforce your understanding "
                f"of the topic ({keywords}):\n\n"
                f"{open_ended_text}\n\n"
                f"{instructions}\n\n"
                f"You can reply as many times as you'd like — only your most recent "
                f"response will be assessed.\n\n"
                f"NOTE: This is an auto-generated message, please let me know if you "
                f"have any questions/concerns/suggestions about it."
            )
            messages.append((student_id, student_name, message_str, f"missed Q{position}"))

        self._sendOrPreviewMessages(messages, subject_str, quiz_name, self.canvas_quiz.points_possible, dry_run)

        # Save the follow-up manifest CSV
        self._saveFollowUpManifest(messages, most_missed_qid, question_mode, subject_str, dry_run)

    def _findMostMissedQuestion(self, subs_by_q_df, question_info):
        """Return the question_id with the highest miss rate, or None if all are perfect.

        Miss rate = fraction of students whose latest attempt scored below
        points_possible for that question.
        """
        # Use only the latest attempt per student (each student has multiple rows — one per question)
        max_attempt_per_student = subs_by_q_df.groupby('id')['attempt'].transform('max')
        latest_attempts = subs_by_q_df[subs_by_q_df['attempt'] == max_attempt_per_student]

        miss_rates = {}
        for qid, info in question_info.items():
            pp = info.get('points_possible')
            if pp is None:
                continue
            q_rows = latest_attempts[latest_attempts['question_id'] == qid]
            if q_rows.empty:
                continue
            n_total = len(q_rows)
            n_missed = sum(1 for _, row in q_rows.iterrows()
                           if pd.notna(row.get('points')) and float(row['points']) < float(pp))
            miss_rates[qid] = n_missed / n_total if n_total > 0 else 0.0

        if not miss_rates or max(miss_rates.values()) == 0:
            return None

        # Report top-5 miss rates for instructor visibility
        sorted_rates = sorted(miss_rates.items(), key=lambda t: t[1], reverse=True)
        print("\nQuestion miss rates (latest attempt):")
        for qid, rate in sorted_rates[:5]:
            info = question_info.get(qid, {})
            label = f"Q{info.get('position', '?')} ({info.get('keywords', '')})"
            print(f"  {label}: {rate:.0%}")

        return sorted_rates[0][0]

    def _findStudentsWhoMissed(self, question_id, subs_by_q_df, question_info):
        """Return a list of student IDs who scored below points_possible on the given question in their latest attempt."""
        pp = question_info.get(question_id, {}).get('points_possible')
        if pp is None:
            return []

        max_attempt_per_student = subs_by_q_df.groupby('id')['attempt'].transform('max')
        latest_attempts = subs_by_q_df[subs_by_q_df['attempt'] == max_attempt_per_student]
        q_rows = latest_attempts[latest_attempts['question_id'] == question_id]

        missed_ids = []
        for _, row in q_rows.iterrows():
            points = row.get('points')
            if points is not None and pd.notna(points) and float(points) < float(pp):
                missed_ids.append(row['id'])
        return missed_ids

    def _loadOpenEndedQuestions(self):
        """Load the latest *_open_ended_*.csv and return a dict keyed by question_id.

        Raises FileNotFoundError if no open-ended CSV exists for the quiz.
        """
        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"
        oe_pattern = file_prefix + "open_ended_"

        all_files = os.listdir(self.config.data_path)
        matching_dates = []
        for f in all_files:
            m = re.search(r'(\d{8})\.csv$', f)
            if m and oe_pattern in f:
                matching_dates.append((m.group(1), f))

        if not matching_dates:
            raise FileNotFoundError(
                f"No *_open_ended_*.csv found for quiz '{self.quiz_name}'. "
                f"Run 'python canvigator.py generate-open-ended-questions' first."
            )

        matching_dates.sort(key=lambda t: t[0])
        latest_file = self.config.data_path / matching_dates[-1][1]
        print(f"  Using open-ended questions from: {latest_file.name}")

        df = pd.read_csv(latest_file)
        if 'open_ended_question' not in df.columns:
            raise RuntimeError(
                f"The file {latest_file.name} has no 'open_ended_question' column. "
                f"Re-run 'generate-open-ended-questions'."
            )

        rows_by_qid = {}
        for _, row in df.iterrows():
            qid = row.get('question_id')
            if pd.notna(qid):
                rows_by_qid[int(qid)] = row.to_dict()
        return rows_by_qid

    def _saveFollowUpManifest(self, messages, question_id, question_mode, subject_str, dry_run):
        """Save a CSV manifest of follow-up messages sent (or previewed in dry-run)."""
        from datetime import datetime
        sent_at = datetime.utcnow().isoformat() + 'Z'
        manifest_rows = []
        for student_id, student_name, _, reason in messages:
            manifest_rows.append({
                'student_id': student_id,
                'student_name': student_name,
                'question_id': question_id,
                'question_mode': question_mode,
                'conversation_subject': subject_str,
                'sent_at': sent_at,
            })

        manifest_df = pd.DataFrame(manifest_rows)
        suffix = "_dryrun" if dry_run else ""
        csv_name = self.config.data_path / (
            f"{self.config.quiz_prefix}{self.canvas_quiz.id}_followup_sent{suffix}_{today_str()}.csv"
        )
        manifest_df.to_csv(csv_name, index=False)
        print(f"  Manifest saved: {csv_name.name}")
        logger.info(f"Follow-up manifest saved: {csv_name}")

    def getFollowUpReplies(self, reply_window_days=5):
        """Retrieve student replies to follow-up questions from Canvas conversations.

        Loads the followup_sent manifest CSV to know which conversations to look
        for, then uses the Canvas conversations API to find matching threads and
        extract student replies (text, audio, image attachments). Downloads media
        to a replies/ subdirectory. Outputs a followup_replies CSV.
        """
        from datetime import datetime, timedelta, timezone

        manifest = self._loadFollowUpManifest()
        subject_str = manifest['conversation_subject'].iloc[0]
        question_id = int(manifest['question_id'].iloc[0])
        question_mode = manifest['question_mode'].iloc[0]

        # Parse the sent_at timestamp to compute the reply window cutoff
        sent_at_str = manifest['sent_at'].iloc[0]
        sent_at = datetime.fromisoformat(sent_at_str.replace('Z', '+00:00'))
        cutoff = sent_at + timedelta(days=reply_window_days)
        now = datetime.now(timezone.utc)

        print(f"\nLooking for replies to: {subject_str}")
        print(f"  Reply window: {reply_window_days} days (cutoff: {cutoff.strftime('%Y-%m-%d %H:%M UTC')})")
        if now < cutoff:
            remaining = cutoff - now
            print(f"  Window still open — {remaining.days}d {remaining.seconds // 3600}h remaining")

        # Build a set of student IDs we expect replies from
        expected_students = set(manifest['student_id'].astype(int))
        student_names = dict(zip(manifest['student_id'].astype(int), manifest['student_name']))

        # Get the instructor's user ID to filter out instructor messages
        instructor_id = self.canvas.get_current_user().id

        # Search sent conversations for threads matching our subject
        print("\n  Scanning sent conversations...")
        matching_convos = self._findConversationsBySubject(subject_str)
        print(f"  Found {len(matching_convos)} matching conversation(s)")

        # Ensure the replies directory exists
        replies_dir = self.config.data_path / "replies"
        replies_dir.mkdir(exist_ok=True)

        # Extract replies from each matching conversation
        all_replies = []
        n_with_reply = 0
        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"

        for convo_summary in matching_convos:
            # Get full conversation with messages (don't mark as read)
            convo = self.canvas.get_conversation(convo_summary.id, auto_mark_as_read=False)
            messages = getattr(convo, 'messages', [])
            audience = set(getattr(convo, 'audience', []))

            # Determine which student this conversation belongs to
            student_ids_in_convo = audience & expected_students
            if not student_ids_in_convo:
                continue
            student_id = next(iter(student_ids_in_convo))
            student_name = student_names.get(student_id, f"Unknown ({student_id})")

            # Collect student replies (messages not from the instructor)
            student_replies = self._extractStudentReplies(
                messages, instructor_id, sent_at, cutoff
            )

            if student_replies:
                n_with_reply += 1

            for i, msg in enumerate(student_replies):
                is_latest = (i == 0)  # messages are newest-first from Canvas
                attachment_path, audio_path = self._downloadReplyMedia(
                    msg, student_id, question_id, file_prefix, replies_dir
                )
                replied_at = msg.get('created_at', '')

                all_replies.append({
                    'student_id': student_id,
                    'student_name': student_name,
                    'question_id': question_id,
                    'question_mode': question_mode,
                    'message_id': msg.get('id', ''),
                    'reply_text': msg.get('body', ''),
                    'has_attachment': bool(attachment_path),
                    'attachment_path': attachment_path or '',
                    'has_audio': bool(audio_path),
                    'audio_path': audio_path or '',
                    'replied_at': replied_at,
                    'latest': is_latest,
                })

        # Save the replies CSV
        replies_df = pd.DataFrame(all_replies)
        csv_name = self.config.data_path / f"{file_prefix}followup_replies_{today_str()}.csv"
        replies_df.to_csv(csv_name, index=False)

        # Print summary
        n_expected = len(expected_students)
        print(f"\n  Students who received follow-up: {n_expected}")
        print(f"  Students who replied: {n_with_reply}")
        print(f"  Students with no reply: {n_expected - n_with_reply}")
        print(f"  Total reply messages: {len(all_replies)}")
        n_attachments = sum(1 for r in all_replies if r['has_attachment'])
        n_audio = sum(1 for r in all_replies if r['has_audio'])
        if n_attachments:
            print(f"  Image attachments downloaded: {n_attachments}")
        if n_audio:
            print(f"  Audio recordings downloaded: {n_audio}")
        print(f"  Replies saved: {csv_name.name}")
        logger.info(f"Follow-up replies saved: {csv_name}")

    def _loadFollowUpManifest(self):
        """Load the latest *_followup_sent_*.csv manifest (non-dryrun only).

        Raises FileNotFoundError if no manifest exists for the quiz.
        """
        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"
        manifest_pattern = file_prefix + "followup_sent_"

        all_files = os.listdir(self.config.data_path)
        matching_dates = []
        for f in all_files:
            # Skip dryrun manifests
            if '_dryrun_' in f:
                continue
            m = re.search(r'(\d{8})\.csv$', f)
            if m and manifest_pattern in f:
                matching_dates.append((m.group(1), f))

        if not matching_dates:
            raise FileNotFoundError(
                f"No *_followup_sent_*.csv found for quiz '{self.quiz_name}'. "
                f"Run 'send-follow-up-question' first."
            )

        matching_dates.sort(key=lambda t: t[0])
        latest_file = self.config.data_path / matching_dates[-1][1]
        print(f"  Using follow-up manifest: {latest_file.name}")

        df = pd.read_csv(latest_file)
        required_cols = {'student_id', 'student_name', 'question_id', 'question_mode', 'conversation_subject', 'sent_at'}
        missing = required_cols - set(df.columns)
        if missing:
            raise RuntimeError(f"Manifest {latest_file.name} is missing columns: {missing}")
        return df

    def _findConversationsBySubject(self, subject_str):
        """Return a list of Conversation objects from sent conversations matching the given subject."""
        matching = []
        for convo in self.canvas.get_conversations(scope='sent'):
            if getattr(convo, 'subject', '') == subject_str:
                matching.append(convo)
        return matching

    def _extractStudentReplies(self, messages, instructor_id, sent_at, cutoff):
        """Filter conversation messages to only student replies within the reply window.

        Returns a list of message dicts, newest first (Canvas's default order).
        Messages from the instructor or outside the window are excluded.
        """
        from datetime import datetime

        replies = []
        for msg in messages:
            # Skip instructor's own messages
            if msg.get('author_id') == instructor_id:
                continue
            # Skip system-generated messages
            if msg.get('generated', False):
                continue
            # Check reply window
            created_str = msg.get('created_at', '')
            if created_str:
                try:
                    created = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                    if created < sent_at or created > cutoff:
                        logger.info(f"Skipping message {msg.get('id')} — outside reply window "
                                    f"(created: {created_str}, window: {sent_at} to {cutoff})")
                        continue
                except (ValueError, TypeError):
                    pass  # If we can't parse the timestamp, include the message
            replies.append(msg)
        return replies

    def _downloadReplyMedia(self, msg, student_id, question_id, file_prefix, replies_dir):
        """Download any attachment or audio from a reply message.

        Returns (attachment_path, audio_path) — each is a string path relative
        to the data directory, or None if no media of that type.
        """
        attachment_path = None
        audio_path = None

        # Download image/file attachments
        attachments = msg.get('attachments', [])
        if attachments:
            att = attachments[0]  # Take the first attachment
            att_url = att.get('url', '')
            filename = att.get('filename', '') or att.get('display_name', 'attachment')
            ext = os.path.splitext(filename)[1] or '.bin'
            local_name = f"{file_prefix}{student_id}_{question_id}{ext}"
            local_path = replies_dir / local_name
            try:
                resp = requests.get(att_url, timeout=30)
                resp.raise_for_status()
                with open(local_path, 'wb') as f:
                    f.write(resp.content)
                attachment_path = str(local_path.relative_to(self.config.data_path.parent))
                logger.info(f"Downloaded attachment for student {student_id}: {local_name}")
            except Exception as e:
                logger.warning(f"Failed to download attachment for student {student_id}: {e}")

        # Download audio/video media comment
        media_comment = msg.get('media_comment')
        if media_comment:
            media_url = media_comment.get('url', '')
            media_type = media_comment.get('media_type', 'audio')
            ext = '.m4a' if media_type == 'audio' else '.mp4'
            local_name = f"{file_prefix}{student_id}_{question_id}{ext}"
            local_path = replies_dir / local_name
            try:
                resp = requests.get(media_url, timeout=60)
                resp.raise_for_status()
                with open(local_path, 'wb') as f:
                    f.write(resp.content)
                audio_path = str(local_path.relative_to(self.config.data_path.parent))
                logger.info(f"Downloaded media for student {student_id}: {local_name}")
            except Exception as e:
                logger.warning(f"Failed to download media for student {student_id}: {e}")

        return attachment_path, audio_path

    def assessFollowUpReplies(self):
        """Assess student replies using the local LLM.

        Loads the latest followup_replies CSV, filters to only the latest reply
        per student, loads the open-ended question context, and calls the LLM
        to produce a pass/fail assessment with feedback. Outputs a
        followup_assessments CSV.
        """
        # Load replies CSV
        replies_df = self._loadFollowUpReplies()
        # Filter to latest reply per student only
        latest_replies = replies_df[replies_df['latest'] == True].to_dict('records')  # noqa: E712

        if not latest_replies:
            print("No student replies to assess.")
            return

        # Load the open-ended questions to get context for assessment
        open_ended_rows = self._loadOpenEndedQuestions()

        # All replies should share the same question_id (Phase 1 sends one question)
        question_id = int(latest_replies[0]['question_id'])
        oe_row = open_ended_rows.get(question_id)
        if oe_row is None:
            print(f"No open-ended question found for question_id {question_id}.")
            return

        print(f"\nAssessing {len(latest_replies)} student replies for question_id {question_id}")
        print(f"  Question: {oe_row.get('open_ended_question', '')[:100]}...")
        print(f"  Mode: {latest_replies[0].get('question_mode', 'explain')}")

        import canvigator_llm
        results = canvigator_llm.assess_replies(latest_replies, oe_row)

        # Save assessments CSV
        out_df = pd.DataFrame(results, columns=[
            'student_id', 'student_name', 'question_id', 'question_mode',
            'result', 'feedback', 'transcript', 'assessed_at',
        ])
        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"
        csv_name = self.config.data_path / f"{file_prefix}followup_assessments_{today_str()}.csv"
        out_df.to_csv(csv_name, index=False)

        n_pass = sum(1 for r in results if r['result'] == 'pass')
        n_fail = sum(1 for r in results if r['result'] == 'fail')
        print(f"\n  Results: {n_pass} pass, {n_fail} fail")
        print(f"  Assessments saved: {csv_name.name}")
        logger.info(f"Follow-up assessments saved: {csv_name}")

    def _loadFollowUpReplies(self):
        """Load the latest *_followup_replies_*.csv.

        Raises FileNotFoundError if no replies CSV exists for the quiz.
        """
        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"
        replies_pattern = file_prefix + "followup_replies_"

        all_files = os.listdir(self.config.data_path)
        matching_dates = []
        for f in all_files:
            m = re.search(r'(\d{8})\.csv$', f)
            if m and replies_pattern in f:
                matching_dates.append((m.group(1), f))

        if not matching_dates:
            raise FileNotFoundError(
                f"No *_followup_replies_*.csv found for quiz '{self.quiz_name}'. "
                f"Run 'get-replies' first."
            )

        matching_dates.sort(key=lambda t: t[0])
        latest_file = self.config.data_path / matching_dates[-1][1]
        print(f"  Using replies from: {latest_file.name}")
        return pd.read_csv(latest_file)

    SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def _spin(self, frame, message, indent=2):
        """Write a single spinner frame with a message to stdout, overwriting the current line."""
        pad = ' ' * indent
        sys.stdout.write(f"\r{pad}{self.SPINNER_FRAMES[frame % len(self.SPINNER_FRAMES)]} {message}  ")
        sys.stdout.flush()

    def _spin_done(self, message, indent=2):
        """Clear the spinner line and write a completion message."""
        pad = ' ' * indent
        sys.stdout.write(f"\r{pad}✓ {message}                              \n")
        sys.stdout.flush()

    def figurePath(self, figure_name):
        """Return a figure output path with the date suffix at the end."""
        return self.config.figures_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_{figure_name}_{today_str()}.png"

    def generateQuestionHistograms(self):
        """Draw a histogram of scores of each question."""
        mpl.style.use('seaborn-v0_8')
        n_questions = len(self.quiz_question_ids)
        figure, axis = plt.subplots(1, n_questions, sharey=True)
        if n_questions == 1:
            axis = [axis]
        figure.set_size_inches(13, 3)
        for i, q in enumerate(self.quiz_question_ids):
            score_col = q + '_score'
            col_data = self.quiz_df[score_col]
            axis[i].hist(col_data, bins=6, facecolor='#00447c', edgecolor='black', alpha=0.8)
            axis[i].axvline(col_data.mean(), color='black', linestyle='dashed', linewidth=1)
            axis[i].set_xlabel('score')
            axis[i].set_title('question: ' + q.split('_')[0])
        axis[0].set_ylabel('# of people')
        plt.tight_layout()  # Or try plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.9)
        fig_path = self.figurePath("histograms")
        figure.savefig(fig_path, dpi=200)
        plt.close('all')
        print(f"  ✓ Saved: {fig_path.name}")

    def _plotPerQuestionHistogram(self, data_by_q, xlabel, facecolor, figure_name):
        """Plot a row of per-question histograms and save the figure."""
        question_ids = self.quiz_question_ids
        n_questions = len(question_ids)
        mpl.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(1, n_questions, sharey=True)
        if n_questions == 1:
            axes = [axes]
        fig.set_size_inches(13, 3)
        for i, qid in enumerate(question_ids):
            data = data_by_q[qid]
            if data:
                bins = range(0, max(data) + 2) if all(isinstance(v, int) for v in data) else 10
                axes[i].hist(data, bins=bins, facecolor=facecolor, edgecolor='black', alpha=0.8)
                axes[i].axvline(sum(data) / len(data), color='black', linestyle='dashed', linewidth=1)
            axes[i].set_xlabel(xlabel)
            axes[i].set_title(f'Q{i + 1}')
        axes[0].set_ylabel('# of students')
        plt.tight_layout()
        fig_path = self.figurePath(figure_name)
        fig.savefig(fig_path, dpi=200)
        plt.close('all')
        print(f"  ✓ Saved: {fig_path.name}")

    def generateFirstAttemptHistograms(self):
        """Generate per-question timing and page-blur histograms for first attempts only.

        Requires getAllSubmissionsAndEvents() to have been run first. Uses
        question_answered events to compute per-question time deltas and counts
        page_blurred events between consecutive answers.
        """
        events_df = getattr(self, 'all_subs_and_events', None)
        if events_df is None or events_df.empty:
            return

        first = events_df[events_df['attempt'] == 1].copy()
        if first.empty:
            return

        first['timestamp'] = pd.to_datetime(first['timestamp'])

        question_ids = self.quiz_question_ids
        if not question_ids:
            return

        timing_by_q = {qid: [] for qid in question_ids}
        blurs_by_q = {qid: [] for qid in question_ids}

        for sid in first['id'].unique():
            student_events = first[first['id'] == sid].sort_values('timestamp').to_dict('records')
            _collect_timing_and_blurs(student_events, timing_by_q, blurs_by_q)

        has_data = any(timing_by_q[qid] or blurs_by_q[qid] for qid in question_ids)
        if not has_data:
            print("  (No per-question event data available for timing/blur histograms)")
            return

        self._plotPerQuestionHistogram(timing_by_q, 'minutes', '#00447c', 'timing_first_attempt')
        self._plotPerQuestionHistogram(blurs_by_q, '# of page blurs', '#c44e52', 'blurs_first_attempt')

    def generateDistanceMatrix(self, only_present, distance_type='euclid'):
        """Calculate vector distance between all possible student pairs."""
        if only_present and getattr(self, 'df_quiz_scores_present', None) is None:
            raise RuntimeError("openPresentCSV() must be called before generateDistanceMatrix(only_present=True)")

        quiz_df_local = self.df_quiz_scores_present.copy() if only_present else self.quiz_df.copy()
        student_ids = list(quiz_df_local['id'])
        student_ids.sort()
        self.dist_matrix = pd.DataFrame(0.0, columns=student_ids, index=student_ids)

        for i, id1 in enumerate(student_ids):
            x = quiz_df_local.loc[quiz_df_local.id == id1, quiz_df_local.columns.str.endswith('_score')].to_numpy().flatten()
            x = [0.0 if pd.isna(val) else val for val in x]
            if self.verbose:
                print(id1, "values =", x)
            for j, id2 in enumerate(student_ids):
                if i < j:
                    y = quiz_df_local.loc[quiz_df_local.id == id2, quiz_df_local.columns.str.endswith('_score')].to_numpy().flatten()
                    y = [0.0 if pd.isna(val) else val for val in y]
                    if self.verbose:
                        print(id2, "    values =", y)
                    if distance_type == 'euclid':
                        dist = distance.euclidean(x, y)
                    elif distance_type == 'cosine':
                        dist = distance.cosine(x, y)
                    if dist == 0:
                        dist = 1E-4
                    self.dist_matrix.loc[id1, id2] = dist
                    self.dist_matrix.loc[id2, id1] = dist

        mpl.style.use('seaborn-v0_8')
        plt.figure(figsize=(16, 16))
        sbn.heatmap(
            self.dist_matrix,
            square=True,
            cmap="YlGnBu",
            linewidth=0.5,
            annot=True,
            cbar=False
        )
        plt.tight_layout()
        plt.rc('font', size=9)
        fig_path = self.figurePath(f"dist_{distance_type}")
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"Saved distance matrix heatmap to {fig_path.name}")

    def openPresentCSV(self, csv_path=None):
        """Prompt user for a local CSV file and return a pandas dataframe."""
        if not csv_path:
            csv_path = self.config.data_path

        selected = selectCSVFromList(
            csv_path, 'present',
            "\nSelect csv of students present from above using index: ",
            verbose=self.verbose
        )

        # Open the file and remove those that are not present today, then return this dataframe.
        df_present_all = pd.read_csv(selected)
        self.df_present = df_present_all[df_present_all['present'] == 1]
        print(f"  *** (double check there are {len(self.df_present)} students present today) ***")

        self.df_quiz_scores_present = pd.merge(self.df_present[['name', 'id']], self.quiz_df, how='left')
        # replace missing vals with zero (for people who missed pre-quiz)
        with pd.option_context("future.no_silent_downcasting", True):
            self.df_quiz_scores_present = self.df_quiz_scores_present.fillna(0).infer_objects(copy=False)
        if self.verbose:
            print(f"self.df_quiz_scores_present.columns = {self.df_quiz_scores_present.columns}")
        assert len(self.df_quiz_scores_present) == len(self.df_present)

    def createStudentPairings(self, method='med', write_csv=True):
        """Generate student pairings using one of several methods, but not saved unless write_csv is True."""
        if self.dist_matrix is None:
            raise RuntimeError("generateDistanceMatrix() must be called before createStudentPairings()")

        dm = self.dist_matrix.copy()
        pairings = []

        while dm.shape[0] > 2:

            # retrieve max entry in each column/row and corresponding index
            # (note that indices and columns are canvas student ids)
            col_maximums = dm.max()
            col_max_indices = dm.idxmax()

            # 'max' is a greedy approach taking largest pair difference first
            if method == 'max':
                person_A = col_maximums.idxmax()
            # 'med' is median difference and generally leads to highest mean diff and lowest variance
            elif method == 'med':
                col_maximums.sort_values(inplace=True)
                person_A = col_maximums.index[len(col_maximums) // 2]
            # 'min' uses conservative approach by taking min pair difference (among maxes) but often leads to high var
            elif method == 'min':
                person_A = col_maximums.idxmin()
            elif method == 'rand':
                people = list(dm.index)
                random.shuffle(people)
                person_A = people[0]
                person_B = people[1]
                pairings.append((person_A, person_B, dm.loc[person_A, person_B]))
            else:
                raise ValueError("vectorDistancePairings(): invalid method")

            if method != 'rand':
                person_B = col_max_indices[person_A]
                pairings.append((person_A, person_B, col_maximums[person_A]))

            dm.drop(index=person_A, axis=0, inplace=True)
            dm.drop(person_A, axis=1, inplace=True)
            dm.drop(index=person_B, axis=0, inplace=True)
            dm.drop(person_B, axis=1, inplace=True)

            assert dm.shape[0] == dm.shape[1]

        # only two people left in dm so they must be paired together
        if dm.shape[0] == 2:
            pairings.append((dm.index[0], dm.index[1], dm.iat[0, 1]))
        # only one person left in dm so add them to the last pair that was created and use max distance among the three 3c2 possible distances among them
        else:
            assert dm.shape == (1, 1)
            temp_tuple = pairings[-1]
            pairings[-1] = (temp_tuple[0], temp_tuple[1], dm.index[0],
                            max(temp_tuple[2], self.dist_matrix.loc[temp_tuple[0], dm.index[0]], self.dist_matrix.loc[temp_tuple[1], dm.index[0]]))

        if self.verbose:
            print("Pairings:")
            print(pairings)

        stats_tmp = [x[-1] for x in pairings]
        mean_pair_dist = sum(stats_tmp) / len(stats_tmp)
        var_pair_dist = sum([(x[-1] - mean_pair_dist)**2 for x in pairings]) / len(stats_tmp)

        if self.verbose:
            print(f"Pairing via {method} method:")
            print(f"    mean(pair distances) = {mean_pair_dist}")
            print(f"     var(pair distances) = {var_pair_dist}")

        if write_csv:
            self.writePairingsCSV(method, pairings)
        return pairings

    def comparePairingMethods(self):
        """Compare the median, max, min, and rand methods of pairing students."""
        if self.dist_matrix is None:
            raise RuntimeError("generateDistanceMatrix() must be called before comparePairingMethods()")

        pairs_med = self.createStudentPairings(method='med', write_csv=False)
        pairs_max = self.createStudentPairings(method='max', write_csv=False)
        pairs_min = self.createStudentPairings(method='min', write_csv=False)
        pairs_rand = self.createStudentPairings(method='rand', write_csv=False)
        pairs_med_distances = pd.Series([x[-1] for x in pairs_med])
        pairs_max_distances = pd.Series([x[-1] for x in pairs_max])
        pairs_min_distances = pd.Series([x[-1] for x in pairs_min])
        pairs_rand_distances = pd.Series([x[-1] for x in pairs_rand])

        plt.figure(edgecolor='black')
        fig, axes = plt.subplots(1, 4, figsize=(20, 4))
        fig.patch.set_facecolor('white')
        bin_breaks = [x / 2 for x in range(0, (8 + 1))]
        pairs_med_distances.hist(bins=bin_breaks, ax=axes[0], edgecolor='black')
        pairs_max_distances.hist(bins=bin_breaks, ax=axes[1], edgecolor='black')
        pairs_min_distances.hist(bins=bin_breaks, ax=axes[2], edgecolor='black')
        pairs_rand_distances.hist(bins=bin_breaks, ax=axes[3], edgecolor='black')
        axes[0].set_xlabel('Pairing via: Median-Max Approach')
        axes[1].set_xlabel('Max-Max Approach')
        axes[2].set_xlabel('Min-Max Approach')
        axes[3].set_xlabel('Randomized Pairs')
        axes[1].set_ylim(0, 9)
        axes[0].set_ylim(0, 9)
        axes[2].set_ylim(0, 9)
        axes[3].set_ylim(0, 9)
        axes[0].set_ylabel('# of Student Pairs')
        plt.tight_layout()
        fig_path = self.figurePath("compare_pairing_methods")
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"Saved pairing method comparison to {fig_path.name}")

    def writePairingsCSV(self, method, pairs):
        """Create an output csv file in data/ with the given student pairings."""
        if getattr(self, 'df_quiz_scores_present', None) is None:
            raise RuntimeError("openPresentCSV() must be called before writePairingsCSV()")

        df = self.df_quiz_scores_present.copy()
        name1 = []
        name2 = []
        name3 = []
        person1 = []
        person2 = []
        person3 = []

        for i, pair in enumerate(pairs):
            name1.append(df.name[df.id == pair[0]].to_string(index=False))
            name2.append(df.name[df.id == pair[1]].to_string(index=False))
            person1.append(pair[0])
            person2.append(pair[1])

            # Only needed for 3-tuple groups since there are 3 possible distances.
            # Note that when there are 3 the first distance is used since first two people were paired intentionally.
            if len(pair) == 2 + 1:
                dists = (self.dist_matrix.loc[pair[0], pair[1]], -1, -1)
            else:
                dists = (self.dist_matrix.loc[pair[0], pair[1]], self.dist_matrix.loc[pair[0], pair[2]], self.dist_matrix.loc[pair[1], pair[2]])

            if self.verbose:
                print(f"pair = {pair}, dists = {dists}")

            if len(pair) == 2 + 1:
                name3.append(None)
                person3.append(-1)
            if len(pair) == 3 + 1:
                name3.append(df.name[df.id == pair[2]].to_string(index=False))
                person3.append(pair[2])
                if self.verbose:
                    print(f"    3-tuple {i + 1:2.0f}: {df.name[df.id == pair[0]].to_string(index=False)}, \
                          {df.name[df.id == pair[1]].to_string(index=False)}, {df.name[df.id == pair[2]].to_string(index=False)}")
                if self.verbose:
                    print(f"p1, p2, dist = {(pair[0], pair[1], self.dist_matrix.loc[pair[0], pair[1]])}")
                    print(f"p1, p3, dist = {(pair[0], pair[2], self.dist_matrix.loc[pair[0], pair[2]])}")
                    print(f"p2, p3, dist = {(pair[1], pair[2], self.dist_matrix.loc[pair[1], pair[2]])}")

        data = {'person1': name1, 'id1': person1, 'person2': name2, 'id2': person2}
        has_triples = any(v is not None for v in name3)
        if has_triples:
            data['person3'] = name3
            data['id3'] = person3
        data['distance'] = [x[-1] for x in pairs]
        df_pairs = pd.DataFrame(data)
        pairs_csv = self.config.data_path / f"pairings_based_on_{self.config.quiz_prefix}{self.canvas_quiz.id}_{today_str()}.csv"
        df_pairs.to_csv(pairs_csv, index=False)
        print(f"Saved pairings to {pairs_csv.name}")

    def _findMatchingPairs(self, student_ids, student_scores, student_timestamps, n_questions,
                           score_threshold, time_threshold_secs, time_overlap_threshold):
        """Compare all student pairs and return edges for those meeting score and timestamp thresholds."""
        partner_edges = []
        print(f"\nComparing {len(student_ids)} students across {n_questions} questions...")

        for i in range(len(student_ids)):
            for j in range(i + 1, len(student_ids)):
                id1, id2 = student_ids[i], student_ids[j]

                # Score comparison: fraction of questions with identical scores
                questions = set(student_scores.get(id1, {}).keys()) & set(student_scores.get(id2, {}).keys())
                if len(questions) < n_questions * score_threshold:
                    continue

                score_matches = sum(1 for q in questions
                                    if abs(student_scores[id1][q] - student_scores[id2][q]) < 0.001)
                score_overlap = score_matches / n_questions

                if score_overlap < score_threshold:
                    continue

                # Timestamp comparison: greedy closest-match on sorted timestamps
                ts1 = student_timestamps.get(id1, [])
                ts2 = student_timestamps.get(id2, [])

                if not ts1 or not ts2:
                    continue

                # Iterate over the shorter list, matching against the longer
                shorter, longer = (ts1, ts2) if len(ts1) <= len(ts2) else (ts2, ts1)
                longer_available = list(longer)
                time_matches = 0

                for t in shorter:
                    if not longer_available:
                        break
                    diffs = [abs((t_l - t).total_seconds()) for t_l in longer_available]
                    min_idx = diffs.index(min(diffs))
                    if diffs[min_idx] <= time_threshold_secs:
                        time_matches += 1
                        longer_available.pop(min_idx)

                time_overlap = time_matches / n_questions if n_questions > 0 else 0

                if time_overlap >= time_overlap_threshold:
                    partner_edges.append((id1, id2, score_overlap, time_overlap))

        return partner_edges

    def _groupPartnerEdges(self, student_ids, partner_edges):
        """Group matching pair edges into connected components using union-find (to detect triples)."""
        parent = {sid: sid for sid in student_ids}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for id1, id2, _, _ in partner_edges:
            union(id1, id2)

        groups = {}
        for id1, id2, so, to in partner_edges:
            root = find(id1)
            if root not in groups:
                groups[root] = {'members': set(), 'edges': []}
            groups[root]['members'].add(id1)
            groups[root]['members'].add(id2)
            groups[root]['edges'].append((id1, id2, so, to))

        return groups

    def _selectSubmissionDate(self):
        """Prompt user to select a date with get-all-subs data, or reuse a previously selected date."""
        if hasattr(self, '_selected_submission_date') and self._selected_submission_date:
            print(f"\nReusing previously selected date: {self._selected_submission_date}")
            return self._selected_submission_date

        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"
        subs_pattern = file_prefix + "all_submissions_"

        all_files = os.listdir(self.config.data_path)
        available_dates = set()
        for f in all_files:
            match = re.search(r'(\d{8})\.csv$', f)
            if match and subs_pattern in f:
                available_dates.add(match.group(1))

        available_dates = sorted(available_dates)
        if not available_dates:
            raise FileNotFoundError(
                f"No all_submissions CSV found for quiz '{self.quiz_name}'. "
                "Run the 'get-all-subs' task first to generate these files."
            )

        print("\nAvailable dates with submission data:")
        for i, d in enumerate(available_dates, start=1):
            print(f"[ {i} ] {d}")

        date_index = prompt_for_index("\nSelect date from above using index: ", len(available_dates) - 1)
        self._selected_submission_date = available_dates[date_index]
        return self._selected_submission_date

    def _selectSubmissionDataByDate(self):
        """Prompt user to select a date for which events and subs_by_question CSVs exist, then load them."""
        selected_date = self._selectSubmissionDate()

        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"
        events_pattern = file_prefix + "all_subs_and_events_"
        by_question_pattern = file_prefix + "all_subs_by_question_"

        events_csv = self.config.data_path / f"{events_pattern}{selected_date}.csv"
        subs_by_q_csv = self.config.data_path / f"{by_question_pattern}{selected_date}.csv"

        if not events_csv.exists() or not subs_by_q_csv.exists():
            raise FileNotFoundError(
                f"No matching all_subs_and_events / subs_by_question CSV pair found for date {selected_date}. "
                "Run the 'get-all-subs' task first to generate these files."
            )

        print(f"\nUsing files from {selected_date}")

        # Load and filter to first attempt
        events_df = pd.read_csv(events_csv)
        subs_by_q_df = pd.read_csv(subs_by_q_csv)

        events_df = events_df[(events_df['attempt'] == 1) & (events_df['event'] == 'question_answered')].copy()
        subs_by_q_df = subs_by_q_df[subs_by_q_df['attempt'] == 1].copy()

        events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

        # Also load the all_submissions CSV for first-attempt timestamps
        subs_pattern = file_prefix + "all_submissions_"
        subs_csv = self.config.data_path / f"{subs_pattern}{selected_date}.csv"
        if subs_csv.exists():
            subs_df = pd.read_csv(subs_csv)
            subs_df = subs_df[subs_df['attempt'] == 1].copy()
        else:
            subs_df = None

        return events_df, subs_by_q_df, subs_df

    def _loadAllSubmissions(self):
        """Load the all_submissions CSV for the selected date (all attempts, unfiltered)."""
        selected_date = self._selectSubmissionDate()

        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"
        subs_pattern = file_prefix + "all_submissions_"
        subs_csv = self.config.data_path / f"{subs_pattern}{selected_date}.csv"

        if not subs_csv.exists():
            raise FileNotFoundError(
                f"Submissions file not found: {subs_csv}. Run the 'get-all-subs' task first."
            )

        return pd.read_csv(subs_csv)

    def _buildFirstAttemptTimes(self, student_ids, events_df, first_attempt_subs):
        """Build a dict mapping student id to first-attempt start/finish/minutes from event and submission data."""
        first_attempt_times = {}
        if first_attempt_subs is not None:
            for _, srow in first_attempt_subs.iterrows():
                first_attempt_times[srow['id']] = {'finish': srow['timestamp']}

        for sid in student_ids:
            ts = events_df[events_df['id'] == sid]['timestamp']
            if len(ts) > 0:
                start = ts.min()
                if sid in first_attempt_times:
                    finish = pd.to_datetime(first_attempt_times[sid]['finish'])
                else:
                    finish = ts.max()
                first_attempt_times.setdefault(sid, {})
                first_attempt_times[sid]['start'] = str(start)
                first_attempt_times[sid]['finish'] = str(finish)
                first_attempt_times[sid]['minutes'] = (finish - start).total_seconds() / 60.0

        return first_attempt_times

    def detectPartners(self, score_threshold=0.8, time_threshold_secs=5, time_overlap_threshold=0.8, bonus_amount=0.15):
        """Detect student partners by comparing first-attempt per-question scores and answer timestamps.

        Finds pairs (or triples) of students whose first-attempt data shows:
        - At least score_threshold fraction of questions with identical scores
        - At least time_overlap_threshold fraction of question_answered timestamps within time_threshold_secs
        """
        events_df, subs_by_q_df, first_attempt_subs = self._selectSubmissionDataByDate()

        # Build per-student score and timestamp profiles
        student_ids = sorted(subs_by_q_df['id'].unique())
        n_questions = subs_by_q_df['question'].nunique()
        name_map = dict(zip(self.quiz_df['id'], self.quiz_df['name']))

        student_scores = {}
        student_timestamps = {}

        for sid in student_ids:
            scores = subs_by_q_df[subs_by_q_df['id'] == sid]
            student_scores[sid] = dict(zip(scores['question'], scores['points']))

            ts = events_df[events_df['id'] == sid].sort_values('timestamp')['timestamp'].tolist()
            student_timestamps[sid] = ts

        # Find all matching pairs, then group into connected components
        partner_edges = self._findMatchingPairs(student_ids, student_scores, student_timestamps, n_questions,
                                                score_threshold, time_threshold_secs, time_overlap_threshold)
        groups = self._groupPartnerEdges(student_ids, partner_edges)

        # Print detected groups
        print(f"\nDetected {len(groups)} partner group(s):")
        for root, group in groups.items():
            members = sorted(group['members'])
            names = [name_map.get(m, str(m)) for m in members]
            print(f"  Group: {', '.join(names)}")
            for id1, id2, so, to in group['edges']:
                print(f"    {name_map.get(id1, str(id1))} & {name_map.get(id2, str(id2))}: "
                      f"score overlap={so:.0%}, time overlap={to:.0%}")

        # Calculate bonus amount
        if bonus_amount < 1.0:
            bonus = round(bonus_amount * self.canvas_quiz.points_possible, 2)
        else:
            bonus = bonus_amount

        # Build df_paired_students (same format awardBonusPoints() expects)
        paired_data = []
        pairs_output = []

        for root, group in groups.items():
            members = sorted(group['members'])
            if len(members) > 3:
                print(f"  WARNING: Group of {len(members)} detected — review manually, skipping bonus for this group")
                continue

            for m in members:
                paired_data.append({'name': name_map.get(m, str(m)), 'id': m, 'bonus': bonus})

            row = {
                'person1': name_map.get(members[0], ''),
                'id1': members[0],
                'person2': name_map.get(members[1], '') if len(members) > 1 else None,
                'id2': members[1] if len(members) > 1 else -1,
                'person3': name_map.get(members[2], '') if len(members) > 2 else None,
                'id3': members[2] if len(members) > 2 else -1,
            }
            avg_score = sum(e[2] for e in group['edges']) / len(group['edges'])
            avg_time = sum(e[3] for e in group['edges']) / len(group['edges'])
            row['score_overlap'] = round(avg_score, 3)
            row['time_overlap'] = round(avg_time, 3)
            pairs_output.append(row)

        self.df_paired_students = pd.DataFrame(paired_data) if paired_data else pd.DataFrame(columns=['name', 'id', 'bonus'])

        # Build first-attempt timestamp lookup for awardBonusPoints
        self.first_attempt_times = self._buildFirstAttemptTimes(student_ids, events_df, first_attempt_subs)

        # Save detected partners CSV
        df_detected = pd.DataFrame(pairs_output)
        detected_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_detected_partners_{today_str()}.csv"
        df_detected.to_csv(detected_csv, index=False)
        print(f"\nSaved detected partners to {detected_csv}")
        logger.info(f"Detected {len(groups)} partner groups, saved to {detected_csv}")

    def detectRetakers(self, min_attempts=3, min_days_between=1, bonus_amount=0.15):
        """Detect students who retook the quiz enough times with sufficient spacing to earn the retake bonus.

        A student qualifies if they have at least min_attempts submissions where each subsequent
        qualifying attempt is at least min_days_between day(s) after the previous one.
        """
        subs_df = self._loadAllSubmissions()
        name_map = dict(zip(self.quiz_df['id'], self.quiz_df['name']))

        if bonus_amount < 1.0:
            bonus = round(bonus_amount * self.canvas_quiz.points_possible, 2)
        else:
            bonus = bonus_amount

        retake_data = []
        for sid in subs_df['id'].unique():
            student_subs = subs_df[subs_df['id'] == sid].sort_values('timestamp')
            timestamps = pd.to_datetime(student_subs['timestamp']).tolist()

            # Count qualifying attempts: first always counts, then each subsequent one
            # must be at least min_days_between days after the previous qualifying attempt
            qualifying = [timestamps[0]]
            for ts in timestamps[1:]:
                if (ts - qualifying[-1]).total_seconds() >= min_days_between * 86400:
                    qualifying.append(ts)

            if len(qualifying) >= min_attempts:
                retake_data.append({
                    'name': name_map.get(sid, str(sid)),
                    'id': sid,
                    'retake_bonus': bonus,
                    'qualifying_attempts': len(qualifying),
                    'total_attempts': len(timestamps)
                })

        self.df_retake_students = pd.DataFrame(retake_data) if retake_data else pd.DataFrame(
            columns=['name', 'id', 'retake_bonus', 'qualifying_attempts', 'total_attempts'])

        print(f"\nDetected {len(retake_data)} student(s) qualifying for retake bonus "
              f"(>= {min_attempts} attempts, >= {min_days_between} day(s) apart):")
        for _, row in self.df_retake_students.iterrows():
            print(f"  {row['name']}: {row['qualifying_attempts']} qualifying attempts "
                  f"(of {row['total_attempts']} total)")

        retake_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_retake_qualified_{today_str()}.csv"
        self.df_retake_students.to_csv(retake_csv, index=False)
        print(f"\nSaved retake-qualified students to {retake_csv}")
        logger.info(f"Detected {len(retake_data)} retake-qualified students, saved to {retake_csv}")

    def _buildSubByQuestionRow(self, user_id, attempt_num, q_idx, qdata):
        """Build one row for the all_subs_by_question CSV from a Canvas submission_data entry.

        Canvas's per-submission submission_data does not reliably include points_possible;
        prefer the authoritative value parsed from the student_analysis CSV header.
        """
        qid = qdata.get('question_id')
        pp = self.question_points_possible.get(qid) if qid is not None else None
        if pp is None:
            pp = qdata.get('points_possible')
        return {
            'id': user_id,
            'attempt': attempt_num,
            'question': q_idx + 1,
            'question_id': qid,
            'points': qdata['points'],
            'points_possible': pp,
            'correct': qdata['correct'],
        }

    def getAllSubmissionsAndEvents(self):
        """Collect per-attempt submission history and events into three CSVs."""
        quiz_takers = self.quiz_df[['name', 'id']].copy()

        # Lists to collect data before creating DataFrames
        submissions_data = []
        subs_by_question_data = []
        subs_and_events_data = []

        n_total = self.n_students or 0
        subs = self.canvas_quiz.get_submissions(include=['submission_history'])
        for i, sub in enumerate(subs):
            self._spin(i, f"Fetching submissions... student {i + 1}/{n_total}")
            if self.verbose:
                print(f"\nProcessing submission for student id {sub.user_id}")
            student_subs = self.canvas_course.canvas_course.get_multiple_submissions(
                student_ids=[sub.user_id],
                assignment_ids=[self.canvas_quiz.assignment_id],
                include=['submission_history'])[0]
            n_attempts = len(student_subs.submission_history)
            for a in range(n_attempts):
                attempt_data = student_subs.submission_history[a]
                attempt_num = attempt_data['attempt']

                # Compute raw score from per-question points to exclude any fudge points
                # that may have been added by a prior award-bonus run
                submission_data = attempt_data.get('submission_data', [])
                raw_score = sum(qdata['points'] for qdata in submission_data) if submission_data else attempt_data['score']

                if submission_data and abs(attempt_data['score'] - raw_score) >= 0.001:
                    logger.info(f"Student {sub.user_id} attempt {attempt_num}: Canvas score={attempt_data['score']}, "
                                f"raw score={raw_score} (fudge points excluded)")

                new_row = {
                    'id': sub.user_id,
                    'attempt': attempt_num,
                    'score': raw_score,
                    'timestamp': attempt_data['submitted_at']
                }
                submissions_data.append(new_row)

                # check that this an attempt/submission exists before trying to get all of the events for it
                try:
                    if attempt_num is None:
                        break
                except Exception:
                    break

                # now get question-level data for this attempt
                for q, qdata in enumerate(attempt_data['submission_data']):
                    subs_by_question_data.append(
                        self._buildSubByQuestionRow(sub.user_id, attempt_num, q, qdata)
                    )

                # see scratch_work.py for getting all events for this submission
                this_submission_events = sub.get_submission_events(attempt=attempt_num)

                try:
                    for event in this_submission_events:
                        subs_and_events_data.append({
                            'id': sub.user_id,
                            'attempt': attempt_num,
                            'event': event.event_type,
                            'timestamp': event.created_at,
                            'question_id': _extract_question_id(event) if event.event_type == 'question_answered' else None,
                        })
                except Exception:
                    print(f"  !!! could not get events for student id {sub.user_id} for attempt {attempt_num}")
                    continue

        # Create DataFrames from the collected lists
        all_submissions = pd.DataFrame(submissions_data)
        all_subs_by_question = pd.DataFrame(subs_by_question_data)
        all_subs_and_events = pd.DataFrame(subs_and_events_data, columns=['id', 'attempt', 'event', 'timestamp', 'question_id'])

        # do a full outer join of quiz_takers on 'id' to get names for the submission data
        if not all_submissions.empty:
            all_submissions = pd.merge(quiz_takers[['name', 'id']], all_submissions, on='id', how='inner')
        if not all_subs_by_question.empty:
            all_subs_by_question = pd.merge(quiz_takers[['name', 'id']], all_subs_by_question, on='id', how='inner')
        if not all_subs_and_events.empty:
            all_subs_and_events = pd.merge(quiz_takers[['name', 'id']], all_subs_and_events, on='id', how='inner')

        all_submissions_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_all_submissions_{today_str()}.csv"
        all_submissions.to_csv(all_submissions_csv, index=False)
        self._spin_done(f"Saved: {all_submissions_csv.name}")

        all_subs_by_question_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_all_subs_by_question_{today_str()}.csv"
        all_subs_by_question.to_csv(all_subs_by_question_csv, index=False)
        print(f"  ✓ Saved: {all_subs_by_question_csv.name}")
        self.all_subs_by_question = all_subs_by_question

        all_sub_and_events_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_all_subs_and_events_{today_str()}.csv"
        all_subs_and_events.to_csv(all_sub_and_events_csv, index=False)
        print(f"  ✓ Saved: {all_sub_and_events_csv.name}")
        self.all_subs_and_events = all_subs_and_events

    def _populateTimestamps(self, quiz_summary, row_index, sub):
        """Fill in start/finish/minutes for a student row using first-attempt times if available."""
        first_times = getattr(self, 'first_attempt_times', {})
        if sub.user_id in first_times:
            fa = first_times[sub.user_id]
            quiz_summary.at[row_index, 'start'] = fa.get('start', 'n/a')
            quiz_summary.at[row_index, 'finish'] = fa.get('finish', 'n/a')
            quiz_summary.at[row_index, 'minutes'] = fa.get('minutes', 0.0)
        else:
            quiz_summary.at[row_index, 'start'] = sub.started_at
            quiz_summary.at[row_index, 'finish'] = sub.finished_at
            quiz_summary.at[row_index, 'minutes'] = sub.time_spent / 60.0

    def awardBonusPoints(self, dry_run=False):
        """Award bonus points to students by setting fudge points (partner bonus, retake bonus, or both)."""
        has_partner = getattr(self, 'df_paired_students', None) is not None and not self.df_paired_students.empty
        has_retake = getattr(self, 'df_retake_students', None) is not None and not self.df_retake_students.empty
        if not has_partner and not has_retake:
            raise RuntimeError("detectPartners() and/or detectRetakers() must be called before awardBonusPoints()")

        # Load all_submissions to find each student's best attempt (highest score, latest attempt if tied)
        subs_df = self._loadAllSubmissions()
        subs_df = subs_df.sort_values(['id', 'score', 'attempt'], ascending=[True, True, True])
        best_attempts = subs_df.groupby('id').last()[['attempt']].rename(columns={'attempt': 'best_attempt'})
        best_attempt_map = best_attempts['best_attempt'].to_dict()

        quiz_summary = self.quiz_df[['name', 'id', 'n_correct', 'n_incorrect', 'score']].copy()

        # Merge in partner and retake bonuses
        quiz_summary['partner_bonus'] = 0.0
        quiz_summary['retake_bonus'] = 0.0

        if has_partner:
            partner_map = self.df_paired_students.set_index('id')['bonus']
            quiz_summary['partner_bonus'] = quiz_summary['id'].map(partner_map).fillna(0.0)

        if has_retake:
            retake_map = self.df_retake_students.set_index('id')['retake_bonus']
            quiz_summary['retake_bonus'] = quiz_summary['id'].map(retake_map).fillna(0.0)

        quiz_summary['bonus'] = quiz_summary['partner_bonus'] + quiz_summary['retake_bonus']

        # Create new columns for start, finish, minutes, and score_w_bonus
        quiz_summary['start'] = 'n/a'
        quiz_summary['finish'] = 'n/a'
        quiz_summary['minutes'] = 0.0
        quiz_summary['score_w_bonus'] = 0.0

        # Sort quiz_summary by the second word in 'name' column
        quiz_summary['lastname'] = quiz_summary['name'].str.split().str[1]
        quiz_summary.sort_values(by='lastname', inplace=True)
        quiz_summary.drop(columns='lastname', inplace=True)
        quiz_summary.reset_index(drop=True, inplace=True)

        # Get the Canvas Assignment corresponding to this quiz (needed for leaving submission comments)
        assignment = self.canvas_course.canvas_course.get_assignment(self.canvas_quiz.assignment_id)

        if dry_run:
            print("\n=== DRY RUN MODE - No changes will be made to Canvas ===\n")

        subs = self.canvas_quiz.get_submissions()
        for _, sub in enumerate(subs):
            # Get row from quiz_summary where column 'id' matches sub.user_id
            row = quiz_summary[quiz_summary['id'] == sub.user_id]

            # Canvas may return a lower score than the student_analysis report if the
            # student retook the quiz and did worse than on their best attempt(s); log a note when they differ.
            report_score = row['score'].values[0]
            if abs(sub.score - report_score) >= 0.001:
                student_name = quiz_summary.at[row.index[0], 'name']
                logger.info(f"Score differs for {student_name} (id: {sub.user_id}): "
                            f"Canvas={sub.score}, report={report_score} (likely a retake improvement)")

            self._populateTimestamps(quiz_summary, row.index[0], sub)

            # Check if bonus needs to be added
            bonus_val = row['bonus'].values[0]
            if bonus_val > 0:

                best_attempt = best_attempt_map.get(sub.user_id, sub.attempt)

                # Build a comment describing the bonus breakdown
                p_bonus = row['partner_bonus'].values[0]
                r_bonus = row['retake_bonus'].values[0]
                comment_parts = []
                if p_bonus > 0:
                    comment_parts.append(f"partner bonus = {p_bonus}")
                if r_bonus > 0:
                    comment_parts.append(f"retake bonus = {r_bonus}")
                comment_text = "Bonus points: " + ", ".join(comment_parts) + f" (total = {bonus_val})"

                if dry_run:
                    student_name = quiz_summary.at[row.index[0], 'name']
                    parts = []
                    if p_bonus > 0:
                        parts.append(f"partner={p_bonus}")
                    if r_bonus > 0:
                        parts.append(f"retake={r_bonus}")
                    detail = ", ".join(parts)
                    attempt_note = f", attempt={best_attempt}" if best_attempt != sub.attempt else ""
                    print(f"  [DRY RUN] Would award {bonus_val} bonus points to {student_name} "
                          f"(id: {sub.user_id}, {detail}{attempt_note})")
                    print(f"            Comment: \"{comment_text}\"")
                    logger.info(f"[DRY RUN] Would award {bonus_val} bonus to student {sub.user_id}")
                else:
                    # Set points before fudge points are added
                    newattributes = {'excused?': True, 'score_before_regrade': sub.score}
                    sub.set_attributes(newattributes)

                    # Now set fudge points on the student's best attempt (combined partner + retake bonus)
                    update_obj = [{'attempt': best_attempt, 'fudge_points': bonus_val}]
                    sub.update_score_and_comments(quiz_submissions=update_obj)

                    # Leave a comment on the assignment submission describing the bonus
                    assgn_sub = assignment.get_submission(sub.user_id)
                    assgn_sub.edit(comment={'text_comment': comment_text})

                    if best_attempt != sub.attempt:
                        logger.info(f"Awarded {bonus_val} bonus to student {sub.user_id} "
                                    f"on best attempt {best_attempt} (latest was {sub.attempt})")
                    else:
                        logger.info(f"Awarded {bonus_val} bonus to student {sub.user_id}")

                quiz_summary.at[row.index[0], 'score_w_bonus'] = row['score'].values[0] + bonus_val

            else:
                quiz_summary.at[row.index[0], 'score_w_bonus'] = row['score'].values[0]

        suffix = "_dryrun" if dry_run else ""
        quiz_summary_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_scores_w_bonus{suffix}_{today_str()}.csv"
        quiz_summary.to_csv(quiz_summary_csv, index=False)

        if dry_run:
            print(f"\n[DRY RUN] No changes were made to Canvas. Review output: {quiz_summary_csv}")
