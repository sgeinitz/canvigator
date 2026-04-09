import sys
import time
import random
import logging
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

        quiz_report_progress = self.canvas.get_progress(request_id)
        while quiz_report_progress.workflow_state != 'completed':
            self.progressBar(quiz_report_progress.completion, 100)
            time.sleep(0.1)
            quiz_report_progress = self.canvas.get_progress(request_id)
        self.progressBar(quiz_report_progress.completion, 100)
        print(f"\n{self.quiz_name} download complete")
        logger.info(f"Quiz report downloaded: {self.quiz_name}")

        quiz_report = self.canvas_quiz.get_quiz_report(quiz_report_request)
        quiz_csv_url = quiz_report.file['url']
        quiz_csv = requests.get(quiz_csv_url)
        csv_name = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_student_analysis_{today_str()}.csv"

        with open(csv_name, 'wb') as f:
            for content in quiz_csv.iter_content(chunk_size=2**20):
                if content:
                    f.write(content)

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

    def getQuizQuestions(self):
        """Save quiz metadata and questions to a CSV file."""
        quiz_id = self.canvas_quiz.id
        assignment_id = getattr(self.canvas_quiz, 'assignment_id', None)

        fields = ['id', 'position', 'question_name', 'question_type', 'question_text', 'points_possible']
        rows = []
        for q in self.quiz_questions:
            row = {'quiz_id': quiz_id, 'assignment_id': assignment_id}
            row.update({f: getattr(q, f, None) for f in fields})
            row['answers'] = json.dumps(getattr(q, 'answers', []))
            rows.append(row)

        columns = ['quiz_id', 'assignment_id'] + fields + ['answers']
        df = pd.DataFrame(rows, columns=columns)
        csv_name = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_data_and_content_{today_str()}.csv"
        df.to_csv(csv_name, index=False)
        print(f"Saved {len(rows)} questions to {csv_name}")
        logger.info(f"Quiz data and content saved: {csv_name}")

    def sendQuizReminders(self, dry_run=False):
        """Send reminder messages to students who haven't taken the quiz or haven't achieved a perfect score."""
        quiz_name = self.canvas_quiz.title
        points_possible = self.canvas_quiz.points_possible

        # Build set of student IDs and scores from the student_analysis report
        quiz_scores = {}
        if self.quiz_df is not None and self.n_students is not None and self.n_students > 0:
            for _, row in self.quiz_df.iterrows():
                student_id = row['id']
                score = row['score']
                if pd.notna(score):
                    quiz_scores[student_id] = score

        # All enrolled students from the course
        enrolled = self.canvas_course.students

        no_attempt_template = (
            "You have not yet attempted {quiz_name}. Be sure to make an attempt soon "
            "to help stay on top of the content in this course. And remember, the best "
            "way to use the quiz as a learning tool is to try to answer the questions "
            "without going to outside references or AI tools. Trying to answer on your "
            "own, even if it feels like a struggle, is the best way to help learn this "
            "material. Good luck!"
        )

        imperfect_template = (
            "Nice work on attempting {quiz_name}, but you don't yet have a perfect "
            "score. Be sure to try it again soon to earn a perfect score. And remember, "
            "quizzes are most effective as learning tools when you try to answer the "
            "questions on your own without using any other resources. Quizzes work best "
            "when it feels like a struggle to recall the concepts and ideas in your mind, "
            "so embrace the struggle. Good luck!"
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
            else:
                continue

            message_str = f"Hello {first_name}, {reminder}"
            messages.append((student_id, student_name, message_str, reason))

        if not messages:
            print("No reminders to send — all students have perfect scores!")
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
                    [str(student_id)], message_str, subject=subject_str
                )
                print(f"  Sent to: {student_name} (id: {student_id}, {reason})")

        n_no_attempt = sum(1 for _, _, _, r in messages if r == "no attempt")
        n_imperfect = len(messages) - n_no_attempt
        action = "would be sent" if dry_run else "sent"
        summary_parts = []
        if n_no_attempt:
            summary_parts.append(f"{n_no_attempt} no-attempt")
        if n_imperfect:
            summary_parts.append(f"{n_imperfect} imperfect-score")
        print(f"\n{len(messages)} reminder(s) {action} ({', '.join(summary_parts)}).")
        logger.info(f"Quiz reminders {'(dry run) ' if dry_run else ''}{action}: {len(messages)} for {quiz_name}")

    def figurePath(self, figure_name):
        """Return a figure output path with the date suffix at the end."""
        return self.config.figures_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_{figure_name}_{today_str()}.png"

    def progressBar(self, current, total, bar_length=20):
        """Displays or updates a console progress bar."""
        progress = current / total
        arrow = '=' * int(progress * bar_length - 1) + '>' if current < total else '=' * bar_length
        spaces = ' ' * (bar_length - len(arrow))
        percent = int(progress * 100)
        sys.stdout.write(f"\r[{arrow}{spaces}] {percent}%")
        sys.stdout.flush()

    def generateQuestionHistograms(self):
        """Draw a histogram of scores of each question."""
        mpl.style.use('seaborn-v0_8')
        figure, axis = plt.subplots(1, len(self.quiz_question_ids), sharey=True)
        figure.set_size_inches(13, 3)
        for i, q in enumerate(self.quiz_question_ids):
            score_col = q + '_score'
            axis[i].hist(self.quiz_df[score_col], bins=6, facecolor='#00447c', edgecolor='black', alpha=0.8)
            axis[i].set_xlabel('score')
            axis[i].set_title('question: ' + q.split('_')[0])
        axis[0].set_ylabel('# of people')
        plt.tight_layout()  # Or try plt.subplots_adjust(left=0.05, right=0.98, bottom=0.15, top=0.9)
        fig_path = self.figurePath("histograms")
        figure.savefig(fig_path, dpi=200)
        plt.close('all')
        print(f"  Saved: {fig_path.name}")

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

    def getAllSubmissionsAndEvents(self):
        """Collect per-attempt submission history and events into three CSVs."""
        quiz_takers = self.quiz_df[['name', 'id']].copy()

        # Lists to collect data before creating DataFrames
        submissions_data = []
        subs_by_question_data = []
        subs_and_events_data = []

        subs = self.canvas_quiz.get_submissions(include=['submission_history'])
        for i, sub in enumerate(subs):
            if self.verbose:
                print(f"Processing submission for student id {sub.user_id}")
            student_subs = self.canvas_course.canvas_course.get_multiple_submissions(
                student_ids=[sub.user_id],
                assignment_ids=[self.canvas_quiz.assignment_id],
                include=['submission_history'])[0]
            n_attempts = len(student_subs.submission_history)
            for i in range(n_attempts):
                attempt_data = student_subs.submission_history[i]
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
                    new_q_row = {'id': sub.user_id,
                                 'attempt': attempt_num,
                                 'question': q + 1,
                                 'points': qdata['points'],
                                 'correct': qdata['correct']}
                    subs_by_question_data.append(new_q_row)

                # see scratch_work.py for getting all events for this submission
                this_submission_events = sub.get_submission_events(attempt=i + 1)  # get sub. events for this attempt

                try:
                    for event in this_submission_events:
                        subs_and_events_data.append({
                            'id': sub.user_id,
                            'attempt': i + 1,
                            'event': event.event_type,
                            'timestamp': event.created_at
                        })
                except Exception:
                    print(f"  !!! could not get events for student id {sub.user_id} for attempt {i + 1}")
                    continue

        # Create DataFrames from the collected lists
        all_submissions = pd.DataFrame(submissions_data)
        all_subs_by_question = pd.DataFrame(subs_by_question_data)
        all_subs_and_events = pd.DataFrame(subs_and_events_data, columns=['id', 'attempt', 'event', 'timestamp'])

        # do a full outer join of quiz_takers on 'id' to get names for the submission data
        if not all_submissions.empty:
            all_submissions = pd.merge(quiz_takers[['name', 'id']], all_submissions, on='id', how='inner')
        if not all_subs_by_question.empty:
            all_subs_by_question = pd.merge(quiz_takers[['name', 'id']], all_subs_by_question, on='id', how='inner')
        if not all_subs_and_events.empty:
            all_subs_and_events = pd.merge(quiz_takers[['name', 'id']], all_subs_and_events, on='id', how='inner')

        all_submissions_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_all_submissions_{today_str()}.csv"
        all_submissions.to_csv(all_submissions_csv, index=False)

        all_subs_by_question_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_all_subs_by_question_{today_str()}.csv"
        all_subs_by_question.to_csv(all_subs_by_question_csv, index=False)

        all_sub_and_events_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_all_subs_and_events_{today_str()}.csv"
        all_subs_and_events.to_csv(all_sub_and_events_csv, index=False)

        print(f"  Saved: {all_submissions_csv.name}")
        print(f"  Saved: {all_subs_by_question_csv.name}")
        print(f"  Saved: {all_sub_and_events_csv.name}")

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
