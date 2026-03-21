import sys
import time
import random
import logging
import requests
import pandas as pd
import scipy.stats as stats
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sbn
import os
import re
from canvigator_utils import today_str, selectCSVFromList, prompt_for_index

logger = logging.getLogger(__name__)


class CanvigatorQuiz:
    """A class for one quiz and associated attributes/data."""

    def __init__(self, canvas, canvas_course, canvas_quiz, config, verbose=False):
        """Initialize quiz object by getting all quiz data from Canvas."""
        self.canvas = canvas
        self.canvas_course = canvas_course
        self.canvas_quiz = canvas_quiz
        self.published = canvas_quiz.published
        self.verbose = verbose
        self.config = config
        self.quiz_name = self.canvas_quiz.title.lower().replace(" ", "_")
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
        if self.published:
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
        figure.savefig(self.figurePath("histograms"), dpi=200)
        plt.close('all')

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
        plt.savefig(self.figurePath(f"dist_{distance_type}"), dpi=200)
        plt.close()

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

    def getPastPairingsCSV(self, csv_path=None):
        """Prompt for a CSV that contains past pairings for this quiz and return a pandas dataframe."""
        if not csv_path:
            csv_path = self.config.data_path

        selected = selectCSVFromList(
            csv_path, 'pairing',
            "\nSelect csv with student pairings from past quiz using index: ",
            verbose=self.verbose
        )

        # Open the file and return the dataframe
        self.df_past_pairings = pd.read_csv(selected)

        # Get list of students who were paired in the past and put into long format
        paired_students1 = self.df_past_pairings[['person1', 'id1']].copy()
        paired_students1.rename(columns={'person1': 'name', 'id1': 'id'}, inplace=True)
        paired_students2 = self.df_past_pairings[['person2', 'id2']].copy()
        paired_students2.rename(columns={'person2': 'name', 'id2': 'id'}, inplace=True)
        paired_students3 = self.df_past_pairings[['person3', 'id3']].copy()
        paired_students3.rename(columns={'person3': 'name', 'id3': 'id'}, inplace=True)

        # Get columns person2, id2, from df_past_pairings and append these rows to paired_students
        paired_students = pd.concat([paired_students1, paired_students2, paired_students3], ignore_index=True)

        # Remove any rows with NaN or -1 values, and reset index
        paired_students.dropna(inplace=True)
        paired_students = paired_students[paired_students['id'] != -1]
        paired_students.reset_index(drop=True, inplace=True)
        self.df_paired_students = paired_students.drop_duplicates()

    def getPastBonusCSV(self, csv_path=None):
        """Prompt for a CSV that contains past pairings for this quiz and return a pandas dataframe."""
        if not csv_path:
            csv_path = self.config.data_path

        selected = selectCSVFromList(
            csv_path, 'w_bonus',
            "\nSelect csv with past student bonus awards using index: ",
            verbose=self.verbose
        )

        # Open the file and return the dataframe
        self.df_past_bonus = pd.read_csv(selected)

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
        plt.savefig(self.figurePath("compare_pairing_methods"), dpi=200)
        plt.close()

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
                    print(f"    3-tuple {i+1:2.0f}: {df.name[df.id == pair[0]].to_string(index=False)}, \
                          {df.name[df.id == pair[1]].to_string(index=False)}, {df.name[df.id == pair[2]].to_string(index=False)}")
                if self.verbose:
                    print(f"p1, p2, dist = {(pair[0], pair[1], self.dist_matrix.loc[pair[0], pair[1]])}")
                    print(f"p1, p3, dist = {(pair[0], pair[2], self.dist_matrix.loc[pair[0], pair[2]])}")
                    print(f"p2, p3, dist = {(pair[1], pair[2], self.dist_matrix.loc[pair[1], pair[2]])}")

        df_pairs = pd.DataFrame({'person1': name1, 'id1': person1, 'person2': name2, 'id2': person2,
                                 'person3': name3, 'id3': person3, 'distance': [x[-1] for x in pairs]})
        pairs_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_pairing_via_{method}_{today_str()}.csv"
        df_pairs.to_csv(pairs_csv, index=False)

    def checkForBonusEarned(self, bonus_amount=0.2):
        """Check if any students have a distance of 0 with their partner."""
        if self.dist_matrix is None:
            raise RuntimeError("generateDistanceMatrix() must be called before checkForBonusEarned()")
        if getattr(self, 'df_past_pairings', None) is None or getattr(self, 'df_paired_students', None) is None:
            raise RuntimeError("getPastPairingsCSV() must be called before checkForBonusEarned()")

        # Rename the column 'distance' to 'previous_distance' in df_past_pairings
        self.df_past_pairings.rename(columns={'distance': 'previous_distance'}, inplace=True)

        # Create a new column 'distance' in df_past_pairings and set it to 0
        self.df_past_pairings['distance'] = 0.0

        # Create a new column 'bonus' in df_paired_students and set it to 0
        self.df_paired_students['bonus'] = 0.0

        # set bonus_to_add to be added as a percentage of the total points possible, or as a fixed number of points
        if bonus_amount < 1.0:
            bonus = round(bonus_amount * self.canvas_quiz.points_possible)
        else:
            bonus = bonus_amount

        # Iterate through each row of df_past_pairings
        for i, row in self.df_past_pairings.iterrows():
            person1 = row['id1']
            person2 = row['id2']
            dist = self.dist_matrix.loc[person1, person2]
            person3 = row['id3']
            if person3 > 0:
                dist = max(dist, self.dist_matrix.loc[person1, person3], self.dist_matrix.loc[person2, person3])
            self.df_past_pairings.at[i, 'distance'] = dist

            if dist <= 0.01:
                self.df_paired_students.loc[self.df_paired_students['id'] == person1, 'bonus'] = bonus
                self.df_paired_students.loc[self.df_paired_students['id'] == person2, 'bonus'] = bonus
                if person3 > 0:
                    self.df_paired_students.loc[self.df_paired_students['id'] == person3, 'bonus'] = bonus

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

    def _selectSubmissionDataByDate(self):
        """Prompt user to select a date for which events and subs_by_question CSVs exist, then load them."""
        file_prefix = f"{self.config.quiz_prefix}{self.canvas_quiz.id}_"
        events_pattern = file_prefix + "all_subs_and_events_"
        by_question_pattern = file_prefix + "all_subs_by_question_"

        all_files = os.listdir(self.config.data_path)
        events_dates = set()
        by_question_dates = set()

        for f in all_files:
            match = re.search(r'(\d{8})\.csv$', f)
            if match:
                date_str = match.group(1)
                if events_pattern in f:
                    events_dates.add(date_str)
                elif by_question_pattern in f:
                    by_question_dates.add(date_str)

        common_dates = sorted(events_dates & by_question_dates)
        if not common_dates:
            raise FileNotFoundError(
                f"No matching all_subs_and_events / subs_by_question CSV pair found for quiz '{self.quiz_name}'. "
                "Run the 'all-subs' task first to generate these files."
            )

        print("\nAvailable dates with submission data:")
        for i, d in enumerate(common_dates):
            print(f"[ {i} ] {d}")

        date_index = prompt_for_index("\nSelect date from above using index: ", len(common_dates) - 1)
        selected_date = common_dates[date_index]

        events_csv = self.config.data_path / f"{events_pattern}{selected_date}.csv"
        subs_by_q_csv = self.config.data_path / f"{by_question_pattern}{selected_date}.csv"
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

    def detectPartners(self, score_threshold=0.8, time_threshold_secs=10, time_overlap_threshold=0.8, bonus_amount=0.2):
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
            bonus = round(bonus_amount * self.canvas_quiz.points_possible)
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

    def getAllSubmissionsAndEvents(self):
        quiz_takers = self.quiz_df[['name', 'id']].copy()

        # Lists to collect data before creating DataFrames
        submissions_data = []
        subs_by_question_data = []
        subs_and_events_data = []

        subs = self.canvas_quiz.get_submissions(include=['submission_history'])
        for i, sub in enumerate(subs):
            if self.verbose:
                print(f"Processing submission for student id {sub.user_id}")
            student_subs = self.canvas_course.canvas_course.get_multiple_submissions(student_ids=[sub.user_id],
                                                                       assignment_ids=[self.canvas_quiz.assignment_id],
                                                                       include=['submission_history'])[0]
            n_attempts = len(student_subs.submission_history)
            for i in range(n_attempts):
                attempt_data = student_subs.submission_history[i]
                attempt_num = attempt_data['attempt']

                new_row = {'id': sub.user_id,
                        'attempt': attempt_num,
                        'score': attempt_data['score'],
                        'timestamp': attempt_data['submitted_at']}
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
                                 'question': q+1,
                                 'points': qdata['points'],
                                 'correct': qdata['correct']}
                    subs_by_question_data.append(new_q_row)

                # see scratch_work.py for getting all events for this submission
                this_submission_events = sub.get_submission_events(attempt=i+1)  # get sub. events for this attempt

                try:
                    for event in this_submission_events:
                        subs_and_events_data.append({
                            'id': sub.user_id,
                            'attempt': i+1,
                            'event': event.event_type,
                            'timestamp': event.created_at
                        })
                except Exception:
                    print(f"  !!! could not get events for student id {sub.user_id} for attempt {i+1}")
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

    def awardBonusPoints(self, dry_run=False):
        """Award bonus points to students who received it by setting fudge points."""
        if getattr(self, 'df_paired_students', None) is None:
            raise RuntimeError("checkForBonusEarned() must be called before awardBonusPoints()")

        quiz_summary = self.quiz_df[['name', 'id', 'n_correct', 'n_incorrect', 'score']].copy()

        # Left join quiz_summary with df_paired_students on 'id' to get bonus points
        quiz_summary = pd.merge(quiz_summary, self.df_paired_students[['id', 'bonus']], on='id', how='left')

        # Fill NaN values in 'bonus' column with -1.0
        with pd.option_context("future.no_silent_downcasting", True):
            quiz_summary = quiz_summary.fillna(-1.0).infer_objects(copy=False)

        # Create new columns for start, finish, minutes, and score_w_bonus and set them to 0
        quiz_summary['start'] = 'n/a'
        quiz_summary['finish'] = 'n/a'
        quiz_summary['minutes'] = 0.0
        quiz_summary['score_w_bonus'] = 0.0

        # Sort quiz_summary by the second word in 'name' column
        quiz_summary['lastname'] = quiz_summary['name'].str.split().str[1]
        quiz_summary.sort_values(by='lastname', inplace=True)
        quiz_summary.drop(columns='lastname', inplace=True)
        quiz_summary.reset_index(drop=True, inplace=True)

        if dry_run:
            print("\n=== DRY RUN MODE - No changes will be made to Canvas ===\n")

        subs = self.canvas_quiz.get_submissions()
        for i, sub in enumerate(subs):
            # Get row from quiz_summary where column 'id' matches sub.user_id
            row = quiz_summary[quiz_summary['id'] == sub.user_id]

            # Confirm that the sub.score matches row['score'] using an assert statement
            assert abs(sub.score - row['score'].values[0]) < 0.001

            # Use first-attempt timestamps if available (set by detectPartners), otherwise use latest
            first_times = getattr(self, 'first_attempt_times', {})
            if sub.user_id in first_times:
                fa = first_times[sub.user_id]
                quiz_summary.at[row.index[0], 'start'] = fa.get('start', 'n/a')
                quiz_summary.at[row.index[0], 'finish'] = fa.get('finish', 'n/a')
                quiz_summary.at[row.index[0], 'minutes'] = fa.get('minutes', 0.0)
            else:
                quiz_summary.at[row.index[0], 'start'] = sub.started_at
                quiz_summary.at[row.index[0], 'finish'] = sub.finished_at
                quiz_summary.at[row.index[0], 'minutes'] = sub.time_spent / 60.0

            # Check if bonus needs to be added
            if row['bonus'].values[0] > 0:

                if dry_run:
                    student_name = quiz_summary.at[row.index[0], 'name']
                    bonus_pts = row['bonus'].values[0]
                    print(f"  [DRY RUN] Would award {bonus_pts} bonus points to {student_name} (id: {sub.user_id})")
                    logger.info(f"[DRY RUN] Would award {bonus_pts} bonus to student {sub.user_id}")
                else:
                    # Set points before fudge points are added
                    newattributes = {'excused?': True, 'score_before_regrade': sub.score}
                    upd1 = sub.set_attributes(newattributes)

                    # Now set fudge points
                    update_obj = [{'attempt': sub.attempt, 'fudge_points': row['bonus'].values[0]}]
                    sub.update_score_and_comments(quiz_submissions=update_obj)
                    logger.info(f"Awarded {row['bonus'].values[0]} bonus to student {sub.user_id}")

                # Set quiz_summary for this user_id and column 'score_w_bonus' with sub.score + row['bonus']
                quiz_summary.at[row.index[0], 'score_w_bonus'] = row['score'].values[0] + row['bonus'].values[0]

            else:
                quiz_summary.at[row.index[0], 'score_w_bonus'] = row['score'].values[0]

        suffix = "_dryrun" if dry_run else ""
        quiz_summary_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_scores_w_bonus{suffix}_{today_str()}.csv"
        quiz_summary.to_csv(quiz_summary_csv, index=False)

        if dry_run:
            print(f"\n[DRY RUN] No changes were made to Canvas. Review output: {quiz_summary_csv}")

    # 1) Need to consider making sure this can run multiple times without continually adding bonus
    #    points (since if it is run twice, then sub.score might include a previously added bonus) one
    #    way is to look and see if sub.fudge_points is already set, and if so, then don't add bonus points
    # 2) also may want to check when get_submissions does not return a student's highest submission, as
    #    happned with K.O. in 4050, Quiz 1B (spring25), where their attempt=3 was 3.93 but get_submissions
    #    returned attempt=4 with a score of 3.73, this would require getting all of the attempts for a person... see
    #    https://community.canvaslms.com/t5/Archived-Questions/Get-All-Quiz-Submissions-API-not-working/m-p/218389
    #    and  https://canvas.instructure.com/doc/api/quiz_submissions.html
    def reAwardBonusPoints(self, dry_run=False):
        """Re-award bonus points by setting fudge points for the highest submission attempt."""
        if getattr(self, 'df_past_bonus', None) is None:
            raise RuntimeError("getPastBonusCSV() must be called before reAwardBonusPoints()")

        past_bonus = self.df_past_bonus.copy()
        past_bonus['new_score'] = past_bonus['score']
        past_bonus['start_new'] = 'n/a'
        past_bonus['finish_new'] = 'n/a'
        past_bonus['minutes_new'] = -1
        past_bonus['minutes_new'] = past_bonus['minutes_new'].astype(float)
        past_bonus['new_score_w_bonus'] = past_bonus['score_w_bonus']

        if dry_run:
            print("\n=== DRY RUN MODE - No changes will be made to Canvas ===\n")

        subs = self.canvas_quiz.get_submissions()
        for i, sub in enumerate(subs):

            # no bonus to award if user_id not in past_bonus
            if sub.user_id not in past_bonus['id'].values:
                continue

            # Get row from quiz_summary where column 'id' matches sub.user_id
            row = past_bonus[past_bonus['id'] == sub.user_id]
            if self.verbose:
                print(f"name = {past_bonus.at[row.index[0], 'name']}")
                print(f"  sub.user_id = {sub.user_id}, sub.attempt = {sub.attempt}, sub.score = {sub.score}, sub.fudge_points = {sub.fudge_points}")

            # Update past_bonus df for this user_id
            past_bonus.at[row.index[0], 'new_score'] = sub.score
            past_bonus.at[row.index[0], 'start_new'] = sub.started_at
            past_bonus.at[row.index[0], 'finish_new'] = sub.finished_at
            if sub.time_spent is not None:
                past_bonus.at[row.index[0], 'minutes_new'] = sub.time_spent / 60.0
            else:
                past_bonus.at[row.index[0], 'minutes_new'] = -1.0

            # Check bonus was received before and new score is better than past score (w/o bonus)
            if row['bonus'].values[0] > 0 and sub.score > row['score'].values[0]:

                if dry_run:
                    student_name = past_bonus.at[row.index[0], 'name']
                    bonus_pts = row['bonus'].values[0]
                    print(f"  [DRY RUN] Would re-award {bonus_pts} bonus points to {student_name} (id: {sub.user_id})")
                    logger.info(f"[DRY RUN] Would re-award {bonus_pts} bonus to student {sub.user_id}")
                else:
                    # Set points before fudge points are added
                    newattributes = {'excused?': True, 'score_before_regrade': sub.score}
                    upd1 = sub.set_attributes(newattributes)

                    # Now set fudge points
                    update_obj = [{'attempt': sub.attempt, 'fudge_points': row['bonus'].values[0]}]
                    sub.update_score_and_comments(quiz_submissions=update_obj)
                    logger.info(f"Re-awarded {row['bonus'].values[0]} bonus to student {sub.user_id}")

                # Update 'new_score_w_bonus' using sub.score + row['bonus']
                past_bonus.at[row.index[0], 'new_score_w_bonus'] = sub.score + row['bonus'].values[0]

        suffix = "_dryrun" if dry_run else ""
        past_bonus_csv = self.config.data_path / f"{self.config.quiz_prefix}{self.canvas_quiz.id}_scores_w_bonus_new{suffix}_{today_str()}.csv"
        past_bonus.to_csv(past_bonus_csv, index=False)

        if dry_run:
            print(f"\n[DRY RUN] No changes were made to Canvas. Review output: {past_bonus_csv}")
