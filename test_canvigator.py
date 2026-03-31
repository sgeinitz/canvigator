"""Tests for canvigator utility functions and core algorithms."""
import hashlib
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


# ---------------------------------------------------------------------------
# canvigator_utils tests
# ---------------------------------------------------------------------------

class TestTodayStr:
    """Tests for today_str date formatting."""

    def test_format(self):
        """Verify today_str returns YYYYMMDD format."""
        from canvigator_utils import today_str
        result = today_str()
        assert len(result) == 8
        assert result.isdigit()
        # Should match today's actual date
        assert result == datetime.today().strftime('%Y%m%d')


class TestCanvigatorConfig:
    """Tests for CanvigatorConfig path management."""

    def test_default_paths(self):
        """Verify config initializes with cwd-relative data and figures paths."""
        from canvigator_utils import CanvigatorConfig
        config = CanvigatorConfig()
        assert config.data_path == Path.cwd() / "data"
        assert config.figures_path == Path.cwd() / "figures"
        assert config.quiz_prefix == "quiz"

    def test_modify_quiz_prefix(self):
        """Verify modifyQuizPrefix updates the prefix string."""
        from canvigator_utils import CanvigatorConfig
        config = CanvigatorConfig()
        config.modifyQuizPrefix("quiz3_")
        assert config.quiz_prefix == "quiz3_"

    def test_add_course_to_path(self, tmp_path):
        """Verify addCourseToPath appends course dir and creates directories."""
        from canvigator_utils import CanvigatorConfig
        config = CanvigatorConfig()
        config.data_path = tmp_path / "data"
        config.figures_path = tmp_path / "figures"
        config.data_path.mkdir()
        config.figures_path.mkdir()

        config.addCourseToPath("/cs3120_12345")
        assert config.data_path == tmp_path / "data" / "cs3120_12345"
        assert config.figures_path == tmp_path / "figures" / "cs3120_12345"
        assert config.data_path.exists()
        assert config.figures_path.exists()

    def test_add_course_to_path_idempotent(self, tmp_path):
        """Calling addCourseToPath when already at that dir does not nest."""
        from canvigator_utils import CanvigatorConfig
        config = CanvigatorConfig()
        config.data_path = tmp_path / "data" / "cs3120_12345"
        config.figures_path = tmp_path / "figures" / "cs3120_12345"
        config.data_path.mkdir(parents=True)
        config.figures_path.mkdir(parents=True)

        config.addCourseToPath("/cs3120_12345")
        # Should NOT nest: data/cs3120_12345/cs3120_12345
        assert config.data_path == tmp_path / "data" / "cs3120_12345"


class TestPromptForIndex:
    """Tests for prompt_for_index interactive input."""

    def test_valid_input(self):
        """Valid numeric input returns 0-based index."""
        from canvigator_utils import prompt_for_index
        with patch('builtins.input', return_value='2'):
            result = prompt_for_index("Pick: ", 4)
        assert result == 1

    def test_strips_non_digits(self):
        """Non-digit characters are stripped before parsing."""
        from canvigator_utils import prompt_for_index
        with patch('builtins.input', return_value='[ 3 ]'):
            result = prompt_for_index("Pick: ", 4)
        assert result == 2

    def test_retries_on_out_of_range(self):
        """Out-of-range input causes retry until valid."""
        from canvigator_utils import prompt_for_index
        with patch('builtins.input', side_effect=['99', '2']):
            result = prompt_for_index("Pick: ", 4)
        assert result == 1

    def test_retries_on_non_numeric(self):
        """Non-numeric input causes retry until valid."""
        from canvigator_utils import prompt_for_index
        with patch('builtins.input', side_effect=['abc', '1']):
            result = prompt_for_index("Pick: ", 4)
        assert result == 0


# ---------------------------------------------------------------------------
# canvigator_course anonymization tests
# ---------------------------------------------------------------------------

class TestMakeAnonId:
    """Tests for _make_anon_id hashing function."""

    def test_deterministic(self):
        """Same input always produces the same anon ID."""
        from canvigator_course import _make_anon_id
        assert _make_anon_id(12345) == _make_anon_id(12345)

    def test_different_inputs_differ(self):
        """Different student IDs produce different anon IDs."""
        from canvigator_course import _make_anon_id
        assert _make_anon_id(12345) != _make_anon_id(67890)

    def test_max_10_digits(self):
        """Anon ID is at most 10 digits."""
        from canvigator_course import _make_anon_id
        for sid in [1, 999999, 123456789]:
            result = _make_anon_id(sid)
            assert 0 <= result < 10_000_000_000

    def test_matches_sha256(self):
        """Verify the hash matches the expected SHA-256 derivation."""
        from canvigator_course import _make_anon_id
        sid = 42
        expected = int(hashlib.sha256(str(sid).encode()).hexdigest(), 16) % 10_000_000_000
        assert _make_anon_id(sid) == expected


class TestCollectStudentIds:
    """Tests for _collectStudentIds CSV scanning."""

    def test_standard_csv(self, tmp_path):
        """Collects IDs from a standard name/id CSV."""
        from canvigator_course import _collectStudentIds
        csv_file = tmp_path / "students.csv"
        df = pd.DataFrame({'name': ['Alice', 'Bob'], 'id': [100, 200]})
        df.to_csv(csv_file, index=False)

        result = _collectStudentIds([csv_file])
        assert 100 in result
        assert 200 in result
        assert result[100]['name'] == 'Alice'

    def test_pairing_csv(self, tmp_path):
        """Collects IDs from pairing-style person1/id1 columns."""
        from canvigator_course import _collectStudentIds
        csv_file = tmp_path / "pairings.csv"
        df = pd.DataFrame({
            'person1': ['Alice'], 'id1': [100],
            'person2': ['Bob'], 'id2': [200],
            'person3': [None], 'id3': [-1],
            'distance': [1.5]
        })
        df.to_csv(csv_file, index=False)

        result = _collectStudentIds([csv_file])
        assert 100 in result
        assert 200 in result
        # id3=-1 should be excluded
        assert -1 not in result

    def test_sis_id_preserved(self, tmp_path):
        """sis_id column is captured when present."""
        from canvigator_course import _collectStudentIds
        csv_file = tmp_path / "with_sis.csv"
        df = pd.DataFrame({'name': ['Alice'], 'id': [100], 'sis_id': ['A001']})
        df.to_csv(csv_file, index=False)

        result = _collectStudentIds([csv_file])
        assert result[100]['sis_id'] == 'A001'

    def test_multiple_files_dedup(self, tmp_path):
        """Same student appearing in multiple files is not duplicated."""
        from canvigator_course import _collectStudentIds
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        pd.DataFrame({'name': ['Alice'], 'id': [100]}).to_csv(f1, index=False)
        pd.DataFrame({'name': ['Alice'], 'id': [100]}).to_csv(f2, index=False)

        result = _collectStudentIds([f1, f2])
        assert len(result) == 1


class TestAnonymizeCsvFile:
    """Tests for _anonymizeCsvFile anonymization logic."""

    def test_standard_file(self, tmp_path):
        """Standard CSV: name/id replaced with anon_id, name and id columns dropped."""
        from canvigator_course import _anonymizeCsvFile
        csv_file = tmp_path / "input.csv"
        anon_dir = tmp_path / "anon"
        anon_dir.mkdir()

        df = pd.DataFrame({'name': ['Alice', 'Bob'], 'id': [100, 200], 'score': [95, 87]})
        df.to_csv(csv_file, index=False)

        id_to_anon = {100: 1111111111, 200: 2222222222}
        modified = _anonymizeCsvFile(csv_file, anon_dir, id_to_anon)

        assert modified is True
        result = pd.read_csv(anon_dir / "input.csv")
        assert 'name' not in result.columns
        assert 'id' not in result.columns
        assert 'anon_id' in result.columns
        assert list(result['anon_id']) == [1111111111, 2222222222]
        assert list(result['score']) == [95, 87]

    def test_pairing_file(self, tmp_path):
        """Pairing CSV: person1/id1 columns replaced with anon_id1."""
        from canvigator_course import _anonymizeCsvFile
        csv_file = tmp_path / "pairs.csv"
        anon_dir = tmp_path / "anon"
        anon_dir.mkdir()

        df = pd.DataFrame({
            'person1': ['Alice'], 'id1': [100],
            'person2': ['Bob'], 'id2': [200],
            'distance': [1.5]
        })
        df.to_csv(csv_file, index=False)

        id_to_anon = {100: 1111111111, 200: 2222222222}
        modified = _anonymizeCsvFile(csv_file, anon_dir, id_to_anon)

        assert modified is True
        result = pd.read_csv(anon_dir / "pairs.csv")
        assert 'person1' not in result.columns
        assert 'id1' not in result.columns
        assert 'anon_id1' in result.columns
        assert 'anon_id2' in result.columns

    def test_unrelated_file_not_modified(self, tmp_path):
        """CSV without name/id or pairing columns passes through unmodified."""
        from canvigator_course import _anonymizeCsvFile
        csv_file = tmp_path / "other.csv"
        anon_dir = tmp_path / "anon"
        anon_dir.mkdir()

        df = pd.DataFrame({'metric': ['latency'], 'value': [42]})
        df.to_csv(csv_file, index=False)

        id_to_anon = {}
        modified = _anonymizeCsvFile(csv_file, anon_dir, id_to_anon)

        assert modified is False
        result = pd.read_csv(anon_dir / "other.csv")
        assert list(result.columns) == ['metric', 'value']


# ---------------------------------------------------------------------------
# canvigator_quiz algorithm tests
# ---------------------------------------------------------------------------

class TestGroupPartnerEdges:
    """Tests for _groupPartnerEdges union-find grouping."""

    def _make_quiz(self):
        """Create a minimal CanvigatorQuiz-like object with just the method we need."""
        from canvigator_quiz import CanvigatorQuiz

        class Dummy:
            pass

        dummy = Dummy()
        dummy._groupPartnerEdges = CanvigatorQuiz._groupPartnerEdges.__get__(dummy)
        return dummy

    def test_single_pair(self):
        """Single edge produces one group with two members."""
        quiz = self._make_quiz()
        edges = [(1, 2, 0.9, 0.85)]
        groups = quiz._groupPartnerEdges([1, 2, 3], edges)
        assert len(groups) == 1
        group = list(groups.values())[0]
        assert group['members'] == {1, 2}

    def test_two_disjoint_pairs(self):
        """Two separate edges produce two groups."""
        quiz = self._make_quiz()
        edges = [(1, 2, 0.9, 0.85), (3, 4, 0.95, 0.9)]
        groups = quiz._groupPartnerEdges([1, 2, 3, 4], edges)
        assert len(groups) == 2

    def test_triple_from_chain(self):
        """Edges (1,2) and (2,3) merge into a single group of three."""
        quiz = self._make_quiz()
        edges = [(1, 2, 0.9, 0.85), (2, 3, 0.9, 0.85)]
        groups = quiz._groupPartnerEdges([1, 2, 3], edges)
        assert len(groups) == 1
        group = list(groups.values())[0]
        assert group['members'] == {1, 2, 3}

    def test_no_edges(self):
        """No edges produces no groups."""
        quiz = self._make_quiz()
        groups = quiz._groupPartnerEdges([1, 2, 3], [])
        assert len(groups) == 0


class TestFindMatchingPairs:
    """Tests for _findMatchingPairs score/timestamp comparison."""

    def _make_quiz(self):
        """Create a minimal object with the _findMatchingPairs method."""
        from canvigator_quiz import CanvigatorQuiz

        class Dummy:
            pass

        dummy = Dummy()
        dummy._findMatchingPairs = CanvigatorQuiz._findMatchingPairs.__get__(dummy)
        return dummy

    def test_identical_students_detected(self):
        """Two students with identical scores and close timestamps are paired."""
        quiz = self._make_quiz()
        base = datetime(2024, 1, 1, 10, 0, 0)
        student_scores = {
            1: {1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 0.0},
            2: {1: 1.0, 2: 0.0, 3: 1.0, 4: 1.0, 5: 0.0},
        }
        # Need >= n_questions * time_overlap_threshold timestamps to match (5 * 0.8 = 4)
        from datetime import timedelta
        student_timestamps = {
            1: [base + timedelta(seconds=i) for i in range(5)],
            2: [base + timedelta(seconds=i) for i in range(5)],
        }
        edges = quiz._findMatchingPairs(
            [1, 2], student_scores, student_timestamps,
            n_questions=5, score_threshold=0.8,
            time_threshold_secs=10, time_overlap_threshold=0.8
        )
        assert len(edges) == 1
        assert edges[0][0] == 1 and edges[0][1] == 2

    def test_different_students_not_paired(self):
        """Students with very different scores are not paired."""
        quiz = self._make_quiz()
        base = datetime(2024, 1, 1, 10, 0, 0)
        student_scores = {
            1: {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0},
            2: {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
        }
        student_timestamps = {
            1: [base],
            2: [base],
        }
        edges = quiz._findMatchingPairs(
            [1, 2], student_scores, student_timestamps,
            n_questions=5, score_threshold=0.8,
            time_threshold_secs=10, time_overlap_threshold=0.8
        )
        assert len(edges) == 0

    def test_close_scores_distant_times_not_paired(self):
        """Matching scores but timestamps too far apart are not paired."""
        quiz = self._make_quiz()
        from datetime import timedelta
        base = datetime(2024, 1, 1, 10, 0, 0)
        student_scores = {
            1: {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0},
            2: {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0},
        }
        # Timestamps 1 hour apart
        student_timestamps = {
            1: [base],
            2: [base + timedelta(hours=1)],
        }
        edges = quiz._findMatchingPairs(
            [1, 2], student_scores, student_timestamps,
            n_questions=5, score_threshold=0.8,
            time_threshold_secs=10, time_overlap_threshold=0.8
        )
        assert len(edges) == 0


class TestCreateStudentPairings:
    """Tests for createStudentPairings pairing algorithm."""

    def _make_quiz_with_matrix(self, dist_matrix):
        """Create a minimal object with createStudentPairings and a distance matrix."""
        from canvigator_quiz import CanvigatorQuiz

        class Dummy:
            pass

        dummy = Dummy()
        dummy.createStudentPairings = CanvigatorQuiz.createStudentPairings.__get__(dummy)
        dummy.dist_matrix = dist_matrix
        dummy.verbose = False
        dummy.writePairingsCSV = lambda *a, **k: None
        return dummy

    def test_four_students_produces_two_pairs(self):
        """Four students yield exactly two pairs."""
        ids = [1, 2, 3, 4]
        dm = pd.DataFrame(0.0, index=ids, columns=ids)
        # Set some distances
        dm.loc[1, 2] = dm.loc[2, 1] = 3.0
        dm.loc[1, 3] = dm.loc[3, 1] = 1.0
        dm.loc[1, 4] = dm.loc[4, 1] = 2.0
        dm.loc[2, 3] = dm.loc[3, 2] = 2.5
        dm.loc[2, 4] = dm.loc[4, 2] = 1.5
        dm.loc[3, 4] = dm.loc[4, 3] = 2.0

        quiz = self._make_quiz_with_matrix(dm)
        pairs = quiz.createStudentPairings(method='med', write_csv=False)
        assert len(pairs) == 2

        # Every student appears exactly once
        all_ids = set()
        for p in pairs:
            for member in p[:-1]:  # last element is distance
                all_ids.add(member)
        assert all_ids == {1, 2, 3, 4}

    def test_odd_students_produces_one_triple(self):
        """Five students yield two pairs where one is a triple."""
        ids = [1, 2, 3, 4, 5]
        dm = pd.DataFrame(1.0, index=ids, columns=ids)
        for i in ids:
            dm.loc[i, i] = 0.0
        # Make one pair very strong
        dm.loc[1, 2] = dm.loc[2, 1] = 3.0
        dm.loc[3, 4] = dm.loc[4, 3] = 2.5

        quiz = self._make_quiz_with_matrix(dm)
        pairs = quiz.createStudentPairings(method='med', write_csv=False)
        assert len(pairs) == 2

        # One pair should have 3 members (triple) and one has 2
        sizes = sorted(len(p) - 1 for p in pairs)  # subtract 1 for distance element
        assert sizes == [2, 3]

    def test_raises_without_matrix(self):
        """Calling without distance matrix raises RuntimeError."""
        from canvigator_quiz import CanvigatorQuiz

        class Dummy:
            pass

        dummy = Dummy()
        dummy.createStudentPairings = CanvigatorQuiz.createStudentPairings.__get__(dummy)
        dummy.dist_matrix = None
        dummy.verbose = False
        with pytest.raises(RuntimeError, match="generateDistanceMatrix"):
            dummy.createStudentPairings(method='med', write_csv=False)

    def test_all_methods_valid(self):
        """All four pairing methods (med, max, min, rand) produce valid results."""
        ids = [1, 2, 3, 4]
        dm = pd.DataFrame(0.0, index=ids, columns=ids)
        dm.loc[1, 2] = dm.loc[2, 1] = 3.0
        dm.loc[1, 3] = dm.loc[3, 1] = 1.0
        dm.loc[1, 4] = dm.loc[4, 1] = 2.0
        dm.loc[2, 3] = dm.loc[3, 2] = 2.5
        dm.loc[2, 4] = dm.loc[4, 2] = 1.5
        dm.loc[3, 4] = dm.loc[4, 3] = 2.0

        for method in ['med', 'max', 'min', 'rand']:
            quiz = self._make_quiz_with_matrix(dm)
            pairs = quiz.createStudentPairings(method=method, write_csv=False)
            assert len(pairs) == 2
            all_ids = set()
            for p in pairs:
                for member in p[:-1]:
                    all_ids.add(member)
            assert all_ids == {1, 2, 3, 4}, f"Method {method} didn't pair all students"

    def test_invalid_method_raises(self):
        """Invalid method name raises ValueError."""
        ids = [1, 2, 3, 4]
        dm = pd.DataFrame(1.0, index=ids, columns=ids)
        for i in ids:
            dm.loc[i, i] = 0.0

        quiz = self._make_quiz_with_matrix(dm)
        with pytest.raises(ValueError, match="invalid method"):
            quiz.createStudentPairings(method='bogus', write_csv=False)


class TestSelectCSVFromList:
    """Tests for selectCSVFromList file selection."""

    def test_finds_matching_files(self, tmp_path):
        """Selects matching CSV by keyword and user input."""
        from canvigator_utils import selectCSVFromList
        (tmp_path / "quiz1_present_20240101.csv").write_text("a,b\n1,2\n")
        (tmp_path / "quiz1_present_20240102.csv").write_text("a,b\n1,2\n")
        (tmp_path / "quiz1_scores.csv").write_text("a,b\n1,2\n")

        with patch('canvigator_utils.prompt_for_index', return_value=0):
            result = selectCSVFromList(str(tmp_path), 'present', "Pick: ")
        assert result.name == "quiz1_present_20240101.csv"

    def test_no_match_raises(self, tmp_path):
        """Raises FileNotFoundError when no files match keyword."""
        from canvigator_utils import selectCSVFromList
        (tmp_path / "unrelated.csv").write_text("a\n1\n")

        with pytest.raises(FileNotFoundError, match="No CSV files"):
            selectCSVFromList(str(tmp_path), 'present', "Pick: ")
