"""Tests for canvigator utility functions and core algorithms."""
import hashlib
import subprocess
import pytest
import pandas as pd
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


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


class _StubQuiz:
    """Minimal stand-in for a canvasapi Quiz object."""

    def __init__(self, published=True, due_at=None):
        """Set published flag and due_at string for the stub."""
        self.published = published
        self.due_at = due_at


class TestIsQuizOpenForReminder:
    """Tests for is_quiz_open_for_reminder filter predicate."""

    def _now(self):
        """Return a frozen UTC reference timestamp for deterministic tests."""
        from datetime import timezone
        return datetime(2026, 4, 24, 12, 0, 0, tzinfo=timezone.utc)

    def test_published_with_future_due(self):
        """Published quiz with due_at after now returns True."""
        from canvigator_utils import is_quiz_open_for_reminder
        quiz = _StubQuiz(published=True, due_at="2026-04-25T12:00:00Z")
        assert is_quiz_open_for_reminder(quiz, now=self._now()) is True

    def test_published_with_past_due(self):
        """Published quiz with due_at before now returns False."""
        from canvigator_utils import is_quiz_open_for_reminder
        quiz = _StubQuiz(published=True, due_at="2026-04-23T12:00:00Z")
        assert is_quiz_open_for_reminder(quiz, now=self._now()) is False

    def test_unpublished_with_future_due(self):
        """Unpublished quiz returns False even with future due_at."""
        from canvigator_utils import is_quiz_open_for_reminder
        quiz = _StubQuiz(published=False, due_at="2026-04-25T12:00:00Z")
        assert is_quiz_open_for_reminder(quiz, now=self._now()) is False

    def test_due_at_none(self):
        """Quiz with due_at=None returns False."""
        from canvigator_utils import is_quiz_open_for_reminder
        quiz = _StubQuiz(published=True, due_at=None)
        assert is_quiz_open_for_reminder(quiz, now=self._now()) is False

    def test_due_at_empty_string(self):
        """Quiz with empty due_at returns False."""
        from canvigator_utils import is_quiz_open_for_reminder
        quiz = _StubQuiz(published=True, due_at="")
        assert is_quiz_open_for_reminder(quiz, now=self._now()) is False

    def test_due_at_malformed(self):
        """Malformed due_at strings are treated as not-open rather than raising."""
        from canvigator_utils import is_quiz_open_for_reminder
        quiz = _StubQuiz(published=True, due_at="not-a-date")
        assert is_quiz_open_for_reminder(quiz, now=self._now()) is False

    def test_due_at_exactly_now(self):
        """due_at equal to now is not strictly future, so returns False."""
        from canvigator_utils import is_quiz_open_for_reminder
        quiz = _StubQuiz(published=True, due_at="2026-04-24T12:00:00Z")
        assert is_quiz_open_for_reminder(quiz, now=self._now()) is False


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

    def test_gradebook_csv(self, tmp_path):
        """Collects IDs from gradebook-style name/id columns."""
        from canvigator_course import _collectStudentIds
        csv_file = tmp_path / "gradebook.csv"
        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'sortable_name': ['Smith, Alice', 'Jones, Bob'],
            'id': [100, 200],
            'assignment_name': ['Quiz 1', 'Quiz 1'],
            'score': [95, 87],
        })
        df.to_csv(csv_file, index=False)

        result = _collectStudentIds([csv_file])
        assert 100 in result
        assert 200 in result
        assert result[100]['name'] == 'Alice'


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

    def test_gradebook_file(self, tmp_path):
        """Gradebook CSV: name/sortable_name/id replaced with anon_id."""
        from canvigator_course import _anonymizeCsvFile
        csv_file = tmp_path / "gradebook_20240101.csv"
        anon_dir = tmp_path / "anon"
        anon_dir.mkdir()

        df = pd.DataFrame({
            'name': ['Alice', 'Bob'],
            'sortable_name': ['Smith, Alice', 'Jones, Bob'],
            'id': [100, 200],
            'assignment_name': ['Quiz 1', 'Quiz 1'],
            'assignment_id': [10, 10],
            'points_possible': [100, 100],
            'grade': ['95', '87'],
            'score': [95, 87],
        })
        df.to_csv(csv_file, index=False)

        id_to_anon = {100: 1111111111, 200: 2222222222}
        modified = _anonymizeCsvFile(csv_file, anon_dir, id_to_anon)

        assert modified is True
        result = pd.read_csv(anon_dir / "gradebook_20240101.csv")
        assert 'name' not in result.columns
        assert 'sortable_name' not in result.columns
        assert 'id' not in result.columns
        assert 'anon_id' in result.columns
        assert list(result['anon_id']) == [1111111111, 2222222222]
        assert list(result['score']) == [95, 87]

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


# ---------------------------------------------------------------------------
# canvigator_llm tests
# ---------------------------------------------------------------------------

class TestStripHtml:
    """Tests for _strip_html HTML-to-text conversion."""

    def test_empty(self):
        """Empty/None input yields empty string."""
        from canvigator_llm import _strip_html
        assert _strip_html("") == ""
        assert _strip_html(None) == ""

    def test_strips_tags(self):
        """Tags are removed and entities unescaped."""
        from canvigator_llm import _strip_html
        result = _strip_html("<p>What is <b>2 &amp; 3</b>?</p>")
        assert result == "What is 2 & 3?"

    def test_converts_breaks(self):
        """Break and closing-p tags become newlines."""
        from canvigator_llm import _strip_html
        result = _strip_html("line1<br>line2<br/>line3")
        assert "line1" in result and "line2" in result and "line3" in result
        assert "<br" not in result

    def test_collapses_whitespace(self):
        """Runs of spaces are collapsed."""
        from canvigator_llm import _strip_html
        assert _strip_html("a    b   c") == "a b c"

    def test_handles_nan_and_other_non_strings(self):
        """Non-string inputs (NaN, ints, lists) yield empty string instead of crashing.

        pd.read_csv returns NaN (a float) for empty cells; bool(NaN) is True so
        the historical `if not text` check let NaN through to the regex, which
        crashed with 'expected string or bytes-like object, got float'.
        """
        from canvigator_llm import _strip_html
        assert _strip_html(float('nan')) == ""
        assert _strip_html(0) == ""
        assert _strip_html(123) == ""
        assert _strip_html([]) == ""


class TestParseTags:
    """Tests for _parse_tags response parser."""

    def test_empty(self):
        """Empty response yields empty list."""
        from canvigator_llm import _parse_tags
        assert _parse_tags("") == []
        assert _parse_tags(None) == []

    def test_basic(self):
        """Comma-separated tags are lowercased and stripped."""
        from canvigator_llm import _parse_tags
        assert _parse_tags("Recursion, Base Case, Stack") == ["recursion", "base case", "stack"]

    def test_truncates_to_three(self):
        """Only the first three unique tags are kept."""
        from canvigator_llm import _parse_tags
        result = _parse_tags("a, b, c, d, e")
        assert result == ["a", "b", "c"]

    def test_dedupes(self):
        """Duplicate tags collapse."""
        from canvigator_llm import _parse_tags
        assert _parse_tags("loops, loops, iteration") == ["loops", "iteration"]

    def test_ignores_extra_lines(self):
        """Only the first non-empty line is parsed."""
        from canvigator_llm import _parse_tags
        assert _parse_tags("recursion, pointers\nExplanation: ...") == ["recursion", "pointers"]

    def test_strips_quotes(self):
        """Wrapping quotes are stripped."""
        from canvigator_llm import _parse_tags
        assert _parse_tags('"big-o", "sorting"') == ["big-o", "sorting"]


# ---------------------------------------------------------------------------
# canvigator_quiz._render_missed_bullets tests
# ---------------------------------------------------------------------------

class TestRenderMissedBullets:
    """Tests for the pure _render_missed_bullets helper."""

    def _question_info(self):
        """Return a simple question_info mapping used across tests."""
        return {
            101: {'position': 1, 'keywords': 'recursion, base case', 'question_name': 'Q1'},
            102: {'position': 2, 'keywords': 'big-o, sorting', 'question_name': 'Q2'},
            103: {'position': 3, 'keywords': 'pointers', 'question_name': 'Q3'},
        }

    def test_returns_none_when_empty(self):
        """Empty input returns None."""
        from canvigator_quiz import _render_missed_bullets
        assert _render_missed_bullets([], self._question_info()) is None

    def test_renders_bullet_lines_in_position_order(self):
        """Bullets are sorted by position and include keywords + score."""
        from canvigator_quiz import _render_missed_bullets
        rows = [
            {'question_id': 102, 'points': 0.0, 'points_possible': 1.0},
            {'question_id': 101, 'points': 0.5, 'points_possible': 1.0},
        ]
        result = _render_missed_bullets(rows, self._question_info())
        assert result is not None
        assert result.startswith("\n\nThe questions that you missed on this most recent attempt")
        lines = result.strip().splitlines()
        # Header + 2 bullets
        assert lines[-2] == "• Q1: recursion, base case (0.50 / 1.00 pts)"
        assert lines[-1] == "• Q2: big-o, sorting (0.00 / 1.00 pts)"

    def test_skips_unknown_question_ids(self):
        """Rows whose question_id isn't in question_info are skipped with a warning."""
        from canvigator_quiz import _render_missed_bullets
        rows = [
            {'question_id': 999, 'points': 0.0, 'points_possible': 1.0},
            {'question_id': 103, 'points': 0.25, 'points_possible': 1.0},
        ]
        result = _render_missed_bullets(rows, self._question_info())
        assert result is not None
        assert "pointers" in result
        assert "999" not in result

    def test_returns_none_when_all_unknown(self):
        """If every row is skipped, the helper returns None (no empty section)."""
        from canvigator_quiz import _render_missed_bullets
        rows = [{'question_id': 999, 'points': 0.0, 'points_possible': 1.0}]
        assert _render_missed_bullets(rows, self._question_info()) is None

    def test_handles_string_question_id(self):
        """question_id stored as a string (e.g. from CSV round-trip) still joins."""
        from canvigator_quiz import _render_missed_bullets
        rows = [{'question_id': '102', 'points': 0.0, 'points_possible': 1.0}]
        result = _render_missed_bullets(rows, self._question_info())
        assert result is not None
        assert "big-o" in result

    def test_score_formatting_uses_two_decimals(self):
        """Scores are formatted with exactly two decimals."""
        from canvigator_quiz import _render_missed_bullets
        rows = [{'question_id': 101, 'points': 0.6666, 'points_possible': 1.0}]
        result = _render_missed_bullets(rows, self._question_info())
        assert "Q1:" in result
        assert "(0.67 / 1.00 pts)" in result


# ---------------------------------------------------------------------------
# canvigator_quiz._render_blur_bullets tests
# ---------------------------------------------------------------------------

class TestRenderBlurBullets:
    """Tests for the pure _render_blur_bullets helper."""

    def _question_info(self):
        """Return a simple question_info mapping used across tests."""
        return {
            101: {'position': 1, 'keywords': 'recursion, base case'},
            102: {'position': 2, 'keywords': 'big-o, sorting'},
            103: {'position': 3, 'keywords': 'pointers'},
        }

    def test_returns_none_when_empty(self):
        """Empty input returns None."""
        from canvigator_quiz import _render_blur_bullets
        assert _render_blur_bullets(set(), self._question_info()) is None

    def test_renders_bullets_in_position_order(self):
        """Bullets are sorted by position and include Q number and keywords."""
        from canvigator_quiz import _render_blur_bullets
        result = _render_blur_bullets({102, 101}, self._question_info())
        assert result is not None
        assert "changed window focus" in result
        lines = result.strip().splitlines()
        assert lines[-2] == "• Q1: recursion, base case"
        assert lines[-1] == "• Q2: big-o, sorting"

    def test_skips_unknown_question_ids(self):
        """Question IDs not in question_info are skipped."""
        from canvigator_quiz import _render_blur_bullets
        result = _render_blur_bullets({999, 103}, self._question_info())
        assert result is not None
        assert "pointers" in result
        assert "999" not in result

    def test_returns_none_when_all_unknown(self):
        """If every question ID is unknown, returns None."""
        from canvigator_quiz import _render_blur_bullets
        assert _render_blur_bullets({999}, self._question_info()) is None

    def test_no_points_in_output(self):
        """Blur bullets should not include point scores."""
        from canvigator_quiz import _render_blur_bullets
        result = _render_blur_bullets({101}, self._question_info())
        assert "/ 1.00 pts)" not in result


# ---------------------------------------------------------------------------
# canvigator_llm._parse_question_mode tests
# ---------------------------------------------------------------------------

class TestParseQuestionMode:
    """Tests for _parse_question_mode classify response parser."""

    def test_returns_explain_for_explain(self):
        """Recognises 'explain' response."""
        from canvigator_llm import _parse_question_mode
        assert _parse_question_mode("explain") == "explain"

    def test_returns_draw_for_draw(self):
        """Recognises 'draw' response."""
        from canvigator_llm import _parse_question_mode
        assert _parse_question_mode("draw") == "draw"

    def test_case_insensitive(self):
        """Handles uppercase or mixed-case responses."""
        from canvigator_llm import _parse_question_mode
        assert _parse_question_mode("DRAW") == "draw"
        assert _parse_question_mode("Explain") == "explain"

    def test_strips_quotes_and_punctuation(self):
        """Strips surrounding quotes and trailing punctuation."""
        from canvigator_llm import _parse_question_mode
        assert _parse_question_mode('"draw"') == "draw"
        assert _parse_question_mode("'explain'.") == "explain"

    def test_defaults_to_explain_on_empty(self):
        """Empty/None defaults to explain."""
        from canvigator_llm import _parse_question_mode
        assert _parse_question_mode("") == "explain"
        assert _parse_question_mode(None) == "explain"

    def test_defaults_to_explain_on_unexpected(self):
        """Unexpected text defaults to explain."""
        from canvigator_llm import _parse_question_mode
        assert _parse_question_mode("I think you should draw it") == "explain"

    def test_uses_first_line_only(self):
        """Only the first line is considered."""
        from canvigator_llm import _parse_question_mode
        assert _parse_question_mode("draw\nBecause it is visual...") == "draw"


# ---------------------------------------------------------------------------
# canvigator_llm._build_classify_prompt tests
# ---------------------------------------------------------------------------

class TestBuildClassifyPrompt:
    """Tests for _build_classify_prompt construction."""

    def test_includes_keywords(self):
        """Keywords appear in the prompt."""
        from canvigator_llm import _build_classify_prompt
        result = _build_classify_prompt("linked lists", "What is a linked list?", None)
        assert "Topic keywords: linked lists" in result

    def test_ends_with_classify_cue(self):
        """The prompt ends with the classification cue."""
        from canvigator_llm import _build_classify_prompt
        result = _build_classify_prompt("sorting", "How does quicksort work?", None)
        assert result.endswith("Best assessment mode (explain or draw):")


# ---------------------------------------------------------------------------
# canvigator_llm._build_open_ended_prompt tests
# ---------------------------------------------------------------------------

class TestBuildOpenEndedPrompt:
    """Tests for _build_open_ended_prompt prompt construction."""

    def test_includes_keywords(self):
        """Keywords appear in the prompt under the Topic keywords label."""
        from canvigator_llm import _build_open_ended_prompt
        result = _build_open_ended_prompt("recursion, base case", "What is recursion?", None, "explain")
        assert "Topic keywords: recursion, base case" in result

    def test_includes_question_text(self):
        """Question text is stripped of HTML and included."""
        from canvigator_llm import _build_open_ended_prompt
        result = _build_open_ended_prompt("loops", "<p>What does a <b>for</b> loop do?</p>", None, "explain")
        assert "What does a for loop do?" in result
        assert "<p>" not in result

    def test_includes_answer_choices(self):
        """Answer choices from JSON are included."""
        import json
        from canvigator_llm import _build_open_ended_prompt
        answers = json.dumps([{"text": "O(n)"}, {"text": "O(n^2)"}])
        result = _build_open_ended_prompt("big-o", "What is the complexity?", answers, "explain")
        assert "O(n)" in result
        assert "O(n^2)" in result

    def test_explain_mode_cue(self):
        """Explain mode ends with the oral explanation cue."""
        from canvigator_llm import _build_open_ended_prompt
        result = _build_open_ended_prompt("sorting", "How does quicksort work?", None, "explain")
        assert result.endswith('Oral explanation question (must start with "Explain"):')

    def test_draw_mode_cue(self):
        """Draw mode ends with the visual assessment cue."""
        from canvigator_llm import _build_open_ended_prompt
        result = _build_open_ended_prompt("trees", "What is a binary tree?", None, "draw")
        assert result.endswith('Visual assessment question (must start with "Draw a diagram" or "Draw a figure"):')

    def test_handles_empty_inputs(self):
        """Empty/None inputs produce a minimal prompt with just the cue."""
        from canvigator_llm import _build_open_ended_prompt
        result = _build_open_ended_prompt("", "", None, "explain")
        assert "Oral explanation question" in result


class TestGenerateOpenEndedQuestionsRecovery:
    """generate_open_ended_questions must survive a per-row LLM failure with partial output."""

    def test_one_failing_row_does_not_kill_the_loop(self):
        """A row that raises mid-pipeline yields placeholder rows; other rows succeed normally."""
        import canvigator_llm
        rows = [
            {'question_id': 1, 'question_name': 'Q1', 'question_text': '<p>Q1</p>', 'keywords': 'a', 'position': 1, 'answers': '[]'},
            {'question_id': 2, 'question_name': float('nan'), 'question_text': float('nan'), 'keywords': 'b', 'position': 2, 'answers': '[]'},
            {'question_id': 3, 'question_name': 'Q3', 'question_text': '<p>Q3</p>', 'keywords': 'c', 'position': 3, 'answers': '[]'},
        ]

        def fake_classify(row, _client, _model):
            # Simulate a transient API failure on the bad row.
            if row.get('question_id') == 2:
                raise RuntimeError("simulated API error on question 2")
            return 'explain'

        def fake_candidates(_row, _client, _model, _mode, n=3):
            return [f'candidate {i + 1}' for i in range(n)]

        def fake_guide(_row, _client, _model, _mode, _cand):
            return 'guide'

        def fake_rubric(_row, _client, _model, _mode, _cand):
            return canvigator_llm._empty_rubric('explain')

        def fake_exemplars(_row, _client, _model, _mode, _cand, _rub):
            return dict(canvigator_llm._EMPTY_EXEMPLARS)

        with patch.object(canvigator_llm, '_make_client', return_value=MagicMock()), \
             patch.object(canvigator_llm, 'classify_question_mode', side_effect=fake_classify), \
             patch.object(canvigator_llm, 'generate_open_ended_candidates', side_effect=fake_candidates), \
             patch.object(canvigator_llm, 'generate_assessment_guide', side_effect=fake_guide), \
             patch.object(canvigator_llm, 'generate_structured_rubric', side_effect=fake_rubric), \
             patch.object(canvigator_llm, 'generate_exemplars', side_effect=fake_exemplars):
            results = canvigator_llm.generate_open_ended_questions(rows, n=3)

        # 3 questions × 3 candidates = 9 rows total, regardless of failure.
        assert len(results) == 9
        # Q1 and Q3 produce real candidates.
        for qid in (1, 3):
            qrows = [r for r in results if r['question_id'] == qid]
            assert all(r['open_ended_question'].startswith('candidate') for r in qrows)
            assert all(r['question_mode'] == 'explain' for r in qrows)
        # Q2 produces 3 placeholder rows: empty open_ended_question, empty mode.
        q2_rows = [r for r in results if r['question_id'] == 2]
        assert len(q2_rows) == 3
        assert all(r['open_ended_question'] == '' for r in q2_rows)
        assert all(r['question_mode'] == '' for r in q2_rows)
        # NaN question_text is coerced to empty string by _strip_html, not a crash.
        assert all(r['original_question_text'] == '' for r in q2_rows)


# ---------------------------------------------------------------------------
# canvigator_llm._parse_candidates tests
# ---------------------------------------------------------------------------

class TestParseCandidates:
    """Tests for _parse_candidates numbered-response parser."""

    def test_empty_and_none(self):
        """Empty string or None returns an empty list."""
        from canvigator_llm import _parse_candidates
        assert _parse_candidates("") == []
        assert _parse_candidates(None) == []

    def test_numbered_dot_prefix(self):
        """Parses '1.' / '2.' / '3.' numbered output."""
        from canvigator_llm import _parse_candidates
        resp = "1. Explain recursion.\n2. Explain the base case.\n3. Explain stack frames."
        assert _parse_candidates(resp) == [
            "Explain recursion.",
            "Explain the base case.",
            "Explain stack frames.",
        ]

    def test_numbered_paren_prefix(self):
        """Parses '1)' / '2)' / '3)' style numbering."""
        from canvigator_llm import _parse_candidates
        resp = "1) First question?\n2) Second question?\n3) Third question?"
        assert _parse_candidates(resp) == [
            "First question?",
            "Second question?",
            "Third question?",
        ]

    def test_bulleted_prefix(self):
        """Parses '- ' and '* ' bulleted lines."""
        from canvigator_llm import _parse_candidates
        resp = "- first\n* second\n- third"
        assert _parse_candidates(resp) == ["first", "second", "third"]

    def test_strips_surrounding_quotes(self):
        """Surrounding single/double quotes are stripped."""
        from canvigator_llm import _parse_candidates
        resp = '1. "Explain A."\n2. \'Explain B.\'\n3. Explain C.'
        assert _parse_candidates(resp) == ["Explain A.", "Explain B.", "Explain C."]

    def test_skips_blank_lines(self):
        """Blank lines between candidates are dropped."""
        from canvigator_llm import _parse_candidates
        resp = "1. One\n\n2. Two\n\n\n3. Three"
        assert _parse_candidates(resp) == ["One", "Two", "Three"]

    def test_truncates_to_n(self):
        """Only the first n candidates are returned."""
        from canvigator_llm import _parse_candidates
        resp = "1. a\n2. b\n3. c\n4. d\n5. e"
        assert _parse_candidates(resp, n=3) == ["a", "b", "c"]

    def test_no_prefix_each_line_is_candidate(self):
        """Plain lines without numbering are still treated as candidates."""
        from canvigator_llm import _parse_candidates
        resp = "Explain one thing.\nExplain another thing.\nExplain a third thing."
        assert _parse_candidates(resp) == [
            "Explain one thing.",
            "Explain another thing.",
            "Explain a third thing.",
        ]

    def test_fewer_than_n_candidates(self):
        """Returns whatever was generated even if fewer than n."""
        from canvigator_llm import _parse_candidates
        resp = "1. Only one."
        assert _parse_candidates(resp) == ["Only one."]


# ---------------------------------------------------------------------------
# canvigator_quiz: follow-up question helper tests
# ---------------------------------------------------------------------------

def _make_subs_by_question_df(rows):
    """Build a DataFrame matching the all_subs_by_question schema."""
    return pd.DataFrame(rows, columns=['name', 'id', 'attempt', 'question', 'question_id', 'points', 'points_possible', 'correct'])


def _make_quiz_stub():
    """Return the CanvigatorQuiz class so unbound helper methods can be called with self=None."""
    from canvigator_quiz import CanvigatorQuiz
    return CanvigatorQuiz


class TestClassifyStudentsByQuestionResult:
    """Tests for CanvigatorQuiz._classifyStudentsByQuestionResult."""

    def _call(self, question_id, subs_rows, question_info):
        """Helper to call the classifier without a full quiz instance."""
        cls = _make_quiz_stub()
        df = _make_subs_by_question_df(subs_rows)
        return cls._classifyStudentsByQuestionResult(None, question_id, df, question_info)

    def test_classifies_correct_and_missed(self):
        """Each attempter is labeled 'missed' (points < pp) or 'correct' (points == pp)."""
        question_info = {100: {'position': 1, 'keywords': 'topic a', 'points_possible': 1.0}}
        rows = [
            ('A', 1, 1, 1, 100, 0.0, 1.0, False),
            ('B', 2, 1, 1, 100, 1.0, 1.0, True),
            ('C', 3, 1, 1, 100, 0.5, 1.0, False),
        ]
        result = self._call(100, rows, question_info)
        assert result == {1: 'missed', 2: 'correct', 3: 'missed'}

    def test_uses_latest_attempt(self):
        """A later attempt overrides the earlier one (fixed on retry → 'correct')."""
        question_info = {100: {'position': 1, 'keywords': 'topic a', 'points_possible': 1.0}}
        rows = [
            ('A', 1, 1, 1, 100, 0.0, 1.0, False),
            ('A', 1, 2, 1, 100, 1.0, 1.0, True),
        ]
        result = self._call(100, rows, question_info)
        assert result == {1: 'correct'}

    def test_all_correct(self):
        """When every latest attempt scored perfectly, every student is 'correct'."""
        question_info = {100: {'position': 1, 'keywords': 'topic a', 'points_possible': 1.0}}
        rows = [
            ('A', 1, 1, 1, 100, 1.0, 1.0, True),
            ('B', 2, 1, 1, 100, 1.0, 1.0, True),
        ]
        result = self._call(100, rows, question_info)
        assert result == {1: 'correct', 2: 'correct'}

    def test_question_missing_from_info_returns_empty(self):
        """Without a points_possible entry the classifier can't label anyone."""
        rows = [('A', 1, 1, 1, 100, 1.0, 1.0, True)]
        result = self._call(100, rows, question_info={})
        assert result == {}

    def test_no_attempts_returns_empty(self):
        """Students who never attempted the quiz are absent from the result."""
        question_info = {100: {'position': 1, 'keywords': 'topic a', 'points_possible': 1.0}}
        result = self._call(100, [], question_info)
        assert result == {}


# ---------------------------------------------------------------------------
# canvigator_quiz: reply extraction tests
# ---------------------------------------------------------------------------

class TestExtractStudentReplies:
    """Tests for CanvigatorQuiz._extractStudentReplies."""

    def _call(self, messages, instructor_id, sent_at, cutoff):
        """Helper to call _extractStudentReplies without a full quiz instance."""
        from canvigator_quiz import CanvigatorQuiz
        return CanvigatorQuiz._extractStudentReplies(None, messages, instructor_id, sent_at, cutoff)

    def _make_times(self):
        """Return (sent_at, cutoff) spanning a 5-day window for testing."""
        from datetime import datetime, timedelta, timezone
        sent = datetime(2026, 4, 10, 12, 0, 0, tzinfo=timezone.utc)
        cutoff = sent + timedelta(days=5)
        return sent, cutoff

    def test_filters_instructor_messages(self):
        """Messages from the instructor are excluded."""
        sent, cutoff = self._make_times()
        messages = [
            {'id': 1, 'author_id': 999, 'body': 'instructor msg', 'created_at': '2026-04-11T10:00:00Z'},
            {'id': 2, 'author_id': 42, 'body': 'student reply', 'created_at': '2026-04-11T10:00:00Z'},
        ]
        result = self._call(messages, instructor_id=999, sent_at=sent, cutoff=cutoff)
        assert len(result) == 1
        assert result[0]['author_id'] == 42

    def test_filters_messages_before_sent_at(self):
        """Messages created before the follow-up was sent are excluded."""
        sent, cutoff = self._make_times()
        messages = [
            {'id': 1, 'author_id': 42, 'body': 'old msg', 'created_at': '2026-04-09T10:00:00Z'},
        ]
        result = self._call(messages, instructor_id=999, sent_at=sent, cutoff=cutoff)
        assert len(result) == 0

    def test_filters_messages_after_cutoff(self):
        """Messages created after the reply window closes are excluded."""
        sent, cutoff = self._make_times()
        messages = [
            {'id': 1, 'author_id': 42, 'body': 'late msg', 'created_at': '2026-04-20T10:00:00Z'},
        ]
        result = self._call(messages, instructor_id=999, sent_at=sent, cutoff=cutoff)
        assert len(result) == 0

    def test_includes_messages_within_window(self):
        """Messages within the reply window from a student are included."""
        sent, cutoff = self._make_times()
        messages = [
            {'id': 1, 'author_id': 42, 'body': 'reply 1', 'created_at': '2026-04-12T10:00:00Z'},
            {'id': 2, 'author_id': 42, 'body': 'reply 2', 'created_at': '2026-04-13T10:00:00Z'},
        ]
        result = self._call(messages, instructor_id=999, sent_at=sent, cutoff=cutoff)
        assert len(result) == 2

    def test_skips_generated_messages(self):
        """System-generated messages are excluded."""
        sent, cutoff = self._make_times()
        messages = [
            {'id': 1, 'author_id': 42, 'body': 'auto', 'created_at': '2026-04-11T10:00:00Z', 'generated': True},
        ]
        result = self._call(messages, instructor_id=999, sent_at=sent, cutoff=cutoff)
        assert len(result) == 0

    def test_preserves_newest_first_order(self):
        """Canvas returns messages newest-first; this order is preserved."""
        sent, cutoff = self._make_times()
        messages = [
            {'id': 2, 'author_id': 42, 'body': 'newer', 'created_at': '2026-04-13T10:00:00Z'},
            {'id': 1, 'author_id': 42, 'body': 'older', 'created_at': '2026-04-11T10:00:00Z'},
        ]
        result = self._call(messages, instructor_id=999, sent_at=sent, cutoff=cutoff)
        assert result[0]['id'] == 2
        assert result[1]['id'] == 1


class TestAssessmentsMerge:
    """Tests for CanvigatorQuiz._mergeAssessments and _indexAssessments."""

    @property
    def COLS(self):
        """Pull the canonical column list from the production class so tests track it."""
        from canvigator_quiz import CanvigatorQuiz
        return list(CanvigatorQuiz.ASSESSMENTS_COLUMNS)

    def _stub(self):
        """Return a CanvigatorQuiz stand-in carrying just ASSESSMENTS_COLUMNS."""
        from canvigator_quiz import CanvigatorQuiz

        class _Stub:
            ASSESSMENTS_COLUMNS = CanvigatorQuiz.ASSESSMENTS_COLUMNS
            _mergeAssessments = CanvigatorQuiz._mergeAssessments
            _indexAssessments = CanvigatorQuiz._indexAssessments

        return _Stub()

    def _row(self, sid, qid, **overrides):
        """Build a canonical assessments row dict."""
        base = {
            'student_id': sid, 'student_name': f's{sid}', 'question_id': qid,
            'question_mode': 'explain', 'conversation_id': 100 + sid,
            'result': 'pass', 'confidence': 'high', 'feedback': 'fb',
            'transcript': '', 'criteria_evaluations': '',
            'assessed_at': '2026-04-22T00:00:00Z', 'sent_assessment': 0, 'sent_at': '',
        }
        base.update(overrides)
        return base

    def test_merge_into_empty(self):
        """Merging into an empty DataFrame produces all new rows."""
        import pandas as pd
        stub = self._stub()
        new = [self._row(1, 10), self._row(2, 10)]
        out = stub._mergeAssessments(pd.DataFrame(columns=self.COLS), new)
        assert len(out) == 2
        assert set(out.columns) == set(self.COLS)

    def test_merge_replaces_existing_match(self):
        """Rows with matching (student_id, question_id) are replaced, not duplicated."""
        import pandas as pd
        stub = self._stub()
        existing = pd.DataFrame([self._row(1, 10, feedback='OLD')], columns=self.COLS)
        new = [self._row(1, 10, feedback='NEW')]
        out = stub._mergeAssessments(existing, new)
        assert len(out) == 1
        assert out.iloc[0]['feedback'] == 'NEW'

    def test_merge_preserves_unrelated_rows(self):
        """Rows with non-matching keys are kept as-is."""
        import pandas as pd
        stub = self._stub()
        existing = pd.DataFrame(
            [self._row(1, 10, feedback='keep'), self._row(2, 10, feedback='also keep')],
            columns=self.COLS,
        )
        new = [self._row(3, 10, feedback='new')]
        out = stub._mergeAssessments(existing, new)
        assert len(out) == 3
        feedback_by_sid = dict(zip(out['student_id'], out['feedback']))
        assert feedback_by_sid[1] == 'keep'
        assert feedback_by_sid[2] == 'also keep'
        assert feedback_by_sid[3] == 'new'

    def test_index_keys_by_student_and_question(self):
        """_indexAssessments returns a dict keyed by (student_id, question_id)."""
        import pandas as pd
        stub = self._stub()
        df = pd.DataFrame(
            [self._row(1, 10), self._row(1, 11), self._row(2, 10)],
            columns=self.COLS,
        )
        index = stub._indexAssessments(df)
        assert set(index.keys()) == {(1, 10), (1, 11), (2, 10)}


class TestComposeFollowUpFeedbackMessage:
    """Tests for canvigator_quiz._composeFollowUpFeedbackMessage."""

    def _call(self, name, feedback, result):
        """Invoke the module-level helper."""
        from canvigator_quiz import _composeFollowUpFeedbackMessage
        return _composeFollowUpFeedbackMessage(name, feedback, result)

    def test_pass_uses_first_name_and_nice_work(self):
        """Pass result uses 'Nice work!' closing with the first name greeting."""
        out = self._call('Victor Salazar', 'Solid Venn diagram.', 'pass')
        assert out.startswith('Hi Victor,\n\n')
        assert 'Solid Venn diagram.' in out
        assert out.endswith('Nice work!')

    def test_fail_uses_try_again_closing(self):
        """Fail result uses the 'try again' closing."""
        out = self._call('Landon Strong', 'Missing the union step.', 'fail')
        assert out.startswith('Hi Landon,\n\n')
        assert out.endswith('Please give it another try when you get a chance.')

    def test_unknown_result_defaults_to_try_again(self):
        """Empty/unknown result falls through to the 'try again' closing."""
        out = self._call('Sam Lee', 'Some feedback.', '')
        assert out.endswith('Please give it another try when you get a chance.')

    def test_sortable_name_uses_part_after_comma(self):
        """A 'Last, First' style name extracts the first name from after the comma."""
        out = self._call('Salazar, Victor', 'fb', 'pass')
        assert out.startswith('Hi Victor,\n\n')

    def test_empty_name_falls_back_to_there(self):
        """An empty name produces a generic 'Hi there,' greeting."""
        out = self._call('', 'fb', 'pass')
        assert out.startswith('Hi there,\n\n')

    def test_pass_is_case_insensitive(self):
        """Result matching is case-insensitive."""
        out = self._call('Sam', 'fb', 'PASS')
        assert out.endswith('Nice work!')


class TestComposeConversationSubject:
    """Tests for canvigator_quiz._composeConversationSubject."""

    def _call(self, course_code, quiz_name, suffix, short_code=False):
        """Invoke the module-level helper."""
        from canvigator_quiz import _composeConversationSubject
        return _composeConversationSubject(course_code, quiz_name, suffix, short_code=short_code)

    def test_drops_section_from_four_part_code(self):
        """A four-part Canvas code drops the section component (3rd part)."""
        out = self._call('CSI-3300-001-12345', 'Quiz 1', 'Q3 Follow-Up')
        assert out == 'CSI-3300-12345 - Quiz 1 - Q3 Follow-Up'

    def test_reminder_suffix(self):
        """Reminder subjects use the 'Reminder' suffix without a Q-number."""
        out = self._call('CSI-3300-001-12345', 'Quiz 2', 'Reminder')
        assert out == 'CSI-3300-12345 - Quiz 2 - Reminder'

    def test_three_part_code_passes_through(self):
        """A code that already lacks a section component is preserved as-is."""
        out = self._call('CSI-3300-12345', 'Quiz 1', 'Reminder')
        assert out == 'CSI-3300-12345 - Quiz 1 - Reminder'

    def test_missing_course_code_falls_back_to_course_label(self):
        """An empty course code still produces a sensible subject with a 'Course' placeholder."""
        out = self._call('', 'Quiz 1', 'Reminder')
        assert out == 'Course - Quiz 1 - Reminder'

    def test_empty_suffix_is_omitted(self):
        """An empty suffix drops the trailing separator instead of leaving a dangling dash."""
        out = self._call('CSI-3300-12345', 'Quiz 1', '')
        assert out == 'CSI-3300-12345 - Quiz 1'

    def test_short_code_truncates_four_part_code(self):
        """short_code=True truncates CSI-3300-001-12345 to CSI-3300 (up to the second hyphen)."""
        out = self._call('CSI-3300-001-12345', 'Quiz 1', 'Q3 Follow-Up', short_code=True)
        assert out == 'CSI-3300 - Quiz 1 - Q3 Follow-Up'

    def test_short_code_truncates_three_part_code(self):
        """short_code=True truncates CSI-3300-12345 to CSI-3300."""
        out = self._call('CSI-3300-12345', 'Quiz 2', 'Q1 Follow-Up', short_code=True)
        assert out == 'CSI-3300 - Quiz 2 - Q1 Follow-Up'

    def test_short_code_preserves_single_hyphen_code(self):
        """A code with only one hyphen is unchanged (still only two parts)."""
        out = self._call('MATH-101', 'Quiz 1', 'Q1 Follow-Up', short_code=True)
        assert out == 'MATH-101 - Quiz 1 - Q1 Follow-Up'

    def test_short_code_preserves_hyphenless_code(self):
        """A code with no hyphens and no spaces passes through entirely."""
        out = self._call('MATH101', 'Quiz 1', 'Q1 Follow-Up', short_code=True)
        assert out == 'MATH101 - Quiz 1 - Q1 Follow-Up'

    def test_short_code_truncates_long_padded_code(self):
        """A long course_code with embedded ' - ' separators truncates to '<prefix>-<number>'."""
        # Models a Canvas course_code where extra metadata follows the section/CRN
        out = self._call('CS-3120-001 - Spring 2026 - CRN: 32093', 'Quiz 8', 'Reminder', short_code=True)
        assert out == 'CS-3120 - Quiz 8 - Reminder'

    def test_short_code_falls_back_to_space_split(self):
        """When the code has no hyphens, the second space is the truncation point."""
        out = self._call('CS 3120 Spring 2026 CRN 32093', 'Quiz 1', 'Reminder', short_code=True)
        assert out == 'CS 3120 - Quiz 1 - Reminder'

    def test_short_code_single_token_is_unchanged(self):
        """A single token (no hyphens, no spaces) is preserved verbatim."""
        out = self._call('STANDALONE', 'Quiz 1', 'Reminder', short_code=True)
        assert out == 'STANDALONE - Quiz 1 - Reminder'


# ---------------------------------------------------------------------------
# canvigator_quiz: histogram bin helpers
# ---------------------------------------------------------------------------

class TestFmtStat:
    """Tests for canvigator_quiz._fmt_stat."""

    def test_trims_trailing_zeros(self):
        """0.50 renders as '0.5', not '0.50'."""
        from canvigator_quiz import _fmt_stat
        assert _fmt_stat(0.5) == '0.5'

    def test_caps_at_two_decimals(self):
        """0.1234 rounds to '0.12'."""
        from canvigator_quiz import _fmt_stat
        assert _fmt_stat(0.1234) == '0.12'

    def test_integer_value_drops_decimal(self):
        """1.0 renders as '1', not '1.0' or '1.00'."""
        from canvigator_quiz import _fmt_stat
        assert _fmt_stat(1.0) == '1'

    def test_zero(self):
        """0.0 renders as '0'."""
        from canvigator_quiz import _fmt_stat
        assert _fmt_stat(0.0) == '0'

    def test_nan_and_none(self):
        """Renders NaN and None as an em-dash placeholder."""
        from canvigator_quiz import _fmt_stat
        import math as _m
        assert _fmt_stat(float('nan')) == '—'
        assert _fmt_stat(None) == '—'
        assert _fmt_stat(_m.nan) == '—'


class TestScoreHistogramBins:
    """Tests for canvigator_quiz._scoreHistogramBins."""

    def test_width_0_1_when_max_pts_1(self):
        """1-point question uses 0.1-wide bins across [0, 1]."""
        from canvigator_quiz import _scoreHistogramBins
        bins = _scoreHistogramBins(1.0)
        assert len(bins) == 11
        assert bins[0] == 0
        assert bins[-1] == 1.0
        assert abs(bins[1] - 0.1) < 1e-9

    def test_width_0_1_at_threshold_1_5(self):
        """max_pts = 1.5 still uses 0.1-wide bins (≤ 1.5 rule)."""
        from canvigator_quiz import _scoreHistogramBins
        bins = _scoreHistogramBins(1.5)
        assert len(bins) == 16

    def test_width_0_2_when_max_pts_2(self):
        """2-point question uses 0.2-wide bins."""
        from canvigator_quiz import _scoreHistogramBins
        bins = _scoreHistogramBins(2.0)
        assert len(bins) == 11
        assert bins[-1] == 2.0

    def test_width_0_5_when_max_pts_3(self):
        """3-point question uses 0.5-wide bins."""
        from canvigator_quiz import _scoreHistogramBins
        bins = _scoreHistogramBins(3.0)
        assert len(bins) == 7
        assert bins[-1] == 3.0

    def test_fallback_when_max_pts_missing(self):
        """None max_pts falls back to the integer bin count 10."""
        from canvigator_quiz import _scoreHistogramBins
        assert _scoreHistogramBins(None) == 10
        assert _scoreHistogramBins(0) == 10


class TestIntegerAlignedBins:
    """Tests for canvigator_quiz._integerAlignedBins."""

    def test_spans_zero_to_ceil_max_plus_one(self):
        """Edges cover [0, ceil(max)+1] so every integer value has a bucket."""
        from canvigator_quiz import _integerAlignedBins
        bins = _integerAlignedBins({'q1': [0, 1, 2], 'q2': [5]})
        assert bins == [0, 1, 2, 3, 4, 5, 6]

    def test_ceils_float_values(self):
        """Non-integer timing values round up so the max is covered."""
        from canvigator_quiz import _integerAlignedBins
        bins = _integerAlignedBins({'q1': [0.3, 2.7]})
        assert bins == [0, 1, 2, 3, 4]

    def test_returns_none_when_empty(self):
        """Empty data across all questions returns None."""
        from canvigator_quiz import _integerAlignedBins
        assert _integerAlignedBins({'q1': [], 'q2': []}) is None


# ---------------------------------------------------------------------------
# canvigator_llm: assessment helper tests
# ---------------------------------------------------------------------------

class TestBuildAssessmentPrompt:
    """Tests for canvigator_llm._build_assessment_prompt."""

    def test_includes_all_fields(self):
        """All provided fields appear in the prompt."""
        from canvigator_llm import _build_assessment_prompt
        result = _build_assessment_prompt(
            keywords="neural networks",
            open_ended_question="Explain backprop.",
            original_question_text="What is backprop?",
            transcript="It's a way to compute gradients.",
        )
        assert "neural networks" in result
        assert "Explain backprop." in result
        assert "What is backprop?" in result
        assert "It's a way to compute gradients." in result

    def test_omits_empty_fields(self):
        """Empty or None fields are omitted cleanly."""
        from canvigator_llm import _build_assessment_prompt
        result = _build_assessment_prompt(keywords="", open_ended_question="Q?", original_question_text="", transcript=None)
        assert "Topic keywords" not in result
        assert "Original quiz question" not in result
        assert "transcript" not in result.lower()
        assert "Q?" in result

    def test_works_for_draw_mode(self):
        """Prompt without transcript is suitable for draw assessments."""
        from canvigator_llm import _build_assessment_prompt
        result = _build_assessment_prompt(
            keywords="binary trees",
            open_ended_question="Draw a binary search tree.",
            original_question_text="What is a BST?",
            mode='draw',
        )
        assert "binary trees" in result
        assert "transcript" not in result.lower()

    def test_includes_assessment_guide(self):
        """Assessment guide is rendered with its instructor-framing label."""
        from canvigator_llm import _build_assessment_prompt
        result = _build_assessment_prompt(
            keywords="binary search",
            open_ended_question="Explain the time complexity.",
            original_question_text="What is O(log n)?",
            transcript="Binary search halves the search space each step.",
            assessment_guide="Student should mention halving, log n, and sorted input.",
        )
        assert "Assessment guide" in result
        assert "halving, log n, and sorted input" in result

    def test_omits_empty_assessment_guide(self):
        """Empty/None assessment_guide doesn't add a stray label."""
        from canvigator_llm import _build_assessment_prompt
        result = _build_assessment_prompt(
            keywords="k",
            open_ended_question="Q",
            original_question_text="O",
            transcript="t",
            assessment_guide="",
        )
        assert "Assessment guide" not in result

    def test_renders_rubric_block(self):
        """A rubric dict is rendered as bulleted sections in the prompt."""
        from canvigator_llm import _build_assessment_prompt
        rubric = {
            'canonical_answer': 'Halve the search space each step.',
            'model_answer': 'Binary search works by dividing the array in half...',
            'pass_criteria': ['mention halving', 'mention log n'],
            'acceptable_alternatives': ['logarithmic time'],
            'common_misconceptions': ['saying it is O(n)'],
            'fatal_errors': ['claims O(1)'],
        }
        result = _build_assessment_prompt(
            keywords="binary search",
            open_ended_question="Explain the complexity.",
            original_question_text="What is the runtime?",
            transcript="It's logarithmic because we halve each step.",
            rubric=rubric,
        )
        assert "Pass criteria" in result
        assert "mention halving" in result
        assert "mention log n" in result
        assert "Acceptable alternative framings" in result
        assert "logarithmic time" in result
        assert "Common misconceptions" in result
        assert "saying it is O(n)" in result
        assert "Fatal errors" in result
        assert "claims O(1)" in result
        assert "Model answer" in result
        assert "Canonical answer" in result

    def test_renders_required_visual_elements_for_draw(self):
        """Draw-mode rubric surfaces required_visual_elements as a checklist."""
        from canvigator_llm import _build_assessment_prompt
        rubric = {
            'canonical_answer': '',
            'model_answer': '',
            'pass_criteria': [],
            'acceptable_alternatives': [],
            'common_misconceptions': [],
            'fatal_errors': [],
            'required_visual_elements': ['root node labeled', 'three child nodes', 'arrows pointing down'],
        }
        result = _build_assessment_prompt(
            keywords="trees",
            open_ended_question="Draw a tree.",
            original_question_text="What is a tree?",
            rubric=rubric,
            mode='draw',
        )
        assert "Required visual elements" in result
        assert "root node labeled" in result
        assert "three child nodes" in result

    def test_renders_exemplars_block(self):
        """Pass/fail exemplars and notes are rendered as a calibration section."""
        from canvigator_llm import _build_assessment_prompt
        exemplars = {
            'exemplar_pass': 'Um, so binary search like cuts the array in half...',
            'exemplar_pass_note': 'Covers halving and runtime, informal but correct.',
            'exemplar_fail': 'It just goes through every element.',
            'exemplar_fail_note': 'Describes linear search, the wrong algorithm.',
        }
        result = _build_assessment_prompt(
            keywords="binary search",
            open_ended_question="Explain.",
            original_question_text="What is binary search?",
            transcript="...",
            exemplars=exemplars,
        )
        assert "Calibration exemplars" in result
        assert "Passing example" in result
        assert "cuts the array in half" in result
        assert "Failing example" in result
        assert "Describes linear search" in result

    def test_renders_locked_examples(self):
        """Instructor-approved few-shot examples are rendered when supplied."""
        from canvigator_llm import _build_assessment_prompt
        locked = [
            {'response': 'Binary search halves the array each step.', 'result': 'pass',
             'feedback': 'Solid — you got the key idea.'},
            {'response': 'It searches one item at a time.', 'result': 'fail',
             'feedback': 'You described linear search instead.'},
        ]
        result = _build_assessment_prompt(
            keywords="binary search",
            open_ended_question="Explain.",
            original_question_text="What is binary search?",
            transcript="...",
            locked_examples=locked,
        )
        assert "Previously-graded examples" in result
        assert "halves the array each step" in result
        assert "Example 1 verdict: pass" in result
        assert "searches one item at a time" in result
        assert "Example 2 verdict: fail" in result

    def test_omits_locked_examples_when_empty(self):
        """No locked-examples header when the list is empty."""
        from canvigator_llm import _build_assessment_prompt
        result = _build_assessment_prompt(
            keywords="k", open_ended_question="Q", original_question_text="O",
            transcript="t", locked_examples=[],
        )
        assert "Previously-graded examples" not in result


class TestBuildAssessmentGuidePrompt:
    """Tests for canvigator_llm._build_assessment_guide_prompt."""

    def test_includes_all_fields_explain(self):
        """Keywords, original question, mode, and open-ended question all appear."""
        from canvigator_llm import _build_assessment_guide_prompt
        result = _build_assessment_guide_prompt(
            keywords="recursion, base case",
            original_question_text="<p>What is a base case?</p>",
            answers_json=None,
            mode="explain",
            open_ended_question="Explain why every recursive function needs a base case.",
        )
        assert "recursion, base case" in result
        assert "What is a base case?" in result  # HTML stripped
        assert "explain" in result
        assert "Explain why every recursive function needs a base case." in result
        assert "Assessment guide:" in result

    def test_draw_mode_label(self):
        """Draw mode is labeled in the prompt."""
        from canvigator_llm import _build_assessment_guide_prompt
        result = _build_assessment_guide_prompt(
            keywords="linked list",
            original_question_text="What is a linked list?",
            answers_json=None,
            mode="draw",
            open_ended_question="Draw a diagram of a singly-linked list with 3 nodes.",
        )
        assert "draw" in result
        assert "Draw a diagram of a singly-linked list with 3 nodes." in result

    def test_includes_answer_choices(self):
        """Answer labels from the original question appear in the prompt."""
        import json
        from canvigator_llm import _build_assessment_guide_prompt
        answers = json.dumps([{"text": "O(1)"}, {"text": "O(log n)"}, {"text": "O(n)"}])
        result = _build_assessment_guide_prompt(
            keywords="big-o",
            original_question_text="What is the complexity?",
            answers_json=answers,
            mode="explain",
            open_ended_question="Explain the complexity.",
        )
        assert "O(log n)" in result


# ---------------------------------------------------------------------------
# canvigator_llm: structured rubric parser tests (Tier 1B coverage)
# ---------------------------------------------------------------------------

class TestParseStructuredRubric:
    """Tests for canvigator_llm._parse_structured_rubric."""

    def _clean_explain_json(self):
        """Fixture: a minimal valid explain-mode rubric as a JSON string."""
        return (
            '{"canonical_answer": "It runs in log time.",'
            ' "model_answer": "Binary search halves the array each step until found.",'
            ' "pass_criteria": ["mention halving", "mention sorted input"],'
            ' "acceptable_alternatives": ["logarithmic"],'
            ' "common_misconceptions": ["claim O(n)"],'
            ' "fatal_errors": ["claim O(1)"]}'
        )

    def test_parses_clean_explain_json(self):
        """Clean JSON with all explain-mode fields is parsed correctly."""
        from canvigator_llm import _parse_structured_rubric
        out = _parse_structured_rubric(self._clean_explain_json(), 'explain')
        assert out['canonical_answer'] == 'It runs in log time.'
        assert out['model_answer'].startswith('Binary search halves')
        assert out['pass_criteria'] == ['mention halving', 'mention sorted input']
        assert out['acceptable_alternatives'] == ['logarithmic']
        assert out['common_misconceptions'] == ['claim O(n)']
        assert out['fatal_errors'] == ['claim O(1)']
        # Explain mode should NOT have required_visual_elements
        assert 'required_visual_elements' not in out

    def test_handles_markdown_fences(self):
        """JSON wrapped in ```json ... ``` fences is unwrapped before parsing."""
        from canvigator_llm import _parse_structured_rubric
        wrapped = "```json\n" + self._clean_explain_json() + "\n```"
        out = _parse_structured_rubric(wrapped, 'explain')
        assert out['canonical_answer'] == 'It runs in log time.'

    def test_handles_prose_prefix(self):
        """JSON preceded by prose ('Here is the rubric: {...}') is still parsed."""
        from canvigator_llm import _parse_structured_rubric
        prose = "Here is the rubric you asked for: " + self._clean_explain_json() + "\nHope that helps!"
        out = _parse_structured_rubric(prose, 'explain')
        assert out['pass_criteria'] == ['mention halving', 'mention sorted input']

    def test_malformed_json_returns_empty_rubric(self):
        """Garbage in -> empty rubric (no crash, no partial fields)."""
        from canvigator_llm import _parse_structured_rubric
        out = _parse_structured_rubric('not json at all', 'explain')
        assert out['canonical_answer'] == ''
        assert out['model_answer'] == ''
        assert out['pass_criteria'] == []
        assert out['fatal_errors'] == []

    def test_empty_string_returns_empty_rubric(self):
        """Empty input returns the canonical empty rubric."""
        from canvigator_llm import _parse_structured_rubric
        out = _parse_structured_rubric('', 'explain')
        assert out['pass_criteria'] == []

    def test_draw_mode_includes_required_visual_elements(self):
        """Draw mode parses required_visual_elements as a list."""
        from canvigator_llm import _parse_structured_rubric
        draw_json = (
            '{"canonical_answer": "Tree with 3 nodes.",'
            ' "model_answer": "Root at top with two children below.",'
            ' "pass_criteria": ["root labeled"],'
            ' "acceptable_alternatives": [],'
            ' "common_misconceptions": [],'
            ' "fatal_errors": [],'
            ' "required_visual_elements": ["root node", "child nodes", "edges"]}'
        )
        out = _parse_structured_rubric(draw_json, 'draw')
        assert out['required_visual_elements'] == ['root node', 'child nodes', 'edges']

    def test_filters_empty_list_items(self):
        """List entries that are None or empty strings are dropped."""
        from canvigator_llm import _parse_structured_rubric
        json_with_empties = (
            '{"canonical_answer": "X",'
            ' "model_answer": "",'
            ' "pass_criteria": ["good", "", "  ", null, "also good"],'
            ' "acceptable_alternatives": [],'
            ' "common_misconceptions": [],'
            ' "fatal_errors": []}'
        )
        out = _parse_structured_rubric(json_with_empties, 'explain')
        assert out['pass_criteria'] == ['good', 'also good']

    def test_non_dict_root_returns_empty(self):
        """A JSON list (not an object) at the root returns empty rubric."""
        from canvigator_llm import _parse_structured_rubric
        out = _parse_structured_rubric('["just a list"]', 'explain')
        assert out['pass_criteria'] == []


# ---------------------------------------------------------------------------
# canvigator_llm: per-criterion assessment parsing
# ---------------------------------------------------------------------------

class TestParsePerCriterionResponse:
    """Tests for canvigator_llm._parse_per_criterion_response."""

    def test_parses_clean_explain_json(self):
        """A well-formed explain-mode response yields all three sections."""
        from canvigator_llm import _parse_per_criterion_response
        resp = (
            '{"pass_criteria_evaluations": ['
            '  {"criterion": "mention halving", "status": "met"},'
            '  {"criterion": "mention sorted input", "status": "missing"}'
            '],'
            ' "fatal_errors_evaluations": ['
            '  {"error": "claim O(1)", "status": "absent"}'
            '],'
            ' "feedback": "Solid coverage of halving but you missed sorted input."}'
        )
        out = _parse_per_criterion_response(resp, 'explain')
        assert out['pass_criteria_evaluations'] == [
            {'criterion': 'mention halving', 'status': 'met'},
            {'criterion': 'mention sorted input', 'status': 'missing'},
        ]
        assert out['fatal_errors_evaluations'] == [
            {'error': 'claim O(1)', 'status': 'absent'}
        ]
        assert 'sorted input' in out['feedback']
        # explain mode does not have visual_elements
        assert 'visual_elements_evaluations' not in out

    def test_parses_draw_mode_with_visual_elements(self):
        """Draw mode parses visual_elements_evaluations."""
        from canvigator_llm import _parse_per_criterion_response
        resp = (
            '{"pass_criteria_evaluations": [],'
            ' "fatal_errors_evaluations": [],'
            ' "visual_elements_evaluations": ['
            '  {"element": "root node", "status": "yes"},'
            '  {"element": "edges", "status": "no"}'
            '],'
            ' "feedback": "Root is there but no edges."}'
        )
        out = _parse_per_criterion_response(resp, 'draw')
        assert out['visual_elements_evaluations'] == [
            {'element': 'root node', 'status': 'yes'},
            {'element': 'edges', 'status': 'no'},
        ]

    def test_handles_markdown_fences(self):
        """Fenced JSON output is unwrapped before parsing."""
        from canvigator_llm import _parse_per_criterion_response
        resp = (
            '```json\n'
            '{"pass_criteria_evaluations": [{"criterion": "X", "status": "met"}],'
            ' "fatal_errors_evaluations": [],'
            ' "feedback": "ok"}'
            '\n```'
        )
        out = _parse_per_criterion_response(resp, 'explain')
        assert out['pass_criteria_evaluations'][0]['status'] == 'met'

    def test_invalid_status_degrades_to_worst_case(self):
        """A bogus status value never silently flips to a charitable default."""
        from canvigator_llm import _parse_per_criterion_response
        resp = (
            '{"pass_criteria_evaluations": [{"criterion": "X", "status": "kinda met"}],'
            ' "fatal_errors_evaluations": [{"error": "Y", "status": "maybe"}],'
            ' "feedback": "test"}'
        )
        out = _parse_per_criterion_response(resp, 'explain')
        assert out['pass_criteria_evaluations'][0]['status'] == 'missing'
        assert out['fatal_errors_evaluations'][0]['status'] == 'present'

    def test_malformed_returns_empty_shape(self):
        """Malformed JSON returns the canonical empty shape, not a crash."""
        from canvigator_llm import _parse_per_criterion_response
        out = _parse_per_criterion_response('not json', 'explain')
        assert out['pass_criteria_evaluations'] == []
        assert out['fatal_errors_evaluations'] == []
        assert out['feedback'] == ''


class TestAggregatePassFail:
    """Tests for canvigator_llm._aggregate_pass_fail."""

    def test_all_met_passes(self):
        """All pass criteria met, no fatal errors -> pass."""
        from canvigator_llm import _aggregate_pass_fail
        evals = {
            'pass_criteria_evaluations': [
                {'criterion': 'a', 'status': 'met'},
                {'criterion': 'b', 'status': 'met'},
            ],
            'fatal_errors_evaluations': [{'error': 'x', 'status': 'absent'}],
        }
        assert _aggregate_pass_fail(evals, 'explain') == 'pass'

    def test_all_missing_fails(self):
        """All criteria missing -> fail."""
        from canvigator_llm import _aggregate_pass_fail
        evals = {
            'pass_criteria_evaluations': [
                {'criterion': 'a', 'status': 'missing'},
                {'criterion': 'b', 'status': 'missing'},
            ],
            'fatal_errors_evaluations': [],
        }
        assert _aggregate_pass_fail(evals, 'explain') == 'fail'

    def test_partial_credit_at_threshold(self):
        """Two partial out of two = 0.5 score = exact threshold = pass."""
        from canvigator_llm import _aggregate_pass_fail
        evals = {
            'pass_criteria_evaluations': [
                {'criterion': 'a', 'status': 'partial'},
                {'criterion': 'b', 'status': 'partial'},
            ],
            'fatal_errors_evaluations': [],
        }
        assert _aggregate_pass_fail(evals, 'explain') == 'pass'

    def test_below_threshold_fails(self):
        """One partial + one missing = 0.25 score = below threshold -> fail."""
        from canvigator_llm import _aggregate_pass_fail
        evals = {
            'pass_criteria_evaluations': [
                {'criterion': 'a', 'status': 'partial'},
                {'criterion': 'b', 'status': 'missing'},
            ],
            'fatal_errors_evaluations': [],
        }
        assert _aggregate_pass_fail(evals, 'explain') == 'fail'

    def test_fatal_error_forces_fail(self):
        """A fatal error present forces fail, even with all criteria met."""
        from canvigator_llm import _aggregate_pass_fail
        evals = {
            'pass_criteria_evaluations': [
                {'criterion': 'a', 'status': 'met'},
                {'criterion': 'b', 'status': 'met'},
            ],
            'fatal_errors_evaluations': [{'error': 'x', 'status': 'present'}],
        }
        assert _aggregate_pass_fail(evals, 'explain') == 'fail'

    def test_empty_criteria_passes(self):
        """No pass criteria + no fatal errors -> pass (default 1.0 score)."""
        from canvigator_llm import _aggregate_pass_fail
        evals = {'pass_criteria_evaluations': [], 'fatal_errors_evaluations': []}
        assert _aggregate_pass_fail(evals, 'explain') == 'pass'

    def test_draw_mode_visual_elements_below_threshold_fails(self):
        """Draw mode: visual elements below threshold forces fail even if criteria met."""
        from canvigator_llm import _aggregate_pass_fail
        evals = {
            'pass_criteria_evaluations': [{'criterion': 'a', 'status': 'met'}],
            'fatal_errors_evaluations': [],
            'visual_elements_evaluations': [
                {'element': 'x', 'status': 'no'},
                {'element': 'y', 'status': 'no'},
            ],
        }
        assert _aggregate_pass_fail(evals, 'draw') == 'fail'

    def test_draw_mode_unclear_counts_as_half(self):
        """Draw mode: unclear visual elements count as 0.5."""
        from canvigator_llm import _aggregate_pass_fail
        evals = {
            'pass_criteria_evaluations': [{'criterion': 'a', 'status': 'met'}],
            'fatal_errors_evaluations': [],
            'visual_elements_evaluations': [
                {'element': 'x', 'status': 'unclear'},
                {'element': 'y', 'status': 'unclear'},
            ],
        }
        # 0.5 + 0.5 = 1.0 / 2 = 0.5 exactly = pass
        assert _aggregate_pass_fail(evals, 'draw') == 'pass'

    def test_draw_mode_no_visual_elements_defaults_to_pass_score(self):
        """Empty visual elements list defaults visual score to 1.0 (auto-pass on that axis)."""
        from canvigator_llm import _aggregate_pass_fail
        evals = {
            'pass_criteria_evaluations': [{'criterion': 'a', 'status': 'met'}],
            'fatal_errors_evaluations': [],
            'visual_elements_evaluations': [],
        }
        assert _aggregate_pass_fail(evals, 'draw') == 'pass'


class TestVoteAssessments:
    """Tests for canvigator_llm._vote_assessments."""

    def test_unanimous_pass_yields_high_confidence(self):
        """3-0 pass -> result=pass, confidence=high."""
        from canvigator_llm import _vote_assessments
        runs = [
            ('pass', 'Good.', {'pass_criteria_evaluations': []}),
            ('pass', 'Solid.', {'pass_criteria_evaluations': []}),
            ('pass', 'OK.', {'pass_criteria_evaluations': []}),
        ]
        result, confidence, feedback, _ = _vote_assessments(runs)
        assert result == 'pass'
        assert confidence == 'high'
        assert feedback == 'Good.'  # First matching run's feedback

    def test_majority_pass_yields_borderline(self):
        """2-1 pass -> result=pass, confidence=borderline."""
        from canvigator_llm import _vote_assessments
        runs = [
            ('fail', 'Off-topic.', {}),
            ('pass', 'Mostly there.', {}),
            ('pass', 'Acceptable.', {}),
        ]
        result, confidence, feedback, _ = _vote_assessments(runs)
        assert result == 'pass'
        assert confidence == 'borderline'
        assert feedback == 'Mostly there.'  # First matching run

    def test_majority_fail_yields_borderline(self):
        """1-2 fail -> result=fail, confidence=borderline."""
        from canvigator_llm import _vote_assessments
        runs = [
            ('pass', 'Looks good.', {}),
            ('fail', 'Missed it.', {}),
            ('fail', 'Off-topic.', {}),
        ]
        result, confidence, feedback, _ = _vote_assessments(runs)
        assert result == 'fail'
        assert confidence == 'borderline'
        assert feedback == 'Missed it.'

    def test_empty_runs_returns_fail(self):
        """Empty runs list returns a sane fail default."""
        from canvigator_llm import _vote_assessments
        result, confidence, feedback, _ = _vote_assessments([])
        assert result == 'fail'
        assert 'No assessment runs' in feedback


# ---------------------------------------------------------------------------
# canvigator_llm: exemplar parsing
# ---------------------------------------------------------------------------

class TestParseExemplars:
    """Tests for canvigator_llm._parse_exemplars."""

    def test_parses_clean_json(self):
        """All four exemplar fields are extracted from a clean JSON object."""
        from canvigator_llm import _parse_exemplars
        resp = (
            '{"exemplar_pass": "It halves the array each step.",'
            ' "exemplar_pass_note": "Captures the core idea.",'
            ' "exemplar_fail": "It looks at every element.",'
            ' "exemplar_fail_note": "Describes linear search instead."}'
        )
        out = _parse_exemplars(resp)
        assert out['exemplar_pass'] == 'It halves the array each step.'
        assert out['exemplar_pass_note'] == 'Captures the core idea.'
        assert out['exemplar_fail'] == 'It looks at every element.'
        assert out['exemplar_fail_note'] == 'Describes linear search instead.'

    def test_handles_fences(self):
        """Markdown fences are stripped."""
        from canvigator_llm import _parse_exemplars
        resp = '```json\n{"exemplar_pass": "X", "exemplar_pass_note": "", "exemplar_fail": "", "exemplar_fail_note": ""}\n```'
        out = _parse_exemplars(resp)
        assert out['exemplar_pass'] == 'X'

    def test_malformed_returns_empty(self):
        """Malformed JSON returns the empty exemplar dict."""
        from canvigator_llm import _parse_exemplars
        out = _parse_exemplars('not json')
        assert out == {
            'exemplar_pass': '', 'exemplar_pass_note': '',
            'exemplar_fail': '', 'exemplar_fail_note': '',
        }


# ---------------------------------------------------------------------------
# canvigator_quiz: locked-example sampler (Tier 3C)
# ---------------------------------------------------------------------------

class TestCollectLockedExamples:
    """Tests for CanvigatorQuiz._collectLockedExamples."""

    def _stub(self):
        """Return a CanvigatorQuiz stand-in carrying just _collectLockedExamples."""
        from canvigator_quiz import CanvigatorQuiz

        class _Stub:
            LOCKED_EXAMPLES_PER_BUCKET = CanvigatorQuiz.LOCKED_EXAMPLES_PER_BUCKET
            _collectLockedExamples = CanvigatorQuiz._collectLockedExamples

        return _Stub()

    def _df(self, rows):
        """Build an assessments DataFrame from row dicts."""
        from canvigator_quiz import CanvigatorQuiz
        return pd.DataFrame(rows, columns=CanvigatorQuiz.ASSESSMENTS_COLUMNS)

    def _row(self, sid, qid, result, transcript, sent=1, mode='explain', feedback='fb'):
        """Convenience row builder."""
        return {
            'student_id': sid, 'student_name': f's{sid}', 'question_id': qid,
            'question_mode': mode, 'conversation_id': 100 + sid,
            'result': result, 'confidence': 'high', 'feedback': feedback,
            'transcript': transcript, 'criteria_evaluations': '',
            'assessed_at': '', 'sent_assessment': sent, 'sent_at': '',
        }

    def test_empty_df_returns_empty(self):
        """An empty DataFrame yields no examples."""
        from canvigator_quiz import CanvigatorQuiz
        out = self._stub()._collectLockedExamples(
            pd.DataFrame(columns=CanvigatorQuiz.ASSESSMENTS_COLUMNS), 42,
        )
        assert out == []

    def test_only_includes_sent_assessments(self):
        """sent_assessment=0 rows are skipped."""
        df = self._df([
            self._row(1, 42, 'pass', 'good response', sent=1),
            self._row(2, 42, 'pass', 'also good', sent=0),
        ])
        out = self._stub()._collectLockedExamples(df, 42)
        responses = [e['response'] for e in out]
        assert 'good response' in responses
        assert 'also good' not in responses

    def test_only_includes_matching_question_id(self):
        """Rows for a different question_id are filtered out."""
        df = self._df([
            self._row(1, 42, 'pass', 'matches'),
            self._row(2, 99, 'pass', 'wrong question'),
        ])
        out = self._stub()._collectLockedExamples(df, 42)
        responses = [e['response'] for e in out]
        assert responses == ['matches']

    def test_skips_rows_with_empty_transcript(self):
        """Rows whose transcript is empty (e.g. draw mode) are skipped."""
        df = self._df([
            self._row(1, 42, 'pass', '', mode='draw'),
            self._row(2, 42, 'pass', 'has text'),
        ])
        out = self._stub()._collectLockedExamples(df, 42)
        responses = [e['response'] for e in out]
        assert responses == ['has text']

    def test_skips_draw_mode_even_with_transcript(self):
        """Draw mode is filtered before sampling, regardless of transcript."""
        df = self._df([
            self._row(1, 42, 'pass', 'something', mode='draw'),
            self._row(2, 42, 'pass', 'explain text', mode='explain'),
        ])
        out = self._stub()._collectLockedExamples(df, 42)
        responses = [e['response'] for e in out]
        assert responses == ['explain text']

    def test_returns_both_passes_and_fails(self):
        """Both pass and fail buckets are sampled."""
        df = self._df([
            self._row(1, 42, 'pass', 'pass-resp'),
            self._row(2, 42, 'fail', 'fail-resp'),
        ])
        out = self._stub()._collectLockedExamples(df, 42)
        results = sorted(e['result'] for e in out)
        assert results == ['fail', 'pass']

    def test_caps_per_bucket(self):
        """No more than LOCKED_EXAMPLES_PER_BUCKET examples per result label."""
        from canvigator_quiz import CanvigatorQuiz
        cap = CanvigatorQuiz.LOCKED_EXAMPLES_PER_BUCKET
        rows = [self._row(i, 42, 'pass', f'resp{i}') for i in range(cap + 4)]
        df = self._df(rows)
        out = self._stub()._collectLockedExamples(df, 42)
        passes = [e for e in out if e['result'] == 'pass']
        assert len(passes) == cap


class TestGetQuizQuestionsPosition:
    """Tests that getQuizQuestions enumerates `position` as 1..N regardless of Canvas's stale `position` field."""

    def _stub_quiz(self, canvas_questions, tmp_path):
        """Build a CanvigatorQuiz instance bypassing __init__ for direct method tests."""
        from canvigator_quiz import CanvigatorQuiz

        class _Config:
            def __init__(self, data_path):
                self.data_path = data_path
                self.quiz_prefix = 'quiz'

        class _CanvasQuiz:
            id = 999
            assignment_id = 12345

        quiz = CanvigatorQuiz.__new__(CanvigatorQuiz)
        quiz.canvas_quiz = _CanvasQuiz()
        quiz.config = _Config(tmp_path)
        quiz.quiz_questions = canvas_questions
        return quiz

    def _question_stub(self, qid, position, name, qtype='multiple_choice_question', text='', pp=1.0):
        """Build a minimal stand-in for a canvasapi QuizQuestion object."""
        class _Q:
            pass
        q = _Q()
        q.id = qid
        q.position = position
        q.question_name = name
        q.question_type = qtype
        q.question_text = text
        q.points_possible = pp
        q.answers = []
        return q

    def test_position_is_one_based_enumeration(self, tmp_path):
        """`position` reflects iteration order, not Canvas's stale per-question position field."""
        # Canvas reports stale positions: out-of-order values, gaps from removed questions
        canvas_questions = [
            self._question_stub(101, position=7, name='Q one'),     # stale 7
            self._question_stub(102, position=2, name='Q two'),     # stale 2
            self._question_stub(103, position=42, name='Q three'),  # stale 42 (gap)
        ]
        quiz = self._stub_quiz(canvas_questions, tmp_path)
        quiz.getQuizQuestions(tag=False)

        out_files = list(tmp_path.glob('quiz999_questions_*.csv'))
        assert len(out_files) == 1
        df = pd.read_csv(out_files[0])

        # Position is 1..N in iteration order, ignoring Canvas's reported values
        assert list(df['position']) == [1, 2, 3]
        assert list(df['question_id']) == [101, 102, 103]
        assert list(df['question_name']) == ['Q one', 'Q two', 'Q three']


# ---------------------------------------------------------------------------
# Multi-quiz reminder tests (--all path for send-quiz-reminder)
# ---------------------------------------------------------------------------

class TestClassifyStudentForQuiz:
    """Tests for CanvigatorQuiz._classifyStudentForQuiz."""

    def _quiz(self):
        """Return a CanvigatorQuiz instance bypassing __init__ for direct method tests."""
        from canvigator_quiz import CanvigatorQuiz
        return CanvigatorQuiz.__new__(CanvigatorQuiz)

    def test_no_attempt(self):
        """Student missing from quiz_scores → 'no attempt' with no bullets."""
        result = self._quiz()._classifyStudentForQuiz(
            student_id=99, quiz_scores={}, points_possible=10.0,
            subs_by_q_df=None, events_df=None, question_info={},
        )
        assert result == ('no attempt', None)

    def test_imperfect_score_with_bullets(self, monkeypatch):
        """Student with score < points_possible → 'score x/y' with bullet section."""
        quiz = self._quiz()
        monkeypatch.setattr(quiz, '_buildMissedBulletsForStudent',
                            lambda sid, df, qi: '\n• Q1: keywords (5/10 pts)')
        result = quiz._classifyStudentForQuiz(
            student_id=42, quiz_scores={42: 7}, points_possible=10,
            subs_by_q_df=None, events_df=None, question_info={},
        )
        assert result[0] == 'score 7/10'
        assert 'Q1: keywords' in result[1]

    def test_perfect_with_blur(self, monkeypatch):
        """Student with perfect score and blur events → 'page blur' with bullets."""
        quiz = self._quiz()
        monkeypatch.setattr(quiz, '_buildBlurBulletsForStudent',
                            lambda sid, df, qi: '\n• Q2: stuff')
        result = quiz._classifyStudentForQuiz(
            student_id=42, quiz_scores={42: 10}, points_possible=10,
            subs_by_q_df=None, events_df=None, question_info={},
        )
        assert result[0] == 'page blur'
        assert 'Q2: stuff' in result[1]

    def test_perfect_no_blur_returns_perfect_clean(self, monkeypatch):
        """Perfect score with no blur events → 'perfect clean' (encourage retake)."""
        quiz = self._quiz()
        monkeypatch.setattr(quiz, '_buildBlurBulletsForStudent', lambda sid, df, qi: None)
        result = quiz._classifyStudentForQuiz(
            student_id=42, quiz_scores={42: 10}, points_possible=10,
            subs_by_q_df=None, events_df=None, question_info={},
        )
        assert result == ('perfect clean', None)


class TestComposeMultiQuizReminder:
    """Tests for canvigator_course._composeMultiQuizReminder."""

    def test_orders_sections_in_input_order(self):
        """Sections are emitted in the order provided in state_list."""
        from canvigator_course import _composeMultiQuizReminder
        state_list = [
            {'quiz_name': 'Quiz A', 'due_at': '2026-05-01T12:00:00Z',
             'points_possible': 10, 'reason': 'no attempt', 'bullets': None},
            {'quiz_name': 'Quiz B', 'due_at': '2026-05-05T12:00:00Z',
             'points_possible': 10, 'reason': 'score 7/10',
             'bullets': '\n• Q1: foo (5/10 pts)'},
            {'quiz_name': 'Quiz C', 'due_at': '2026-05-10T12:00:00Z',
             'points_possible': 10, 'reason': 'page blur', 'bullets': '\n• Q2: bar'},
        ]
        body = _composeMultiQuizReminder('Alex', state_list)

        # Greeting and structure
        assert body.startswith('Hello Alex,')

        # Order check: A appears before B which appears before C
        i_a = body.index('Quiz A')
        i_b = body.index('Quiz B')
        i_c = body.index('Quiz C')
        assert i_a < i_b < i_c

        # Per-section formatting
        assert 'Quiz A (due 2026-05-01) — not yet attempted' in body
        assert 'Quiz B (due 2026-05-05) — score 7/10' in body
        assert '• Q1: foo (5/10 pts)' in body
        assert 'Quiz C (due 2026-05-10) — perfect score' in body
        assert '• Q2: bar' in body

        # NOTE disclaimer present
        assert 'auto-generated message' in body

    def test_handles_missing_due_at(self):
        """A quiz with no due_at falls back to '?' in the section header."""
        from canvigator_course import _composeMultiQuizReminder
        body = _composeMultiQuizReminder('Sam', [
            {'quiz_name': 'Q', 'due_at': None,
             'points_possible': 5, 'reason': 'no attempt', 'bullets': None},
        ])
        assert 'Q (due ?)' in body

    def test_skips_bullets_when_none(self):
        """Section with bullets=None renders just the header, no nested bullet block."""
        from canvigator_course import _composeMultiQuizReminder
        body = _composeMultiQuizReminder('Sam', [
            {'quiz_name': 'Q1', 'due_at': '2026-05-01T12:00:00Z',
             'points_possible': 5, 'reason': 'no attempt', 'bullets': None},
        ])
        assert 'Q1 (due 2026-05-01) — not yet attempted' in body
        # Only the section header bullet appears — no nested bullets below it.
        assert body.count('•') == 1

    def test_perfect_clean_section(self):
        """A 'perfect clean' state renders a retake-encouraging section header."""
        from canvigator_course import _composeMultiQuizReminder
        body = _composeMultiQuizReminder('Sam', [
            {'quiz_name': 'Q5', 'due_at': '2026-05-15T12:00:00Z',
             'points_possible': 10, 'reason': 'perfect clean', 'bullets': None},
        ])
        assert 'Q5 (due 2026-05-15) — nice work, perfect score with no window changes' in body
        assert 'consider retaking' in body

    def test_single_quiz_keeps_top_level_bullets(self):
        """With only one quiz, bullets stay at top level (preamble preserved, no indent)."""
        from canvigator_course import _composeMultiQuizReminder
        bullets_block = (
            "\n\nThe questions that you missed on this most recent attempt covered "
            "the concepts/topics:\n• Q1: foo (5/10 pts)\n• Q3: bar (2/10 pts)"
        )
        body = _composeMultiQuizReminder('Sam', [
            {'quiz_name': 'Quiz Solo', 'due_at': '2026-05-01T12:00:00Z',
             'points_possible': 10, 'reason': 'score 7/10', 'bullets': bullets_block},
        ])
        # Preamble is preserved
        assert 'covered the concepts/topics' in body
        # Bullets are NOT indented (no '  •' anywhere)
        assert '\n  •' not in body
        # Top-level bullets present
        assert '\n• Q1: foo (5/10 pts)' in body

    def test_multi_quiz_indents_bullets_as_sub_bullets(self):
        """With 2+ quizzes, per-question bullets become 2-space-indented sub-bullets and the preamble is dropped."""
        from canvigator_course import _composeMultiQuizReminder
        missed_block = (
            "\n\nThe questions that you missed on this most recent attempt covered "
            "the concepts/topics:\n• Q1: foo (5/10 pts)\n• Q3: bar (2/10 pts)"
        )
        blur_block = (
            "\n\nThe questions that you changed window focus on covered the concepts/topics:"
            "\n• Q2: baz"
        )
        body = _composeMultiQuizReminder('Sam', [
            {'quiz_name': 'Quiz A', 'due_at': '2026-05-01T12:00:00Z',
             'points_possible': 10, 'reason': 'score 7/10', 'bullets': missed_block},
            {'quiz_name': 'Quiz B', 'due_at': '2026-05-05T12:00:00Z',
             'points_possible': 10, 'reason': 'page blur', 'bullets': blur_block},
        ])
        # Quiz names render as top-level bullets
        assert '\n• Quiz A (due 2026-05-01) — score 7/10' in body
        assert '\n• Quiz B (due 2026-05-05) — perfect score' in body
        # Per-question bullets are 2-space-indented sub-bullets
        assert '\n  • Q1: foo (5/10 pts)' in body
        assert '\n  • Q3: bar (2/10 pts)' in body
        assert '\n  • Q2: baz' in body
        # Preamble dropped in nested mode
        assert 'covered the concepts/topics' not in body


class TestSendAllQuizRemindersFiltering:
    """Tests for sendAllQuizReminders quiz filtering and missing-CSV skip."""

    def _make_course_stub(self, quizzes, students):
        """Build a CanvigatorCourse instance bypassing __init__ for direct method tests."""
        from canvigator_course import CanvigatorCourse

        class _CanvasCourse:
            def __init__(self, quizzes):
                self._quizzes = quizzes
                self.course_code = 'CSI-3300-001-12345'

            def get_quizzes(self):
                return self._quizzes

        course = CanvigatorCourse.__new__(CanvigatorCourse)
        course.canvas = None
        course.canvas_course = _CanvasCourse(quizzes)
        course.config = None
        course.students = students
        course.verbose = False
        return course

    def test_filters_to_eligible_quizzes_and_skips_when_empty(self, capsys, monkeypatch):
        """When no quiz passes is_quiz_open_for_reminder, the method returns early."""
        from datetime import timezone as _tz
        # Past-due quiz only
        quizzes = [_StubQuiz(published=True, due_at='2026-01-01T12:00:00Z')]
        course = self._make_course_stub(quizzes, students=[{'id': 1, 'name': 'Alice A'}])

        # Freeze "now" so the past quiz remains past
        from canvigator_utils import is_quiz_open_for_reminder as orig
        ref_now = datetime(2026, 4, 24, 12, 0, 0, tzinfo=_tz.utc)
        monkeypatch.setattr('canvigator_course.cu.is_quiz_open_for_reminder',
                            lambda q: orig(q, now=ref_now))

        course.sendAllQuizReminders(dry_run=True)
        captured = capsys.readouterr().out
        assert 'No published quizzes with a future due date' in captured

    def test_skips_quizzes_missing_tagged_csv(self, capsys, monkeypatch, caplog):
        """A quiz whose _loadQuestionInfo raises FileNotFoundError is skipped with a warning."""
        import logging as _logging
        from datetime import timezone as _tz
        # One eligible quiz; the only quiz lacks the tagged CSV
        quizzes = [_StubQuiz(published=True, due_at='2026-12-31T12:00:00Z')]
        # give it required attrs the constructor will need
        quizzes[0].id = 999
        quizzes[0].title = 'Quiz Z'
        quizzes[0].points_possible = 10
        course = self._make_course_stub(quizzes, students=[{'id': 1, 'name': 'Alice A'}])

        from canvigator_utils import is_quiz_open_for_reminder as orig
        ref_now = datetime(2026, 4, 24, 12, 0, 0, tzinfo=_tz.utc)
        monkeypatch.setattr('canvigator_course.cu.is_quiz_open_for_reminder',
                            lambda q: orig(q, now=ref_now))

        # Stub the CanvigatorQuiz constructor to return an object whose
        # _loadQuestionInfo raises FileNotFoundError.
        class _FakeQuiz:
            def __init__(self, *a, **kw):
                pass

            def _loadQuestionInfo(self):
                raise FileNotFoundError('No *_questions_w_tags_*.csv found for quiz')
        monkeypatch.setattr('canvigator_course.cq.CanvigatorQuiz', _FakeQuiz)

        with caplog.at_level(_logging.WARNING, logger='canvigator_course'):
            course.sendAllQuizReminders(dry_run=True)
        captured = capsys.readouterr().out
        assert "Skipping 'Quiz Z'" in captured
        assert any("Skipping 'Quiz Z'" in r.message for r in caplog.records)
        # No reminder-worthy state, so we hit the "all caught up" branch
        assert 'All students are caught up' in captured


# ---------------------------------------------------------------------------
# canvigator_assignment tests
# ---------------------------------------------------------------------------

class TestCanvigatorAssignment:
    """Tests for canvigator_assignment helpers and class methods."""

    def _make_assignment_obj(self, points_possible=1):
        """Build a minimal stand-in for a Canvas Assignment object."""
        return SimpleNamespace(id=999, points_possible=points_possible)

    def _make_instance(self, points_possible=1):
        """Build a CanvigatorAssignment with mocked dependencies."""
        from canvigator_assignment import CanvigatorAssignment
        return CanvigatorAssignment(
            canvas=MagicMock(),
            course=MagicMock(),
            canvas_assignment=self._make_assignment_obj(points_possible),
            config=MagicMock(),
        )

    def test_is_media_recording_assignment_filter(self):
        """Filter passes only assignments whose submission_types is exactly ['media_recording']."""
        from canvigator_assignment import _isMediaRecordingAssignment
        assert _isMediaRecordingAssignment(SimpleNamespace(submission_types=['media_recording'])) is True
        assert _isMediaRecordingAssignment(SimpleNamespace(submission_types=['online_text_entry'])) is False
        # Mixed types must NOT match — we want recording-only assignments.
        assert _isMediaRecordingAssignment(SimpleNamespace(submission_types=['media_recording', 'online_upload'])) is False
        assert _isMediaRecordingAssignment(SimpleNamespace(submission_types=None)) is False
        assert _isMediaRecordingAssignment(SimpleNamespace(submission_types=[])) is False

    def test_extract_audio_url_from_media_comment(self):
        """media_comment with an audio media_type yields a .m4a extension."""
        ca = self._make_instance()
        sub = SimpleNamespace(media_comment={'url': 'https://canvas/audio.m4a', 'media_type': 'audio'})
        url, ext = ca._extractAudioUrl(sub)
        assert url == 'https://canvas/audio.m4a'
        assert ext == '.m4a'

    def test_extract_audio_url_video_media_comment_uses_mp4(self):
        """A video media_comment yields an .mp4 extension instead of .m4a."""
        ca = self._make_instance()
        sub = SimpleNamespace(media_comment={'url': 'https://canvas/v.mp4', 'media_type': 'video'})
        url, ext = ca._extractAudioUrl(sub)
        assert url == 'https://canvas/v.mp4'
        assert ext == '.mp4'

    def test_extract_audio_url_from_attachment_fallback(self):
        """When no media_comment is present, audio attachments are picked up by mime type."""
        ca = self._make_instance()
        sub = SimpleNamespace(
            media_comment=None,
            attachments=[{
                'url': 'https://canvas/file.webm',
                'filename': 'recording.webm',
                'content-type': 'audio/webm',
            }],
        )
        url, ext = ca._extractAudioUrl(sub)
        assert url == 'https://canvas/file.webm'
        assert ext == '.webm'

    def test_extract_audio_url_returns_none_when_empty(self):
        """A submission with neither media_comment nor attachments returns (None, None)."""
        ca = self._make_instance()
        sub = SimpleNamespace(media_comment=None, attachments=[])
        assert ca._extractAudioUrl(sub) == (None, None)

    def test_extract_audio_url_skips_non_audio_attachments(self):
        """Non-audio attachments (e.g. PDFs) are ignored."""
        ca = self._make_instance()
        sub = SimpleNamespace(
            media_comment=None,
            attachments=[
                {'url': 'https://canvas/notes.pdf', 'filename': 'notes.pdf', 'content-type': 'application/pdf'},
            ],
        )
        assert ca._extractAudioUrl(sub) == (None, None)

    def test_build_recording_row_shape(self):
        """_buildRecordingRow returns a dict with all expected columns populated; grade defaults to empty."""
        ca = self._make_instance()
        sub = SimpleNamespace(user_id=42, id=7, submitted_at='2026-05-01T12:00:00Z')
        row = ca._buildRecordingRow(sub, 'Alice', 'data/course/media_recordings/assignment999/42_7.wav', 'hello')
        assert row['student_id'] == 42
        assert row['student_name'] == 'Alice'
        assert row['submission_id'] == 7
        assert row['submitted_at'] == '2026-05-01T12:00:00Z'
        assert row['audio_path'] == 'data/course/media_recordings/assignment999/42_7.wav'
        assert row['transcript'] == 'hello'
        assert 'transcribed_at' in row and row['transcribed_at']
        # Default grade=None yields empty grade and graded_at columns.
        assert row['grade'] == ''
        assert row['graded_at'] == ''

    def test_build_recording_row_records_grade_and_graded_at(self):
        """When a grade is passed, _buildRecordingRow records the value and a graded_at timestamp."""
        ca = self._make_instance()
        sub = SimpleNamespace(user_id=42, id=7, submitted_at='2026-05-01T12:00:00Z')
        row = ca._buildRecordingRow(sub, 'Alice', 'audio.wav', 'hello', grade=0.5)
        assert row['grade'] == 0.5
        assert row['graded_at']  # non-empty ISO timestamp

    def test_post_grade_dry_run_does_not_call_edit(self):
        """In dry-run mode, _postGrade skips Submission.edit and returns False."""
        ca = self._make_instance(points_possible=3)
        sub = MagicMock()
        sub.user_id = 42
        result = ca._postGrade(sub, 3, dry_run=True)
        assert result is False
        sub.edit.assert_not_called()

    def test_post_grade_real_call_uses_points_possible(self):
        """A real grade post calls Submission.edit with posted_grade=points_possible."""
        ca = self._make_instance(points_possible=5)
        sub = MagicMock()
        sub.user_id = 42
        result = ca._postGrade(sub, 5, dry_run=False)
        assert result is True
        sub.edit.assert_called_once_with(submission={'posted_grade': 5})

    def test_post_grade_logs_warning_on_failure(self, caplog):
        """When Submission.edit raises, _postGrade logs a warning and returns False."""
        import logging as _logging
        ca = self._make_instance(points_possible=2)
        sub = MagicMock()
        sub.user_id = 42
        sub.edit.side_effect = RuntimeError("Canvas down")
        with caplog.at_level(_logging.WARNING, logger='canvigator_assignment'):
            result = ca._postGrade(sub, 2, dry_run=False)
        assert result is False
        assert any('Grade post failed' in r.message for r in caplog.records)

    def test_fetch_audio_invokes_ffmpeg_with_url_and_whitelist(self):
        """_fetchAudio passes the URL directly to ffmpeg with an https-capable protocol whitelist."""
        from canvigator_assignment import _fetchAudio
        audio = Path('/tmp/data/course/media_recordings/assignment999/42_7.wav')
        with patch('canvigator_assignment.subprocess.run') as mock_run:
            mock_run.return_value = SimpleNamespace(returncode=0)
            result = _fetchAudio('https://canvas/manifest.mpd', audio)
        assert result is True
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == 'ffmpeg'
        assert '-protocol_whitelist' in cmd
        whitelist = cmd[cmd.index('-protocol_whitelist') + 1]
        # https must be allowed so ffmpeg can follow DASH/HLS fragment URLs.
        assert 'https' in whitelist
        # Input URL is passed directly, not a local file.
        assert 'https://canvas/manifest.mpd' in cmd
        # Output is audio-only PCM WAV at 16 kHz mono — the format Gemma's
        # audio path accepts. Anything else (e.g. AAC m4a) gets rejected as
        # "image: unknown format" by Ollama's media-type detector.
        assert '-vn' in cmd
        assert 'pcm_s16le' in cmd
        assert cmd[cmd.index('-ar') + 1] == '16000'
        assert cmd[cmd.index('-ac') + 1] == '1'
        assert str(audio) in cmd

    def test_fetch_audio_handles_missing_ffmpeg(self, caplog):
        """When ffmpeg is not on PATH, _fetchAudio returns False and logs a warning."""
        import logging as _logging
        from canvigator_assignment import _fetchAudio
        audio = Path('/tmp/missing/v.m4a')
        with patch('canvigator_assignment.subprocess.run', side_effect=FileNotFoundError()):
            with caplog.at_level(_logging.WARNING, logger='canvigator_assignment'):
                result = _fetchAudio('https://x/y', audio)
        assert result is False
        assert any('ffmpeg not found' in r.message for r in caplog.records)

    def test_fetch_audio_handles_ffmpeg_failure(self, caplog):
        """When ffmpeg exits non-zero, _fetchAudio returns False and logs a warning."""
        import logging as _logging
        from canvigator_assignment import _fetchAudio
        audio = Path('/tmp/v.m4a')
        err = subprocess.CalledProcessError(1, 'ffmpeg', stderr=b'bad input')
        with patch('canvigator_assignment.subprocess.run', side_effect=err):
            with caplog.at_level(_logging.WARNING, logger='canvigator_assignment'):
                result = _fetchAudio('https://x/y', audio)
        assert result is False
        assert any('ffmpeg failed' in r.message for r in caplog.records)

    def test_prompt_for_grade_returns_full_credit_on_empty(self, monkeypatch):
        """Empty input on the prompt awards points_possible (full credit)."""
        from canvigator_assignment import _promptForGrade
        monkeypatch.setattr('builtins.input', lambda _prompt='': '')
        result = _promptForGrade('Alice', 'transcript text', 'audio.wav', points_possible=1)
        assert result == 1.0

    def test_prompt_for_grade_returns_none_on_skip(self, monkeypatch):
        """'s', 'skip' (case-insensitive) record no grade for the student."""
        from canvigator_assignment import _promptForGrade
        for raw in ('s', 'S', 'skip', 'SKIP'):
            monkeypatch.setattr('builtins.input', lambda _prompt='', _raw=raw: _raw)
            assert _promptForGrade('Alice', 'transcript', 'audio.wav', points_possible=1) is None

    def test_prompt_for_grade_returns_numeric_value(self, monkeypatch):
        """A numeric value is returned as a float, even when smaller or larger than points_possible."""
        from canvigator_assignment import _promptForGrade
        monkeypatch.setattr('builtins.input', lambda _prompt='': '0.5')
        assert _promptForGrade('Alice', 't', 'a.wav', points_possible=1) == 0.5
        monkeypatch.setattr('builtins.input', lambda _prompt='': '2')
        assert _promptForGrade('Alice', 't', 'a.wav', points_possible=1) == 2.0

    def test_prompt_for_grade_reprompts_on_invalid(self, monkeypatch):
        """Negative and non-numeric input re-prompt; the eventual valid input is returned."""
        from canvigator_assignment import _promptForGrade
        inputs = iter(['-1', 'oops', '1'])
        monkeypatch.setattr('builtins.input', lambda _prompt='': next(inputs))
        assert _promptForGrade('Alice', 't', 'a.wav', points_possible=1) == 1.0

    def test_collect_unique_tags_lowercases_and_dedupes(self):
        """_collectUniqueTags collapses duplicates across rows, lowercases, and sorts."""
        from canvigator_assignment import _collectUniqueTags
        series = pd.Series([
            'graphs, recursion, hash tables',
            'Hash Tables, big-o',
            'graphs',
            None,
            '   ',
        ])
        assert _collectUniqueTags(series) == ['big-o', 'graphs', 'hash tables', 'recursion']

    def test_render_analysis_report_orders_tags_by_count(self):
        """Tag-grounded analysis section ranks tags by descending student count."""
        from canvigator_assignment import _renderAnalysisReport
        transcripts = [('Alice', 'a'), ('Bob', 'b'), ('Carol', 'c')]
        tag_to_indices = {
            'graphs': [1, 2],
            'recursion': [1, 2, 3],
            'big-o': [],
        }
        report = _renderAnalysisReport(
            'Quiz 1 Check-in', transcripts, ['big-o', 'graphs', 'recursion'],
            tag_to_indices, '- **Confusion**: ...', 5, 2,
        )
        # 'recursion' appears before 'graphs' (3 > 2 students); 'big-o' is reported as unmentioned.
        assert report.index('| recursion |') < report.index('| graphs |')
        assert 'Tags not referenced by anyone' in report
        assert 'big-o' in report.split('Tags not referenced by anyone')[1]
        # Roster section lists each student with their 1-based index.
        assert '\n1. Alice' in report
        assert '\n3. Carol' in report
        # Coverage line surfaces the total/empty counts.
        assert '3 transcripts analyzed' in report
        assert '2 empty' in report


class TestRecordingAnalysisLLM:
    """Tests for the recording-analysis prompt builders and parsers in canvigator_llm."""

    def test_classify_response_filters_to_valid_tags(self):
        """_parse_classify_response only keeps tags that appear in the valid list."""
        from canvigator_llm import _parse_classify_response
        valid = ['graphs', 'recursion', 'hash tables']
        assert _parse_classify_response('graphs, recursion', valid) == ['graphs', 'recursion']
        # Invented tag is dropped; valid tags are preserved.
        assert _parse_classify_response('graphs, dynamic programming', valid) == ['graphs']
        # Case-insensitive match canonicalizes back to the valid form.
        assert _parse_classify_response('Graphs, RECURSION', valid) == ['graphs', 'recursion']

    def test_classify_response_handles_none(self):
        """A 'none' response yields an empty list."""
        from canvigator_llm import _parse_classify_response
        assert _parse_classify_response('none', ['graphs']) == []
        assert _parse_classify_response('  None.  ', ['graphs']) == []
        assert _parse_classify_response('', ['graphs']) == []

    def test_classify_response_dedupes(self):
        """Duplicate tags in the response are emitted once."""
        from canvigator_llm import _parse_classify_response
        assert _parse_classify_response('graphs, graphs, GRAPHS', ['graphs']) == ['graphs']

    def test_build_classify_prompt_includes_tags_and_transcript(self):
        """The classify prompt embeds both the tag list and the transcript verbatim."""
        from canvigator_llm import _build_classify_recording_prompt
        prompt = _build_classify_recording_prompt('I struggled with graphs', ['graphs', 'recursion'])
        assert 'graphs, recursion' in prompt
        assert 'I struggled with graphs' in prompt

    def test_build_themes_prompt_numbers_students(self):
        """The themes prompt labels each transcript with a 1-based 'Student N:' index."""
        from canvigator_llm import _build_themes_prompt
        prompt = _build_themes_prompt(['first', 'second'], ['graphs'])
        assert 'Student 1: first' in prompt
        assert 'Student 2: second' in prompt


# ---------------------------------------------------------------------------
# canvigator_llm quiz-question generation helpers
# ---------------------------------------------------------------------------

class TestQuizQuestionHelpers:
    """Tests for _build_quiz_question_prompt and _parse_quiz_question."""

    def test_build_quiz_question_prompt_includes_seed(self):
        """The instructor's seed appears verbatim in the user prompt."""
        from canvigator_llm import _build_quiz_question_prompt
        seed = "a question on Big-O for binary search"
        prompt = _build_quiz_question_prompt(seed)
        assert seed in prompt
        assert "JSON" in prompt

    def test_build_quiz_question_prompt_handles_empty(self):
        """Empty/None seed still returns a string (the wrapper text)."""
        from canvigator_llm import _build_quiz_question_prompt
        assert isinstance(_build_quiz_question_prompt(""), str)
        assert isinstance(_build_quiz_question_prompt(None), str)

    def test_parse_quiz_question_accepts_multiple_choice(self):
        """A well-formed multiple_choice_question payload round-trips and points_possible is stripped."""
        import json as _json
        from canvigator_llm import _parse_quiz_question
        payload = _json.dumps({
            "question_type": "multiple_choice_question",
            "question_name": "Binary search complexity",
            "question_text": "What is the time complexity of binary search?",
            "points_possible": 5,
            "answers": [
                {"answer_text": "O(log n)", "answer_weight": 100},
                {"answer_text": "O(n)", "answer_weight": 0},
                {"answer_text": "O(n log n)", "answer_weight": 0},
                {"answer_text": "O(1)", "answer_weight": 0},
            ],
        })
        result = _parse_quiz_question(payload)
        assert result is not None
        assert result["question_type"] == "multiple_choice_question"
        assert result["question_name"] == "Binary search complexity"
        assert "points_possible" not in result
        assert len(result["answers"]) == 4

    def test_parse_quiz_question_handles_markdown_fence(self):
        """JSON wrapped in ```json fences is tolerated."""
        from canvigator_llm import _parse_quiz_question
        wrapped = (
            "```json\n"
            '{"question_type": "true_false_question", "question_name": "T/F",'
            ' "question_text": "Quicksort is stable.",'
            ' "answers": [{"answer_text": "True", "answer_weight": 0},'
            ' {"answer_text": "False", "answer_weight": 100}]}'
            "\n```"
        )
        result = _parse_quiz_question(wrapped)
        assert result is not None
        assert result["question_type"] == "true_false_question"
        assert len(result["answers"]) == 2

    def test_parse_quiz_question_rejects_unknown_type(self):
        """Unknown question_type returns None."""
        import json as _json
        from canvigator_llm import _parse_quiz_question
        payload = _json.dumps({
            "question_type": "essay_question",
            "question_name": "Discuss",
            "question_text": "Discuss recursion.",
            "answers": [],
        })
        assert _parse_quiz_question(payload) is None

    def test_parse_quiz_question_rejects_missing_text(self):
        """Missing or empty question_text returns None."""
        import json as _json
        from canvigator_llm import _parse_quiz_question
        payload = _json.dumps({
            "question_type": "multiple_choice_question",
            "question_name": "X",
            "answers": [],
        })
        assert _parse_quiz_question(payload) is None

    def test_parse_quiz_question_allows_empty_answers_for_calculated(self):
        """A calculated_question with formulas/variables and an empty answers list is accepted."""
        import json as _json
        from canvigator_llm import _parse_quiz_question
        payload = _json.dumps({
            "question_type": "calculated_question",
            "question_name": "Doubling",
            "question_text": "What is [x] doubled?",
            "answers": [],
            "variables": [{"name": "x", "min": 1, "max": 10, "scale": 0}],
            "formulas": [{"formula": "x*2"}],
            "formula_decimal_places": 0,
        })
        result = _parse_quiz_question(payload)
        assert result is not None
        assert result["question_type"] == "calculated_question"
        assert result["formulas"] == [{"formula": "x*2"}]

    def test_parse_quiz_question_defaults_missing_name(self):
        """A payload without question_name gets a fallback name."""
        import json as _json
        from canvigator_llm import _parse_quiz_question
        payload = _json.dumps({
            "question_type": "multiple_choice_question",
            "question_text": "Pick one.",
            "answers": [{"answer_text": "A", "answer_weight": 100}],
        })
        result = _parse_quiz_question(payload)
        assert result is not None
        assert result["question_name"] == "Generated question"

    def test_parse_quiz_question_handles_malformed_json(self):
        """Invalid JSON returns None rather than raising."""
        from canvigator_llm import _parse_quiz_question
        assert _parse_quiz_question("not json at all") is None
        assert _parse_quiz_question("") is None
        assert _parse_quiz_question(None) is None

    def test_summarize_draft_renders_multiple_choice(self):
        """A multiple-choice draft summary marks the correct answer with '*' and lists distractors with '-'."""
        from canvigator_llm import _summarize_draft_for_prompt
        draft = {
            "question_type": "multiple_choice_question",
            "question_text": "What is 2+2?",
            "answers": [
                {"answer_text": "3", "answer_weight": 0},
                {"answer_text": "4", "answer_weight": 100},
                {"answer_text": "5", "answer_weight": 0},
            ],
        }
        summary = _summarize_draft_for_prompt(draft)
        assert "What is 2+2?" in summary
        assert "* 4" in summary
        assert "- 3" in summary
        assert "- 5" in summary

    def test_summarize_draft_renders_matching(self):
        """A matching-question draft summary lists 'left -> right' pairs."""
        from canvigator_llm import _summarize_draft_for_prompt
        draft = {
            "question_type": "matching_question",
            "question_text": "Match terms.",
            "answers": [
                {"answer_match_left": "Tree", "answer_match_right": "Hierarchical"},
                {"answer_match_left": "Graph", "answer_match_right": "Network"},
            ],
        }
        summary = _summarize_draft_for_prompt(draft)
        assert "Tree -> Hierarchical" in summary
        assert "Graph -> Network" in summary

    def test_summarize_draft_handles_non_dict(self):
        """Non-dict input returns an empty string rather than raising."""
        from canvigator_llm import _summarize_draft_for_prompt
        assert _summarize_draft_for_prompt(None) == ""
        assert _summarize_draft_for_prompt("not a dict") == ""

    def test_build_quiz_question_prompt_no_prior_drafts(self):
        """Without prior_drafts the prompt has no rejected-drafts section."""
        from canvigator_llm import _build_quiz_question_prompt
        prompt = _build_quiz_question_prompt("a Big-O question", prior_drafts=None)
        assert "REJECTED" not in prompt
        assert "SUBSTANTIVELY DIFFERENT" not in prompt
        assert "a Big-O question" in prompt

    def test_build_quiz_question_prompt_with_prior_drafts(self):
        """With prior_drafts the prompt instructs the model to diverge and includes draft summaries."""
        from canvigator_llm import _build_quiz_question_prompt
        prior = [{
            "question_type": "multiple_choice_question",
            "question_text": "What is the time complexity of binary search?",
            "answers": [
                {"answer_text": "O(log n)", "answer_weight": 100},
                {"answer_text": "O(n)", "answer_weight": 0},
            ],
        }]
        prompt = _build_quiz_question_prompt("a Big-O question", prior_drafts=prior)
        assert "REJECTED" in prompt
        assert "SUBSTANTIVELY DIFFERENT" in prompt
        assert "What is the time complexity of binary search?" in prompt
        assert "Rejected draft 1" in prompt


# ---------------------------------------------------------------------------
# canvigator_digest tests
# ---------------------------------------------------------------------------


class TestFindCsvsInWindow:
    """Tests for canvigator_utils.find_csvs_in_window."""

    def _write(self, tmp_path, name):
        (tmp_path / name).write_text('x')

    def test_returns_only_files_at_or_after_cutoff(self, tmp_path):
        """Files dated >= since_date are returned; older are filtered out."""
        from canvigator_utils import find_csvs_in_window
        from datetime import date
        self._write(tmp_path, 'foo_20260420.csv')
        self._write(tmp_path, 'foo_20260424.csv')
        self._write(tmp_path, 'foo_20260501.csv')
        result = find_csvs_in_window(tmp_path, 'foo', date(2026, 4, 24))
        names = [p.name for p in result]
        assert names == ['foo_20260424.csv', 'foo_20260501.csv']

    def test_pattern_substring_filter(self, tmp_path):
        """Only files whose name contains the pattern substring are returned."""
        from canvigator_utils import find_csvs_in_window
        from datetime import date
        self._write(tmp_path, 'recordings_20260501.csv')
        self._write(tmp_path, 'gradebook_20260501.csv')
        result = find_csvs_in_window(tmp_path, 'recordings', date(2026, 1, 1))
        assert [p.name for p in result] == ['recordings_20260501.csv']

    def test_exclude_substr_skips_matches(self, tmp_path):
        """Files containing exclude_substr are dropped even if they match the pattern."""
        from canvigator_utils import find_csvs_in_window
        from datetime import date
        self._write(tmp_path, 'foo_20260501.csv')
        self._write(tmp_path, 'foo_dryrun_20260501.csv')
        result = find_csvs_in_window(tmp_path, 'foo', date(2026, 1, 1), exclude_substr='dryrun')
        assert [p.name for p in result] == ['foo_20260501.csv']

    def test_missing_directory_returns_empty(self, tmp_path):
        """A non-existent data_path returns an empty list rather than raising."""
        from canvigator_utils import find_csvs_in_window
        from datetime import date
        result = find_csvs_in_window(tmp_path / 'does_not_exist', 'foo', date(2026, 1, 1))
        assert result == []

    def test_results_sorted_ascending_by_date(self, tmp_path):
        """Output is sorted ascending so callers can reason chronologically."""
        from canvigator_utils import find_csvs_in_window
        from datetime import date
        self._write(tmp_path, 'foo_20260501.csv')
        self._write(tmp_path, 'foo_20260424.csv')
        self._write(tmp_path, 'foo_20260427.csv')
        result = find_csvs_in_window(tmp_path, 'foo', date(2026, 4, 1))
        assert [p.name for p in result] == ['foo_20260424.csv', 'foo_20260427.csv', 'foo_20260501.csv']


class TestFindRecentSubmissionCsvs:
    """Tests for canvigator_quiz._findRecentSubmissionCsvs."""

    SUFFIXES = ('all_submissions', 'all_subs_by_question', 'all_subs_and_events')

    def _write_csv(self, tmp_path, name, age_seconds=0):
        """Write a CSV in ``tmp_path``; backdate its mtime by ``age_seconds`` if positive."""
        import os
        import time
        path = tmp_path / name
        path.write_text('id\n1\n')
        if age_seconds > 0:
            ts = time.time() - age_seconds
            os.utime(path, (ts, ts))

    def _write_all_three(self, tmp_path, prefix='quiz', quiz_id=123, date_str='20260510', age_seconds=0):
        """Write all three submission CSVs for ``prefix{quiz_id}`` with the given mtime offset."""
        for suffix in self.SUFFIXES:
            self._write_csv(tmp_path, f"{prefix}{quiz_id}_{suffix}_{date_str}.csv", age_seconds=age_seconds)

    def test_all_three_fresh_returns_paths(self, tmp_path):
        """When all three CSVs exist with current mtimes, the helper returns a 3-tuple of Paths."""
        from canvigator_quiz import _findRecentSubmissionCsvs
        self._write_all_three(tmp_path)
        result = _findRecentSubmissionCsvs(tmp_path, 'quiz', 123, max_age_minutes=10)
        assert result is not None
        assert len(result) == 3
        for p, suffix in zip(result, self.SUFFIXES):
            assert suffix in p.name

    def test_missing_one_returns_none(self, tmp_path):
        """If any of the three CSVs is absent, the helper returns None."""
        from canvigator_quiz import _findRecentSubmissionCsvs
        self._write_csv(tmp_path, 'quiz123_all_submissions_20260510.csv')
        self._write_csv(tmp_path, 'quiz123_all_subs_by_question_20260510.csv')
        # all_subs_and_events deliberately missing
        assert _findRecentSubmissionCsvs(tmp_path, 'quiz', 123, max_age_minutes=10) is None

    def test_one_stale_returns_none(self, tmp_path):
        """A single file outside the freshness window invalidates the cache."""
        from canvigator_quiz import _findRecentSubmissionCsvs
        self._write_csv(tmp_path, 'quiz123_all_submissions_20260510.csv')
        self._write_csv(tmp_path, 'quiz123_all_subs_by_question_20260510.csv')
        # 15 minutes is past the 10-minute window
        self._write_csv(tmp_path, 'quiz123_all_subs_and_events_20260510.csv', age_seconds=15 * 60)
        assert _findRecentSubmissionCsvs(tmp_path, 'quiz', 123, max_age_minutes=10) is None

    def test_all_stale_returns_none(self, tmp_path):
        """All-stale CSVs return None."""
        from canvigator_quiz import _findRecentSubmissionCsvs
        self._write_all_three(tmp_path, age_seconds=20 * 60)
        assert _findRecentSubmissionCsvs(tmp_path, 'quiz', 123, max_age_minutes=10) is None

    def test_respects_quiz_prefix(self, tmp_path):
        """A non-default quiz_prefix is honored when looking up files."""
        from canvigator_quiz import _findRecentSubmissionCsvs
        self._write_all_three(tmp_path, prefix='midterm')
        # Default prefix should miss
        assert _findRecentSubmissionCsvs(tmp_path, 'quiz', 123, max_age_minutes=10) is None
        # Matching prefix should hit
        result = _findRecentSubmissionCsvs(tmp_path, 'midterm', 123, max_age_minutes=10)
        assert result is not None

    def test_picks_latest_dated_per_pattern(self, tmp_path):
        """When multiple dated copies exist, the helper inherits ``find_latest_csv``'s newest-wins semantics."""
        from canvigator_quiz import _findRecentSubmissionCsvs
        # Older copies are stale; newer copies are fresh — newest dated should win and pass freshness.
        self._write_all_three(tmp_path, date_str='20260401', age_seconds=60 * 60)
        self._write_all_three(tmp_path, date_str='20260510')
        result = _findRecentSubmissionCsvs(tmp_path, 'quiz', 123, max_age_minutes=10)
        assert result is not None
        for p in result:
            assert '20260510' in p.name


class TestLoadQuizMisses:
    """Tests for canvigator_digest._loadQuizMisses."""

    def _write_subs(self, tmp_path, prefix, date_str, rows):
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / f"{prefix}all_subs_by_question_{date_str}.csv", index=False)

    def _write_tags(self, tmp_path, prefix, date_str, rows):
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / f"{prefix}questions_w_tags_{date_str}.csv", index=False)

    def test_per_tag_miss_count_basic(self, tmp_path):
        """A miss row credits each of its question's keywords once."""
        from canvigator_digest import _loadQuizMisses
        from datetime import date
        self._write_subs(tmp_path, 'quiz1_111_', '20260501', [
            {'id': 7, 'question_id': 1, 'points': 0, 'points_possible': 1},
            {'id': 7, 'question_id': 2, 'points': 1, 'points_possible': 1},
            {'id': 8, 'question_id': 1, 'points': 0, 'points_possible': 1},
        ])
        self._write_tags(tmp_path, 'quiz1_111_', '20260501', [
            {'question_id': 1, 'keywords': 'recursion, base case'},
            {'question_id': 2, 'keywords': 'sorting'},
        ])
        result = _loadQuizMisses(tmp_path, date(2026, 4, 1))
        assert len(result) == 1
        entry = result[0]
        assert entry['quiz_id'] == '111'
        assert entry['quiz_name'] == 'quiz1'
        assert entry['n_missed_per_tag'] == {'recursion': 2, 'base case': 2}

    def test_partial_credit_counts_as_miss(self, tmp_path):
        """`points < points_possible` is the miss rule — partial credit counts."""
        from canvigator_digest import _loadQuizMisses
        from datetime import date
        self._write_subs(tmp_path, 'quiz1_111_', '20260501', [
            {'id': 7, 'question_id': 1, 'points': 0.5, 'points_possible': 1},
        ])
        self._write_tags(tmp_path, 'quiz1_111_', '20260501', [
            {'question_id': 1, 'keywords': 'recursion'},
        ])
        result = _loadQuizMisses(tmp_path, date(2026, 4, 1))
        assert result[0]['n_missed_per_tag']['recursion'] == 1

    def test_quiz_without_tags_csv_is_skipped(self, tmp_path):
        """A quiz with subs in window but no questions_w_tags CSV is skipped with a warning."""
        from canvigator_digest import _loadQuizMisses
        from datetime import date
        self._write_subs(tmp_path, 'quiz1_111_', '20260501', [
            {'id': 7, 'question_id': 1, 'points': 0, 'points_possible': 1},
        ])
        result = _loadQuizMisses(tmp_path, date(2026, 4, 1))
        assert result == []

    def test_subs_outside_window_excluded(self, tmp_path):
        """Subs files dated before since_date are not loaded."""
        from canvigator_digest import _loadQuizMisses
        from datetime import date
        self._write_subs(tmp_path, 'quiz1_111_', '20260301', [
            {'id': 7, 'question_id': 1, 'points': 0, 'points_possible': 1},
        ])
        self._write_tags(tmp_path, 'quiz1_111_', '20260301', [
            {'question_id': 1, 'keywords': 'recursion'},
        ])
        result = _loadQuizMisses(tmp_path, date(2026, 4, 1))
        assert result == []


class TestLoadFollowupAssessments:
    """Tests for canvigator_digest._loadFollowupAssessments."""

    def _write_assessments(self, tmp_path, prefix, rows):
        df = pd.DataFrame(rows)
        df.to_csv(tmp_path / f"{prefix}followup_assessments.csv", index=False)

    def test_filters_to_window_and_struggling_rows(self, tmp_path):
        """Only fail or borderline rows whose assessed_at >= since_date are kept."""
        from canvigator_digest import _loadFollowupAssessments
        from datetime import date
        self._write_assessments(tmp_path, 'quiz1_111_', [
            {'student_id': 1, 'question_id': 5, 'question_mode': 'explain',
             'result': 'fail', 'confidence': 'high',
             'transcript': 'I missed it', 'criteria_evaluations': '',
             'feedback': 'wrong on X', 'assessed_at': '2026-05-01T12:00:00+00:00'},
            {'student_id': 2, 'question_id': 5, 'question_mode': 'explain',
             'result': 'pass', 'confidence': 'high',
             'transcript': 'good answer', 'criteria_evaluations': '',
             'feedback': 'great', 'assessed_at': '2026-05-01T12:00:00+00:00'},
            {'student_id': 3, 'question_id': 5, 'question_mode': 'explain',
             'result': 'pass', 'confidence': 'borderline',
             'transcript': 'shaky', 'criteria_evaluations': '',
             'feedback': 'almost', 'assessed_at': '2026-05-01T12:00:00+00:00'},
            {'student_id': 4, 'question_id': 5, 'question_mode': 'explain',
             'result': 'fail', 'confidence': 'high',
             'transcript': 'old miss', 'criteria_evaluations': '',
             'feedback': 'wrong', 'assessed_at': '2026-03-01T12:00:00+00:00'},
        ])
        result = _loadFollowupAssessments(tmp_path, date(2026, 4, 1))
        assert len(result) == 1
        entry = result[0]
        assert entry['quiz_id'] == '111'
        assert entry['quiz_name'] == 'quiz1'
        # Pass+high is dropped; fail+high and pass+borderline are kept.
        kept_ids = {row['student_id'] for row in entry['rows_by_question'][5]}
        assert kept_ids == {1, 3}

    def test_drops_nat_rows_and_records_count(self, tmp_path):
        """Rows whose assessed_at is unparseable are dropped and counted via n_dropped_nat."""
        from canvigator_digest import _loadFollowupAssessments
        from datetime import date
        self._write_assessments(tmp_path, 'quiz1_111_', [
            {'student_id': 1, 'question_id': 5, 'question_mode': 'explain',
             'result': 'fail', 'confidence': 'high', 'transcript': '',
             'criteria_evaluations': '', 'feedback': '', 'assessed_at': 'not-a-date'},
            {'student_id': 2, 'question_id': 5, 'question_mode': 'explain',
             'result': 'fail', 'confidence': 'high', 'transcript': '',
             'criteria_evaluations': '', 'feedback': '', 'assessed_at': '2026-05-01T12:00:00+00:00'},
        ])
        result = _loadFollowupAssessments(tmp_path, date(2026, 4, 1))
        assert len(result) == 1
        assert result[0]['n_dropped_nat'] == 1


class TestBuildFollowupThemePrompt:
    """Tests for canvigator_digest._buildFollowupThemePrompt."""

    def test_explain_mode_includes_transcript(self):
        """Explain-mode prompts include the transcript and the grader feedback."""
        from canvigator_digest import _buildFollowupThemePrompt
        rows = [
            {'result': 'fail', 'confidence': 'high',
             'transcript': 'I confused base case with recursive case',
             'feedback': 'missed key idea', 'criteria_evaluations': ''},
        ]
        prompt = _buildFollowupThemePrompt('quiz1', 5, rows, 'explain')
        assert 'Mode: explain' in prompt
        assert 'I confused base case' in prompt
        assert 'missed key idea' in prompt

    def test_draw_mode_omits_transcript_includes_feedback(self):
        """Draw-mode prompts skip transcript (always empty for draw) but keep feedback."""
        from canvigator_digest import _buildFollowupThemePrompt
        rows = [
            {'result': 'fail', 'confidence': 'high',
             'transcript': '', 'feedback': 'BST drawn without root',
             'criteria_evaluations': ''},
        ]
        prompt = _buildFollowupThemePrompt('quiz1', 5, rows, 'draw')
        assert 'Mode: draw' in prompt
        assert 'BST drawn without root' in prompt

    def test_caps_rows_at_max(self):
        """Row count is capped at _MAX_FOLLOWUP_ROWS_PER_PROMPT to bound context size."""
        from canvigator_digest import _buildFollowupThemePrompt, _MAX_FOLLOWUP_ROWS_PER_PROMPT
        rows = [
            {'result': 'fail', 'confidence': 'high',
             'transcript': f'transcript {i}', 'feedback': '', 'criteria_evaluations': ''}
            for i in range(50)
        ]
        prompt = _buildFollowupThemePrompt('quiz1', 5, rows, 'explain')
        assert f"Number of struggling responses: {_MAX_FOLLOWUP_ROWS_PER_PROMPT}" in prompt
        # Student 13+ should not appear (we keep only the first 12).
        assert "Student 13" not in prompt

    def test_criteria_evaluations_rendered_as_bullets(self):
        """JSON criteria_evaluations are summarized as compact bullets in the prompt."""
        from canvigator_digest import _buildFollowupThemePrompt
        import json as _json
        crit_blob = _json.dumps({
            'pass_criteria_evaluations': [
                {'criterion': 'mentions base case', 'status': 'missing'},
                {'criterion': 'explains stack frames', 'status': 'partial'},
            ],
            'fatal_errors_evaluations': [
                {'error': 'asserts recursion is just a loop', 'status': 'present'},
            ],
        })
        rows = [{'result': 'fail', 'confidence': 'high', 'transcript': '',
                 'feedback': '', 'criteria_evaluations': crit_blob}]
        prompt = _buildFollowupThemePrompt('quiz1', 5, rows, 'explain')
        assert 'mentions base case' in prompt
        assert 'asserts recursion is just a loop' in prompt
        assert '[FATAL]' in prompt


class TestBuildDigestPriorities:
    """Tests for canvigator_digest._buildDigestPriorities."""

    def test_ranks_by_descending_miss_count(self):
        """Higher miss counts come first; tied tags break alphabetically."""
        from canvigator_digest import _buildDigestPriorities
        quiz_misses = [{
            'quiz_id': '111', 'quiz_name': 'quiz1', 'total_attempts': 10,
            'n_missed_per_tag': {'recursion': 5, 'sorting': 2, 'arrays': 5},
        }]
        result = _buildDigestPriorities(quiz_misses, {}, [], top_n=10)
        tags = [p['tag'] for p in result]
        assert tags[:3] == ['arrays', 'recursion', 'sorting']

    def test_three_sources_attributed(self):
        """A tag that appears via quiz misses + recordings credits both sources."""
        from canvigator_digest import _buildDigestPriorities
        quiz_misses = [{
            'quiz_id': '111', 'quiz_name': 'quiz1', 'total_attempts': 10,
            'n_missed_per_tag': {'recursion': 3},
        }]
        recording_results = [{
            'assignment_id': '999',
            'transcripts_with_names': [],
            'tags': ['recursion'],
            'tag_to_indices': {'recursion': [1, 2]},
            'themes_md': '',
        }]
        result = _buildDigestPriorities(quiz_misses, {}, recording_results, top_n=10)
        recursion_entry = next(p for p in result if p['tag'] == 'recursion')
        assert set(recursion_entry['sources']) == {'quiz', 'recording'}
        # 3 quiz misses + 2 recording mentions = 5
        assert recursion_entry['miss_count'] == 5

    def test_followup_themes_become_synthetic_tags(self):
        """Follow-up themes get a synthetic tag attribution since they aren't tag-keyed."""
        from canvigator_digest import _buildDigestPriorities
        followup_themes = {('111', 5): '- **Base case confusion**: students forget terminator'}
        result = _buildDigestPriorities([], followup_themes, [], top_n=10)
        assert len(result) == 1
        entry = result[0]
        assert 'quiz 111' in entry['tag']
        assert entry['sources'] == ['followup']
        assert entry['evidence_snippets'] == ['- **Base case confusion**: students forget terminator']

    def test_top_n_caps_results(self):
        """top_n trims the priorities list to that many entries."""
        from canvigator_digest import _buildDigestPriorities
        quiz_misses = [{
            'quiz_id': '111', 'quiz_name': 'quiz1', 'total_attempts': 10,
            'n_missed_per_tag': {f't{i}': i for i in range(20)},
        }]
        result = _buildDigestPriorities(quiz_misses, {}, [], top_n=3)
        assert len(result) == 3


class TestBuildDiscussionPromptCloud:
    """Privacy guardrail: cloud prompt must not leak transcripts or theme content."""

    def test_cloud_prompt_excludes_evidence_snippets(self):
        """The redacted prompt embeds tag names + counts but no snippet text."""
        from canvigator_digest import _buildDiscussionPromptCloud
        priorities = [{
            'tag': 'recursion',
            'miss_count': 7,
            'sources': ['quiz', 'followup'],
            'evidence_snippets': [
                'STUDENT_TRANSCRIPT: I confused base case with recursive call',
                'CRITERIA: missing(mentions base case)',
            ],
        }]
        prompt = _buildDiscussionPromptCloud(priorities)
        assert 'recursion' in prompt
        assert '7' in prompt
        # Privacy: no snippet text leaks through the redacted prompt.
        assert 'STUDENT_TRANSCRIPT' not in prompt
        assert 'CRITERIA' not in prompt
        assert 'base case' not in prompt

    def test_cloud_prompt_includes_theme_count_only(self):
        """The cloud prompt mentions how many evidence snippets exist, not their content."""
        from canvigator_digest import _buildDiscussionPromptCloud
        priorities = [{
            'tag': 'sorting',
            'miss_count': 4,
            'sources': ['quiz'],
            'evidence_snippets': ['snip A', 'snip B'],
        }]
        prompt = _buildDiscussionPromptCloud(priorities)
        assert '2 related theme cluster(s)' in prompt
        assert 'snip A' not in prompt
        assert 'snip B' not in prompt

    def test_local_prompt_includes_evidence(self):
        """By contrast, the local prompt does include the evidence snippets verbatim."""
        from canvigator_digest import _buildDiscussionPromptLocal
        priorities = [{
            'tag': 'recursion',
            'miss_count': 7,
            'sources': ['quiz'],
            'evidence_snippets': ['STUDENT_TRANSCRIPT: I confused base case'],
        }]
        prompt = _buildDiscussionPromptLocal(priorities)
        assert 'STUDENT_TRANSCRIPT' in prompt
        assert 'base case' in prompt


class TestSuggestDiscussionQuestions:
    """Tests for canvigator_digest._suggestDiscussionQuestions."""

    def test_empty_priorities_skips_llm_call(self):
        """An empty priorities list returns the no-gaps string and never builds a client."""
        from canvigator_digest import _suggestDiscussionQuestions
        with patch('canvigator_digest._make_client') as mk:
            result = _suggestDiscussionQuestions([], cloud_questions=False)
        mk.assert_not_called()
        assert 'no significant gaps' in result

    def test_local_path_uses_local_client(self):
        """cloud_questions=False routes through _make_client(cloud=False)."""
        from canvigator_digest import _suggestDiscussionQuestions
        priorities = [{'tag': 'recursion', 'miss_count': 3, 'sources': ['quiz'], 'evidence_snippets': []}]
        fake_client = MagicMock()
        with patch('canvigator_digest._make_client', return_value=fake_client) as mk, \
             patch('canvigator_digest._chat_with_retry',
                   return_value={'message': {'content': '- discuss recursion'}}):
            result = _suggestDiscussionQuestions(priorities, cloud_questions=False)
        mk.assert_called_once_with(cloud=False)
        assert '- discuss recursion' in result

    def test_cloud_path_uses_cloud_client(self):
        """cloud_questions=True routes through _make_client(cloud=True)."""
        from canvigator_digest import _suggestDiscussionQuestions
        priorities = [{'tag': 'recursion', 'miss_count': 3, 'sources': ['quiz'], 'evidence_snippets': ['secret']}]
        fake_client = MagicMock()
        captured = {}

        def _capture(client, **kwargs):
            captured['messages'] = kwargs.get('messages')
            return {'message': {'content': '- discuss recursion'}}

        with patch('canvigator_digest._make_client', return_value=fake_client) as mk, \
             patch('canvigator_digest._chat_with_retry', side_effect=_capture):
            _suggestDiscussionQuestions(priorities, cloud_questions=True)
        mk.assert_called_once_with(cloud=True)
        # Privacy guardrail: the secret evidence snippet must not appear in the user message.
        user_msg = captured['messages'][1]['content']
        assert 'secret' not in user_msg


class TestRenderDigest:
    """Tests for canvigator_digest._renderDigest."""

    def test_five_sections_present_in_order(self):
        """The rendered Markdown contains all five sections in the documented order."""
        from canvigator_digest import _renderDigest
        from datetime import date
        out = _renderDigest(
            course_label='CSI-3300',
            since_date=date(2026, 4, 24),
            today=date(2026, 5, 1),
            quiz_misses=[],
            followup_themes={},
            followup_blocks=[],
            recording_results=[],
            discussion_md='- example question',
            days=7,
        )
        # All four data-source sections plus the discussion section are present.
        for header in ['## Quiz performance', '## Follow-up reply themes',
                       '## Media-recording themes', '## Suggested in-class discussion questions']:
            assert header in out
        # And they appear in order.
        idx_quiz = out.index('## Quiz performance')
        idx_followup = out.index('## Follow-up reply themes')
        idx_recording = out.index('## Media-recording themes')
        idx_discussion = out.index('## Suggested in-class discussion questions')
        assert idx_quiz < idx_followup < idx_recording < idx_discussion

    def test_header_surfaces_window_dates(self):
        """The header line includes the window range and day count."""
        from canvigator_digest import _renderDigest
        from datetime import date
        out = _renderDigest(
            course_label='CSI-3300',
            since_date=date(2026, 4, 24),
            today=date(2026, 5, 1),
            quiz_misses=[],
            followup_themes={},
            followup_blocks=[],
            recording_results=[],
            discussion_md='',
            days=7,
        )
        assert '2026-04-24' in out
        assert '2026-05-01' in out
        assert '7 day' in out

    def test_quiz_performance_table_renders_with_data(self):
        """When quiz misses exist, the section renders a Markdown table sorted by descending miss count."""
        from canvigator_digest import _renderDigest
        from datetime import date
        from collections import Counter
        quiz_misses = [{
            'quiz_id': '111', 'quiz_name': 'quiz1', 'total_attempts': 10,
            'n_missed_per_tag': Counter({'recursion': 5, 'sorting': 1}),
        }]
        out = _renderDigest(
            'CSI-3300', date(2026, 4, 24), date(2026, 5, 1),
            quiz_misses, {}, [], [], 'discussion', 7,
        )
        assert '| Tag | Total misses | Contributing quizzes |' in out
        # recursion (5) listed before sorting (1)
        idx_r = out.index('| recursion |')
        idx_s = out.index('| sorting |')
        assert idx_r < idx_s


class TestPrepClassDigestEmptyWindow:
    """Tests for the orchestrator's empty-window guardrail."""

    def test_empty_window_writes_no_file(self, tmp_path, capsys):
        """With no CSVs in the data directory, the orchestrator returns None and writes nothing."""
        from canvigator_digest import prepClassDigest
        course = SimpleNamespace(
            config=SimpleNamespace(data_path=tmp_path),
            canvas_course=SimpleNamespace(course_code='CSI-3300', name='Test'),
        )
        result = prepClassDigest(course, days=7, cloud_questions=False)
        assert result is None
        # No output file created.
        assert list(tmp_path.glob('class_digest_*.md')) == []
        # User-facing message confirms the no-op.
        captured = capsys.readouterr()
        assert 'nothing to digest' in captured.out


# ---------------------------------------------------------------------------
# canvigator_help tests (per-task --help)
# ---------------------------------------------------------------------------

class TestPerTaskHelp:
    """Integrity + rendering + CLI-dispatch tests for the per-task help system."""

    def _all_task_names(self):
        """Pull the list of tasks from canvigator.py without executing it."""
        # canvigator.py is a top-level script; importing it would attempt to
        # parse sys.argv. We instead read task_groups via runpy in a guarded
        # mode that stops before the parse loop.
        import ast
        src = Path(__file__).parent.joinpath('canvigator.py').read_text()
        tree = ast.parse(src)
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == 'task_groups':
                        # task_groups = [(header, [(name, desc), ...]), ...]
                        for group_tuple in node.value.elts:
                            items_list = group_tuple.elts[1]
                            for entry in items_list.elts:
                                names.append(entry.elts[0].value)
        return names

    def test_every_task_has_help_entry(self):
        """Every task name in canvigator.py has a TASK_HELP entry."""
        from canvigator_help import TASK_HELP
        task_names = self._all_task_names()
        assert task_names, "expected to find tasks in canvigator.py"
        missing = sorted(set(task_names) - set(TASK_HELP))
        assert not missing, f"missing TASK_HELP entries: {missing}"
        extra = sorted(set(TASK_HELP) - set(task_names))
        assert not extra, f"TASK_HELP has stale entries: {extra}"

    def test_required_fields_non_empty(self):
        """Every entry has non-empty description, inputs, outputs, examples."""
        from canvigator_help import TASK_HELP
        for name, entry in TASK_HELP.items():
            assert entry.get('description', '').strip(), f"{name}: empty description"
            assert entry.get('inputs'), f"{name}: empty inputs"
            assert entry.get('outputs'), f"{name}: empty outputs"
            assert entry.get('examples'), f"{name}: empty examples"
            # These keys must exist (may be empty lists) so the renderer can
            # iterate without KeyError.
            for key in ('prerequisites', 'flags', 'run_before', 'run_after'):
                assert key in entry, f"{name}: missing key '{key}'"
                assert isinstance(entry[key], list), f"{name}: '{key}' must be a list"

    def test_flag_references_resolve(self):
        """Every flag listed in any entry is defined in FLAG_DESCRIPTIONS."""
        from canvigator_help import TASK_HELP, FLAG_DESCRIPTIONS
        for name, entry in TASK_HELP.items():
            for flag in entry.get('flags', []):
                assert flag in FLAG_DESCRIPTIONS, (
                    f"{name}: references unknown flag '{flag}'"
                )

    def test_run_before_after_targets_exist(self):
        """run_before / run_after entries are themselves valid task names."""
        from canvigator_help import TASK_HELP
        all_tasks = set(TASK_HELP)
        for name, entry in TASK_HELP.items():
            for ref in entry.get('run_before', []) + entry.get('run_after', []):
                assert ref in all_tasks, (
                    f"{name}: run_before/run_after references unknown task '{ref}'"
                )

    def test_print_task_help_render(self, capsys):
        """The rendered output contains the description, prereq, flags, and example."""
        from canvigator_help import print_task_help
        print_task_help('send-quiz-reminder')
        out = capsys.readouterr().out
        assert 'send-quiz-reminder' in out
        assert 'Send Canvas reminder messages' in out
        assert 'Prerequisites:' in out
        assert 'get-quiz-questions' in out  # cited as a prerequisite
        assert '--all' in out
        assert '--dry-run' in out
        assert 'Examples:' in out
        assert 'python canvigator.py' in out
        assert 'Workflow:' in out
        assert 'Run before this:' in out
        # Universal --crn flag should be auto-appended to the Flags section.
        assert '--crn' in out

    def test_print_task_help_unknown_task(self, capsys):
        """Unknown task names render a friendly message instead of crashing."""
        from canvigator_help import print_task_help
        print_task_help('not-a-real-task')
        out = capsys.readouterr().out
        assert 'not-a-real-task' in out

    def _run_cli(self, *args):
        """Invoke canvigator.py as a subprocess; return (rc, stdout, stderr).

        Sets CANVAS_URL/CANVAS_TOKEN to empty so the subprocess can prove the
        --help path exits before constructing the Canvas client.
        """
        env = {'PATH': '/usr/bin:/bin', 'PYTHONPATH': ''}
        # Inherit some basics so Python can find its stdlib in the test runner.
        import os
        for k in ('PATH', 'HOME', 'PYTHONPATH', 'PYTHONHOME'):
            if k in os.environ:
                env[k] = os.environ[k]
        # Explicitly NOT setting CANVAS_URL / CANVAS_TOKEN — the help path
        # must short-circuit before the env-var check.
        env['CANVAS_URL'] = ''
        env['CANVAS_TOKEN'] = ''
        import sys
        cwd = Path(__file__).parent
        result = subprocess.run(
            [sys.executable, 'canvigator.py', *args],
            capture_output=True, text=True, cwd=cwd, env=env, timeout=15,
        )
        return result.returncode, result.stdout, result.stderr

    def test_cli_dispatch_global_help(self):
        """`canvigator.py --help` exits 0 and prints the task list."""
        rc, stdout, stderr = self._run_cli('--help')
        assert rc == 0, f"stderr={stderr!r}"
        assert 'Usage: canvigator.py' in stdout
        # The new footer line should be present.
        assert "<task> --help" in stdout
        # A representative task name should appear.
        assert 'send-quiz-reminder' in stdout

    def test_cli_dispatch_task_help(self):
        """`canvigator.py <task> --help` exits 0 and prints task-specific help."""
        rc, stdout, stderr = self._run_cli('create-pairs', '--help')
        assert rc == 0, f"stderr={stderr!r}"
        assert 'create-pairs' in stdout
        assert 'Prerequisites:' in stdout
        assert 'present_*.csv' in stdout

    def test_cli_dispatch_task_help_short_form(self):
        """`canvigator.py <task> -h` works the same as --help."""
        rc, stdout, stderr = self._run_cli('send-quiz-reminder', '-h')
        assert rc == 0, f"stderr={stderr!r}"
        assert 'send-quiz-reminder' in stdout
        assert 'Workflow:' in stdout

    def test_cli_invalid_task_with_help(self):
        """`canvigator.py bogus --help` still rejects the unknown task."""
        rc, stdout, _ = self._run_cli('bogus-task', '--help')
        assert rc == 1
        assert 'Invalid task' in stdout
