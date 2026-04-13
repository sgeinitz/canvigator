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


# ---------------------------------------------------------------------------
# canvigator_quiz: follow-up question helper tests
# ---------------------------------------------------------------------------

def _make_subs_by_question_df(rows):
    """Build a DataFrame matching the all_subs_by_question schema."""
    return pd.DataFrame(rows, columns=['name', 'id', 'attempt', 'question', 'question_id', 'points', 'points_possible', 'correct'])


def _make_quiz_stub():
    """Create a minimal mock object with the methods _findMostMissedQuestion and _findStudentsWhoMissed."""
    from canvigator_quiz import CanvigatorQuiz
    # We only need the unbound methods — call them with an explicit self=None
    # since they don't use self at all (only their arguments).
    return CanvigatorQuiz


class TestFindMostMissedQuestion:
    """Tests for CanvigatorQuiz._findMostMissedQuestion."""

    def _call(self, subs_rows, question_info):
        """Helper to call _findMostMissedQuestion without a full quiz instance."""
        cls = _make_quiz_stub()
        df = _make_subs_by_question_df(subs_rows)
        return cls._findMostMissedQuestion(None, df, question_info)

    def test_returns_most_missed(self):
        """Question with higher miss rate is returned."""
        question_info = {
            100: {'position': 1, 'keywords': 'topic a', 'points_possible': 1.0},
            200: {'position': 2, 'keywords': 'topic b', 'points_possible': 1.0},
        }
        rows = [
            ('A', 1, 1, 1, 100, 1.0, 1.0, True),   # student 1 got Q100 right
            ('A', 1, 1, 2, 200, 0.0, 1.0, False),   # student 1 missed Q200
            ('B', 2, 1, 1, 100, 0.0, 1.0, False),   # student 2 missed Q100
            ('B', 2, 1, 2, 200, 0.0, 1.0, False),   # student 2 missed Q200
        ]
        result = self._call(rows, question_info)
        assert result == 200  # both students missed Q200, only one missed Q100

    def test_returns_none_when_all_perfect(self):
        """Returns None when no questions are missed."""
        question_info = {
            100: {'position': 1, 'keywords': 'topic a', 'points_possible': 1.0},
        }
        rows = [
            ('A', 1, 1, 1, 100, 1.0, 1.0, True),
            ('B', 2, 1, 1, 100, 1.0, 1.0, True),
        ]
        result = self._call(rows, question_info)
        assert result is None

    def test_uses_latest_attempt(self):
        """Only the latest attempt per student counts toward miss rate."""
        question_info = {
            100: {'position': 1, 'keywords': 'topic a', 'points_possible': 1.0},
        }
        rows = [
            # Student 1: missed on attempt 1, got it right on attempt 2
            ('A', 1, 1, 1, 100, 0.0, 1.0, False),
            ('A', 1, 2, 1, 100, 1.0, 1.0, True),
        ]
        result = self._call(rows, question_info)
        assert result is None  # latest attempt is perfect


class TestFindStudentsWhoMissed:
    """Tests for CanvigatorQuiz._findStudentsWhoMissed."""

    def _call(self, question_id, subs_rows, question_info):
        """Helper to call _findStudentsWhoMissed without a full quiz instance."""
        cls = _make_quiz_stub()
        df = _make_subs_by_question_df(subs_rows)
        return cls._findStudentsWhoMissed(None, question_id, df, question_info)

    def test_returns_students_who_missed(self):
        """Only students who scored below points_possible are returned."""
        question_info = {100: {'position': 1, 'keywords': 'topic a', 'points_possible': 1.0}}
        rows = [
            ('A', 1, 1, 1, 100, 0.0, 1.0, False),
            ('B', 2, 1, 1, 100, 1.0, 1.0, True),
            ('C', 3, 1, 1, 100, 0.5, 1.0, False),
        ]
        result = self._call(100, rows, question_info)
        assert sorted(result) == [1, 3]

    def test_uses_latest_attempt(self):
        """Student who fixed the question on a later attempt is excluded."""
        question_info = {100: {'position': 1, 'keywords': 'topic a', 'points_possible': 1.0}}
        rows = [
            ('A', 1, 1, 1, 100, 0.0, 1.0, False),
            ('A', 1, 2, 1, 100, 1.0, 1.0, True),
        ]
        result = self._call(100, rows, question_info)
        assert result == []

    def test_empty_when_no_misses(self):
        """Returns empty list when all students scored perfectly."""
        question_info = {100: {'position': 1, 'keywords': 'topic a', 'points_possible': 1.0}}
        rows = [
            ('A', 1, 1, 1, 100, 1.0, 1.0, True),
        ]
        result = self._call(100, rows, question_info)
        assert result == []
