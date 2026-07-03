"""
Tests for rag_datasets.py — deterministic random question selection.

Loaders use a seeded shuffle (random.Random(seed)) to select n questions
deterministically. Corpus remains sequential and complete regardless of seed.
All tests mock load_dataset to avoid network calls.
"""

import random
import sys
import os
from unittest.mock import patch

import pytest

# Make rag_datasets importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ragbench_row(question, response, documents, **extra):
    """Helper to build a single RAGBench dataset row dict."""
    return {"question": question, "response": response, "documents": documents, **extra}


def _make_fetaqa_row(question, answer, headers, data_rows, page_title="TestTable", section_title="TestSection"):
    """Helper to build a single FeTaQA dataset row dict."""
    return {
        "question": question,
        "answer": answer,
        "table_array": [headers] + data_rows,
        "table_page_title": page_title,
        "table_section_title": section_title,
    }


def _make_stratrag_row(query, reference_answer, doc_pool, **extra):
    """Helper to build a single StratRAG dataset row dict."""
    return {"query": query, "reference_answer": reference_answer, "doc_pool": doc_pool, **extra}


# ---------------------------------------------------------------------------
# RAGBench fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_ragbench_sequential():
    """5 RAGBench rows with distinct documents — sequential."""
    return [
        _make_ragbench_row("Q0", "A0", ["doc_a"]),
        _make_ragbench_row("Q1", "A1", ["doc_b1", "doc_b2"]),
        _make_ragbench_row("Q2", "A2", ["doc_c1", "doc_c2", "doc_c3"]),
        _make_ragbench_row("Q3", "A3", ["doc_d"]),
        _make_ragbench_row("Q4", "A4", ["doc_e1", "doc_e2"]),
    ]


@pytest.fixture
def mock_ragbench_sequential_dup():
    """Rows with some overlapping document text."""
    return [
        _make_ragbench_row("Q0", "A0", ["shared", "unique_a"]),
        _make_ragbench_row("Q1", "A1", ["shared", "unique_b"]),
        _make_ragbench_row("Q2", "A2", ["unique_c"]),
    ]


# ---------------------------------------------------------------------------
# FeTaQA fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_fetaqa_sparse():
    """FeTaQA dataset with one table: 3 headers, 1 data row of only 2 values."""
    return [_make_fetaqa_row(
        "What is X?", "X is Y",
        headers=["Col1", "Col2", "Col3"],
        data_rows=[["val1", "val2"]],
    )]


@pytest.fixture
def mock_fetaqa_normal():
    """FeTaQA dataset with one table: matching headers and data."""
    return [_make_fetaqa_row(
        "Who?", "Person",
        headers=["Name", "Age", "City"],
        data_rows=[["Alice", "30", "NYC"]],
        page_title="People",
        section_title="Employees",
    )]


@pytest.fixture
def mock_fetaqa_long():
    """FeTaQA dataset with one table: data row longer than headers."""
    return [_make_fetaqa_row(
        "Q?", "A",
        headers=["Col1", "Col2"],
        data_rows=[["a", "b", "c", "d"]],
    )]


@pytest.fixture
def mock_fetaqa_sequential():
    """3 FeTaQA tables with 2+ rows each (no header-only)."""
    return [
        _make_fetaqa_row("Q0", "A0", ["H1"], [["r0"]], page_title="T0", section_title="S0"),
        _make_fetaqa_row("Q1", "A1", ["H1", "H2"], [["a", "b"], ["c", "d"]], page_title="T1", section_title="S1"),
        _make_fetaqa_row("Q2", "A2", ["Col"], [["x"], ["y"]], page_title="T2", section_title="S2"),
    ]


# ---------------------------------------------------------------------------
# StratRAG fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_stratrag_sequential():
    """4 StratRAG rows with distinct doc pools."""
    return [
        _make_stratrag_row("Q0", "A0", [{"text": "doc_a", "source": "sA"}]),
        _make_stratrag_row("Q1", "A1", [{"text": "doc_b1", "source": "sB"}, {"text": "doc_b2", "source": "sB"}]),
        _make_stratrag_row("Q2", "A2", [{"text": "doc_c1", "source": "sC"}, {"text": "doc_c2", "source": "sC"}, {"text": "doc_c3", "source": "sC"}]),
        _make_stratrag_row("Q3", "A3", [{"text": "doc_d", "source": "sD"}]),
    ]


@pytest.fixture
def mock_stratrag_empty_text():
    """Empty text and missing text fields should be filtered out."""
    return [
        _make_stratrag_row("Q0", "A0", [{"text": "", "source": "sA"}, {"text": "valid", "source": "sA"}]),
        _make_stratrag_row("Q1", "A1", [{"source": "sB"}, {"text": "fill1", "source": "sB"}]),
    ]


# ---------------------------------------------------------------------------
# Jagged row padding (FeTaQA — formatting logic retained)
# ---------------------------------------------------------------------------

class TestJaggedRowPadding:
    """Table formatting logic is preserved from before simplification."""

    def test_fetaqa_sparse_row_is_padded(self, mock_fetaqa_sparse):
        """3-col header + 2-col data row → missing third value is padded."""
        from rag_datasets import load_fetaqa

        with patch("rag_datasets.load_dataset", return_value=mock_fetaqa_sparse):
            rows, corpus = load_fetaqa(n=1)

        assert len(rows) == 1
        assert rows[0]["question"] == "What is X?"
        assert len(corpus) == 1
        assert "Col3:" in corpus[0]

    def test_fetaqa_equal_row_normal(self, mock_fetaqa_normal):
        """Matching header/data → one document per table row."""
        from rag_datasets import load_fetaqa

        with patch("rag_datasets.load_dataset", return_value=mock_fetaqa_normal):
            rows, corpus = load_fetaqa(n=1)

        assert len(rows) == 1
        assert len(corpus) == 1
        doc = corpus[0]
        assert "Name: Alice" in doc
        assert "Age: 30" in doc
        assert "City: NYC" in doc

    def test_fetaqa_long_row_clipped(self, mock_fetaqa_long):
        """Data row longer than headers → doc uses only header count of columns."""
        from rag_datasets import load_fetaqa

        with patch("rag_datasets.load_dataset", return_value=mock_fetaqa_long):
            rows, corpus = load_fetaqa(n=1)

        assert len(corpus) == 1
        doc = corpus[0]
        assert "Col1: a" in doc
        assert "Col2: b" in doc
        assert "Col1: c" not in doc
        assert "Col2: d" not in doc


# ---------------------------------------------------------------------------
# Sequential corpus ordering tests (replaces anchor-first tests)
# ---------------------------------------------------------------------------

class TestRagbenchSequential:
    """After simplification: corpus is sequential, no dedup, no anchor-first."""

    def test_ragbench_sequential_corpus(self, mock_ragbench_sequential):
        """Corpus is all docs from all rows in sequential order."""
        from rag_datasets import load_ragbench

        with patch("rag_datasets.load_dataset", return_value=mock_ragbench_sequential):
            rows, corpus = load_ragbench(n=3)

        assert len(rows) == 3
        # Sequential: first doc from first row should come first
        assert corpus[0] == "doc_a"
        # All 9 docs present
        assert len(corpus) == 9
        # No internal field leak
        assert list(rows[0].keys()) == ["question", "grading_notes"]

    def test_ragbench_no_dedup(self, mock_ragbench_sequential_dup):
        """Documents are NOT deduplicated — 'shared' appears twice."""
        from rag_datasets import load_ragbench

        with patch("rag_datasets.load_dataset", return_value=mock_ragbench_sequential_dup):
            rows, corpus = load_ragbench(n=3)

        assert len(rows) == 3
        assert corpus.count("shared") == 2, "No dedup: 'shared' should appear twice"


class TestFetaqaSequential:
    """After simplification: fetaqa corpus is sequential, no dedup, no _source_title."""

    def test_fetaqa_sequential_corpus(self, mock_fetaqa_sequential):
        """Corpus returns one doc per table row, sequential."""
        from rag_datasets import load_fetaqa

        with patch("rag_datasets.load_dataset", return_value=mock_fetaqa_sequential):
            rows, corpus = load_fetaqa(n=3)

        assert len(rows) == 3
        assert len(corpus) == 3
        # First doc from first table
        assert "r0" in corpus[0]
        # No _source_title leak
        for r in rows:
            assert "_source_title" not in r
            assert set(r.keys()) == {"question", "grading_notes"}


class TestStratragSequential:
    """After simplification: stratrag corpus is sequential, no dedup, no [source] prefix."""

    def test_stratrag_sequential_corpus(self, mock_stratrag_sequential):
        """Corpus is all docs from all rows, sequential, plain text (no [source]).
        Questions are selected deterministically by seed=42, sorted by original index."""
        from rag_datasets import load_stratrag

        with patch("rag_datasets.load_dataset", return_value=mock_stratrag_sequential):
            rows, corpus = load_stratrag(n=2)

        assert len(rows) == 2
        # Selected rows must be sorted by original index
        questions = [r["question"] for r in rows]
        indices = [int(q[1:]) for q in questions]
        assert indices == sorted(indices), "Questions must be in original index order"
        # Fields must not leak internal keys
        for r in rows:
            assert set(r.keys()) == {"question", "grading_notes"}
        # Corpus: all docs, sequential, first doc from first row
        assert len(corpus) >= 4
        assert corpus[0] == "doc_a"
        # No [source]\ntext format — just plain text
        assert "\n" not in corpus[0], "No [source] prefix → plain text only"

    def test_stratrag_empty_text_filtered(self, mock_stratrag_empty_text):
        """Empty text and missing 'text' key are filtered out."""
        from rag_datasets import load_stratrag

        with patch("rag_datasets.load_dataset", return_value=mock_stratrag_empty_text):
            rows, corpus = load_stratrag(n=2)

        assert len(rows) == 2
        # Empty text filtered, missing text key filtered → only 'valid' and 'fill1'
        assert "" not in corpus
        assert "valid" in corpus
        assert "fill1" in corpus


# ---------------------------------------------------------------------------
# Shortfall warning (retained from original)
# ---------------------------------------------------------------------------

class TestShortfallWarning:

    def test_ragbench_shortfall_warning(self, capsys):
        """n=5, 2 rows in dataset → warning printed."""
        from rag_datasets import load_ragbench

        mock_ds = [
            {"question": "Q0", "response": "A0", "documents": ["d0"]},
            {"question": "Q1", "response": "A1", "documents": ["d1"]},
        ]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_ragbench(n=5)
        assert len(rows) == 2
        captured = capsys.readouterr()
        assert "requested 5 questions but only 2 available" in captured.out

    def test_fetaqa_shortfall_warning(self, capsys):
        """fetaqa shortfall warning."""
        from rag_datasets import load_fetaqa

        mock_ds = [
            _make_fetaqa_row("Q0", "A0", ["H1"], [["r1"]]),
        ]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_fetaqa(n=3)
        assert len(rows) == 1
        captured = capsys.readouterr()
        assert "requested 3 questions but only 1 available" in captured.out

    def test_stratrag_shortfall_warning(self, capsys):
        """stratrag shortfall warning."""
        from rag_datasets import load_stratrag

        mock_ds = [
            {"query": "Q0", "reference_answer": "A0", "doc_pool": [{"text": "d0", "source": "s"}]},
        ]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_stratrag(n=3)
        assert len(rows) == 1
        captured = capsys.readouterr()
        assert "requested 3 questions but only 1 available" in captured.out

    def test_no_warning_when_n_fits(self, capsys):
        """No false warning when n <= available rows."""
        from rag_datasets import load_ragbench

        mock_ds = [
            {"question": "Q0", "response": "A0", "documents": ["d0"]},
            {"question": "Q1", "response": "A1", "documents": ["d1"]},
        ]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_ragbench(n=2)
        assert len(rows) == 2
        captured = capsys.readouterr()
        assert "requested" not in captured.out


# ---------------------------------------------------------------------------
# Zero questions / offset beyond end
# ---------------------------------------------------------------------------

class TestZeroQuestions:
    """n=0 returns empty question list, but corpus is still full."""

    def test_ragbench_zero_questions(self, mock_ragbench_sequential):
        from rag_datasets import load_ragbench

        with patch("rag_datasets.load_dataset", return_value=mock_ragbench_sequential):
            rows, corpus = load_ragbench(n=0)
        assert rows == []
        assert len(corpus) == 9  # full corpus

    def test_fetaqa_zero_questions(self, mock_fetaqa_sequential):
        from rag_datasets import load_fetaqa

        with patch("rag_datasets.load_dataset", return_value=mock_fetaqa_sequential):
            rows, corpus = load_fetaqa(n=0)
        assert rows == []
        assert len(corpus) == 3

    def test_stratrag_zero_questions(self, mock_stratrag_sequential):
        from rag_datasets import load_stratrag

        with patch("rag_datasets.load_dataset", return_value=mock_stratrag_sequential):
            rows, corpus = load_stratrag(n=0)
        assert rows == []
        assert len(corpus) >= 4


# ---------------------------------------------------------------------------
# load_one / load_all wrappers
# ---------------------------------------------------------------------------

class TestLoadOne:
    """load_one dispatches to the correct loader with seed passthrough."""

    def test_load_one_ragbench(self, mock_ragbench_sequential):
        from rag_datasets import load_one

        with patch("rag_datasets.load_dataset", return_value=mock_ragbench_sequential):
            rows, corpus = load_one("ragbench", n=2, seed=42)

        assert len(rows) == 2
        assert len(corpus) == 9

    def test_load_one_unknown_dataset(self):
        from rag_datasets import load_one

        with pytest.raises(ValueError, match="Unknown dataset"):
            load_one("nonexistent")

    def test_load_one_default_seed(self, mock_ragbench_sequential):
        """load_one defaults seed to 42 when not specified."""
        from rag_datasets import load_one

        with patch("rag_datasets.load_dataset", return_value=mock_ragbench_sequential):
            rows_a, corpus_a = load_one("ragbench", n=2)
            rows_b, corpus_b = load_one("ragbench", n=2)

        # Both calls with default seed=42 should be identical
        assert [r["question"] for r in rows_a] == [r["question"] for r in rows_b]

    def test_load_one_with_offset_typeerror(self, mock_ragbench_sequential):
        """load_one raises TypeError when offset keyword is passed."""
        from rag_datasets import load_one

        with patch("rag_datasets.load_dataset", return_value=mock_ragbench_sequential):
            with pytest.raises(TypeError):
                load_one("ragbench", n=2, offset=2)


class TestLoadAll:
    """load_all loads all three datasets, forwarding seed."""

    def test_load_all(self):
        from rag_datasets import load_all

        mock_ragbench_ds = [_make_ragbench_row("Q0", "A0", ["d0"])]
        mock_fetaqa_ds = [_make_fetaqa_row("Q0", "A0", ["H1"], [["r1"]])]
        mock_stratrag_ds = [{"query": "Q0", "reference_answer": "A0", "doc_pool": [{"text": "d0", "source": "s"}]}]

        def mock_load_dataset(name, *args, **kwargs):
            if "ragbench" in str(name):
                return mock_ragbench_ds
            elif "FeTaQA" in str(name):
                return mock_fetaqa_ds
            elif "StratRAG" in str(name):
                return mock_stratrag_ds
            return []

        with patch("rag_datasets.load_dataset", side_effect=mock_load_dataset):
            result = load_all(n=1, seed=42)

        assert set(result.keys()) == {"ragbench", "fetaqa", "stratrag"}
        for name, (rows, corpus) in result.items():
            assert len(rows) == 1
            assert len(corpus) >= 1

    def test_load_all_deterministic(self):
        """load_all with same seed returns identical results."""
        from rag_datasets import load_all

        mock_ds = [_make_ragbench_row(f"Q{i}", f"A{i}", [f"d{i}"]) for i in range(5)]
        mock_ds_fetaqa = [_make_fetaqa_row(f"Q{i}", f"A{i}", ["H1"], [[f"r{i}"]]) for i in range(3)]
        mock_ds_stratrag = [{"query": f"Q{i}", "reference_answer": f"A{i}", "doc_pool": [{"text": f"d{i}", "source": "s"}]} for i in range(3)]

        def mock_load_dataset(name, *args, **kwargs):
            if "ragbench" in str(name):
                return mock_ds
            elif "FeTaQA" in str(name):
                return mock_ds_fetaqa
            elif "StratRAG" in str(name):
                return mock_ds_stratrag
            return []

        with patch("rag_datasets.load_dataset", side_effect=mock_load_dataset):
            result_a = load_all(n=2, seed=42)
            result_b = load_all(n=2, seed=42)

        for name in result_a:
            rows_a, _ = result_a[name]
            rows_b, _ = result_b[name]
            assert [r["question"] for r in rows_a] == [r["question"] for r in rows_b]


# ---------------------------------------------------------------------------
# Seed determinism tests (spec: rag-eval-dataset-loading)
# ---------------------------------------------------------------------------

class TestSeedDeterminism:
    """Deterministic random question selection via seed parameter."""

    def test_same_seed_produces_identical_questions(self):
        """Same seed → identical question lists and identical corpus."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_row(f"Q{i}", f"A{i}", [f"d{i}"]) for i in range(10)]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows_a, corpus_a = load_ragbench(n=5, seed=42)
            rows_b, corpus_b = load_ragbench(n=5, seed=42)

        questions_a = [r["question"] for r in rows_a]
        questions_b = [r["question"] for r in rows_b]
        assert questions_a == questions_b, f"Same seed should produce identical questions"
        assert corpus_a == corpus_b, "Corpus must be identical"

    def test_different_seeds_produce_different_questions(self):
        """Different seeds → different question lists with high probability."""
        from rag_datasets import load_ragbench

        # 100 rows to make collision extremely unlikely with n=5
        mock_ds = [_make_ragbench_row(f"Q{i}", f"A{i}", [f"d{i}"]) for i in range(100)]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows_a, _ = load_ragbench(n=5, seed=42)
            rows_b, _ = load_ragbench(n=5, seed=99)

        questions_a = [r["question"] for r in rows_a]
        questions_b = [r["question"] for r in rows_b]
        assert questions_a != questions_b, (
            f"Different seeds should produce different questions (probabilistic)"
        )

    def test_questions_returned_sorted_by_original_index(self):
        """Selected questions appear sorted by their original dataset row index."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_row(f"Q{i}", f"A{i}", [f"d{i}"]) for i in range(10)]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=5, seed=42)

        questions = [r["question"] for r in rows]
        # Verify they are sorted: Q0 < Q1 < Q2 ...
        indices = [int(q[1:]) for q in questions]
        assert indices == sorted(indices), (
            f"Questions must be sorted by original index, got {questions}"
        )

    def test_corpus_unchanged_regardless_of_seed(self):
        """Corpus is complete and sequential regardless of seed value."""
        from rag_datasets import load_ragbench

        mock_ds = [
            _make_ragbench_row("Q0", "A0", ["d0", "d1"]),
            _make_ragbench_row("Q1", "A1", ["d2"]),
            _make_ragbench_row("Q2", "A2", ["d3", "d4"]),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            _, corpus_42 = load_ragbench(n=1, seed=42)
            _, corpus_99 = load_ragbench(n=1, seed=99)
            _, corpus_0 = load_ragbench(n=2, seed=0)

        expected = ["d0", "d1", "d2", "d3", "d4"]
        assert corpus_42 == expected, f"seed=42 corpus mismatch: {corpus_42}"
        assert corpus_99 == expected, f"seed=99 corpus mismatch: {corpus_99}"
        assert corpus_0 == expected, f"seed=0 corpus mismatch: {corpus_0}"

    def test_seed_edge_cases(self):
        """Zero and negative seeds are valid and deterministic."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_row(f"Q{i}", f"A{i}", [f"d{i}"]) for i in range(10)]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows_0a, _ = load_ragbench(n=3, seed=0)
            rows_0b, _ = load_ragbench(n=3, seed=0)
            rows_neg_a, _ = load_ragbench(n=3, seed=-1)
            rows_neg_b, _ = load_ragbench(n=3, seed=-1)

        # Same seed must be deterministic
        assert [r["question"] for r in rows_0a] == [r["question"] for r in rows_0b]
        assert [r["question"] for r in rows_neg_a] == [r["question"] for r in rows_neg_b]


class TestSeedIsolation:
    """Loader must not mutate global random module state."""

    def test_global_random_state_unchanged_after_load(self):
        """Global random.getstate() is unchanged after any dataset load."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_row(f"Q{i}", f"A{i}", [f"d{i}"]) for i in range(10)]

        state_before = random.getstate()
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            _ = load_ragbench(n=5, seed=42)
        state_after = random.getstate()

        assert state_before == state_after, (
            "Global random state must be unchanged after loader call"
        )


# ---------------------------------------------------------------------------
# ALL_DATASETS / LOADERS constants
# ---------------------------------------------------------------------------

def test_all_datasets_constant():
    from rag_datasets import ALL_DATASETS
    assert ALL_DATASETS == ["ragbench", "fetaqa", "stratrag"]


def test_loaders_constant():
    from rag_datasets import LOADERS
    assert set(LOADERS.keys()) == {"ragbench", "fetaqa", "stratrag"}
