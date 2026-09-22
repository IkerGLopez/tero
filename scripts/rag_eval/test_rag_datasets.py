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


def _make_fetaqa_row(question, answer, headers, data_rows, page_title="TestTable", section_title="TestSection",
                     feta_id=0, highlighted_cell_ids=None):
    """Helper to build a single FeTaQA dataset row dict.

    ``feta_id`` and ``highlighted_cell_ids`` mirror the HF dataset fields the
    loader consumes for gold linkage (defaults keep legacy fixtures terse).
    """
    return {
        "question": question,
        "answer": answer,
        "table_array": [headers] + data_rows,
        "table_page_title": page_title,
        "table_section_title": section_title,
        "feta_id": feta_id,
        "highlighted_cell_ids": [] if highlighted_cell_ids is None else highlighted_cell_ids,
    }


def _make_stratrag_row(query, reference_answer, doc_pool, *, row_id=None,
                       gold_doc_indices=None, question_type=None, **extra):
    """Helper to build a single StratRAG dataset row dict.

    ``row_id``, ``gold_doc_indices`` and ``question_type`` mirror the HF dataset
    fields the loader consumes for row exposure and gold linkage. They are
    keyword-only and optional so legacy fixtures stay terse; ``question_type``
    nests under ``metadata`` exactly like the live dataset, and ``**extra``
    remains an escape hatch for ad-hoc fields.
    """
    row = {"query": query, "reference_answer": reference_answer, "doc_pool": doc_pool, **extra}
    if row_id is not None:
        row["id"] = row_id
    if gold_doc_indices is not None:
        row["gold_doc_indices"] = gold_doc_indices
    if question_type is not None:
        row["metadata"] = {**(row.get("metadata") or {}), "question_type": question_type}
    return row


def _make_ragbench_gold_row(question, response, documents, *, sentence_keys=None,
                            documents_sentences=None):
    """RAGBench row carrying the live gold-linkage fields (verified 2026-09-22).

    ``documents_sentences`` is the nested pair structure the HF dataset exposes:
    ``documents_sentences[doc][sentence] == [key, text]``. ``sentence_keys``
    mirrors ``all_relevant_sentence_keys`` (keys like ``"4a"`` / ``"4ab"``).
    """
    row = {"question": question, "response": response, "documents": documents}
    if sentence_keys is not None:
        row["all_relevant_sentence_keys"] = sentence_keys
    if documents_sentences is not None:
        row["documents_sentences"] = documents_sentences
    return row


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


@pytest.fixture
def mock_ragbench_gold_eight_rows():
    """8 rows × 5 documents — row 7's documents start at corpus index 35."""
    return [
        _make_ragbench_gold_row(
            f"Q{i}", f"A{i}", [f"doc_{i}_{j}" for j in range(5)],
            sentence_keys=(["0a", "2a", "4a"] if i == 7 else []),
        )
        for i in range(8)
    ]


# ---------------------------------------------------------------------------
# FeTaQA fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_fetaqa_sparse():
    """FeTaQA dataset with one table: 3 headers, one short row + one full row."""
    return [_make_fetaqa_row(
        "What is X?", "X is Y",
        headers=["Col1", "Col2", "Col3"],
        data_rows=[["val1", "val2"], ["val3", "val4", "val5"]],
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


@pytest.fixture
def mock_stratrag_gold_pair():
    """Pad-free rows whose gold pair is the first two real documents.

    Row 0's pair resolves to corpus indices 0 and 1 — index 0 is a valid gold
    id (design R13), which is why gold resolution must use a membership lookup
    and never a truthiness check.
    """
    return [
        _make_stratrag_row(
            "Q0", "A0",
            [{"text": "doc_0_0", "source": "s0"},
             {"text": "doc_0_1", "source": "s0"},
             {"text": "doc_0_2", "source": "s0"}],
            row_id="val_000000", gold_doc_indices=[0, 1], question_type="bridge",
        ),
        _make_stratrag_row(
            "Q1", "A1",
            [{"text": "doc_1_0", "source": "s1"},
             {"text": "doc_1_1", "source": "s1"}],
            row_id="val_000001", gold_doc_indices=[0, 1],
        ),
    ]


@pytest.fixture
def mock_stratrag_skipped_empty_entry():
    """Rows whose skipped empty entries shift the corpus offsets.

    Row 0's first entry carries no text, so its kept documents land one corpus
    position earlier than their `doc_pool` position — a position-only
    resolution cannot see that (design AD-1).
    """
    return [
        _make_stratrag_row(
            "Q0", "A0",
            [{"text": "", "source": "s0"},
             {"text": "real_a", "source": "s0"},
             {"text": "real_b", "source": "s0"}],
            row_id="val_000000", gold_doc_indices=[1],
        ),
        _make_stratrag_row(
            "Q1", "A1",
            [{"text": "real_c", "source": "s1"},
             {"text": "real_d", "source": "s1"}],
            row_id="val_000001", gold_doc_indices=[0],
        ),
    ]


# ---------------------------------------------------------------------------
# Jagged row padding (FeTaQA — formatting logic retained)
# ---------------------------------------------------------------------------

class TestJaggedRowPadding:
    """Table formatting logic is preserved from before simplification."""

    def test_fetaqa_sparse_row_is_padded(self, mock_fetaqa_sparse):
        """3-col header + 2-col data row → short rows stay aligned, no invented value."""
        from rag_datasets import load_fetaqa

        with patch("rag_datasets.load_dataset", return_value=mock_fetaqa_sparse):
            rows, corpus = load_fetaqa(n=1)

        assert len(rows) == 1
        assert rows[0]["question"] == "What is X?"
        assert len(corpus) == 1
        doc_lines = corpus[0].splitlines()
        # Header keeps every column; the short row ends after its own cells.
        assert doc_lines[3] == "[0_0] | Col1 | Col2 | Col3 |"
        assert doc_lines[4] == "[0_1] | val1 | val2 |"
        assert doc_lines[5] == "[0_2] | val3 | val4 | val5 |"

    def test_fetaqa_equal_row_normal(self, mock_fetaqa_normal):
        """Matching header/data → pipe row with the row values."""
        from rag_datasets import load_fetaqa

        with patch("rag_datasets.load_dataset", return_value=mock_fetaqa_normal):
            rows, corpus = load_fetaqa(n=1)

        assert len(rows) == 1
        assert len(corpus) == 1
        doc = corpus[0]
        assert doc.startswith("Title: People\nSection: Employees\n")
        assert "| Alice | 30 | NYC |" in doc

    def test_fetaqa_long_row_clipped(self, mock_fetaqa_long):
        """Data row longer than headers → doc uses only header count of columns."""
        from rag_datasets import load_fetaqa

        with patch("rag_datasets.load_dataset", return_value=mock_fetaqa_long):
            rows, corpus = load_fetaqa(n=1)

        assert len(corpus) == 1
        doc = corpus[0]
        assert "| a | b |" in doc
        assert "| c |" not in doc
        assert "| d |" not in doc


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
        # No internal field leak — the row payload is the documented six-field shape
        assert list(rows[0].keys()) == [
            "question", "grading_notes", "gold_doc_ids", "gold_sentences",
            "gold_sentence_doc_ids", "no_gold_labels",
        ]

    def test_ragbench_no_dedup(self, mock_ragbench_sequential_dup):
        """Documents are NOT deduplicated — 'shared' appears twice."""
        from rag_datasets import load_ragbench

        with patch("rag_datasets.load_dataset", return_value=mock_ragbench_sequential_dup):
            rows, corpus = load_ragbench(n=3)

        assert len(rows) == 3
        assert corpus.count("shared") == 2, "No dedup: 'shared' should appear twice"


class TestRagbenchGoldLinkage:
    """load_ragbench resolves relevant sentence keys into corpus-level gold linkage.

    Spec: rag-eval-dataset-loading — ADDED *RAGBench gold document ids*,
    *gold sentences with source document ids* and *no-label rows*.
    Design: AD-1 (pair decoding), AD-2 (key decoding, parallel alignment).
    """

    def test_corpus_index_derivation(self, mock_ragbench_gold_eight_rows):
        """Spec: row 7 with keys on documents 0/2/4 → corpus indices 35/37/39."""
        from rag_datasets import load_ragbench

        with patch("rag_datasets.load_dataset", return_value=mock_ragbench_gold_eight_rows):
            rows, corpus = load_ragbench(n=8)

        assert len(corpus) == 40
        row7 = next(r for r in rows if r["question"] == "Q7")
        assert row7["gold_doc_ids"] == [35, 37, 39]

    def test_duplicate_keys_collapse_to_one_ascending_id(self):
        """Spec: repeated keys contribute one id each, and ids ascend."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_gold_row(
            "Q0", "A0", ["d0", "d1", "d2"],
            sentence_keys=["2a", "0a", "2b"],
        )]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=1)

        assert rows[0]["gold_doc_ids"] == [0, 2]

    def test_out_of_range_and_unparseable_keys_are_rejected(self):
        """Spec: out-of-range keys emit no id and leave the other ids intact."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_gold_row(
            "Q0", "A0", ["d0", "d1"],
            sentence_keys=["0a", "5a", "not-a-key"],
        )]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=1)

        assert rows[0]["gold_doc_ids"] == [0]
        assert rows[0]["no_gold_labels"] is False

    def test_pair_entry_is_decoded_and_stripped(self):
        """Design AD-1: the pair ["0a", " RELEASE NOTES ABSTRACT"] stores its text."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_gold_row(
            "Q0", "A0", ["d0"],
            sentence_keys=["0a"],
            documents_sentences=[[["0a", " RELEASE NOTES ABSTRACT"]]],
        )]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=1)

        assert rows[0]["gold_sentences"] == ["RELEASE NOTES ABSTRACT"]
        assert rows[0]["gold_sentence_doc_ids"] == [0]
        assert rows[0]["gold_doc_ids"] == [0]

    def test_pre_joined_string_entry_is_tolerated(self):
        """The other dataset shape — the key and text already joined — works too."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_gold_row(
            "Q0", "A0", ["d0"],
            sentence_keys=["0a"],
            documents_sentences=[["0a RELEASE NOTES ABSTRACT"]],
        )]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=1)

        assert rows[0]["gold_sentences"] == ["RELEASE NOTES ABSTRACT"]
        assert rows[0]["gold_sentence_doc_ids"] == [0]

    def test_sentence_whitespace_is_normalized(self):
        """Spec: whitespace runs collapse to single spaces, outer whitespace gone."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_gold_row(
            "Q0", "A0", ["d0"],
            sentence_keys=["0a"],
            documents_sentences=[[["0a", "  lots   of\r\nspace "]]],
        )]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=1)

        assert rows[0]["gold_sentences"] == ["lots of space"]

    def test_multi_letter_key_decodes_to_the_sentence_position(self):
        """Spec (*Multi-letter key decoded*): `4ab` → document 4, sentence 27."""
        from rag_datasets import load_ragbench

        doc4_sentences = [[f"4{chr(97 + k)}", f"filler {k}"] for k in range(26)]
        doc4_sentences.append(["4aa", "twenty sixth sentence"])
        doc4_sentences.append(["4ab", "TWENTY SEVENTH SENTENCE"])
        mock_ds = [_make_ragbench_gold_row(
            "Q0", "A0", ["d0", "d1", "d2", "d3", "d4"],
            sentence_keys=["4ab"],
            documents_sentences=[[], [], [], [], doc4_sentences],
        )]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=1)

        assert rows[0]["gold_doc_ids"] == [4]
        assert rows[0]["gold_sentences"] == ["TWENTY SEVENTH SENTENCE"]
        assert rows[0]["gold_sentence_doc_ids"] == [4]

    def test_parallel_alignment_across_documents(self):
        """Spec: gold_sentences[k] and gold_sentence_doc_ids[k] stay index-aligned."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_gold_row(
            "Q0", "A0", ["d0", "d1", "d2"],
            sentence_keys=["0a", "2a"],
            documents_sentences=[
                [["0a", " from doc zero"]],
                [],
                [["2a", " from doc two"]],
            ],
        )]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=1)

        row = rows[0]
        assert len(row["gold_sentences"]) == len(row["gold_sentence_doc_ids"]) == 2
        assert dict(zip(row["gold_sentence_doc_ids"], row["gold_sentences"])) == {
            0: "from doc zero", 2: "from doc two",
        }
        assert row["gold_doc_ids"] == [0, 2]

    def test_dotted_relevant_keys_keep_their_gold_linkage(self):
        """OP-1 finding: 11 live rows label with `4o.`-style keys (trailing dot).

        Rejecting the dot would strip those rows of every gold document and
        report them as out-of-corpus despite carrying labels.
        """
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_gold_row(
            "Q0", "A0", ["d0", "d1", "d2"],
            sentence_keys=["2b.", "0a."],
            documents_sentences=[
                [["0a", " from doc zero"]],
                [],
                [["2a", " a filler sentence"], ["2b", " from doc two"]],
            ],
        )]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=1)

        row = rows[0]
        assert row["gold_doc_ids"] == [0, 2]
        assert row["no_gold_labels"] is False
        assert dict(zip(row["gold_sentence_doc_ids"], row["gold_sentences"])) == {
            0: "from doc zero", 2: "from doc two",
        }

    def test_unlabeled_row_is_flagged_with_empty_gold_fields(self):
        """Spec: no relevant keys → flag true and three empty gold lists."""
        from rag_datasets import load_ragbench

        mock_ds = [
            _make_ragbench_gold_row("Q0", "A0", ["d0"], sentence_keys=[]),
            _make_ragbench_gold_row("Q1", "A1", ["d1"]),  # field absent entirely
        ]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=2)

        for row in rows:
            assert row["no_gold_labels"] is True
            assert row["gold_doc_ids"] == []
            assert row["gold_sentences"] == []
            assert row["gold_sentence_doc_ids"] == []

    def test_labeled_row_is_not_flagged(self):
        """Spec: at least one relevant key → the no-label flag is false."""
        from rag_datasets import load_ragbench

        mock_ds = [_make_ragbench_gold_row("Q0", "A0", ["d0"], sentence_keys=["0a"])]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_ragbench(n=1)

        assert rows[0]["no_gold_labels"] is False
        assert rows[0]["gold_doc_ids"] == [0]


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
        # FeTaQA rows now carry gold linkage fields alongside the question payload
        for r in rows:
            assert "_source_title" not in r
            assert set(r.keys()) == {
                "question", "grading_notes", "feta_id", "gold_values",
                "gold_doc_id", "gold_out_of_corpus",
            }


class TestFetaqaGoldLinkage:
    """load_fetaqa retains feta_id, resolved gold values, and skip-aware gold_doc_id.

    Spec: rag-eval-dataset-loading — requirements D3 (gold linkage fields),
    D4 (skip-aware gold_doc_id), D5 (gold-out-of-corpus marking).
    """

    def test_cell_resolution_indexes_table_array_with_header_row_zero(self):
        """highlighted_cell_ids are [row, col] over table_array; header row 0 counts."""
        from rag_datasets import load_fetaqa

        mock_ds = [_make_fetaqa_row(
            "When did Andy Karl win the Olivier Award?", "In 2017 for Groundhog Day",
            headers=["Year", "Award", "Category", "Work", "Result"],
            data_rows=[
                ["2013", "Drama Desk Award", "Outstanding Featured Actor", "The Mystery of Edwin Drood", "Nominated"],
                ["2014", "Tony Award", "Best Actor", "Rocky", "Nominated"],
                ["2017", "Laurence Olivier Award", "Best Actor", "Groundhog Day", "Won"],
            ],
            feta_id=2275,
            highlighted_cell_ids=[[0, 2], [3, 1]],
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_fetaqa(n=1)

        # [0,2] resolves inside the header row itself; [3,1] inside the last data row
        assert rows[0]["gold_values"] == ["Category", "Laurence Olivier Award"]

    @pytest.mark.parametrize("highlighted, expected", [
        ([[0, 1], [1, 0]], ["Age", "Alice"]),
        ([[1, 2], [1, 1]], ["NYC", "30"]),
        ([[2, 0]], ["Bob"]),
    ])
    def test_cell_resolution_multiple_pairs(self, highlighted, expected):
        """Different [row, col] combinations resolve to distinct gold values."""
        from rag_datasets import load_fetaqa

        mock_ds = [_make_fetaqa_row(
            "Who?", "Alice",
            headers=["Name", "Age", "City"],
            data_rows=[["Alice", "30", "NYC"], ["Bob", "40", "LA"]],
            feta_id=42,
            highlighted_cell_ids=highlighted,
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_fetaqa(n=1)

        assert rows[0]["gold_values"] == expected

    def test_empty_highlighted_cell_ids_yields_empty_gold_values(self):
        """No highlighted cells → empty gold list, gold_doc_id still computed."""
        from rag_datasets import load_fetaqa

        mock_ds = [_make_fetaqa_row(
            "Question with no denotation?", "Some answer",
            headers=["H1", "H2"], data_rows=[["a", "b"]],
            feta_id=11, highlighted_cell_ids=[],
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_fetaqa(n=1)

        assert rows[0]["gold_values"] == []
        assert rows[0]["gold_doc_id"] == 0

    def test_row_payload_carries_gold_linkage_fields(self):
        """Each row carries feta_id, a gold values list, and gold_doc_id."""
        from rag_datasets import load_fetaqa

        mock_ds = [_make_fetaqa_row(
            "Who?", "Alice",
            headers=["Name", "Age"], data_rows=[["Alice", "30"]],
            feta_id=900, highlighted_cell_ids=[[1, 0]],
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_fetaqa(n=1)

        row = rows[0]
        assert set(row.keys()) == {
            "question", "grading_notes", "feta_id", "gold_values",
            "gold_doc_id", "gold_out_of_corpus",
        }
        assert row["question"] == "Who?"
        assert row["grading_notes"] == "Alice"
        assert row["feta_id"] == 900
        assert row["gold_values"] == ["Alice"]
        assert row["gold_doc_id"] == 0
        assert row["gold_out_of_corpus"] is False

    def test_gold_doc_id_counts_only_kept_tables(self):
        """Skipped tables before/between gold rows must not shift gold_doc_id."""
        from rag_datasets import load_fetaqa

        mock_ds = [
            _make_fetaqa_row("Q skipped first", "A", ["OnlyHeader"], [],
                             feta_id=1, highlighted_cell_ids=[[0, 0]]),
            _make_fetaqa_row("Q kept first", "A", ["H1"], [["alpha"]],
                             feta_id=2, highlighted_cell_ids=[[1, 0]]),
            _make_fetaqa_row("Q skipped between", "A", ["OnlyHeader"], [],
                             feta_id=3, highlighted_cell_ids=[[0, 0]]),
            _make_fetaqa_row("Q kept second", "A", ["H1"], [["beta"]],
                             feta_id=4, highlighted_cell_ids=[[1, 0]]),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_fetaqa(n=4)

        by_id = {r["feta_id"]: r for r in rows}
        assert len(corpus) == 2
        assert by_id[2]["gold_doc_id"] == 0
        assert by_id[4]["gold_doc_id"] == 1

    def test_gold_table_skipped_is_marked_out_of_corpus(self):
        """A row whose table is never serialized gets no bogus id: marked out-of-corpus."""
        from rag_datasets import load_fetaqa

        mock_ds = [
            _make_fetaqa_row("Q skipped", "A", ["OnlyHeader"], [],
                             feta_id=1, highlighted_cell_ids=[[0, 0]]),
            _make_fetaqa_row("Q kept", "A", ["H1"], [["kept-value"]],
                             feta_id=2, highlighted_cell_ids=[[1, 0]]),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_fetaqa(n=2)

        by_id = {r["feta_id"]: r for r in rows}
        assert by_id[1]["gold_doc_id"] is None
        assert by_id[1]["gold_out_of_corpus"] is True
        assert by_id[2]["gold_doc_id"] == 0
        assert by_id[2]["gold_out_of_corpus"] is False

    def test_identity_alignment_corpus_gold_doc_id_resolves_own_table(self):
        """corpus[gold_doc_id] is the serialized table of the row's own question."""
        from rag_datasets import load_fetaqa

        mock_ds = [
            _make_fetaqa_row("Q alpha", "A", ["H1"], [["alpha-marker"], ["alpha-extra"]],
                             feta_id=101, highlighted_cell_ids=[[1, 0]]),
            _make_fetaqa_row("Q beta", "A", ["H1"], [["beta-marker"]],
                             feta_id=102, highlighted_cell_ids=[[1, 0]]),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_fetaqa(n=2)

        by_id = {r["feta_id"]: r for r in rows}
        alpha_doc = corpus[by_id[101]["gold_doc_id"]]
        beta_doc = corpus[by_id[102]["gold_doc_id"]]

        assert by_id[101]["gold_doc_id"] == 0
        assert by_id[102]["gold_doc_id"] == 1
        assert "alpha-marker" in alpha_doc
        assert "alpha-extra" in alpha_doc
        assert "beta-marker" not in alpha_doc
        assert "beta-marker" in beta_doc
        assert by_id[101]["gold_values"] == ["alpha-marker"]
        assert by_id[102]["gold_values"] == ["beta-marker"]


# ---------------------------------------------------------------------------
# Opción B serialization + cleanup (spec: rag-eval-dataset-loading — D1, D2)
# ---------------------------------------------------------------------------

# Mirrors the docs-tool chunker (`src/backend/tero/tools/docs/tool.py`):
# MarkdownTextSplitter with DOCS_TOOL_CHUNK_SIZE=4000 / OVERLAP=200 (no `.env` edits).
FETAQA_CHUNK_SIZE = 4000
FETAQA_CHUNK_OVERLAP = 200


def _split_like_docs_tool(text: str) -> list[str]:
    """Split a corpus document exactly like the backend docs tool does."""
    from langchain_text_splitters import MarkdownTextSplitter

    splitter = MarkdownTextSplitter(
        chunk_size=FETAQA_CHUNK_SIZE,
        chunk_overlap=FETAQA_CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


class TestFetaqaOpcionBSerialization:
    """Opción B: Title/Section metadata, pipe-delimited rows, per-row UIDs.

    Spec: rag-eval-dataset-loading — D1 (FeTaQA Opción B serialization).
    UIDs are `[feta_id_rownum]` with `rownum` indexing `table_array` (row 0 is
    the header, matching `highlighted_cell_ids`).
    """

    def test_document_starts_with_title_and_section_lines(self):
        """The document begins with Title:/Section: lines from the row fields."""
        from rag_datasets import load_fetaqa

        mock_ds = [_make_fetaqa_row(
            "Who?", "Person",
            headers=["Name", "Age", "City"],
            data_rows=[["Alice", "30", "NYC"]],
            page_title="People",
            section_title="Employees",
            feta_id=42,
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            _, corpus = load_fetaqa(n=1)

        lines = corpus[0].splitlines()
        assert lines[0] == "Title: People"
        assert lines[1] == "Section: Employees"
        assert lines[2] == ""
        assert lines[3] == "[42_0] | Name | Age | City |"
        assert lines[4] == "[42_1] | Alice | 30 | NYC |"

    def test_rows_are_pipe_delimited_with_the_row_feta_id_uid(self):
        """UIDs use the row's own `feta_id` (the dataset field, not the position)."""
        from rag_datasets import load_fetaqa

        mock_ds = [
            _make_fetaqa_row("Q0", "A0", ["H1"], [["first"]], feta_id=1),
            _make_fetaqa_row(
                "Q1", "A1",
                headers=["Year", "Result"],
                data_rows=[["2017", "Won"], ["2013", "Nominated"]],
                feta_id=2275,
            ),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            _, corpus = load_fetaqa(n=2)

        assert corpus[0].splitlines()[3] == "[1_0] | H1 |"
        second = corpus[1].splitlines()
        assert second[3] == "[2275_0] | Year | Result |"
        assert second[4] == "[2275_1] | 2017 | Won |"
        assert second[5] == "[2275_2] | 2013 | Nominated |"

    def test_one_table_is_one_chunk_at_configured_size_and_overlap(self):
        """Split with 4000/200: every document yields exactly one chunk."""
        from rag_datasets import load_fetaqa

        wide_row = [
            "James R. Thompson incumbent", "1,816,101", "49.44",
            "Adlai Stevenson III", "1,811,027", "49.30",
        ]
        mock_ds = [
            _make_fetaqa_row("Q0", "A0", ["H1"], [["small"]], feta_id=1),
            _make_fetaqa_row(
                "Q1", "A1",
                headers=["Candidate", "Votes", "Percent", "Party", "Region", "Note"],
                data_rows=[wide_row for _ in range(34)],
                feta_id=2,
            ),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            _, corpus = load_fetaqa(n=2)

        assert len(corpus) == 2
        # The near-worst-case table must be genuinely large for this assertion
        # to mean anything (FeTaQA tables reach ~34 data rows).
        assert len(corpus[1]) > 3000
        for doc in corpus:
            assert len(_split_like_docs_tool(doc)) == 1

    def test_splitter_control_an_oversized_table_does_split(self):
        """Control: the 1-chunk assertion above is sensitive to document size."""
        from rag_datasets import load_fetaqa

        wide_row = [
            "James R. Thompson incumbent", "1,816,101", "49.44",
            "Adlai Stevenson III", "1,811,027", "49.30",
        ]
        mock_ds = [_make_fetaqa_row(
            "Q", "A",
            headers=["Candidate", "Votes", "Percent", "Party", "Region", "Note"],
            data_rows=[wide_row for _ in range(70)],
            feta_id=2,
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            _, corpus = load_fetaqa(n=1)

        assert len(corpus) == 1
        assert len(corpus[0]) > FETAQA_CHUNK_SIZE
        assert len(_split_like_docs_tool(corpus[0])) > 1


class TestFetaqaCleanupLosslessDeterministic:
    """Deterministic cleanup that never drops gold-referenced cells.

    Spec: rag-eval-dataset-loading — D2 (lossless serialization cleanup):
    duplicate/degenerate columns (ToTTo merged-cell artifact) are removed,
    gold cell values survive, and repeated runs are identical.
    """

    def test_degenerate_and_duplicate_columns_are_dropped(self):
        """All-`-` columns (degenerate) and repeated headers are removed."""
        from rag_datasets import load_fetaqa

        mock_ds = [_make_fetaqa_row(
            "Q", "A",
            headers=["Party", "Party", "Candidate", "Notes"],
            data_rows=[
                ["Republican", "-", "James R. Thompson", "-"],
                ["-", "-", "Adlai Stevenson III", "-"],
            ],
            feta_id=7,
            highlighted_cell_ids=[[1, 2]],
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_fetaqa(n=1)

        lines = corpus[0].splitlines()
        assert lines[3] == "[7_0] | Party | Candidate |"
        assert lines[4] == "[7_1] | Republican | James R. Thompson |"
        assert lines[5] == "[7_2] | - | Adlai Stevenson III |"
        assert rows[0]["gold_values"] == ["James R. Thompson"]

    def test_gold_cell_in_a_duplicate_column_survives_cleanup(self):
        """A duplicate-header column holding a gold value is kept (lossless)."""
        from rag_datasets import load_fetaqa

        mock_ds = [_make_fetaqa_row(
            "Q", "A",
            headers=["Party", "Party", "Candidate"],
            data_rows=[
                ["Republican", "Republican-Alternate", "James R. Thompson"],
                ["-", "-", "-"],
            ],
            feta_id=7,
            highlighted_cell_ids=[[1, 1]],
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_fetaqa(n=1)

        assert rows[0]["gold_values"] == ["Republican-Alternate"]
        doc = corpus[rows[0]["gold_doc_id"]]
        # Without gold protection the duplicate column is dropped and the value
        # disappears, so cell_recall would miss a gold cell the table contains.
        assert "[7_1] | Republican | Republican-Alternate | James R. Thompson |" in doc

    def test_cleanup_and_serialization_are_deterministic(self):
        """Repeated loads of the same dataset produce byte-identical documents."""
        from rag_datasets import load_fetaqa

        mock_ds = [_make_fetaqa_row(
            "Q", "A",
            headers=["Year", "Year", "Result", "Notes"],
            data_rows=[
                ["2017", "2017", "Won", "-"],
                ["2013", "2013", "Nominated", "-"],
            ],
            feta_id=2275,
            highlighted_cell_ids=[[1, 2]],
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            _, first = load_fetaqa(n=1)
            _, second = load_fetaqa(n=1)

        assert first == second
        assert first[0].splitlines()[3] == "[2275_0] | Year | Result |"

    def test_identity_alignment_after_cleanup_resolves_the_serialized_table(self):
        """`gold_doc_id` still resolves to the row's own cleaned document."""
        from rag_datasets import load_fetaqa

        mock_ds = [
            _make_fetaqa_row("Q skipped", "A", ["OnlyHeader"], [],
                             feta_id=1, highlighted_cell_ids=[[0, 0]]),
            _make_fetaqa_row(
                "Q gold", "A",
                headers=["Party", "Party", "Candidate"],
                data_rows=[["-", "-", "Adlai Stevenson III"]],
                feta_id=2,
                highlighted_cell_ids=[[1, 2]],
            ),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_fetaqa(n=2)

        row = {r["feta_id"]: r for r in rows}[2]
        doc = corpus[row["gold_doc_id"]]
        assert row["gold_doc_id"] == 0  # the skipped table never counted
        assert doc.startswith("Title: TestTable\nSection: TestSection\n")
        assert "[2_1] | Adlai Stevenson III |" in doc
        assert row["gold_values"] == ["Adlai Stevenson III"]

    def test_empty_first_duplicate_keeps_the_column_that_has_values(self):
        """The degenerate rule runs before dedup: the first `Party` goes, not the second."""
        from rag_datasets import load_fetaqa

        mock_ds = [_make_fetaqa_row(
            "Q", "A",
            headers=["Party", "Party", "Candidate"],
            data_rows=[
                ["-", "Republican", "James R. Thompson"],
                ["-", "Democratic", "Adlai Stevenson III"],
            ],
            feta_id=7,
            highlighted_cell_ids=[[1, 2]],
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            _, corpus = load_fetaqa(n=1)

        lines = corpus[0].splitlines()
        assert lines[3] == "[7_0] | Party | Candidate |"
        assert lines[4] == "[7_1] | Republican | James R. Thompson |"

    def test_gold_header_value_in_a_duplicate_column_survives_cleanup(self):
        """Gold may reference a header cell; its column is kept (header protection)."""
        from rag_datasets import load_fetaqa

        mock_ds = [_make_fetaqa_row(
            "Q", "A",
            headers=["Category", "Category", "Name"],
            data_rows=[["Cat", "Cat", "Alice"]],
            feta_id=7,
            highlighted_cell_ids=[[0, 1]],
        )]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_fetaqa(n=1)

        assert rows[0]["gold_values"] == ["Category"]
        # Both duplicate columns survive: the second one is gold-referenced, so
        # the dedup rule cannot drop it.
        assert corpus[0].splitlines()[3] == "[7_0] | Category | Category | Name |"


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
        # The row contract (exact keys and order) is pinned by
        # `test_row_shape_is_pinned`.
        # Corpus: all docs, sequential, first doc from first row
        assert len(corpus) >= 4
        assert corpus[0] == "doc_a"
        # No [source]\ntext format — just plain text
        assert "\n" not in corpus[0], "No [source] prefix → plain text only"

    def test_row_shape_is_pinned(self, mock_stratrag_sequential):
        """The row payload is the loader's contract with runner, probe and CSV.

        Exact key order is pinned (design AD-9, mirroring the ragbench
        precedent) so an accidental key reorder or an internal-field leak fails
        a test instead of silently reshaping a CSV column.
        """
        from rag_datasets import load_stratrag

        with patch("rag_datasets.load_dataset", return_value=mock_stratrag_sequential):
            rows, corpus = load_stratrag(n=2)

        assert len(rows) == 2
        for r in rows:
            assert list(r.keys()) == [
                "question",
                "grading_notes",
                "row_id",
                "question_type",
                "gold_doc_ids",
                "no_gold_labels",
            ]

    def test_stratrag_empty_text_filtered(self, mock_stratrag_empty_text):
        """Empty text and missing 'text' key are filtered out.

        Skipped entries leave no gap: the kept documents stay contiguous and in
        dataset order (spec M1 *Empty-text entries remain filtered*).
        """
        from rag_datasets import load_stratrag

        with patch("rag_datasets.load_dataset", return_value=mock_stratrag_empty_text):
            rows, corpus = load_stratrag(n=2)

        assert len(rows) == 2
        # Empty text filtered, missing text key filtered → only 'valid' and 'fill1',
        # contiguous and in dataset order — the skipped entries leave no gap.
        assert "" not in corpus
        assert corpus == ["valid", "fill1"]


class TestStratragGoldLinkage:
    """Gold ids resolve through the position → corpus-index map (spec D1)."""

    def test_gold_pair_resolved_through_the_position_map(self, mock_stratrag_gold_pair):
        """A `[0, 1]` gold pair resolves to `[0, 1]` on the first row.

        Corpus index 0 is a legitimate gold id, so a falsy-index check would
        silently drop it (design R13). Ids point at the row's real documents, in
        dataset order.
        """
        from rag_datasets import load_stratrag

        with patch("rag_datasets.load_dataset", return_value=mock_stratrag_gold_pair):
            rows, corpus = load_stratrag(n=2)

        assert len(rows) == 2
        assert rows[0]["gold_doc_ids"] == [0, 1]
        assert rows[0]["no_gold_labels"] is False
        assert [corpus[i] for i in rows[0]["gold_doc_ids"]] == ["doc_0_0", "doc_0_1"]
        # Row 1's pair continues the offsets — the third corpus document.
        assert rows[1]["gold_doc_ids"] == [3, 4]
        assert rows[1]["no_gold_labels"] is False
        assert [corpus[i] for i in rows[1]["gold_doc_ids"]] == ["doc_1_0", "doc_1_1"]

    def test_dynamic_offsets_via_a_skipped_empty_entry(self, mock_stratrag_skipped_empty_entry):
        """Row 0's gold position 1 is `real_a` at corpus index 0, not 1.

        The offset is produced by the skipped entry, which only the map can see
        — position math would emit 1 and point at `real_b`.
        """
        from rag_datasets import load_stratrag

        with patch("rag_datasets.load_dataset", return_value=mock_stratrag_skipped_empty_entry):
            rows, corpus = load_stratrag(n=2)

        assert corpus == ["real_a", "real_b", "real_c", "real_d"]
        assert rows[0]["gold_doc_ids"] == [0]
        assert corpus[rows[0]["gold_doc_ids"][0]] == "real_a"
        # Row 1 starts after two kept documents: its position 0 is corpus index 2.
        assert rows[1]["gold_doc_ids"] == [2]
        assert corpus[rows[1]["gold_doc_ids"][0]] == "real_c"

    def test_gold_index_on_a_skipped_entry_emits_no_id(self):
        """A gold position on a skipped entry emits nothing and remaps nothing."""
        from rag_datasets import load_stratrag

        mock_ds = [
            _make_stratrag_row(
                "Q0", "A0",
                [{"text": "", "source": "s0"}, {"text": "real_a", "source": "s0"}],
                row_id="val_000000", gold_doc_indices=[0, 1],
            ),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_stratrag(n=1)

        # Position 0 is skipped → no id; position 1 is `real_a` at corpus index 0.
        assert rows[0]["gold_doc_ids"] == [0]
        assert corpus == ["real_a"]
        assert rows[0]["no_gold_labels"] is False

    @pytest.mark.parametrize(
        "extra",
        [{}, {"gold_doc_indices": []}],
        ids=["missing-field", "empty-list"],
    )
    def test_index_less_row_is_flagged(self, extra):
        """No `gold_doc_indices` (or an empty list) → flag `True`, no ids."""
        from rag_datasets import load_stratrag

        mock_ds = [
            _make_stratrag_row(
                "Q0", "A0",
                [{"text": "real_a", "source": "s0"}],
                row_id="val_000000",
                **extra,
            ),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_stratrag(n=1)

        assert rows[0]["gold_doc_ids"] == []
        assert rows[0]["no_gold_labels"] is True

    @pytest.mark.parametrize(
        "gold_doc_indices,expected_ids",
        [
            (["1"], [1]),
            (["not-a-position", 0], [0]),
            (["not-a-position"], []),
        ],
        ids=["string-position-accepted", "unparseable-skipped", "all-unparseable"],
    )
    def test_non_integer_gold_position_is_coerced_or_skipped(self, gold_doc_indices, expected_ids):
        """`_as_int` coercion: string positions are accepted, junk is skipped.

        Skipping never changes the flag — label availability is still what the
        row carries, not what resolved (design AD-3).
        """
        from rag_datasets import load_stratrag

        mock_ds = [
            _make_stratrag_row(
                "Q0", "A0",
                [{"text": "real_a", "source": "s0"}, {"text": "real_b", "source": "s0"}],
                row_id="val_000000", gold_doc_indices=gold_doc_indices,
            ),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_stratrag(n=1)

        assert rows[0]["gold_doc_ids"] == expected_ids
        assert rows[0]["no_gold_labels"] is False


class TestStratragRowExposure:
    """Rows expose dataset identity and question type (spec D2)."""

    def test_row_identity_and_question_type_exposed(self):
        """`id` and `metadata.question_type` reach the payload; QA fields survive."""
        from rag_datasets import load_stratrag

        # `val_000089` is the registry row for the S2 grading-notes correction,
        # so the fixture stores the corrected value: asserting a stored answer
        # that the registry rewrites would flip an S1 test inside S2, and S2
        # owns no by-design flips.
        mock_ds = [
            _make_stratrag_row(
                "Who produced Being John Malkovich?",
                "Vincent Landay",
                [{"text": "doc_89_a", "source": "s0"}, {"text": "doc_89_b", "source": "s0"}],
                row_id="val_000089", gold_doc_indices=[0, 1], question_type="bridge",
            ),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_stratrag(n=1)

        assert rows[0]["row_id"] == "val_000089"
        assert rows[0]["question_type"] == "bridge"
        assert rows[0]["question"] == "Who produced Being John Malkovich?"
        assert rows[0]["grading_notes"] == "Vincent Landay"

    @pytest.mark.parametrize(
        "extra",
        [{}, {"metadata": {}}],
        ids=["no-metadata", "metadata-without-key"],
    )
    def test_question_type_defaults_to_unknown(self, extra):
        """Missing `metadata.question_type` → `"unknown"`, row still loads."""
        from rag_datasets import load_stratrag

        mock_ds = [
            _make_stratrag_row(
                "Q0", "A0",
                [{"text": "real_a", "source": "s0"}],
                row_id="val_000000", gold_doc_indices=[0],
                **extra,
            ),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, _ = load_stratrag(n=1)

        assert rows[0]["question_type"] == "unknown"


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

    def test_shortfall_warning_preserved_with_linkage(self, capsys):
        """`n` above the row count keeps the warning text and the new fields."""
        from rag_datasets import load_stratrag

        mock_ds = [
            _make_stratrag_row(
                "Q0", "A0",
                [{"text": "real_a", "source": "s0"}, {"text": "real_b", "source": "s0"}],
                row_id="val_000000", gold_doc_indices=[0, 1],
            ),
        ]
        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            rows, corpus = load_stratrag(n=3)

        assert len(rows) == 1
        captured = capsys.readouterr()
        assert "requested 3 questions but only 1 available" in captured.out
        assert rows[0]["row_id"] == "val_000000"
        assert rows[0]["gold_doc_ids"] == [0, 1]
        assert rows[0]["no_gold_labels"] is False

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

    def test_zero_questions_still_builds_the_full_corpus(self):
        """`n=0` returns no rows but the full kept corpus, and the linkage over
        that corpus is the one a non-zero load resolves (spec D1 *Zero questions
        still resolves linkage over the full corpus*)."""
        from rag_datasets import load_stratrag

        mock_ds = [
            _make_stratrag_row(
                "Q0", "A0",
                [{"text": "", "source": "s0"},
                 {"text": "real_a", "source": "s0"},
                 {"text": "real_b", "source": "s0"}],
                row_id="val_000000", gold_doc_indices=[1, 2],
            ),
            _make_stratrag_row(
                "Q1", "A1",
                [{"text": "real_c", "source": "s1"},
                 {"text": "real_d", "source": "s1"}],
                row_id="val_000001", gold_doc_indices=[0],
            ),
        ]

        with patch("rag_datasets.load_dataset", return_value=mock_ds):
            zero_rows, zero_corpus = load_stratrag(n=0)
            rows, corpus = load_stratrag(n=2)

        assert zero_rows == []
        assert zero_corpus == ["real_a", "real_b", "real_c", "real_d"]
        # The corpus is the same one the linkage resolves against.
        assert zero_corpus == corpus
        assert rows[0]["gold_doc_ids"] == [0, 1]
        assert rows[1]["gold_doc_ids"] == [2]
        assert [corpus[i] for i in rows[1]["gold_doc_ids"]] == ["real_c"]


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
