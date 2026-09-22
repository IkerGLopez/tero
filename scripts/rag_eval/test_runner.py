"""
Tests for runner.py — recursionLimitExceeded detection, NaN faithfulness logging,
and relevant_chunk_position computation.
"""
import sys
import os
import asyncio
import argparse
import math
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# Make runner importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# TASK-2.3 — DP5: relevant_chunk_position (pure function tests)
# ---------------------------------------------------------------------------

class TestRelevantChunkIndex:
    """Tests for _find_relevant_chunk_index pure function — 20-char minimum + token-overlap hybrid."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        """Import the function once the stub exists in runner.py."""
        from runner import _find_relevant_chunk_index
        self._find_relevant_chunk_index = _find_relevant_chunk_index

    # ── Exact substring match (20+ chars) ──

    def test_relevant_chunk_first_position(self):
        """Grading notes substring found in retrieved context at 1-based index 3."""
        grading_notes = "Barack Hussein Obama was born in Hawaii"
        contexts = [
            "Some unrelated context about politics",
            "Another unrelated context about elections",
            "Biography: Barack Hussein Obama was born in Hawaii in 1961...",
            "More unrelated text",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == 3, f"Expected 3, got {result}"

    def test_relevant_chunk_case_insensitive(self):
        """Substring match is case-insensitive."""
        grading_notes = "BARACK HUSSEIN OBAMA WAS BORN IN HAWAII"
        contexts = [
            "Context A",
            "barack hussein obama was born in hawaii biography",
            "Context C",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == 2, f"Case-insensitive match should find at position 2, got {result}"

    def test_relevant_chunk_exact_20_char_match(self):
        """Minimum 20-char substring exactly matches."""
        grading_notes = "12345678901234567890extra"
        contexts = ["12345678901234567890"]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == 1, f"Expected 1 for exact 20-char match, got {result}"

    def test_relevant_chunk_19_char_no_match(self):
        """19-char grading notes should NOT match (minimum 20 chars required)."""
        grading_notes = "1234567890123456789"
        contexts = ["1234567890123456789"]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == -1, f"Expected -1 for <20 char grading notes, got {result}"

    # ── Token overlap (word reorder) ──

    def test_kathleen_williams_token_overlap_reorder(self):
        """'Landay, Vincent' in context — token overlap resolves word-order reorder.
        Uses 20+ char grading notes to meet minimum length requirement."""
        grading_notes = "Landay, Vincent film producer"
        contexts = [
            "Some random film trivia here",
            "Vincent Landay is a film producer known for his work",
            "Another unrelated document",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == 2, f"Expected 2 for token-overlap match, got {result}"

    def test_token_overlap_resolves_word_order(self):
        """'Vincent Landay producer' matches 'Landay, Vincent is a film producer'."""
        grading_notes = "Vincent Landay producer"
        contexts = ["Landay, Vincent is a film producer"]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == 1, f"Expected 1 for token-overlap match, got {result}"

    # ── No-match scenarios ──

    def test_relevant_chunk_not_found(self):
        """No token overlap and no 20+ char substring match → returns -1."""
        grading_notes = "Advanced carbon dating methodology explained thoroughly"
        contexts = [
            "Geology of the Grand Canyon formations",
            "History of radiometric techniques study",
            "Fossil record sedimentary analysis",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == -1, f"Expected -1, got {result}"

    def test_relevant_chunk_empty_contexts(self):
        """Empty contexts list → returns -1."""
        grading_notes = "Some relevant content here, long enough"
        contexts = []
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == -1, f"Expected -1 for empty contexts, got {result}"

    def test_relevant_chunk_with_short_grading_notes(self):
        """Grading notes shorter than 20 chars → returns -1 (no 20-char substring possible)."""
        grading_notes = "Short grading note"
        contexts = [
            "This is a short note context, but not long enough",
            "Another context",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == -1, f"Expected -1 for short grading_notes (< 20 chars), got {result}"

    # ── Regression — existing behavior preserved ──

    def test_substring_bonus_boosts_exact_match(self):
        """20+ char substring + token overlap → highest score wins (regression)."""
        grading_notes = "Kathleen Williams was born in Portland Oregon"
        contexts = [
            "Nothing about anyone here",
            "Kathleen Williams was born in Portland and grew up...",
            "Portland is a city in Oregon",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == 2, f"Expected index 2 (substring match), got {result}"

    def test_highest_score_context_selected(self):
        """Multiple partial matches → highest scoring context wins."""
        grading_notes = "the president of France in 2024"
        contexts = [
            "France has a president",
            "In 2024 the president of France visited Germany",
            "France is in Europe",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == 2, f"Expected index 2 (most token overlap + substring), got {result}"


# ---------------------------------------------------------------------------
# Helpers for async runner tests
# ---------------------------------------------------------------------------

def _make_mock_tero(answer_text="Normal answer", contexts=None, citations=None, latency_ms=100):
    """Create a mock TeroClient with configurable ask_question response."""
    mock = AsyncMock()
    mock.create_thread.return_value = "thread_test_123"
    mock.ask_question.return_value = {
        "answer_text": answer_text,
        "retrieved_contexts": contexts or ["ctx1", "ctx2", "ctx3"],
        "citations": citations or ["cite1", "cite2"],
        "latency_ms": latency_ms,
    }
    return mock


def _make_mock_metrics(recall_val=0.8, precision_val=0.7, faith_val=0.9, correctness_val="3"):
    """Create mock RAGAS metric objects with configurable return values."""
    mock_recall = AsyncMock()
    mock_recall.single_turn_ascore.return_value = recall_val

    mock_precision = AsyncMock()
    mock_precision.single_turn_ascore.return_value = precision_val

    mock_faith = AsyncMock()
    mock_faith.single_turn_ascore.return_value = faith_val

    mock_correctness = AsyncMock()
    mock_correctness.ascore.return_value = MagicMock(value=correctness_val)

    mock_cite_faith = AsyncMock()
    mock_cite_faith.ascore.return_value = MagicMock(value="supported")

    return mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith


def _make_test_row(question="Test question?", grading_notes="Test grading notes", **extra):
    """Create a test row dict with required fields."""
    row = {
        "question": question,
        "grading_notes": grading_notes,
    }
    row.update(extra)
    return row


# ---------------------------------------------------------------------------
# TASK-2.1 — Bug 2: recursionLimitExceeded detection
# ---------------------------------------------------------------------------

class TestRecursionLimitExceeded:
    """Tests for recursionLimitExceeded error detection in _process_question_result."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _process_question_result
        self._process_question_result = _process_question_result

    def test_recursion_limit_detected_and_all_metrics_none(self, capsys):
        """When answer_text starts with recursionLimitExceeded, all metrics are None and error is set."""
        mock_tero = _make_mock_tero(
            answer_text="recursionLimitExceeded: max recursion depth reached",
        )
        row = _make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        result = asyncio.run(
            self._process_question_result(
                mock_tero,
                row,
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

        assert result["error"] == "recursionLimitExceeded"
        assert result["correctness"] is None
        assert result["faithfulness"] is None
        assert result["context_recall"] is None
        assert result["context_precision"] is None
        assert result["citation_faithfulness"] is None
        assert result["grounded_correctness"] is None
        assert result["response"] == "recursionLimitExceeded: max recursion depth reached"
        assert result["latency_ms"] == 100

        # RAGAS metrics must NOT have been called
        mock_recall.single_turn_ascore.assert_not_called()
        mock_precision.single_turn_ascore.assert_not_called()
        mock_faith.single_turn_ascore.assert_not_called()
        mock_correctness.ascore.assert_not_called()

        # WARNING should be printed
        captured = capsys.readouterr()
        assert "WARNING: backend error for question" in captured.out
        assert "recursionLimitExceeded" in captured.out

    def test_normal_answer_has_error_none(self):
        """Normal answer should have error=None and all metrics computed."""
        mock_tero = _make_mock_tero(answer_text="The capital of France is Paris.")
        row = _make_test_row(question="What is the capital of France?", grading_notes="Paris")
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        result = asyncio.run(
            self._process_question_result(
                mock_tero,
                row,
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

        assert result["error"] is None
        # Metrics should be computed (not None)
        assert result["correctness"] == 3  # from mock correctness_val="3"
        assert result["faithfulness"] == 0.9
        assert isinstance(result["context_recall"], float)
        assert isinstance(result["context_precision"], float)
        # Original row fields should be present
        assert result["question"] == "What is the capital of France?"
        assert result["grading_notes"] == "Paris"

    def test_recursion_limit_non_starting_substring(self):
        """recursionLimitExceeded in mid-answer is NOT flagged (REQ-006: startswith only)."""
        mock_tero = _make_mock_tero(
            answer_text="Error details: recursionLimitExceeded during processing",
        )
        row = _make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        result = asyncio.run(
            self._process_question_result(
                mock_tero, row, None,
                mock_recall, mock_precision, mock_faith,
                mock_correctness, mock_cite_faith,
            )
        )

        # Mid-answer substring should NOT be flagged as recursionLimitExceeded
        assert result["error"] is None
        # RAGAS metrics should be computed normally
        assert result["correctness"] is not None


# ---------------------------------------------------------------------------
# TASK-2.2 — Bug 3: NaN faithfulness logging
# ---------------------------------------------------------------------------

class TestNanFaithfulnessLogging:
    """Tests for NaN faithfulness detection and WARNING logging."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _process_question_result
        self._process_question_result = _process_question_result

    def test_nan_faithfulness_logs_warning(self, capsys):
        """When faithfulness returns NaN, a WARNING is printed and faith_val is None."""
        mock_tero = _make_mock_tero(answer_text="Normal answer")
        row = _make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics(
            faith_val=float("nan"),
        )

        result = asyncio.run(
            self._process_question_result(
                mock_tero, row, None,
                mock_recall, mock_precision, mock_faith,
                mock_correctness, mock_cite_faith,
            )
        )

        captured = capsys.readouterr()
        assert "WARNING: faithfulness=NaN for question" in captured.out
        # REQ-003: NaN → None, not 0.0
        assert result["faithfulness"] is None
        # grounded_correctness should be None when faith is invalid
        assert result["grounded_correctness"] is None

    def test_none_faithfulness_logs_warning(self, capsys):
        """When faithfulness returns None, a WARNING is printed and faith_val is None."""
        mock_tero = _make_mock_tero(answer_text="Normal answer")
        row = _make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics(
            faith_val=None,
        )

        result = asyncio.run(
            self._process_question_result(
                mock_tero, row, None,
                mock_recall, mock_precision, mock_faith,
                mock_correctness, mock_cite_faith,
            )
        )

        captured = capsys.readouterr()
        assert "WARNING: faithfulness=NaN for question" in captured.out
        # REQ-003: None → None
        assert result["faithfulness"] is None

    def test_valid_faithfulness_no_warning(self, capsys):
        """When faithfulness returns a valid value, no WARNING is printed."""
        mock_tero = _make_mock_tero(answer_text="Normal answer")
        row = _make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics(
            faith_val=0.85,
        )

        result = asyncio.run(
            self._process_question_result(
                mock_tero, row, None,
                mock_recall, mock_precision, mock_faith,
                mock_correctness, mock_cite_faith,
            )
        )

        captured = capsys.readouterr()
        assert "WARNING: faithfulness=NaN" not in captured.out
        assert result["faithfulness"] == 0.85


# ---------------------------------------------------------------------------
# TASK-3.1 — REMOVED: Corpus cache tests (Phase 2 strips corpus_state)
# TASK-3.2 — REMOVED: Default corpus_size (Phase 2 strips corpus_size)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TASK-1.1 — Citation pairing tests (8 cases)
# ---------------------------------------------------------------------------

class TestCitationPairing:
    """Tests for _pair_citations_with_contexts pure function — chunk_N marker matching."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _pair_citations_with_contexts
        self._func = _pair_citations_with_contexts

    def test_normal_chunk_n_mapping(self):
        """chunk_1 → contexts[0], chunk_3 → contexts[2]."""
        citations = ["[source](chunk_1)", "[source](chunk_3)"]
        contexts = ["ctx_a", "ctx_b", "ctx_c", "ctx_d"]
        result = self._func(citations, contexts)
        assert len(result) == 2
        assert result[0] == (citations[0], contexts[0])
        assert result[1] == (citations[1], contexts[2])

    def test_out_of_range_chunk_index_skipped(self):
        """chunk_10 with only 3 contexts → citation skipped."""
        citations = ["[source](chunk_10)", "[source](chunk_1)"]
        contexts = ["ctx_a", "ctx_b", "ctx_c"]
        result = self._func(citations, contexts)
        assert len(result) == 1
        assert result[0] == (citations[1], contexts[0])

    def test_duplicate_chunk_n_first_wins(self):
        """Two citations for chunk_1 → only first occurrence used."""
        citations = ["[cite](chunk_1)", "[cite](chunk_1)"]
        contexts = ["ctx_a", "ctx_b"]
        result = self._func(citations, contexts)
        assert len(result) == 1
        assert result[0] == (citations[0], contexts[0])

    def test_no_chunk_n_markers_returns_empty(self):
        """No chunk_N markers → returns empty list (no positional fallback)."""
        citations = ["[source](doc_1)", "[source](ref_2)"]
        contexts = ["ctx_a", "ctx_b"]
        result = self._func(citations, contexts)
        assert result == []

    def test_mismatched_lengths_no_crash(self):
        """More citations than contexts — doesn't crash."""
        citations = [f"[src](chunk_{i})" for i in range(1, 20)]
        contexts = ["ctx_a", "ctx_b", "ctx_c"]
        result = self._func(citations, contexts)
        # Only chunk_1, chunk_2, chunk_3 should match (0-based 0,1,2)
        assert len(result) == 3
        for i, (cite, ctx) in enumerate(result):
            assert ctx == contexts[i]

    def test_empty_citations_returns_empty(self):
        """Empty citations → empty result."""
        result = self._func([], ["ctx_a", "ctx_b"])
        assert result == []

    def test_empty_contexts_returns_empty(self):
        """Empty contexts → empty result (all indices OOB)."""
        citations = ["[source](chunk_1)", "[source](chunk_2)"]
        result = self._func(citations, [])
        assert result == []

    def test_mixed_valid_invalid_markers(self):
        """chunk_N mixed with non-chunk-index markers → only chunk_N used."""
        citations = [
            "[source](chunk_1)",
            "[source](ref_abc)",
            "[source](chunk_5)",
        ]
        contexts = ["ctx_a", "ctx_b", "ctx_c", "ctx_d", "ctx_e", "ctx_f"]
        result = self._func(citations, contexts)
        assert len(result) == 2
        assert result[0] == (citations[0], contexts[0])
        assert result[1] == (citations[2], contexts[4])


# ---------------------------------------------------------------------------
# TASK-1.4 — Error resilience test
# ---------------------------------------------------------------------------

class TestErrorResilience:
    """Tests that _process_question_result catches httpx.ConnectError gracefully."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _process_question_result
        self._func = _process_question_result

    def test_connect_error_returns_error_row(self):
        """When ask_question raises httpx.ConnectError, return error dict with all metrics None."""
        import httpx

        mock_tero = AsyncMock()
        mock_tero.create_thread.return_value = "thread_test_err"
        mock_tero.ask_question.side_effect = httpx.ConnectError("Connection refused")

        row = _make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        result = asyncio.run(
            self._func(
                mock_tero, row, None,
                mock_recall, mock_precision, mock_faith,
                mock_correctness, mock_cite_faith,
            )
        )

        assert result["error"] is not None
        assert "Connection refused" in result["error"]
        assert result["correctness"] is None
        assert result["faithfulness"] is None
        assert result["context_recall"] is None
        assert result["context_precision"] is None
        assert result["citation_faithfulness"] is None
        assert result["grounded_correctness"] is None
        # Original row fields preserved
        assert result["question"] == row["question"]

    def test_openai_api_error_returns_error_row(self):
        """When any step raises Exception (generic), return error dict with all metrics None."""
        mock_tero = _make_mock_tero(answer_text="Normal answer")
        row = _make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()
        mock_recall.single_turn_ascore.side_effect = Exception("API error")

        result = asyncio.run(
            self._func(
                mock_tero, row, None,
                mock_recall, mock_precision, mock_faith,
                mock_correctness, mock_cite_faith,
            )
        )

        assert result["error"] is not None
        assert "API error" in str(result["error"])
        assert result["context_recall"] is None


# ---------------------------------------------------------------------------
# TASK-1.5 — NaN faithfulness mapping tests
# ---------------------------------------------------------------------------

class TestNanFaithfulnessMapping:
    """Tests that NaN faithfulness → None, faithfulness_valid=False, grounded_correctness=None."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _process_question_result
        self._func = _process_question_result

    def test_nan_faith_returns_none_and_valid_false(self):
        """NaN faith → faithfulness=None, faithfulness_valid=False, grounded_correctness=None."""
        mock_tero = _make_mock_tero(answer_text="Normal answer")
        row = _make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics(
            faith_val=float("nan"),
        )

        result = asyncio.run(
            self._func(
                mock_tero, row, None,
                mock_recall, mock_precision, mock_faith,
                mock_correctness, mock_cite_faith,
            )
        )

        assert result["faithfulness"] is None
        assert result["grounded_correctness"] is None

    def test_none_faith_returns_none_and_valid_false(self):
        """None faith → faithfulness=None, faithfulness_valid=False."""
        mock_tero = _make_mock_tero(answer_text="Normal answer")
        row = _make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics(
            faith_val=None,
        )

        result = asyncio.run(
            self._func(
                mock_tero, row, None,
                mock_recall, mock_precision, mock_faith,
                mock_correctness, mock_cite_faith,
            )
        )

        assert result["faithfulness"] is None
        assert result["grounded_correctness"] is None

    def test_valid_faith_returns_value_and_valid_true(self):
        """Valid faith → faithfulness=value, faithfulness_valid=True, grounded_correctness computed."""
        mock_tero = _make_mock_tero(answer_text="Normal answer")
        row = _make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics(
            faith_val=0.85, correctness_val="4",
        )

        result = asyncio.run(
            self._func(
                mock_tero, row, None,
                mock_recall, mock_precision, mock_faith,
                mock_correctness, mock_cite_faith,
            )
        )

        assert result["faithfulness"] == 0.85
        assert result["grounded_correctness"] is not None
        assert result["grounded_correctness"] == round((4 / 4) * 0.85, 3)


# ---------------------------------------------------------------------------
# T-002 — JSON column parsing
# ---------------------------------------------------------------------------

class TestParseJsonColumn:
    """Tests for _parse_json_column — JSON array parsing from CSV cells.

    FIX-1: _parse_json_column now returns tuple[list, bool] where bool
    signals a parse error. Callers must check the flag and skip metric
    computation on malformed rows (REQ-OFFLINE-003).
    """

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _parse_json_column
        self._func = _parse_json_column

    def test_valid_json_array(self):
        """Valid JSON array string → (list, False)."""
        result, had_error = self._func('["a", "b", "c"]', "retrieved_contexts", 0)
        assert result == ["a", "b", "c"]
        assert had_error is False

    def test_empty_string_returns_empty_list(self):
        """Empty string → ([], False)."""
        result, had_error = self._func("", "retrieved_contexts", 0)
        assert result == []
        assert had_error is False

    def test_malformed_json_logs_warning_and_returns_empty_with_error(self, capsys):
        """Malformed JSON → ([], True) + WARNING."""
        result, had_error = self._func("not valid json", "retrieved_contexts", 3)
        assert result == []
        assert had_error is True
        captured = capsys.readouterr()
        combined = (captured.out or "") + (captured.err or "")
        assert "WARNING" in combined
        assert "retrieved_contexts" in combined

    def test_nan_returns_empty_list_no_error(self):
        """NaN float → ([], False)."""
        import math
        result, had_error = self._func(float("nan"), "retrieved_contexts", 0)
        assert result == []
        assert had_error is False

    def test_single_element_array(self):
        """JSON array with one element → ([element], False)."""
        result, had_error = self._func('["only one"]', "citations", 1)
        assert result == ["only one"]
        assert had_error is False

    def test_empty_json_array(self):
        """Empty JSON array '[]' → ([], False)."""
        result, had_error = self._func("[]", "retrieved_contexts", 0)
        assert result == []
        assert had_error is False

    def test_non_array_json_signals_error(self, capsys):
        """JSON object (not array) → ([], True) + WARNING."""
        result, had_error = self._func('{"key": "value"}', "retrieved_contexts", 5)
        assert result == []
        assert had_error is True
        captured = capsys.readouterr()
        combined = (captured.out or "") + (captured.err or "")
        assert "WARNING" in combined
        assert "not a JSON array" in combined

    def test_whitespace_only_string_returns_empty_list(self):
        """B6: Whitespace-only string → ([], False)."""
        result, had_error = self._func("   ", "retrieved_contexts", 0)
        assert result == []
        assert had_error is False

    def test_none_value_returns_empty_list(self):
        """B6: None value → ([], False)."""
        result, had_error = self._func(None, "retrieved_contexts", 0)
        assert result == []
        assert had_error is False


# ---------------------------------------------------------------------------
# T-004 — _compute_metrics_from_sample tests
# ---------------------------------------------------------------------------

class TestComputeMetricsFromSample:
    """Tests for _compute_metrics_from_sample — shared metric computation."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        import runner
        self._compute = runner._compute_metrics_from_sample

    def _make_test_row(self, **kwargs):
        return {"question": "Test Q?", "grading_notes": "Test notes", **kwargs}

    def test_all_metrics_computed(self):
        """Happy path: all metrics return valid values → full dict with 14+ columns."""
        row = self._make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics(
            recall_val=0.8, precision_val=0.7, faith_val=0.9, correctness_val="3",
        )

        result = asyncio.run(
            self._compute(
                row=row,
                answer="Paris is the capital.",
                retrieved_contexts=["ctx1", "ctx2", "ctx3"],
                citations=["chunk_1: ctx1", "chunk_2: ctx2"],
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
                latency_ms=150.5,
            )
        )

        assert result["question"] == "Test Q?"
        assert result["grading_notes"] == "Test notes"
        assert result["response"] == "Paris is the capital."
        assert result["correctness"] == 3
        assert result["faithfulness"] == 0.9
        assert isinstance(result["context_recall"], float)
        assert isinstance(result["context_precision"], float)
        assert result["grounded_correctness"] == round((3 / 4) * 0.9, 3)
        assert result["latency_ms"] == 150.5
        assert result["error"] is None
        # Verify all 13+ output keys exist
        expected_keys = {
            "question", "grading_notes", "response", "retrieved_contexts",
            "citations", "latency_ms", "correctness", "faithfulness",
            "context_recall", "context_precision",
            "citation_faithfulness", "grounded_correctness",
            "relevant_chunk_position", "error",
        }
        assert expected_keys.issubset(set(result.keys())), f"Missing keys: {expected_keys - set(result.keys())}"

    def test_nan_faith_maps_to_none(self):
        """NaN faithfulness → None, faithfulness_valid=False, grounded_correctness=None."""
        row = self._make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics(
            faith_val=float("nan"), correctness_val="3",
        )

        result = asyncio.run(
            self._compute(
                row=row,
                answer="Paris is the capital.",
                retrieved_contexts=["ctx1", "ctx2"],
                citations=["chunk_1: ctx1"],
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

        assert result["faithfulness"] is None
        assert result["grounded_correctness"] is None

    def test_no_citations_citation_faithfulness_none(self):
        """Empty citations → citation_faithfulness=None."""
        row = self._make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        result = asyncio.run(
            self._compute(
                row=row,
                answer="Paris is the capital.",
                retrieved_contexts=["ctx1", "ctx2"],
                citations=[],
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

        assert result["citation_faithfulness"] is None
        # Other metrics still computed
        assert result["correctness"] == 3
        assert result["faithfulness"] == 0.9

    def test_no_chunk_n_markers_citation_faithfulness_none(self):
        """Citations without chunk_N markers → citation_faithfulness=None."""
        row = self._make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        result = asyncio.run(
            self._compute(
                row=row,
                answer="Paris is the capital.",
                retrieved_contexts=["ctx1", "ctx2"],
                citations=["no_chunk_marker_here", "also_no_marker"],
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

        assert result["citation_faithfulness"] is None

    def test_citation_faithfulness_computed_when_supported(self):
        """Citations with chunk_N markers → citation_faithfulness computed."""
        row = self._make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        result = asyncio.run(
            self._compute(
                row=row,
                answer="Paris is the capital.",
                retrieved_contexts=["ctx1", "ctx2"],
                citations=["chunk_1: ctx1", "chunk_2: ctx2"],
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

        # citation_faithfulness should be computed (2 supported out of 2)
        assert result["citation_faithfulness"] is not None
        assert result["citation_faithfulness"] == 1.0  # 2/2 supported

    def test_metric_computation_error_caught(self):
        """Exception during metric computation → error dict with metrics None."""
        row = self._make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()
        mock_recall.single_turn_ascore.side_effect = Exception("API error")

        result = asyncio.run(
            self._compute(
                row=row,
                answer="Paris is the capital.",
                retrieved_contexts=["ctx1", "ctx2"],
                citations=["chunk_1: ctx1"],
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

        assert result["error"] is not None
        assert "API error" in str(result["error"])
        assert result["correctness"] is None
        assert result["faithfulness"] is None
        assert result["context_recall"] is None
        assert result["context_precision"] is None
        assert result["citation_faithfulness"] is None
        assert result["grounded_correctness"] is None

    def test_missing_grading_notes_yields_correctness_none(self):
        """FIX-2: When row has no grading_notes, correctness MUST be None (REQ-OFFLINE-005)."""
        row = self._make_test_row(grading_notes="")  # empty grading_notes
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        result = asyncio.run(
            self._compute(
                row=row,
                answer="Paris is the capital.",
                retrieved_contexts=["ctx1", "ctx2"],
                citations=["chunk_1: ctx1"],
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

        # correctness MUST be None when grading_notes absent/empty
        assert result["correctness"] is None
        # correctness.ascore MUST NOT have been called
        mock_correctness.ascore.assert_not_called()
        # Other metrics still computed
        assert result["faithfulness"] == 0.9
        assert result["correctness"] is None
        # grounded_correctness requires both — also None
        assert result["grounded_correctness"] is None

    def test_empty_retrieved_contexts_yields_none_context_metrics(self):
        """FIX-4: Empty retrieved_contexts → context_recall, context_precision, faithfulness all None (REQ-OFFLINE-005)."""
        row = self._make_test_row()
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        result = asyncio.run(
            self._compute(
                row=row,
                answer="test answer",
                retrieved_contexts=[],  # EMPTY — triggers graceful degradation
                citations=[],
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

        # Context-dependent metrics → None
        assert result["context_recall"] is None
        assert result["context_precision"] is None
        assert result["faithfulness"] is None
        # grounded_correctness → None when faith is None
        assert result["grounded_correctness"] is None
        # correctness still computed (has grading_notes)
        assert result["correctness"] == 3
        # citation_faithfulness → None (no citations)
        assert result["citation_faithfulness"] is None
        # Core RAGAS metric computation must NOT have been called
        mock_recall.single_turn_ascore.assert_not_called()
        mock_precision.single_turn_ascore.assert_not_called()
        mock_faith.single_turn_ascore.assert_not_called()

    def test_mocked_metrics_produce_consistent_outputs(self):
        """B8: Two calls with identical mocked inputs → identical outputs — tests mock consistency, not real RAGAS determinism."""
        row = self._make_test_row()
        m_recall, m_precision, m_faith, m_correctness, m_cite_faith = _make_mock_metrics()

        common_kwargs = dict(
            row=row,
            answer="X marks the spot",
            retrieved_contexts=["ctx_a"],
            citations=["chunk_1: ctx_a"],
            judge_llm=None,
            context_recall=m_recall,
            context_precision=m_precision,
            faithfulness=m_faith,
            correctness=m_correctness,
            citation_faithfulness=m_cite_faith,
        )

        result1 = asyncio.run(self._compute(**common_kwargs))
        result2 = asyncio.run(self._compute(**common_kwargs))

        # All metric keys must match — identical inputs → identical outputs
        metric_keys = [
            "correctness", "faithfulness",
            "context_recall", "context_precision",
            "citation_faithfulness", "grounded_correctness",
        ]
        for key in metric_keys:
            assert result1[key] == result2[key], \
                f"Mismatch on {key}: {result1[key]} != {result2[key]}"

        # Non-metric keys must also match
        for key in ("question", "grading_notes", "response", "error", "relevant_chunk_position"):
            assert result1[key] == result2[key], \
                f"Mismatch on {key}: {result1[key]} != {result2[key]}"


# ---------------------------------------------------------------------------
# T-006 — CSV column validation tests
# ---------------------------------------------------------------------------

class TestValidateCsvColumns:
    """Tests for _validate_csv_columns — required column checking."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _validate_csv_columns
        self._func = _validate_csv_columns

    def _make_df(self, columns: list[str]) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(columns=columns)

    def test_all_required_columns_present(self):
        """All required columns present → no error."""
        df = self._make_df(["question", "response", "retrieved_contexts", "extra_col"])
        self._func(df)

    def test_missing_response_raises_valueerror(self):
        """Missing 'response' column → ValueError naming it."""
        df = self._make_df(["question", "retrieved_contexts"])
        with pytest.raises(ValueError) as exc_info:
            self._func(df)
        assert "response" in str(exc_info.value)

    def test_missing_retrieved_contexts_raises_valueerror(self):
        """Missing 'retrieved_contexts' column → ValueError naming it."""
        df = self._make_df(["question", "response"])
        with pytest.raises(ValueError) as exc_info:
            self._func(df)
        assert "retrieved_contexts" in str(exc_info.value)

    def test_missing_question_raises_valueerror(self):
        """Missing 'question' column → ValueError naming it."""
        df = self._make_df(["response", "retrieved_contexts"])
        with pytest.raises(ValueError) as exc_info:
            self._func(df)
        assert "question" in str(exc_info.value)

    def test_multiple_missing_columns(self):
        """Multiple missing columns → ValueError naming all of them."""
        df = self._make_df(["question"])
        with pytest.raises(ValueError) as exc_info:
            self._func(df)
        error_msg = str(exc_info.value)
        assert "response" in error_msg
        assert "retrieved_contexts" in error_msg


# ---------------------------------------------------------------------------
# T-008 — CLI argument validation tests
# ---------------------------------------------------------------------------

class TestCliArgs:
    """Tests for --from-csv and --models mutual exclusion + runtime validation."""

    def _parse_args(self, arg_list: list[str]) -> "argparse.Namespace":
        import argparse
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import rag_datasets as ds_module

        parser = argparse.ArgumentParser()
        parser.add_argument("--bearer-token", default=None)
        parser.add_argument("--base-url", default="http://localhost:8000")
        parser.add_argument("--models", default=None,
                           help="Comma-separated Tero model IDs to evaluate.")
        parser.add_argument("--only", choices=ds_module.ALL_DATASETS, default=None)
        parser.add_argument("--n", type=int, default=1)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--update-baseline", action="store_true")
        parser.add_argument("--compare", action="store_true")
        parser.add_argument("--agent-id", type=int, default=None)
        parser.add_argument("--from-csv", default=None,
                            help="CSV file for offline evaluation (skips Tero).")
        return parser.parse_args(arg_list)

    def test_from_csv_flag_accepted(self):
        """--from-csv with a file path is accepted."""
        args = self._parse_args(["--from-csv", "data.csv"])
        assert args.from_csv == "data.csv"
        assert args.models is None  # models not required

    def test_models_still_accepted(self):
        """--models with a dataset is still accepted (backward compat)."""
        args = self._parse_args(["--models", "gpt-5", "--only", "fetaqa"])
        assert args.models == "gpt-5"
        assert args.only == "fetaqa"

    def test_neither_models_nor_from_csv(self):
        """Neither --models nor --from-csv → both None (runtime validation catches this)."""
        args = self._parse_args(["--only", "fetaqa"])
        assert args.models is None
        assert args.from_csv is None

    def test_both_models_and_from_csv(self):
        """Both --models and --from-csv provided → both values set (runtime validation catches this)."""
        args = self._parse_args(["--models", "gpt-5", "--from-csv", "data.csv", "--only", "fetaqa"])
        assert args.models == "gpt-5"
        assert args.from_csv == "data.csv"

    def test_seed_flag_accepted(self):
        """--seed 123 is parsed correctly as int."""
        args = self._parse_args(["--seed", "123"])
        assert args.seed == 123

    def test_seed_default_is_42(self):
        """--seed defaults to 42 when omitted."""
        args = self._parse_args([])
        assert args.seed == 42


# ---------------------------------------------------------------------------
# T-013 — E2E integration test for CSV offline mode
# ---------------------------------------------------------------------------

class TestCsvModeE2E:
    """End-to-end tests for _run_csv_mode — full pipeline from CSV to stats."""

    def _make_result_row(self, question, correctness=3, faith=0.9, recall=0.8, precision=0.7, cite_faith=1.0, grounded=0.675):
        return {
            "question": question,
            "grading_notes": "Test notes",
            "error": None,
            "response": f"Answer to {question}",
            "retrieved_contexts": "ctx1 | ctx2",
            "citations": "chunk_1: ctx1",
            "latency_ms": 100.0,
            "correctness": correctness,
            "faithfulness": faith,
            "context_recall": recall,
            "context_precision": precision,
            "citation_faithfulness": cite_faith,
            "grounded_correctness": grounded,
            "relevant_chunk_position": 1,
        }

    async def _mock_compute(self, row, answer, retrieved_contexts, citations,
                            judge_llm, context_recall, context_precision,
                            faithfulness, correctness, citation_faithfulness,
                            latency_ms=None):
        q = row["question"]
        key = q[:20]
        data = {
            "What is the capital": self._make_result_row(q, correctness=3, faith=0.9, recall=0.8, precision=0.7, cite_faith=1.0, grounded=0.675),
            "What is 2+2?": self._make_result_row(q, correctness=4, faith=0.95, recall=0.85, precision=0.75, cite_faith=1.0, grounded=0.95),
            "Who wrote Hamlet?": self._make_result_row(q, correctness=2, faith=0.7, recall=0.6, precision=0.55, cite_faith=1.0, grounded=0.35),
        }
        for prefix, result in data.items():
            if q.startswith(prefix):
                return result
        return self._make_result_row(q)

    def test_full_csv_pipeline_writes_output_and_computes_stats(self):
        """3-row valid_full.csv → _run_csv_mode → output CSV with 14+ cols, stats computed."""
        import tempfile
        from pathlib import Path
        import runner

        # Create temp CSV from our fixture
        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "valid_full.csv"

        # Create args object
        args = argparse.Namespace()
        args.from_csv = str(fixture_path)

        # Mock the LLM creation and metric computation
        with patch.object(runner, '_build_metrics', return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )):
            with patch.object(runner, '_compute_metrics_from_sample', side_effect=self._mock_compute):
                with patch.object(runner, 'AsyncOpenAI'):
                    # Patch EVALS_DIR for output isolation
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir)
                        with patch.object(runner, 'EVALS_DIR', tmp_path):
                            asyncio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

                        # Verify output CSV was written
                        offline_dir = tmp_path / "experiments" / "offline"
                        csvs = list(offline_dir.glob("*.csv"))
                        assert len(csvs) == 1, f"Expected 1 output CSV, got {len(csvs)}"
                        output_csv = csvs[0]

                        # Read the output CSV
                        df = pd.read_csv(output_csv, sep=";")
                        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

                        # Verify 13+ output columns
                        expected_cols = {"question", "response", "retrieved_contexts", "citations",
                                         "latency_ms", "correctness", "faithfulness",
                                         "context_recall", "context_precision", "citation_faithfulness",
                                         "grounded_correctness", "relevant_chunk_position", "error",
                                         "grading_notes"}
                        assert expected_cols.issubset(set(df.columns)), \
                            f"Missing columns: {expected_cols - set(df.columns)}"

                        # Verify stats computed (indirect: check the file structure)
                        assert "model_id" in df.columns, "model_id column present from valid_full.csv"
                        assert set(df["model_id"].unique()) == {"gpt-5"}

                        # Verify correctness values from mock
                        correctness_vals = list(df["correctness"])
                        assert 3 in correctness_vals
                        assert 4 in correctness_vals
                        assert 2 in correctness_vals

    def test_csv_mode_zero_tero_imports(self):
        """_run_csv_mode must NOT import TeroClient."""
        import runner
        import inspect

        src = inspect.getsource(runner._run_csv_mode)
        # Check for actual import statements, not docstring mentions
        assert "import TeroClient" not in src, \
            "_run_csv_mode must not import TeroClient"
        assert "from tero_client" not in src, \
            "_run_csv_mode must not import from tero_client"

    def test_minimal_csv_produces_results(self):
        """valid_minimal.csv (only required columns) → results with correctness=None, context metrics computed."""
        import tempfile
        from pathlib import Path
        import runner

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "valid_minimal.csv"

        args = argparse.Namespace()
        args.from_csv = str(fixture_path)

        with patch.object(runner, '_build_metrics', return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )):
            with patch.object(runner, '_compute_metrics_from_sample', side_effect=self._mock_compute):
                with patch.object(runner, 'AsyncOpenAI'):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir)
                        with patch.object(runner, 'EVALS_DIR', tmp_path):
                            asyncio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

                        offline_dir = tmp_path / "experiments" / "offline"
                        csvs = list(offline_dir.glob("*.csv"))
                        assert len(csvs) == 1
                        df = pd.read_csv(csvs[0], sep=";")
                        assert len(df) == 2  # 2 rows in valid_minimal.csv

    def test_missing_csv_file_exits(self):
        """Non-existent CSV file → SystemExit."""
        import runner

        args = argparse.Namespace()
        args.from_csv = "nonexistent_file.csv"

        with pytest.raises(SystemExit):
            asyncio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

    def test_multi_model_csv_groups_and_compares(self):
        """FIX-5: multi_model.csv with 2 models → per-model stats + both models in output."""
        import tempfile
        from pathlib import Path
        import runner

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "multi_model.csv"

        args = argparse.Namespace()
        args.from_csv = str(fixture_path)

        with patch.object(runner, '_build_metrics', return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )):
            with patch.object(runner, '_compute_metrics_from_sample', side_effect=self._mock_compute):
                with patch.object(runner, 'AsyncOpenAI'):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir)
                        with patch.object(runner, 'EVALS_DIR', tmp_path):
                            asyncio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

                        offline_dir = tmp_path / "experiments" / "offline"
                        csvs = list(offline_dir.glob("*.csv"))
                        assert len(csvs) == 1, f"Expected 1 output CSV, got {len(csvs)}"
                        df = pd.read_csv(csvs[0], sep=";")
                        assert len(df) == 4, f"Expected 4 rows, got {len(df)}"

                        # Multi-model grouping: both model_id values present
                        assert "model_id" in df.columns, "model_id column must be present"
                        unique_models = set(df["model_id"].unique())
                        assert unique_models == {"gpt-5", "claude"}, \
                            f"Expected both gpt-5 and claude, got {unique_models}"

                        # Verify row counts per model
                        gpt5_rows = df[df["model_id"] == "gpt-5"]
                        claude_rows = df[df["model_id"] == "claude"]
                        assert len(gpt5_rows) == 2, f"Expected 2 gpt-5 rows, got {len(gpt5_rows)}"
                        assert len(claude_rows) == 2, f"Expected 2 claude rows, got {len(claude_rows)}"

    def test_malformed_json_csv_yields_error_rows(self):
        """B2: malformed_json.csv → error rows with all-None metrics for malformed JSON."""
        import tempfile
        from pathlib import Path
        import runner

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "malformed_json.csv"

        args = argparse.Namespace()
        args.from_csv = str(fixture_path)

        with patch.object(runner, '_build_metrics', return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )):
            with patch.object(runner, '_compute_metrics_from_sample', side_effect=self._mock_compute):
                with patch.object(runner, 'AsyncOpenAI'):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir)
                        with patch.object(runner, 'EVALS_DIR', tmp_path):
                            asyncio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

                        offline_dir = tmp_path / "experiments" / "offline"
                        csvs = list(offline_dir.glob("*.csv"))
                        assert len(csvs) == 1
                        df = pd.read_csv(csvs[0], sep=";")
                        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

                        # Row 0: malformed JSON "this is not valid json at all" → error
                        row0 = df.iloc[0]
                        assert row0["error"] == "json_parse_error"
                        assert row0["correctness"] is None or (isinstance(row0["correctness"], float) and math.isnan(row0["correctness"]))
                        assert row0["faithfulness"] is None or (isinstance(row0["faithfulness"], float) and math.isnan(row0["faithfulness"]))
                        assert row0["context_recall"] is None or (isinstance(row0["context_recall"], float) and math.isnan(row0["context_recall"]))

                        # Row 1: valid JSON → no error (mocked compute returns values)
                        row1 = df.iloc[1]
                        assert row1["error"] is None or (isinstance(row1["error"], float) and math.isnan(row1["error"]))

                        # Row 2: truncated JSON → error
                        row2 = df.iloc[2]
                        assert row2["error"] == "json_parse_error"

    def test_missing_column_csv_raises_valueerror(self):
        """B2: missing_column.csv (no 'response' column) → ValueError."""
        import runner
        from pathlib import Path

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "missing_column.csv"

        args = argparse.Namespace()
        args.from_csv = str(fixture_path)

        with patch.object(runner, '_build_metrics', return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )):
            with patch.object(runner, 'AsyncOpenAI'):
                with pytest.raises(SystemExit):
                    asyncio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

    def test_nan_question_skipped_with_error(self):
        """NaN question or response → row skipped with missing_required_value error, all-None metrics."""
        import tempfile
        from pathlib import Path
        import runner

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "nan_question.csv"

        args = argparse.Namespace()
        args.from_csv = str(fixture_path)

        with patch.object(runner, '_build_metrics', return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )):
            with patch.object(runner, '_compute_metrics_from_sample', side_effect=self._mock_compute):
                with patch.object(runner, 'AsyncOpenAI'):
                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir)
                        with patch.object(runner, 'EVALS_DIR', tmp_path):
                            asyncio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

                        offline_dir = tmp_path / "experiments" / "offline"
                        csvs = list(offline_dir.glob("*.csv"))
                        assert len(csvs) == 1
                        df = pd.read_csv(csvs[0], sep=";")
                        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

                        # Row 1 (index 1): NaN question → should be skipped with error
                        row1 = df.iloc[1]
                        assert row1["error"] == "missing_required_value"
                        assert row1["correctness"] is None or (isinstance(row1["correctness"], float) and math.isnan(row1["correctness"]))
                        assert row1["faithfulness"] is None or (isinstance(row1["faithfulness"], float) and math.isnan(row1["faithfulness"]))

                        # Row 0 and 2: valid rows → no error
                        row0 = df.iloc[0]
                        assert row0["error"] is None or (isinstance(row0["error"], float) and math.isnan(row0["error"]))
                        row2 = df.iloc[2]
                        assert row2["error"] is None or (isinstance(row2["error"], float) and math.isnan(row2["error"]))


# ---------------------------------------------------------------------------
# Task 2.1 — configure_docs_tool with config parameter (POST-based)
# ---------------------------------------------------------------------------

class TestConfigureDocsTool:
    """Tests for TeroClient.configure_docs_tool(config) — POST-based config-aware tool setup."""

    @pytest.fixture(autouse=True)
    def _import_class(self):
        from tero_client import TeroClient
        self._TeroClient = TeroClient

    def test_configure_docs_tool_with_config_sends_post(self):
        """When config is provided, POST body includes config dict instead of {}."""
        import httpx
        from unittest.mock import AsyncMock, patch, MagicMock

        client = self._TeroClient("http://localhost:8000", 9, "token")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx, 'AsyncClient') as mock_client_class:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_ctx.post = AsyncMock(return_value=mock_resp)
            mock_client_class.return_value = mock_ctx

            asyncio.run(client.configure_docs_tool(
                config={"skipDescriptions": True, "advancedFileProcessing": False}))

            # Verify POST was called with the config dict, not {}
            mock_ctx.post.assert_called_once()
            call_kwargs = mock_ctx.post.call_args
            payload = call_kwargs[1]["json"]
            assert payload["config"] == {"skipDescriptions": True, "advancedFileProcessing": False}

    def test_configure_docs_tool_without_config_uses_empty_dict(self):
        """When config is None (default), POST body config is {} (backward compat)."""
        import httpx
        from unittest.mock import AsyncMock, patch, MagicMock

        client = self._TeroClient("http://localhost:8000", 9, "token")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx, 'AsyncClient') as mock_client_class:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_ctx.post = AsyncMock(return_value=mock_resp)
            mock_client_class.return_value = mock_ctx

            asyncio.run(client.configure_docs_tool())

            mock_ctx.post.assert_called_once()
            call_kwargs = mock_ctx.post.call_args
            payload = call_kwargs[1]["json"]
            assert payload["config"] == {}

    def test_configure_docs_tool_overwrites_when_already_configured(self):
        """When docs tool already configured, POST still fires (overwrite behavior)."""
        import httpx
        from unittest.mock import AsyncMock, patch, MagicMock

        client = self._TeroClient("http://localhost:8000", 9, "token")

        mock_resp = MagicMock()
        mock_resp.raise_for_status = MagicMock()

        with patch.object(httpx, 'AsyncClient') as mock_client_class:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_ctx)
            mock_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_ctx.post = AsyncMock(return_value=mock_resp)
            mock_client_class.return_value = mock_ctx

            asyncio.run(client.configure_docs_tool(
                config={"skipDescriptions": True}))

            # POST must always be called — overwrites any existing config
            mock_ctx.post.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 4: cost_report() tests (Task 4.3)
# ---------------------------------------------------------------------------

class TestCostReport:
    """Tests for cost_report() — embedding token cost output format."""

    def test_cost_report_output_format(self, capsys):
        """cost_report prints all expected fields: model, tokens, cost, LLM desc note."""
        from runner import cost_report
        cost_report(1000000)
        captured = capsys.readouterr()
        assert "=== Cost Report ===" in captured.out
        assert "Embedding model      : text-embedding-3-small" in captured.out
        assert "1,000,000" in captured.out
        assert "Total estimated USD  : $" in captured.out
        assert "LLM description tokens" not in captured.out

    def test_cost_report_custom_model_name(self, capsys):
        """cost_report accepts custom model name parameter."""
        from runner import cost_report
        cost_report(500, model="text-embedding-ada-002")
        captured = capsys.readouterr()
        assert "text-embedding-ada-002" in captured.out

    def test_cost_report_zero_tokens(self, capsys):
        """cost_report with zero tokens shows $0.000000 total."""
        from runner import cost_report
        cost_report(0)
        captured = capsys.readouterr()
        assert "0" in captured.out
        assert "$0.000000" in captured.out

    def test_cost_report_small_token_count(self, capsys):
        """cost_report handles small token counts correctly."""
        from runner import cost_report
        cost_report(100)
        captured = capsys.readouterr()
        assert "100" in captured.out
        # 100/1000 * 0.00002 = 0.000002
        assert "$0.000002" in captured.out


# ---------------------------------------------------------------------------
# Phase 4: do_index() tests — index subcommand (Task 4.2)
# ---------------------------------------------------------------------------

class TestIndexSubcommand:
    """Tests for do_index() — mocked Tero, tiktoken, and dataset loading."""

    @staticmethod
    def _make_index_args(**overrides):
        """Build an argparse.Namespace matching the index subparser."""
        import argparse
        defaults = dict(
            dataset="ragbench",
            agent_id=9,
            max_docs=None,
            bearer_token="test-token",
            base_url="http://localhost:8000",
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_do_index_loads_corpus_and_uploads(self):
        """Full do_index workflow: load corpus (n=0), configure agent, upload docs, cost_report."""
        import runner
        from unittest.mock import AsyncMock, MagicMock, patch
        import asyncio as aio

        args = self._make_index_args()
        mock_corpus = ["doc one text", "doc two", "doc three text"]

        mock_tero = AsyncMock()
        mock_tero.list_file_ids = AsyncMock(return_value=[])
        mock_tero.configure_docs_tool = AsyncMock()
        mock_tero.upload_document = AsyncMock(side_effect=[101, 102, 103])
        mock_tero.wait_files_processed = AsyncMock()

        # Mock tiktoken: each char = 1 token for predictable counts
        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(side_effect=lambda s: list(range(len(s))))

        with patch("tero_client.TeroClient", return_value=mock_tero):
            with patch.object(runner.ds_module, "load_one", return_value=([], mock_corpus)) as mock_load:
                with patch("tiktoken.get_encoding", return_value=mock_enc):
                    with patch.object(runner, "cost_report") as mock_cost:
                        aio.run(runner.do_index(args))

        # Corpus loaded with n=0 (full corpus, zero questions)
        mock_load.assert_called_once_with("ragbench", n=0)

        # Agent configured with skipDescriptions=true
        mock_tero.configure_docs_tool.assert_called_once()
        config_arg = mock_tero.configure_docs_tool.call_args[0][0]
        assert config_arg == {"skipDescriptions": True}

        # All 3 docs uploaded
        assert mock_tero.upload_document.call_count == 3
        mock_tero.wait_files_processed.assert_called_once_with([101, 102, 103])

        # cost_report called with total token count (12+7+14 = 33 chars = 33 tokens)
        mock_cost.assert_called_once_with(33)

    def test_do_index_max_docs_truncation(self):
        """When --max-docs is set, corpus is truncated before upload."""
        import runner
        from unittest.mock import AsyncMock, MagicMock, patch
        import asyncio as aio

        args = self._make_index_args(max_docs=2)
        mock_corpus = ["d1", "d2", "d3", "d4", "d5"]

        mock_tero = AsyncMock()
        mock_tero.list_file_ids = AsyncMock(return_value=[])  # first-run: no existing files
        mock_tero.delete_all_files = AsyncMock(return_value=0)
        mock_tero.delete_docs_tool = AsyncMock()
        mock_tero.configure_docs_tool = AsyncMock()
        mock_tero.upload_document = AsyncMock(side_effect=[201, 202])
        mock_tero.wait_files_processed = AsyncMock()

        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(side_effect=lambda s: list(range(len(s))))

        with patch("tero_client.TeroClient", return_value=mock_tero):
            with patch.object(runner.ds_module, "load_one", return_value=([], mock_corpus)):
                with patch("tiktoken.get_encoding", return_value=mock_enc):
                    with patch.object(runner, "cost_report"):
                        aio.run(runner.do_index(args))

        # Only 2 docs uploaded (truncated)
        assert mock_tero.upload_document.call_count == 2
        mock_tero.wait_files_processed.assert_called_once_with([201, 202])

    def test_do_index_missing_bearer_token_exits(self):
        """When bearer_token is missing (None + empty env), SystemExit(1)."""
        import runner
        import asyncio as aio

        args = self._make_index_args(bearer_token=None)
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                aio.run(runner.do_index(args))
        assert exc_info.value.code == 1

    def test_do_index_bearer_token_from_env(self):
        """bearer_token falls back to BEARER_TOKEN env var."""
        import runner
        from unittest.mock import AsyncMock, MagicMock, patch
        import asyncio as aio

        args = self._make_index_args(bearer_token=None)
        mock_corpus = ["doc"]
        mock_tero = AsyncMock()
        mock_tero.configure_docs_tool = AsyncMock()
        mock_tero.upload_document = AsyncMock(return_value=1)
        mock_tero.wait_files_processed = AsyncMock()
        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(return_value=[])

        with patch.dict("os.environ", {"BEARER_TOKEN": "env-token"}):
            with patch("tero_client.TeroClient", return_value=mock_tero) as mock_tc:
                with patch.object(runner.ds_module, "load_one", return_value=([], mock_corpus)):
                    with patch("tiktoken.get_encoding", return_value=mock_enc):
                        with patch.object(runner, "cost_report"):
                            aio.run(runner.do_index(args))

        # TeroClient should be created with the env token
        mock_tc.assert_called_once_with("http://localhost:8000", 9, "env-token")


# ---------------------------------------------------------------------------
# Phase 4: do_eval() tests — eval subcommand (Task 4.2)
# ---------------------------------------------------------------------------

class TestEvalSubcommand:
    """Tests for do_eval() — CSV offline mode, live mode, missing required flags."""

    @staticmethod
    def _make_eval_args(**overrides):
        """Build an argparse.Namespace matching the eval subparser."""
        import argparse
        defaults = dict(
            dataset="ragbench",
            agent_id=9,
            max_questions=1,
            models="gpt-5",
            bearer_token="test-token",
            base_url="http://localhost:8000",
            from_csv=None,
            update_baseline=False,
            compare=False,
            n=1,
            seed=42,
            only=None,
            judge_model="gemini-3.5-flash",
            concurrency=5,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_eval_missing_agent_id_live_mode_exits(self):
        """When --agent-id is None in live mode (no --from-csv), SystemExit(1)."""
        import runner
        import asyncio as aio

        args = self._make_eval_args(agent_id=None)
        with pytest.raises(SystemExit) as exc_info:
            aio.run(runner.do_eval(args))
        assert exc_info.value.code == 1

    def test_eval_missing_models_live_mode_exits(self):
        """When --models is None/empty in live mode, SystemExit(1)."""
        import runner
        import asyncio as aio

        args = self._make_eval_args(models=None)
        with pytest.raises(SystemExit) as exc_info:
            aio.run(runner.do_eval(args))
        assert exc_info.value.code == 1

    def test_csv_mode_skips_agent_id_check(self):
        """When --from-csv is provided, agent_id check is bypassed — CSV mode activates."""
        import runner
        from pathlib import Path
        from unittest.mock import AsyncMock, patch
        import asyncio as aio

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "valid_full.csv"
        args = self._make_eval_args(
            agent_id=None,
            models=None,
            from_csv=str(fixture_path),
        )

        with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
            with patch.object(runner, "_run_csv_mode", new=AsyncMock()) as mock_csv:
                aio.run(runner.do_eval(args))

        # _run_csv_mode should be called (CSV path taken, agent_id not needed)
        mock_csv.assert_called_once()

    def test_csv_mode_rejects_models_flag(self):
        """--from-csv and --models are mutually exclusive → SystemExit(1)."""
        import runner
        import asyncio as aio

        args = self._make_eval_args(from_csv="data.csv", models="gpt-5")
        with pytest.raises(SystemExit) as exc_info:
            aio.run(runner.do_eval(args))
        assert exc_info.value.code == 1

    def test_eval_empty_models_string_exits(self):
        """--models with empty/whitespace string → SystemExit(1)."""
        import runner
        import asyncio as aio

        args = self._make_eval_args(models="  ,  ")
        with pytest.raises(SystemExit) as exc_info:
            aio.run(runner.do_eval(args))
        assert exc_info.value.code == 1

    def test_csv_mode_missing_google_api_key_exits(self):
        """--from-csv with gemini judge model and no GOOGLE_API_KEY → SystemExit(1)."""
        import runner
        from pathlib import Path
        import asyncio as aio

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "valid_minimal.csv"
        args = self._make_eval_args(
            agent_id=None,
            models=None,
            from_csv=str(fixture_path),
        )

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                aio.run(runner.do_eval(args))
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Phase 1: JudgeCostTracker — token counting and cost reporting
# ---------------------------------------------------------------------------

class TestJudgeCostTracker:
    """Tests for JudgeCostTracker class (Phase 1, Task 1.3) — pure-function tests
    for token accumulation and cost conversion."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from runner import JudgeCostTracker
        self.Tracker = JudgeCostTracker

    def _make_mock_client(self):
        """Create a mock AsyncOpenAI with a replaceable chat.completions.create."""
        from unittest.mock import MagicMock, AsyncMock
        client = MagicMock()
        client.chat.completions.create = AsyncMock()
        return client

    def _make_mock_response(self, prompt_tokens=1000, completion_tokens=500):
        """Create a mock chat completion response with usage info."""
        from unittest.mock import MagicMock
        response = MagicMock()
        response.usage.prompt_tokens = prompt_tokens
        response.usage.completion_tokens = completion_tokens
        return response

    # ── RED-1: Initial state ──

    def test_initial_state_zero_tokens(self):
        """Tracker starts with zero prompt and completion tokens."""
        client = self._make_mock_client()
        tracker = self.Tracker(client)
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0

    # ── RED-2: Single call accumulation ──

    def test_accumulates_tokens_from_single_call(self):
        """One chat completion → tracker accumulates prompt + completion tokens."""
        client = self._make_mock_client()
        mock_resp = self._make_mock_response(prompt_tokens=1000, completion_tokens=500)
        client.chat.completions.create.return_value = mock_resp

        tracker = self.Tracker(client)
        import asyncio
        asyncio.run(client.chat.completions.create(model="gemini-3.5-flash", messages=[]))

        assert tracker.prompt_tokens == 1000
        assert tracker.completion_tokens == 500

    # ── RED-3: Multiple calls sum correctly ──

    def test_accumulates_across_multiple_calls(self):
        """Three calls → tokens accumulate correctly across all calls."""
        client = self._make_mock_client()
        call_responses = [
            self._make_mock_response(prompt_tokens=100, completion_tokens=50),
            self._make_mock_response(prompt_tokens=200, completion_tokens=100),
            self._make_mock_response(prompt_tokens=300, completion_tokens=150),
        ]
        client.chat.completions.create.side_effect = call_responses

        tracker = self.Tracker(client)
        import asyncio
        async def _run():
            await client.chat.completions.create(model="gemini", messages=[])
            await client.chat.completions.create(model="gemini", messages=[])
            await client.chat.completions.create(model="gemini", messages=[])
        asyncio.run(_run())

        assert tracker.prompt_tokens == 600   # 100+200+300
        assert tracker.completion_tokens == 300  # 50+100+150

    # ── RED-4: Response without usage is safe ──

    def test_response_without_usage_is_safe(self):
        """Response missing usage attribute doesn't crash token counting."""
        client = self._make_mock_client()
        # Use a plain object without .usage so getattr falls back to None
        class BareResponse:
            pass
        client.chat.completions.create.return_value = BareResponse()

        tracker = self.Tracker(client)
        import asyncio
        asyncio.run(client.chat.completions.create(model="gemini", messages=[]))

        # Should not crash — tokens stay at zero
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0

    # ── RED-5: cost_summary format ──

    def test_cost_summary_format(self, capsys):
        """cost_summary prints all expected fields: tokens, rates, total USD."""
        client = self._make_mock_client()
        tracker = self.Tracker(client)
        tracker.prompt_tokens = 15000
        tracker.completion_tokens = 3000

        tracker.cost_summary()

        captured = capsys.readouterr()
        assert "=== Judge Cost Report ===" in captured.out
        assert "Prompt tokens        : 15,000" in captured.out
        assert "Completion tokens    : 3,000" in captured.out
        assert "Prompt cost/1K       : $" in captured.out
        assert "Completion cost/1K   : $" in captured.out
        assert "Total judge USD      : $" in captured.out

    # ── RED-6: Zero tokens → zero cost ──

    def test_cost_summary_zero_tokens_shows_zero(self, capsys):
        """Zero tokens produces $0.000000 total."""
        client = self._make_mock_client()
        tracker = self.Tracker(client)
        tracker.cost_summary()
        captured = capsys.readouterr()
        assert "$0.000000" in captured.out

    # ── RED-7: Cost calculation with real numbers ──

    def test_cost_calculation_accuracy(self, monkeypatch):
        """Cost formula: (prompt_tokens/1000)*prompt_rate + (completion_tokens/1000)*completion_rate."""
        import runner
        monkeypatch.setattr(runner, "JUDGE_COST_PER_1K_PROMPT_TOKENS", 0.15)
        monkeypatch.setattr(runner, "JUDGE_COST_PER_1K_COMPLETION_TOKENS", 0.60)

        client = self._make_mock_client()
        tracker = runner.JudgeCostTracker(client)
        tracker.prompt_tokens = 15000
        tracker.completion_tokens = 3000

        # prompt_cost = (15000/1000)*0.15 = 2.25
        # completion_cost = (3000/1000)*0.60 = 1.80
        # total = 4.05
        import io, sys
        old_stdout = sys.stdout
        try:
            captured = io.StringIO()
            sys.stdout = captured
            tracker.cost_summary()
            output = captured.getvalue()
        finally:
            sys.stdout = old_stdout

        # cost_summary prints the rates and the total
        assert "Prompt cost/1K       : $0.150000" in output
        assert "Completion cost/1K   : $0.600000" in output
        assert "Total judge USD      : $4.050000" in output


# ---------------------------------------------------------------------------
# Phase 3: Eval wiring — JudgeCostTracker integration
# ---------------------------------------------------------------------------

class TestEvalJudgeCostWiring:
    """Tests that JudgeCostTracker wraps AsyncOpenAI in do_eval() and _run_csv_mode()
    (Phase 3, Tasks 3.1-3.4)."""

    def test_run_csv_mode_creates_judge_cost_tracker(self):
        """_run_csv_mode wraps the AsyncOpenAI client via JudgeCostTracker."""
        import runner
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock, patch
        import asyncio as aio
        import tempfile

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "valid_full.csv"
        args = MagicMock()
        args.from_csv = str(fixture_path)

        with patch.object(runner, '_build_metrics', return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )):
            with patch.object(runner, '_compute_metrics_from_sample', new=AsyncMock(
                return_value={"question": "Q", "grading_notes": "", "error": None,
                              "correctness": 3, "faithfulness": 0.9,
                              "context_recall": 0.8, "context_precision": 0.7,
                              "citation_faithfulness": 1.0, "grounded_correctness": 0.675,
                              "response": "A", "retrieved_contexts": "c", "citations": "",
                              "latency_ms": 100.0, "relevant_chunk_position": 1}
            )):
                with patch.object(runner, 'JudgeCostTracker') as mock_tracker_cls:
                    mock_tracker = MagicMock()
                    mock_tracker.prompt_tokens = 0
                    mock_tracker.completion_tokens = 0
                    mock_tracker_cls.return_value = mock_tracker

                    with tempfile.TemporaryDirectory() as tmpdir:
                        tmp_path = Path(tmpdir)
                        with patch.object(runner, 'EVALS_DIR', tmp_path):
                            aio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

                    # JudgeCostTracker must be instantiated with an AsyncOpenAI client
                    mock_tracker_cls.assert_called_once()
                    # cost_summary must be called after all rows
                    mock_tracker.cost_summary.assert_called_once()

    def test_do_eval_live_creates_judge_cost_tracker(self):
        """do_eval() live path instantiates JudgeCostTracker and calls cost_summary()."""
        import runner
        from unittest.mock import AsyncMock, MagicMock, patch
        import asyncio as aio
        import argparse
        import tempfile
        from pathlib import Path

        args = argparse.Namespace(
            dataset="ragbench", agent_id=9, max_questions=1, models="gpt-5",
            bearer_token="test-token", base_url="http://localhost:8000",
            from_csv=None, update_baseline=False, compare=False,
            n=1, seed=42, only=None, judge_model="gemini-3.5-flash",
            concurrency=5,
        )

        mock_tero = AsyncMock()
        mock_tero.set_agent_model = AsyncMock()

        class _FakeExperimentResult:
            def save(self):
                pass

        def _make_experiment_decorator():
            def decorator(fn):
                async def arun(dataset):
                    return _FakeExperimentResult()
                fn.arun = arun
                return fn
            return decorator

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            with patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}):
                with patch("tero_client.TeroClient", return_value=mock_tero):
                    with patch.object(runner.ds_module, "load_one", return_value=(
                        [{"question": "Q?", "grading_notes": "notes"}], []
                    )):
                        with patch.object(runner, "AsyncOpenAI") as mock_ai:
                            with patch.object(runner, "JudgeCostTracker") as mock_tracker_cls:
                                mock_tracker = MagicMock()
                                mock_tracker_cls.return_value = mock_tracker
                                with patch.object(runner, "EXPERIMENTS_DIR", tmp_path):
                                    with patch("ragas.Dataset") as mock_ragas_ds:
                                        mock_ragas_ds.return_value = MagicMock()
                                        with patch.object(
                                            runner, "experiment",
                                            side_effect=_make_experiment_decorator,
                                        ):
                                            with patch.object(runner, "llm_factory"):
                                                with patch.object(runner, "_build_metrics", return_value=(
                                                    MagicMock(), MagicMock(), MagicMock(),
                                                    MagicMock(), MagicMock(),
                                                )):
                                                    with patch.object(runner.analysis, "compute_stats"):
                                                        with patch.object(runner.analysis, "print_summary") as mock_ps:
                                                            aio.run(runner.do_eval(args))
                            mock_tracker_cls.assert_called_once()
                            mock_tracker.cost_summary.assert_called_once()


# ---------------------------------------------------------------------------
# Phase 3 — Sanity wiring tests
# ---------------------------------------------------------------------------

class TestSanityWiring:
    """Tests that runner paths wire run_sanity_checks and pass df+dataset to print_summary."""

    def test_run_csv_mode_calls_run_sanity_checks(self):
        """_run_csv_mode calls run_sanity_checks after writing results CSV."""
        import runner
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock, patch
        import asyncio as aio
        import tempfile
        import argparse

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "valid_full.csv"
        args = argparse.Namespace()
        args.from_csv = str(fixture_path)
        # Dataset is pulled from the args object — patch it
        args.dataset = "ragbench"

        with patch.object(runner, '_build_metrics', return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )):
            with patch.object(runner, '_compute_metrics_from_sample', new=AsyncMock(
                return_value={"question": "Q", "grading_notes": "", "error": None,
                              "correctness": 3, "faithfulness": 0.9,
                              "context_recall": 0.8, "context_precision": 0.7,
                              "citation_faithfulness": 1.0, "grounded_correctness": 0.675,
                              "response": "A", "retrieved_contexts": "c", "citations": "",
                              "latency_ms": 100.0, "relevant_chunk_position": 1}
            )):
                with patch.object(runner, 'AsyncOpenAI'):
                    with patch.object(runner, 'JudgeCostTracker') as mock_tracker_cls:
                        mock_tracker = MagicMock()
                        mock_tracker.prompt_tokens = 0
                        mock_tracker.completion_tokens = 0
                        mock_tracker_cls.return_value = mock_tracker

                        with patch("sanity_checks.run_sanity_checks") as mock_rsc:
                            with patch.object(runner.analysis, "print_summary") as mock_ps:
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    tmp_path = Path(tmpdir)
                                    with patch.object(runner, 'EVALS_DIR', tmp_path):
                                        aio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

                        # Verify run_sanity_checks called after CSV write
                        mock_rsc.assert_called_once()
                        call_args = mock_rsc.call_args
                        assert call_args[0][1] == "ragbench", \
                            f"Expected dataset='ragbench', got {call_args[0][1]}"

    def test_run_csv_mode_passes_df_and_dataset_to_print_summary(self):
        """print_summary called with df= and dataset= keyword args."""
        import runner
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock, patch
        import asyncio as aio
        import tempfile
        import argparse

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "valid_full.csv"
        args = argparse.Namespace()
        args.from_csv = str(fixture_path)
        args.dataset = "fetaqa"

        with patch.object(runner, '_build_metrics', return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )):
            with patch.object(runner, '_compute_metrics_from_sample', new=AsyncMock(
                return_value={"question": "Q", "grading_notes": "", "error": None,
                              "correctness": 3, "faithfulness": 0.9,
                              "context_recall": 0.8, "context_precision": 0.7,
                              "citation_faithfulness": 1.0, "grounded_correctness": 0.675,
                              "response": "A", "retrieved_contexts": "c", "citations": "",
                              "latency_ms": 100.0, "relevant_chunk_position": 1}
            )):
                with patch.object(runner, 'AsyncOpenAI'):
                    with patch.object(runner, 'JudgeCostTracker') as mock_tracker_cls:
                        mock_tracker = MagicMock()
                        mock_tracker.prompt_tokens = 0
                        mock_tracker.completion_tokens = 0
                        mock_tracker_cls.return_value = mock_tracker
                        with patch("sanity_checks.run_sanity_checks"):
                            with patch.object(runner.analysis, "print_summary") as mock_ps:
                                with tempfile.TemporaryDirectory() as tmpdir:
                                    tmp_path = Path(tmpdir)
                                    with patch.object(runner, 'EVALS_DIR', tmp_path):
                                        aio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

                        # Verify print_summary called with df and dataset keyword args
                        # valid_full.csv has model_id → multi-model path uses "offline/{model}"
                        assert mock_ps.call_count >= 1
                        call_kwargs = mock_ps.call_args[1]
                        assert "df" in call_kwargs, "Expected df= keyword argument"
                        assert "dataset" in call_kwargs, "Expected dataset= keyword argument"

    def test_csv_rewrite_on_parametric_suspect_column(self):
        """When parametric_suspect column is added, CSV is re-written."""
        import runner
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock, patch
        import asyncio as aio
        import tempfile
        import argparse
        import pandas as pd

        fixture_path = Path(__file__).parent / "evals" / "test_fixtures" / "valid_full.csv"
        args = argparse.Namespace()
        args.from_csv = str(fixture_path)
        args.dataset = "fetaqa"

        with patch.object(runner, '_build_metrics', return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )):
            with patch.object(runner, '_compute_metrics_from_sample', new=AsyncMock(
                return_value={"question": "Q", "grading_notes": "", "error": None,
                              "correctness": 3, "faithfulness": 0.9,
                              "context_recall": 0.8, "context_precision": 0.7,
                              "citation_faithfulness": 1.0, "grounded_correctness": 0.675,
                              "response": "A", "retrieved_contexts": "c", "citations": "",
                              "latency_ms": 100.0, "relevant_chunk_position": 1}
            )):

                with patch.object(runner, 'AsyncOpenAI'):
                    with patch.object(runner, 'JudgeCostTracker') as mock_tracker_cls:
                        mock_tracker = MagicMock()
                        mock_tracker.prompt_tokens = 0
                        mock_tracker.completion_tokens = 0
                        mock_tracker_cls.return_value = mock_tracker

                        with tempfile.TemporaryDirectory() as tmpdir:
                            tmp_path = Path(tmpdir)
                            # Don't mock run_sanity_checks — let it run to add the column
                            with patch.object(runner, 'EVALS_DIR', tmp_path):
                                aio.run(runner._run_csv_mode(args, "gemini-3.5-flash"))

                            # Verify output CSV has parametric_suspect column
                            offline_dir = tmp_path / "experiments" / "offline"
                            csvs = list(offline_dir.glob("*.csv"))
                            assert len(csvs) == 1, f"Expected 1 output CSV, got {len(csvs)}"
                            output_df = pd.read_csv(csvs[0], sep=";")
                            assert "parametric_suspect" in output_df.columns, \
                                "Expected parametric_suspect column in output CSV"


# ---------------------------------------------------------------------------
# Bedrock RAGAS Judge: TestAnthropicCostTracker
# ---------------------------------------------------------------------------

class TestAnthropicCostTracker:
    """Tests for JudgeCostTracker with provider="anthropic" — Anthropic Bedrock
    token accumulation via monkey-patched client.messages.create."""

    @pytest.fixture(autouse=True)
    def _import(self):
        from runner import JudgeCostTracker
        self.Tracker = JudgeCostTracker

    @staticmethod
    def _make_anthropic_mock_client():
        """Create a mock Anthropic client with a replaceable messages.create."""
        client = MagicMock()
        client.messages.create = AsyncMock()
        return client

    @staticmethod
    def _make_anthropic_response(input_tokens=1000, output_tokens=500):
        """Create a mock Anthropic messages.create response with usage info."""
        response = MagicMock()
        response.usage.input_tokens = input_tokens
        response.usage.output_tokens = output_tokens
        return response

    # ── RED-1: Anthropic token accumulation from single call ──

    def test_anthropic_token_accumulation_from_single_call(self):
        """provider='anthropic' reads usage.input_tokens / usage.output_tokens."""
        client = self._make_anthropic_mock_client()
        mock_resp = self._make_anthropic_response(input_tokens=1000, output_tokens=500)
        client.messages.create.return_value = mock_resp

        tracker = self.Tracker(client, provider="anthropic",
                               model_label="claude-sonnet-4",
                               prompt_cost_per_1k=0.00300,
                               completion_cost_per_1k=0.01500)
        asyncio.run(client.messages.create(model="claude-sonnet-4", messages=[]))

        assert tracker.prompt_tokens == 1000
        assert tracker.completion_tokens == 500

    # ── RED-2: Multi-call accumulation ──

    def test_anthropic_multi_call_accumulation(self):
        """Three Anthropic calls → tokens accumulate correctly."""
        client = self._make_anthropic_mock_client()
        call_responses = [
            self._make_anthropic_response(input_tokens=100, output_tokens=50),
            self._make_anthropic_response(input_tokens=200, output_tokens=100),
            self._make_anthropic_response(input_tokens=300, output_tokens=150),
        ]
        client.messages.create.side_effect = call_responses

        tracker = self.Tracker(client, provider="anthropic",
                               model_label="claude-sonnet-4",
                               prompt_cost_per_1k=0.00300,
                               completion_cost_per_1k=0.01500)

        async def _run():
            await client.messages.create(model="claude-sonnet-4", messages=[])
            await client.messages.create(model="claude-sonnet-4", messages=[])
            await client.messages.create(model="claude-sonnet-4", messages=[])
        asyncio.run(_run())

        assert tracker.prompt_tokens == 600   # 100+200+300
        assert tracker.completion_tokens == 300  # 50+100+150

    # ── RED-3: Cost calculation accuracy (haiku pricing) ──

    def test_anthropic_cost_calculation_haiku(self):
        """Haiku 4.5: (input/1K)*0.001 + (output/1K)*0.005.
        10,000 input + 2,000 output = (10*0.001) + (2*0.005) = 0.01 + 0.01 = 0.02.
        """
        client = self._make_anthropic_mock_client()
        tracker = self.Tracker(client, provider="anthropic",
                               model_label="claude-haiku-4-5",
                               prompt_cost_per_1k=0.00100,
                               completion_cost_per_1k=0.00500)
        tracker.prompt_tokens = 10000
        tracker.completion_tokens = 2000

        import io
        old_stdout = sys.stdout
        try:
            captured = io.StringIO()
            sys.stdout = captured
            tracker.cost_summary()
            output = captured.getvalue()
        finally:
            sys.stdout = old_stdout

        assert "Prompt cost/1K       : $0.001000" in output
        assert "Completion cost/1K   : $0.005000" in output
        assert "Total judge USD      : $0.020000" in output

    # ── RED-4: Zero-token safety ──

    def test_anthropic_zero_tokens_safe(self):
        """provider='anthropic' with zero tokens → no crash, zero cost."""
        client = self._make_anthropic_mock_client()
        tracker = self.Tracker(client, provider="anthropic",
                               model_label="claude-sonnet-4",
                               prompt_cost_per_1k=0.00300,
                               completion_cost_per_1k=0.01500)
        # No calls made — tokens stay at 0
        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0

        # cost_summary with zero tokens → $0.000000
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            tracker.cost_summary()
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old
        assert "$0.000000" in output

    # ── RED-5: Wraps messages.create, NOT chat.completions.create ──

    def test_anthropic_wraps_messages_create_not_chat_completions(self):
        """provider='anthropic' monkey-patches client.messages.create
        and leaves client.chat.completions untouched."""
        client = self._make_anthropic_mock_client()
        # Use a plain object for chat.completions so hasattr doesn't lie
        class PlainCompletions:
            pass
        client.chat = MagicMock()
        client.chat.completions = PlainCompletions()

        tracker = self.Tracker(client, provider="anthropic",
                               model_label="claude-sonnet-4",
                               prompt_cost_per_1k=0.00300,
                               completion_cost_per_1k=0.01500)

        # messages.create should be patched (tracked)
        assert getattr(client.messages, "_tracked_by_judge_cost", None) is True, \
            "messages.create must be tracked (_tracked_by_judge_cost=True)"
        # chat.completions.create should be UNTOUCHED (plain object, no attr set)
        assert not hasattr(client.chat.completions, "_tracked_by_judge_cost"), \
            "chat.completions.create must NOT be tracked for provider=anthropic"

    # ── RED-6: Response without usage is safe ──

    def test_anthropic_response_without_usage_is_safe(self):
        """Anthropic response missing usage attribute doesn't crash."""
        client = self._make_anthropic_mock_client()

        class BareResponse:
            pass
        client.messages.create.return_value = BareResponse()

        tracker = self.Tracker(client, provider="anthropic",
                               model_label="claude-sonnet-4",
                               prompt_cost_per_1k=0.00300,
                               completion_cost_per_1k=0.01500)
        asyncio.run(client.messages.create(model="claude-sonnet-4", messages=[]))

        assert tracker.prompt_tokens == 0
        assert tracker.completion_tokens == 0

    # ── RED-7: Double-wrap guard ──

    def test_anthropic_double_wrap_guard(self):
        """Wrapping an already-tracked Anthropic client is a no-op."""
        client = self._make_anthropic_mock_client()
        # Set up the original create's return_value BEFORE wrapping
        # (after wrapping, client.messages.create is the tracked function)
        mock_resp = self._make_anthropic_response(input_tokens=500, output_tokens=250)
        client.messages.create.return_value = mock_resp

        tracker1 = self.Tracker(client, provider="anthropic",
                                model_label="claude-sonnet-4",
                                prompt_cost_per_1k=0.00300,
                                completion_cost_per_1k=0.01500)

        # Second wrap on same client should be a no-op (guard fires)
        tracker2 = self.Tracker(client, provider="anthropic",
                                model_label="claude-haiku-4-5",
                                prompt_cost_per_1k=0.00100,
                                completion_cost_per_1k=0.00500)

        # Call through the tracked client — response goes to original create
        asyncio.run(client.messages.create(model="claude-sonnet-4", messages=[]))

        # Tokens should have accumulated on tracker1 (the first wrapper)
        assert tracker1.prompt_tokens == 500
        assert tracker1.completion_tokens == 250
        # tracker2 should have 0 tokens (it was a no-op)
        assert tracker2.prompt_tokens == 0
        assert tracker2.completion_tokens == 0


# ---------------------------------------------------------------------------
# Bedrock RAGAS Judge: TestAnthropicJudgeClient
# ---------------------------------------------------------------------------

class TestAnthropicJudgeClient:
    """Tests for _build_judge_client() Bedrock branch — model resolution,
    credential validation, and client construction."""

    @pytest.fixture(autouse=True)
    def _import_func(self):
        import runner
        self._build = runner._build_judge_client
        self._runner = runner

    # ── RED-8: Valid model lookup ──

    def test_valid_claude_model_resolves_to_profile_id(self):
        """claude-sonnet-4 → us.anthropic.claude-sonnet-4-20250514-v1:0."""
        with patch.dict("os.environ", {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_REGION": "us-east-1",
        }, clear=True):
            with patch("anthropic.AsyncAnthropicBedrock") as mock_bedrock:
                mock_client = MagicMock()
                mock_bedrock.return_value = mock_client

                with patch.object(self._runner, "JudgeCostTracker") as mock_tracker_cls:
                    mock_tracker = MagicMock()
                    mock_tracker_cls.return_value = mock_tracker

                    with patch.object(self._runner, "llm_factory") as mock_llm:
                        mock_llm.return_value = MagicMock()

                        client, judge_llm, cost_tracker, model_label = \
                            self._build("claude-sonnet-4")

        # Verify correct inference profile ID was used
        mock_bedrock.assert_called_once_with(
            aws_access_key="test-key",
            aws_secret_key="test-secret",
            aws_region="us-east-1",
        )
        # Verify llm_factory called with inference profile ID
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args
        assert call_kwargs[0][0] == "us.anthropic.claude-sonnet-4-20250514-v1:0", \
            f"Expected inference profile ID, got {call_kwargs[0][0]}"
        assert call_kwargs[1].get("provider") == "anthropic"
        assert call_kwargs[1].get("max_tokens") == 16384
        # model_label uses friendly name
        assert model_label == "claude-sonnet-4"

        # Verify cost tracker wrapped with provider="anthropic"
        mock_tracker_cls.assert_called_once()
        tracker_kwargs = mock_tracker_cls.call_args[1]
        assert tracker_kwargs.get("provider") == "anthropic"
        assert tracker_kwargs.get("model_label") == "claude-sonnet-4"

    # ── RED-9: Unknown model exits ──

    def test_unknown_claude_model_exits_with_valid_list(self):
        """claude-opus-4-nonexistent → SystemExit(1) listing valid models."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                self._build("claude-opus-4-nonexistent")
        assert exc_info.value.code == 1

    # ── RED-10: Missing AWS_ACCESS_KEY_ID exits ──

    def test_missing_aws_access_key_id_exits(self):
        """Missing AWS_ACCESS_KEY_ID → SystemExit(1) with descriptive message."""
        with patch.dict("os.environ", {
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_REGION": "us-east-1",
        }, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                self._build("claude-sonnet-4")
        assert exc_info.value.code == 1

    # ── RED-11: Missing AWS_SECRET_ACCESS_KEY exits ──

    def test_missing_aws_secret_access_key_exits(self):
        """Missing AWS_SECRET_ACCESS_KEY → SystemExit(1)."""
        with patch.dict("os.environ", {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_REGION": "us-east-1",
        }, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                self._build("claude-sonnet-4")
        assert exc_info.value.code == 1

    # ── RED-12: Missing AWS_REGION exits ──

    def test_missing_aws_region_exits(self):
        """Missing AWS_REGION → SystemExit(1)."""
        with patch.dict("os.environ", {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
        }, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                self._build("claude-sonnet-4")
        assert exc_info.value.code == 1

    # ── RED-13: All Claude aliases resolve correctly ──

    @pytest.mark.parametrize("model_name,expected_profile", [
        ("claude-sonnet-4", "us.anthropic.claude-sonnet-4-20250514-v1:0"),
        ("claude-haiku-4-5", "us.anthropic.claude-haiku-4-5-20251001-v1:0"),
        ("claude-sonnet-4-5", "us.anthropic.claude-sonnet-4-5-20251001-v1:0"),
        ("claude-sonnet-4-6", "us.anthropic.claude-sonnet-4-6"),
    ])
    def test_all_claude_models_resolve(self, model_name, expected_profile):
        """All 4 Claude model aliases resolve to correct inference profile IDs."""
        with patch.dict("os.environ", {
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "AWS_REGION": "us-east-1",
        }, clear=True):
            with patch("anthropic.AsyncAnthropicBedrock") as mock_bedrock:
                mock_client = MagicMock()
                mock_bedrock.return_value = mock_client

                with patch.object(self._runner, "JudgeCostTracker") as mock_tracker_cls:
                    mock_tracker = MagicMock()
                    mock_tracker_cls.return_value = mock_tracker

                    with patch.object(self._runner, "llm_factory") as mock_llm:
                        mock_llm.return_value = MagicMock()

                        client, judge_llm, cost_tracker, model_label = \
                            self._build(model_name)

            # Verify the llm_factory receives the inference profile ID
            call_kwargs = mock_llm.call_args
            assert call_kwargs[0][0] == expected_profile, \
                f"Expected {expected_profile}, got {call_kwargs[0][0]}"
            # model_label is the friendly name
            assert model_label == model_name


# ---------------------------------------------------------------------------
# Phase 2 (A2) — context dedupe + deterministic @5 metrics
# ---------------------------------------------------------------------------

# Shared fixtures: distinct contexts whose token overlap with the grading notes
# is zero, plus one gold-shaped table context that contains the grading notes
# and a whole cell value ("1867").
_CTX_ALPHA = "Unrelated alpha passage about geology and rocks"
_CTX_BETA = "Unrelated beta passage about music and tours"
_CTX_GOLD = (
    "Name | Year\n"
    "Name: Kathleen Williams | Year: 1867\n"
    "Kathleen Williams was born in Portland Oregon"
)
_GRADING_NOTES = "Kathleen Williams was born in Portland Oregon"


class TestContextDedupe:
    """Deduplicate contexts before any metric — first-occurrence order, per question."""

    @pytest.fixture(autouse=True)
    def _import_functions(self):
        from runner import _dedupe_first_occurrence, _process_question_result
        self._dedupe_first_occurrence = _dedupe_first_occurrence
        self._process_question_result = _process_question_result

    # ── Pure function: first-occurrence order ──

    def test_first_occurrence_preserved(self):
        """[A, B, A, C] → [A, B, C] keeping first-occurrence order."""
        contexts = [_CTX_ALPHA, _CTX_BETA, _CTX_ALPHA, _CTX_GOLD]
        result = self._dedupe_first_occurrence(contexts)
        assert result == [_CTX_ALPHA, _CTX_BETA, _CTX_GOLD]

    def test_no_duplicates_returns_same_order(self):
        """Distinct contexts are returned unchanged, in order."""
        contexts = ["one", "two", "three"]
        assert self._dedupe_first_occurrence(contexts) == ["one", "two", "three"]

    def test_all_duplicates_collapse_to_first(self):
        """Repeated copies collapse to a single first occurrence."""
        assert self._dedupe_first_occurrence(["dup", "dup", "dup"]) == ["dup"]

    # ── Pipeline: deduped list feeds every metric consumer ──

    def test_deduped_list_feeds_all_metrics(self):
        """RAGAS sample, relevance position, and citation pairing all see the deduped list."""
        duplicate_contexts = [_CTX_ALPHA, _CTX_BETA, _CTX_ALPHA, _CTX_GOLD]
        mock_tero = _make_mock_tero(
            answer_text="Kathleen Williams was born in Portland Oregon.",
            contexts=list(duplicate_contexts),
            citations=["chunk_3"],
        )
        row = _make_test_row(question="Where was Kathleen Williams born?", grading_notes=_GRADING_NOTES)
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        result = asyncio.run(
            self._process_question_result(
                mock_tero, row,
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

        # Row column carries the deduped list (CSV persistence)
        assert result["retrieved_contexts"] == " | ".join([_CTX_ALPHA, _CTX_BETA, _CTX_GOLD])

        # RAGAS context metrics received the deduped list
        sample = mock_recall.single_turn_ascore.call_args[0][0]
        assert sample.retrieved_contexts == [_CTX_ALPHA, _CTX_BETA, _CTX_GOLD]

        # Relevance-position heuristic ran on the deduped list: gold is 3rd unique,
        # 4th in the raw accumulated list.
        assert result["relevant_chunk_position"] == 3

        # Citation pairing indexes the deduped list: chunk_3 → 3rd unique context
        assert mock_cite_faith.ascore.call_args.kwargs["cited_chunk"] == _CTX_GOLD

    def test_dedup_scope_is_per_question(self):
        """The same chunk retrieved by two questions stays in both lists (no cross-thread dedup)."""
        shared = "Shared chunk text retrieved by both questions"
        mock_tero = _make_mock_tero(contexts=[shared, shared])
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()

        def _run(row):
            return asyncio.run(
                self._process_question_result(
                    mock_tero, row,
                    judge_llm=None,
                    context_recall=mock_recall,
                    context_precision=mock_precision,
                    faithfulness=mock_faith,
                    correctness=mock_correctness,
                    citation_faithfulness=mock_cite_faith,
                )
            )

        first = _run(_make_test_row(question="First question?"))
        second = _run(_make_test_row(question="Second question?"))

        assert first["retrieved_contexts"] == shared
        assert second["retrieved_contexts"] == shared


class TestAnnotateGold:
    """Gold annotation: resolve gold content and compose the out-of-corpus flag."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _annotate_gold
        self._annotate_gold = _annotate_gold

    def _corpus(self):
        return ["doc0", "doc1", "doc2", "doc3", "doc4"]

    def test_resolves_gold_content_for_in_corpus_row(self):
        """gold_doc_id inside the corpus resolves gold_content and stays in the aggregate."""
        rows = [{
            "question": "q",
            "gold_values": ["v"],
            "gold_doc_id": 2,
            "gold_out_of_corpus": False,
        }]
        annotated = self._annotate_gold(rows, self._corpus(), None)

        assert annotated[0]["gold_content"] == "doc2"
        assert annotated[0]["gold_out_of_corpus"] is False
        # Pure: the input row is not mutated
        assert "gold_content" not in rows[0]

    def test_skipped_table_row_is_out_of_corpus(self):
        """Loader flag / gold_doc_id=None → out-of-corpus, no gold content."""
        rows = [{
            "question": "q",
            "gold_values": ["v"],
            "gold_doc_id": None,
            "gold_out_of_corpus": True,
        }]
        annotated = self._annotate_gold(rows, self._corpus(), None)

        assert annotated[0]["gold_content"] is None
        assert annotated[0]["gold_out_of_corpus"] is True

    def test_gold_beyond_max_docs_is_out_of_corpus(self):
        """An indexed prefix shorter than the corpus pushes higher gold ids out of corpus."""
        rows = [
            {"question": "in", "gold_doc_id": 2, "gold_out_of_corpus": False},
            {"question": "out", "gold_doc_id": 4, "gold_out_of_corpus": False},
        ]
        annotated = self._annotate_gold(rows, self._corpus(), 3)

        assert annotated[0]["gold_content"] == "doc2"
        assert annotated[0]["gold_out_of_corpus"] is False
        assert annotated[1]["gold_content"] is None
        assert annotated[1]["gold_out_of_corpus"] is True

    def test_rows_without_gold_linkage_are_unchanged(self):
        """Rows from datasets without any gold linkage pass through untouched."""
        rows = [{"question": "q", "grading_notes": "g"}]
        annotated = self._annotate_gold(rows, self._corpus(), None)

        assert annotated == [{"question": "q", "grading_notes": "g"}]

    def test_multi_gold_fully_covered_row(self):
        """Spec: two gold ids inside the prefix → both available, not out of corpus."""
        rows = [{
            "question": "q",
            "gold_doc_ids": [1, 3],
            "gold_sentences": ["s1", "s3"],
            "gold_sentence_doc_ids": [1, 3],
            "no_gold_labels": False,
        }]
        annotated = self._annotate_gold(rows, self._corpus(), None)

        assert annotated[0]["gold_doc_ids_in_prefix"] == [1, 3]
        assert annotated[0]["gold_contents_in_prefix"] == ["doc1", "doc3"]
        assert annotated[0]["gold_sentences_in_prefix"] == ["s1", "s3"]
        assert annotated[0]["gold_out_of_corpus"] is False
        assert annotated[0]["no_gold_labels"] is False
        # Legacy key stays absent for multi-gold rows (FeTaQA CSV shape preserved)
        assert "gold_content" not in annotated[0]

    def test_multi_gold_partially_covered_row(self):
        """Spec: only the in-prefix subset is scored; out-of-prefix gold is not a miss."""
        rows = [{
            "question": "q",
            "gold_doc_ids": [1, 3, 9],
            "gold_sentences": ["s1", "s3", "s9"],
            "gold_sentence_doc_ids": [1, 3, 9],
            "no_gold_labels": False,
        }]
        annotated = self._annotate_gold(rows, self._corpus(), None)

        assert annotated[0]["gold_doc_ids_in_prefix"] == [1, 3]
        assert annotated[0]["gold_contents_in_prefix"] == ["doc1", "doc3"]
        assert annotated[0]["gold_sentences_in_prefix"] == ["s1", "s3"]
        assert annotated[0]["gold_out_of_corpus"] is False

    def test_multi_gold_none_inside_the_prefix_is_out_of_corpus(self):
        """Spec: labels exist but no gold id is inside the prefix → out of corpus."""
        rows = [{
            "question": "q",
            "gold_doc_ids": [7, 9],
            "gold_sentences": ["s7", "s9"],
            "gold_sentence_doc_ids": [7, 9],
            "no_gold_labels": False,
        }]
        annotated = self._annotate_gold(rows, self._corpus(), 3)

        assert annotated[0]["gold_out_of_corpus"] is True
        assert annotated[0]["gold_doc_ids_in_prefix"] == []
        assert annotated[0]["gold_contents_in_prefix"] == []
        assert annotated[0]["gold_sentences_in_prefix"] == []

    def test_no_label_row_has_no_scoring_target(self):
        """Spec: no-label rows carry no gold target and are never out of corpus."""
        rows = [{
            "question": "q",
            "gold_doc_ids": [],
            "gold_sentences": [],
            "gold_sentence_doc_ids": [],
            "no_gold_labels": True,
        }]
        annotated = self._annotate_gold(rows, self._corpus(), None)

        assert annotated[0]["no_gold_labels"] is True
        assert annotated[0]["gold_out_of_corpus"] is False
        assert annotated[0]["gold_doc_ids_in_prefix"] == []
        assert annotated[0]["gold_contents_in_prefix"] == []

    def test_legacy_rows_gain_additive_keys_without_changing_the_legacy_ones(self):
        """Spec: FeTaQA rows keep gold_content and the flag rules exactly."""
        rows = [{"question": "q", "gold_values": ["v"], "gold_doc_id": 2, "gold_out_of_corpus": False}]
        annotated = self._annotate_gold(rows, self._corpus(), None)

        assert annotated[0]["gold_content"] == "doc2"
        assert annotated[0]["gold_out_of_corpus"] is False
        assert annotated[0]["gold_doc_ids_in_prefix"] == [2]
        assert annotated[0]["gold_contents_in_prefix"] == ["doc2"]
        assert annotated[0]["gold_sentences_in_prefix"] == []
        assert annotated[0]["no_gold_labels"] is False

    def test_multi_gold_input_rows_are_not_mutated(self):
        """Annotation is pure: the caller's rows never gain annotation keys."""
        rows = [{
            "question": "q",
            "gold_doc_ids": [1],
            "gold_sentences": ["s1"],
            "gold_sentence_doc_ids": [1],
            "no_gold_labels": False,
        }]
        self._annotate_gold(rows, self._corpus(), None)

        assert set(rows[0].keys()) == {
            "question", "gold_doc_ids", "gold_sentences", "gold_sentence_doc_ids", "no_gold_labels",
        }


class TestTableRecallAt5:
    """Deterministic table_recall@5 — exact content identity over first five unique contexts."""

    @pytest.fixture(autouse=True)
    def _import_functions(self):
        from runner import _dedupe_first_occurrence, _table_recall_5
        self._dedupe_first_occurrence = _dedupe_first_occurrence
        self._table_recall_5 = _table_recall_5

    def test_gold_third_unique_context_scores_1(self):
        """Gold as the third unique context → 1."""
        contexts = [_CTX_ALPHA, _CTX_BETA, _CTX_GOLD]
        assert self._table_recall_5(contexts, _CTX_GOLD, False) == 1.0

    def test_duplicates_do_not_shift_k(self):
        """Spec scenario: [A, A, B, C, D, E, gold] → first five unique exclude gold → 0."""
        contexts = ["a", "a", "b", "c", "d", "e", _CTX_GOLD]
        deduped = self._dedupe_first_occurrence(contexts)
        assert self._table_recall_5(deduped, _CTX_GOLD, False) == 0.0

    def test_dedupe_promotes_gold_into_top_five(self):
        """[A, A, B, C, D, gold] → deduped gold is the fifth unique → 1."""
        contexts = ["a", "a", "b", "c", "d", _CTX_GOLD]
        deduped = self._dedupe_first_occurrence(contexts)
        assert self._table_recall_5(deduped, _CTX_GOLD, False) == 1.0

    def test_line_ending_and_whitespace_normalized_match(self):
        """Identity is exact content after line-ending normalization + strip."""
        gold = "\r\nName | Year\r\nName: Kathleen Williams | Year: 1867\r\n  "
        contexts = ["Name | Year\nName: Kathleen Williams | Year: 1867"]
        assert self._table_recall_5(contexts, gold, False) == 1.0

    def test_gold_beyond_top_five_scores_0(self):
        """Gold present but beyond the first five unique contexts → 0 (not None)."""
        contexts = ["1", "2", "3", "4", "5", _CTX_GOLD]
        assert self._table_recall_5(contexts, _CTX_GOLD, False) == 0.0

    def test_out_of_corpus_excluded(self):
        """Out-of-corpus questions are excluded from the aggregate, never scored 0."""
        assert self._table_recall_5([_CTX_GOLD], _CTX_GOLD, True) is None

    def test_missing_gold_content_not_applicable(self):
        """No gold linkage (legacy CSV row) → not applicable."""
        assert self._table_recall_5([_CTX_GOLD], None, False) is None


class TestCellRecallAt5:
    """Deterministic cell_recall@5 — whole-cell equality, degenerate values filtered."""

    @pytest.fixture(autouse=True)
    def _import_functions(self):
        from runner import _dedupe_first_occurrence, _cell_recall_5
        self._dedupe_first_occurrence = _dedupe_first_occurrence
        self._cell_recall_5 = _cell_recall_5

    def test_partial_recall_two_of_three(self):
        """Three non-degenerate gold values, two matched whole cells → 2/3."""
        context = "Name | Year | Title\nName: Kathleen Williams | Year: 1867 | Title: Other"
        gold_values = ["Kathleen Williams", "1867", "Hairshirt"]
        result = self._cell_recall_5([context], gold_values, False)
        assert result == pytest.approx(2 / 3)

    def test_substring_rejected(self):
        """Gold `12` vs context cell `123` → not a match."""
        context = "Name | Score\nName: Alice | Score: 123"
        assert self._cell_recall_5([context], ["12"], False) == 0.0

    def test_degenerate_values_filtered(self):
        """`-,` empty and whitespace-only values leave the numerator and denominator."""
        context = "Name | Year\nName: X | Year: 1867"
        result = self._cell_recall_5([context], ["-", "", "1867"], False)
        assert result == 1.0

    def test_whitespace_only_value_is_degenerate(self):
        """Whitespace-only gold values are degenerate → 1/1 when only 1867 remains."""
        context = "Name | Year\nName: X | Year: 1867"
        assert self._cell_recall_5([context], ["   ", "1867"], False) == 1.0

    def test_all_degenerate_is_not_applicable(self):
        """All gold values degenerate → not-applicable, never 0."""
        assert self._cell_recall_5(["Year: 1867"], ["-", "", "  "], False) is None

    def test_match_limited_to_first_five_deduped_contexts(self):
        """A whole cell only present in the sixth unique context → 0."""
        contexts = ["c1", "c2", "c3", "c4", "c5", "Year: 1867"]
        assert self._cell_recall_5(contexts, ["1867"], False) == 0.0

    def test_dedupe_promotes_cell_into_top_five(self):
        """A duplicate in the first five does not push the gold cell out of scope."""
        contexts = ["c1", "c1", "c2", "c3", "c4", "Year: 1867"]
        deduped = self._dedupe_first_occurrence(contexts)
        assert self._cell_recall_5(deduped, ["1867"], False) == 1.0

    def test_out_of_corpus_is_not_applicable(self):
        """Out-of-corpus → excluded from aggregates, never 0."""
        assert self._cell_recall_5(["Year: 1867"], ["1867"], True) is None

    def test_no_gold_values_is_not_applicable(self):
        """Rows without gold values → not applicable."""
        assert self._cell_recall_5(["Year: 1867"], [], False) is None


class TestExtractCells:
    """_extract_cells: Opción B rows keep cells whole; legacy `h: v` → `v`."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _extract_cells
        self._extract_cells = _extract_cells

    def test_legacy_rows_normalize_header_value_pairs(self):
        """Legacy format: header cells kept, `header: value` rows reduced to values."""
        context = "Name | Year\nName: Kathleen Williams | Year: 1867"
        assert self._extract_cells(context) == ["Name", "Year", "Kathleen Williams", "1867"]

    def test_legacy_colon_values_are_normalized(self):
        """Legacy rows keep the `h: v` → `v` normalization."""
        context = "Name | Note\nName: Alice | Note: lead role"
        assert self._extract_cells(context) == ["Name", "Note", "Alice", "lead role"]

    def test_opcion_b_metadata_lines_are_not_cells(self):
        """`Title:`/`Section:` metadata lines never become cells."""
        context = (
            "Title: Filmography\n"
            "Section: Acting credits\n"
            "\n"
            "[feta_0042_0] | Year | Title |\n"
            "[feta_0042_1] | 1998 | Hairshirt |"
        )
        assert self._extract_cells(context) == ["Year", "Title", "1998", "Hairshirt"]

    def test_opcion_b_rows_keep_cells_whole_including_colons(self):
        """Opción B cells are values: the legacy `h: v` split must not mangle them."""
        context = (
            "Title: Filmography\n"
            "Section: Acting credits\n"
            "\n"
            "[feta_0042_0] | Year | Time | Title |\n"
            "[feta_0042_1] | 1998 | 12:30 | Hairshirt: The Musical |"
        )
        assert self._extract_cells(context) == [
            "Year", "Time", "Title", "1998", "12:30", "Hairshirt: The Musical",
        ]

    def test_opcion_b_row_with_only_empty_cells_yields_nothing(self):
        """A UID'd row whose cells are empty or whitespace-only contributes no cells."""
        context = (
            "[feta_7_0] | |\n"
            "[feta_7_1] | Notes |\n"
            "[feta_7_2] |   | x |"
        )
        assert self._extract_cells(context) == ["Notes", "x"]


class TestOpcionBSerializerRoundTrip:
    """The C serializer's documents feed `_extract_cells`/`_cell_recall_5` unchanged."""

    @pytest.fixture(autouse=True)
    def _import_functions(self):
        from rag_datasets import _serialize_fetaqa_table
        from runner import _cell_recall_5, _extract_cells
        self._serialize_fetaqa_table = _serialize_fetaqa_table
        self._cell_recall_5 = _cell_recall_5
        self._extract_cells = _extract_cells

    def test_real_serializer_output_round_trips_cells(self):
        """Cells extracted from a real serialized document are its values."""
        doc = self._serialize_fetaqa_table(
            [["Year", "Title"], ["1998", "Hairshirt: The Musical"]],
            feta_id=42,
            page_title="Filmography",
            section_title="Acting credits",
        )
        assert self._extract_cells(doc) == ["Year", "Title", "1998", "Hairshirt: The Musical"]

    def test_colon_gold_value_matches_its_cell(self):
        """A gold value containing a colon is matched whole (complete cell)."""
        doc = self._serialize_fetaqa_table(
            [["Year", "Title"], ["2017", "Groundhog Day: The Musical"]],
            feta_id=2275,
        )
        assert self._cell_recall_5([doc], ["Groundhog Day: The Musical"], False) == 1.0
        assert self._cell_recall_5([doc], ["Groundhog Day"], False) == 0.0


class TestMetricRowPersistence:
    """The @5 metrics persist as per-question row fields (CSV columns), additive and readable."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _process_question_result
        self._process_question_result = _process_question_result

    def _run(self, row, contexts, citations=None):
        mock_tero = _make_mock_tero(contexts=list(contexts), citations=citations)
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()
        return asyncio.run(
            self._process_question_result(
                mock_tero, row,
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )

    def test_row_carries_at5_metric_columns(self):
        """Gold-linked question in a FeTaQA run → both @5 metrics persisted per question."""
        row = _make_test_row(
            question="Where was Kathleen Williams born?",
            grading_notes=_GRADING_NOTES,
            gold_values=["1867"],
            gold_content=_CTX_GOLD,
            gold_out_of_corpus=False,
        )
        result = self._run(row, [_CTX_ALPHA, _CTX_BETA, _CTX_ALPHA, _CTX_GOLD])

        assert result["table_recall_5"] == 1.0
        assert result["cell_recall_5"] == 1.0

    def test_row_without_gold_linkage_reports_not_applicable(self):
        """Legacy CSV rows without gold columns remain readable: columns present, value N/A."""
        row = _make_test_row(question="Legacy row?", grading_notes=_GRADING_NOTES)
        result = self._run(row, [_CTX_ALPHA, _CTX_GOLD])

        assert "table_recall_5" in result
        assert "cell_recall_5" in result
        assert result["table_recall_5"] is None
        assert result["cell_recall_5"] is None

    def test_error_row_carries_at5_columns_as_none(self):
        """recursionLimitExceeded rows expose the @5 columns as None (stable CSV schema)."""
        row = _make_test_row(question="Backend error?")
        result = self._run(row, [_CTX_ALPHA], citations=[])

        # Sanity: this really is the backend-error path (normal answers never start
        # with the marker).
        mock_tero = _make_mock_tero(answer_text="recursionLimitExceeded: depth")
        mock_recall, mock_precision, mock_faith, mock_correctness, mock_cite_faith = _make_mock_metrics()
        error_result = asyncio.run(
            self._process_question_result(
                mock_tero, row,
                judge_llm=None,
                context_recall=mock_recall,
                context_precision=mock_precision,
                faithfulness=mock_faith,
                correctness=mock_correctness,
                citation_faithfulness=mock_cite_faith,
            )
        )
        assert error_result["error"] == "recursionLimitExceeded"
        assert error_result["table_recall_5"] is None
        assert error_result["cell_recall_5"] is None

        # The normal-path row carries the keys too (non-error control)
        assert "table_recall_5" in result
        assert "cell_recall_5" in result


class TestAnalysisMetricRows:
    """New-metric visibility: summary/delta rows, existing reporting unchanged."""

    def test_stats_and_summary_include_new_metrics(self, capsys):
        """Both @5 metrics appear as summary rows without disturbing existing rows."""
        import analysis

        df = pd.DataFrame({
            "grounded_correctness": [0.1, 0.2],
            "context_recall": [0.0, 0.5],
            "table_recall_5": [1.0, 0.0],
            "cell_recall_5": [2 / 3, 1 / 3],
        })
        stats = analysis._stats_from_df(df)

        assert stats["table_recall_5"]["mean"] == 0.5
        assert stats["cell_recall_5"]["mean"] == pytest.approx(0.5)

        analysis.print_summary(stats)
        out = capsys.readouterr().out

        assert "table_recall_5" in out
        assert "cell_recall_5" in out
        # Existing rows are unchanged: grounded_correctness still leads, before
        # context_recall, and before the appended @5 rows.
        assert out.index("grounded_correctness") < out.index("context_recall")
        assert out.index("context_recall") < out.index("table_recall_5")

    def test_comparison_reports_new_metric_deltas(self, capsys):
        """Baseline comparison prints delta rows for both @5 metrics."""
        import analysis

        baseline = {
            "table_recall_5": {"mean": 0.1},
            "cell_recall_5": {"mean": 0.2},
        }
        current = {
            "table_recall_5": {"mean": 0.6},
            "cell_recall_5": {"mean": 0.25},
        }
        analysis.print_comparison(baseline, current)
        out = capsys.readouterr().out

        assert "table_recall_5" in out
        assert "+0.5000" in out
        assert "cell_recall_5" in out
        assert "+0.0500" in out

    def test_distribution_unchanged_when_at5_columns_present(self, capsys):
        """No new distribution buckets: the breakdown stays context_recall-only."""
        import analysis

        df = pd.DataFrame({"context_recall": [0.0, 0.8], "table_recall_5": [1.0, 0.0]})
        analysis.print_distribution(df)
        out = capsys.readouterr().out

        assert "CONTEXT RECALL DISTRIBUTION" in out
        assert "table_recall_5" not in out


# ---------------------------------------------------------------------------
# Phase 4 — A4 comparability protocol (D7)
# ---------------------------------------------------------------------------

def _comparability_inputs(**overrides):
    """Baseline/current comparability dict copied for A4 tests."""
    metadata = {"agent_id": 9, "seed": 14, "judge_model": "gemini-3.5-flash", "top_k": 5}
    metadata.update(overrides)
    return metadata


class TestComparabilityMetadata:
    """Baseline metadata records the run's comparability inputs: agent, seed, judge, top_k=5."""

    def test_save_baseline_records_comparability_metadata(self, tmp_path, monkeypatch):
        """Spec: with --update-baseline the agent id is recorded alongside seed/judge/top_k."""
        import json
        import runner

        monkeypatch.setattr(runner, "BASELINE_DIR", tmp_path)
        stats = {"grounded_correctness": {"mean": 0.5}}

        runner._save_baseline(
            "gpt-5", "fetaqa", stats, 20,
            agent_id=10, seed=14, judge_model="gemini-3.5-flash",
        )

        saved = json.loads((tmp_path / "gpt-5" / "fetaqa.json").read_text())
        assert saved["agent_id"] == 10
        assert saved["seed"] == 14
        assert saved["judge_model"] == "gemini-3.5-flash"
        assert saved["top_k"] == 5
        # Legacy keys stay intact for existing baseline consumers (analysis.py CLI)
        assert saved["model_id"] == "gpt-5"
        assert saved["dataset"] == "fetaqa"
        assert saved["n"] == 20
        assert saved["stats"] == stats
        assert "generated_at" in saved

    def test_baseline_metadata_round_trips_via_load(self, tmp_path, monkeypatch):
        """A later --compare run loads the recorded comparability inputs back."""
        import runner

        monkeypatch.setattr(runner, "BASELINE_DIR", tmp_path)
        runner._save_baseline(
            "gpt-5", "fetaqa", {"grounded_correctness": {"mean": 0.5}}, 20,
            agent_id=11, seed=14, judge_model="gemini-3.5-flash",
        )

        baseline = runner._load_baseline("gpt-5", "fetaqa")
        assert baseline["agent_id"] == 11
        assert baseline["seed"] == 14
        assert baseline["judge_model"] == "gemini-3.5-flash"
        assert baseline["top_k"] == 5

    def test_comparability_metadata_pins_protocol_values(self):
        """Current-run metadata carries top_k=5 — the protocol constant for comparisons."""
        from runner import _comparability_metadata

        metadata = _comparability_metadata(agent_id=10, seed=14, judge_model="gemini-3.5-flash")
        assert metadata == {
            "agent_id": 10,
            "seed": 14,
            "judge_model": "gemini-3.5-flash",
            "top_k": 5,
        }


class TestBaselineDeviations:
    """Comparison inputs (agent/seed/judge/top_k) are checked — never silently assumed."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _baseline_deviations
        self._baseline_deviations = _baseline_deviations

    def test_matching_metadata_reports_no_deviations(self):
        assert self._baseline_deviations(_comparability_inputs(), _comparability_inputs()) == []

    def test_different_agent_id_is_reported(self):
        """Spec: a later comparison against a different agent id surfaces the mismatch."""
        deviations = self._baseline_deviations(
            _comparability_inputs(agent_id=9), _comparability_inputs(agent_id=10)
        )
        assert len(deviations) == 1
        assert "agent_id" in deviations[0]
        assert "9" in deviations[0] and "10" in deviations[0]

    def test_different_seed_is_reported(self):
        deviations = self._baseline_deviations(
            _comparability_inputs(seed=42), _comparability_inputs(seed=14)
        )
        assert len(deviations) == 1
        assert "seed" in deviations[0]

    def test_different_judge_model_is_reported(self):
        deviations = self._baseline_deviations(
            _comparability_inputs(judge_model="gpt-5"),
            _comparability_inputs(judge_model="gemini-3.5-flash"),
        )
        assert len(deviations) == 1
        assert "judge_model" in deviations[0]

    def test_different_top_k_is_reported(self):
        deviations = self._baseline_deviations(
            _comparability_inputs(top_k=10), _comparability_inputs(top_k=5)
        )
        assert len(deviations) == 1
        assert "top_k" in deviations[0]

    def test_every_deviation_is_reported(self):
        deviations = self._baseline_deviations(
            _comparability_inputs(agent_id=9, seed=42, judge_model="gpt-5", top_k=10),
            _comparability_inputs(agent_id=10, seed=14, judge_model="gemini-3.5-flash", top_k=5),
        )
        assert len(deviations) == 4
        assert {deviation.split(":")[0] for deviation in deviations} == {
            "agent_id", "seed", "judge_model", "top_k",
        }

    def test_legacy_baseline_without_metadata_is_not_flagged(self):
        """Baselines written before D7 carry no comparability keys — nothing to compare."""
        legacy = {"model_id": "gpt-5", "dataset": "fetaqa", "n": 20, "stats": {}}
        assert self._baseline_deviations(legacy, _comparability_inputs()) == []


class TestComparabilityReport:
    """The --compare guard prints deviations instead of comparing silently."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _print_comparability_report
        self._print_comparability_report = _print_comparability_report

    def test_deviations_are_printed_and_returned(self, capsys):
        deviations = self._print_comparability_report(
            _comparability_inputs(agent_id=9), _comparability_inputs(agent_id=10)
        )
        out = capsys.readouterr().out
        assert len(deviations) == 1
        assert "agent_id" in out
        assert "9" in out and "10" in out

    def test_agreement_is_printed_when_inputs_match(self, capsys):
        deviations = self._print_comparability_report(
            _comparability_inputs(), _comparability_inputs()
        )
        out = capsys.readouterr().out
        assert deviations == []
        assert "Comparability" in out
        assert "deviation" not in out.lower()


class TestEvalComparabilityCli:
    """The real eval parser exposes --max-docs and keeps the seed-14 protocol default."""

    @pytest.fixture(autouse=True)
    def _parser(self):
        import runner
        self._parser = runner.build_parser()

    def _eval_args(self, argv=None):
        return self._parser.parse_args(
            ["eval", "--dataset", "fetaqa", "--models", "gpt-5", *(argv or [])]
        )

    def test_eval_accepts_max_docs(self):
        """--max-docs mirrors the indexed prefix that gold annotation should assume."""
        assert self._eval_args(["--max-docs", "500"]).max_docs == 500

    def test_eval_max_docs_defaults_to_none(self):
        assert self._eval_args().max_docs is None

    def test_eval_seed_defaults_to_protocol_14(self):
        """Comparison runs pin seed 14 (fixture loaders use 42; deviations guard the rest)."""
        assert self._eval_args().seed == 14

    def test_eval_accepts_explicit_seed_14(self):
        assert self._eval_args(["--seed", "14"]).seed == 14

    def test_build_parser_still_exposes_index_max_docs(self):
        """build_parser() extraction keeps the index subcommand intact."""
        args = self._parser.parse_args(["index", "--dataset", "fetaqa", "--max-docs", "100"])
        assert args.max_docs == 100


class TestEvalComparabilityWiring:
    """do_eval live path wires --max-docs, baseline metadata, and the --compare guard."""

    @staticmethod
    def _make_live_args(**overrides):
        args = argparse.Namespace(
            dataset="fetaqa", agent_id=9, max_questions=1, models="gpt-5",
            bearer_token="test-token", base_url="http://localhost:8000",
            from_csv=None, update_baseline=False, compare=False,
            n=1, seed=14, only=None, judge_model="gemini-3.5-flash",
            concurrency=5, max_docs=None,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    class _FakeExperimentResult:
        def save(self):
            pass

    @staticmethod
    def _fake_experiment_decorator():
        def decorator(fn):
            async def arun(dataset):
                return TestEvalComparabilityWiring._FakeExperimentResult()
            fn.arun = arun
            return fn
        return decorator

    def _run_live_eval(self, args, tmp_path):
        """Run do_eval() with the network/LLM seams patched (house pattern from
        test_do_eval_live_creates_judge_cost_tracker). Returns the annotation spy record."""
        import contextlib
        import runner

        mock_tero = AsyncMock()
        mock_tero.set_agent_model = AsyncMock()
        recorded = {}

        def annotate_spy(rows, corpus, max_docs=None):
            recorded["max_docs"] = max_docs
            return rows

        stack = contextlib.ExitStack()
        stack.enter_context(patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}))
        stack.enter_context(patch("tero_client.TeroClient", return_value=mock_tero))
        stack.enter_context(patch.object(runner.ds_module, "load_one", return_value=(
            [{"question": "Q?", "grading_notes": "notes"}], ["doc"],
        )))
        stack.enter_context(patch.object(runner, "_annotate_gold", annotate_spy))
        stack.enter_context(patch.object(runner, "AsyncOpenAI"))
        stack.enter_context(patch.object(runner, "JudgeCostTracker"))
        stack.enter_context(patch.object(runner, "llm_factory"))
        stack.enter_context(patch.object(runner, "_build_metrics", return_value=(
            MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
        )))
        stack.enter_context(patch.object(runner.analysis, "compute_stats", return_value={
            "grounded_correctness": {"mean": 0.6, "std": 0.0, "n": 1},
        }))
        stack.enter_context(patch.object(runner.analysis, "print_summary"))
        stack.enter_context(patch("ragas.Dataset"))
        stack.enter_context(patch.object(
            runner, "experiment", side_effect=self._fake_experiment_decorator,
        ))
        stack.enter_context(patch.object(runner, "EXPERIMENTS_DIR", tmp_path / "experiments"))

        with stack:
            asyncio.run(runner.do_eval(args))

        return recorded

    def test_update_baseline_records_metadata_and_max_docs_reaches_gold_annotation(
        self, tmp_path, monkeypatch, capsys
    ):
        """One live run proves both A4 wires: metadata written + --max-docs used for gold."""
        import json
        import runner

        monkeypatch.setattr(runner, "BASELINE_DIR", tmp_path / "baseline")
        args = self._make_live_args(update_baseline=True, max_docs=500)

        recorded = self._run_live_eval(args, tmp_path)

        saved = json.loads((tmp_path / "baseline" / "gpt-5" / "fetaqa.json").read_text())
        assert saved["agent_id"] == 9
        assert saved["seed"] == 14
        assert saved["judge_model"] == "gemini-3.5-flash"
        assert saved["top_k"] == 5
        # A2 seam: the new CLI flag reaches gold annotation with the run's value
        assert recorded["max_docs"] == 500
        assert "Baseline saved to" in capsys.readouterr().out

    def test_compare_reports_deviations_and_still_prints_deltas(
        self, tmp_path, monkeypatch, capsys
    ):
        """Spec: mismatched agent id / seed are reported, and the comparison still runs."""
        import json
        import runner

        baseline_dir = tmp_path / "baseline"
        baseline_path = baseline_dir / "gpt-5" / "fetaqa.json"
        baseline_path.parent.mkdir(parents=True)
        baseline_stats = {"grounded_correctness": {"mean": 0.1, "std": 0.0, "n": 1}}
        baseline_path.write_text(json.dumps({
            "generated_at": "2026-01-01T00:00:00+00:00",
            "model_id": "gpt-5", "dataset": "fetaqa", "n": 20,
            "agent_id": 9, "seed": 42, "judge_model": "gemini-3.5-flash", "top_k": 5,
            "stats": baseline_stats,
        }))
        monkeypatch.setattr(runner, "BASELINE_DIR", baseline_dir)

        args = self._make_live_args(compare=True, agent_id=10, seed=14)
        with patch.object(runner.analysis, "print_comparison") as mock_compare:
            self._run_live_eval(args, tmp_path)

        out = capsys.readouterr().out
        assert "Comparability deviations" in out
        assert "agent_id: baseline 9 != current 10" in out
        assert "seed: baseline 42 != current 14" in out
        mock_compare.assert_called_once()
        assert mock_compare.call_args.args[0] == baseline_stats
        assert mock_compare.call_args.args[1]["grounded_correctness"]["mean"] == 0.6


# ---------------------------------------------------------------------------
# Phase 6 — B2 probe CLI + gate (D6)
# ---------------------------------------------------------------------------

def _probe_report(**overrides):
    """Synthetic report dict matching pool_probe.probe_pool's frozen shape (B1 docstring)."""
    import pool_probe

    report = {
        "model": "text-embedding-3-small",
        "depths": [100, 500],
        "corpus_size": 1001,
        "effective_corpus_size": 1001,
        "n_questions": 10,
        "n_scored": 9,
        "n_out_of_corpus": 1,
        "rates": {"in": {100: 0.9, 500: 0.95}, "out": {100: 0.1, 500: 0.05}},
        "per_question": [{
            "feta_id": 2275,
            "question": "Q?",
            "gold_doc_id": 37,
            "gold_rank": 37,
            "gold_in_pool": {100: True, 500: True},
            "gold_out_of_corpus": False,
        }],
        "caveat": pool_probe.CAVEAT,
    }
    report.update(overrides)
    return report


def _direction_embedder(gold_text, question_text):
    """Offline deterministic embedder: the gold doc and its question share a direction."""
    def embed(texts):
        return [
            [1.0, 0.0] if text in (gold_text, question_text) else [0.0, 1.0]
            for text in texts
        ]
    return embed


class TestPoolProbeCli:
    """The pool-probe subcommand exposes the D6 flags with the protocol defaults."""

    @pytest.fixture(autouse=True)
    def _parser(self):
        import runner
        self._parser = runner.build_parser()

    def _probe_args(self, argv=None):
        return self._parser.parse_args(["pool-probe", "--questions", "25", *(argv or [])])

    def test_seed_defaults_to_protocol_14(self):
        """Gate runs pin seed 14 (D7: comparison and probe runs pass --seed explicitly)."""
        assert self._probe_args().seed == 14

    def test_embedding_model_defaults_to_indexed_model(self):
        """The probe must embed with the same model the backend indexed with (spec R2)."""
        import pool_probe
        assert self._probe_args().embedding_model == pool_probe.DEFAULT_EMBEDDING_MODEL

    def test_depths_default_to_100_and_500(self):
        assert self._probe_args().depths == [100, 500]

    def test_max_docs_defaults_to_none(self):
        """No prefix flag means the full loader corpus is probed."""
        assert self._probe_args().max_docs is None

    def test_cache_dir_defaults_to_gitignored_evals_subdir(self):
        """Embeddings land under the gitignored evals/ tree — never committed."""
        import runner
        cache_dir = Path(self._probe_args().cache_dir)
        assert cache_dir == runner.DEFAULT_PROBE_CACHE_DIR
        assert cache_dir.name == "pool_probe_cache"
        assert runner.EVALS_DIR in cache_dir.parents

    def test_accepts_explicit_flag_values(self):
        args = self._probe_args([
            "--seed", "7", "--embedding-model", "other-embed", "--max-docs", "500",
            "--depths", "10", "50", "--cache-dir", "custom-cache",
        ])
        assert args.questions == 25
        assert args.seed == 7
        assert args.embedding_model == "other-embed"
        assert args.max_docs == 500
        assert args.depths == [10, 50]
        assert args.cache_dir == "custom-cache"

    def test_questions_is_required(self):
        """The gate's sample size must be explicit — no silent default."""
        with pytest.raises(SystemExit):
            self._parser.parse_args(["pool-probe"])


class TestProbeGateVerdict:
    """The verdict maps deepest-pool reachability to the rama A / rama B decision."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from runner import _probe_gate_verdict
        self._verdict = _probe_gate_verdict

    def test_gold_reaching_the_pool_is_rama_a(self):
        verdict = self._verdict(0.9)
        assert "rama A" in verdict
        assert "reranker" in verdict

    def test_gold_outside_the_pool_is_rama_b(self):
        verdict = self._verdict(0.2)
        assert "rama B" in verdict
        assert "representation" in verdict

    def test_majority_boundary_counts_as_rama_a(self):
        assert "rama A" in self._verdict(0.5)

    def test_just_below_majority_is_rama_b(self):
        assert "rama B" in self._verdict(0.499)

    def test_no_scorable_questions_is_inconclusive(self):
        """An all-out-of-corpus run is not evidence for either rama."""
        assert "inconclusive" in self._verdict(None)


class TestProbeGateReport:
    """The CLI prints dataset-level rates at both depths, the verdict and the caveat."""

    @pytest.fixture(autouse=True)
    def _import_module(self):
        import runner
        self._runner = runner

    def test_report_prints_rates_counts_and_gate(self, capsys):
        verdict = self._runner._print_probe_gate_report(_probe_report())
        out = capsys.readouterr().out
        assert "Gold-in-pool @100" in out
        assert "Gold-in-pool @500" in out
        assert "90.0%" in out and "95.0%" in out
        assert "scored 9" in out
        assert "out-of-corpus 1" in out
        assert "rama A" in verdict

    def test_report_states_the_replication_caveat(self, capsys):
        """Spec: every report states the replication caveat — no parity claimed."""
        import pool_probe
        self._runner._print_probe_gate_report(_probe_report())
        out = capsys.readouterr().out
        assert "Caveat" in out
        assert pool_probe.CAVEAT in out
        assert "necessary but not sufficient" in out

    def test_gold_outside_the_pool_prints_rama_b(self, capsys):
        report = _probe_report(
            rates={"in": {100: 0.1, 500: 0.2}, "out": {100: 0.9, 500: 0.8}},
        )
        verdict = self._runner._print_probe_gate_report(report)
        out = capsys.readouterr().out
        assert "10.0%" in out and "20.0%" in out
        assert "rama B" in verdict

    def test_unscorable_run_prints_na_rates_and_inconclusive_gate(self, capsys):
        report = _probe_report(
            n_scored=0,
            n_out_of_corpus=10,
            rates={"in": {100: None, 500: None}, "out": {100: None, 500: None}},
        )
        verdict = self._runner._print_probe_gate_report(report)
        out = capsys.readouterr().out
        assert "N/A" in out
        assert "inconclusive" in verdict

    def test_verdict_uses_the_deepest_probed_depth(self, capsys):
        """Shallow in-pool but deep out-of-pool is rama B — the deepest pool decides."""
        report = _probe_report(
            depths=[50, 500],
            rates={"in": {50: 0.9, 500: 0.2}, "out": {50: 0.1, 500: 0.8}},
        )
        verdict = self._runner._print_probe_gate_report(report)
        out = capsys.readouterr().out
        assert "Gold-in-pool @50" in out
        assert "Gold-in-pool @500" in out
        assert "rama B" in verdict


class TestPoolProbeWiring:
    """do_pool_probe wires the loader, the OpenAI embedder seam and the gate report."""

    @staticmethod
    def _make_pool_probe_args(**overrides):
        args = argparse.Namespace(
            command="pool-probe", questions=3, seed=14,
            embedding_model="text-embedding-3-small", max_docs=None,
            depths=[100, 500], cache_dir=None,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_probe_runs_offline_and_prints_gate_report(self, tmp_path, capsys):
        """Spec: without any DB/backend the probe completes and prints a gold-in-pool report."""
        import pool_probe
        import runner

        corpus = ["gold doc content", "other doc 0", "other doc 1"]
        rows = [
            {"feta_id": 2275, "question": "Q?", "gold_doc_id": 0},
            {"feta_id": 2276, "question": "Q2?", "gold_doc_id": None},
        ]
        cache_dir = tmp_path / "cache"
        recorded = {}

        def fake_build_openai_embed_fn(model=None, api_key=None, **kwargs):
            recorded["model"] = model
            recorded["api_key"] = api_key
            return _direction_embedder("gold doc content", "Q?")

        args = self._make_pool_probe_args(questions=2, cache_dir=str(cache_dir))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch.object(runner.ds_module, "load_one", return_value=(rows, corpus)) as load_spy, \
             patch("pool_probe.build_openai_embed_fn", side_effect=fake_build_openai_embed_fn):
            runner.do_pool_probe(args)

        load_spy.assert_called_once_with("fetaqa", n=2, seed=14)
        assert recorded == {"model": "text-embedding-3-small", "api_key": "test-key"}
        # The cache seam reached the B1 content-addressed store.
        assert (cache_dir / pool_probe.CACHE_FILENAME).exists()
        out = capsys.readouterr().out
        assert "scored 1" in out
        assert "out-of-corpus 1" in out
        assert "100.0%" in out
        assert "rama A" in out
        assert pool_probe.CAVEAT in out

    def test_missing_openai_key_exits_before_loading(self, monkeypatch, capsys):
        """Without OPENAI_API_KEY the probe exits with a clear operator error."""
        import runner

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with patch.object(runner.ds_module, "load_one") as load_spy:
            with pytest.raises(SystemExit) as excinfo:
                runner.do_pool_probe(self._make_pool_probe_args(cache_dir="unused"))

        assert excinfo.value.code == 1
        load_spy.assert_not_called()
        assert "OPENAI_API_KEY" in capsys.readouterr().err

    def test_max_docs_excludes_gold_as_out_of_corpus_not_a_miss(self, tmp_path, capsys):
        """Spec: gold beyond --max-docs is out-of-corpus, never counted as out-of-pool."""
        import runner

        corpus = ["gold doc content", "other doc 0", "other doc 1"]
        rows = [
            {"feta_id": 1, "question": "Q?", "gold_doc_id": 0},
            {"feta_id": 2, "question": "Q2?", "gold_doc_id": 2},  # beyond the probed prefix
        ]
        args = self._make_pool_probe_args(
            questions=2, max_docs=2, cache_dir=str(tmp_path / "cache"),
        )

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch.object(runner.ds_module, "load_one", return_value=(rows, corpus)), \
             patch("pool_probe.build_openai_embed_fn",
                   side_effect=lambda **kwargs: _direction_embedder("gold doc content", "Q?")):
            runner.do_pool_probe(args)

        out = capsys.readouterr().out
        assert "scored 1" in out
        assert "out-of-corpus 1" in out
        # The single scorable question is in-pool; the excluded one must not dilute the rate.
        assert "100.0%" in out
        assert "50.0%" not in out

    def test_second_run_reuses_cache_without_changing_results(self, tmp_path, capsys):
        """Spec: a re-run on the same dataset/questions/model is identical and may reuse cache."""
        import runner

        corpus = ["gold doc content", "other doc 0", "other doc 1"]
        rows = [{"feta_id": 2275, "question": "Q?", "gold_doc_id": 0}]
        cache_dir = tmp_path / "cache"
        embed_calls = []

        def fake_build_openai_embed_fn(model=None, api_key=None, **kwargs):
            def embed(texts):
                embed_calls.append(list(texts))
                return [[1.0, 0.0] if text in ("gold doc content", "Q?") else [0.0, 1.0]
                        for text in texts]
            return embed

        args = self._make_pool_probe_args(questions=1, cache_dir=str(cache_dir))
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}), \
             patch.object(runner.ds_module, "load_one", return_value=(rows, corpus)), \
             patch("pool_probe.build_openai_embed_fn", side_effect=fake_build_openai_embed_fn):
            runner.do_pool_probe(args)
            first_out = capsys.readouterr().out
            first_run_calls = list(embed_calls)
            embed_calls.clear()
            runner.do_pool_probe(args)
            second_out = capsys.readouterr().out

        assert first_run_calls  # the first run embedded the corpus and the question
        assert embed_calls == []  # the warm cache skipped the embedder entirely
        assert second_out == first_out


# ---------------------------------------------------------------------------
# Multi-gold deterministic retrieval metrics (spec: eval-runner-metrics)
# ---------------------------------------------------------------------------

# Gold-document fixtures for the multi-gold metrics.
_GOLD_DOC_ONE = "Golden document one: alpha row | beta row | gamma row"
_GOLD_DOC_TWO = "Golden document two: delta row | epsilon row"
_GOLD_CHUNK_OF_ONE = "alpha row | beta row"


class _RetrievalMetricsBase:
    """Shared import seam for the deterministic @5 retrieval metrics."""

    @pytest.fixture(autouse=True)
    def _import_metric_functions(self):
        import runner
        self._dedupe_first_occurrence = runner._dedupe_first_occurrence
        self._doc_recall_5 = runner._doc_recall_5
        self._sentence_recall_5 = runner._sentence_recall_5
        self._doc_coverage_5 = runner._doc_coverage_5


class TestDocRecall5(_RetrievalMetricsBase):
    """doc_recall_5 — any in-prefix gold document among the first five unique contexts."""

    def test_any_gold_match_at_the_third_unique_context(self):
        """Spec scenario: the second gold document is the third unique context → 1."""
        contexts = [_CTX_ALPHA, _CTX_BETA, _GOLD_DOC_TWO]
        assert self._doc_recall_5(contexts, [_GOLD_DOC_ONE, _GOLD_DOC_TWO], False) == 1.0

    def test_split_document_containment_scores_1(self):
        """Spec scenario: a retrieved chunk contained in the gold document → 1."""
        contexts = [_CTX_ALPHA, _GOLD_CHUNK_OF_ONE]
        assert self._doc_recall_5(contexts, [_GOLD_DOC_ONE], False) == 1.0

    def test_duplicates_do_not_shift_k(self):
        """Spec scenario: [A, A, B, C, D, E, gold] deduped first → the gold is 6th → 0."""
        contexts = [
            "unrelated passage one", "unrelated passage one", "unrelated passage two",
            "unrelated passage three", "unrelated passage four", "unrelated passage five",
            _GOLD_DOC_ONE,
        ]
        deduped = self._dedupe_first_occurrence(contexts)
        assert len(deduped) == 6
        assert self._doc_recall_5(deduped, [_GOLD_DOC_ONE], False) == 0.0

    def test_no_match_scores_zero_and_stays_in_the_denominator(self):
        """Spec scenario: labeled, in-corpus, no match → 0 (not None)."""
        contexts = [_CTX_ALPHA, _CTX_BETA]
        assert self._doc_recall_5(contexts, [_GOLD_DOC_ONE], False) == 0.0

    def test_out_of_corpus_is_not_applicable(self):
        """Spec scenario: gold-out-of-corpus → None, excluded from the denominator."""
        contexts = [_GOLD_DOC_ONE]
        assert self._doc_recall_5(contexts, [_GOLD_DOC_ONE], True) is None

    def test_no_gold_labels_is_not_applicable(self):
        """Spec scenario: no-label → None even when a context would look like gold."""
        assert self._doc_recall_5([_GOLD_DOC_ONE], [], False) is None

    def test_only_the_first_five_unique_contexts_count(self):
        """A gold document in the sixth unique context is not recalled @5."""
        contexts = ["c1", "c2", "c3", "c4", "c5", _GOLD_DOC_ONE]
        assert self._doc_recall_5(contexts, [_GOLD_DOC_ONE], False) == 0.0

    def test_gold_matched_by_several_contexts_is_still_one(self):
        """Any-gold semantics: repeated matches never exceed 1."""
        contexts = [_GOLD_CHUNK_OF_ONE, _GOLD_DOC_ONE]
        assert self._doc_recall_5(contexts, [_GOLD_DOC_ONE], False) == 1.0


class TestSentenceRecall5(_RetrievalMetricsBase):
    """sentence_recall_5 — fraction of usable in-prefix gold sentences found @5."""

    def test_partial_recall_is_two_thirds(self):
        """Spec scenario: three usable sentences, two found → 2/3."""
        contexts = [_CTX_ALPHA, "contains sentence one and sentence two"]
        gold_sentences = ["sentence one", "sentence two", "sentence three"]
        assert self._sentence_recall_5(contexts, gold_sentences, False) == pytest.approx(2 / 3)

    def test_sentence_found_only_in_the_fifth_context_counts(self):
        """Spec scenario: a sentence appearing only in the fifth unique context counts."""
        contexts = ["c1", "c2", "c3", "c4", "the golden sentence lives here"]
        assert self._sentence_recall_5(contexts, ["the golden sentence"], False) == 1.0

    def test_no_match_scores_zero(self):
        """Spec scenario: usable sentences, none found → 0."""
        contexts = [_CTX_ALPHA, _CTX_BETA]
        assert self._sentence_recall_5(contexts, ["a sentence that is absent"], False) == 0.0

    def test_degenerate_sentences_leave_the_denominator(self):
        """Spec scenario: empty/whitespace-only sentences leave numerator and denominator."""
        contexts = ["only the real sentence appears here"]
        gold_sentences = ["", "   ", "\r\n", "the real sentence"]
        assert self._sentence_recall_5(contexts, gold_sentences, False) == 1.0

    def test_all_degenerate_sentences_is_not_applicable(self):
        """Spec scenario: no usable sentence remains → None, never 0."""
        assert self._sentence_recall_5(["anything"], ["", "  \n "], False) is None

    def test_empty_sentence_list_is_not_applicable(self):
        """No in-prefix sentences (or no labels at all) → None."""
        assert self._sentence_recall_5(["anything"], [], False) is None

    def test_out_of_corpus_is_not_applicable(self):
        """Spec scenario: gold-out-of-corpus → None."""
        assert self._sentence_recall_5(["anything"], ["a sentence"], True) is None

    def test_sentence_found_in_another_documents_context_still_counts(self):
        """The spec criterion is a normalized substring of any of the five contexts."""
        contexts = ["this context quotes the golden sentence verbatim"]
        assert self._sentence_recall_5(contexts, ["the golden sentence"], False) == 1.0


class TestDocCoverage5(_RetrievalMetricsBase):
    """doc_coverage_5 — fraction of in-prefix gold documents matched @5."""

    def test_multi_gold_coverage_is_two_thirds(self):
        """Spec scenario: three gold documents, two among the first five → 2/3."""
        contexts = [_CTX_ALPHA, _GOLD_DOC_ONE, _GOLD_DOC_TWO, "c4", "c5", "c6"]
        gold_contents = [_GOLD_DOC_ONE, _GOLD_DOC_TWO, "a gold document that is absent"]
        assert self._doc_coverage_5(contexts, gold_contents, False) == pytest.approx(2 / 3)

    def test_single_gold_degenerates_to_doc_recall(self):
        """Spec scenario: single-gold coverage equals the doc_recall_5 value."""
        contexts = [_CTX_ALPHA, _GOLD_DOC_ONE]
        assert self._doc_coverage_5(contexts, [_GOLD_DOC_ONE], False) == 1.0
        assert self._doc_coverage_5([_CTX_ALPHA], [_GOLD_DOC_ONE], False) == 0.0
        assert self._doc_recall_5([_CTX_ALPHA], [_GOLD_DOC_ONE], False) == 0.0

    def test_each_gold_document_counts_at_most_once(self):
        """Repeated chunk matches for one gold document never exceed its share."""
        contexts = [_GOLD_CHUNK_OF_ONE, _GOLD_DOC_ONE]
        assert self._doc_coverage_5(contexts, [_GOLD_DOC_ONE, "absent gold"], False) == 0.5

    def test_exclusions_are_not_applicable(self):
        """Spec scenario: out-of-corpus or no-label → None."""
        assert self._doc_coverage_5([_GOLD_DOC_ONE], [_GOLD_DOC_ONE], True) is None
        assert self._doc_coverage_5([_GOLD_DOC_ONE], [], False) is None


class TestRetrievalMetricErrorRows:
    """Every error/guard row carries the five deterministic metric columns as None."""

    METRIC_COLUMNS = (
        "table_recall_5", "cell_recall_5",
        "doc_recall_5", "sentence_recall_5", "doc_coverage_5",
    )

    @pytest.fixture(autouse=True)
    def _import_runner(self):
        import runner
        self._runner = runner

    def _assert_all_metric_columns_none(self, row: dict):
        for column in self.METRIC_COLUMNS:
            assert column in row, f"missing metric column in the row: {column}"
            assert row[column] is None, f"{column} must be None on an error row"

    def test_metric_defaults_cover_all_five_columns(self):
        """One constant fronts every error-row default (no drifting literals)."""
        assert self._runner._RETRIEVAL_METRIC_DEFAULTS == {
            "table_recall_5": None,
            "cell_recall_5": None,
            "doc_recall_5": None,
            "sentence_recall_5": None,
            "doc_coverage_5": None,
        }

    def test_compute_metrics_error_row_carries_the_metric_columns(self):
        """A judge failure returns an error row with all five metrics None."""
        exploding_recall = AsyncMock()
        exploding_recall.single_turn_ascore.side_effect = RuntimeError("judge exploded")

        result = asyncio.run(self._runner._compute_metrics_from_sample(
            row={"question": "Q?", "grading_notes": "G"},
            answer="an answer",
            retrieved_contexts=["ctx"],
            citations=[],
            judge_llm=None,
            context_recall=exploding_recall,
            context_precision=MagicMock(),
            faithfulness=MagicMock(),
            correctness=MagicMock(),
            citation_faithfulness=MagicMock(),
        ))

        assert result["error"] == "judge exploded"
        self._assert_all_metric_columns_none(result)

    def test_process_question_error_row_carries_the_metric_columns(self):
        """A Tero failure returns an error row with all five metrics None."""
        mock_tero = AsyncMock()
        mock_tero.create_thread.side_effect = RuntimeError("backend down")

        result = asyncio.run(self._runner._process_question_result(
            mock_tero, _make_test_row(), None, None, None, None, None, None,
        ))

        assert result["error"] == "backend down"
        self._assert_all_metric_columns_none(result)

    def test_recursion_limit_row_carries_the_metric_columns(self):
        """The recursionLimitExceeded row carries all five metrics as None."""
        mock_tero = _make_mock_tero(answer_text="recursionLimitExceeded: loop detected")

        result = asyncio.run(self._runner._process_question_result(
            mock_tero, _make_test_row(), None, None, None, None, None, None,
        ))

        assert result["error"] == "recursionLimitExceeded"
        self._assert_all_metric_columns_none(result)

    def _write_guard_csv(self, path):
        """Three rows, one per offline guard: missing value, bad JSON, recursion."""
        path.write_text(
            "question;response;retrieved_contexts\n"
            ";an answer for a missing question;ctx\n"
            "Q2;an answer;123\n"
            "Q3;recursionLimitExceeded(loop);ctx\n",
            encoding="utf-8",
        )
        return path

    def test_csv_mode_guard_rows_carry_the_metric_columns(self, tmp_path):
        """The three _run_csv_mode guards all emit the five metric columns as None."""
        import argparse
        import pandas as pd

        csv_path = self._write_guard_csv(tmp_path / "guards.csv")
        args = argparse.Namespace()
        args.from_csv = str(csv_path)

        with patch.object(self._runner, "_build_judge_client",
                          return_value=(None, None, None, "mock-judge")), \
                patch.object(self._runner, "_build_metrics", return_value=(
                    MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock(),
                )), patch.object(self._runner, "EVALS_DIR", tmp_path):
            asyncio.run(self._runner._run_csv_mode(args, "gemini-3.5-flash"))

        csvs = list((tmp_path / "experiments" / "offline").glob("*.csv"))
        assert len(csvs) == 1
        df = pd.read_csv(csvs[0], sep=";")
        assert list(df["error"]) == [
            "missing_required_value", "json_parse_error", "recursionLimitExceeded",
        ]
        for column in self.METRIC_COLUMNS:
            assert column in df.columns
            assert df[column].isna().all(), f"{column} must be empty on every guard row"


class TestAlignmentReport:
    """Population counts separate scorable / out-of-corpus / no-label / unlinked."""

    @pytest.fixture(autouse=True)
    def _import_functions(self):
        import runner
        self._population_counts = runner._population_counts
        self._print_alignment_report = runner._print_alignment_report

    def _scorable_row(self, question):
        return {
            "question": question,
            "gold_doc_ids_in_prefix": [0],
            "gold_contents_in_prefix": ["doc0"],
            "gold_out_of_corpus": False,
            "no_gold_labels": False,
        }

    def _out_of_corpus_row(self, question):
        return {
            "question": question,
            "gold_doc_ids_in_prefix": [],
            "gold_contents_in_prefix": [],
            "gold_out_of_corpus": True,
            "no_gold_labels": False,
        }

    def _no_label_row(self, question):
        return {
            "question": question,
            "gold_doc_ids_in_prefix": [],
            "gold_contents_in_prefix": [],
            "gold_out_of_corpus": False,
            "no_gold_labels": True,
        }

    def test_populations_are_counted_separately(self):
        """Spec: scorable, out-of-corpus, no-label and unlinked are distinct counts."""
        rows = [
            self._scorable_row("s1"),
            self._scorable_row("s2"),
            self._out_of_corpus_row("o1"),
            self._no_label_row("n1"),
            {"question": "u1", "grading_notes": "g"},
        ]
        assert self._population_counts(rows) == {
            "n_questions": 5,
            "n_scorable": 2,
            "n_out_of_corpus": 1,
            "n_no_gold_labels": 1,
            "n_without_gold_linkage": 1,
        }

    def test_counts_account_for_every_question(self):
        """Invariant: every question lands in exactly one population bucket."""
        rows = [
            self._scorable_row("s1"),
            self._out_of_corpus_row("o1"),
            self._no_label_row("n1"),
            {"question": "u1", "grading_notes": "g"},
        ]
        counts = self._population_counts(rows)
        assert counts["n_questions"] == (
            counts["n_scorable"] + counts["n_out_of_corpus"]
            + counts["n_no_gold_labels"] + counts["n_without_gold_linkage"]
        )

    def test_no_label_question_is_never_counted_as_a_miss(self, capsys):
        """Spec scenario: a no-label question with no gold is not a miss."""
        import pandas as pd

        rows = [self._no_label_row("n1"), self._scorable_row("s1")]
        results_df = pd.DataFrame([
            {"question": "n1", "doc_recall_5": None},
            {"question": "s1", "doc_recall_5": 1.0},
        ])

        self._print_alignment_report(rows, results_df)

        out = capsys.readouterr().out
        assert "No gold labels       : 1" in out
        assert "Out-of-corpus        : 0" in out
        assert "Scorable             : 1" in out
        assert "misses 0" in out

    def test_report_separates_misses_from_exclusions_and_unmeasured(self, capsys):
        """Misses, exclusions and unmeasured rows print independently."""
        import pandas as pd

        rows = [
            self._scorable_row("miss"),
            self._scorable_row("hit"),
            self._scorable_row("never measured"),
            self._out_of_corpus_row("excluded"),
            self._no_label_row("unlabeled"),
        ]
        results_df = pd.DataFrame([
            {"question": "miss", "doc_recall_5": 0.0},
            {"question": "hit", "doc_recall_5": 1.0},
            {"question": "excluded", "doc_recall_5": None},
            {"question": "unlabeled", "doc_recall_5": None},
        ])

        self._print_alignment_report(rows, results_df)

        out = capsys.readouterr().out
        assert "Scorable             : 3" in out
        assert "Out-of-corpus        : 1" in out
        assert "No gold labels       : 1" in out
        assert "measured 2, misses 1, unmeasured 1" in out

    def test_report_without_a_results_frame_prints_populations_only(self, capsys):
        """Annotations alone still report the populations (no metric line)."""
        rows = [self._scorable_row("s1"), self._out_of_corpus_row("o1")]

        self._print_alignment_report(rows)

        out = capsys.readouterr().out
        assert "Scorable             : 1" in out
        assert "Out-of-corpus        : 1" in out
        assert "misses" not in out

    def test_scorable_count_matches_the_metric_denominator(self):
        """Invariant: the scorable count is exactly the doc_recall_5 denominator."""
        rows = [
            self._scorable_row("s1"),
            self._scorable_row("s2"),
            self._out_of_corpus_row("o1"),
            self._no_label_row("n1"),
        ]
        counts = self._population_counts(rows)

        measured = [
            self._doc_recall_value(row)
            for row in rows
        ]
        denominator = [value for value in measured if value is not None]
        assert counts["n_scorable"] == len(denominator) == 2

    def _doc_recall_value(self, row):
        import runner
        return runner._doc_recall_5(
            ["doc0"], row.get("gold_contents_in_prefix"),
            bool(row.get("gold_out_of_corpus", False)),
        )

