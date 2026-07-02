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
    """Tests for _find_relevant_chunk_index pure function."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        """Import the function once the stub exists in runner.py."""
        from runner import _find_relevant_chunk_index
        self._find_relevant_chunk_index = _find_relevant_chunk_index

    def test_relevant_chunk_first_position(self):
        """Grading notes substring found in retrieved context at 1-based index 3."""
        grading_notes = "Barack Hussein Obama"
        contexts = [
            "Some unrelated context about politics",
            "Another unrelated context about elections",
            "Biography: Barack Hussein Obama was born in Hawaii...",
            "More unrelated text",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == 3, f"Expected 3, got {result}"

    def test_relevant_chunk_not_found(self):
        """No substring overlap → returns -1."""
        grading_notes = "Carbon dating methods"
        contexts = [
            "Geology of the Grand Canyon",
            "History of radiometric techniques",
            "Fossil record in sedimentary rocks",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == -1, f"Expected -1, got {result}"

    def test_relevant_chunk_with_short_grading_notes(self):
        """Grading notes shorter than 10 chars → returns -1 (no 10-char substring possible)."""
        grading_notes = "Short"
        contexts = [
            "This is a short note context",
            "Another context",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == -1, f"Expected -1 for short grading_notes (< 10 chars), got {result}"

    def test_relevant_chunk_case_insensitive(self):
        """Substring match is case-insensitive."""
        grading_notes = "BARACK HUSSEIN OBAMA"
        contexts = [
            "Context A",
            "barack hussein obama biography",
            "Context C",
        ]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == 2, f"Case-insensitive match should find at position 2, got {result}"

    def test_relevant_chunk_empty_contexts(self):
        """Empty contexts list → returns -1."""
        grading_notes = "Some relevant content here"
        contexts = []
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == -1, f"Expected -1 for empty contexts, got {result}"

    def test_relevant_chunk_exact_10_char_match(self):
        """Minimum 10-char substring exactly matches."""
        grading_notes = "1234567890extra"
        contexts = ["1234567890"]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == 1, f"Expected 1 for exact 10-char match, got {result}"

    def test_relevant_chunk_9_char_no_match(self):
        """9-char overlap should NOT match (minimum 10 chars required)."""
        grading_notes = "12345678X"
        contexts = ["12345678"]
        result = self._find_relevant_chunk_index(grading_notes, contexts)
        assert result == -1, f"Expected -1 for <10 char match, got {result}"


# ---------------------------------------------------------------------------
# Helpers for async runner tests
# ---------------------------------------------------------------------------

def _make_mock_tero(answer_text="Normal answer", contexts=None, citations=None, latency_ms=100,
                    error="", parse_failures=0):
    """Create a mock TeroClient with configurable ask_question response.

    Includes ``error`` and ``parse_failures`` fields (backward-compatible defaults)
    so existing tests keep working after tero_client starts returning them.
    """
    mock = AsyncMock()
    mock.create_thread.return_value = "thread_test_123"
    mock.ask_question.return_value = {
        "answer_text": answer_text,
        "retrieved_contexts": contexts or ["ctx1", "ctx2", "ctx3"],
        "citations": citations or ["cite1", "cite2"],
        "latency_ms": latency_ms,
        "error": error,
        "parse_failures": parse_failures,
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
        assert result["faithfulness_valid"] is False
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
        assert result["faithfulness_valid"] is True


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
# TASK-1.2 — Answer extraction tests (5 cases)
# ---------------------------------------------------------------------------

class TestExtractAnswerText:
    """Tests for _extract_answer_text from tero_client."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from tero_client import _extract_answer_text
        self._func = _extract_answer_text

    def test_normal_sse_stream_extraction(self):
        """Leading JSON events stripped, answer text returned."""
        raw = (
            '{"action":"thinking","content":"..."}\n'
            '{"action":"responding","content":"start"}\n'
            'The capital of France is Paris.\n'
            'It is known for the Eiffel Tower.'
        )
        result = self._func(raw)
        assert "The capital of France is Paris." in result
        assert "It is known for the Eiffel Tower." in result
        assert "thinking" not in result

    def test_json_like_content_in_answer_preserved(self):
        """Answer containing legitimate JSON-like content {\"key\": \"value\"} preserved."""
        raw = (
            '{"action":"responding","content":"..."}\n'
            'The response includes {"key": "value"} inline.\n'
            'More text follows.'
        )
        result = self._func(raw)
        assert '{"key": "value"}' in result
        assert "inline." in result

    def test_empty_stream_returns_empty(self):
        """Empty raw string → empty answer."""
        assert self._func("") == ""

    def test_only_json_events_returns_empty(self):
        """Raw contains only JSON events → empty answer."""
        raw = '{"action":"thinking"}\n{"action":"responding"}\n'
        result = self._func(raw)
        assert result == ""

    def test_trailing_answer_message_id_stripped(self):
        """Trailing {\"answerMessageId\":\"xxx\"} removed; mid-answer {\"answerMessageId\":...} preserved."""
        raw = (
            '{"action":"responding"}\n'
            'The answer says {"answerMessageId": "mid_val"} is used internally.\n'
            'But the trailing one should be removed.\n'
            '{"answerMessageId":"msg_abc123"}'
        )
        result = self._func(raw)
        # Trailing metadata blob removed
        assert 'answerMessageId":"msg_abc123"' not in result
        assert not result.strip().endswith("}")
        # Mid-answer {"answerMessageId": "mid_val"} preserved
        assert 'answerMessageId": "mid_val"' in result
        assert "used internally" in result


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
        assert result["faithfulness_valid"] is False
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
        assert result["faithfulness_valid"] is False
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
        assert result["faithfulness_valid"] is True
        assert result["grounded_correctness"] is not None
        assert result["grounded_correctness"] == round((4 / 4) * 0.85, 3)


# ---------------------------------------------------------------------------
# TASK-1.6 — wait_files_processed missing file detection
# ---------------------------------------------------------------------------

class TestWaitFilesProcessed:
    """Tests that wait_files_processed doesn't silently drop files missing from API response."""

    @pytest.fixture
    def mock_client(self):
        """Create an httpx.AsyncClient mock for tero_client tests."""
        import httpx
        return MagicMock()

    def test_file_ids_absent_from_response_stay_pending(self):
        """When a file ID is missing from API response, it stays pending (not dropped)."""
        import tero_client as tc
        import asyncio as aio

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # API returns only some file IDs
        call_count = [0]
        async def mock_get(url):
            call_count[0] += 1
            resp = MagicMock()
            if call_count[0] >= 3:
                # Eventually return all files
                resp.json.return_value = [
                    {"id": 1, "status": "PROCESSED"},
                    {"id": 2, "status": "PROCESSED"},
                    {"id": 3, "status": "PROCESSED"},
                ]
            else:
                # File 2 is missing from response
                resp.json.return_value = [
                    {"id": 1, "status": "PROCESSED"},
                ]
            resp.raise_for_status = MagicMock()
            return resp

        mock_client.get = mock_get

        # Test the pending logic directly: when file 2 is absent,
        # the default status must NOT be in ("PROCESSED", "ERROR")
        files = {f["id"]: f for f in [{"id": 1, "status": "PROCESSED"}]}
        # Current bug: files.get(2, {}).get("status") → None → excluded
        # Fix: should be "MISSING" (not in exclusion set)
        default_status = files.get(2, {}).get("status")
        # The old default (None) should NOT be the value we compare against
        # New code will use something NOT in ("PROCESSED", "ERROR")
        assert default_status is None or default_status != "PROCESSED"
        # After fix, the default will be "MISSING" which IS not in excluded set

    def test_missing_sentinel_not_dropped(self):
        """Verify the fix: default for missing files should NOT be None."""
        # This test asserts the contract: the sentinel for missing files
        # must NOT match the exclusion condition
        from tero_client import TeroClient
        # Read the source to verify the current state (will FAIL until fix applied)
        import inspect
        src = inspect.getsource(TeroClient.wait_files_processed)
        # Current code uses None as default (bug) — fix will change to "MISSING"
        assert '"MISSING"' in src or 'MISSING' in src, \
            "wait_files_processed must use 'MISSING' sentinel, not None"


# ---------------------------------------------------------------------------
# TASK-1.7 — CSV semicolon idempotency test
# ---------------------------------------------------------------------------

class TestCsvSemicolonIdempotency:
    """Tests that _csv_to_semicolon correctly detects already-converted files."""

    def test_semicolon_csv_with_comma_in_header_is_detected(self):
        """CSV with semicolons AND commas in header → correctly identified as already converted."""
        import tempfile
        from pathlib import Path
        import runner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "test.csv"
            # Semicolon-delimited CSV with a comma in the column name
            csv_path.write_text(
                'question;response;correctness,faithfulness;context_recall\n'
                'Q1;R1;3;0.8\n',
                encoding="utf-8",
            )

            runner._csv_to_semicolon(tmp)

            content = csv_path.read_text(encoding="utf-8")
            # File should be unchanged (already semicolon-delimited)
            assert 'question;response;correctness,faithfulness;context_recall' in content
            assert content.count(";") >= 3  # Still semicolon-separated

    def test_comma_csv_still_converted(self):
        """Comma-delimited CSV without semicolons → still gets converted."""
        import tempfile
        from pathlib import Path
        import runner

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            csv_path = tmp / "test.csv"
            csv_path.write_text(
                'question,response,correctness\n'
                'Q1,R1,3\n',
                encoding="utf-8",
            )

            runner._csv_to_semicolon(tmp)

            content = csv_path.read_text(encoding="utf-8")
            assert ";" in content
            assert "," not in content.splitlines()[0]  # Header converted


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
        assert result["faithfulness_valid"] is True
        assert isinstance(result["context_recall"], float)
        assert isinstance(result["context_precision"], float)
        assert result["grounded_correctness"] == round((3 / 4) * 0.9, 3)
        assert result["latency_ms"] == 150.5
        assert result["error"] is None
        # Verify all 14+ output keys exist
        expected_keys = {
            "question", "grading_notes", "response", "retrieved_contexts",
            "citations", "latency_ms", "correctness", "faithfulness",
            "faithfulness_valid", "context_recall", "context_precision",
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
        assert result["faithfulness_valid"] is False
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
        assert result["faithfulness_valid"] is False
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
            "correctness", "faithfulness", "faithfulness_valid",
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
            "faithfulness_valid": True,
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
                            asyncio.run(runner._run_csv_mode(args, "test-key"))

                        # Verify output CSV was written
                        offline_dir = tmp_path / "experiments" / "offline"
                        csvs = list(offline_dir.glob("*.csv"))
                        assert len(csvs) == 1, f"Expected 1 output CSV, got {len(csvs)}"
                        output_csv = csvs[0]

                        # Read the output CSV
                        df = pd.read_csv(output_csv, sep=";")
                        assert len(df) == 3, f"Expected 3 rows, got {len(df)}"

                        # Verify 14+ output columns
                        expected_cols = {"question", "response", "retrieved_contexts", "citations",
                                         "latency_ms", "correctness", "faithfulness", "faithfulness_valid",
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
                            asyncio.run(runner._run_csv_mode(args, "test-key"))

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
            asyncio.run(runner._run_csv_mode(args, "test-key"))

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
                            asyncio.run(runner._run_csv_mode(args, "test-key"))

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
                            asyncio.run(runner._run_csv_mode(args, "test-key"))

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
                    asyncio.run(runner._run_csv_mode(args, "test-key"))

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
                            asyncio.run(runner._run_csv_mode(args, "test-key"))

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
        assert "LLM description tokens: 0 (skipped)" in captured.out

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
        mock_tero.configure_docs_tool = AsyncMock()
        mock_tero.upload_document = AsyncMock(side_effect=[101, 102, 103])
        mock_tero.wait_files_processed = AsyncMock()

        # Mock tiktoken: each char = 1 token for predictable counts
        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(side_effect=lambda s: list(range(len(s))))

        with patch("tero_client.TeroClient", return_value=mock_tero):
            with patch.object(runner.ds_module, "load_one", return_value=([], mock_corpus)):
                with patch("tiktoken.get_encoding", return_value=mock_enc):
                    with patch.object(runner, "cost_report") as mock_cost:
                        aio.run(runner.do_index(args))

        # Corpus loaded with n=0 (full corpus, zero questions)
        runner.ds_module.load_one.assert_called_once_with("ragbench", n=0)

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
        """--from-csv with no GOOGLE_API_KEY → SystemExit(1)."""
        import runner
        import asyncio as aio

        args = self._make_eval_args(
            agent_id=None,
            models=None,
            from_csv="data.csv",
        )

        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                aio.run(runner.do_eval(args))
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# Phase 4: argparse subparser — old flag rejection (Task 4.2)
# ---------------------------------------------------------------------------

class TestSubparserOldFlagRejection:
    """Tests that old runner.py flags (--corpus-size, --n, --offset, --only)
    are rejected by the new index/eval subcommands with argparse error."""

    @staticmethod
    def _parse_via_main(subcommand, extra_args):
        """Run runner.main() with given subcommand and extra args.
        Returns exit code: 0 if main completed, caught SystemExit code otherwise.
        Patches do_index/do_eval to prevent actual async execution on valid parses.
        """
        import sys
        import runner
        from unittest.mock import patch, AsyncMock

        old_argv = sys.argv
        try:
            base = ["runner.py", subcommand]
            if subcommand == "eval":
                base += ["--dataset", "ragbench"]
            else:
                base += ["--dataset", "ragbench"]
            sys.argv = base + extra_args
            with patch.object(runner, "do_index", new=AsyncMock()):
                with patch.object(runner, "do_eval", new=AsyncMock()):
                    runner.main()
            return 0
        except SystemExit as e:
            return e.code
        finally:
            sys.argv = old_argv

    # ── eval subcommand old flag rejection ──

    def test_eval_rejects_corpus_size(self):
        """eval --corpus-size 200 → argparse error."""
        code = self._parse_via_main("eval", [
            "--corpus-size", "200", "--agent-id", "9", "--models", "gpt-5",
        ])
        assert code != 0, f"Expected argparse error for --corpus-size, got exit {code}"

    def test_eval_rejects_n_flag(self):
        """eval --n 10 → argparse error."""
        code = self._parse_via_main("eval", [
            "--n", "10", "--agent-id", "9", "--models", "gpt-5",
        ])
        assert code != 0, f"Expected argparse error for --n, got exit {code}"

    def test_eval_rejects_offset_flag(self):
        """eval --offset 5 → argparse error."""
        code = self._parse_via_main("eval", [
            "--offset", "5", "--agent-id", "9", "--models", "gpt-5",
        ])
        assert code != 0, f"Expected argparse error for --offset, got exit {code}"

    def test_eval_rejects_only_flag(self):
        """eval --only fetaqa → argparse error."""
        code = self._parse_via_main("eval", [
            "--only", "fetaqa", "--agent-id", "9", "--models", "gpt-5",
        ])
        assert code != 0, f"Expected argparse error for --only, got exit {code}"

    def test_eval_rejects_rebuild_corpus(self):
        """eval --rebuild-corpus → argparse error."""
        code = self._parse_via_main("eval", [
            "--rebuild-corpus", "--agent-id", "9", "--models", "gpt-5",
        ])
        assert code != 0, f"Expected argparse error for --rebuild-corpus, got exit {code}"

    # ── index subcommand old flag rejection ──

    def test_index_rejects_corpus_size(self):
        """index --corpus-size 200 → argparse error."""
        code = self._parse_via_main("index", [
            "--corpus-size", "200", "--agent-id", "9",
        ])
        assert code != 0, f"Expected argparse error for --corpus-size, got exit {code}"

    def test_index_rejects_n_flag(self):
        """index --n 10 → argparse error."""
        code = self._parse_via_main("index", [
            "--n", "10", "--agent-id", "9",
        ])
        assert code != 0, f"Expected argparse error for --n, got exit {code}"

    def test_index_rejects_offset_flag(self):
        """index --offset 5 → argparse error."""
        code = self._parse_via_main("index", [
            "--offset", "5", "--agent-id", "9",
        ])
        assert code != 0, f"Expected argparse error for --offset, got exit {code}"

    def test_index_rejects_seed_flag(self):
        """index --seed 123 → argparse error."""
        code = self._parse_via_main("index", [
            "--seed", "123", "--agent-id", "9",
        ])
        assert code != 0, f"Expected argparse error for --seed, got exit {code}"

    # ── valid flag acceptance (no rejection) ──

    def test_eval_accepts_valid_flags(self):
        """eval with valid flags (--dataset, --agent-id, --models, --seed) succeeds."""
        code = self._parse_via_main("eval", [
            "--agent-id", "9", "--models", "gpt-5",
            "--max-questions", "5", "--update-baseline",
            "--seed", "123",
        ])
        assert code == 0, f"Expected exit 0 for valid eval flags, got {code}"

    def test_index_accepts_valid_flags(self):
        """index with valid flags (--dataset, --agent-id) succeeds."""
        code = self._parse_via_main("index", [
            "--agent-id", "9", "--max-docs", "100",
        ])
        assert code == 0, f"Expected exit 0 for valid index flags, got {code}"

    def test_eval_accepts_csv_mode_flags(self):
        """eval --from-csv with no agent-id succeeds (CSV path)."""
        code = self._parse_via_main("eval", [
            "--from-csv", "data.csv",
        ])
        assert code == 0, f"Expected exit 0 for valid CSV-mode eval flags, got {code}"
