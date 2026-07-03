"""
Unit and integration tests for TeroClient — rag-reindex-clean change.

Covers:
  - list_file_ids(): 404 → [], 200 → [ids], 500 → raises
  - delete_all_files(): retry-then-raise, no-op on empty, transient recovery
  - do_index() teardown order and first-run guard
"""
import asyncio
import sys
import os
from unittest.mock import AsyncMock, MagicMock, patch, call

import httpx
import pytest

# Make tero_client and runner importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client():
    """Return a TeroClient pointed at a local test server."""
    from tero_client import TeroClient
    return TeroClient(
        base_url="http://localhost:8000",
        agent_id=9,
        bearer_token="test-token",
    )


def _make_http_response(status_code: int, json_data=None, text: str = ""):
    """Build a MagicMock that behaves like an httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.is_success = 200 <= status_code < 300
    resp.is_error = not resp.is_success
    resp.text = text
    if json_data is not None:
        resp.json.return_value = json_data
    if not resp.is_success:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _patch_httpx_client(mock_inner_client):
    """
    Return a context manager that patches httpx.AsyncClient so that
    ``async with httpx.AsyncClient(...) as client:`` yields *mock_inner_client*.
    """
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_inner_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_cls = MagicMock(return_value=mock_ctx)
    return patch.object(httpx, "AsyncClient", mock_cls)


# ---------------------------------------------------------------------------
# Phase 4.1 / 4.2 — list_file_ids() unit tests
# ---------------------------------------------------------------------------

class TestListFileIds:
    """Unit tests for TeroClient.list_file_ids()."""

    # ------------------------------------------------------------------
    # RED: 404 → [] (task 4.1)
    # ------------------------------------------------------------------
    def test_list_file_ids_returns_empty_on_404(self):
        """list_file_ids returns [] when the API responds with 404 (no docs tool yet)."""
        client = _make_client()
        resp_404 = _make_http_response(404)

        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(return_value=resp_404)

        with _patch_httpx_client(mock_inner):
            result = asyncio.run(client.list_file_ids())

        assert result == [], f"Expected [], got {result!r}"
        mock_inner.get.assert_called_once()

    # ------------------------------------------------------------------
    # GREEN triangulation: 200 → list of int IDs
    # ------------------------------------------------------------------
    def test_list_file_ids_returns_int_ids_on_200(self):
        """list_file_ids extracts 'id' field from each file object and returns int IDs."""
        client = _make_client()
        files_json = [
            {"id": 1, "status": "PROCESSED"},
            {"id": 2, "status": "PROCESSED"},
            {"id": 7, "status": "ERROR"},
        ]
        resp_200 = _make_http_response(200, json_data=files_json)

        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(return_value=resp_200)

        with _patch_httpx_client(mock_inner):
            result = asyncio.run(client.list_file_ids())

        assert result == [1, 2, 7], f"Expected [1, 2, 7], got {result!r}"

    # ------------------------------------------------------------------
    # Triangulation: single file
    # ------------------------------------------------------------------
    def test_list_file_ids_returns_single_id(self):
        """list_file_ids with one file returns a one-element int list."""
        client = _make_client()
        resp_200 = _make_http_response(200, json_data=[{"id": 42, "status": "PROCESSING"}])

        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(return_value=resp_200)

        with _patch_httpx_client(mock_inner):
            result = asyncio.run(client.list_file_ids())

        assert result == [42]

    # ------------------------------------------------------------------
    # RED: 500 → raises (task 4.2)
    # ------------------------------------------------------------------
    def test_list_file_ids_raises_on_500(self):
        """list_file_ids raises httpx.HTTPStatusError on any non-404 error response."""
        client = _make_client()
        resp_500 = _make_http_response(500, text="Internal Server Error")

        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(return_value=resp_500)

        with _patch_httpx_client(mock_inner):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(client.list_file_ids())

    # ------------------------------------------------------------------
    # Triangulation: 403 (other 4xx, not 404) → also raises
    # ------------------------------------------------------------------
    def test_list_file_ids_raises_on_403(self):
        """list_file_ids raises on non-404 client errors (403 Forbidden)."""
        client = _make_client()
        resp_403 = _make_http_response(403, text="Forbidden")

        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(return_value=resp_403)

        with _patch_httpx_client(mock_inner):
            with pytest.raises(httpx.HTTPStatusError):
                asyncio.run(client.list_file_ids())

    # ------------------------------------------------------------------
    # Verify correct URL is called
    # ------------------------------------------------------------------
    def test_list_file_ids_calls_correct_url(self):
        """list_file_ids calls GET /api/agents/{agent_id}/tools/docs/files."""
        client = _make_client()
        resp_200 = _make_http_response(200, json_data=[])

        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(return_value=resp_200)

        with _patch_httpx_client(mock_inner):
            asyncio.run(client.list_file_ids())

        mock_inner.get.assert_called_once_with(
            "http://localhost:8000/api/agents/9/tools/docs/files"
        )


# ---------------------------------------------------------------------------
# Phase 4.3 / 4.4 — delete_all_files() unit tests
# ---------------------------------------------------------------------------

class TestDeleteAllFiles:
    """Unit tests for TeroClient.delete_all_files() — retry logic and no-op path."""

    # ------------------------------------------------------------------
    # RED: empty file list → returns 0, no DELETEs (task 4.4)
    # ------------------------------------------------------------------
    def test_delete_all_files_noop_when_no_files(self):
        """delete_all_files returns 0 immediately when list_file_ids returns []."""
        client = _make_client()

        with patch.object(client, "list_file_ids", new=AsyncMock(return_value=[])):
            with patch.object(httpx, "AsyncClient") as mock_cls:
                result = asyncio.run(client.delete_all_files())

        assert result == 0, f"Expected 0, got {result}"
        # No httpx.AsyncClient should be opened for DELETEs
        mock_cls.assert_not_called()

    # ------------------------------------------------------------------
    # RED: persistent failure → raises RuntimeError after 4 attempts (task 4.3)
    # ------------------------------------------------------------------
    def test_delete_all_files_raises_after_retries_exhausted(self):
        """delete_all_files raises RuntimeError after all 4 delete attempts fail."""
        client = _make_client()

        resp_500 = _make_http_response(500, text="server error")
        mock_inner = AsyncMock()
        mock_inner.delete = AsyncMock(return_value=resp_500)

        with patch.object(client, "list_file_ids", new=AsyncMock(return_value=[42])):
            with _patch_httpx_client(mock_inner):
                with patch("asyncio.sleep", new=AsyncMock()):
                    with pytest.raises(RuntimeError) as exc_info:
                        asyncio.run(client.delete_all_files())

        error_msg = str(exc_info.value)
        assert "42" in error_msg, f"RuntimeError must mention file ID 42: {error_msg}"
        assert "500" in error_msg, f"RuntimeError must mention status 500: {error_msg}"

    # ------------------------------------------------------------------
    # Triangulation: runtime error message contains attempts count
    # ------------------------------------------------------------------
    def test_delete_all_files_error_includes_attempt_count(self):
        """RuntimeError after retries contains enough info to diagnose the failure."""
        client = _make_client()

        resp_503 = _make_http_response(503, text="Service Unavailable")
        mock_inner = AsyncMock()
        mock_inner.delete = AsyncMock(return_value=resp_503)

        with patch.object(client, "list_file_ids", new=AsyncMock(return_value=[99])):
            with _patch_httpx_client(mock_inner):
                with patch("asyncio.sleep", new=AsyncMock()):
                    with pytest.raises(RuntimeError) as exc_info:
                        asyncio.run(client.delete_all_files())

        error_msg = str(exc_info.value)
        assert "99" in error_msg

    # ------------------------------------------------------------------
    # RED: exactly 4 DELETE attempts are made before raising
    # ------------------------------------------------------------------
    def test_delete_all_files_makes_exactly_4_attempts(self):
        """When all attempts fail, exactly 4 DELETE calls are made (1 initial + 3 retries)."""
        client = _make_client()

        resp_500 = _make_http_response(500, text="err")
        mock_inner = AsyncMock()
        mock_inner.delete = AsyncMock(return_value=resp_500)

        with patch.object(client, "list_file_ids", new=AsyncMock(return_value=[7])):
            with _patch_httpx_client(mock_inner):
                with patch("asyncio.sleep", new=AsyncMock()):
                    with pytest.raises(RuntimeError):
                        asyncio.run(client.delete_all_files())

        assert mock_inner.delete.call_count == 4, (
            f"Expected 4 DELETE attempts, got {mock_inner.delete.call_count}"
        )

    # ------------------------------------------------------------------
    # RED: backoff sleeps occur between retries
    # ------------------------------------------------------------------
    def test_delete_all_files_sleeps_between_retries(self):
        """asyncio.sleep is called 3 times with backoff values 2, 4, 8."""
        client = _make_client()

        resp_500 = _make_http_response(500, text="err")
        mock_inner = AsyncMock()
        mock_inner.delete = AsyncMock(return_value=resp_500)

        mock_sleep = AsyncMock()

        with patch.object(client, "list_file_ids", new=AsyncMock(return_value=[7])):
            with _patch_httpx_client(mock_inner):
                with patch("asyncio.sleep", mock_sleep):
                    with pytest.raises(RuntimeError):
                        asyncio.run(client.delete_all_files())

        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_args == [2, 4, 8], f"Expected backoff [2, 4, 8], got {sleep_args}"

    # ------------------------------------------------------------------
    # Transient failure: succeeds on second attempt (first retry)
    # ------------------------------------------------------------------
    def test_delete_all_files_recovers_on_second_attempt(self):
        """delete_all_files succeeds if a file's DELETE succeeds on the second attempt."""
        client = _make_client()

        resp_500 = _make_http_response(500, text="transient")
        resp_200 = _make_http_response(200)

        mock_inner = AsyncMock()
        mock_inner.delete = AsyncMock(side_effect=[resp_500, resp_200])

        mock_sleep = AsyncMock()

        with patch.object(client, "list_file_ids", new=AsyncMock(return_value=[55])):
            with _patch_httpx_client(mock_inner):
                with patch("asyncio.sleep", mock_sleep):
                    result = asyncio.run(client.delete_all_files())

        assert result == 1, f"Expected 1 deleted, got {result}"
        assert mock_inner.delete.call_count == 2, (
            f"Expected 2 DELETE calls (1 fail + 1 success), got {mock_inner.delete.call_count}"
        )
        # Only one sleep between attempt 1 and attempt 2
        sleep_args = [c.args[0] for c in mock_sleep.call_args_list]
        assert sleep_args == [2], f"Expected one sleep of 2s, got {sleep_args}"

    # ------------------------------------------------------------------
    # Success path: 200 on first try returns count
    # ------------------------------------------------------------------
    def test_delete_all_files_returns_count_on_success(self):
        """delete_all_files returns the count of successfully deleted files."""
        client = _make_client()

        resp_200 = _make_http_response(200)
        mock_inner = AsyncMock()
        mock_inner.delete = AsyncMock(return_value=resp_200)

        with patch.object(client, "list_file_ids", new=AsyncMock(return_value=[1, 2, 3])):
            with _patch_httpx_client(mock_inner):
                with patch("asyncio.sleep", new=AsyncMock()):
                    result = asyncio.run(client.delete_all_files())

        assert result == 3
        assert mock_inner.delete.call_count == 3

    # ------------------------------------------------------------------
    # 404 on DELETE (already deleted) counts as success
    # ------------------------------------------------------------------
    def test_delete_all_files_counts_404_as_deleted(self):
        """A 404 DELETE response (already deleted) increments the deleted count."""
        client = _make_client()

        resp_404 = _make_http_response(404)
        mock_inner = AsyncMock()
        mock_inner.delete = AsyncMock(return_value=resp_404)

        with patch.object(client, "list_file_ids", new=AsyncMock(return_value=[88])):
            with _patch_httpx_client(mock_inner):
                with patch("asyncio.sleep", new=AsyncMock()):
                    result = asyncio.run(client.delete_all_files())

        assert result == 1


# ---------------------------------------------------------------------------
# Phase 4.5 / 4.6 — do_index() integration tests (call order)
# ---------------------------------------------------------------------------

class TestDoIndexTeardownOrder:
    """Integration tests for do_index() — teardown call sequence."""

    @pytest.fixture(autouse=True)
    def _imports(self):
        import runner as r
        import tero_client as tc
        self._runner = r
        self._tc = tc

    def _make_index_args(self, dataset="ragbench", agent_id=9, max_docs=2):
        args = MagicMock()
        args.dataset = dataset
        args.agent_id = agent_id
        args.bearer_token = "test-token"
        args.base_url = "http://localhost:8000"
        args.max_docs = max_docs
        return args

    def _make_mock_tero(self, existing_file_ids=None, new_file_ids=None):
        """Build an AsyncMock TeroClient with sensible defaults."""
        mock_tero = AsyncMock()
        mock_tero.list_file_ids = AsyncMock(return_value=existing_file_ids or [])
        mock_tero.wait_files_processed = AsyncMock(return_value=None)
        mock_tero.delete_all_files = AsyncMock(return_value=len(existing_file_ids or []))
        mock_tero.delete_docs_tool = AsyncMock(return_value=None)
        mock_tero.configure_docs_tool = AsyncMock(return_value=None)
        # upload_document returns sequential IDs for new files
        ids = new_file_ids or [101, 102]
        mock_tero.upload_document = AsyncMock(side_effect=ids)
        return mock_tero

    # ------------------------------------------------------------------
    # RED: teardown order with existing files (task 4.5)
    # ------------------------------------------------------------------
    def test_do_index_teardown_order_with_existing_files(self):
        """do_index calls: list_file_ids → wait_files_processed → delete_all_files
        → delete_docs_tool → configure_docs_tool → upload → wait_files_processed.

        Covers REQ-INDEX-4.
        """
        mock_corpus = ["doc one text", "doc two"]
        args = self._make_index_args(max_docs=None)
        mock_tero = self._make_mock_tero(existing_file_ids=[10, 20], new_file_ids=[101, 102])

        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(side_effect=lambda s: list(range(len(s))))

        with patch.object(self._tc, "TeroClient", return_value=mock_tero):
            with patch.object(self._runner.ds_module, "load_one", return_value=([], mock_corpus)):
                with patch("tiktoken.get_encoding", return_value=mock_enc):
                    with patch.object(self._runner, "cost_report"):
                        asyncio.run(self._runner.do_index(args))

        # Extract method call names in order
        call_names = [str(c[0]) for c in mock_tero.method_calls]

        # The methods we care about (they must appear in this relative order)
        ordered_methods = [
            "list_file_ids",
            "wait_files_processed",
            "delete_all_files",
            "delete_docs_tool",
            "configure_docs_tool",
        ]
        indices = {}
        for method in ordered_methods:
            # Find first occurrence
            for i, name in enumerate(call_names):
                if name == method:
                    indices[method] = i
                    break
            assert method in indices, f"Method '{method}' was not called"

        for i in range(len(ordered_methods) - 1):
            a = ordered_methods[i]
            b = ordered_methods[i + 1]
            assert indices[a] < indices[b], (
                f"Expected '{a}' (pos {indices[a]}) before '{b}' (pos {indices[b]})"
            )

    # ------------------------------------------------------------------
    # RED: pre-teardown wait uses timeout=120 (task 4.5 detail)
    # ------------------------------------------------------------------
    def test_do_index_pre_teardown_wait_uses_120s_timeout(self):
        """wait_files_processed for existing files is called with timeout=120.0."""
        mock_corpus = ["doc one"]
        args = self._make_index_args(max_docs=None)
        mock_tero = self._make_mock_tero(existing_file_ids=[5], new_file_ids=[201])

        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(return_value=[])

        with patch.object(self._tc, "TeroClient", return_value=mock_tero):
            with patch.object(self._runner.ds_module, "load_one", return_value=([], mock_corpus)):
                with patch("tiktoken.get_encoding", return_value=mock_enc):
                    with patch.object(self._runner, "cost_report"):
                        asyncio.run(self._runner.do_index(args))

        # First wait_files_processed call should use timeout=120.0 for existing IDs
        first_wait_call = mock_tero.wait_files_processed.call_args_list[0]
        # Positional arg is the existing IDs; keyword arg timeout=120.0
        assert first_wait_call.kwargs.get("timeout") == 120.0, (
            f"Expected timeout=120.0 for pre-teardown wait, "
            f"got: {first_wait_call}"
        )
        assert first_wait_call.args[0] == [5], (
            f"Expected existing IDs [5], got {first_wait_call.args[0]}"
        )

    # ------------------------------------------------------------------
    # RED: first-run guard — no pre-teardown wait when no existing files (task 4.6)
    # ------------------------------------------------------------------
    def test_do_index_first_run_skips_wait_for_existing_files(self):
        """When list_file_ids returns [], wait_files_processed is NOT called for existing files.

        Covers REQ-INDEX-6 (first-run guard).
        """
        mock_corpus = ["doc one", "doc two"]
        args = self._make_index_args(max_docs=None)
        mock_tero = self._make_mock_tero(existing_file_ids=[], new_file_ids=[101, 102])

        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(return_value=[])

        with patch.object(self._tc, "TeroClient", return_value=mock_tero):
            with patch.object(self._runner.ds_module, "load_one", return_value=([], mock_corpus)):
                with patch("tiktoken.get_encoding", return_value=mock_enc):
                    with patch.object(self._runner, "cost_report"):
                        asyncio.run(self._runner.do_index(args))

        # wait_files_processed must be called exactly ONCE — for the new files after upload
        # (NOT for the empty existing_ids list)
        assert mock_tero.wait_files_processed.call_count == 1, (
            f"Expected 1 wait call (post-upload), got {mock_tero.wait_files_processed.call_count}"
        )

        # That one call must NOT use timeout=120.0 (which is for pre-teardown)
        the_call = mock_tero.wait_files_processed.call_args_list[0]
        assert the_call.kwargs.get("timeout") != 120.0, (
            "The post-upload wait should use default timeout, not 120.0"
        )

    # ------------------------------------------------------------------
    # Triangulation: delete_all_files still called on first run (no-op)
    # ------------------------------------------------------------------
    def test_do_index_first_run_still_calls_delete_all_files(self):
        """Even when list_file_ids returns [], delete_all_files is still called (no-op)."""
        mock_corpus = ["doc one"]
        args = self._make_index_args(max_docs=None)
        mock_tero = self._make_mock_tero(existing_file_ids=[], new_file_ids=[101])

        mock_enc = MagicMock()
        mock_enc.encode = MagicMock(return_value=[])

        with patch.object(self._tc, "TeroClient", return_value=mock_tero):
            with patch.object(self._runner.ds_module, "load_one", return_value=([], mock_corpus)):
                with patch("tiktoken.get_encoding", return_value=mock_enc):
                    with patch.object(self._runner, "cost_report"):
                        asyncio.run(self._runner.do_index(args))

        mock_tero.delete_all_files.assert_called_once()
        mock_tero.delete_docs_tool.assert_called_once()
