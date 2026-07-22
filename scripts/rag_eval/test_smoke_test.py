"""
Tests for smoke_test.py — agent resolution and corpus upload without JSON state.
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            message=f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


def _patch_httpx_client(mock_inner_client):
    """Patch httpx.AsyncClient so ``async with`` yields *mock_inner_client*."""
    import httpx
    mock_ctx = AsyncMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_inner_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_cls = MagicMock(return_value=mock_ctx)
    return patch.object(httpx, "AsyncClient", mock_cls)


def _run(coro):
    """Run an async coroutine in the event loop."""
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# TASK-2.2 — inlined resolve_eval_agent
# ---------------------------------------------------------------------------

class TestResolveEvalAgent:
    """Tests for the inlined agent resolution helper."""

    @pytest.fixture(autouse=True)
    def _imports(self):
        import smoke_test as st
        self._st = st

    def test_reads_existing_agent_id(self, tmp_path: Path):
        """If eval_agent.json exists, return its id without any HTTP calls."""
        state_file = tmp_path / "eval_agent.json"
        state_file.write_text(json.dumps({"agent_id": 42}))

        with patch("httpx.AsyncClient") as mock_client_cls:
            result = _run(self._st._resolve_eval_agent(
                base_url="http://localhost:8000",
                bearer_token="test-token",
                evals_dir=tmp_path,
            ))

        assert result == 42
        mock_client_cls.assert_not_called()

    def test_creates_agent_when_state_missing(self, tmp_path: Path):
        """When eval_agent.json is absent, create via POST and persist the id."""
        mock_inner = AsyncMock()
        mock_inner.post = AsyncMock(return_value=_make_http_response(200, {"id": 99}))
        mock_inner.put = AsyncMock(return_value=_make_http_response(200, {}))

        with _patch_httpx_client(mock_inner):
            result = _run(self._st._resolve_eval_agent(
                base_url="http://localhost:8000",
                bearer_token="test-token",
                evals_dir=tmp_path,
            ))

        assert result == 99
        state_file = tmp_path / "eval_agent.json"
        assert state_file.exists()
        assert json.loads(state_file.read_text()) == {"agent_id": 99}

        mock_inner.post.assert_called_once_with("http://localhost:8000/api/agents")
        mock_inner.put.assert_called_once_with(
            "http://localhost:8000/api/agents/99",
            json={"name": "RAG Evaluation Agent"},
        )


# ---------------------------------------------------------------------------
# TASK-2.3 / 2.4 — corpus upload uses list_file_ids, export_dataset, no JSON state
# ---------------------------------------------------------------------------

class TestUploadCorpus:
    """Tests for _upload_corpus using server state and shared export helper."""

    @pytest.fixture(autouse=True)
    def _imports(self):
        import smoke_test as st
        self._st = st

    def _make_tero(self, existing_file_ids=None, new_file_ids=None):
        mock = AsyncMock()
        mock.configure_docs_tool = AsyncMock()
        mock.list_file_ids = AsyncMock(return_value=existing_file_ids or [])
        mock.delete_all_files = AsyncMock(return_value=len(existing_file_ids or []))
        mock.upload_document = AsyncMock(side_effect=new_file_ids or [])
        mock.wait_files_processed = AsyncMock()
        return mock

    def test_skips_upload_when_files_already_present(self, tmp_path: Path):
        """If list_file_ids returns files and not rebuilding, upload is skipped."""
        tero = self._make_tero(existing_file_ids=[1, 2, 3])

        with patch.object(self._st, "export_dataset") as mock_export:
            result = _run(self._st._upload_corpus(
                tero=tero,
                agent_id=7,
                dataset="ragbench",
                rebuild=False,
                corpus_size=2,
            ))

        assert result == [1, 2, 3]
        tero.list_file_ids.assert_called_once()
        tero.delete_all_files.assert_not_called()
        mock_export.assert_not_called()
        tero.upload_document.assert_not_called()
        tero.configure_docs_tool.assert_called_once()
        assert not (tmp_path / "corpus_state_7.json").exists()

    def test_rebuild_deletes_existing_files_before_upload(self, tmp_path: Path):
        """With rebuild=True, existing files are deleted before export/upload."""
        tero = self._make_tero(existing_file_ids=[1, 2], new_file_ids=[10, 11])

        with patch.object(self._st, "export_dataset", return_value=([], ["doc a", "doc b"])) as mock_export:
            result = _run(self._st._upload_corpus(
                tero=tero,
                agent_id=7,
                dataset="ragbench",
                rebuild=True,
                corpus_size=2,
            ))

        tero.delete_all_files.assert_called_once()
        mock_export.assert_called_once_with("ragbench", n=10, corpus_size=2)
        assert result == [10, 11]

    def test_fresh_upload_exports_and_uploads_documents(self, tmp_path: Path):
        """Empty agent exports corpus via export_dataset and uploads returned docs."""
        tero = self._make_tero(existing_file_ids=[], new_file_ids=[101, 102])

        with patch.object(self._st, "export_dataset", return_value=([], ["doc a", "doc b"])) as mock_export:
            result = _run(self._st._upload_corpus(
                tero=tero,
                agent_id=7,
                dataset="ragbench",
                rebuild=False,
                corpus_size=2,
            ))

        mock_export.assert_called_once_with("ragbench", n=10, corpus_size=2)
        # smoke_test no longer calls load_one itself — export_dataset is the single loader
        assert tero.upload_document.call_count == 2
        tero.wait_files_processed.assert_called_once_with([101, 102])
        assert result == [101, 102]

        # CRITICAL: no per-agent corpus_state JSON is written
        assert not (tmp_path / "corpus_state_7.json").exists()
        assert not list(tmp_path.glob("corpus_state_*.json"))


# ---------------------------------------------------------------------------
# CLI / import smoke
# ---------------------------------------------------------------------------

def test_smoke_test_imports_without_tero_client_helpers():
    """smoke_test no longer depends on tero_client.resolve_eval_agent."""
    import smoke_test as st
    assert hasattr(st, "_resolve_eval_agent")
    assert hasattr(st, "_upload_corpus")
    assert not hasattr(st, "resolve_eval_agent")
