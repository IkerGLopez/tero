"""Shared bootstrap, paths and constants for the rag_eval scripts.

Paths are anchored to this file so every tool works from any cwd; importing
this module also loads `.env` from the repo root.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (two levels up from scripts/rag_eval/)
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

# Resolve paths relative to this file so the script works from any cwd
SCRIPT_DIR = Path(__file__).parent
EVALS_DIR = SCRIPT_DIR / "evals"
BASELINE_DIR = EVALS_DIR / "baseline"
EXPERIMENTS_DIR = EVALS_DIR / "experiments"

# Pool probe (B2): the embedding cache lives under the gitignored evals/ tree
# so embeddings are never committed.
DEFAULT_PROBE_CACHE_DIR = EVALS_DIR / "pool_probe_cache"

# Comparability protocol (D7): eval runs measure the docs tool's default top-5
# retention (rerank off), so every baseline records the same top_k.
EVAL_TOP_K = 5

# Reranker banner: keep the default scoring model in sync with
# Settings.set_defaults in src/backend/tero/core/env.py.
_RERANKER_TRUTHY = {"true", "1", "yes", "on"}


def _reranker_state_line() -> str:
    """One-line reranker state for the eval console.

    Mirrors the backend's DOCS_TOOL_RERANK* settings as loaded from .env; the
    backend only applies changes after its container is recreated.
    """
    enabled = (os.environ.get("DOCS_TOOL_RERANK") or "").strip().lower() in _RERANKER_TRUTHY
    if not enabled:
        return (
            "[reranker] OFF | DOCS_TOOL_RERANK is not enabled — retrieval returns "
            "DOCS_TOOL_RETRIEVE_TOP chunks directly"
        )
    fetch_k = (os.environ.get("DOCS_TOOL_RERANK_FETCH_K") or "").strip() or "20"
    top_n = (os.environ.get("DOCS_TOOL_RERANK_TOP_N") or "").strip() or "5"
    model = (os.environ.get("DOCS_TOOL_RERANK_MODEL") or "").strip() or "gpt-6-luna"
    return (
        f"[reranker] ON | DOCS_TOOL_RERANK=true | fetch_k={fetch_k} | top_n={top_n} | "
        f"model={model} — recreate the backend container to apply"
    )
