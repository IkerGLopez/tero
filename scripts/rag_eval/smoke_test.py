"""
Smoke test for the RAG pipeline — no judge LLM, no RAGAS metrics.

Verifies that:
  1. The eval agent exists and responds.
  2. The docs tool is active on the agent.
  3. A question returns non-empty retrieved_contexts.

Usage:
  python scripts/rag_eval/smoke_test.py --bearer-token <JWT> [--dataset fetaqa]

  # Upload corpus for a dataset without running any evaluation:
  python scripts/rag_eval/smoke_test.py --bearer-token <JWT> --dataset fetaqa --upload-corpus

  # Force re-upload even if corpus was already uploaded:
  python scripts/rag_eval/smoke_test.py --bearer-token <JWT> --dataset fetaqa --upload-corpus --rebuild

  # Test with a custom question against whatever corpus is already loaded:
  python scripts/rag_eval/smoke_test.py --bearer-token <JWT> --question "What is X?"

  # Env vars are loaded automatically from .env at the repo root.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

SCRIPT_DIR = Path(__file__).parent
EVALS_DIR = SCRIPT_DIR / "evals"
CORPUS_DIR = EVALS_DIR / "corpus"

sys.path.insert(0, str(SCRIPT_DIR))
from tero_client import TeroClient, resolve_eval_agent
import rag_datasets as ds_module

# One representative question per dataset, to avoid loading the full HF dataset.
# Questions must be specific enough to trigger the docs tool — vague questions
# cause the model to ask for clarification instead of searching.
_DEFAULT_QUESTIONS = {
    "ragbench": "What security vulnerabilities are described in the bulletin?",
    "fetaqa": "How many episodes does the show Fargo have?",
    "stratrag": "What does the document say about retrieval augmented generation?",
}


def _load_corpus_state(agent_id: int) -> dict:
    path = EVALS_DIR / f"corpus_state_{agent_id}.json"
    if path.exists():
        state = json.loads(path.read_text())
        # Normalize legacy list format to dict format
        for ds in list(state.keys()):
            if isinstance(state[ds], list):
                state[ds] = {"file_ids": state[ds], "corpus_size": len(state[ds])}
        return state
    return {}


def _save_corpus_state(agent_id: int, state: dict) -> None:
    EVALS_DIR.mkdir(parents=True, exist_ok=True)
    path = EVALS_DIR / f"corpus_state_{agent_id}.json"
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(state, indent=2))
    os.replace(tmp_path, path)


async def _upload_corpus(tero: TeroClient, agent_id: int, dataset: str, rebuild: bool, corpus_size: int = 10) -> list[int]:
    import httpx

    state = _load_corpus_state(agent_id)
    await tero.configure_docs_tool()

    if rebuild and dataset in state:
        # Clean old documents from the agent before uploading the new corpus.
        print("Cleaning old documents from agent before rebuild...")
        await tero.delete_all_files()
        del state[dataset]

    if not rebuild and dataset in state:
        cached = state[dataset]
        # Normalized to dict format by _load_corpus_state
        file_count = len(cached["file_ids"]) if isinstance(cached, dict) else len(cached)
        print(f"Corpus for '{dataset}' already uploaded ({file_count} files). Use --rebuild to re-upload.")
        return cached["file_ids"] if isinstance(cached, dict) else cached

    _, corpus = ds_module.load_one(dataset, n=10)
    corpus = corpus[:corpus_size]  # Limit locally instead of relying on loader parameter
    corpus_path = CORPUS_DIR / dataset
    corpus_path.mkdir(parents=True, exist_ok=True)

    print(f"Uploading {len(corpus)} documents for '{dataset}'...")
    file_ids = []
    for i, doc_text in enumerate(corpus):
        filename = f"doc_{i:04d}.txt"
        filepath = corpus_path / filename
        filepath.write_text(doc_text, encoding="utf-8")
        file_id = await tero.upload_document(filename, doc_text.encode("utf-8"))
        file_ids.append(file_id)
        print(f"  Uploaded {filename} (id={file_id})")

    print("Waiting for document processing...")
    await tero.wait_files_processed(file_ids)
    print("All documents processed.")

    state[dataset] = {"file_ids": file_ids, "corpus_size": len(file_ids)}
    _save_corpus_state(agent_id, state)
    return file_ids


async def run(args: argparse.Namespace) -> None:
    bearer_token = (args.bearer_token or os.environ.get("BEARER_TOKEN", "")).strip()
    if not bearer_token:
        print("ERROR: --bearer-token is required (or set BEARER_TOKEN env var).", file=sys.stderr)
        sys.exit(1)

    base_url = args.base_url
    if args.agent_id is not None:
        agent_id = args.agent_id
        print(f"Using agent ID from --agent-id: {agent_id}")
    else:
        agent_id = await resolve_eval_agent(base_url, bearer_token, EVALS_DIR)
    tero = TeroClient(base_url, agent_id, bearer_token)

    if args.upload_corpus:
        await _upload_corpus(tero, agent_id, args.dataset, rebuild=args.rebuild, corpus_size=args.corpus_size)
        print(f"\nCorpus for '{args.dataset}' ready. Run without --upload-corpus to test retrieval.")
        return

    question = args.question
    if not question:
        question = _DEFAULT_QUESTIONS.get(args.dataset, "What is the main topic of the documents?")

    print(f"\nAgent ID : {agent_id}")
    print(f"Dataset  : {args.dataset}")
    print(f"Question : {question}\n")

    thread_id = await tero.create_thread()
    result = await tero.ask_question(thread_id, question)

    n_ctx = len(result["retrieved_contexts"])
    status = "OK" if n_ctx > 0 else "FAIL — retrieved_contexts is empty"

    print(f"Status            : {status}")
    print(f"Retrieved contexts: {n_ctx}")
    print(f"Latency           : {result['latency_ms']} ms")
    print(f"\nAnswer:\n{result['answer_text']}\n")

    if n_ctx > 0:
        print("First retrieved context (truncated to 300 chars):")
        print(result["retrieved_contexts"][0][:300])
    else:
        print(
            "\nDiagnosis: the agent responded without using any document.\n"
            "  - Check that the corpus was uploaded for this dataset (corpus_state_<agent_id>.json).\n"
            "  - Re-run runner.py with --rebuild-corpus to force re-upload.\n"
            "  - Verify the docs tool is enabled on the agent in the Tero UI."
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the Tero RAG pipeline.")
    parser.add_argument("--bearer-token", default=None)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--agent-id",
        type=int,
        default=None,
        help="Tero agent ID to use. If omitted, reads from evals/eval_agent.json (or creates a new agent).",
    )
    parser.add_argument(
        "--dataset",
        choices=ds_module.ALL_DATASETS,
        default="ragbench",
        help="Dataset whose corpus should be loaded on the agent (default: ragbench).",
    )
    parser.add_argument(
        "--question",
        default=None,
        help="Custom question to ask. Overrides the default question for the dataset.",
    )
    parser.add_argument(
        "--upload-corpus",
        action="store_true",
        help="Upload the dataset corpus to the agent and exit without running any question.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force re-upload even if the corpus was already uploaded (use with --upload-corpus).",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=10,
        help="Max corpus documents to upload (default: 10, small for smoke tests).",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
