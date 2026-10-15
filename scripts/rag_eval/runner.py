from __future__ import annotations

"""
RAG evaluation runner for Tero using RAGAS.

Judge LLM: GPT-4o, Gemini, or Claude via AWS Bedrock
Models:     Any Tero model ID, passed via --models

Separates indexing from evaluation into two subcommands.

Usage:
  # Index a dataset into an agent (one-time, no LLM description cost)
  python scripts/rag_eval/runner.py index --dataset ragbench --agent-id 9 --max-docs 500

  # Evaluate against a pre-indexed agent (repeatable, no uploads)
  python scripts/rag_eval/runner.py eval --dataset ragbench --agent-id 9 --models gpt-5 --max-questions 30

  # Save and compare baselines
  python scripts/rag_eval/runner.py eval --dataset ragbench --agent-id 9 --models gpt-5 --update-baseline
  python scripts/rag_eval/runner.py eval --dataset ragbench --agent-id 9 --models gpt-5 --compare

  # Offline CSV evaluation
  python scripts/rag_eval/runner.py eval --dataset ragbench --agent-id 9 --from-csv path/to/file.csv

  # Offline dense-pool gate (Fase 2: is the FeTaQA gold document reachable at depth?)
  python scripts/rag_eval/runner.py pool-probe --questions 100 --seed 14

  # Env vars are loaded automatically from .env at the repo root.
  # Required in .env: GOOGLE_API_KEY
  # Required via CLI: --bearer-token (JWT expires, easier to pass each time)
"""

import argparse
import asyncio

# Shared bootstrap (repo-root .env, paths, constants) - importing config.py
# also inserts the script directory on sys.path, so the imports below resolve
# from any entry point; config imports nothing from this module.
from config import DEFAULT_PROBE_CACHE_DIR

from rag_datasets import ALL_DATASETS

# T6: the subcommand bodies (index, eval live, eval CSV and pool probe) now
# live in the sibling flows/ package; runner keeps the CLI (parser plus
# dispatch) and re-exports the flow entry points, so patch.object(runner,
# do_index/do_eval/do_pool_probe) still intercepts the dispatch calls.
from flows.index import do_index
from flows.eval_live import do_eval
from flows.probe import do_pool_probe


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the index/eval/pool-probe subcommands (test seam)."""
    parser = argparse.ArgumentParser(description="Tero RAG evaluation — index, eval and pool-probe subcommands.")
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Subcommand: index, eval or pool-probe")

    # --- index subcommand ---
    index_parser = subparsers.add_parser("index", help="Index a dataset corpus into an agent")
    index_parser.add_argument("--dataset", required=True, choices=ALL_DATASETS,
                              help="Dataset to index (ragbench, fetaqa, stratrag)")
    index_parser.add_argument("--agent-id", type=int, default=None,
                              help="Tero agent ID to index into. Falls back to RAG_EVAL_AGENT_ID env var.")
    index_parser.add_argument("--max-docs", type=int, default=None,
                              help="Maximum number of documents to index (default: all)")
    index_parser.add_argument("--bearer-token", default=None,
                              help="Tero JWT bearer token. Falls back to BEARER_TOKEN env var.")
    index_parser.add_argument("--base-url", default="http://localhost:8000",
                              help="Tero API base URL")

    # --- eval subcommand ---
    eval_parser = subparsers.add_parser("eval", help="Evaluate a pre-indexed agent with RAGAS")
    eval_parser.add_argument("--dataset", required=True, choices=ALL_DATASETS,
                             help="Dataset to evaluate (ragbench, fetaqa, stratrag)")
    eval_parser.add_argument("--agent-id", type=int, default=None,
                             help="Tero agent ID to evaluate (must be pre-indexed; not needed with --from-csv)")
    eval_parser.add_argument("--max-questions", type=int, default=1,
                             help="Number of questions to evaluate (default: 1)")
    eval_parser.add_argument("--max-docs", type=int, default=None,
                             help="Maximum number of indexed documents to assume (mirrors index "
                                  "--max-docs; gold beyond this prefix counts as out-of-corpus. "
                                  "Default: full corpus)")
    eval_parser.add_argument("--models", default=None,
                             help="Comma-separated Tero model IDs to evaluate (e.g. gpt-5,claude-sonnet-4)")
    eval_parser.add_argument("--bearer-token", default=None,
                             help="Tero JWT bearer token. Falls back to BEARER_TOKEN env var.")
    eval_parser.add_argument("--base-url", default="http://localhost:8000",
                             help="Tero API base URL")
    eval_parser.add_argument("--from-csv", default=None,
                             help="CSV file path for offline RAG evaluation (skips Tero HTTP calls)")
    eval_parser.add_argument("--update-baseline", action="store_true",
                             help="Save current results as new baseline")
    eval_parser.add_argument("--compare", action="store_true",
                             help="Compare results against existing baseline")
    eval_parser.add_argument("--seed", type=int, default=14,
                             help="Random seed for deterministic question selection")
    eval_parser.add_argument("--judge-model", default="gemini-3.5-flash",
                             help="Judge LLM for RAGAS metrics. Vendor auto-detected from prefix: "
                                  "claude* → Anthropic, gpt*/o1*/o3*/o4* → OpenAI, gemini* → Google. "
                                  "(default: gemini-3.5-flash)")
    eval_parser.add_argument("--concurrency", type=int, default=5,
                             help="Maximum concurrent questions sent to the Tero agent. "
                                  "Prevents LLM API rate-limiting and DB pool exhaustion. "
                                  "(default: 5)")

    # --- pool-probe subcommand (B2) ---
    probe_parser = subparsers.add_parser(
        "pool-probe",
        help="Offline dense-pool probe: is the gold document reachable at pool depths?",
    )
    probe_parser.add_argument("--dataset", required=True, choices=ALL_DATASETS,
                              help="Dataset to probe (ragbench, fetaqa, stratrag)")
    probe_parser.add_argument("--questions", type=int, required=True,
                              help="Questions to probe — the gate's sample size "
                                   "(must be explicit; the corpus is embedded once and cached)")
    probe_parser.add_argument("--seed", type=int, default=14,
                              help="Random seed for deterministic question selection (default: 14)")
    probe_parser.add_argument("--embedding-model", default="text-embedding-3-small",
                              help="Embedding model — must match the indexed corpus "
                                   "(default: text-embedding-3-small)")
    probe_parser.add_argument("--max-docs", type=int, default=None,
                              help="Probed corpus prefix, mirrors the indexed run "
                                   "(default: full corpus)")
    probe_parser.add_argument("--depths", type=int, nargs="+", default=[20, 50, 100, 500],
                              help="Pool depths to report gold-in-pool rates for "
                                   "(default: 20 50 100 500)")
    probe_parser.add_argument("--cache-dir", default=str(DEFAULT_PROBE_CACHE_DIR),
                              help="Embedding cache directory (default: gitignored "
                                   "evals/pool_probe_cache)")

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "index":
        asyncio.run(do_index(args))
    elif args.command == "eval":
        asyncio.run(do_eval(args))
    elif args.command == "pool-probe":
        do_pool_probe(args)


if __name__ == "__main__":
    main()
