from __future__ import annotations

"""
RAG evaluation runner for Tero using RAGAS.

Judge LLM: Gemini 2.5 Flash (requires GOOGLE_API_KEY env var)
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

  # Env vars are loaded automatically from .env at the repo root.
  # Required in .env: GOOGLE_API_KEY
  # Required via CLI: --bearer-token (JWT expires, easier to pass each time)
"""

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from uuid import uuid4
import pandas as pd
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root (two levels up from scripts/rag_eval/)
_REPO_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_REPO_ROOT / ".env")

from openai import AsyncOpenAI
from ragas import SingleTurnSample, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness

# Resolve paths relative to this file so the script works from any cwd
SCRIPT_DIR = Path(__file__).parent
EVALS_DIR = SCRIPT_DIR / "evals"
BASELINE_DIR = EVALS_DIR / "baseline"
EXPERIMENTS_DIR = EVALS_DIR / "experiments"

# Fixed agent IDs per dataset — each dataset has its own isolated corpus index.
DATASET_AGENT_IDS: dict[str, int] = {
    "ragbench": 9,
    "stratrag": 8,
    "fetaqa": 7,
}

sys.path.insert(0, str(SCRIPT_DIR))
import rag_datasets as ds_module
from rag_datasets import ALL_DATASETS
# NOTE: TeroClient import is lazy — only imported in do_eval() live path
# to satisfy REQ-OFFLINE-009: zero Tero dependency in CSV mode.
import analysis


# ------------------------------------------------------------------
# RAGAS metrics
# ------------------------------------------------------------------

def _build_metrics(llm):
    context_recall = LLMContextRecall(llm=llm)
    context_precision = ContextPrecision(llm=llm)
    faithfulness = Faithfulness(llm=llm)

    correctness = DiscreteMetric(
        name="correctness",
        prompt=(
            "Rate the response against the grading notes using this scale:\n"
            "0: incorrect, irrelevant, or covers less than 20% of the expected key claims.\n"
            "1: contains between 20% and 50% of the expected key claims.\n"
            "2: contains between 50% and 70% of the expected key claims.\n"
            "3: contains between 70% and 90% of the expected key claims, minor omissions.\n"
            "4: contains more than 90% of the expected key claims, no relevant errors.\n"
            "Return only the number (0, 1, 2, 3, or 4).\n"
            "Response: {response}\n"
            "Grading Notes: {grading_notes}"
        ),
        allowed_values=["0", "1", "2", "3", "4"],
    )

    citation_faithfulness = DiscreteMetric(
        name="citation_faithfulness",
        prompt=(
            "A RAG system answered a question by citing a document chunk. "
            "Determine if the cited chunk actually supports the claims made in the answer.\n"
            "Answer: {response}\n"
            "Cited chunk: {cited_chunk}\n"
            "Return 'supported' if the chunk backs the answer, 'unsupported' otherwise."
        ),
        allowed_values=["supported", "unsupported"],
    )

    return context_recall, context_precision, faithfulness, correctness, citation_faithfulness


# ------------------------------------------------------------------
# Row evaluation helpers (extracted for testability)
# ------------------------------------------------------------------

def _pair_citations_with_contexts(
    citations: list[str],
    contexts: list[str],
) -> list[tuple[str, str]]:
    """Pair citations with retrieved contexts via chunk_N markers.

    Returns empty list when no valid chunk_N markers found.
    No positional fallback — only chunk_N index matching is used.
    """
    _CHUNK_IDX_RE = re.compile(r"chunk_(\d+)")
    pairs: list[tuple[str, str]] = []
    seen: set[int] = set()
    for cite in citations:
        m = _CHUNK_IDX_RE.search(cite)
        if m:
            idx = int(m.group(1)) - 1  # chunk_1 → contexts[0]
            if 0 <= idx < len(contexts) and idx not in seen:
                pairs.append((cite, contexts[idx]))
                seen.add(idx)
    return pairs


def _find_relevant_chunk_index(grading_notes: str, contexts: list[str]) -> int:
    """Return 1-based index of first context with 10+ char substring overlap with grading_notes.

    Returns -1 if grading_notes < 10 chars or no match found.
    Match is case-insensitive.
    """
    if len(grading_notes) < 10:
        return -1

    grading_lower = grading_notes.lower()
    substrings = [grading_lower[i:i + 10] for i in range(len(grading_lower) - 9)]

    for idx, ctx in enumerate(contexts):
        ctx_lower = ctx.lower()
        for chunk in substrings:
            if chunk in ctx_lower:
                return idx + 1  # 1-based

    return -1


def _parse_json_column(value, col_name: str, row_idx: int) -> tuple[list, bool]:
    """Parse JSON array string from CSV cell.

    Returns (parsed_list, had_parse_error).
    - Empty/NaN → ([], False) — not an error
    - Valid JSON array → (parsed_list, False)
    - Malformed JSON / not a list → ([], True) — WARNING emitted
    """
    if isinstance(value, float) and math.isnan(value):
        return [], False
    if not isinstance(value, str) or value.strip() == "":
        return [], False
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed, False
        # Not a JSON array — treat as malformed
        print(f"  WARNING: Row {row_idx}: {col_name} is not a JSON array — {str(value)[:100]}")
        return [], True
    except json.JSONDecodeError:
        print(f"  WARNING: Row {row_idx}: failed to parse {col_name} as JSON — {str(value)[:100]}")
        return [], True


def _validate_csv_columns(df: pd.DataFrame) -> None:
    """Validate that required columns are present in the CSV DataFrame.

    Raises ValueError naming missing columns if any required column is absent.
    Required columns: question, response, retrieved_contexts.
    """
    required = {"question", "response", "retrieved_contexts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")


def _is_missing_value(value) -> bool:
    """Return True if value is None, NaN, or pd.NA.
    
    Used by _run_csv_mode guards for citations, latency_ms, and grading_notes.
    """
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


async def _compute_metrics_from_sample(
    row: dict,
    answer: str,
    retrieved_contexts: list[str],
    citations: list[str],
    judge_llm,
    context_recall,
    context_precision,
    faithfulness,
    correctness,
    citation_faithfulness,
    latency_ms: float | None = None,
) -> dict:
    """Compute all RAGAS metrics from a pre-built sample. Shared by live and CSV paths.

    Returns a dict with all 14 output columns. Encapsulates: relevant_chunk_position,
    RAGAS ascore calls, NaN→None mapping, citation pairing, grounded_correctness,
    and error-resilience try/except for metric computation failures.
    """
    import httpx
    try:
        from openai import APIError as OpenaiAPIError
    except ImportError:
        OpenaiAPIError = Exception  # type: ignore[assignment]

    def _error_row(exc: Exception) -> dict:
        return {
            **row,
            "error": str(exc),
            "correctness": None,
            "faithfulness": None,
            "faithfulness_valid": False,
            "context_recall": None,
            "context_precision": None,
            "citation_faithfulness": None,
            "grounded_correctness": None,
            "response": answer if answer else "",
            "retrieved_contexts": " | ".join(retrieved_contexts) if retrieved_contexts else "",
            "citations": " | ".join(citations) if citations else "",
            "latency_ms": latency_ms,
            "relevant_chunk_position": -1,
        }

    try:
        # Compute relevant_chunk_position diagnostic
        relevant_chunk_position = _find_relevant_chunk_index(
            row.get("grading_notes", ""), retrieved_contexts
        )

        # T-012: Graceful degradation — context-dependent metrics → None when no contexts
        if not retrieved_contexts:
            recall_val = None
            precision_val = None
            faith_val = None
        else:
            # Build RAGAS sample and compute core metrics
            sample = SingleTurnSample(
                user_input=row["question"],
                response=answer,
                retrieved_contexts=retrieved_contexts,
                reference=row.get("grading_notes", ""),
            )

            recall = await context_recall.single_turn_ascore(sample)
            precision = await context_precision.single_turn_ascore(sample)
            faith = await faithfulness.single_turn_ascore(sample)

            recall_val = round(recall, 3)
            precision_val = round(precision, 3)

            # REQ-003: NaN faith → None, faithfulness_valid column
            if faith is None or (isinstance(faith, float) and math.isnan(faith)):
                print(f"  WARNING: faithfulness=NaN for question '{row['question'][:80]}'")
                faith_val = None
            else:
                faith_val = round(faith, 3)

        # FIX-2: correctness=None when no grading_notes (REQ-OFFLINE-005)
        if not row.get("grading_notes", "").strip():
            correctness_val = None
        else:
            correctness_result = await correctness.ascore(
                llm=judge_llm,
                response=answer,
                grading_notes=row.get("grading_notes", ""),
            )
            # REQ-005: Safe int() parsing with fallback to None
            try:
                correctness_val = int(correctness_result.value)
            except (ValueError, TypeError):
                print(f"  WARNING: non-conforming correctness value '{correctness_result.value}' "
                      f"for question '{row['question'][:80]}'")
                correctness_val = None

        # REQ-004: Use _pair_citations_with_contexts for chunk_N matching
        citation_faith = None
        if citations and retrieved_contexts:
            pairs = _pair_citations_with_contexts(citations, retrieved_contexts)
            if not pairs:
                print(f"  WARNING: no chunk_N markers in citations for question "
                      f"'{row['question'][:80]}' — citation_faithfulness=None")
                citation_faith = None
            else:
                citation_tasks = [
                    citation_faithfulness.ascore(
                        llm=judge_llm,
                        response=answer,
                        cited_chunk=ctx,
                    )
                    for _, ctx in pairs
                ]
                citation_results = await asyncio.gather(*citation_tasks)
                supported = sum(1 for r in citation_results if r.value == "supported")
                citation_faith = round(supported / len(citation_results), 3)

        # REQ-003: grounded_correctness = None when faith is invalid
        if faith_val is not None and correctness_val is not None:
            grounded_correctness = round((correctness_val / 4) * faith_val, 3)
        else:
            grounded_correctness = None

        return {
            **row,
            "error": None,
            "faithfulness_valid": faith_val is not None and faith_val == faith_val,
            "response": answer,
            "retrieved_contexts": " | ".join(retrieved_contexts),
            "citations": " | ".join(citations),
            "latency_ms": latency_ms,
            "correctness": correctness_val,
            "faithfulness": faith_val,
            "context_recall": recall_val,
            "context_precision": precision_val,
            "citation_faithfulness": citation_faith,
            "grounded_correctness": grounded_correctness,
            "relevant_chunk_position": relevant_chunk_position,
        }
    except (httpx.HTTPError, OpenaiAPIError, Exception) as exc:
        print(f"  ERROR computing metrics for question '{row['question'][:80]}': {exc}")
        error_dict = _error_row(exc)
        if answer:
            error_dict["response"] = answer
        if retrieved_contexts:
            error_dict["retrieved_contexts"] = " | ".join(retrieved_contexts)
        if citations:
            error_dict["citations"] = " | ".join(citations)
        error_dict["latency_ms"] = latency_ms
        return error_dict


async def _process_question_result(
    tero,
    row: dict,
    judge_llm,
    context_recall,
    context_precision,
    faithfulness,
    correctness,
    citation_faithfulness,
) -> dict:
    """Process a single question: ask Tero, detect errors, then delegate to shared metric computation.

    Returns a dict with all fields for the output CSV row.
    """
    import httpx
    try:
        from openai import APIError as OpenaiAPIError
    except ImportError:
        OpenaiAPIError = Exception  # type: ignore[assignment]

    def _error_row(exc: Exception) -> dict:
        return {
            **row,
            "error": str(exc),
            "correctness": None,
            "faithfulness": None,
            "faithfulness_valid": False,
            "context_recall": None,
            "context_precision": None,
            "citation_faithfulness": None,
            "grounded_correctness": None,
            "response": "",
            "retrieved_contexts": "",
            "citations": "",
            "latency_ms": None,
            "relevant_chunk_position": -1,
        }

    try:
        thread_id = await tero.create_thread()
        result = await tero.ask_question(thread_id, row["question"])
    except (httpx.HTTPError, OpenaiAPIError, Exception) as exc:
        print(f"  ERROR processing question '{row['question'][:80]}': {exc}")
        return _error_row(exc)

    answer = result["answer_text"]
    retrieved_contexts = result["retrieved_contexts"]
    citations = result["citations"]
    latency_ms = result["latency_ms"]

    # TASK-2.1: Detect recursionLimitExceeded — skip RAGAS entirely
    # REQ-006: Use startswith to avoid false positives on mid-answer mentions
    if answer.startswith("recursionLimitExceeded"):
        print(f"  WARNING: backend error for question '{row['question'][:80]}': recursionLimitExceeded")
        return {
            **row,
            "error": "recursionLimitExceeded",
            "faithfulness_valid": False,
            "correctness": None,
            "faithfulness": None,
            "context_recall": None,
            "context_precision": None,
            "citation_faithfulness": None,
            "grounded_correctness": None,
            "response": answer,
            "retrieved_contexts": " | ".join(retrieved_contexts),
            "citations": " | ".join(citations),
            "latency_ms": latency_ms,
            "relevant_chunk_position": -1,
        }

    return await _compute_metrics_from_sample(
        row, answer, retrieved_contexts, citations,
        judge_llm, context_recall, context_precision, faithfulness,
        correctness, citation_faithfulness, latency_ms=latency_ms,
    )


# ------------------------------------------------------------------
# Baseline management
# ------------------------------------------------------------------

def _load_baseline(model_id: str, dataset: str) -> dict | None:
    path = BASELINE_DIR / model_id / f"{dataset}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_baseline(model_id: str, dataset: str, stats: dict, n: int) -> None:
    path = BASELINE_DIR / model_id / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "dataset": dataset,
        "n": n,
        "stats": stats,
    }
    path.write_text(json.dumps(baseline, indent=2))
    print(f"\nBaseline saved to {path}")


def _csv_to_semicolon(directory: Path) -> None:
    """Rewrite all CSVs in directory using semicolon separator (Excel-friendly).

    Idempotent: skips files already delimited by semicolons.
    Detects semicolons by counting separators in first 5 lines.
    """
    for csv_path in directory.glob("*.csv"):
        # REQ-009: Count semicolons in first 5 lines to detect already-converted files
        try:
            lines = csv_path.read_text(encoding="utf-8").splitlines()[:5]
        except (FileNotFoundError, IndexError, PermissionError):
            continue
        if not lines:
            continue

        # If every non-empty line has at least one semicolon, it's already converted
        if all(";" in line for line in lines if line.strip()):
            continue  # already converted

        try:
            df = pd.read_csv(csv_path, sep=",")
        except pd.errors.ParserError as e:
            print(f"Warning: Could not convert {csv_path.name} to semicolon format: {e}")
            continue
        df.to_csv(csv_path, sep=";", index=False)


# ------------------------------------------------------------------
# CSV offline mode
# ------------------------------------------------------------------

async def _run_csv_mode(args: argparse.Namespace, google_api_key: str) -> None:
    """Run RAGAS evaluation from a pre-prepared CSV file — no Tero HTTP dependency.

    1. Read CSV, validate required columns
    2. Create judge LLM + RAGAS metrics
    3. Per row: parse JSON columns, guard recursionLimitExceeded, build sample,
       call _compute_metrics_from_sample()
    4. Write semicolon CSV to experiments/offline/
    5. Group by model_id (if present) → per-model stats + comparison
    6. Print summary
    """
    csv_path = Path(args.from_csv)
    if not csv_path.exists():
        print(f"ERROR: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Read CSV + validate columns
    try:
        df = pd.read_csv(csv_path, sep=None, engine="python", encoding="utf-8-sig")
    except (pd.errors.ParserError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    try:
        _validate_csv_columns(df)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    # B9: Early exit for 0-row CSV
    if len(df) == 0:
        print("CSV has 0 data rows, nothing to evaluate.")
        return

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"Columns: {', '.join(df.columns)}")

    # 2. Create judge LLM + RAGAS metrics (no TeroClient import)
    _openai_client = AsyncOpenAI(
        api_key=google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    cost_tracker = JudgeCostTracker(_openai_client)
    judge_llm = llm_factory("gemini-2.5-flash", client=_openai_client, max_tokens=16384)
    context_recall, context_precision, faithfulness, correctness, citation_faithfulness = _build_metrics(judge_llm)

    # 3. Per-row metric computation
    results: list[dict] = []
    for idx, (_, row_data) in enumerate(df.iterrows()):
        # Guard against NaN/missing values in required columns
        if _is_missing_value(row_data["question"]) or _is_missing_value(row_data["response"]):
            print(f"  WARNING: Row {idx}: missing question or response — skipping")
            result_row = {
                "question": str(row_data.get("question", "")),
                "grading_notes": "",
                "error": "missing_required_value",
                "faithfulness_valid": False,
                "response": str(row_data.get("response", "")),
                "retrieved_contexts": "",
                "citations": "",
                "latency_ms": None,
                "correctness": None,
                "faithfulness": None,
                "context_recall": None,
                "context_precision": None,
                "citation_faithfulness": None,
                "grounded_correctness": None,
                "relevant_chunk_position": -1,
            }
            if "model_id" in df.columns:
                result_row["model_id"] = str(row_data["model_id"])
            results.append(result_row)
            continue

        question = str(row_data["question"])
        answer = str(row_data["response"])

        # Parse retrieved_contexts (required)
        ctx_raw = row_data["retrieved_contexts"]
        retrieved_contexts, ctx_error = _parse_json_column(ctx_raw, "retrieved_contexts", idx)

        # Parse citations (optional)
        cite_error = False
        if "citations" in df.columns and not _is_missing_value(row_data["citations"]):
            citations, cite_error = _parse_json_column(row_data["citations"], "citations", idx)
        else:
            citations = []

        # Parse latency_ms (optional)
        latency_ms = None
        if "latency_ms" in df.columns:
            val = row_data["latency_ms"]
            if not _is_missing_value(val):
                try:
                    latency_ms = float(val)
                except (ValueError, TypeError):
                    pass

        # Build row dict for _compute_metrics_from_sample
        grading_notes = ""
        if "grading_notes" in df.columns:
            gn_val = row_data["grading_notes"]
            if not _is_missing_value(gn_val):
                grading_notes = str(gn_val)

        row_dict = {
            "question": question,
            "grading_notes": grading_notes,
        }

        # FIX-1: Malformed JSON → all metrics None (REQ-OFFLINE-003)
        if ctx_error or cite_error:
            print(f"  WARNING: Row {idx}: json_parse_error — all metrics set to None")
            result_row = {
                **row_dict,
                "error": "json_parse_error",
                "faithfulness_valid": False,
                "correctness": None,
                "faithfulness": None,
                "context_recall": None,
                "context_precision": None,
                "citation_faithfulness": None,
                "grounded_correctness": None,
                "response": answer,
                "retrieved_contexts": " | ".join(retrieved_contexts),
                "citations": " | ".join(citations),
                "latency_ms": latency_ms,
                "relevant_chunk_position": -1,
            }
        # T-011: recursionLimitExceeded guard (same behavior as live mode)
        elif answer.startswith("recursionLimitExceeded"):
            print(f"  WARNING: Row {idx}: recursionLimitExceeded in response")
            result_row = {
                **row_dict,
                "error": "recursionLimitExceeded",
                "faithfulness_valid": False,
                "correctness": None,
                "faithfulness": None,
                "context_recall": None,
                "context_precision": None,
                "citation_faithfulness": None,
                "grounded_correctness": None,
                "response": answer,
                "retrieved_contexts": " | ".join(retrieved_contexts),
                "citations": " | ".join(citations),
                "latency_ms": latency_ms,
                "relevant_chunk_position": -1,
            }
        else:
            result_row = await _compute_metrics_from_sample(
                row_dict, answer, retrieved_contexts, citations,
                judge_llm, context_recall, context_precision, faithfulness,
                correctness, citation_faithfulness, latency_ms=latency_ms,
            )

        # Attach model_id if present in source CSV
        if "model_id" in df.columns:
            result_row["model_id"] = str(row_data["model_id"])

        results.append(result_row)

        # A2: sanitize question for terminal printing (strip non-printable/escape sequences)
        safe_question = repr(question)[:80]
        # A3: replace None/inf/nan with "N/A" in status prints
        c_val = result_row.get("correctness")
        f_val = result_row.get("faithfulness")
        c_str = "N/A" if c_val is None or (isinstance(c_val, float) and (math.isnan(c_val) or math.isinf(c_val))) else c_val
        f_str = "N/A" if f_val is None or (isinstance(f_val, float) and (math.isnan(f_val) or math.isinf(f_val))) else f_val
        print(f"  Row {idx}: question={safe_question} → "
              f"correctness={c_str}, faithfulness={f_str}")

    # 4. Write results CSV
    results_df = pd.DataFrame(results)
    offline_dir = EVALS_DIR / "experiments" / "offline"
    offline_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = offline_dir / f"{timestamp}.csv"
    results_df.to_csv(output_path, sep=";", index=False)
    print(f"\nResults written to {output_path}")

    # 5. Multi-model grouping (T-010)
    has_model_id = "model_id" in df.columns
    if has_model_id:
        all_model_stats: dict[str, dict[str, dict]] = {}
        for model, group_df in results_df.groupby("model_id"):
            stats = analysis._stats_from_df(group_df)
            all_model_stats[model] = {"offline": stats}
            print(f"\n=== Model: {model} ({len(group_df)} rows) ===")
            analysis.print_summary(stats, dataset=f"offline/{model}")
        if len(all_model_stats) > 1:
            analysis.print_model_comparison(all_model_stats)
    else:
        # 6. Single-model stats
        stats = analysis._stats_from_df(results_df)
        analysis.print_summary(stats, dataset="offline")

    cost_tracker.cost_summary()


# ------------------------------------------------------------------
# Cost reporting
# ------------------------------------------------------------------

try:
    EMBEDDING_COST_PER_1K_TOKENS = float(os.environ.get("EMBEDDING_COST_PER_1K_TOKENS", "0.00002"))
except (ValueError, TypeError):
    sys.exit(
        f"ERROR: EMBEDDING_COST_PER_1K_TOKENS must be a number, "
        f"got: {os.environ.get('EMBEDDING_COST_PER_1K_TOKENS', '')!r}"
    )

try:
    JUDGE_COST_PER_1K_PROMPT_TOKENS = float(
        os.environ.get("JUDGE_COST_PER_1K_PROMPT_TOKENS", "0.00015")
    )
except (ValueError, TypeError):
    sys.exit(
        f"ERROR: JUDGE_COST_PER_1K_PROMPT_TOKENS must be a number, "
        f"got: {os.environ.get('JUDGE_COST_PER_1K_PROMPT_TOKENS', '')!r}"
    )

try:
    JUDGE_COST_PER_1K_COMPLETION_TOKENS = float(
        os.environ.get("JUDGE_COST_PER_1K_COMPLETION_TOKENS", "0.00060")
    )
except (ValueError, TypeError):
    sys.exit(
        f"ERROR: JUDGE_COST_PER_1K_COMPLETION_TOKENS must be a number, "
        f"got: {os.environ.get('JUDGE_COST_PER_1K_COMPLETION_TOKENS', '')!r}"
    )


class JudgeCostTracker:
    """Token-counting wrapper around the RAGAS judge LLM's AsyncOpenAI client.

    Intercepts ``AsyncOpenAI.chat.completions.create`` to accumulate real
    ``usage.prompt_tokens`` and ``usage.completion_tokens`` from every API
    response.  Exposes ``cost_summary()`` to print a per-run USD cost report
    formatted consistently with the existing ``cost_report()`` output.

    Pricing is configurable via ``JUDGE_COST_PER_1K_PROMPT_TOKENS`` and
    ``JUDGE_COST_PER_1K_COMPLETION_TOKENS`` environment variables; defaults
    to Gemini 2.5 Flash public pricing.
    """

    def __init__(self, client):
        """Wrap *client* (an ``AsyncOpenAI`` instance) for token counting.

        The constructor monkey-patches ``client.chat.completions.create``
        so every subsequent call is intercepted.
        """
        # Guard against double-wrapping — skip if already tracked
        if getattr(client.chat.completions, "_tracked_by_judge_cost", None) is True:
            return
        self._client = client
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self._original_create = client.chat.completions.create
        self._wrap()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap(self) -> None:
        """Replace ``client.chat.completions.create`` with a tracked version."""
        original = self._original_create
        # NOTE: deliberately creates reference cycle (self → _client → create →
        # closure → tracker → self). Safe for CLI scripts; call unwrap() if
        # reusing the tracker in long-lived contexts.
        tracker = self

        async def _tracked_create(*args, **kwargs):
            response = await original(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                tracker.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                tracker.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            return response

        self._client.chat.completions.create = _tracked_create
        self._client.chat.completions._tracked_by_judge_cost = True

    def unwrap(self) -> None:
        """Restore the original (untracked) ``chat.completions.create``.

        Call before disposing of the tracker in long-lived contexts
        to break the reference cycle.
        """
        self._client.chat.completions.create = self._original_create

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cost_summary(self) -> None:
        """Print a per-run judge-LLM cost report to stdout.

        Format mirrors ``cost_report()``:
            === Judge Cost Report ===
            Judge LLM            : Gemini 2.5 Flash
            Prompt tokens        : N,NNN
            Completion tokens    : N,NNN
            Prompt cost/1K       : $X.XXXXXX
            Completion cost/1K   : $X.XXXXXX
            Total judge USD      : $X.XXXXXX
        """
        prompt_cost = (self.prompt_tokens / 1000) * JUDGE_COST_PER_1K_PROMPT_TOKENS
        completion_cost = (self.completion_tokens / 1000) * JUDGE_COST_PER_1K_COMPLETION_TOKENS
        total_cost = prompt_cost + completion_cost

        print()
        print("=== Judge Cost Report ===")
        print(f"Judge LLM            : Gemini 2.5 Flash")
        print(f"Prompt tokens        : {self.prompt_tokens:,}")
        print(f"Completion tokens    : {self.completion_tokens:,}")
        print(f"Prompt cost/1K       : ${JUDGE_COST_PER_1K_PROMPT_TOKENS:.6f}")
        print(f"Completion cost/1K   : ${JUDGE_COST_PER_1K_COMPLETION_TOKENS:.6f}")
        print(f"Total judge USD      : ${total_cost:.6f}")


def cost_report(embedding_tokens: int, model: str = "text-embedding-3-small") -> None:
    """Print embedding token count and estimated USD cost.

    Uses cl100k_base encoding for token estimation (text-embedding-3-small/ada-002).
    Cost per 1K tokens is configurable via EMBEDDING_COST_PER_1K_TOKENS env var
    (default: $0.00002 — text-embedding-3-small at $0.02/1M tokens).
    """
    cost = (embedding_tokens / 1000) * EMBEDDING_COST_PER_1K_TOKENS
    print(f"\n=== Cost Report ===")
    print(f"Embedding model      : {model}")
    print(f"Embedding tokens     : {embedding_tokens:,}")
    print(f"Cost per 1K tokens   : ${EMBEDDING_COST_PER_1K_TOKENS:.6f}")
    print(f"Total estimated USD  : ${cost:.6f}")
    print(f"LLM description tokens: 0 (skipped)")


# ------------------------------------------------------------------
# Index subcommand
# ------------------------------------------------------------------

async def do_index(args: argparse.Namespace) -> None:
    """Index a dataset's full corpus into an agent with skipDescriptions=true.

    1. Loads corpus from dataset (no questions)
    2. Configures agent with skipDescriptions=true
    3. Uploads all documents
    4. Reports embedding token count and estimated cost
    """
    import tiktoken
    from tero_client import TeroClient

    t0 = time.monotonic()

    bearer_token = (args.bearer_token or os.environ.get("BEARER_TOKEN", "")).strip()
    if not bearer_token:
        print("ERROR: --bearer-token is required (or set BEARER_TOKEN env var).", file=sys.stderr)
        sys.exit(1)

    tero = TeroClient(args.base_url, args.agent_id, bearer_token)

    # 1. Load full corpus (n=0 → no questions, full corpus)
    print(f"Loading corpus for dataset '{args.dataset}'...")
    _, corpus = ds_module.load_one(args.dataset, n=0)

    if args.max_docs and args.max_docs < len(corpus):
        corpus = corpus[:args.max_docs]
        print(f"Truncated to {args.max_docs} documents.")

    print(f"Corpus: {len(corpus)} documents loaded.")

    # 2. Estimate embedding tokens via cl100k_base (used by text-embedding-3-small/ada-002)
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(enc.encode(doc)) for doc in corpus)

    # 3. Clean teardown: list existing files, wait for in-flight processing,
    #    delete all files, then delete tool. Skip wait if no existing files (first run).
    print(f"\nResetting agent {args.agent_id} docs tool...")
    existing_ids = await tero.list_file_ids()
    if existing_ids:
        print(f"  Waiting for {len(existing_ids)} existing file(s) to settle...")
        await tero.wait_files_processed(existing_ids, timeout=120.0)
    await tero.delete_all_files()
    await tero.delete_docs_tool()

    print(f"Configuring agent {args.agent_id} with skipDescriptions=true...")
    config = {"skipDescriptions": True}
    await tero.configure_docs_tool(config)

    # 4. Upload all documents
    print(f"Uploading {len(corpus)} documents...")
    file_ids: list[int] = []
    for idx, doc in enumerate(corpus):
        filename = f"doc_{idx:06d}.txt"
        fid = await tero.upload_document(filename, doc.encode("utf-8"))
        file_ids.append(fid)
        if (idx + 1) % 100 == 0:
            print(f"  Uploaded {idx + 1}/{len(corpus)} documents...")

    print(f"All {len(file_ids)} documents uploaded. Waiting for processing...")
    await tero.wait_files_processed(file_ids)
    print("All documents processed.")

    # 5. Cost report and wall-clock time
    elapsed = time.monotonic() - t0
    cost_report(total_tokens)
    print(f"Index wall-clock time: {elapsed:.1f} seconds")


# ------------------------------------------------------------------
# Eval subcommand
# ------------------------------------------------------------------

async def do_eval(args: argparse.Namespace) -> None:
    """Run RAGAS evaluation against a pre-indexed agent.

    Does NOT upload files or modify tool config. Supports CSV offline mode,
    baseline update/comparison, and all existing RAGAS metrics.
    """
    # ── CSV mode: early return, no Tero dependency ──
    if args.from_csv:
        if args.models:
            print("ERROR: --from-csv and --models are mutually exclusive.", file=sys.stderr)
            sys.exit(1)
        google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not google_api_key:
            print("ERROR: GOOGLE_API_KEY environment variable is not set (required for RAGAS judge).", file=sys.stderr)
            sys.exit(1)
        # Compatibility: set removed attributes that _run_csv_mode may check
        args.n = 1
        args.only = None
        await _run_csv_mode(args, google_api_key)
        return

    # ── Live mode ──
    if args.agent_id is None:
        print("ERROR: --agent-id is required for live eval (or use --from-csv for offline mode).", file=sys.stderr)
        sys.exit(1)
    if not args.models:
        print("ERROR: --models is required for live eval (or use --from-csv for offline mode).", file=sys.stderr)
        sys.exit(1)

    # FIX-3: Lazy TeroClient import — only in live path (REQ-OFFLINE-009)
    from tero_client import TeroClient

    bearer_token = (args.bearer_token or os.environ.get("BEARER_TOKEN", "")).strip()
    if not bearer_token:
        print("ERROR: --bearer-token is required (or set BEARER_TOKEN env var).", file=sys.stderr)
        sys.exit(1)

    google_api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY environment variable is not set (required for RAGAS judge).", file=sys.stderr)
        sys.exit(1)

    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        print("ERROR: --models must contain at least one Tero model ID.", file=sys.stderr)
        sys.exit(1)

    agent_id = args.agent_id
    tero = TeroClient(args.base_url, agent_id, bearer_token)

    print(f"Dataset : {args.dataset} — agent ID: {agent_id}")
    print("Evaluating against pre-indexed agent (no uploads).")

    _openai_client = AsyncOpenAI(
        api_key=google_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    cost_tracker = JudgeCostTracker(_openai_client)
    judge_llm = llm_factory("gemini-2.5-flash", client=_openai_client, max_tokens=16384)
    context_recall, context_precision, faithfulness, correctness, citation_faithfulness = _build_metrics(judge_llm)

    # Load questions only (seed=args.seed for deterministic selection, corpus discarded)
    rows, _ = ds_module.load_one(args.dataset, n=args.max_questions, seed=args.seed)
    loaded_datasets: dict[str, tuple[list[dict], list[str]]] = {args.dataset: (rows, [])}

    all_stats: dict[str, dict[str, dict]] = {}  # {model_id: {dataset: stats}}

    for model_id in model_ids:
        print(f"\n{'#'*60}")
        print(f"# Model: {model_id}")
        print(f"{'#'*60}")
        await tero.set_agent_model(model_id)
        all_stats[model_id] = {}

        for dataset_name in loaded_datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name} (n={args.max_questions})")
            print(f"{'='*60}")

            rows, _ = loaded_datasets[dataset_name]

            @experiment()
            async def run_experiment(row):
                return await _process_question_result(
                    tero, row, judge_llm,
                    context_recall, context_precision, faithfulness,
                    correctness, citation_faithfulness,
                )

            from ragas import Dataset as RagasDataset
            run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_name = f"{model_id}_{run_ts}_{uuid4().hex[:6]}"
            dataset_experiments_dir = EXPERIMENTS_DIR / dataset_name
            dataset_experiments_dir.mkdir(parents=True, exist_ok=True)
            ragas_dataset = RagasDataset(
                name=run_name,
                backend="local/csv",
                root_dir=str(dataset_experiments_dir),
            )
            for row in rows:
                ragas_dataset.append(row)
            ragas_dataset.save()

            print(f"\nRunning evaluation over {len(rows)} questions...")
            experiment_results = await run_experiment.arun(ragas_dataset)
            experiment_results.save()

            # Re-write CSV with semicolon separator for Excel compatibility
            _csv_to_semicolon(dataset_experiments_dir / "experiments")

            print(f"Evaluation complete for '{dataset_name}' / '{model_id}'. CSV: {run_name}.csv")

            stats = analysis.compute_stats(experiment_results, dataset_name)
            analysis.print_summary(stats, dataset=dataset_name)
            all_stats[model_id][dataset_name] = stats

            if args.update_baseline:
                _save_baseline(model_id, dataset_name, stats, args.max_questions)

            if args.compare:
                baseline = _load_baseline(model_id, dataset_name)
                if baseline is None:
                    print(f"\nNo baseline for '{model_id}/{dataset_name}'. Run with --update-baseline first.")
                else:
                    analysis.print_comparison(baseline["stats"], stats)

    if len(model_ids) > 1:
        analysis.print_model_comparison(all_stats)

    cost_tracker.cost_summary()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tero RAG evaluation — index and eval subcommands.")
    subparsers = parser.add_subparsers(dest="command", required=True,
                                       help="Subcommand: index or eval")

    # --- index subcommand ---
    index_parser = subparsers.add_parser("index", help="Index a dataset corpus into an agent")
    index_parser.add_argument("--dataset", required=True, choices=ALL_DATASETS,
                              help="Dataset to index (ragbench, fetaqa, stratrag)")
    index_parser.add_argument("--agent-id", type=int, required=True,
                              help="Tero agent ID to index into")
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

    args = parser.parse_args()

    if args.command == "index":
        asyncio.run(do_index(args))
    elif args.command == "eval":
        asyncio.run(do_eval(args))


if __name__ == "__main__":
    main()
