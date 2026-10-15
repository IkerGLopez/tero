"""Offline CSV eval flow: RAGAS evaluation from a prepared CSV, no Tero HTTP dependency."""

import config  # Bootstrap: repo .env + sys.path

import argparse
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import analysis
from config import EVALS_DIR, _reranker_state_line
from judge import _build_judge_client
from metrics.deterministic import _RETRIEVAL_METRIC_DEFAULTS
from metrics.gold import _is_gold_linked, _print_alignment_report
from metrics.ragas import _build_metrics
from metrics.sample import _compute_metrics_from_sample
from reporting.csv_io import (
    _is_missing_value,
    _move_model_id_last,
    _parse_json_column,
    _validate_csv_columns,
)


async def _run_csv_mode(args: argparse.Namespace, judge_model: str) -> None:
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
    print(f"{_reranker_state_line()} — offline CSV run: the reranker is not applied")
    print(f"Columns: {', '.join(df.columns)}")

    # 2. Create judge LLM + RAGAS metrics (no TeroClient import)
    _openai_client, judge_llm, cost_tracker, _judge_label = _build_judge_client(judge_model)
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
                **_RETRIEVAL_METRIC_DEFAULTS,
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
                **_RETRIEVAL_METRIC_DEFAULTS,
            }
        # T-011: recursionLimitExceeded guard (same behavior as live mode)
        elif answer.startswith("recursionLimitExceeded"):
            print(f"  WARNING: Row {idx}: recursionLimitExceeded in response")
            result_row = {
                **row_dict,
                "error": "recursionLimitExceeded",
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
                **_RETRIEVAL_METRIC_DEFAULTS,
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
    # Alignment report is available offline whenever the rows carry annotated
    # gold fields (the live path persists them; a plain CSV has none).
    if any(_is_gold_linked(result) for result in results):
        _print_alignment_report(results, results_df)
    offline_dir = EVALS_DIR / "experiments" / "offline"
    offline_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = offline_dir / f"{timestamp}.csv"
    results_df.to_csv(output_path, sep=";", index=False)
    print(f"\nResults written to {output_path}")

    # Run sanity checks on results DataFrame
    from sanity_checks import run_sanity_checks as _run_sc
    _run_sc(results_df, getattr(args, "dataset", "unknown"))
    if "parametric_suspect" in results_df.columns:
        # Re-write CSV to persist the new column. Keep model_id as the last
        # column: sanity checks append after the per-row stamp.
        results_df = _move_model_id_last(results_df)
        results_df.to_csv(output_path, sep=";", index=False)

    # 5. Multi-model grouping (T-010)
    has_model_id = "model_id" in df.columns
    if has_model_id:
        all_model_stats: dict[str, dict[str, dict]] = {}
        for model, group_df in results_df.groupby("model_id"):
            stats = analysis._stats_from_df(group_df)
            all_model_stats[model] = {"offline": stats}
            print(f"\n=== Model: {model} ({len(group_df)} rows) ===")
            analysis.print_summary(stats, df=group_df, dataset=f"offline/{model}")
        if len(all_model_stats) > 1:
            analysis.print_model_comparison(all_model_stats)
    else:
        # 6. Single-model stats
        stats = analysis._stats_from_df(results_df)
        analysis.print_summary(stats, df=results_df, dataset="offline")

    if cost_tracker:
        cost_tracker.cost_summary()
