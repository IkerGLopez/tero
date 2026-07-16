"""
Shared sanity check framework for RAG evaluation results.

Checks run as DataFrame operations against post-evaluation CSV data
and detect anomalies defined by dataset-specific source plans — no LLM calls.

Per-dataset activation config lives in rag_datasets.SANITY_CHECKS.
"""
from __future__ import annotations

import json
import sys

import pandas as pd


def run_sanity_checks(df: pd.DataFrame, dataset: str) -> None:
    """Load per-dataset config, run active checks, print warnings.

    Mutates df for parametric_suspect column (if check runs).
    This function is idempotent: if parametric_suspect already exists,
    check_parametric_suspect skips the mutation.
    """
    # Load config from rag_datasets (import here to avoid circular deps)
    from rag_datasets import SANITY_CHECKS

    active = SANITY_CHECKS.get(dataset, [])

    print(f"\n=== SANITY CHECKS [{dataset}] ===")

    if not active:
        print("  No checks configured for this dataset.")
        return

    for check_name in active:
        if check_name == "faithfulness_zero_with_citations":
            check_faithfulness_zero_with_citations(df)
        elif check_name == "parametric_suspect":
            check_parametric_suspect(df)
        elif check_name == "correctness_grounded_divergence":
            check_correctness_grounded_divergence(df)


def check_faithfulness_zero_with_citations(df: pd.DataFrame) -> None:
    """Warn rows: faithfulness == 0.0 AND citations >= 3 entries.

    RAGBench F3.2: Zero faith with sufficient citations is suspicious
    — the judge may have mis-evaluated.
    """
    if "faithfulness" not in df.columns:
        return
    if "citations" not in df.columns:
        return

    for idx, row in df.iterrows():
        faith = row["faithfulness"]
        if faith != 0.0:
            continue

        # Parse citations to count non-empty entries
        cite_val = row["citations"]
        num_citations = _count_citation_entries(cite_val)

        if num_citations >= 3:
            question = row.get("question", f"<row {idx}>")
            print(
                f"  WARNING [faithfulness_zero_with_citations]: "
                f"Q: {question} — faithfulness=0.0 but "
                f"response has {num_citations} citation(s). "
                f"The zero score may be a judge error."
            )


def check_parametric_suspect(df: pd.DataFrame) -> None:
    """Add bool column: correctness >= 2 AND context_recall == 0.0 -> True.

    FeTaQA F2.2: High correctness with zero recall suggests the LLM
    answered from parametric knowledge, not retrieved context.

    Idempotent: if column already exists, mutation is skipped.
    """
    if "correctness" not in df.columns or "context_recall" not in df.columns:
        return

    if "parametric_suspect" in df.columns:
        # Already computed — do not overwrite
        return

    df["parametric_suspect"] = False
    for idx, row in df.iterrows():
        correctness = row["correctness"]
        context_recall = row["context_recall"]

        # NaN-safe guard
        try:
            c_ok = float(correctness) >= 2.0
        except (TypeError, ValueError):
            c_ok = False
        try:
            r_zero = float(context_recall) == 0.0
        except (TypeError, ValueError):
            r_zero = False

        if c_ok and r_zero:
            df.at[idx, "parametric_suspect"] = True
            question = row.get("question", f"<row {idx}>")
            print(
                f"  WARNING [parametric_suspect]: "
                f"Q: {question} — correctness={correctness}, "
                f"context_recall={context_recall}. "
                f"Possible parametric answer (not from retrieved context)."
            )


def check_correctness_grounded_divergence(
    df: pd.DataFrame, threshold: float = 0.5
) -> None:
    """Warn rows: abs(correctness - grounded_correctness) > threshold.

    FeTaQA F2.1: Large divergence suggests the LLM answered from memory
    rather than retrieved context — the correctness score reflects
    parametric knowledge while grounded_correctness reflects document usage.
    """
    if "correctness" not in df.columns or "grounded_correctness" not in df.columns:
        return

    for idx, row in df.iterrows():
        c = row["correctness"]
        g = row["grounded_correctness"]

        # NaN-safe guard
        try:
            c_f = float(c)
        except (TypeError, ValueError):
            continue
        try:
            g_f = float(g)
        except (TypeError, ValueError):
            continue

        divergence = abs(c_f - g_f)
        if divergence > threshold:
            question = row.get("question", f"<row {idx}>")
            print(
                f"  WARNING [correctness_grounded_divergence]: "
                f"Q: {question} — correctness={c_f}, "
                f"grounded_correctness={g_f}, "
                f"divergence={divergence:.2f} > {threshold}. "
                f"Possible parametric answer (LLM answered from memory)."
            )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _count_citation_entries(cite_value) -> int:
    """Parse a JSON-string citations cell and return count of non-empty entries.

    Returns 0 for NaN, empty, non-string, or malformed JSON values.
    """
    if isinstance(cite_value, float):
        return 0  # NaN or other float
    if not isinstance(cite_value, str) or cite_value.strip() == "":
        return 0
    try:
        parsed = json.loads(cite_value)
        if isinstance(parsed, list):
            return sum(1 for entry in parsed if entry and str(entry).strip() != "")
        return 0
    except (json.JSONDecodeError, TypeError):
        return 0
