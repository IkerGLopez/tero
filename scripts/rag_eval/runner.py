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
import hashlib
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

# Pool probe (B2): the embedding cache lives under the gitignored evals/ tree
# so embeddings are never committed.
DEFAULT_PROBE_CACHE_DIR = EVALS_DIR / "pool_probe_cache"


sys.path.insert(0, str(SCRIPT_DIR))
import rag_datasets as ds_module
from rag_datasets import ALL_DATASETS
from retrieval_matching import (
    contains_normalized,
    context_matches_gold_doc,
    normalize_whitespace,
    resolve_gold_targets,
)
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


def _sanitize_error(exc: Exception, max_len: int = 500) -> str:
    """Collapse multi-line exception text into a single CSV-safe string.

    Replaces newlines with `` | `` and truncates to *max_len* chars so that
    ``df.to_csv()`` never emits a broken row when the RAGAS judge produces a
    multi-line traceback.
    """
    flat = str(exc).replace("\n", " | ").replace("\r", "")
    if len(flat) > max_len:
        flat = flat[:max_len - 3] + "..."
    return flat


def _find_relevant_chunk_index(grading_notes: str, contexts: list[str]) -> int:
    """Return 1-based index of highest-scoring context via token overlap + 20-char substring hybrid.

    Returns -1 if grading_notes < 20 chars or no context scores above 0.
    Match is case-insensitive.

    Algorithm:
      1. If grading_notes < 20 chars → return -1 (unchanged contract)
      2. Tokenize grading_notes: whitespace-split, lowercase → set of tokens
      3. Build 20+ character substrings from grading_notes (sliding window, step=1)
      4. For each context, compute score:
         - token overlap: count of grading_notes tokens found in context
         - substring bonus: +1 per 20+ char substring found in context (via `in`)
      5. Return 1-based index of highest-scoring context, or -1 if max score is 0
    """
    if len(grading_notes) < 20:
        return -1

    grading_lower = grading_notes.lower()

    # Tokenize: whitespace-split, lowercase, filter empties
    tokens = set(t for t in grading_lower.split() if t)

    # Build 20+ char substrings from grading_notes
    substrings_20 = [grading_lower[i:i + 20] for i in range(len(grading_lower) - 19)]

    best_score = 0
    best_idx = -1

    for idx, ctx in enumerate(contexts):
        ctx_lower = ctx.lower()

        # Token overlap score
        token_score = sum(1 for t in tokens if t in ctx_lower)

        # Substring bonus: count 20+ char substrings found in context
        substring_bonus = sum(1 for ss in substrings_20 if ss in ctx_lower)

        score = token_score + substring_bonus

        if score > best_score:
            best_score = score
            best_idx = idx + 1  # 1-based

    return best_idx if best_score > 0 else -1


# ------------------------------------------------------------------
# A2 — dedupe + deterministic gold @5 metrics
# ------------------------------------------------------------------

# k for the deterministic FeTaQA retrieval metrics (top-k pinned to 5).
_METRIC_K = 5

# Opción B metadata lines are not table cells.
_CELL_METADATA_PREFIXES = ("Title:", "Section:")

# `[uid]` prefixes mark table rows in the retrieved serialization.
_UID_PREFIX_RE = re.compile(r"^\[[^\]]*\]\s*")


def _dedupe_first_occurrence(contexts: list[str]) -> list[str]:
    """Deduplicate contexts by exact content, keeping first-occurrence order.

    Reuses the `rag-chunk-dedup` semantics (content/hash equality,
    first-occurrence order, before scoring). Scope is one question: callers
    pass a single question's accumulated contexts, so dedup never spans threads.
    """
    seen: set[str] = set()
    unique: list[str] = []
    for context in contexts:
        key = hashlib.sha256(context.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(context)
    return unique


def _normalize_content(text: str) -> str:
    """Normalize line endings and strip outer whitespace for content identity."""
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _extract_cells(context: str) -> list[str]:
    """Extract whole cell values from one serialized table context.

    Opción B rows carry a line-start `[uid]` prefix and pipe-delimited VALUES:
    their cells are kept whole (colons included), so a gold value containing a
    colon still matches its complete cell. Legacy rows carry `header: value`
    pairs, normalized to their value so cell matching compares values, not
    labels. `Title:`/`Section:` metadata lines never become cells.
    """
    cells: list[str] = []
    for raw_line in context.splitlines():
        stripped = raw_line.strip()
        uid_match = _UID_PREFIX_RE.match(stripped)
        line = stripped[uid_match.end():] if uid_match else stripped
        if not line or line.startswith(_CELL_METADATA_PREFIXES):
            continue
        for part in line.split("|"):
            cell = part.strip()
            if not uid_match and ":" in cell:
                cell = cell.split(":", 1)[1].strip()
            if cell:
                cells.append(cell)
    return cells


def _annotate_gold(rows: list[dict], corpus: list[str], max_docs: int | None = None) -> list[dict]:
    """Attach the in-prefix gold targets to gold-linked rows.

    Single annotation seam (design AD-6), additive output:
      - legacy FeTaQA rows keep `gold_content` and `gold_out_of_corpus` exactly
        as before (the loader flag OR a missing id OR an id at/after the
        effective prefix marks the question out of corpus);
      - every gold-linked row also gains `gold_doc_ids_in_prefix`,
        `gold_contents_in_prefix`, `gold_sentences_in_prefix` and
        `no_gold_labels`, with the in-prefix subset scored (a partially covered
        row is never out of corpus and its out-of-prefix gold is never a miss);
      - rows from datasets without gold linkage pass through untouched.

    `max_docs` mirrors the indexed prefix so gold beyond it is excluded instead
    of scored 0. Input rows are not mutated.
    """
    effective_len = len(corpus) if max_docs is None else min(max_docs, len(corpus))
    annotated: list[dict] = []
    for row in rows:
        enriched = dict(row)
        targets = resolve_gold_targets(row, effective_len)
        if targets is None:
            annotated.append(enriched)
            continue

        in_prefix_ids = targets["gold_doc_ids"]
        out_of_corpus = targets["gold_out_of_corpus"]
        enriched["gold_out_of_corpus"] = out_of_corpus
        enriched["no_gold_labels"] = targets["no_gold_labels"]
        enriched["gold_doc_ids_in_prefix"] = in_prefix_ids
        enriched["gold_contents_in_prefix"] = [corpus[doc_id] for doc_id in in_prefix_ids]
        enriched["gold_sentences_in_prefix"] = _in_prefix_gold_sentences(row, in_prefix_ids)

        if "gold_doc_id" in row or "gold_out_of_corpus" in row:
            # Legacy FeTaQA behaviour, preserved exactly (same CSV column)
            enriched["gold_content"] = (
                None if out_of_corpus or not in_prefix_ids else corpus[in_prefix_ids[0]]
            )
        annotated.append(enriched)
    return annotated


def _in_prefix_gold_sentences(row: dict, in_prefix_ids: list[int]) -> list[str]:
    """Gold sentence texts whose source document is inside the indexed prefix.

    Sentences from out-of-prefix documents leave the denominator entirely (spec:
    *Out-of-prefix sentences leave the denominator*), so the metric never scores
    a sentence the indexed run could not have retrieved.
    """
    sentences = row.get("gold_sentences")
    sentence_doc_ids = row.get("gold_sentence_doc_ids")
    if not sentences or not sentence_doc_ids:
        return []
    in_prefix = set(in_prefix_ids)
    return [text for text, doc_id in zip(sentences, sentence_doc_ids) if doc_id in in_prefix]


def _doc_recall_5(
    contexts: list[str], gold_contents: list[str] | None, out_of_corpus: bool,
) -> float | None:
    """1 when any in-prefix gold document matches one of the first five contexts.

    Identity is `context_matches_gold_doc` (normalized equality OR containment,
    covering documents the chunking splits). Out-of-corpus questions and rows
    without a gold target return None so they leave the denominator instead of
    being scored 0; a labeled in-corpus question with no match is a real 0.
    """
    if out_of_corpus or not gold_contents:
        return None
    for context in contexts[:_METRIC_K]:
        for gold_doc in gold_contents:
            if context_matches_gold_doc(context, gold_doc):
                return 1.0
    return 0.0


def _sentence_recall_5(
    contexts: list[str], gold_sentences: list[str] | None, out_of_corpus: bool,
) -> float | None:
    """Fraction of usable in-prefix gold sentences found in the first five contexts.

    Degenerate (empty/whitespace-only) sentences leave both numerator and
    denominator; when no usable sentence remains — or the question is
    out-of-corpus / has no gold target — the metric is not-applicable (None),
    never 0. A sentence counts when it appears in *any* of the five contexts,
    with no document constraint (spec: sentence recall).
    """
    if out_of_corpus or not gold_sentences:
        return None
    usable = [str(sentence) for sentence in gold_sentences if normalize_whitespace(str(sentence))]
    if not usable:
        return None
    top_five = contexts[:_METRIC_K]
    found = sum(
        1 for sentence in usable
        if any(contains_normalized(context, sentence) for context in top_five)
    )
    return found / len(usable)


def _doc_coverage_5(
    contexts: list[str], gold_contents: list[str] | None, out_of_corpus: bool,
) -> float | None:
    """Fraction of the question's in-prefix gold documents matched by the top five contexts.

    Same identity rule as `doc_recall_5`; each gold document counts at most once.
    Secondary metric: it never replaces `doc_recall_5` as the primary signal and
    follows the same not-applicable exclusions.
    """
    if out_of_corpus or not gold_contents:
        return None
    top_five = contexts[:_METRIC_K]
    matched = sum(
        1 for gold_doc in gold_contents
        if any(context_matches_gold_doc(context, gold_doc) for context in top_five)
    )
    return matched / len(gold_contents)


# Error/guard rows carry every deterministic retrieval metric as None so the
# CSV shape never depends on how far a row got (design: lossless exclusions).
_RETRIEVAL_METRIC_DEFAULTS = {
    "table_recall_5": None,
    "cell_recall_5": None,
    "doc_recall_5": None,
    "sentence_recall_5": None,
    "doc_coverage_5": None,
}


def _is_gold_linked(row: dict) -> bool:
    """True when the row carries a resolved gold target (annotation happened).

    Pass-through rows from datasets without gold linkage carry none of these
    keys, so they are never mistaken for scorable or excluded questions.
    """
    return any(
        key in row
        for key in ("gold_doc_ids_in_prefix", "gold_sentences_in_prefix", "no_gold_labels")
    )


def _population_counts(rows: list[dict]) -> dict[str, int]:
    """Count the question populations the alignment report separates (spec: alignment).

    Scorable = labeled with at least one in-prefix gold document, which is
    exactly the metric denominator. Out-of-corpus = labeled but no gold inside
    the indexed prefix. No-label = no gold labels at all — never conflated with
    either. The four buckets plus `n_without_gold_linkage` account for every
    question, so the invariant `n_questions == n_scorable + n_out_of_corpus +
    n_no_gold_labels + n_without_gold_linkage` always holds.
    """
    counts = {
        "n_questions": len(rows),
        "n_scorable": 0,
        "n_out_of_corpus": 0,
        "n_no_gold_labels": 0,
        "n_without_gold_linkage": 0,
    }
    for row in rows:
        if not _is_gold_linked(row):
            counts["n_without_gold_linkage"] += 1
        elif row.get("no_gold_labels"):
            counts["n_no_gold_labels"] += 1
        elif row.get("gold_out_of_corpus"):
            counts["n_out_of_corpus"] += 1
        else:
            counts["n_scorable"] += 1
    return counts


def _print_alignment_report(rows: list[dict], results_df=None) -> None:
    """Print the retrieval populations, and the miss accounting when results exist.

    Scorable, out-of-corpus and no-label questions are reported separately and
    by count. When the results frame carries `doc_recall_5`, the report adds
    `measured` / `misses` / `unmeasured` so misses stay distinct from both
    exclusion populations and from scorable questions whose metric never ran
    (an error row), keeping `denominator(doc_recall_5) == n_scorable` honest on
    error-free runs. Also usable from the offline CSV path whenever the rows
    carry annotated gold fields.
    """
    counts = _population_counts(rows)
    print("\n=== Retrieval alignment ===")
    print(f"Questions            : {counts['n_questions']}")
    print(f"Scorable             : {counts['n_scorable']}")
    print(f"Out-of-corpus        : {counts['n_out_of_corpus']}")
    print(f"No gold labels       : {counts['n_no_gold_labels']}")
    print(f"Without gold linkage : {counts['n_without_gold_linkage']}")

    if results_df is None or "doc_recall_5" not in getattr(results_df, "columns", []):
        return
    values = pd.to_numeric(results_df["doc_recall_5"], errors="coerce")
    measured = int(values.notna().sum())
    misses = int((values == 0.0).sum())
    unmeasured = max(counts["n_scorable"] - measured, 0)
    print(f"doc_recall_5         : measured {measured}, misses {misses}, unmeasured {unmeasured}")



def _table_recall_5(contexts: list[str], gold_content: str | None, out_of_corpus: bool) -> float | None:
    """1 when the gold document is among the first five unique contexts, else 0.

    Identity is exact content match after line-ending normalization + strip.
    Out-of-corpus questions and rows without gold linkage return None so they
    are excluded from aggregates instead of scored 0.
    """
    if out_of_corpus or gold_content is None:
        return None
    gold_key = _normalize_content(gold_content)
    for context in contexts[:_METRIC_K]:
        if _normalize_content(context) == gold_key:
            return 1.0
    return 0.0


def _is_degenerate_gold_value(value) -> bool:
    """Degenerate gold cells: missing, empty, or the FeTaQA `-` placeholder."""
    if value is None:
        return True
    return str(value).strip() in ("", "-")


def _cell_recall_5(contexts: list[str], gold_values: list[str] | None, out_of_corpus: bool) -> float | None:
    """Fraction of non-degenerate gold cells matched as whole cells in the top 5.

    Matching requires equality with a complete extracted cell (no substrings).
    Degenerate gold values leave both numerator and denominator; when none
    remain — or the question has no gold linkage / is out of corpus — the
    metric is not-applicable (None), never 0.
    """
    if out_of_corpus or not gold_values:
        return None
    usable = [
        str(value).strip()
        for value in gold_values
        if not _is_degenerate_gold_value(value)
    ]
    if not usable:
        return None
    cells: set[str] = set()
    for context in contexts[:_METRIC_K]:
        cells.update(_extract_cells(context))
    matched = sum(1 for value in usable if value in cells)
    return matched / len(usable)


def _parse_json_column(value, col_name: str, row_idx: int) -> tuple[list, bool]:
    """Parse JSON array string from CSV cell.

    Returns (parsed_list, had_parse_error).
    - Empty/NaN → ([], False) — not an error
    - Valid JSON array → (parsed_list, False)
    - Pipe-separated text → split by " | " → (parsed_list, False)
    - Malformed JSON / not a list and not pipe-separated → ([], True) — WARNING emitted
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
        # FALLBACK 1: pipe-separated plain text (output format uses " | ".join)
        if " | " in value:
            parts = [p.strip() for p in value.split(" | ") if p.strip()]
            if parts:
                return parts, False
        # FALLBACK 2: single plain-text value → return as single-element list
        # (handles citations like ["text"](chunk_N), grading notes, etc.)
        stripped = value.strip()
        if stripped:
            return [stripped], False
        return [], False


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
            "error": _sanitize_error(exc),
            "correctness": None,
            "faithfulness": None,
            "context_recall": None,
            "context_precision": None,
            "citation_faithfulness": None,
            "grounded_correctness": None,
            "response": answer if answer else "",
            "retrieved_contexts": " | ".join(retrieved_contexts) if retrieved_contexts else "",
            "citations": " | ".join(citations) if citations else "",
            "latency_ms": latency_ms,
            "relevant_chunk_position": -1,
            **_RETRIEVAL_METRIC_DEFAULTS,
        }

    try:
        # Compute relevant_chunk_position diagnostic
        relevant_chunk_position = _find_relevant_chunk_index(
            row.get("grading_notes", ""), retrieved_contexts
        )

        # A2: deterministic gold @5 metrics (None when no gold linkage / out of corpus)
        gold_out_of_corpus = bool(row.get("gold_out_of_corpus", False))
        gold_contents = row.get("gold_contents_in_prefix")
        table_recall_5 = _table_recall_5(
            retrieved_contexts, row.get("gold_content"), gold_out_of_corpus
        )
        cell_recall_5 = _cell_recall_5(
            retrieved_contexts, row.get("gold_values"), gold_out_of_corpus
        )
        # Multi-gold deterministic retrieval metrics (spec: eval-runner-metrics).
        # `_process_question_result` deduped the contexts before this call.
        doc_recall_5 = _doc_recall_5(retrieved_contexts, gold_contents, gold_out_of_corpus)
        sentence_recall_5 = _sentence_recall_5(
            retrieved_contexts, row.get("gold_sentences_in_prefix"), gold_out_of_corpus
        )
        doc_coverage_5 = _doc_coverage_5(retrieved_contexts, gold_contents, gold_out_of_corpus)

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

            # Internal flag: does faithfulness have a usable numeric value?
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
            "table_recall_5": table_recall_5,
            "cell_recall_5": cell_recall_5,
            "doc_recall_5": doc_recall_5,
            "sentence_recall_5": sentence_recall_5,
            "doc_coverage_5": doc_coverage_5,
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
            "error": _sanitize_error(exc),
            "correctness": None,
            "faithfulness": None,
            "context_recall": None,
            "context_precision": None,
            "citation_faithfulness": None,
            "grounded_correctness": None,
            "response": "",
            "retrieved_contexts": "",
            "citations": "",
            "latency_ms": None,
            "relevant_chunk_position": -1,
            **_RETRIEVAL_METRIC_DEFAULTS,
        }

    try:
        thread_id = await tero.create_thread()
        result = await tero.ask_question(thread_id, row["question"])
    except (httpx.HTTPError, OpenaiAPIError, Exception) as exc:
        print(f"  ERROR processing question '{row['question'][:80]}': {exc}")
        return _error_row(exc)

    answer = result["answer_text"]
    # A2: dedupe this question's accumulated contexts BEFORE any metric consumer;
    # scope is the question (thread), never global.
    retrieved_contexts = _dedupe_first_occurrence(result["retrieved_contexts"])
    citations = result["citations"]
    latency_ms = result["latency_ms"]

    # TASK-2.1: Detect recursionLimitExceeded — skip RAGAS entirely
    # REQ-006: Use startswith to avoid false positives on mid-answer mentions
    if answer.startswith("recursionLimitExceeded"):
        print(f"  WARNING: backend error for question '{row['question'][:80]}': recursionLimitExceeded")
        return {
            **row,
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

    return await _compute_metrics_from_sample(
        row, answer, retrieved_contexts, citations,
        judge_llm, context_recall, context_precision, faithfulness,
        correctness, citation_faithfulness, latency_ms=latency_ms,
    )


# ------------------------------------------------------------------
# Baseline management
# ------------------------------------------------------------------

# Comparability protocol (D7): eval runs measure the docs tool's default top-5
# retention (rerank off), so every baseline records the same top_k.
EVAL_TOP_K = 5


def _comparability_metadata(agent_id: int | None, seed: int, judge_model: str) -> dict:
    """Current run's comparability inputs, keyed like the baseline metadata (D7)."""
    return {
        "agent_id": agent_id,
        "seed": seed,
        "judge_model": judge_model,
        "top_k": EVAL_TOP_K,
    }


def _baseline_deviations(baseline: dict, metadata: dict) -> list[str]:
    """Compare a baseline's comparability inputs against the current run (D7).

    Returns one human-readable line per deviation. Keys missing from the
    baseline (written before D7) are skipped — nothing was recorded, so there
    is nothing to compare against.
    """
    deviations: list[str] = []
    for key in ("agent_id", "seed", "judge_model", "top_k"):
        if key not in baseline:
            continue
        if baseline[key] != metadata[key]:
            deviations.append(f"{key}: baseline {baseline[key]!r} != current {metadata[key]!r}")
    return deviations


def _print_comparability_report(baseline: dict, metadata: dict) -> list[str]:
    """Print the comparability guard for a --compare run; returns the deviations (D7)."""
    deviations = _baseline_deviations(baseline, metadata)
    if deviations:
        print("\nComparability deviations (baseline vs current run):")
        for deviation in deviations:
            print(f"  - {deviation}")
        print("Deltas below are not like-for-like — align the inputs or update the baseline first.")
    else:
        print("\nComparability: baseline and current run agree on agent, seed, judge and top_k.")
    return deviations


def _load_baseline(model_id: str, dataset: str) -> dict | None:
    path = BASELINE_DIR / model_id / f"{dataset}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_baseline(model_id: str, dataset: str, stats: dict, n: int,
                   *, agent_id: int | None, seed: int, judge_model: str) -> None:
    """Persist the baseline stats plus the comparability inputs that produced them (D7)."""
    path = BASELINE_DIR / model_id / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "dataset": dataset,
        "n": n,
        "agent_id": agent_id,
        "seed": seed,
        "judge_model": judge_model,
        "top_k": EVAL_TOP_K,
        "stats": stats,
    }
    path.write_text(json.dumps(baseline, indent=2))
    print(f"\nBaseline saved to {path}")


def _csv_to_semicolon(directory: Path) -> None:
    """Rewrite all CSVs in directory using semicolon separator (Excel-friendly).

    Idempotent: skips files already delimited by semicolons (detected via csv.Sniffer).
    Uses two-phase parsing: pandas python engine for well-formed CSVs,
    csv module fallback for files with malformed quoting or encoding issues.
    Phase 2 writes to a temp file first, then replaces atomically to avoid data loss.
    """
    import csv as csv_module

    for csv_path in directory.glob("*.csv"):
        # REQ-009: Use csv.Sniffer on first 8KB to detect the actual delimiter
        try:
            raw = csv_path.open("rb").read(8192)
        except OSError:
            continue
        if not raw:
            continue

        # If the file is already semicolon-delimited, skip conversion
        try:
            dialect = csv_module.Sniffer().sniff(raw[:8192].decode("utf-8-sig", errors="replace"))
            if dialect.delimiter == ";":
                continue  # already converted
        except csv_module.Error:
            pass  # sniffer failed — fall through to Phase 1

        # Phase 1: pandas with python engine (handles more edge cases than C engine)
        try:
            df = pd.read_csv(csv_path, sep=",", engine="python", encoding="utf-8-sig")
        except (pd.errors.ParserError, UnicodeDecodeError, ValueError):
            pass  # parsing failed — fall through to Phase 2
        else:
            tmp_path = csv_path.with_name(csv_path.name + ".tmp")
            try:
                df.to_csv(tmp_path, sep=";", index=False)
                tmp_path.replace(csv_path)
            except OSError:
                print(f"Warning: Could not convert {csv_path.name} to semicolon format: write/replace failed")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
            continue

        # Phase 2: csv module fallback for files with embedded commas
        #           (e.g. error trace cells that break fixed-width parsing)
        try:
            with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
                reader = csv_module.reader(f)
                rows = list(reader)
            if rows:
                # Write to temp file first, then atomically replace (avoid truncation loss)
                tmp_path = csv_path.with_name(csv_path.name + ".tmp")
                try:
                    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                        csv_module.writer(f, delimiter=";").writerows(rows)
                    tmp_path.replace(csv_path)
                finally:
                    if tmp_path.exists():
                        tmp_path.unlink()
                continue
        except Exception as phase2_exc:
            print(f"Warning: Could not convert {csv_path.name} to semicolon format: {phase2_exc}")
            continue

        print(f"Warning: Could not convert {csv_path.name} to semicolon format: unparseable content — both pandas and csv module failed to read the file")


# ------------------------------------------------------------------
# CSV offline mode
# ------------------------------------------------------------------

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
        # Re-write CSV to persist the new column
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

# Per-model judge pricing (USD per 1K tokens), derived from public per-1M pricing:
#   gemini-3.5-flash : $1.50 / $9.00   per 1M  ->  $0.0015 / $0.0090 per 1K
#   gpt-4o           : $2.50 / $10.00  per 1M  ->  $0.0025 / $0.0100 per 1K
#   gpt-4o-mini      : $0.150 / $0.600 per 1M  ->  $0.00015 / $0.00060 per 1K
# Keys are matched case-insensitively against the --judge-model argument.
JUDGE_PRICING_TABLE = {
    "gemini-3.5-flash": (0.0015, 0.0090),
    "gemini-2.5-flash": (0.00030, 0.00250),
    "gpt-4o": (0.0025, 0.0100),
    "claude-sonnet-4": (0.00300, 0.01500),
    "claude-sonnet-4-5": (0.00300, 0.01500),
    "claude-sonnet-4-6": (0.00300, 0.01500),
    "claude-haiku-4-5": (0.00100, 0.00500),
}

# Bedrock inference profile IDs for supported Claude judge models.
# Keys match --judge-model values; values are AWS Bedrock inference profile IDs.
# Prefix depends on your AWS region — currently set for eu-west-1 (eu.*).
# Run `aws bedrock list-inference-profiles` to find your IDs.
BEDROCK_JUDGE_MODELS: dict[str, str] = {
    "claude-sonnet-4":   "eu.anthropic.claude-sonnet-4-20250514-v1:0",
    "claude-haiku-4-5":  "eu.anthropic.claude-haiku-4-5-20251001-v1:0",
    "claude-sonnet-4-5": "eu.anthropic.claude-sonnet-4-5-20250929-v1:0",
    "claude-sonnet-4-6": "eu.anthropic.claude-sonnet-4-6",
}

def _resolve_judge_pricing(model_id: str) -> tuple:
    """Resolve per-1K-token pricing for a judge model.

    Precedence (highest wins):
      1. Explicit env vars JUDGE_COST_PER_1K_PROMPT_TOKENS /
         JUDGE_COST_PER_1K_COMPLETION_TOKENS (override everything).
      2. JUDGE_PRICING_TABLE lookup by model_id (case-insensitive).
      3. Module-level fallback constants (loaded from env with their own
         defaults; used for unknown models not in the table).

    Returns (prompt_cost_per_1k, completion_cost_per_1k) as floats.
    """
    env_prompt = os.environ.get("JUDGE_COST_PER_1K_PROMPT_TOKENS")
    env_completion = os.environ.get("JUDGE_COST_PER_1K_COMPLETION_TOKENS")
    if env_prompt is not None and env_completion is not None:
        return float(env_prompt), float(env_completion)
    entry = JUDGE_PRICING_TABLE.get(model_id.lower())
    if entry is not None:
        return entry
    return JUDGE_COST_PER_1K_PROMPT_TOKENS, JUDGE_COST_PER_1K_COMPLETION_TOKENS


class JudgeCostTracker:
    """Token-counting wrapper around the RAGAS judge LLM's API client.

    Intercepts ``chat.completions.create`` (OpenAI) or ``messages.create``
    (Anthropic Bedrock) to accumulate real ``usage`` token counts from every
    API response.  Exposes ``cost_summary()`` to print a per-run USD cost
    report formatted consistently with the existing ``cost_report()`` output.

    Pricing is resolved per model via ``JUDGE_PRICING_TABLE`` (see
    ``_resolve_judge_pricing``). Explicit ``JUDGE_COST_PER_1K_PROMPT_TOKENS`` /
    ``JUDGE_COST_PER_1K_COMPLETION_TOKENS`` env vars override the table; unknown
    models fall back to the module-level default constants.
    """

    def __init__(self, client, model_label: str = "Gemini 3.5 Flash",
                 prompt_cost_per_1k: float | None = None,
                 completion_cost_per_1k: float | None = None,
                 provider: str = "openai"):
        """Wrap *client* for token counting.

        The constructor monkey-patches the appropriate create method so
        every subsequent call is intercepted.

        *provider* ``"openai"`` (default) patches ``client.chat.completions.create``
        and reads ``usage.prompt_tokens`` / ``usage.completion_tokens``.
        ``"anthropic"`` patches ``client.messages.create`` and reads
        ``usage.input_tokens`` / ``usage.output_tokens``.
        """
        self._provider = provider

        if provider == "anthropic":
            # Guard against double-wrapping on messages namespace
            if getattr(client.messages, "_tracked_by_judge_cost", None) is True:
                # Already tracked — expose minimal attributes for access safety
                self._client = client
                self._model_label = model_label
                self.prompt_cost_per_1k = 0
                self.completion_cost_per_1k = 0
                self.prompt_tokens: int = 0
                self.completion_tokens: int = 0
                self._original_create = None
                return
            self._client = client
            self._model_label = model_label
            self.prompt_cost_per_1k = (
                prompt_cost_per_1k if prompt_cost_per_1k is not None
                else JUDGE_COST_PER_1K_PROMPT_TOKENS
            )
            self.completion_cost_per_1k = (
                completion_cost_per_1k if completion_cost_per_1k is not None
                else JUDGE_COST_PER_1K_COMPLETION_TOKENS
            )
            self.prompt_tokens: int = 0
            self.completion_tokens: int = 0
            self._original_create = client.messages.create
            self._wrap_anthropic()
        else:
            # OpenAI (default, backward-compatible)
            if getattr(client.chat.completions, "_tracked_by_judge_cost", None) is True:
                # Already tracked — expose minimal attributes for access safety
                self._client = client
                self._model_label = model_label
                self.prompt_cost_per_1k = 0
                self.completion_cost_per_1k = 0
                self.prompt_tokens: int = 0
                self.completion_tokens: int = 0
                self._original_create = None
                return
            self._client = client
            self._model_label = model_label
            self.prompt_cost_per_1k = (
                prompt_cost_per_1k if prompt_cost_per_1k is not None
                else JUDGE_COST_PER_1K_PROMPT_TOKENS
            )
            self.completion_cost_per_1k = (
                completion_cost_per_1k if completion_cost_per_1k is not None
                else JUDGE_COST_PER_1K_COMPLETION_TOKENS
            )
            self.prompt_tokens: int = 0
            self.completion_tokens: int = 0
            self._original_create = client.chat.completions.create
            self._wrap_openai()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _wrap_openai(self) -> None:
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

    def _wrap_anthropic(self) -> None:
        """Replace ``client.messages.create`` with a tracked version
        that reads ``usage.input_tokens`` / ``usage.output_tokens``.

        Also sanitises kwargs: drops ``top_p`` and defaults ``temperature``
        to 0 (deterministic) because Bedrock Claude rejects having both set.
        """
        original = self._original_create
        tracker = self

        async def _tracked_messages_create(*args, **kwargs):
            # Bedrock Claude rejects temperature + top_p together.
            # RAGAS/instructor may inject both; strip top_p and default
            # temperature to 0 for deterministic judge output.
            kwargs.pop("top_p", None)
            if "temperature" not in kwargs:
                kwargs["temperature"] = 0
            response = await original(*args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                tracker.prompt_tokens += getattr(usage, "input_tokens", 0) or 0
                tracker.completion_tokens += getattr(usage, "output_tokens", 0) or 0
            return response

        self._client.messages.create = _tracked_messages_create
        self._client.messages._tracked_by_judge_cost = True

    def unwrap(self) -> None:
        """Restore the original (untracked) create method.

        Call before disposing of the tracker in long-lived contexts
        to break the reference cycle.
        """
        if self._original_create is None:
            return  # double-wrap guard left no original to restore
        if self._provider == "anthropic":
            self._client.messages.create = self._original_create
        else:
            self._client.chat.completions.create = self._original_create

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cost_summary(self) -> None:
        """Print a per-run judge-LLM cost report to stdout.

        Format mirrors ``cost_report()``:
            === Judge Cost Report ===
            Judge LLM            : Gemini 3.5 Flash
            Prompt tokens        : N,NNN
            Completion tokens    : N,NNN
            Prompt cost/1K       : $X.XXXXXX
            Completion cost/1K   : $X.XXXXXX
            Total judge USD      : $X.XXXXXX
        """
        prompt_cost = (self.prompt_tokens / 1000) * self.prompt_cost_per_1k
        completion_cost = (self.completion_tokens / 1000) * self.completion_cost_per_1k
        total_cost = prompt_cost + completion_cost

        print()
        print("=== Judge Cost Report ===")
        print(f"Judge LLM            : {self._model_label}")
        print(f"Prompt tokens        : {self.prompt_tokens:,}")
        print(f"Completion tokens    : {self.completion_tokens:,}")
        print(f"Prompt cost/1K       : ${self.prompt_cost_per_1k:.6f}")
        print(f"Completion cost/1K   : ${self.completion_cost_per_1k:.6f}")
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


# ------------------------------------------------------------------
# Judge LLM factory
# ------------------------------------------------------------------

def _build_judge_client(model_id: str) -> tuple:
    """Build API client, RAGAS judge LLM, and cost tracker.

    Detects vendor from model name prefix:
      - gpt*, o1*, o3*, o4* → OpenAI (OPENAI_API_KEY)
      - gemini*  → Google (GOOGLE_API_KEY)
      - claude*  → Anthropic Bedrock (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION)

    Returns (client, judge_llm, cost_tracker, model_label).
    Exits with error on unknown vendor/model or missing credentials.
    """
    model_lower = model_id.lower()

    if model_lower.startswith("claude"):
        # Resolve friendly model name to Bedrock inference profile ID
        inference_profile = BEDROCK_JUDGE_MODELS.get(model_lower)
        if not inference_profile:
            valid = list(BEDROCK_JUDGE_MODELS)
            print(
                f"ERROR: Unknown Claude model '{model_id}'. "
                f"Valid: {valid}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Validate AWS credentials
        for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"):
            if not os.environ.get(var):
                print(
                    f"ERROR: {var} env var is required for Bedrock judge models.",
                    file=sys.stderr,
                )
                sys.exit(1)

        # Lazy import — anthropic SDK only loaded for Bedrock judge models
        from anthropic import AsyncAnthropicBedrock

        client = AsyncAnthropicBedrock(
            aws_access_key=os.environ["AWS_ACCESS_KEY_ID"],
            aws_secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
            aws_region=os.environ["AWS_REGION"],
            **(dict(aws_session_token=os.environ["AWS_SESSION_TOKEN"]) if os.environ.get("AWS_SESSION_TOKEN") else {}),
        )

        _prompt_rate, _completion_rate = _resolve_judge_pricing(model_id)
        cost_tracker = JudgeCostTracker(
            client,
            provider="anthropic",
            model_label=model_id,
            prompt_cost_per_1k=_prompt_rate,
            completion_cost_per_1k=_completion_rate,
        )
        judge_llm = llm_factory(
            inference_profile,
            provider="anthropic",
            client=client,
            max_tokens=16384,
        )

        return client, judge_llm, cost_tracker, model_id

    if model_lower.startswith(("gpt", "o1", "o3", "o4")):
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            print("ERROR: OPENAI_API_KEY env var is required for OpenAI judge models.", file=sys.stderr)
            sys.exit(1)
        base_url = "https://api.openai.com/v1/"
        model_label = model_id
    elif model_lower.startswith("gemini"):
        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            print("ERROR: GOOGLE_API_KEY env var is required for Gemini judge models.", file=sys.stderr)
            sys.exit(1)
        base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
        model_label = model_id
    else:
        print(
            f"ERROR: Unknown judge model vendor for '{model_id}'. "
            f"Expected prefix: claude*, gpt*, o1*, o3*, o4*, or gemini*.",
            file=sys.stderr,
        )
        sys.exit(1)

    openai_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    _prompt_rate, _completion_rate = _resolve_judge_pricing(model_id)
    cost_tracker = JudgeCostTracker(
        openai_client,
        model_label=model_label,
        prompt_cost_per_1k=_prompt_rate,
        completion_cost_per_1k=_completion_rate,
    )
    judge_llm = llm_factory(model_id, client=openai_client, max_tokens=16384)

    return openai_client, judge_llm, cost_tracker, model_label


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

    agent_id = args.agent_id if args.agent_id is not None else int(os.environ.get("RAG_EVAL_AGENT_ID", "0"))
    if agent_id <= 0:
        print("ERROR: --agent-id is required for index (or set RAG_EVAL_AGENT_ID env var).", file=sys.stderr)
        sys.exit(1)

    bearer_token = (args.bearer_token or os.environ.get("BEARER_TOKEN", "")).strip()
    if not bearer_token:
        print("ERROR: --bearer-token is required (or set BEARER_TOKEN env var).", file=sys.stderr)
        sys.exit(1)

    tero = TeroClient(args.base_url, agent_id, bearer_token)

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
    print(f"\nResetting agent {agent_id} docs tool...")
    existing_ids = await tero.list_file_ids()
    if existing_ids:
        print(f"  Waiting for {len(existing_ids)} existing file(s) to settle...")
        await tero.wait_files_processed(existing_ids, timeout=120.0)
    await tero.delete_all_files()
    await tero.delete_docs_tool()

    print(f"Configuring agent {agent_id} with skipDescriptions=true...")
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
        await asyncio.sleep(0.05)  # throttle to avoid saturating JWT auth calls to Keycloak

    print(f"All {len(file_ids)} documents uploaded. Waiting for processing...")
    await tero.wait_files_processed(file_ids)
    print("All documents processed.")

    # 5. Retry failed files — transient errors (LangChain time-sync) resolve on re-upload.
    #     Re-uploads create new file IDs; we track them and wait on the new ones.
    seen_errors: set[int] = set()
    replaced: set[int] = set()  # old ERROR IDs that were successfully re-uploaded
    for attempt in range(1, 4):
        error_files = await tero.list_error_files()
        fresh = [(fid, name) for fid, name in error_files if fid not in seen_errors]
        if not fresh:
            break
        print(f"\nRetry attempt {attempt}: {len(fresh)} file(s) in ERROR. Re-uploading...")
        retry_ids: list[int] = []
        for fid, name in fresh:
            # Parse document index from filename (doc_NNNNNN.txt)
            try:
                idx = int(name.removeprefix("doc_").removesuffix(".txt"))
            except ValueError:
                print(f"  WARNING: skipping {name} — unexpected filename format")
                seen_errors.add(fid)
                continue
            try:
                new_fid = await tero.upload_document(name, corpus[idx].encode("utf-8"))
                retry_ids.append(new_fid)
                replaced.add(fid)
                seen_errors.add(fid)
            except Exception as exc:
                print(f"  WARNING: re-upload failed for {name}: {exc}")
            await asyncio.sleep(2)
        if retry_ids:
            await tero.wait_files_processed(retry_ids, timeout=120.0)
        remaining = [(fid, _) for fid, _ in await tero.list_error_files() if fid not in seen_errors]
        print(f"  After retry: {len(remaining)} new ERROR(s).")
    else:
        still = [(fid, _) for fid, _ in await tero.list_error_files() if fid not in seen_errors]
        if still:
            print(f"WARNING: {len(still)} file(s) still in ERROR after 3 retry attempts.")

    # Clean up orphaned ERROR records that were successfully replaced
    if replaced:
        deleted = await tero.delete_files(list(replaced))
        print(f"Cleaned up {deleted} orphaned ERROR record(s).")

    # 6. Cost report and wall-clock time
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
        await _run_csv_mode(args, args.judge_model)
        return

    # ── Live mode ──
    if args.agent_id is None:
        env_id = os.environ.get("RAG_EVAL_AGENT_ID", "").strip()
        if env_id:
            try:
                args.agent_id = int(env_id)
            except ValueError:
                print(f"ERROR: RAG_EVAL_AGENT_ID must be an integer, got: {env_id!r}", file=sys.stderr)
                sys.exit(1)
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


    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        print("ERROR: --models must contain at least one Tero model ID.", file=sys.stderr)
        sys.exit(1)

    agent_id = args.agent_id
    tero = TeroClient(args.base_url, agent_id, bearer_token)

    print(f"Dataset : {args.dataset} — agent ID: {agent_id}")
    print("Evaluating against pre-indexed agent (no uploads).")

    _openai_client, judge_llm, cost_tracker, judge_label = _build_judge_client(args.judge_model)
    print(f"Judge LLM: {judge_label}")
    context_recall, context_precision, faithfulness, correctness, citation_faithfulness = _build_metrics(judge_llm)

    # Load questions + corpus (seed=args.seed for deterministic selection);
    # the corpus resolves gold content for the deterministic @5 metrics.
    rows, corpus = ds_module.load_one(args.dataset, n=args.max_questions, seed=args.seed)
    rows = _annotate_gold(rows, corpus, max_docs=getattr(args, "max_docs", None))
    _print_alignment_report(rows)
    loaded_datasets: dict[str, tuple[list[dict], list[str]]] = {args.dataset: (rows, corpus)}

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

            print(f"\nRunning evaluation over {len(rows)} questions (concurrency={args.concurrency})...")
            _concurrency_sem = asyncio.Semaphore(args.concurrency)

            @experiment()
            async def run_experiment(row):
                async with _concurrency_sem:
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

            experiment_results = await run_experiment.arun(ragas_dataset)
            experiment_results.save()

            # Re-write CSV with semicolon separator for Excel compatibility
            _csv_to_semicolon(dataset_experiments_dir / "experiments")

            print(f"Evaluation complete for '{dataset_name}' / '{model_id}'.")

            # Build results DataFrame from ragas experiment results directly
            # (avoids CSV quoting issues from _csv_to_semicolon conversion).
            results_df = None
            try:
                results_df = experiment_results.to_pandas()
            except Exception:
                pass  # Graceful degradation — proceed without df

            stats = analysis.compute_stats(experiment_results, dataset_name)

            # Run sanity checks on results (after CSV write)
            if results_df is not None:
                from sanity_checks import run_sanity_checks as _run_sc
                _run_sc(results_df, dataset_name)
                if "parametric_suspect" in results_df.columns:
                    # Find the CSV ragas just wrote and re-save with the new column
                    experiments_dir = dataset_experiments_dir / "experiments"
                    csv_files = sorted(experiments_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if csv_files:
                        results_df.to_csv(csv_files[0], sep=";", index=False)

            analysis.print_summary(stats, df=results_df, dataset=dataset_name)
            _print_alignment_report(rows, results_df)
            all_stats[model_id][dataset_name] = stats

            if args.update_baseline:
                _save_baseline(model_id, dataset_name, stats, args.max_questions,
                               agent_id=agent_id, seed=args.seed, judge_model=args.judge_model)

            if args.compare:
                baseline = _load_baseline(model_id, dataset_name)
                if baseline is None:
                    print(f"\nNo baseline for '{model_id}/{dataset_name}'. Run with --update-baseline first.")
                else:
                    _print_comparability_report(
                        baseline,
                        _comparability_metadata(agent_id, args.seed, args.judge_model),
                    )
                    analysis.print_comparison(baseline["stats"], stats)

    if len(model_ids) > 1:
        analysis.print_model_comparison(all_stats)

    if cost_tracker:
        cost_tracker.cost_summary()


# ------------------------------------------------------------------
# B2 — pool-probe subcommand + gate report (D6/AD-8)
# ------------------------------------------------------------------

# Gate bands evaluated at the reranker's fetch_k depth (proposal: ≥ ~90%
# gold-in-pool → proceed, 70–90% → conditional, < 70% → hold). The thresholds
# are documented defaults calibrated on the actual gate report; the final
# rerank authorization stays a documented operator decision.
_PROBE_GATE_PROCEED_MIN = 0.90
_PROBE_GATE_CONDITIONAL_MIN = 0.70
# The reranker only reorders the fetch_k pool, so the verdict reads the deepest
# configured depth inside this band instead of the deepest pool probed.
_PROBE_GATE_MAX_DEPTH = 50

_PROBE_GATE_PROCEED = (
    "proceed — the gold document reaches the dense pool at the fetch_k depth: "
    "a reranker over the pool is viable (raise fetch_k and rerank)."
)
_PROBE_GATE_CONDITIONAL = (
    "conditional — gold coverage at the fetch_k depth is partial: reranking can "
    "only help the questions whose gold document is already in the pool, so "
    "document that coverage caveat with the decision."
)
_PROBE_GATE_HOLD = (
    "hold — the gold document mostly falls outside the dense pool at the "
    "fetch_k depth: fix retrieval coverage before enabling a reranker."
)
_PROBE_GATE_INCONCLUSIVE = (
    "inconclusive — no scorable questions (every question is out of corpus or "
    "carries no gold labels)."
)


def _format_rate(rate: float | None) -> str:
    """Render a 0..1 rate for the terminal; unscorable (None) prints as N/A."""
    return "N/A" if rate is None else f"{rate:.1%}"


def _probe_gate_depth(depths: Sequence[int]) -> int:
    """Depth the band decision is evaluated at: deepest configured depth ≤ 50.

    Falls back to the shallowest configured depth when the list has no depth in
    the fetch_k band, so the verdict always names a depth that was probed.
    """
    ordered = sorted(int(depth) for depth in depths)
    within_band = [depth for depth in ordered if depth <= _PROBE_GATE_MAX_DEPTH]
    return max(within_band) if within_band else ordered[0]


def _probe_gate_verdict(in_rate: float | None, gate_depth: int) -> str:
    """Band decision from the in-pool rate at the gate depth.

    `None` (no scorable question) is inconclusive; otherwise the proposal's
    pinned bands apply: ≥ 0.90 proceed, 0.70–0.90 conditional, < 0.70 hold.
    """
    if in_rate is None:
        return _PROBE_GATE_INCONCLUSIVE
    if in_rate >= _PROBE_GATE_PROCEED_MIN:
        return _PROBE_GATE_PROCEED
    if in_rate >= _PROBE_GATE_CONDITIONAL_MIN:
        return _PROBE_GATE_CONDITIONAL
    return _PROBE_GATE_HOLD


def _print_probe_gate_report(report: dict) -> str:
    """Print the dataset-level gate report; return the verdict line (spec: aggregate gate).

    Consumes the frozen `pool_probe.probe_pool` report shape: in/out rates at
    every probed depth, the scorable / out-of-corpus / no-label populations
    separately, the row-identity source, the duplicate-document count and the
    mandatory replication caveat. All labels are dataset-generic.
    """
    depths = report["depths"]
    gate_depth = _probe_gate_depth(depths)
    print("\n=== Dense pool gate (gold-in-pool) ===")
    print(f"Embedding model      : {report['model']}")
    print(f"Questions            : {report['n_questions']} "
          f"(scored {report['n_scored']}, out-of-corpus {report['n_out_of_corpus']}, "
          f"no-label {report['n_no_gold_labels']})")
    print(f"Corpus               : {report['corpus_size']} documents "
          f"(probed prefix {report['effective_corpus_size']}, "
          f"duplicate documents {report['n_duplicate_documents']})")
    print(f"Row identity         : {report['row_id_source']}")
    for depth in depths:
        print(f"Gold-in-pool @{depth:<6}: in {_format_rate(report['rates']['in'][depth])}, "
              f"out {_format_rate(report['rates']['out'][depth])}")
    print(f"Gate depth           : {gate_depth} "
          f"(deepest configured depth <= {_PROBE_GATE_MAX_DEPTH}; a reranker only "
          "reorders the fetch_k pool)")
    verdict = _probe_gate_verdict(report["rates"]["in"][gate_depth], gate_depth)
    print(f"Gate                 : {verdict}")
    print(f"Caveat               : {report['caveat']}")
    return verdict


def do_pool_probe(args: argparse.Namespace) -> None:
    """Run the offline dense-pool probe for the selected dataset and print the gate report.

    Offline by contract (spec R1): rows and corpus come from the local loader
    and embeddings go straight to the OpenAI API — no PostgreSQL/PGVector
    driver, no backend calls. `pool_probe` is imported lazily so the other
    subcommands never load it.
    """
    import pool_probe

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY env var is required for the pool probe "
              "(it embeds via the OpenAI API — no backend is used).", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.dataset} rows (n={args.questions}, seed={args.seed})...")
    rows, corpus = ds_module.load_one(args.dataset, n=args.questions, seed=args.seed)
    print(f"Loaded {len(rows)} questions and {len(corpus)} corpus documents.")

    embed_fn = pool_probe.build_openai_embed_fn(model=args.embedding_model, api_key=api_key)
    report = pool_probe.probe_pool(
        rows, corpus, embed_fn,
        model=args.embedding_model,
        depths=args.depths,
        max_docs=args.max_docs,
        cache_dir=args.cache_dir,
    )
    print(f"Embedding cache      : {args.cache_dir}")
    _print_probe_gate_report(report)


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
