"""Per-sample metric computation and question-result processing for the rag_eval runner."""

import config  # Bootstrap: repo .env + sys.path
import asyncio
import hashlib
import math
import time

from ragas import SingleTurnSample

from metrics.deterministic import (
    _RETRIEVAL_METRIC_DEFAULTS,
    _doc_recall_5,
    _sentence_recall_5,
    _doc_coverage_5,
    _table_recall_5,
    _cell_recall_5,
)
from metrics.gold import _find_relevant_chunk_index
from metrics.ragas import _pair_citations_with_contexts


def _sanitize_error(exc: Exception, max_len: int = 500) -> str:
    """Collapse multi-line exception text into a single CSV-safe, typed string.

    The result is ``"{ExceptionType}: {message}"`` — or the bare type name when
    the flattened message is empty (httpx timeout exceptions stringify to ``""``).
    The message is flattened (newlines to `` | ``) and truncated to *max_len*
    chars so that ``df.to_csv()`` never emits a broken row; the type prefix is
    added *after* truncation, so it is never the part that gets cut.
    """
    type_name = type(exc).__name__
    flat = str(exc).replace("\n", " | ").replace("\r", "")
    if len(flat) > max_len:
        flat = flat[:max_len - 3] + "..."
    return f"{type_name}: {flat}" if flat else type_name


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
    except Exception as exc:
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
    def _error_row(exc: Exception, latency_ms: float) -> dict:
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
            "latency_ms": latency_ms,
            "relevant_chunk_position": -1,
            **_RETRIEVAL_METRIC_DEFAULTS,
        }

    # REQ-PIPELINE-HARDENING-025: measure before create_thread() so the create
    # phase itself is covered, not only mid-stream ask_question failures.
    start = time.monotonic()
    try:
        thread_id = await tero.create_thread()
        result = await tero.ask_question(thread_id, row["question"])
    except Exception as exc:
        print(f"  ERROR processing question '{row['question'][:80]}': {_sanitize_error(exc)}")
        return _error_row(exc, latency_ms=round((time.monotonic() - start) * 1000, 2))

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
