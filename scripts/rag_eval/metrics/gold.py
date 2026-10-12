"""Gold annotation and retrieval-alignment reporting for the rag_eval runner."""

import config  # Bootstrap: repo .env + sys.path

import pandas as pd

from retrieval_matching import resolve_gold_targets


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
