"""Deterministic @5 retrieval metrics (table/cell/document/sentence recall)."""

import config  # Bootstrap: repo .env + sys.path
import re

from retrieval_matching import (
    contains_normalized,
    context_matches_gold_doc,
    normalize_whitespace,
)

# k for the deterministic FeTaQA retrieval metrics (top-k pinned to 5).
_METRIC_K = 5

# Opción B metadata lines are not table cells.
_CELL_METADATA_PREFIXES = ("Title:", "Section:")

# `[uid]` prefixes mark table rows in the retrieved serialization.
_UID_PREFIX_RE = re.compile(r"^\[[^\]]*\]\s*")


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
