"""
Shared, stdlib-only retrieval-matching primitives.

One deterministic implementation of whitespace normalization, sentence-key
decoding and gold-document resolution, importable by the loader
(`rag_datasets`), the runner (`runner`) and the offline probe (`pool_probe`)
without dragging the heavy `datasets` package into unit tests (design AD-4).

Sentence keys (verified against the live `galileo-ai/ragbench` techqa split,
2026-09-22) have the shape `<doc digits><sentence letters>`: `"4a"`, `"4ab"`.
The digit run is the row-local 0-based document index; the letter run is a
bijective base-26 number offset by one (`a`=0 … `z`=25, `aa`=26, `ab`=27).
"""

from __future__ import annotations

import re

# Leading sentence-key prefix stripped from a stored sentence entry
# (spec: `^[0-9]+[a-z]+\s+`).
SENTENCE_KEY_RE: re.Pattern[str] = re.compile(r"^[0-9]+[a-z]+\s+")

# A whole key: digits (document index) + letters (sentence index), nothing else.
# Live label keys carry a second shape with one trailing dot (`4o.`, `4ab.`) —
# 262 keys across the techqa split, 11 rows would lose every gold document if
# the dot were rejected (OP-1 audit, 2026-09-22). Stored sentence entries never
# carry the dot; only the relevant-keys field does.
_SENTENCE_KEY_RE: re.Pattern[str] = re.compile(r"^([0-9]+)([a-z]+)\.?$")


def normalize_whitespace(text: str) -> str:
    """Line endings normalised, whitespace runs collapsed to one space, outer stripped.

    Collapsing *every* whitespace run — newlines included — is what makes
    substring matching deterministic: a gold sentence containing a line break
    still matches the same sentence inside a retrieved context. Idempotent by
    construction.
    """
    line_endings = text.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"\s+", " ", line_endings).strip()


def strip_sentence_key_prefix(text: str) -> str:
    """Remove a leading `SENTENCE_KEY_RE` match (no-op when absent)."""
    match = SENTENCE_KEY_RE.match(text)
    return text[match.end():] if match else text


def sentence_key_doc_index(key: str) -> int | None:
    """0-based document index of `<digits><base26 letters>`; None when unparseable.

    Letters are bijective base-26 minus one: a=0 … z=25, aa=26, ab=27.
    """
    parsed = _parse_sentence_key(key)
    return None if parsed is None else parsed[0]


def sentence_key_sentence_index(key: str) -> int | None:
    """0-based sentence index inside the key's document; None when unparseable.

    The letter component is bijective base-26 with a zero offset (`a`=0 …
    `z`=25, `aa`=26, `ab`=27); it addresses `documents_sentences[doc][sentence]`.
    """
    parsed = _parse_sentence_key(key)
    return None if parsed is None else parsed[1]


def _parse_sentence_key(key: str) -> tuple[int, int] | None:
    """(document index, sentence index) for one sentence key, or None.

    Keys are lowercased defensively (design AD-2) so a stray uppercase letter
    never drops a gold sentence.
    """
    if not isinstance(key, str):
        return None
    match = _SENTENCE_KEY_RE.match(key.strip().lower())
    if match is None:
        return None
    return int(match.group(1)), _decode_base26_letters(match.group(2))


def _decode_base26_letters(letters: str) -> int:
    """Bijective base-26 minus one: `a`→0, `z`→25, `aa`→26, `ab`→27."""
    value = 0
    for letter in letters:
        value = value * 26 + (ord(letter) - ord("a") + 1)
    return value - 1


def contains_normalized(haystack: str, needle: str) -> bool:
    """Non-empty needle found as a normalized substring of haystack.

    Both sides are whitespace-normalized first, so line breaks and whitespace
    runs never block a real match. An empty (or whitespace-only) needle never
    matches, and containment is one-way: the needle must fit inside the haystack.
    """
    normalized_needle = normalize_whitespace(needle)
    if not normalized_needle:
        return False
    return normalized_needle in normalize_whitespace(haystack)


def context_matches_gold_doc(context: str, gold_doc: str) -> bool:
    """Normalized equality OR normalized containment (context inside gold document).

    Containment in the *context ⊆ gold document* direction is what covers
    documents the indexed chunking splits into several pieces: a retrieved
    chunk is a fragment of its document. An empty normalized context never
    matches (design AD-5); a pathologically short context could match a gold
    document loosely, which is a recorded residual risk, not a hidden threshold.
    """
    normalized_context = normalize_whitespace(context)
    if not normalized_context:
        return False
    normalized_gold = normalize_whitespace(gold_doc)
    if not normalized_gold:
        return False
    return normalized_context == normalized_gold or normalized_context in normalized_gold


def resolve_gold_targets(row: dict, effective_len: int) -> dict | None:
    """Resolve one row's gold linkage against the effective corpus prefix.

    Returns ``None`` when the row carries no gold linkage at all (datasets the
    runner never annotated, which pass through untouched). Otherwise:

        {"gold_doc_ids": list[int],        # in-prefix ids, sorted, de-duplicated
         "gold_out_of_corpus": bool,       # labels exist, none inside the prefix
         "no_gold_labels": bool}           # label availability only

    Both row shapes resolve here: the legacy single `gold_doc_id` (+ the
    loader's `gold_out_of_corpus` flag) and the multi-gold `gold_doc_ids` shape.
    Negative ids and ids at or beyond `effective_len` are never in-prefix. A
    declared out-of-corpus flag is authoritative and leaves no scoring target;
    a no-label row keeps `gold_out_of_corpus` False, because "no labels" and
    "labels outside the prefix" are different populations (spec: *no-gold-label
    rows*, *Gold-out-of-corpus reported separately*).
    """
    has_multi_gold = "gold_doc_ids" in row
    has_legacy_gold = "gold_doc_id" in row or "gold_out_of_corpus" in row
    if not has_multi_gold and not has_legacy_gold:
        return None

    no_gold_labels = bool(row.get("no_gold_labels", False)) if has_multi_gold else False
    if has_multi_gold:
        raw_ids = row.get("gold_doc_ids") or []
    else:
        legacy_id = row.get("gold_doc_id")
        raw_ids = [] if legacy_id is None else [legacy_id]

    gold_doc_ids: list[int] = []
    for raw_id in raw_ids:
        doc_id = int(raw_id)
        if 0 <= doc_id < effective_len and doc_id not in gold_doc_ids:
            gold_doc_ids.append(doc_id)
    gold_doc_ids.sort()

    out_of_corpus = False
    if no_gold_labels:
        gold_doc_ids = []
    else:
        out_of_corpus = bool(row.get("gold_out_of_corpus", False)) or not gold_doc_ids
        if out_of_corpus:
            gold_doc_ids = []

    return {
        "gold_doc_ids": gold_doc_ids,
        "gold_out_of_corpus": out_of_corpus,
        "no_gold_labels": no_gold_labels,
    }
