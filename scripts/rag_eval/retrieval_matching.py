"""
Shared, stdlib-only retrieval-matching primitives.

One deterministic implementation of whitespace normalization, sentence-key
decoding and gold-document resolution, importable by the loader
(`rag_datasets`), the runner (`runner`) and the offline probe (`pool_probe`)
without dragging the heavy `datasets` package into unit tests (design AD-4).
The module grows by slice: key primitives first, gold resolution when the
runner needs it.

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
