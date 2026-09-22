"""
Tests for retrieval_matching.py — stdlib-only matching primitives shared by the
loader, the runner and the offline pool probe.

Spec: rag-eval-dataset-loading (key prefix stripping, multi-letter key decoding)
and eval-runner-metrics (normalized document identity).
Design: AD-2 (key decoding), AD-4 (one shared stdlib-only module).

Every test here is a pure-function test: no mocks, no network, no file I/O.
"""

import os
import sys

# Make retrieval_matching importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# normalize_whitespace
# ---------------------------------------------------------------------------

class TestNormalizeWhitespace:
    """Whitespace runs collapse to one space, outer whitespace is stripped."""

    def test_line_endings_are_normalized(self):
        """CRLF and lone CR collapse exactly like LF — same stored text."""
        from retrieval_matching import normalize_whitespace

        assert normalize_whitespace("line one\r\nline two") == "line one line two"
        assert normalize_whitespace("line one\rline two") == "line one line two"
        assert normalize_whitespace("line one\nline two") == "line one line two"

    def test_whitespace_runs_collapse_to_single_space(self):
        """Tabs, newlines and space runs all become one space."""
        from retrieval_matching import normalize_whitespace

        assert normalize_whitespace("lots   of \t\n  space") == "lots of space"

    def test_outer_whitespace_is_stripped(self):
        """Leading and trailing whitespace never survives."""
        from retrieval_matching import normalize_whitespace

        assert normalize_whitespace("  padded text \n ") == "padded text"

    def test_empty_and_whitespace_only_inputs(self):
        """Degenerate inputs normalize to the empty string, never to ' '."""
        from retrieval_matching import normalize_whitespace

        assert normalize_whitespace("") == ""
        assert normalize_whitespace("   \t\r\n ") == ""

    def test_idempotent(self):
        """Normalizing an already-normalized string changes nothing."""
        from retrieval_matching import normalize_whitespace

        nasty = "\r\n  Mixed \t\r endings\n\nand   runs \r\n"
        once = normalize_whitespace(nasty)
        assert once == "Mixed endings and runs"
        assert normalize_whitespace(once) == once


# ---------------------------------------------------------------------------
# strip_sentence_key_prefix
# ---------------------------------------------------------------------------

class TestStripSentenceKeyPrefix:
    """`^[0-9]+[a-z]+\\s+` is removed; a text without a key prefix is untouched."""

    def test_strips_single_letter_key(self):
        """Spec scenario: "4a RELEASE NOTES ABSTRACT" → "RELEASE NOTES ABSTRACT"."""
        from retrieval_matching import strip_sentence_key_prefix

        assert strip_sentence_key_prefix("4a RELEASE NOTES ABSTRACT") == "RELEASE NOTES ABSTRACT"

    def test_strips_multi_letter_key(self):
        """Spec scenario: keys beyond `z` (for example `4aa`) strip fully."""
        from retrieval_matching import strip_sentence_key_prefix

        assert strip_sentence_key_prefix("4aa SOMETHING ELSE") == "SOMETHING ELSE"
        assert strip_sentence_key_prefix("0ab ANOTHER ONE") == "ANOTHER ONE"

    def test_noop_when_prefix_absent(self):
        """A text without a key prefix passes through unchanged."""
        from retrieval_matching import strip_sentence_key_prefix

        assert strip_sentence_key_prefix("RELEASE NOTES ABSTRACT") == "RELEASE NOTES ABSTRACT"
        assert strip_sentence_key_prefix("") == ""

    def test_only_the_leading_run_is_consumed(self):
        """The whitespace run after the key is consumed; the rest is preserved."""
        from retrieval_matching import strip_sentence_key_prefix

        assert strip_sentence_key_prefix("4a  padded body") == "padded body"
        assert strip_sentence_key_prefix("4a body with 4a inside") == "body with 4a inside"


# ---------------------------------------------------------------------------
# sentence_key_doc_index
# ---------------------------------------------------------------------------

class TestSentenceKeyDocIndex:
    """The leading digit run of `<digits><letters>` is the 0-based document index."""

    def test_single_letter_keys(self):
        from retrieval_matching import sentence_key_doc_index

        assert sentence_key_doc_index("0a") == 0
        assert sentence_key_doc_index("4a") == 4
        assert sentence_key_doc_index("0z") == 0

    def test_multi_digit_and_multi_letter_keys(self):
        """Verified live shapes: `4a`, `4ab` keep doc index 4."""
        from retrieval_matching import sentence_key_doc_index

        assert sentence_key_doc_index("35c") == 35
        assert sentence_key_doc_index("4ab") == 4
        assert sentence_key_doc_index("1519a") == 1519

    def test_uppercase_is_decoded_defensively(self):
        """Keys are lowercased before matching (AD-2 defensive lowercasing)."""
        from retrieval_matching import sentence_key_doc_index

        assert sentence_key_doc_index("4A") == 4

    def test_trailing_dot_is_tolerated(self):
        """Live label keys carry a second shape — `4o.`, `4ab.` (OP-1 audit).

        The trailing dot appears only in the relevant-keys field (262 keys,
        11 rows would lose every gold document if the dot were rejected), never
        in the stored sentence entries.
        """
        from retrieval_matching import sentence_key_doc_index

        assert sentence_key_doc_index("4o.") == 4
        assert sentence_key_doc_index("4ab.") == 4
        assert sentence_key_doc_index("0a.") == 0

    def test_unparseable_keys_return_none(self):
        """No digits, no letters, or stray characters → None, never a crash."""
        from retrieval_matching import sentence_key_doc_index

        assert sentence_key_doc_index("a") is None
        assert sentence_key_doc_index("4") is None
        assert sentence_key_doc_index("") is None
        assert sentence_key_doc_index("4a b") is None
        assert sentence_key_doc_index("x4a") is None
        assert sentence_key_doc_index("4a..") is None
        assert sentence_key_doc_index("4a.5") is None


# ---------------------------------------------------------------------------
# sentence_key_sentence_index
# ---------------------------------------------------------------------------

class TestSentenceKeySentenceIndex:
    """The letter component is bijective base-26 minus one: a=0 … z=25, aa=26."""

    def test_single_letters(self):
        from retrieval_matching import sentence_key_sentence_index

        assert sentence_key_sentence_index("0a") == 0
        assert sentence_key_sentence_index("4z") == 25

    def test_multi_letter_rollover(self):
        """Spec: base-26 rollover — `aa` is 26, `ab` is 27."""
        from retrieval_matching import sentence_key_sentence_index

        assert sentence_key_sentence_index("0aa") == 26
        assert sentence_key_sentence_index("0ab") == 27
        assert sentence_key_sentence_index("4ab") == 27

    def test_dotted_key_keeps_its_sentence_position(self):
        """The tolerated trailing dot does not change the decoded position."""
        from retrieval_matching import sentence_key_sentence_index

        assert sentence_key_sentence_index("4o.") == 14
        assert sentence_key_sentence_index("4ab.") == 27

    def test_unparseable_keys_return_none(self):
        from retrieval_matching import sentence_key_sentence_index

        assert sentence_key_sentence_index("a") is None
        assert sentence_key_sentence_index("4") is None
        assert sentence_key_sentence_index("") is None
        assert sentence_key_sentence_index("4a b") is None
