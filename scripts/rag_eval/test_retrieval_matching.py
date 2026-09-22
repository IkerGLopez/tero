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


# ---------------------------------------------------------------------------
# contains_normalized
# ---------------------------------------------------------------------------

class TestContainsNormalized:
    """A non-empty needle found as a normalized substring of the haystack."""

    def test_finds_needle_across_normalized_whitespace(self):
        """Line breaks and runs in either side do not block a real match."""
        from retrieval_matching import contains_normalized

        assert contains_normalized("first  line\r\nsecond\tline", "first line second") is True
        assert contains_normalized("prefix TEXT suffix", "TEXT") is True

    def test_direction_matters(self):
        """Containment is one-way: the needle must fit inside the haystack."""
        from retrieval_matching import contains_normalized

        assert contains_normalized("a short haystack", "short") is True
        assert contains_normalized("short", "a short haystack") is False

    def test_empty_needle_never_matches(self):
        """A degenerate needle is not a match, even against a degenerate haystack."""
        from retrieval_matching import contains_normalized

        assert contains_normalized("some text", "") is False
        assert contains_normalized("some text", "   \n ") is False
        assert contains_normalized("", "") is False


# ---------------------------------------------------------------------------
# context_matches_gold_doc
# ---------------------------------------------------------------------------

class TestContextMatchesGoldDoc:
    """Normalized equality OR normalized containment (context inside gold document)."""

    def test_normalized_equality(self):
        from retrieval_matching import context_matches_gold_doc

        assert context_matches_gold_doc("Full document text", "Full document text") is True
        assert context_matches_gold_doc("Full  document\ntext ", "Full document text") is True
        assert context_matches_gold_doc("something else", "Full document text") is False

    def test_split_document_containment(self):
        """Spec scenario: a retrieved chunk contained in the gold document matches."""
        from retrieval_matching import context_matches_gold_doc

        gold = "Title: T\nSection: S\nrow one | row two | row three"
        assert context_matches_gold_doc("row one | row two", gold) is True

    def test_containment_direction_is_context_inside_gold(self):
        """A context longer than the gold document is not a match."""
        from retrieval_matching import context_matches_gold_doc

        assert context_matches_gold_doc(
            "gold document plus a lot of unrelated trailing text",
            "gold document",
        ) is False

    def test_empty_context_never_matches(self):
        """AD-5: an empty normalized context never matches, even an empty gold doc."""
        from retrieval_matching import context_matches_gold_doc

        assert context_matches_gold_doc("", "gold document") is False
        assert context_matches_gold_doc("   \r\n ", "gold document") is False
        assert context_matches_gold_doc("", "") is False


# ---------------------------------------------------------------------------
# resolve_gold_targets
# ---------------------------------------------------------------------------

class TestResolveGoldTargets:
    """Resolution table across the legacy, multi-gold, partial and no-label shapes."""

    EFFECTIVE_LEN = 5

    def _resolve(self, row):
        from retrieval_matching import resolve_gold_targets

        return resolve_gold_targets(row, self.EFFECTIVE_LEN)

    def test_rows_without_gold_linkage_pass_through(self):
        """A row carrying no gold keys at all resolves to None (pass-through dataset)."""
        assert self._resolve({"question": "q", "grading_notes": "g"}) is None

    def test_legacy_single_gold_in_prefix(self):
        """FeTaQA shape: one gold id inside the prefix resolves to itself."""
        assert self._resolve({"gold_doc_id": 2, "gold_out_of_corpus": False}) == {
            "gold_doc_ids": [2], "gold_out_of_corpus": False, "no_gold_labels": False,
        }

    def test_legacy_gold_outside_the_prefix_is_out_of_corpus(self):
        """FeTaQA shape: id >= effective_len stays out of corpus."""
        assert self._resolve({"gold_doc_id": 5, "gold_out_of_corpus": False}) == {
            "gold_doc_ids": [], "gold_out_of_corpus": True, "no_gold_labels": False,
        }

    def test_legacy_missing_id_is_out_of_corpus(self):
        """FeTaQA shape: a skipped table (gold_doc_id None) is out of corpus."""
        assert self._resolve({"gold_doc_id": None, "gold_out_of_corpus": True}) == {
            "gold_doc_ids": [], "gold_out_of_corpus": True, "no_gold_labels": False,
        }

    def test_legacy_declared_flag_wins_over_an_in_range_id(self):
        """The loader flag is authoritative: no scoring target when it says so."""
        assert self._resolve({"gold_doc_id": 2, "gold_out_of_corpus": True}) == {
            "gold_doc_ids": [], "gold_out_of_corpus": True, "no_gold_labels": False,
        }

    def test_multi_gold_fully_covered(self):
        """All gold ids inside the prefix → sorted, de-duplicated, not out of corpus."""
        assert self._resolve({
            "gold_doc_ids": [3, 1, 1, 0], "no_gold_labels": False,
        }) == {
            "gold_doc_ids": [0, 1, 3], "gold_out_of_corpus": False, "no_gold_labels": False,
        }

    def test_multi_gold_partially_covered(self):
        """Only the in-prefix subset survives; the row is not out of corpus."""
        assert self._resolve({
            "gold_doc_ids": [1, 4, 9], "no_gold_labels": False,
        }) == {
            "gold_doc_ids": [1, 4], "gold_out_of_corpus": False, "no_gold_labels": False,
        }

    def test_multi_gold_none_inside_the_prefix(self):
        """Labels exist but no gold id is inside the prefix → out of corpus."""
        assert self._resolve({
            "gold_doc_ids": [7, 9], "no_gold_labels": False,
        }) == {
            "gold_doc_ids": [], "gold_out_of_corpus": True, "no_gold_labels": False,
        }

    def test_no_label_row_is_not_out_of_corpus(self):
        """Spec: no-label is its own population — never conflated with out-of-corpus."""
        assert self._resolve({
            "gold_doc_ids": [], "no_gold_labels": True,
        }) == {
            "gold_doc_ids": [], "gold_out_of_corpus": False, "no_gold_labels": True,
        }

    def test_negative_and_boundary_ids_are_not_in_prefix(self):
        """Negative ids and ids at the boundary never count as in-prefix."""
        assert self._resolve({
            "gold_doc_ids": [-1, 4, 5], "no_gold_labels": False,
        }) == {
            "gold_doc_ids": [4], "gold_out_of_corpus": False, "no_gold_labels": False,
        }
