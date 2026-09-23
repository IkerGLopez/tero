"""
Tests for sanity_checks.py — per-dataset evaluation quality checks.

All checks are pure DataFrame operations. No LLM calls.
"""
import os
import sys

import pandas as pd
import pytest

# Make sanity_checks importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# check_faithfulness_zero_with_citations
# ---------------------------------------------------------------------------

class TestCheckFaithfulnessZeroWithCitations:
    """Tests for check_faithfulness_zero_with_citations — RAGBench F3.2."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from sanity_checks import check_faithfulness_zero_with_citations
        self._func = check_faithfulness_zero_with_citations

    def _make_df(self, faithfulness, citations_value):
        """Build a single-row DataFrame with faithfulness and citations columns."""
        return pd.DataFrame({
            "question": ["Test Q1"],
            "faithfulness": [faithfulness],
            "citations": [citations_value],
        })

    def test_zero_faith_with_three_citations_warns(self, capsys):
        """faithfulness=0.0 + ≥3 citations → WARNING printed, score unchanged."""
        df = self._make_df(0.0, '["cite1", "cite2", "cite3"]')
        self._func(df)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "faithfulness=0.0" in captured.out
        assert "citations" in captured.out.lower()
        # DataFrame faithfulness value remains 0.0 (not modified)
        assert df["faithfulness"].iloc[0] == 0.0

    def test_zero_faith_with_fewer_than_three_citations_no_warn(self, capsys):
        """faithfulness=0.0 but <3 citations → no warning."""
        df = self._make_df(0.0, '["cite1", "cite2"]')
        self._func(df)
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_nonzero_faith_no_warning(self, capsys):
        """faithfulness=0.5 with many citations → no warning."""
        df = self._make_df(0.5, '["cite1", "cite2", "cite3", "cite4"]')
        self._func(df)
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_missing_citations_column_no_crash(self, capsys):
        """DataFrame without 'citations' column → does not crash."""
        df = pd.DataFrame({
            "question": ["Q"],
            "faithfulness": [0.0],
        })
        self._func(df)  # should not raise
        captured = capsys.readouterr()
        # No warnings expected when citations column is missing
        assert captured.out.strip() == ""

    def test_malformed_citations_json_handled_gracefully(self, capsys):
        """Malformed JSON in citations cell → treated as empty, no crash."""
        df = pd.DataFrame({
            "question": ["Q"],
            "faithfulness": [0.0],
            "citations": ["not valid json [[["],
        })
        self._func(df)  # should not crash
        # Malformed JSON → no entries → no warning
        captured = capsys.readouterr()
        assert captured.out.strip() == ""


# ---------------------------------------------------------------------------
# check_parametric_suspect
# ---------------------------------------------------------------------------

class TestCheckParametricSuspect:
    """Tests for check_parametric_suspect — FeTaQA F2.2."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from sanity_checks import check_parametric_suspect
        self._func = check_parametric_suspect

    def _make_df(self, correctness, context_recall):
        return pd.DataFrame({
            "question": ["Test Q1"],
            "correctness": [correctness],
            "context_recall": [context_recall],
        })

    def test_high_correctness_zero_recall_is_flagged(self, capsys):
        """correctness=3, context_recall=0.0 → parametric_suspect=True + WARNING."""
        df = self._make_df(3, 0.0)
        self._func(df)
        assert "parametric_suspect" in df.columns
        assert bool(df["parametric_suspect"].iloc[0]) is True
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "parametric" in captured.out.lower()

    def test_correctness_exactly_2_zero_recall_flagged(self, capsys):
        """correctness=2 (boundary), context_recall=0.0 → flagged."""
        df = self._make_df(2, 0.0)
        self._func(df)
        assert bool(df["parametric_suspect"].iloc[0]) is True

    def test_valid_recall_not_flagged(self, capsys):
        """correctness=3, context_recall=0.5 → parametric_suspect=False, no warning."""
        df = self._make_df(3, 0.5)
        self._func(df)
        assert bool(df["parametric_suspect"].iloc[0]) is False
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_low_correctness_zero_recall_not_flagged(self, capsys):
        """correctness=1, context_recall=0.0 → NOT flagged (< 2 threshold)."""
        df = self._make_df(1, 0.0)
        self._func(df)
        assert bool(df["parametric_suspect"].iloc[0]) is False

    def test_column_already_exists_is_idempotent(self, capsys):
        """If parametric_suspect column already exists, skip mutation."""
        df = self._make_df(3, 0.0)
        df["parametric_suspect"] = "existing_value"
        self._func(df)
        # Value preserved — not overwritten
        assert df["parametric_suspect"].iloc[0] == "existing_value"


# ---------------------------------------------------------------------------
# check_correctness_grounded_divergence
# ---------------------------------------------------------------------------

class TestCheckCorrectnessGroundedDivergence:
    """Tests for check_correctness_grounded_divergence — FeTaQA F2.1."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from sanity_checks import check_correctness_grounded_divergence
        self._func = check_correctness_grounded_divergence

    def _make_df(self, correctness, grounded_correctness):
        return pd.DataFrame({
            "question": ["Test Q1"],
            "correctness": [correctness],
            "grounded_correctness": [grounded_correctness],
        })

    def test_large_divergence_triggers_warning(self, capsys):
        """correctness=4, grounded=0.0 → divergence |1.0 - 0.0| = 1.0 > 0.5 → WARNING."""
        df = self._make_df(4, 0.0)
        self._func(df)
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "divergence" in captured.out.lower()
        # Scores remain unchanged
        assert df["correctness"].iloc[0] == 4
        assert df["grounded_correctness"].iloc[0] == 0.0

    def test_small_divergence_no_warning(self, capsys):
        """correctness=3, grounded=0.55 → divergence |0.75 - 0.55| ≈ 0.2 ≤ 0.5 → no warning."""
        df = self._make_df(3, 0.55)
        self._func(df)
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_perfect_grounding_no_warning(self, capsys):
        """Regression (scale-mix bug): correctness=4.0, grounded=1.0 → no warning.

        The pre-fix formula mixed scales (|4.0 - 1.0| = 3.0 > 0.5) and fired
        on every row with correctness ≥ ~2, even with perfect grounding.
        """
        df = self._make_df(4.0, 1.0)
        self._func(df)
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_custom_threshold_respected(self, capsys):
        """threshold=1.0: divergence |0.75 - 0.0| = 0.75 ≤ 1.0 → no warning.

        The same row would warn at the default threshold (0.75 > 0.5).
        """
        df = self._make_df(3, 0.0)
        self._func(df, threshold=1.0)
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_exact_boundary_no_warning(self, capsys):
        """divergence exactly equals threshold (0.5) → no warning (> not ≥)."""
        df = self._make_df(4.0, 0.5)
        self._func(df)  # default threshold=0.5, divergence=0.5
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_missing_columns_no_crash(self, capsys):
        """DataFrame missing grounded_correctness → does not crash, no warnings."""
        df = pd.DataFrame({
            "question": ["Q"],
            "correctness": [4],
        })
        self._func(df)
        captured = capsys.readouterr()
        assert captured.out.strip() == ""

    def test_nan_values_handled_gracefully(self, capsys):
        """NaN in either column → skipped gracefully, no crash."""
        df = pd.DataFrame({
            "question": ["Q"],
            "correctness": [None],
            "grounded_correctness": [None],
        })
        self._func(df)  # should not crash


# ---------------------------------------------------------------------------
# run_sanity_checks — per-dataset activation
# ---------------------------------------------------------------------------

class TestRunSanityChecks:
    """Tests for run_sanity_checks — per-dataset config loading and dispatch."""

    @pytest.fixture(autouse=True)
    def _import_function(self):
        from sanity_checks import run_sanity_checks
        self._func = run_sanity_checks

    def _make_df(self):
        return pd.DataFrame({
            "question": ["Q1", "Q2"],
            "faithfulness": [0.0, 0.5],
            "citations": ['["a", "b", "c"]', '["a"]'],
            "correctness": [3, 1],
            "context_recall": [0.0, 0.5],
            "grounded_correctness": [0.0, 0.5],
        })

    def test_ragbench_activates_faithfulness_only(self, capsys):
        """RAGBench config → only faithfulness check runs."""
        df = self._make_df()
        self._func(df, "ragbench")
        captured = capsys.readouterr()
        assert "SANITY CHECKS" in captured.out
        assert "ragbench" in captured.out.lower()
        # parametric_suspect column NOT added for ragbench
        assert "parametric_suspect" not in df.columns

    def test_fetaqa_activates_both_checks(self, capsys):
        """FeTaQA config → parametric_suspect + divergence checks run."""
        df = self._make_df()
        self._func(df, "fetaqa")
        captured = capsys.readouterr()
        assert "SANITY CHECKS" in captured.out
        assert "fetaqa" in captured.out.lower()
        # parametric_suspect column IS added
        assert "parametric_suspect" in df.columns

    def test_stratrag_activates_both_checks(self, capsys):
        """StratRAG config → parametric_suspect + divergence checks run.

        Q1 (correctness=3, context_recall=0.0) is flagged and warned; the
        faithfulness check stays skipped even though Q1 carries 0.0 faithfulness
        with 3 citations.
        """
        df = self._make_df()
        self._func(df, "stratrag")
        captured = capsys.readouterr()
        assert "SANITY CHECKS" in captured.out
        assert "stratrag" in captured.out.lower()
        # parametric_suspect column IS added, and only Q1 is flagged
        assert "parametric_suspect" in df.columns
        assert bool(df["parametric_suspect"].iloc[0]) is True
        assert bool(df["parametric_suspect"].iloc[1]) is False
        # correctness_grounded_divergence warns on Q1 (|3/4 - 0.0| = 0.75 > 0.5)
        assert "correctness_grounded_divergence" in captured.out
        # faithfulness_zero_with_citations is NOT registered for stratrag
        assert "faithfulness_zero_with_citations" not in captured.out

    def test_unknown_dataset_defaults_empty_checks(self, capsys):
        """Unknown dataset → no checks run, no crash."""
        df = self._make_df()
        self._func(df, "unknown_dataset")
        # Should not crash
        assert "parametric_suspect" not in df.columns

    def test_sanity_checks_header_printed(self, capsys):
        """The header '=== SANITY CHECKS ===' is always printed."""
        df = self._make_df()
        self._func(df, "ragbench")
        captured = capsys.readouterr()
        assert "=== SANITY CHECKS" in captured.out
