"""
Tests for analysis.py — Bug 5: Fix standalone CSV path resolution.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

# Make analysis importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# TASK-1.3 — std() guard for n=1
# ---------------------------------------------------------------------------

class TestStdGuard:
    """Tests that _stats_from_df returns None for std on single value."""

    def _make_df(self, data: dict) -> "pd.DataFrame":
        import pandas as pd
        return pd.DataFrame(data)

    def test_single_value_std_is_none(self):
        """Single row → std omitted from stats (not NaN)."""
        from analysis import _stats_from_df
        df = self._make_df({
            "correctness": [3],
            "faithfulness": [0.9],
        })
        stats = _stats_from_df(df)
        assert "correctness" in stats
        assert stats["correctness"]["mean"] == pytest.approx(3.0)
        assert "std" not in stats["correctness"], "std should be omitted for n=1"
        # Or alternatively: assert stats["correctness"]["std"] is None

    def test_multi_value_std_is_float(self):
        """Multiple rows → std is a valid float."""
        from analysis import _stats_from_df
        df = self._make_df({
            "correctness": [3, 4, 2],
            "faithfulness": [0.9, 0.8, 0.7],
        })
        stats = _stats_from_df(df)
        assert "correctness" in stats
        assert isinstance(stats["correctness"]["std"], float)
        assert stats["correctness"]["std"] > 0


class TestComputeStatsFromCsv:
    """Bug 5: compute_stats_from_csv must search the experiments/ subdirectory."""

    def test_compute_stats_from_csv_reads_from_experiments_subdir(self):
        """CSVs in fetaqa/experiments/ are discovered and stats computed."""
        from analysis import compute_stats_from_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Create the nested structure that matches RAGAS output
            run_dir = tmp / "fetaqa" / "experiments"
            run_dir.mkdir(parents=True)

            # Write a semicolon-separated CSV matching RAGAS output format
            csv_path = run_dir / "results.csv"
            csv_path.write_text(
                "question;response;correctness;faithfulness;context_recall;"
                "context_precision;citation_faithfulness;grounded_correctness\n"
                "Q1;R1;3;0.9;0.8;0.7;supported;0.675\n"
                "Q2;R2;4;0.95;0.85;0.75;supported;0.7125\n",
                encoding="utf-8",
            )

            # Patch EXPERIMENTS_DIR so compute_stats_from_csv uses our temp dir
            with patch("analysis.EXPERIMENTS_DIR", tmp):
                stats = compute_stats_from_csv("fetaqa")

            assert isinstance(stats, dict), "Stats must be a dict"
            assert "correctness" in stats
            assert "faithfulness" in stats
            assert stats["correctness"]["mean"] == pytest.approx(3.5, abs=0.01)
            assert stats["faithfulness"]["mean"] == pytest.approx(0.925, abs=0.01)

    def test_compute_stats_from_csv_handles_missing_dir(self):
        """If fetaqa/experiments/ does not exist, raises FileNotFoundError."""
        from analysis import compute_stats_from_csv

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            # Create ONLY the dataset dir, NOT experiments/ subdir
            (tmp / "fetaqa").mkdir(parents=True)

            with patch("analysis.EXPERIMENTS_DIR", tmp):
                with pytest.raises(FileNotFoundError):
                    compute_stats_from_csv("fetaqa")


# ---------------------------------------------------------------------------
# Phase 2 — print_summary() ordering, distribution, and sanity integration
# ---------------------------------------------------------------------------

class TestPrintSummaryOrdering:
    """Tests that print_summary() prints grounded_correctness first."""

    def test_grounded_correctness_appears_first_in_output(self, capsys):
        """grounded_correctness must be the first metric printed, regardless of dict order."""
        from analysis import print_summary
        # Pass stats in the OLD order (correctness first, grounded last)
        # print_summary must reorder per METRICS list
        stats = {
            "correctness": {"mean": 3.5, "std": 0.5},
            "faithfulness": {"mean": 0.9, "std": 0.1},
            "context_recall": {"mean": 0.8, "std": 0.3},
            "context_precision": {"mean": 0.7, "std": 0.2},
            "citation_faithfulness": {"mean": 0.95, "std": 0.05},
            "grounded_correctness": {"mean": 0.675, "std": 0.1},
        }
        print_summary(stats)
        captured = capsys.readouterr()
        lines = captured.out.split("\n")
        # Find the first metric line (after the header/separator)
        metric_start = False
        first_metric = None
        for line in lines:
            if "---" in line:
                metric_start = True
                continue
            if metric_start and line.strip():
                first_metric = line.strip()
                break
        assert first_metric is not None, "No metrics found in output"
        assert "grounded_correctness" in first_metric, \
            f"Expected grounded_correctness first, got: {first_metric}"

    def test_missing_metric_skipped_gracefully(self, capsys):
        """Stats without grounded_correctness → prints available metrics without error."""
        from analysis import print_summary
        stats = {
            "correctness": {"mean": 3.5},
            "faithfulness": {"mean": 0.9},
        }
        print_summary(stats)
        captured = capsys.readouterr()
        # Should not raise error
        assert "correctness" in captured.out
        assert "faithfulness" in captured.out


class TestPrintDistribution:
    """Tests for print_distribution() — context recall distribution breakdown."""

    def test_distribution_buckets_and_percentages(self, capsys):
        """Distribution shows 3 buckets with counts and percentages."""
        import pandas as pd
        from analysis import print_distribution
        df = pd.DataFrame({
            "context_recall": [0.0, 0.0, 0.3, 0.8, 0.9],
        })
        print_distribution(df)
        captured = capsys.readouterr()
        out = captured.out
        assert "0.0" in out, f"Expected '0.0' bucket, got: {out}"
        assert "40.0%" in out, f"Expected 40.0% for 0.0 bucket (2/5), got: {out}"
        assert "20.0%" in out, f"Expected 20.0% for 0.0-0.5 bucket (1/5), got: {out}"
        assert "0.5" in out, "Expected >0.5 bucket label"

    def test_distribution_handles_missing_column(self, capsys):
        """DataFrame without context_recall → prints N/A, no crash."""
        import pandas as pd
        from analysis import print_distribution
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        print_distribution(df)
        captured = capsys.readouterr()
        assert "N/A" in captured.out or "not found" in captured.out.lower(), \
            f"Expected N/A for missing column, got: {captured.out}"

    def test_distribution_empty_df(self, capsys):
        """Empty DataFrame → does not crash."""
        import pandas as pd
        from analysis import print_distribution
        df = pd.DataFrame({"context_recall": []})
        print_distribution(df)
        captured = capsys.readouterr()
        # Should not crash — empty distribution is valid
        assert True

    def test_distribution_all_zero_recall(self, capsys):
        """All values exactly 0.0 → 100% in 0.0 bucket."""
        import pandas as pd
        from analysis import print_distribution
        df = pd.DataFrame({"context_recall": [0.0, 0.0, 0.0]})
        print_distribution(df)
        captured = capsys.readouterr()
        out = captured.out
        assert "100%" in out or "100.0%" in out, f"Expected 100% for all-zero, got: {out}"


class TestPrintSummarySanity:
    """Tests that print_summary() integrates sanity checks and distribution."""

    def test_sanity_checks_called_before_table(self, capsys):
        """When df is passed, sanity warnings appear before the metric summary."""
        import pandas as pd
        from analysis import print_summary
        df = pd.DataFrame({
            "question": ["Q1"],
            "faithfulness": [0.0],
            "citations": ['["a", "b", "c"]'],
            "correctness": [3],
            "context_recall": [0.5],
            "grounded_correctness": [0.0],
        })
        stats = {
            "grounded_correctness": {"mean": 0.675},
        }
        print_summary(stats, df=df, dataset="ragbench")
        captured = capsys.readouterr()
        out = captured.out
        # Sanity checks section appears before the metrics table
        sanity_pos = out.find("SANITY CHECKS")
        summary_pos = out.find("RAGAS METRICS SUMMARY")
        assert sanity_pos >= 0, "Expected SANITY CHECKS section"
        assert summary_pos >= 0, "Expected RAGAS METRICS SUMMARY"
        assert sanity_pos < summary_pos, \
            "Sanity checks must appear BEFORE metrics summary"

    def test_distribution_called_after_table(self, capsys):
        """Distribution appears after the metric summary table."""
        import pandas as pd
        from analysis import print_summary
        df = pd.DataFrame({
            "context_recall": [0.0, 0.5],
            "grounded_correctness": [0.0, 0.5],
            "correctness": [3, 4],
        })
        stats = {"grounded_correctness": {"mean": 0.25}}
        print_summary(stats, df=df)
        captured = capsys.readouterr()
        out = captured.out
        summary_pos = out.find("RAGAS METRICS SUMMARY")
        dist_pos = out.find("DISTRIBUTION")
        assert summary_pos >= 0, "Expected RAGAS METRICS SUMMARY"
        assert dist_pos >= 0 or "Distribution" in out, \
            "Expected distribution section"
        if dist_pos >= 0:
            assert summary_pos < dist_pos, \
                "Distribution must appear AFTER metrics summary"

    def test_no_df_still_works_backward_compat(self, capsys):
        """print_summary without df works as before (backward compat)."""
        from analysis import print_summary
        stats = {"correctness": {"mean": 3.5, "std": 0.5}}
        print_summary(stats)
        captured = capsys.readouterr()
        assert "RAGAS METRICS SUMMARY" in captured.out
        assert "correctness" in captured.out
