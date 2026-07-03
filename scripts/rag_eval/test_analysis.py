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
