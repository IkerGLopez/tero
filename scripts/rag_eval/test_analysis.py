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


# ---------------------------------------------------------------------------
# Multi-gold metric visibility (spec: eval-runner-metrics — New-metric visibility)
# ---------------------------------------------------------------------------

class TestRetrievalMetricVisibility:
    """The deterministic retrieval metrics are visible in summary and CSV rows."""

    EXISTING_ORDER = [
        "grounded_correctness",
        "correctness",
        "faithfulness",
        "context_recall",
        "context_precision",
        "citation_faithfulness",
        "table_recall_5",
        "cell_recall_5",
    ]
    NEW_METRICS = ["doc_recall_5", "sentence_recall_5", "doc_coverage_5"]

    def test_new_metrics_are_appended_after_cell_recall(self):
        """Spec: ordering of existing metrics is preserved and new rows are appended."""
        from analysis import METRICS

        assert METRICS[:len(self.EXISTING_ORDER)] == self.EXISTING_ORDER
        assert METRICS[len(self.EXISTING_ORDER):] == self.NEW_METRICS

    def test_summary_prints_the_new_metric_rows(self, capsys):
        """Spec scenario: a run with the new columns prints them as summary rows."""
        import pandas as pd
        from analysis import _stats_from_df, print_summary

        df = pd.DataFrame({
            "doc_recall_5": [1.0, 0.0, 1.0],
            "sentence_recall_5": [0.5, 0.25, 1.0],
            "doc_coverage_5": [0.75, 0.5, 1.0],
        })
        print_summary(_stats_from_df(df))

        out = capsys.readouterr().out
        for metric in self.NEW_METRICS:
            assert metric in out, f"{metric} must appear in the summary"
        assert "0.6667" in out, "doc_recall_5 mean (2/3) must be printed"

    def test_summary_is_unchanged_without_the_new_columns(self, capsys):
        """Spec scenario: runs without the new columns keep their previous output."""
        from analysis import print_summary

        print_summary({"correctness": {"mean": 3.5, "std": 0.5}})

        out = capsys.readouterr().out
        for metric in self.NEW_METRICS:
            assert metric not in out

    def test_new_metrics_persist_as_per_question_columns(self, tmp_path):
        """Spec scenario: each metric is a per-question CSV column; None is excluded."""
        import pandas as pd
        from analysis import _stats_from_df

        df = pd.DataFrame({
            "question": ["q1", "q2"],
            "table_recall_5": [None, None],
            "cell_recall_5": [None, None],
            "doc_recall_5": [1.0, None],
            "sentence_recall_5": [0.5, None],
            "doc_coverage_5": [0.5, None],
        })
        csv_path = tmp_path / "results.csv"
        df.to_csv(csv_path, sep=";", index=False)

        loaded = pd.read_csv(csv_path, sep=";")
        # Existing columns stay readable next to the new per-question columns
        assert {"question", "table_recall_5", "cell_recall_5"}.issubset(loaded.columns)
        assert list(loaded["doc_recall_5"].dropna()) == [1.0]

        stats = _stats_from_df(loaded)
        # The excluded question (None) is dropped from the denominator — never counted 0
        assert stats["doc_recall_5"]["mean"] == pytest.approx(1.0)
        assert stats["sentence_recall_5"]["mean"] == pytest.approx(0.5)
        assert stats["doc_coverage_5"]["mean"] == pytest.approx(0.5)
        # An all-None column contributes no summary row at all
        assert "table_recall_5" not in stats
        assert "cell_recall_5" not in stats


# ---------------------------------------------------------------------------
# Paired analysis — bootstrap CI + exact McNemar (design AD-7)
# ---------------------------------------------------------------------------

class TestExactMcNemar:
    """Two-sided exact McNemar over the discordant pairs, hand-checkable."""

    def test_matches_the_hand_computed_binomial_sum(self):
        """b=1, c=8 → 2·(C(9,0)+C(9,1))/2⁹ = 20/512."""
        from analysis import exact_mcnemar

        assert exact_mcnemar(1, 8) == pytest.approx(20 / 512)

    def test_symmetric_in_its_arguments(self):
        from analysis import exact_mcnemar

        assert exact_mcnemar(1, 8) == pytest.approx(exact_mcnemar(8, 1))

    def test_no_discordant_pairs_is_one(self):
        from analysis import exact_mcnemar

        assert exact_mcnemar(0, 0) == 1.0

    def test_tied_discordant_pairs_are_one(self):
        from analysis import exact_mcnemar

        assert exact_mcnemar(3, 3) == 1.0

    def test_lopsided_table_is_significant(self):
        """b=0, c=10 → 2·C(10,0)/2¹⁰."""
        from analysis import exact_mcnemar

        assert exact_mcnemar(0, 10) == pytest.approx(2 / 1024)


class TestMcNemarTable:
    """The 2×2 table of paired binary outcomes."""

    def _pairs(self):
        return [(1.0, 1.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)]

    def test_counts_the_four_cells(self):
        from analysis import mcnemar_table

        assert mcnemar_table(self._pairs()) == {
            "a_only": 1, "b_only": 1, "both": 1, "neither": 1,
        }

    def test_any_non_zero_value_counts_as_a_hit(self):
        from analysis import mcnemar_table

        table = mcnemar_table([(0.5, 0.0), (0.0, 0.25), (1.0, 1.0)])
        assert table == {"a_only": 1, "b_only": 1, "both": 1, "neither": 0}

    def test_empty_pairs_give_an_empty_table(self):
        from analysis import mcnemar_table

        assert mcnemar_table([]) == {"a_only": 0, "b_only": 0, "both": 0, "neither": 0}


class TestPairedBootstrapCi:
    """Bootstrap CI over the paired difference vector (BCa, percentile fallback)."""

    def test_point_estimate_is_the_mean_delta(self):
        from analysis import paired_bootstrap_ci

        result = paired_bootstrap_ci([0.1, 0.2, 0.3])
        assert result["delta"] == pytest.approx(0.2)

    def test_interval_brackets_the_point_estimate(self):
        from analysis import paired_bootstrap_ci

        result = paired_bootstrap_ci([0.1, 0.2, 0.3, 0.15, 0.25])
        assert result["ci_low"] <= result["delta"] <= result["ci_high"]

    def test_seeded_reproducibility(self):
        from analysis import paired_bootstrap_ci

        deltas = [0.1, -0.05, 0.3, 0.2, 0.0]
        assert paired_bootstrap_ci(deltas, rng_seed=14) == paired_bootstrap_ci(deltas, rng_seed=14)

    def test_bca_is_the_default_method(self):
        from analysis import paired_bootstrap_ci

        result = paired_bootstrap_ci([0.1, 0.2, 0.3, 0.4])
        assert result["ci_method"] == "BCa"
        assert result["degenerate"] is False

    def test_degenerate_vector_falls_back_to_percentile(self):
        from analysis import paired_bootstrap_ci

        result = paired_bootstrap_ci([0.5] * 8)
        assert result["ci_method"] == "percentile"
        assert result["degenerate"] is True
        assert result["ci_low"] == pytest.approx(0.5)
        assert result["ci_high"] == pytest.approx(0.5)

    def test_fewer_than_two_pairs_is_insufficient(self):
        from analysis import paired_bootstrap_ci

        result = paired_bootstrap_ci([0.3])
        assert result["ci_method"] == "insufficient"
        assert result["ci_low"] is None
        assert result["ci_high"] is None
        assert result["delta"] == pytest.approx(0.3)

    def test_records_the_seed_and_resample_count(self):
        from analysis import paired_bootstrap_ci

        result = paired_bootstrap_ci([0.1, 0.2], rng_seed=7, n_resamples=1999)
        assert result["rng_seed"] == 7
        assert result["n_resamples"] == 1999


class TestBuildPairs:
    """Pairing two result frames by question, dropping unusable pairs loudly."""

    METRIC = "doc_recall_5"

    def _frame(self, questions, values):
        import pandas as pd

        return pd.DataFrame({"question": questions, self.METRIC: values})

    def test_pairs_matched_questions(self):
        from analysis import build_pairs

        result = build_pairs(
            self._frame(["q1", "q2"], [1.0, 0.0]),
            self._frame(["q1", "q2"], [0.0, 1.0]),
            metric=self.METRIC,
        )

        assert result["n_pairs"] == 2
        assert result["pairs"] == [(1.0, 0.0), (0.0, 1.0)]
        assert result["questions"] == ["q1", "q2"]
        assert result["n_dropped_not_applicable"] == 0
        assert result["n_dropped_unmatched"] == 0

    def test_drops_non_applicable_pairs(self):
        """A metric missing on either side (excluded question) is not a pair."""
        from analysis import build_pairs

        result = build_pairs(
            self._frame(["q1", "q2", "q3"], [1.0, None, 0.5]),
            self._frame(["q1", "q2", "q3"], [0.0, 1.0, float("nan")]),
            metric=self.METRIC,
        )

        assert result["n_pairs"] == 1
        assert result["pairs"] == [(1.0, 0.0)]
        assert result["n_dropped_not_applicable"] == 2
        assert any("not-applicable" in warning for warning in result["warnings"])

    def test_drops_unmatched_questions_with_a_warning(self):
        from analysis import build_pairs

        result = build_pairs(
            self._frame(["q1", "q2"], [1.0, 0.0]),
            self._frame(["q1", "q3"], [0.0, 1.0]),
            metric=self.METRIC,
        )

        assert result["n_pairs"] == 1
        assert result["n_dropped_unmatched"] == 2
        assert any("unmatched" in warning for warning in result["warnings"])

    def test_missing_metric_column_raises(self):
        from analysis import build_pairs

        with pytest.raises(ValueError, match="Missing required columns"):
            build_pairs(self._frame(["q1"], [1.0]), self._frame(["q1"], [0.0]), metric="absent_metric")


class TestPairedMetricReport:
    """Point delta + CI + exact McNemar composed into one report."""

    METRIC = "doc_recall_5"

    def _frame(self, questions, values):
        import pandas as pd

        return pd.DataFrame({"question": questions, self.METRIC: values})

    def test_composes_delta_interval_and_mcnemar(self):
        from analysis import paired_metric_report

        report = paired_metric_report(
            self._frame(["q1", "q2", "q3", "q4"], [1.0, 1.0, 1.0, 1.0]),
            self._frame(["q1", "q2", "q3", "q4"], [1.0, 0.0, 0.0, 0.0]),
            metric=self.METRIC,
            n_resamples=999,
        )

        assert report["metric"] == self.METRIC
        assert report["n_pairs"] == 4
        assert report["delta"] == pytest.approx(0.75)
        assert report["ci_low"] <= report["delta"] <= report["ci_high"]
        assert report["mcnemar"]["both"] == 1
        assert report["mcnemar"]["a_only"] == 3
        assert report["mcnemar"]["b_only"] == 0
        assert report["mcnemar"]["p_value"] == pytest.approx(exact_mcnemar_of(3, 0))

    def test_single_pair_reports_insufficient_interval(self):
        from analysis import paired_metric_report

        report = paired_metric_report(
            self._frame(["q1"], [1.0]),
            self._frame(["q1"], [0.0]),
            metric=self.METRIC,
        )

        assert report["n_pairs"] == 1
        assert report["ci_method"] == "insufficient"
        assert report["ci_low"] is None


def exact_mcnemar_of(a_only, b_only):
    """Local reference implementation for the hand-checked McNemar expectation."""
    import math

    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(a_only, b_only) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


class TestPairedCli:
    """`analysis.py --paired-a/--paired-b --metric` prints and persists the report."""

    METRIC = "doc_recall_5"

    def _write_arms(self, tmp_path):
        arm_a = tmp_path / "arm_a.csv"
        arm_b = tmp_path / "arm_b.csv"
        header = f"question;response;{self.METRIC}\n"
        arm_a.write_text(header + "q1;answer;1\nq2;answer;1\nq3;answer;0\n", encoding="utf-8")
        arm_b.write_text(header + "q1;answer;0\nq2;answer;1\nq3;answer;0\n", encoding="utf-8")
        return arm_a, arm_b

    def _run_cli(self, argv):
        import sys as _sys
        from unittest.mock import patch
        import analysis

        with patch.object(_sys, "argv", ["analysis.py", *argv]):
            analysis.main()

    def test_cli_prints_the_paired_report(self, tmp_path, capsys):
        arm_a, arm_b = self._write_arms(tmp_path)

        self._run_cli([
            "--paired-a", str(arm_a), "--paired-b", str(arm_b),
            "--metric", self.METRIC, "--n-resamples", "199",
        ])

        out = capsys.readouterr().out
        assert self.METRIC in out
        assert "Delta" in out
        assert "McNemar" in out
        assert "no detectable difference" in out or "favouring arm" in out

    def test_cli_writes_the_json_report(self, tmp_path):
        import json

        arm_a, arm_b = self._write_arms(tmp_path)
        out_path = tmp_path / "report.json"

        self._run_cli([
            "--paired-a", str(arm_a), "--paired-b", str(arm_b),
            "--metric", self.METRIC, "--n-resamples", "199", "--out", str(out_path),
        ])

        report = json.loads(out_path.read_text(encoding="utf-8"))
        assert report["metric"] == self.METRIC
        assert report["n_pairs"] == 3
        assert report["delta"] == pytest.approx(1 / 3)
        assert "mcnemar" in report

    def test_cli_requires_both_arms(self, tmp_path, capsys):
        arm_a, _ = self._write_arms(tmp_path)

        with pytest.raises(SystemExit):
            self._run_cli(["--paired-a", str(arm_a), "--metric", self.METRIC])

        assert "paired-a" in capsys.readouterr().err


