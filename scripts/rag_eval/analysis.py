"""
Analysis of RAGAS experiment results.

Can be called from runner.py or run standalone:
  python scripts/rag_eval/analysis.py --dataset ragbench
"""

import argparse
import json
import math
import statistics
import sys
import warnings
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
EVALS_DIR = SCRIPT_DIR / "evals"
EXPERIMENTS_DIR = EVALS_DIR / "experiments"
BASELINE_DIR = EVALS_DIR / "baseline"

METRICS = [
    "grounded_correctness",
    "correctness",
    "faithfulness",
    "context_recall",
    "context_precision",
    "citation_faithfulness",
    "table_recall_5",
    "cell_recall_5",
    # Deterministic multi-gold retrieval metrics (RAGBench gold linkage);
    # appended so every pre-existing metric keeps its position.
    "doc_recall_5",
    "sentence_recall_5",
    "doc_coverage_5",
]


def compute_stats(experiment_results, dataset: str) -> dict:
    """
    Compute mean and std for each metric from experiment results.
    experiment_results is the object returned by run_experiment.arun().
    """
    base = EXPERIMENTS_DIR / dataset
    csv_path = base / "experiments" / f"{experiment_results.name}.csv"

    if csv_path.exists() and csv_path.stat().st_size > 0:
        try:
            df = pd.read_csv(csv_path, sep=None, engine="python")
        except pd.errors.ParserError:
            df = pd.DataFrame(list(experiment_results))
    else:
        # Fallback: build DataFrame from the experiment results directly
        df = pd.DataFrame(list(experiment_results))

    return _stats_from_df(df)


def compute_stats_from_csv(dataset: str) -> dict:
    """Load the most recent experiment CSV for a dataset and compute stats."""
    dataset_dir = EXPERIMENTS_DIR / dataset
    csvs = sorted((dataset_dir / "experiments").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        raise FileNotFoundError(f"No experiment CSVs found in {dataset_dir}")
    df = pd.read_csv(csvs[0], sep=";")
    return _stats_from_df(df)


def _stats_from_df(df: pd.DataFrame) -> dict:
    stats = {}
    for metric in METRICS:
        if metric not in df.columns:
            continue
        col = pd.to_numeric(df[metric], errors="coerce").dropna()
        if col.empty:
            continue
        metric_stats: dict = {
            "mean": round(col.mean(), 4),
        }
        # REQ-013: std is None for single value (not NaN)
        if len(col) >= 2:
            metric_stats["std"] = round(col.std(), 4)
        stats[metric] = metric_stats
    return stats


def print_model_comparison(all_stats: dict[str, dict[str, dict]]) -> None:
    """
    Print a cross-model comparison table.

    all_stats structure: {model_id: {dataset: {metric: {mean, std}}}}
    """
    model_ids = list(all_stats.keys())
    datasets = sorted({ds for m in all_stats.values() for ds in m})

    col_w = max(18, max(len(m) for m in model_ids) + 2)
    header = f"{'Metric':<28}" + "".join(f"{m:>{col_w}}" for m in model_ids)

    for dataset in datasets:
        print(f"\n=== MODEL COMPARISON — {dataset.upper()} ===")
        print(header)
        print("-" * (28 + col_w * len(model_ids)))
        for metric in METRICS:
            row = f"{metric:<28}"
            for model_id in model_ids:
                val = all_stats.get(model_id, {}).get(dataset, {}).get(metric)
                if val is not None:
                    row += f"{val['mean']:>{col_w}.4f}"
                else:
                    row += f"{'N/A':>{col_w}}"
            print(row)


def print_summary(stats: dict, df=None, dataset: str = "") -> None:
    # Sanity checks (if DataFrame provided)
    if df is not None:
        from sanity_checks import run_sanity_checks
        try:
            run_sanity_checks(df, dataset)
        except Exception as exc:
            print(f"  WARNING: sanity checks failed — {exc}")

    print("\n=== RAGAS METRICS SUMMARY ===")
    print(f"{'Metric':<28} {'Mean':>8} {'Std':>8}")
    print("-" * 46)
    for metric in METRICS:
        values = stats.get(metric)
        if values is None:
            continue
        std_val = values.get("std")
        std_str = f"{std_val:>8.4f}" if std_val is not None else f"{'N/A':>8}"
        print(f"{metric:<28} {values['mean']:>8.4f} {std_str}")

    # Distribution breakdown
    if df is not None:
        print_distribution(df)
    elif df is None:
        # Called without df — no distribution to print
        pass


def print_distribution(df) -> None:
    """Print context recall per-question distribution breakdown.

    Three buckets: exactly 0.0, 0.0–0.5, > 0.5.
    Shows absolute count and percentage for each bucket.
    Always visible — no CLI flag required.
    """
    import pandas as pd

    print("\n=== CONTEXT RECALL DISTRIBUTION ===")

    if "context_recall" not in df.columns:
        print("  N/A — context_recall column not found")
        return

    col = pd.to_numeric(df["context_recall"], errors="coerce").dropna()
    total = len(col)

    if total == 0:
        print("  No data")
        return

    zero_count = int((col == 0.0).sum())
    low_count = int(((col > 0.0) & (col <= 0.5)).sum())
    high_count = int((col > 0.5).sum())

    print(f"  0.0:       {zero_count} ({_pct(zero_count, total)})")
    print(f"  0.0 – 0.5: {low_count} ({_pct(low_count, total)})")
    print(f"  > 0.5:     {high_count} ({_pct(high_count, total)})")


def _pct(count: int, total: int) -> str:
    """Format count/total as percentage string."""
    if total == 0:
        return "0.0%"
    return f"{count / total * 100:.1f}%"


def print_comparison(baseline_stats: dict, current_stats: dict) -> None:
    print("\n=== COMPARISON VS BASELINE ===")
    print(f"{'Metric':<28} {'Baseline':>10} {'Current':>10} {'Delta':>10} {'':>4}")
    print("-" * 60)
    # REQ-017: Per-metric relative thresholds
    THRESHOLDS = {
        "correctness": 0.5,
        "faithfulness": 0.05,
        "context_recall": 0.02,
        "context_precision": 0.02,
        "citation_faithfulness": 0.05,
        "grounded_correctness": 0.05,
    }
    for metric in METRICS:
        if metric not in baseline_stats or metric not in current_stats:
            continue
        base_mean = baseline_stats[metric]["mean"]
        curr_mean = current_stats[metric]["mean"]
        delta = curr_mean - base_mean
        threshold = THRESHOLDS.get(metric, 0.01)
        indicator = "↑" if delta > threshold else ("↓" if delta < -threshold else "→")
        print(f"{metric:<28} {base_mean:>10.4f} {curr_mean:>10.4f} {delta:>+10.4f} {indicator:>4}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse RAGAS experiment results.")
    parser.add_argument("--dataset", default=None, help="Dataset name (ragbench, fetaqa, stratrag)")
    parser.add_argument("--model-id", default="", help="Model ID for baseline lookup")
    parser.add_argument("--compare", action="store_true", help="Compare against saved baseline")
    parser.add_argument("--paired-a", default=None,
                        help="Results CSV of arm A — enables the paired-analysis mode")
    parser.add_argument("--paired-b", default=None,
                        help="Results CSV of arm B — enables the paired-analysis mode")
    parser.add_argument("--metric", default="doc_recall_5",
                        help="Metric to pair on (default: doc_recall_5)")
    parser.add_argument("--rng-seed", type=int, default=14,
                        help="Bootstrap RNG seed (default: 14)")
    parser.add_argument("--n-resamples", type=int, default=9999,
                        help="Bootstrap resamples (default: 9999)")
    parser.add_argument("--out", default=None,
                        help="Write the paired report as JSON to this path")
    args = parser.parse_args()

    if args.paired_a or args.paired_b:
        run_paired_mode(args)
        return

    if not args.dataset:
        parser.error("--dataset is required unless --paired-a/--paired-b are used")

    stats = compute_stats_from_csv(args.dataset)

    dataset_dir = EXPERIMENTS_DIR / args.dataset
    csvs = sorted((dataset_dir / "experiments").glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    df = None
    if csvs:
        df = pd.read_csv(csvs[0], sep=None, engine="python")

    print_summary(stats, df=df, dataset=args.dataset)

    if args.compare:
        # REQ-016: Baseline path is BASELINE_DIR/model_id/dataset.json
        # When --model-id not given, search subdirs for most recent baseline.
        baseline_path = None
        if args.model_id:
            candidate = BASELINE_DIR / args.model_id / f"{args.dataset}.json"
            if candidate.exists():
                baseline_path = candidate
        else:
            # Search all subdirectories, pick most recent by generated_at
            best_ts = ""
            for subdir in sorted(BASELINE_DIR.glob("*")):
                if not subdir.is_dir():
                    continue
                candidate = subdir / f"{args.dataset}.json"
                if candidate.exists():
                    try:
                        data = json.loads(candidate.read_text())
                        ts = data.get("generated_at", "")
                        if ts > best_ts:
                            best_ts = ts
                            baseline_path = candidate
                    except (json.JSONDecodeError, KeyError):
                        continue

        if baseline_path is None:
            print("\nNo baseline found. Run runner.py with --update-baseline first.")
        else:
            baseline = json.loads(baseline_path.read_text())
            print_comparison(baseline["stats"], stats)


# ------------------------------------------------------------------
# Paired analysis — bootstrap CI + exact McNemar (protocol, design AD-7)
# ------------------------------------------------------------------

def exact_mcnemar(a_only: int, b_only: int) -> float:
    """Two-sided exact McNemar p-value over the discordant pairs.

    Exact binomial with p = 0.5 over `n = a_only + b_only` discordant pairs,
    doubled for the two-sided test and capped at 1:
    `2 * sum_{k <= min(a_only, b_only)} C(n, k) / 2^n`. Implemented with
    `math.comb` so no statsmodels dependency is needed (design AD-7).
    """
    discordant = int(a_only) + int(b_only)
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(int(a_only), int(b_only)) + 1))
    return min(1.0, 2 * tail / (2 ** discordant))


def mcnemar_table(pairs: list[tuple[float, float]]) -> dict:
    """2×2 table of the paired binary outcomes.

    A pair counts as a hit when its metric value is greater than zero — the
    deterministic retrieval metrics are 0/1, and a fractional `doc_coverage_5`
    above zero is a hit too. `a_only` / `b_only` are the discordant cells the
    exact test consumes.
    """
    table = {"a_only": 0, "b_only": 0, "both": 0, "neither": 0}
    for a_value, b_value in pairs:
        a_hit = float(a_value) > 0.0
        b_hit = float(b_value) > 0.0
        if a_hit and b_hit:
            table["both"] += 1
        elif a_hit:
            table["a_only"] += 1
        elif b_hit:
            table["b_only"] += 1
        else:
            table["neither"] += 1
    return table


def paired_bootstrap_ci(
    deltas: list[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 9999,
    rng_seed: int = 14,
) -> dict:
    """Bootstrap confidence interval for the mean paired difference.

    Resamples the per-question difference vector (SciPy's paired semantics on
    the difference vector) with a fixed RNG seed, so a re-run reproduces the
    interval exactly. BCa is the default; a degenerate vector (every delta
    identical) makes BCa undefined, so the interval falls back to the
    percentile method and is flagged `degenerate`. Fewer than two pairs is
    `insufficient` — no interval is fabricated from a single observation.
    """
    values = [float(value) for value in deltas]
    result = {
        "delta": statistics.fmean(values) if values else None,
        "ci_low": None,
        "ci_high": None,
        "ci_method": "insufficient",
        "degenerate": False,
        "rng_seed": rng_seed,
        "n_resamples": n_resamples,
    }
    if len(values) < 2:
        return result

    # Lazy imports: analysis.py is imported by the runner, which must keep
    # working in an eval venv where the analysis extras are not installed yet.
    import numpy as np
    from scipy.stats import DegenerateDataWarning, bootstrap

    sample = (np.asarray(values, dtype=float),)
    try:
        with warnings.catch_warnings():
            # A degenerate vector makes BCa's acceleration/bias math undefined;
            # SciPy signals it with either warning depending on the version.
            warnings.simplefilter("error", DegenerateDataWarning)
            warnings.simplefilter("error", RuntimeWarning)
            interval = bootstrap(
                sample, np.mean, confidence_level=confidence,
                n_resamples=n_resamples, method="BCa",
                random_state=np.random.default_rng(rng_seed),
            )
        ci_method, degenerate = "BCa", False
    except (DegenerateDataWarning, RuntimeWarning):
        interval = bootstrap(
            sample, np.mean, confidence_level=confidence,
            n_resamples=n_resamples, method="percentile",
            random_state=np.random.default_rng(rng_seed),
        )
        ci_method, degenerate = "percentile", True

    return {
        **result,
        "ci_low": float(interval.confidence_interval.low),
        "ci_high": float(interval.confidence_interval.high),
        "ci_method": ci_method,
        "degenerate": degenerate,
    }


def build_pairs(df_a: pd.DataFrame, df_b: pd.DataFrame, *, metric: str, key: str = "question") -> dict:
    """Align two result frames by question, dropping pairs that carry no signal.

    A pair is dropped as `not_applicable` when the metric is missing on either
    side (the lossless exclusion for out-of-corpus and no-label questions) and
    as `unmatched` when the question appears in only one arm. Both drops are
    counted and echoed as warnings, so a paired sample smaller than the arms'
    question counts is never silent.
    """
    missing_columns = [
        f"{label}.{column}"
        for label, frame in (("a", df_a), ("b", df_b))
        for column in (key, metric)
        if column not in frame.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    a_values = dict(zip(df_a[key].astype(str), pd.to_numeric(df_a[metric], errors="coerce")))
    b_values = dict(zip(df_b[key].astype(str), pd.to_numeric(df_b[metric], errors="coerce")))

    pairs: list[tuple[float, float]] = []
    questions: list[str] = []
    n_dropped_not_applicable = 0
    n_dropped_unmatched = 0
    for question in sorted(set(a_values) | set(b_values)):
        a_value = a_values.get(question)
        b_value = b_values.get(question)
        if a_value is None or b_value is None:
            n_dropped_unmatched += 1
            continue
        if pd.isna(a_value) or pd.isna(b_value):
            n_dropped_not_applicable += 1
            continue
        pairs.append((float(a_value), float(b_value)))
        questions.append(question)

    warnings_out: list[str] = []
    if n_dropped_unmatched:
        warnings_out.append(
            f"{n_dropped_unmatched} question(s) unmatched between the two arms were dropped")
    if n_dropped_not_applicable:
        warnings_out.append(
            f"{n_dropped_not_applicable} pair(s) dropped as not-applicable "
            "(metric missing on one side)")

    return {
        "pairs": pairs,
        "questions": questions,
        "n_pairs": len(pairs),
        "n_dropped_not_applicable": n_dropped_not_applicable,
        "n_dropped_unmatched": n_dropped_unmatched,
        "warnings": warnings_out,
    }


def paired_metric_report(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    metric: str,
    rng_seed: int = 14,
    n_resamples: int = 9999,
) -> dict:
    """Point delta + bootstrap CI + exact McNemar for one metric across two arms.

    `delta` is the mean of `metric_A - metric_B`, so a positive value favours
    arm A. The claim rule stays with the caller: an improvement is claimed only
    when the 95% CI excludes 0 *and* the exact McNemar p < 0.05; otherwise the
    honest report is "no detectable difference at this sample size".
    """
    pairing = build_pairs(df_a, df_b, metric=metric)
    deltas = [a_value - b_value for a_value, b_value in pairing["pairs"]]
    interval = paired_bootstrap_ci(deltas, rng_seed=rng_seed, n_resamples=n_resamples)
    table = mcnemar_table(pairing["pairs"])
    return {
        "metric": metric,
        "n_pairs": pairing["n_pairs"],
        "n_dropped_not_applicable": pairing["n_dropped_not_applicable"],
        "n_dropped_unmatched": pairing["n_dropped_unmatched"],
        "warnings": pairing["warnings"],
        "delta": interval["delta"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "ci_method": interval["ci_method"],
        "degenerate": interval["degenerate"],
        "rng_seed": interval["rng_seed"],
        "n_resamples": interval["n_resamples"],
        "mcnemar": {**table, "p_value": exact_mcnemar(table["a_only"], table["b_only"])},
    }


def print_paired_report(report: dict) -> None:
    """Print a paired metric report, including the pre-registered claim rule."""
    print("\n=== PAIRED METRIC REPORT ===")
    print(f"Metric        : {report['metric']}")
    print(f"Paired sample : {report['n_pairs']} question(s) "
          f"(dropped: {report['n_dropped_not_applicable']} not-applicable, "
          f"{report['n_dropped_unmatched']} unmatched)")
    for warning in report["warnings"]:
        print(f"  WARNING: {warning}")

    delta = report["delta"]
    if delta is None:
        print("Delta (A - B) : N/A — no paired questions")
    else:
        print(f"Delta (A - B) : {delta:+.4f}")
    if report["ci_method"] == "insufficient":
        print("95% CI        : N/A — fewer than two paired questions")
    else:
        flags = ", degenerate" if report["degenerate"] else ""
        print(f"95% CI        : [{report['ci_low']:+.4f}, {report['ci_high']:+.4f}] "
              f"({report['ci_method']}{flags})")

    table = report["mcnemar"]
    print(f"McNemar table : a_only {table['a_only']}, b_only {table['b_only']}, "
          f"both {table['both']}, neither {table['neither']}")
    print(f"Exact McNemar : p = {table['p_value']:.4f}")

    if delta is None or report["ci_low"] is None or report["ci_high"] is None:
        print("Claim         : no detectable difference at this sample size")
        return
    excludes_zero = report["ci_low"] > 0 or report["ci_high"] < 0
    if excludes_zero and table["p_value"] < 0.05:
        favoured = "A" if delta > 0 else "B"
        print(f"Claim         : detectable difference favouring arm {favoured} "
              "(CI excludes 0 and p < 0.05)")
    else:
        print("Claim         : no detectable difference at this sample size")


def run_paired_mode(args: argparse.Namespace) -> None:
    """CLI branch: pair two result CSVs and print (optionally persist) the report."""
    if not (args.paired_a and args.paired_b):
        print("ERROR: --paired-a and --paired-b must be given together.", file=sys.stderr)
        sys.exit(1)

    df_a = pd.read_csv(args.paired_a, sep=None, engine="python", encoding="utf-8-sig")
    df_b = pd.read_csv(args.paired_b, sep=None, engine="python", encoding="utf-8-sig")
    report = paired_metric_report(
        df_a, df_b,
        metric=args.metric,
        rng_seed=args.rng_seed,
        n_resamples=args.n_resamples,
    )
    print_paired_report(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
