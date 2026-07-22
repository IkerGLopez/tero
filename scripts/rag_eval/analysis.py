"""
Analysis of RAGAS experiment results.

Can be called from runner.py or run standalone:
  python scripts/rag_eval/analysis.py --dataset ragbench
"""

import argparse
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).parent
EVALS_DIR = SCRIPT_DIR / "evals"
EXPERIMENTS_DIR = EVALS_DIR / "experiments"
BASELINE_DIR = EVALS_DIR / "baseline"

METRICS = ["grounded_correctness", "correctness", "faithfulness", "context_recall", "context_precision", "citation_faithfulness"]


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
    parser.add_argument("--dataset", required=True, help="Dataset name (ragbench, fetaqa, stratrag)")
    parser.add_argument("--model-id", default="", help="Model ID for baseline lookup")
    parser.add_argument("--compare", action="store_true", help="Compare against saved baseline")
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
