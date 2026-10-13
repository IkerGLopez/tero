"""Baseline persistence and comparability guards for the rag_eval runner."""

import json
from datetime import datetime, timezone

import config  # Bootstrap: repo .env + sys.path
from config import BASELINE_DIR, EVAL_TOP_K


def _comparability_metadata(agent_id: int | None, seed: int, judge_model: str) -> dict:
    """Current run's comparability inputs, keyed like the baseline metadata (D7)."""
    return {
        "agent_id": agent_id,
        "seed": seed,
        "judge_model": judge_model,
        "top_k": EVAL_TOP_K,
    }


def _baseline_deviations(baseline: dict, metadata: dict) -> list[str]:
    """Compare a baseline's comparability inputs against the current run (D7).

    Returns one human-readable line per deviation. Keys missing from the
    baseline (written before D7) are skipped — nothing was recorded, so there
    is nothing to compare against.
    """
    deviations: list[str] = []
    for key in ("agent_id", "seed", "judge_model", "top_k"):
        if key not in baseline:
            continue
        if baseline[key] != metadata[key]:
            deviations.append(f"{key}: baseline {baseline[key]!r} != current {metadata[key]!r}")
    return deviations


def _print_comparability_report(baseline: dict, metadata: dict) -> list[str]:
    """Print the comparability guard for a --compare run; returns the deviations (D7)."""
    deviations = _baseline_deviations(baseline, metadata)
    if deviations:
        print("\nComparability deviations (baseline vs current run):")
        for deviation in deviations:
            print(f"  - {deviation}")
        print("Deltas below are not like-for-like — align the inputs or update the baseline first.")
    else:
        print("\nComparability: baseline and current run agree on agent, seed, judge and top_k.")
    return deviations


def _load_baseline(model_id: str, dataset: str) -> dict | None:
    path = BASELINE_DIR / model_id / f"{dataset}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _save_baseline(model_id: str, dataset: str, stats: dict, n: int,
                   *, agent_id: int | None, seed: int, judge_model: str) -> None:
    """Persist the baseline stats plus the comparability inputs that produced them (D7)."""
    path = BASELINE_DIR / model_id / f"{dataset}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    baseline = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "dataset": dataset,
        "n": n,
        "agent_id": agent_id,
        "seed": seed,
        "judge_model": judge_model,
        "top_k": EVAL_TOP_K,
        "stats": stats,
    }
    path.write_text(json.dumps(baseline, indent=2))
    print(f"\nBaseline saved to {path}")
