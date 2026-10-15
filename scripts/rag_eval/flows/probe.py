"""Dense-pool probe flow: offline gold-in-pool gate report for the reranker decision."""

import config  # Bootstrap: repo .env + sys.path

import argparse
import os
import sys
from collections.abc import Sequence

import rag_datasets as ds_module


# Gate bands evaluated at the reranker's fetch_k depth (proposal: ≥ ~90%
# gold-in-pool → proceed, 70–90% → conditional, < 70% → hold). The thresholds
# are documented defaults calibrated on the actual gate report; the final
# rerank authorization stays a documented operator decision.
_PROBE_GATE_PROCEED_MIN = 0.90
_PROBE_GATE_CONDITIONAL_MIN = 0.70
# The reranker only reorders the fetch_k pool, so the verdict reads the deepest
# configured depth inside this band instead of the deepest pool probed.
_PROBE_GATE_MAX_DEPTH = 50

_PROBE_GATE_PROCEED = (
    "proceed — the gold document reaches the dense pool at the fetch_k depth: "
    "a reranker over the pool is viable (raise fetch_k and rerank)."
)
_PROBE_GATE_CONDITIONAL = (
    "conditional — gold coverage at the fetch_k depth is partial: reranking can "
    "only help the questions whose gold document is already in the pool, so "
    "document that coverage caveat with the decision."
)
_PROBE_GATE_HOLD = (
    "hold — the gold document mostly falls outside the dense pool at the "
    "fetch_k depth: fix retrieval coverage before enabling a reranker."
)
_PROBE_GATE_INCONCLUSIVE = (
    "inconclusive — no scorable questions (every question is out of corpus or "
    "carries no gold labels)."
)


def _format_rate(rate: float | None) -> str:
    """Render a 0..1 rate for the terminal; unscorable (None) prints as N/A."""
    return "N/A" if rate is None else f"{rate:.1%}"


def _probe_gate_depth(depths: Sequence[int]) -> int:
    """Depth the band decision is evaluated at: deepest configured depth ≤ 50.

    Falls back to the shallowest configured depth when the list has no depth in
    the fetch_k band, so the verdict always names a depth that was probed.
    """
    ordered = sorted(int(depth) for depth in depths)
    within_band = [depth for depth in ordered if depth <= _PROBE_GATE_MAX_DEPTH]
    return max(within_band) if within_band else ordered[0]


def _probe_gate_verdict(in_rate: float | None) -> str:
    """Band decision from the in-pool rate at the gate depth.

    `None` (no scorable question) is inconclusive; otherwise the proposal's
    pinned bands apply: ≥ 0.90 proceed, 0.70–0.90 conditional, < 0.70 hold.
    """
    if in_rate is None:
        return _PROBE_GATE_INCONCLUSIVE
    if in_rate >= _PROBE_GATE_PROCEED_MIN:
        return _PROBE_GATE_PROCEED
    if in_rate >= _PROBE_GATE_CONDITIONAL_MIN:
        return _PROBE_GATE_CONDITIONAL
    return _PROBE_GATE_HOLD


def _print_probe_gate_report(report: dict) -> str:
    """Print the dataset-level gate report; return the verdict line (spec: aggregate gate).

    Consumes the frozen `pool_probe.probe_pool` report shape: in/out rates at
    every probed depth, the scorable / out-of-corpus / no-label populations
    separately, the row-identity source, the duplicate-document count and the
    mandatory replication caveat. All labels are dataset-generic.
    """
    depths = report["depths"]
    gate_depth = _probe_gate_depth(depths)
    print("\n=== Dense pool gate (gold-in-pool) ===")
    print(f"Embedding model      : {report['model']}")
    print(f"Questions            : {report['n_questions']} "
          f"(scored {report['n_scored']}, out-of-corpus {report['n_out_of_corpus']}, "
          f"no-label {report['n_no_gold_labels']})")
    print(f"Corpus               : {report['corpus_size']} documents "
          f"(probed prefix {report['effective_corpus_size']}, "
          f"duplicate documents {report['n_duplicate_documents']})")
    print(f"Row identity         : {report['row_id_source']}")
    for depth in depths:
        print(f"Gold-in-pool @{depth:<6}: in {_format_rate(report['rates']['in'][depth])}, "
              f"out {_format_rate(report['rates']['out'][depth])}")
    print(f"Gate depth           : {gate_depth} "
          f"(deepest configured depth <= {_PROBE_GATE_MAX_DEPTH}; a reranker only "
          "reorders the fetch_k pool)")
    verdict = _probe_gate_verdict(report["rates"]["in"][gate_depth])
    print(f"Gate                 : {verdict}")
    print(f"Caveat               : {report['caveat']}")
    return verdict


def do_pool_probe(args: argparse.Namespace) -> None:
    """Run the offline dense-pool probe for the selected dataset and print the gate report.

    Offline by contract (spec R1): rows and corpus come from the local loader
    and embeddings go straight to the OpenAI API — no PostgreSQL/PGVector
    driver, no backend calls. `pool_probe` is imported lazily so the other
    subcommands never load it.
    """
    import pool_probe

    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        print("ERROR: OPENAI_API_KEY env var is required for the pool probe "
              "(it embeds via the OpenAI API — no backend is used).", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {args.dataset} rows (n={args.questions}, seed={args.seed})...")
    rows, corpus = ds_module.load_one(args.dataset, n=args.questions, seed=args.seed)
    print(f"Loaded {len(rows)} questions and {len(corpus)} corpus documents.")

    embed_fn = pool_probe.build_openai_embed_fn(model=args.embedding_model, api_key=api_key)
    report = pool_probe.probe_pool(
        rows, corpus, embed_fn,
        model=args.embedding_model,
        depths=args.depths,
        max_docs=args.max_docs,
        cache_dir=args.cache_dir,
    )
    print(f"Embedding cache      : {args.cache_dir}")
    _print_probe_gate_report(report)
