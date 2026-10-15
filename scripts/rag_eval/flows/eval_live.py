"""Live eval flow: RAGAS evaluation against a pre-indexed Tero agent."""

import config  # Bootstrap: repo .env + sys.path

import argparse
import asyncio
import os
import sys
from datetime import datetime, timezone
from uuid import uuid4

from ragas import experiment

import analysis
import rag_datasets as ds_module
from config import EXPERIMENTS_DIR, _reranker_state_line
from flows.eval_csv import _run_csv_mode
from judge import _build_judge_client
from metrics.gold import _annotate_gold, _print_alignment_report
from metrics.ragas import _build_metrics
from metrics.sample import _process_question_result
from reporting.baseline import (
    _comparability_metadata,
    _load_baseline,
    _print_comparability_report,
    _save_baseline,
)
from reporting.csv_io import _csv_to_semicolon


async def do_eval(args: argparse.Namespace) -> None:
    """Run RAGAS evaluation against a pre-indexed agent.

    Does NOT upload files or modify tool config. Supports CSV offline mode,
    baseline update/comparison, and all existing RAGAS metrics.
    """
    # ── CSV mode: early return, no Tero dependency ──
    if args.from_csv:
        if args.models:
            print("ERROR: --from-csv and --models are mutually exclusive.", file=sys.stderr)
            sys.exit(1)
        await _run_csv_mode(args, args.judge_model)
        return

    # ── Live mode ──
    if args.agent_id is None:
        env_id = os.environ.get("RAG_EVAL_AGENT_ID", "").strip()
        if env_id:
            try:
                args.agent_id = int(env_id)
            except ValueError:
                print(f"ERROR: RAG_EVAL_AGENT_ID must be an integer, got: {env_id!r}", file=sys.stderr)
                sys.exit(1)
    if args.agent_id is None:
        print("ERROR: --agent-id is required for live eval (or use --from-csv for offline mode).", file=sys.stderr)
        sys.exit(1)
    if not args.models:
        print("ERROR: --models is required for live eval (or use --from-csv for offline mode).", file=sys.stderr)
        sys.exit(1)

    # FIX-3: Lazy TeroClient import — only in live path (REQ-OFFLINE-009)
    from tero_client import TeroClient

    bearer_token = (args.bearer_token or os.environ.get("BEARER_TOKEN", "")).strip()
    if not bearer_token:
        print("ERROR: --bearer-token is required (or set BEARER_TOKEN env var).", file=sys.stderr)
        sys.exit(1)


    model_ids = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_ids:
        print("ERROR: --models must contain at least one Tero model ID.", file=sys.stderr)
        sys.exit(1)

    agent_id = args.agent_id
    tero = TeroClient(args.base_url, agent_id, bearer_token)

    print(f"Dataset : {args.dataset} — agent ID: {agent_id}")
    print("Evaluating against pre-indexed agent (no uploads).")
    print(_reranker_state_line())

    _openai_client, judge_llm, cost_tracker, judge_label = _build_judge_client(args.judge_model)
    print(f"Judge LLM: {judge_label}")
    context_recall, context_precision, faithfulness, correctness, citation_faithfulness = _build_metrics(judge_llm)

    # Load questions + corpus (seed=args.seed for deterministic selection);
    # the corpus resolves gold content for the deterministic @5 metrics.
    rows, corpus = ds_module.load_one(args.dataset, n=args.max_questions, seed=args.seed)
    rows = _annotate_gold(rows, corpus, max_docs=getattr(args, "max_docs", None))
    _print_alignment_report(rows)
    loaded_datasets: dict[str, tuple[list[dict], list[str]]] = {args.dataset: (rows, corpus)}

    all_stats: dict[str, dict[str, dict]] = {}  # {model_id: {dataset: stats}}

    for model_id in model_ids:
        print(f"\n{'#'*60}")
        print(f"# Model: {model_id}")
        print(f"{'#'*60}")
        await tero.set_agent_model(model_id)
        all_stats[model_id] = {}

        for dataset_name in loaded_datasets:
            print(f"\n{'='*60}")
            print(f"Dataset: {dataset_name} (n={args.max_questions})")
            print(f"{'='*60}")

            rows, _ = loaded_datasets[dataset_name]

            print(f"\nRunning evaluation over {len(rows)} questions (concurrency={args.concurrency})...")
            _concurrency_sem = asyncio.Semaphore(args.concurrency)

            @experiment()
            async def run_experiment(row):
                async with _concurrency_sem:
                    result = await _process_question_result(
                        tero, row, judge_llm,
                        context_recall, context_precision, faithfulness,
                        correctness, citation_faithfulness,
                    )
                    # Stamp the evaluated model as the last row key, so every
                    # results CSV carries which model was used for the run.
                    result["model_id"] = model_id
                    return result

            from ragas import Dataset as RagasDataset
            run_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_name = f"{model_id}_{run_ts}_{uuid4().hex[:6]}"
            dataset_experiments_dir = EXPERIMENTS_DIR / dataset_name
            dataset_experiments_dir.mkdir(parents=True, exist_ok=True)
            ragas_dataset = RagasDataset(
                name=run_name,
                backend="local/csv",
                root_dir=str(dataset_experiments_dir),
            )
            for row in rows:
                ragas_dataset.append(row)
            ragas_dataset.save()

            experiment_results = await run_experiment.arun(ragas_dataset)
            experiment_results.save()

            # Re-write CSV with semicolon separator for Excel compatibility
            _csv_to_semicolon(dataset_experiments_dir / "experiments")

            print(f"Evaluation complete for '{dataset_name}' / '{model_id}'.")

            # Build results DataFrame from ragas experiment results directly
            # (avoids CSV quoting issues from _csv_to_semicolon conversion).
            results_df = None
            try:
                results_df = experiment_results.to_pandas()
            except Exception:
                pass  # Graceful degradation — proceed without df

            stats = analysis.compute_stats(experiment_results, dataset_name)

            # Run sanity checks on results (after CSV write)
            if results_df is not None:
                from sanity_checks import run_sanity_checks as _run_sc
                _run_sc(results_df, dataset_name)
                if "parametric_suspect" in results_df.columns:
                    # Find the CSV ragas just wrote and re-save with the new column.
                    # model_id goes back to the end: sanity columns append after it.
                    results_df = _move_model_id_last(results_df)
                    experiments_dir = dataset_experiments_dir / "experiments"
                    csv_files = sorted(experiments_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
                    if csv_files:
                        results_df.to_csv(csv_files[0], sep=";", index=False)

            analysis.print_summary(stats, df=results_df, dataset=dataset_name)
            _print_alignment_report(rows, results_df)
            all_stats[model_id][dataset_name] = stats

            if args.update_baseline:
                _save_baseline(model_id, dataset_name, stats, args.max_questions,
                               agent_id=agent_id, seed=args.seed, judge_model=args.judge_model)

            if args.compare:
                baseline = _load_baseline(model_id, dataset_name)
                if baseline is None:
                    print(f"\nNo baseline for '{model_id}/{dataset_name}'. Run with --update-baseline first.")
                else:
                    _print_comparability_report(
                        baseline,
                        _comparability_metadata(agent_id, args.seed, args.judge_model),
                    )
                    analysis.print_comparison(baseline["stats"], stats)

    if len(model_ids) > 1:
        analysis.print_model_comparison(all_stats)

    if cost_tracker:
        cost_tracker.cost_summary()
