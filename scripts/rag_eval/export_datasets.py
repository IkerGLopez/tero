"""
Export HuggingFace RAG datasets to local files for manual upload to Tero's UI.

Loads datasets via the existing rag_datasets loaders (anchor-first corpus ordering
preserved) and saves two artifacts per dataset:
  - evals/corpus/<dataset>/doc_0000.txt … doc_NNNN.txt   (corpus documents)
  - evals/corpus/<dataset>/questions.json                (rows: question + grading_notes)

Zero Tero dependency — no bearer token, no HTTP calls, no agent ID needed.

Usage:
  # Export all three datasets with default sizes
  python scripts/rag_eval/export_datasets.py

  # Single dataset, custom sizes
  python scripts/rag_eval/export_datasets.py --dataset fetaqa --n 20 --corpus-size 400

  # Questions only, no corpus
  python scripts/rag_eval/export_datasets.py --dataset ragbench --corpus-size 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EVALS_DIR = SCRIPT_DIR / "evals"
CORPUS_DIR = EVALS_DIR / "corpus"

sys.path.insert(0, str(SCRIPT_DIR))
import rag_datasets as ds_module


def export_dataset(name: str, n: int, corpus_size: int) -> None:
    """Load one dataset and save corpus + questions locally.

    Clears any existing files from the dataset subdirectory before writing,
    so that stale docs from a previous export (different --n or --corpus-size)
    don't pollute the corpus.
    """
    loader = ds_module.LOADERS[name]
    rows, corpus = loader(n=n)
    # Apply corpus_size slice locally (removed from loader signature)
    corpus = corpus[:corpus_size]

    dataset_dir = CORPUS_DIR / name

    # Wipe stale files from prior exports
    if dataset_dir.exists():
        for old_file in dataset_dir.iterdir():
            old_file.unlink()
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # --- Corpus: one .txt per document ---
    for i, doc_text in enumerate(corpus):
        filepath = dataset_dir / f"doc_{i:04d}.txt"
        filepath.write_text(doc_text, encoding="utf-8")

    # --- Questions: JSON with question + grading_notes ---
    questions_path = dataset_dir / "questions.json"
    questions_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"  {name}: {len(corpus)} corpus docs, {len(rows)} questions "
          f"→ {dataset_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export HuggingFace RAG datasets to local files for Tero UI upload.",
    )
    parser.add_argument(
        "--dataset",
        choices=ds_module.ALL_DATASETS,
        default=None,
        help="Export a single dataset (default: all three).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of questions to select per dataset (default: 10).",
    )
    parser.add_argument(
        "--corpus-size",
        type=int,
        default=200,
        help="Max corpus documents per dataset (default: 200). Use 0 for questions only.",
    )
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else ds_module.ALL_DATASETS

    print(f"Exporting {len(datasets)} dataset(s) "
          f"(n={args.n}, corpus_size={args.corpus_size})\n")

    for ds_name in datasets:
        export_dataset(ds_name, n=args.n, corpus_size=args.corpus_size)

    print(f"\nDone. Files are in: {CORPUS_DIR}")


if __name__ == "__main__":
    main()
