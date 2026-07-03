"""
HuggingFace dataset loaders for RAG evaluation.

Each loader returns:
  - rows: list of {question, grading_notes} dicts
  - corpus: list of document strings (the FULL corpus of the dataset)

Loaders use a seeded shuffle (random.Random(seed)) to select n questions
deterministically. Corpus is always sequential and complete regardless of seed.
"""

import random
from itertools import zip_longest

from datasets import load_dataset


ALL_DATASETS = ["ragbench", "fetaqa", "stratrag"]


def _shuffle_select(all_rows: list[dict], n: int, seed: int) -> list[dict]:
    """Select n rows via deterministic shuffle, returned in original order."""
    rng = random.Random(seed)
    indices = list(range(len(all_rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:n])
    return [all_rows[i] for i in selected]


def load_ragbench(n: int = 10, seed: int = 14) -> tuple[list[dict], list[str]]:
    """
    RAGBench (techqa subset) — technical QA with grounding labels.
    Corpus: all passages from the dataset documents (sequential, no dedup).
    Questions: n rows selected via deterministic shuffle (sorted by original index).
    """
    ds = load_dataset("galileo-ai/ragbench", "techqa", split="validation")

    all_rows: list[dict] = []
    corpus: list[str] = []

    for idx, row in enumerate(ds):
        # Collect ALL documents from ALL rows (sequential, no dedup)
        corpus.extend(row.get("documents", []))

        # Collect all rows for shuffle-and-select
        all_rows.append({
            "question": row["question"],
            "grading_notes": row["response"],
        })

    if n == 0:
        return [], corpus

    rows = _shuffle_select(all_rows, n, seed)

    if len(rows) < n:
        print(f"WARNING: ragbench: requested {n} questions but only {len(rows)} available")

    return rows, corpus


def load_fetaqa(n: int = 10, seed: int = 14) -> tuple[list[dict], list[str]]:
    """
    FeTaQA — table-grounded QA requiring free-form answers from structured data.
    Corpus: one document per table row (sequential, no dedup). Each document is
    multi-line: header row on line 1, then "header: value" data rows below.
    Questions: n rows selected via deterministic shuffle (sorted by original index).
    """
    ds = load_dataset("DongfuJiang/FeTaQA", split="validation")

    all_rows: list[dict] = []
    corpus: list[str] = []

    for idx, row in enumerate(ds):
        # Build one corpus document per table row (skip header-only tables)
        table_array = row.get("table_array", [])
        if len(table_array) >= 2:
            headers = table_array[0]
            doc_lines = [" | ".join(headers)]
            for data_row in table_array[1:]:
                pairs = list(zip_longest(headers, data_row, fillvalue=""))[:len(headers)]
                doc_lines.append(" | ".join(f"{h}: {v}" for h, v in pairs))
            corpus.append("\n".join(doc_lines))

        # Collect all rows for shuffle-and-select
        all_rows.append({
            "question": row["question"],
            "grading_notes": row["answer"],
        })

    if n == 0:
        return [], corpus

    rows = _shuffle_select(all_rows, n, seed)

    if len(rows) < n:
        print(f"WARNING: fetaqa: requested {n} questions but only {len(rows)} available")

    return rows, corpus


def load_stratrag(n: int = 10, seed: int = 14) -> tuple[list[dict], list[str]]:
    """
    StratRAG — multi-hop QA with distractor documents.
    Corpus: all document texts (sequential, no dedup, no [source] prefix).
    Questions: n rows selected via deterministic shuffle (sorted by original index).
    """
    ds = load_dataset("Aryanp088/StratRAG", split="validation")

    all_rows: list[dict] = []
    corpus: list[str] = []

    for idx, row in enumerate(ds):
        # Collect ALL docs from ALL rows (sequential, no dedup, no [source] prefix)
        for doc in row.get("doc_pool", []):
            text = doc.get("text", "")
            if text:  # filter empty text only
                corpus.append(text)

        # Collect all rows for shuffle-and-select
        all_rows.append({
            "question": row["query"],
            "grading_notes": row["reference_answer"],
        })

    if n == 0:
        return [], corpus

    rows = _shuffle_select(all_rows, n, seed)

    if len(rows) < n:
        print(f"WARNING: stratrag: requested {n} questions but only {len(rows)} available")

    return rows, corpus


LOADERS: dict[str, callable] = {
    "ragbench": load_ragbench,
    "fetaqa": load_fetaqa,
    "stratrag": load_stratrag,
}


def load_all(n: int = 10, seed: int = 14) -> dict[str, tuple[list[dict], list[str]]]:
    """Load all three datasets. Returns {dataset_name: (rows, corpus)}."""
    return {name: loader(n=n, seed=seed) for name, loader in LOADERS.items()}


def load_one(dataset: str, n: int = 10, seed: int = 14) -> tuple[list[dict], list[str]]:
    """Load a single dataset by name. Used for isolated testing."""
    if dataset not in LOADERS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {', '.join(ALL_DATASETS)}")
    return LOADERS[dataset](n=n, seed=seed)
