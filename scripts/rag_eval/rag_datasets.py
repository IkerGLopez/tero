"""
HuggingFace dataset loaders for RAG evaluation.

Each loader returns:
  - rows: list of question dicts. Every loader returns `question` and
    `grading_notes`; `load_fetaqa` additionally returns gold linkage fields
    (`feta_id`, `gold_values`, `gold_doc_id`, `gold_out_of_corpus`).
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


def _serialize_fetaqa_table(table_array: list[list[str]]) -> str:
    """Serialize one FeTaQA table: header row line + "header: value" data rows.

    Jagged rows are padded/clipped to the header width so every document keeps
    the same column contract.
    """
    headers = table_array[0]
    doc_lines = [" | ".join(headers)]
    for data_row in table_array[1:]:
        pairs = list(zip_longest(headers, data_row, fillvalue=""))[:len(headers)]
        doc_lines.append(" | ".join(f"{h}: {v}" for h, v in pairs))
    return "\n".join(doc_lines)


def load_fetaqa(n: int = 10, seed: int = 14) -> tuple[list[dict], list[str]]:
    """
    FeTaQA — table-grounded QA requiring free-form answers from structured data.
    Corpus: one document per kept table (tables with fewer than 2 rows are
    skipped), sequential, no dedup. Each document is multi-line: header row on
    line 1, then "header: value" data rows below.
    Questions: n rows selected via deterministic shuffle (sorted by original index).

    Gold linkage fields on each row:
      - feta_id: dataset instance id (NOT the dataset position)
      - gold_values: cell values resolved from `highlighted_cell_ids` as
        `table_array[row][col]`; the header row at index 0 participates
      - gold_doc_id: corpus index of the row's serialized table, computed by
        counting only kept tables (matches the `doc_{idx:06d}` upload index);
        None when the row's table was skipped and never serialized
      - gold_out_of_corpus: True when the row's gold document is not in the
        corpus (its table was skipped), so it is never scored as a recall miss
    """
    ds = load_dataset("DongfuJiang/FeTaQA", split="validation")

    all_rows: list[dict] = []
    corpus: list[str] = []

    for idx, row in enumerate(ds):
        # Build one corpus document per table (skip header-only tables).
        # gold_doc_id counts only kept tables, so skipped tables never shift it.
        table_array = row.get("table_array", [])
        gold_doc_id: int | None = None
        if len(table_array) >= 2:
            gold_doc_id = len(corpus)
            corpus.append(_serialize_fetaqa_table(table_array))

        # Resolve gold cell values: highlighted_cell_ids are [row, col] pairs
        # over table_array, with the header row at index 0.
        gold_values = [table_array[r][c] for r, c in row["highlighted_cell_ids"]]

        # Collect all rows for shuffle-and-select
        all_rows.append({
            "question": row["question"],
            "grading_notes": row["answer"],
            "feta_id": row["feta_id"],
            "gold_values": gold_values,
            "gold_doc_id": gold_doc_id,
            "gold_out_of_corpus": gold_doc_id is None,
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


SANITY_CHECKS: dict[str, list[str]] = {
    "ragbench": ["faithfulness_zero_with_citations"],
    "fetaqa": ["parametric_suspect", "correctness_grounded_divergence"],
    "stratrag": [],
}

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
