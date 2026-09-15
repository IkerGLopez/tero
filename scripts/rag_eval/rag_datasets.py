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


def _is_degenerate_cell(value) -> bool:
    """Degenerate cells: missing, empty, or the FeTaQA `-` placeholder."""
    if value is None:
        return True
    return str(value).strip() in ("", "-")


def _kept_column_indices(table_array: list[list[str]], protected_values) -> list[int]:
    """Deterministic Opción B cleanup: indices of the columns to keep, left to right.

    ToTTo merged cells leave duplicate/degenerate columns behind (`Party |
    Party` with `-` values). Rules, evaluated per column:
      1. A column referenced by a gold cell value — its header or any of its
         data cells, compared stripped — is always kept, so cleanup never
         drops a cell the gold metrics could match. Degenerate gold values are
         excluded (the metrics filter them, spec D2/D4).
      2. An all-degenerate column (every data cell empty/`-`) is dropped.
      3. A column whose header repeats an earlier KEPT header is dropped.
    Pure and deterministic: same table + gold values → same indices.
    """
    headers = table_array[0]
    data_rows = table_array[1:]
    protected = {
        str(value).strip()
        for value in protected_values
        if not _is_degenerate_cell(value)
    }
    kept: list[int] = []
    kept_headers: set[str] = set()
    for col in range(len(headers)):
        values = [row[col] if col < len(row) else "" for row in data_rows]
        header = str(headers[col]).strip()
        if header in protected or any(str(v).strip() in protected for v in values):
            kept.append(col)
            kept_headers.add(header)
            continue
        if values and all(_is_degenerate_cell(v) for v in values):
            continue
        if header and header in kept_headers:
            continue
        kept.append(col)
        kept_headers.add(header)
    return kept


def _serialize_fetaqa_table(
    table_array: list[list[str]],
    feta_id,
    page_title: str = "",
    section_title: str = "",
    protected_values=(),
) -> str:
    """Serialize one FeTaQA table in the Opción B format.

    Layout: `Title:`/`Section:` metadata lines, a blank line, then one
    pipe-delimited line per `table_array` row. Every line carries a
    `[feta_id_rownum]` UID where `rownum` is the `table_array` index (row 0 is
    the header, matching `highlighted_cell_ids` indexing). Columns are cleaned
    by `_kept_column_indices` first; rows shorter than the kept header emit
    only their own cells (no invented trailing values).
    """
    kept = _kept_column_indices(table_array, protected_values)
    lines = [f"Title: {page_title}", f"Section: {section_title}", ""]
    for row_idx, row in enumerate(table_array):
        cells = [str(row[col]) if col < len(row) else "" for col in kept]
        while cells and cells[-1] == "":
            cells.pop()
        body = " | ".join(cells)
        lines.append(f"[{feta_id}_{row_idx}] | {body} |" if body else f"[{feta_id}_{row_idx}] |")
    return "\n".join(lines)


def load_fetaqa(n: int = 10, seed: int = 14) -> tuple[list[dict], list[str]]:
    """
    FeTaQA — table-grounded QA requiring free-form answers from structured data.
    Corpus: one document per kept table (tables with fewer than 2 rows are
    skipped), sequential, no dedup. Each document is the Opción B format:
    `Title:`/`Section:` metadata lines, a blank line, then one
    `[feta_id_rownum]` pipe-delimited line per `table_array` row (row 0 is the
    header), with duplicate/degenerate columns cleaned deterministically.
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

        # Resolve gold cell values BEFORE serialization: cleanup must never
        # drop them. highlighted_cell_ids are [row, col] pairs over
        # table_array, with the header row at index 0.
        gold_values = [table_array[r][c] for r, c in row["highlighted_cell_ids"]]
        feta_id = row["feta_id"]

        gold_doc_id: int | None = None
        if len(table_array) >= 2:
            gold_doc_id = len(corpus)
            corpus.append(_serialize_fetaqa_table(
                table_array,
                feta_id,
                page_title=row.get("table_page_title", ""),
                section_title=row.get("table_section_title", ""),
                protected_values=gold_values,
            ))

        # Collect all rows for shuffle-and-select
        all_rows.append({
            "question": row["question"],
            "grading_notes": row["answer"],
            "feta_id": feta_id,
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
