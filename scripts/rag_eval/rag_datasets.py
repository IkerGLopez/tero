"""
HuggingFace dataset loaders for RAG evaluation.

Each loader returns:
  - rows: list of question dicts. Every loader returns `question` and
    `grading_notes`; `load_ragbench` additionally returns the RAGBench gold
    linkage fields (`gold_doc_ids`, `gold_sentences`, `gold_sentence_doc_ids`,
    `no_gold_labels`), `load_fetaqa` returns (`feta_id`, `gold_values`,
    `gold_doc_id`, `gold_out_of_corpus`), and `load_stratrag` returns
    (`row_id`, `question_type`, `gold_doc_ids`, `no_gold_labels`).
  - corpus: list of document strings (the FULL corpus of the dataset)

Loaders use a seeded shuffle (random.Random(seed)) to select n questions
deterministically. Corpus is always sequential and complete regardless of seed.
"""

import random

from datasets import load_dataset

from retrieval_matching import (
    normalize_whitespace,
    sentence_key_doc_index,
    sentence_key_sentence_index,
    strip_sentence_key_prefix,
)


ALL_DATASETS = ["ragbench", "fetaqa", "stratrag"]


def _shuffle_select(all_rows: list[dict], n: int, seed: int) -> list[dict]:
    """Select n rows via deterministic shuffle, returned in original order."""
    rng = random.Random(seed)
    indices = list(range(len(all_rows)))
    rng.shuffle(indices)
    selected = sorted(indices[:n])
    return [all_rows[i] for i in selected]


def _sentence_entry_text(documents_sentences, doc_index: int, sentence_index: int) -> str | None:
    """Raw stored sentence at `documents_sentences[doc_index][sentence_index]`.

    The live dataset stores nested pairs `[key, text]` — the text is NOT joined
    with its key (design AD-1) — so the pair is rebuilt into a key-prefixed
    entry. The pre-joined string shape is tolerated as well. A missing document
    or sentence row is a miss, never a crash.
    """
    if not documents_sentences:
        return None
    try:
        sentences = documents_sentences[doc_index]
    except (IndexError, TypeError):
        return None
    try:
        entry = sentences[sentence_index]
    except (IndexError, TypeError):
        return None
    if isinstance(entry, (list, tuple)):
        if len(entry) >= 2:
            return f"{entry[0]} {entry[1]}"
        if len(entry) == 1:
            return str(entry[0])
        return None
    if isinstance(entry, str):
        return entry
    return None


def _gold_linkage(row: dict, row_doc_start: int) -> dict:
    """Resolve one RAGBench row's relevant sentence keys into gold linkage fields.

    Keys are `<row-local document index><sentence letters>` (design AD-2); the
    corpus index of a gold document is the row's document-start offset plus its
    index inside the row. A key whose document index falls outside the row's
    documents — or that cannot be parsed at all — contributes neither a gold id
    nor a sentence, so `gold_sentences` and `gold_sentence_doc_ids` stay
    parallel and every emitted sentence keeps a valid source document index.

    `no_gold_labels` describes label availability only: it is true exactly when
    the row carries no relevant sentence keys, never a statement about corpus
    coverage (which the pipeline decides against the indexed prefix).
    """
    documents = row.get("documents") or []
    documents_sentences = row.get("documents_sentences")
    relevant_keys = row.get("all_relevant_sentence_keys") or []

    gold_doc_ids: set[int] = set()
    gold_sentences: list[str] = []
    gold_sentence_doc_ids: list[int] = []
    seen_keys: set[str] = set()

    for key in relevant_keys:
        if key in seen_keys:
            continue
        seen_keys.add(key)

        doc_index = sentence_key_doc_index(key)
        sentence_index = sentence_key_sentence_index(key)
        if doc_index is None or sentence_index is None:
            continue
        if not 0 <= doc_index < len(documents):
            continue

        corpus_index = row_doc_start + doc_index
        gold_doc_ids.add(corpus_index)

        raw_text = _sentence_entry_text(documents_sentences, doc_index, sentence_index)
        if raw_text is None:
            continue
        text = normalize_whitespace(strip_sentence_key_prefix(raw_text))
        if not text:
            continue
        gold_sentences.append(text)
        gold_sentence_doc_ids.append(corpus_index)

    return {
        "gold_doc_ids": sorted(gold_doc_ids),
        "gold_sentences": gold_sentences,
        "gold_sentence_doc_ids": gold_sentence_doc_ids,
        "no_gold_labels": not relevant_keys,
    }


def load_ragbench(n: int = 10, seed: int = 14) -> tuple[list[dict], list[str]]:
    """
    RAGBench (techqa subset) — technical QA with grounding labels.
    Corpus: all passages from the dataset documents (sequential, no dedup).
    Questions: n rows selected via deterministic shuffle (sorted by original index).

    Gold linkage fields on each row (spec: *RAGBench gold document ids*,
    *gold sentences with source document ids*, *no-gold-label rows*):
      - gold_doc_ids: sorted, de-duplicated corpus indices of the row's gold
        documents (`row_doc_start + key document index`, bounds-checked)
      - gold_sentences: relevant sentence texts with the key prefix stripped
        (`^[0-9]+[a-z]+\\s+`) and whitespace-normalized
      - gold_sentence_doc_ids: corpus index of each sentence's source document,
        index-aligned with `gold_sentences`
      - no_gold_labels: label availability only — true when the row has no
        relevant sentence keys
    """
    ds = load_dataset("galileo-ai/ragbench", "techqa", split="validation")

    all_rows: list[dict] = []
    corpus: list[str] = []

    for idx, row in enumerate(ds):
        # The row's documents start where the corpus currently ends, so a gold
        # key's document index maps to `row_doc_start + docIndex`.
        row_doc_start = len(corpus)

        # Collect ALL documents from ALL rows (sequential, no dedup)
        corpus.extend(row.get("documents", []))

        # Collect all rows for shuffle-and-select
        all_rows.append({
            "question": row["question"],
            "grading_notes": row["response"],
            **_gold_linkage(row, row_doc_start),
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


def _as_int(value) -> int | None:
    """Defensive coercion of a StratRAG gold position (design R5).

    The verified split carries integers; a string position is accepted rather
    than silently dropped, and an unparseable one is skipped (`None`) instead of
    crashing the load. Intentional, tested behavior — not loose typing.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stratrag_question_type(row: dict) -> str:
    """`metadata.question_type` when present, else `"unknown"` (design AD-5)."""
    metadata = row.get("metadata") or {}
    question_type = metadata.get("question_type")
    if isinstance(question_type, str) and question_type:
        return question_type
    return "unknown"


def load_stratrag(n: int = 10, seed: int = 14) -> tuple[list[dict], list[str]]:
    """
    StratRAG — multi-hop QA with distractor documents.
    Corpus: all document texts (sequential, no dedup, no [source] prefix).
    Questions: n rows selected via deterministic shuffle (sorted by original index).

    Gold linkage fields on each row:
      - row_id: the dataset `id` (e.g. "val_000089"); None when the row has none
      - question_type: `metadata.question_type`, else "unknown"
      - gold_doc_ids: sorted, de-duplicated corpus indices resolved through a
        `position → corpus-index` map built while iterating that row's
        `doc_pool`. Never stride arithmetic: `val_000030` carries 7 real
        documents, so a fixed block size mislinks every later row.
      - no_gold_labels: label availability only — true when the row carries no
        `gold_doc_indices`. A position that cannot be resolved (skipped entry,
        out-of-range or unparseable) emits no id and never flips the flag.
    """
    ds = load_dataset("Aryanp088/StratRAG", split="validation")

    all_rows: list[dict] = []
    corpus: list[str] = []

    for idx, row in enumerate(ds):
        # One pass: the branch that appends a document is the branch that
        # records its corpus index, so the map cannot desynchronize from the
        # corpus (design AD-1). Skipped entries leave no position behind.
        position_to_index: dict[int, int] = {}
        for position, doc in enumerate(row.get("doc_pool") or []):
            text = doc.get("text", "")
            if not text:  # filter empty text only
                continue
            position_to_index[position] = len(corpus)
            corpus.append(text)

        raw_indices = row.get("gold_doc_indices") or []
        gold_doc_ids: set[int] = set()
        for raw in raw_indices:
            # Membership lookup, never truthiness: corpus index 0 is a valid
            # gold id (design R13).
            index = position_to_index.get(_as_int(raw))
            if index is not None:
                gold_doc_ids.add(index)

        # Collect all rows for shuffle-and-select
        all_rows.append({
            "question": row["query"],
            "grading_notes": row["reference_answer"],
            "row_id": row.get("id"),
            "question_type": _stratrag_question_type(row),
            "gold_doc_ids": sorted(gold_doc_ids),
            "no_gold_labels": not raw_indices,
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
