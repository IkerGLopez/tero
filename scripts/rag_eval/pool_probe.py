"""
Offline retrieval-pool probe for any gold-linked dataset.

The eval side has no PostgreSQL/PGVector driver, so this module replicates
embedding + exact cosine similarity locally to answer one question before the
rerank phase invests effort:

    does the gold document reach the configured pool depths of the exact
    similarity ranking for the probed corpus prefix?

Design pins (design.md AD-8):
  - Cache keys are content-addressed: ``sha256(model + text)``. Changed content
    misses by construction, and a warm cache lets re-runs skip the embedder
    without changing results.
  - Ranking uses exact cosine over full vectors — never an approximate index.
  - Gold resolution is shared with the runner (`resolve_gold_targets`), so a
    question is scored over its in-prefix gold subset, out-of-corpus questions
    (``--max-docs`` truncation, loader flags) are reported separately from
    out-of-pool ones, and rows without gold labels form their own population —
    never counted as out-of-corpus or as misses.
  - The embed function is injected (test seam). ``build_openai_embed_fn`` is the
    only network path and is never exercised by unit tests.

Report shape (consumed by the `runner.py pool-probe` subcommand):

    {
      "model": str,                       # embedding model identifier
      "depths": [int, ...],               # probed pool depths (gate default 20/50/100/500)
      "corpus_size": int,                 # full loader corpus size
      "effective_corpus_size": int,       # indexed prefix actually probed
      "n_questions": int,
      "n_scored": int,                    # questions with gold inside the prefix
      "n_out_of_corpus": int,             # labeled, no gold inside the prefix
      "n_no_gold_labels": int,            # no gold labels (or no linkage at all)
      "row_id_source": str,               # "loader" | "selection_index"
      "n_duplicate_documents": int,       # repeated content in the FULL corpus
      "rates": {"in": {depth: float | None}, "out": {depth: float | None}},
      "per_question": [
        {
          "row_id": int | str,             # never a `feta_id` key
          "question": str,
          "gold_doc_ids": [int, ...],      # in-prefix gold documents, sorted
          "gold_rank": int | None,         # 1-based best rank; None when unscored
          "gold_ranks": [int, ...],        # one rank per in-prefix gold document
          "n_gold_docs": int,
          "gold_in_pool": {depth: bool | None},
          "gold_out_of_corpus": bool,
          "no_gold_labels": bool,
        },
        ...
      ],
      "caveat": str,                        # mandatory replication caveat
    }

Rates are ``None`` when no question is scorable (empty denominator = N/A, not 0),
and every question lands in exactly one population:
``n_questions == n_scored + n_out_of_corpus + n_no_gold_labels``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from openai import OpenAI

from retrieval_matching import resolve_gold_targets


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
# Gate default: 20/50 are the reranker's fetch_k range, 100/500 show headroom.
DEFAULT_DEPTHS: tuple[int, ...] = (20, 50, 100, 500)
DEFAULT_BATCH_SIZE = 64

# File inside `--cache-dir` holding one JSON object per line: {"key": ..., "vector": ...}.
CACHE_FILENAME = "embeddings.jsonl"

# Mandatory in every report (spec: "Replication caveat").
CAVEAT = (
    "Replicates embedding + exact cosine similarity only; it does NOT reproduce "
    "PGVector ANN/index internals, so these results are necessary but not "
    "sufficient evidence for the live retriever. It also embeds each corpus "
    "document whole, so documents that the indexed chunking splits into several "
    "pieces are a documented approximation: no exact chunk-level parity is "
    "claimed for them."
)

# Batch embedder contract: one vector per input text, same input order.
Embedder = Callable[[Sequence[str]], list[list[float]]]

# Row identity keys, most explicit first. `row_id` is the generic loader field;
# `feta_id` is the FeTaQA dataset identifier kept for backward-compatible values.
_ROW_ID_KEYS: tuple[str, ...] = ("row_id", "feta_id")

# A row the probe cannot score because it carries no gold linkage at all
# (a dataset nobody labeled) counts as no-label: nothing is scored and no
# corpus-coverage claim is made about it.
_UNLINKED_GOLD_TARGET = {"gold_doc_ids": [], "gold_out_of_corpus": False, "no_gold_labels": True}


# ------------------------------------------------------------------
# Content-addressed embedding cache
# ------------------------------------------------------------------

def embedding_cache_key(model: str, text: str) -> str:
    """Cache key for one (model, text) pair: sha256 over model + text.

    The model is part of the key so switching models never reuses stale vectors;
    the text is part of the key so changed content (for example a new corpus
    serialization) invalidates automatically. A NUL byte separates the two parts
    so the two concatenations can never alias.
    """
    digest = hashlib.sha256(model.encode("utf-8") + b"\x00" + text.encode("utf-8"))
    return digest.hexdigest()


class EmbeddingCache:
    """File-backed, content-addressed embedding cache.

    ``cache_dir=None`` keeps the cache in memory for the current run only.
    With a ``cache_dir`` the cache is a JSONL file (one ``{"key", "vector"}``
    object per line) that grows append-only, so a deeper corpus re-run embeds
    only the documents it has never seen.
    """

    def __init__(self, model: str, cache_dir: str | Path | None = None) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._vectors: dict[str, list[float]] = {}
        if self.cache_dir is not None:
            self._vectors.update(self._load())

    def embed(self, texts: Sequence[str], embed_fn: Embedder) -> list[list[float]]:
        """Return one vector per text, calling `embed_fn` only for cache misses.

        Duplicate texts inside one call share a single embedding, and the
        embedder receives the misses as one batch in first-occurrence order.
        """
        missing_keys: list[str] = []
        missing_texts: list[str] = []
        scheduled: set[str] = set()
        for text in texts:
            key = embedding_cache_key(self.model, text)
            if key in self._vectors or key in scheduled:
                continue
            scheduled.add(key)
            missing_keys.append(key)
            missing_texts.append(text)

        if missing_texts:
            vectors = embed_fn(missing_texts)
            if len(vectors) != len(missing_texts):
                raise ValueError(
                    f"embed_fn returned {len(vectors)} vectors for {len(missing_texts)} texts")
            new_entries: dict[str, list[float]] = {}
            for key, vector in zip(missing_keys, vectors):
                stored = [float(value) for value in vector]
                self._vectors[key] = stored
                new_entries[key] = stored
            self._append(new_entries)

        return [self._vectors[embedding_cache_key(self.model, text)] for text in texts]

    # -- persistence -------------------------------------------------

    @property
    def _path(self) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / CACHE_FILENAME

    def _load(self) -> dict[str, list[float]]:
        """Read the cache file; corrupt lines are a miss, never a crash."""
        path = self._path
        if path is None or not path.exists():
            return {}
        vectors: dict[str, list[float]] = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    key = entry["key"]
                    vector = entry["vector"]
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
                if isinstance(key, str) and isinstance(vector, list):
                    vectors[key] = vector
        return vectors

    def _append(self, entries: dict[str, list[float]]) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            for key, vector in entries.items():
                handle.write(json.dumps({"key": key, "vector": vector},
                                        separators=(",", ":")) + "\n")


# ------------------------------------------------------------------
# Exact cosine ranking
# ------------------------------------------------------------------

def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Exact cosine similarity between two full vectors (no ANN, no approximation).

    A zero-norm vector has no angle; it scores 0.0 (no similarity) instead of
    raising, so a degenerate embedding cannot abort a probe run.
    """
    a = np.asarray(vector_a, dtype=np.float64)
    b = np.asarray(vector_b, dtype=np.float64)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def rank_corpus(query_vector: Sequence[float], doc_vectors: Sequence[Sequence[float]]) -> list[int]:
    """Corpus indices ordered by cosine similarity to the query, best first.

    Ties break by corpus index (stable order) so repeated runs rank identically.
    """
    docs = np.asarray(doc_vectors, dtype=np.float64)
    if docs.size == 0:
        return []
    query = np.asarray(query_vector, dtype=np.float64)
    denominator = np.linalg.norm(docs, axis=1) * float(np.linalg.norm(query))
    scores = np.zeros(len(docs), dtype=np.float64)
    comparable = denominator > 0.0
    scores[comparable] = (docs[comparable] @ query) / denominator[comparable]
    return [int(index) for index in np.argsort(-scores, kind="stable")]


def gold_rank(order: Sequence[int], gold_doc_id: int) -> int:
    """1-based rank of `gold_doc_id` in the similarity `order` from rank_corpus."""
    try:
        return list(order).index(gold_doc_id) + 1
    except ValueError as exc:
        raise ValueError(f"gold document {gold_doc_id} is not in the probed corpus") from exc


# ------------------------------------------------------------------
# Probe
# ------------------------------------------------------------------

def _effective_corpus_size(corpus_size: int, max_docs: int | None) -> int:
    """Size of the probed prefix: `--max-docs` mirrors the indexed corpus prefix."""
    return corpus_size if max_docs is None else min(max_docs, corpus_size)


def _resolve_row_id(row: dict, position: int) -> tuple[object, str]:
    """Question identity plus how it was obtained (design AD-3).

    An explicit `row_id` wins; otherwise the row's own dataset identifier is
    used (`feta_id` for FeTaQA, so its numeric values stay recognizable); the
    last resort is the row's 0-based position in the loaded rows list, which is
    stable for a fixed dataset/n/seed.
    """
    for key in _ROW_ID_KEYS:
        value = row.get(key)
        if value is not None:
            return value, "loader"
    return position, "selection_index"


def _duplicate_document_count(corpus: Sequence[str]) -> int:
    """Documents whose content repeats in the FULL loader corpus.

    Identity/dedupe semantics are unchanged — this only reports how much
    duplicated evidence the probed corpus contains, so a reader can weigh it.
    Counted over the whole corpus (not the probed prefix) so the number is a
    stable property of the dataset and comparable across `--max-docs` runs.
    """
    return len(corpus) - len(set(corpus))


def _in_pool_rate(scored: list[dict], depth: int) -> float | None:
    """Share of scorable questions whose gold document is inside `depth`."""
    if not scored:
        return None
    inside = sum(1 for question in scored if question["gold_in_pool"][depth])
    return inside / len(scored)


def _question_entry(
    row: dict, gold: dict, order: Sequence[int], depths: Sequence[int], row_id,
) -> dict:
    """One per-question report entry: gold ranks plus in-pool flags at every depth.

    `gold_rank` is the best (minimum) rank across the question's in-prefix gold
    documents, and the depth flags follow it. Unscorable questions (out of
    corpus, or no gold labels at all) carry `None` flags — they are unrankable,
    not misses.
    """
    if gold["gold_doc_ids"]:
        ranks = [gold_rank(order, doc_id) for doc_id in gold["gold_doc_ids"]]
        best_rank = min(ranks)
        flags: dict[int, bool | None] = {depth: best_rank <= depth for depth in depths}
    else:
        ranks = []
        best_rank = None
        flags = {depth: None for depth in depths}
    return {
        "row_id": row_id,
        "question": row["question"],
        "gold_doc_ids": gold["gold_doc_ids"],
        "gold_rank": best_rank,
        "gold_ranks": ranks,
        "n_gold_docs": len(gold["gold_doc_ids"]),
        "gold_in_pool": flags,
        "gold_out_of_corpus": gold["gold_out_of_corpus"],
        "no_gold_labels": gold["no_gold_labels"],
    }


def probe_pool(
    rows: list[dict],
    corpus: list[str],
    embed_fn: Embedder,
    *,
    model: str = DEFAULT_EMBEDDING_MODEL,
    depths: Sequence[int] = DEFAULT_DEPTHS,
    max_docs: int | None = None,
    cache_dir: str | Path | None = None,
) -> dict:
    """Probe gold-document reachability at the requested pool depths.

    `rows` and `corpus` come from the dataset loader (`rag_datasets.load_one`);
    only the effective corpus prefix is embedded, so the probe measures the same
    documents the indexed run saw. `embed_fn` is injected — unit tests pass a
    deterministic fake, the CLI passes `build_openai_embed_fn(model)`.
    """
    depth_list = [int(depth) for depth in depths]
    effective_len = _effective_corpus_size(len(corpus), max_docs)
    probed_corpus = list(corpus[:effective_len])

    cache = EmbeddingCache(model=model, cache_dir=cache_dir)
    doc_vectors = cache.embed(probed_corpus, embed_fn)
    question_vectors = cache.embed([row["question"] for row in rows], embed_fn)

    per_question: list[dict] = []
    row_id_sources: set[str] = set()
    for position, (row, query_vector) in enumerate(zip(rows, question_vectors)):
        row_id, row_id_source = _resolve_row_id(row, position)
        row_id_sources.add(row_id_source)
        gold = resolve_gold_targets(row, effective_len) or dict(_UNLINKED_GOLD_TARGET)
        per_question.append(
            _question_entry(row, gold, rank_corpus(query_vector, doc_vectors), depth_list, row_id)
        )

    # Unscorable questions are excluded from both rates: they are not misses.
    scored = [
        question for question in per_question
        if not question["gold_out_of_corpus"] and not question["no_gold_labels"]
    ]
    in_rates = {depth: _in_pool_rate(scored, depth) for depth in depth_list}
    rates = {
        "in": in_rates,
        "out": {depth: None if rate is None else 1.0 - rate
                for depth, rate in in_rates.items()},
    }

    return {
        "model": model,
        "depths": depth_list,
        "corpus_size": len(corpus),
        "effective_corpus_size": effective_len,
        "n_questions": len(rows),
        "n_scored": len(scored),
        "n_out_of_corpus": sum(1 for q in per_question if q["gold_out_of_corpus"]),
        "n_no_gold_labels": sum(1 for q in per_question if q["no_gold_labels"]),
        # Any positional fallback weakens the whole report's identity claim.
        "row_id_source": "loader" if row_id_sources <= {"loader"} else "selection_index",
        "n_duplicate_documents": _duplicate_document_count(corpus),
        "rates": rates,
        "per_question": per_question,
        "caveat": CAVEAT,
    }


# ------------------------------------------------------------------
# Real embedder — the only network path
# ------------------------------------------------------------------

def build_openai_embed_fn(
    model: str = DEFAULT_EMBEDDING_MODEL,
    api_key: str | None = None,
    base_url: str | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Embedder:
    """Build the real embedder: OpenAI embeddings API, batched, order-preserving.

    The same `model` identifier embeds both the corpus and the questions, so the
    probe replicates the model the backend indexed with. Unit tests never call
    this function's returned embedder; they inject a deterministic fake instead.
    """
    client = OpenAI(api_key=api_key, base_url=base_url)

    def embed(texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start:start + batch_size])
            response = client.embeddings.create(model=model, input=batch)
            vectors.extend([list(entry.embedding) for entry in response.data])
        return vectors

    return embed
