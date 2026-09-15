"""
Offline retrieval-pool probe for the FeTaQA corpus.

The eval side has no PostgreSQL/PGVector driver, so this module replicates
embedding + exact cosine similarity locally to answer one question before the
representation (C) and reranker (D) phases invest effort:

    does the gold document reach the top-100 / top-500 of the exact
    similarity ranking for the probed corpus prefix?

Design pins (design.md D6):
  - Cache keys are content-addressed: ``sha256(model + text)``. Changed content
    (for example the Opción B serialization) misses by construction, and a warm
    cache lets re-runs skip the embedder without changing results.
  - Ranking uses exact cosine over full vectors — never an approximate index.
  - Out-of-corpus questions (loader-skipped table, ``--max-docs`` truncation)
    are reported separately from out-of-pool questions and never counted as
    recall misses.
  - The embed function is injected (test seam). ``build_openai_embed_fn`` is the
    only network path and is never exercised by unit tests.

Report shape (consumed by the `runner.py pool-probe` subcommand):

    {
      "model": str,                       # embedding model identifier
      "depths": [int, ...],               # probed pool depths
      "corpus_size": int,                 # full loader corpus size
      "effective_corpus_size": int,       # indexed prefix actually probed
      "n_questions": int,
      "n_scored": int,                    # questions with gold inside the prefix
      "n_out_of_corpus": int,
      "rates": {"in": {depth: float | None}, "out": {depth: float | None}},
      "per_question": [
        {
          "feta_id": int | None,
          "question": str,
          "gold_doc_id": int | None,
          "gold_rank": int | None,          # 1-based; None when out of corpus
          "gold_in_pool": {depth: bool | None},
          "gold_out_of_corpus": bool,
        },
        ...
      ],
      "caveat": str,                        # mandatory replication caveat
    }

Rates are ``None`` when no question is scorable (empty denominator = N/A, not 0).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from openai import OpenAI


DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_DEPTHS: tuple[int, ...] = (100, 500)
DEFAULT_BATCH_SIZE = 64

# File inside `--cache-dir` holding one JSON object per line: {"key": ..., "vector": ...}.
CACHE_FILENAME = "embeddings.jsonl"

# Mandatory in every report (spec: "Replication caveat").
CAVEAT = (
    "Replicates embedding + exact cosine similarity only; it does NOT reproduce "
    "PGVector ANN/index internals, so these results are necessary but not "
    "sufficient evidence for the live retriever."
)

# Batch embedder contract: one vector per input text, same input order.
Embedder = Callable[[Sequence[str]], list[list[float]]]


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


def _prepare_questions(rows: list[dict], effective_len: int) -> list[dict]:
    """Resolve each row's gold linkage against the probed corpus prefix.

    Mirrors `runner._annotate_gold`: the loader flag, a missing `gold_doc_id`,
    or a `gold_doc_id` at/after the effective corpus length all mark the
    question out of corpus. Input rows are not mutated.
    """
    prepared: list[dict] = []
    for row in rows:
        gold_doc_id = row.get("gold_doc_id")
        out_of_corpus = (
            bool(row.get("gold_out_of_corpus", False))
            or gold_doc_id is None
            or gold_doc_id >= effective_len
        )
        prepared.append({
            "gold_doc_id": None if out_of_corpus else gold_doc_id,
            "gold_out_of_corpus": out_of_corpus,
        })
    return prepared


def _in_pool_rate(scored: list[dict], depth: int) -> float | None:
    """Share of scorable questions whose gold document is inside `depth`."""
    if not scored:
        return None
    inside = sum(1 for question in scored if question["gold_in_pool"][depth])
    return inside / len(scored)


def _question_entry(row: dict, gold: dict, order: Sequence[int], depths: Sequence[int]) -> dict:
    """One per-question report entry: gold rank plus in-pool flags at every depth.

    Out-of-corpus questions carry `None` flags — they are unrankable, not misses.
    """
    if gold["gold_out_of_corpus"]:
        position = None
        flags: dict[int, bool | None] = {depth: None for depth in depths}
    else:
        position = gold_rank(order, gold["gold_doc_id"])
        flags = {depth: position <= depth for depth in depths}
    return {
        "feta_id": row.get("feta_id"),
        "question": row["question"],
        "gold_doc_id": gold["gold_doc_id"],
        "gold_rank": position,
        "gold_in_pool": flags,
        "gold_out_of_corpus": gold["gold_out_of_corpus"],
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

    `rows` and `corpus` come from the dataset loader (`rag_datasets.load_fetaqa`);
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

    prepared = _prepare_questions(rows, effective_len)
    per_question = [
        _question_entry(row, gold, rank_corpus(query_vector, doc_vectors), depth_list)
        for row, gold, query_vector in zip(rows, prepared, question_vectors)
    ]

    # Out-of-corpus questions are excluded from both rates: they are not misses.
    scored = [question for question in per_question if not question["gold_out_of_corpus"]]
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
        "n_out_of_corpus": len(per_question) - len(scored),
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
