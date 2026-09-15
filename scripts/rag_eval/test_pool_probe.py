"""
Tests for pool_probe.py — the offline retrieval-pool probe.

Spec: rag-eval-pool-probe (R1-R5). Every test injects a deterministic fake
embedder, so no test touches the OpenAI API, PostgreSQL, or any backend.
"""
import math
import os
import sys

import pytest
from unittest.mock import MagicMock, patch

# Make pool_probe importable from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---------------------------------------------------------------------------
# Helpers — synthetic corpora + injected embedders (offline)
# ---------------------------------------------------------------------------

def _make_embed_fn(mapping: dict):
    """Injected deterministic embedder: one known vector per text, records calls."""
    calls: list[list[str]] = []

    def embed(texts):
        calls.append(list(texts))
        return [mapping[text] for text in texts]

    embed.calls = calls
    return embed


def _corpus_vectors(n_corpus: int) -> list[list[float]]:
    """Corpus vectors whose cosine to ``[1.0, 0.0]`` decreases strictly with index.

    With the probe query vector ``[1.0, 0.0]``, document ``i`` ranks exactly
    ``i + 1`` — the synthetic ranking used by the flag/rate tests.
    """
    return [[float(n_corpus - i), 1.0] for i in range(n_corpus)]


def _make_probe_inputs(n_corpus: int, gold_doc_ids: list):
    """Build (rows, corpus, mapping) mirroring the loader's gold linkage fields.

    A ``None`` gold id mirrors a loader-skipped table: the loader emits
    ``gold_doc_id=None`` plus ``gold_out_of_corpus=True``.
    """
    corpus = [f"doc{i}" for i in range(n_corpus)]
    mapping = dict(zip(corpus, _corpus_vectors(n_corpus)))
    rows = []
    for index, gold_doc_id in enumerate(gold_doc_ids):
        question = f"question {index}"
        mapping[question] = [1.0, 0.0]
        rows.append({
            "question": question,
            "feta_id": 2000 + index,
            "gold_doc_id": gold_doc_id,
            "gold_out_of_corpus": gold_doc_id is None,
        })
    return rows, corpus, mapping


# ---------------------------------------------------------------------------
# Cache keys — content addressed (sha256 over model + text)
# ---------------------------------------------------------------------------

class TestEmbeddingCacheKey:
    """Cache keys bind both the model and the exact text (design D6)."""

    def test_key_is_a_stable_hex_digest(self):
        from pool_probe import embedding_cache_key

        key = embedding_cache_key("text-embedding-3-small", "Year | Award\n2013 | Won")
        assert key == embedding_cache_key("text-embedding-3-small", "Year | Award\n2013 | Won")
        assert len(key) == 64
        assert set(key) <= set("0123456789abcdef")

    def test_key_changes_when_the_model_changes(self):
        from pool_probe import embedding_cache_key

        assert embedding_cache_key("text-embedding-3-small", "same text") != \
            embedding_cache_key("text-embedding-3-large", "same text")

    def test_key_changes_when_the_text_changes(self):
        from pool_probe import embedding_cache_key

        assert embedding_cache_key("text-embedding-3-small", "header | old") != \
            embedding_cache_key("text-embedding-3-small", "header | new")


# ---------------------------------------------------------------------------
# Cache reuse — warm caches skip the embedder, changed content misses
# ---------------------------------------------------------------------------

class TestEmbeddingCacheReuse:
    """Spec: deterministic re-runs; cached embeddings MAY be reused unchanged."""

    MODEL = "text-embedding-3-small"

    def test_first_run_embeds_unique_texts_once_in_input_order(self, tmp_path):
        from pool_probe import EmbeddingCache

        embed_fn = _make_embed_fn({"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [1.0, 1.0]})
        cache = EmbeddingCache(model=self.MODEL, cache_dir=tmp_path)

        vectors = cache.embed(["a", "b", "c"], embed_fn)

        assert embed_fn.calls == [["a", "b", "c"]]
        assert vectors == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]

    def test_duplicate_texts_share_one_embedding(self, tmp_path):
        from pool_probe import EmbeddingCache

        embed_fn = _make_embed_fn({"dup": [0.5, 0.5]})
        cache = EmbeddingCache(model=self.MODEL, cache_dir=tmp_path)

        vectors = cache.embed(["dup", "dup", "dup"], embed_fn)

        assert embed_fn.calls == [["dup"]]
        assert vectors == [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]

    def test_warm_cache_skips_the_embedder_entirely(self, tmp_path):
        from pool_probe import EmbeddingCache

        EmbeddingCache(model=self.MODEL, cache_dir=tmp_path).embed(
            ["a", "b"], _make_embed_fn({"a": [1.0, 0.0], "b": [0.0, 1.0]}))

        def exploding_embed_fn(texts):
            raise AssertionError(f"warm cache must not embed, got {texts}")

        warm = EmbeddingCache(model=self.MODEL, cache_dir=tmp_path)
        assert warm.embed(["a", "b"], exploding_embed_fn) == [[1.0, 0.0], [0.0, 1.0]]

    def test_changed_text_invalidates_its_entry(self, tmp_path):
        """Post-Opción B serialization change → new content → new cache key."""
        from pool_probe import EmbeddingCache

        EmbeddingCache(model=self.MODEL, cache_dir=tmp_path).embed(
            ["header | value"], _make_embed_fn({"header | value": [1.0, 0.0]}))

        embed_fn = _make_embed_fn({"Title: t\nheader | value": [0.0, 1.0]})
        cache = EmbeddingCache(model=self.MODEL, cache_dir=tmp_path)

        assert cache.embed(["Title: t\nheader | value"], embed_fn) == [[0.0, 1.0]]
        assert embed_fn.calls == [["Title: t\nheader | value"]]

    def test_changed_model_invalidates_its_entry(self, tmp_path):
        from pool_probe import EmbeddingCache

        EmbeddingCache(model="text-embedding-3-small", cache_dir=tmp_path).embed(
            ["a"], _make_embed_fn({"a": [1.0, 0.0]}))

        embed_fn = _make_embed_fn({"a": [0.0, 1.0]})
        cache = EmbeddingCache(model="text-embedding-3-large", cache_dir=tmp_path)

        assert cache.embed(["a"], embed_fn) == [[0.0, 1.0]]
        assert embed_fn.calls == [["a"]]

    def test_partial_cache_reuse_embeds_only_new_documents(self, tmp_path):
        """A deeper corpus re-run reuses the cached prefix and embeds the rest."""
        from pool_probe import EmbeddingCache

        mapping = {"doc0": [1.0, 0.0], "doc1": [0.0, 1.0], "doc2": [1.0, 1.0], "doc3": [2.0, 0.0]}
        EmbeddingCache(model=self.MODEL, cache_dir=tmp_path).embed(
            ["doc0", "doc1"], _make_embed_fn(mapping))

        embed_fn = _make_embed_fn(mapping)
        cache = EmbeddingCache(model=self.MODEL, cache_dir=tmp_path)
        vectors = cache.embed(["doc0", "doc1", "doc2", "doc3"], embed_fn)

        assert embed_fn.calls == [["doc2", "doc3"]]
        assert vectors == [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]]

    def test_cache_round_trips_vectors_exactly(self, tmp_path):
        from pool_probe import EmbeddingCache

        vector = [0.1, -1.0 / 3.0, math.sqrt(2.0)]
        EmbeddingCache(model=self.MODEL, cache_dir=tmp_path).embed(
            ["x"], _make_embed_fn({"x": vector}))

        reloaded = EmbeddingCache(model=self.MODEL, cache_dir=tmp_path)
        assert reloaded.embed(["x"], _make_embed_fn({})) == [vector]

    def test_in_memory_cache_does_not_persist_without_a_cache_dir(self):
        from pool_probe import EmbeddingCache

        EmbeddingCache(model=self.MODEL).embed(["a"], _make_embed_fn({"a": [1.0, 0.0]}))

        embed_fn = _make_embed_fn({"a": [1.0, 0.0]})
        EmbeddingCache(model=self.MODEL).embed(["a"], embed_fn)
        assert embed_fn.calls == [["a"]]

    def test_corrupt_cache_line_is_a_miss_not_a_crash(self, tmp_path):
        from pool_probe import CACHE_FILENAME, EmbeddingCache

        (tmp_path / CACHE_FILENAME).write_text(
            'not json\n{"key": "missing-vector"}\n', encoding="utf-8")

        embed_fn = _make_embed_fn({"a": [1.0, 0.0]})
        cache = EmbeddingCache(model=self.MODEL, cache_dir=tmp_path)

        assert cache.embed(["a"], embed_fn) == [[1.0, 0.0]]
        assert embed_fn.calls == [["a"]]

    def test_mismatched_embed_fn_result_raises(self, tmp_path):
        from pool_probe import EmbeddingCache

        cache = EmbeddingCache(model=self.MODEL, cache_dir=tmp_path)
        with pytest.raises(ValueError, match="embed_fn returned 1 vectors for 2 texts"):
            cache.embed(["a", "b"], lambda texts: [[1.0, 0.0]])

    def test_repeated_calls_in_one_session_reuse_in_memory_entries(self, tmp_path):
        from pool_probe import EmbeddingCache

        embed_fn = _make_embed_fn({"a": [1.0, 0.0]})
        cache = EmbeddingCache(model=self.MODEL, cache_dir=tmp_path)

        cache.embed(["a"], embed_fn)
        assert cache.embed(["a"], embed_fn) == [[1.0, 0.0]]
        assert embed_fn.calls == [["a"]]

    def test_no_texts_means_no_embedder_call(self, tmp_path):
        from pool_probe import EmbeddingCache

        embed_fn = _make_embed_fn({})
        cache = EmbeddingCache(model=self.MODEL, cache_dir=tmp_path)

        assert cache.embed([], embed_fn) == []
        assert embed_fn.calls == []
        assert not (tmp_path / "embeddings.jsonl").exists()


# ---------------------------------------------------------------------------
# Exact cosine (no ANN)
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    """Spec R2/R3: ranking uses exact cosine over full vectors."""

    @pytest.mark.parametrize("a, b, expected", [
        ([1.0, 0.0], [1.0, 0.0], 1.0),
        ([1.0, 0.0], [0.0, 1.0], 0.0),
        ([1.0, 0.0], [-1.0, 0.0], -1.0),
        ([3.0, 4.0], [6.0, 8.0], 1.0),
        ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], 1.0),
    ])
    def test_known_angles(self, a, b, expected):
        from pool_probe import cosine_similarity

        assert cosine_similarity(a, b) == pytest.approx(expected)

    def test_forty_five_degrees(self):
        from pool_probe import cosine_similarity

        assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(1.0 / math.sqrt(2.0))

    def test_zero_vector_scores_zero(self):
        from pool_probe import cosine_similarity

        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


class TestRankCorpus:
    """Full similarity ordering, best first, deterministic ties."""

    def test_best_first_order_matches_exact_cosine(self):
        from pool_probe import rank_corpus

        vectors = [[0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
        assert rank_corpus([1.0, 0.0], vectors) == [1, 2, 0]

    def test_ties_break_by_corpus_index(self):
        from pool_probe import rank_corpus

        vectors = [[1.0, 1.0], [2.0, 2.0], [1.0, 0.0]]
        assert rank_corpus([1.0, 0.0], vectors) == [2, 0, 1]

    def test_scale_invariance_keeps_the_ranking(self):
        from pool_probe import rank_corpus

        assert rank_corpus([1.0, 0.0], [[100.0, 1.0], [1.0, 1.0]]) == [0, 1]

    def test_empty_corpus_returns_an_empty_order(self):
        from pool_probe import rank_corpus

        assert rank_corpus([1.0, 0.0], []) == []

    def test_zero_vector_document_scores_zero_and_keeps_index_order(self):
        from pool_probe import rank_corpus

        assert rank_corpus([1.0, 0.0], [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]) == [1, 0, 2]


class TestGoldRank:
    """1-based rank of the gold document in the similarity order."""

    def test_rank_is_the_one_based_position(self):
        from pool_probe import gold_rank

        assert gold_rank([5, 2, 9], 5) == 1
        assert gold_rank([5, 2, 9], 9) == 3

    def test_missing_gold_document_raises(self):
        from pool_probe import gold_rank

        with pytest.raises(ValueError):
            gold_rank([0, 1, 2], 7)


# ---------------------------------------------------------------------------
# Gold-in-pool flags (spec scenarios @100/@500)
# ---------------------------------------------------------------------------

class TestProbeFlags:
    """Spec: per-question flags at 100 and 500 plus the reported rank."""

    def test_gold_inside_top_100_flags_both_depths(self):
        """Spec scenario: gold ranks 37th → gold-in-pool@100 and @500 are true."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [36])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        question = report["per_question"][0]
        assert question["gold_rank"] == 37
        assert question["gold_in_pool"] == {100: True, 500: True}
        assert question["gold_out_of_corpus"] is False

    def test_gold_outside_top_500_flags_both_false_and_reports_the_rank(self):
        """Spec scenario: gold ranks 612th → both flags false, rank reported."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(650, [611])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        question = report["per_question"][0]
        assert question["gold_rank"] == 612
        assert question["gold_in_pool"] == {100: False, 500: False}

    @pytest.mark.parametrize("gold_index, expected", [
        (99, {100: True, 500: True}),     # rank 100 — last document inside @100
        (100, {100: False, 500: True}),   # rank 101 — first document outside @100
        (499, {100: False, 500: True}),   # rank 500 — last document inside @500
        (500, {100: False, 500: False}),  # rank 501 — first document outside @500
    ])
    def test_rank_boundaries_at_both_depths(self, gold_index, expected):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(520, [gold_index])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["per_question"][0]["gold_rank"] == gold_index + 1
        assert report["per_question"][0]["gold_in_pool"] == expected

    def test_custom_depths_are_respected(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(20, [4])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), depths=(3, 5))

        assert report["depths"] == [3, 5]
        assert report["per_question"][0]["gold_in_pool"] == {3: False, 5: True}
        assert report["rates"]["in"] == {3: 0.0, 5: 1.0}

    def test_flags_follow_the_injected_embedder(self):
        """Nothing is hardcoded: the vectors decide gold reachability."""
        from pool_probe import probe_pool

        rows, corpus, _ = _make_probe_inputs(10, [7])
        query = {row["question"]: [1.0, 0.0] for row in rows}
        gold_least_similar = dict(query, **{text: [0.0, 1.0] for text in corpus})
        gold_most_similar = dict(gold_least_similar, **{corpus[7]: [1.0, 0.0]})

        worst = probe_pool(rows, corpus, _make_embed_fn(gold_least_similar), depths=(1,))
        best = probe_pool(rows, corpus, _make_embed_fn(gold_most_similar), depths=(1,))

        assert worst["per_question"][0]["gold_rank"] == 8
        assert worst["per_question"][0]["gold_in_pool"] == {1: False}
        assert best["per_question"][0]["gold_rank"] == 1
        assert best["per_question"][0]["gold_in_pool"] == {1: True}


# ---------------------------------------------------------------------------
# Dataset-level rates (the gate for C/D)
# ---------------------------------------------------------------------------

class TestProbeRates:
    """Spec: dataset-level gold-in-pool rates at both depths."""

    def test_rates_at_both_depths_over_scored_questions(self):
        from pool_probe import probe_pool

        # ranks 37, 612, 99 → in@100 and in@500 are 2/3
        rows, corpus, mapping = _make_probe_inputs(650, [36, 611, 98])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["n_questions"] == 3
        assert report["n_scored"] == 3
        assert report["rates"]["in"][100] == pytest.approx(2 / 3)
        assert report["rates"]["in"][500] == pytest.approx(2 / 3)
        assert report["rates"]["out"][100] == pytest.approx(1 / 3)
        assert report["rates"]["out"][500] == pytest.approx(1 / 3)

    def test_all_reachable_reports_full_in_rates(self):
        from pool_probe import probe_pool

        # ranks 1 and 300 → in@100 = 1/2, in@500 = 2/2
        rows, corpus, mapping = _make_probe_inputs(300, [0, 299])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["rates"]["in"][100] == pytest.approx(0.5)
        assert report["rates"]["in"][500] == pytest.approx(1.0)
        assert report["rates"]["out"][100] == pytest.approx(0.5)
        assert report["rates"]["out"][500] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Out-of-corpus ≠ out-of-pool
# ---------------------------------------------------------------------------

class TestProbeOutOfCorpusSeparation:
    """Spec: out-of-corpus questions are reported separately, never as misses."""

    def test_max_docs_truncation_marks_the_question_out_of_corpus(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [550])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=500)

        question = report["per_question"][0]
        assert question["gold_out_of_corpus"] is True
        assert question["gold_rank"] is None
        assert question["gold_in_pool"] == {100: None, 500: None}
        assert report["n_out_of_corpus"] == 1
        assert report["n_scored"] == 0

    def test_out_of_corpus_is_excluded_from_the_pool_rates(self):
        from pool_probe import probe_pool

        # q0: gold at rank 501 → genuinely out of pool; q1: gold truncated → out of corpus
        rows, corpus, mapping = _make_probe_inputs(600, [500, 550])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=520)

        assert report["n_scored"] == 1
        assert report["n_out_of_corpus"] == 1
        assert report["rates"]["in"][500] == 0.0
        assert report["rates"]["out"][500] == 1.0
        assert report["per_question"][0]["gold_out_of_corpus"] is False
        assert report["per_question"][1]["gold_out_of_corpus"] is True

    def test_loader_skipped_table_is_out_of_corpus(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(10, [None])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        question = report["per_question"][0]
        assert question["gold_out_of_corpus"] is True
        assert question["gold_doc_id"] is None
        assert question["gold_rank"] is None
        assert question["gold_in_pool"] == {100: None, 500: None}
        assert report["rates"]["in"][100] is None
        assert report["rates"]["out"][500] is None

    def test_gold_exactly_at_the_max_docs_boundary_is_out_of_corpus(self):
        """--max-docs keeps the first N documents, so index N is already outside."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [500])
        truncated = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=500)
        included = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=501)

        assert truncated["per_question"][0]["gold_out_of_corpus"] is True
        assert truncated["n_scored"] == 0
        assert included["per_question"][0]["gold_out_of_corpus"] is False
        assert included["per_question"][0]["gold_rank"] == 501
        assert included["rates"]["in"][500] == 0.0

    def test_max_docs_beyond_the_corpus_keeps_the_whole_corpus(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(5, [4])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=99999)

        assert report["effective_corpus_size"] == 5
        assert report["per_question"][0]["gold_rank"] == 5
        assert report["rates"]["in"][100] == 1.0

    def test_empty_corpus_reports_none_rates_without_crashing(self):
        from pool_probe import probe_pool

        rows = [{"question": "who?", "feta_id": 1, "gold_doc_id": None, "gold_out_of_corpus": True}]
        report = probe_pool(rows, [], _make_embed_fn({"who?": [1.0, 0.0]}))

        assert report["corpus_size"] == 0
        assert report["n_out_of_corpus"] == 1
        assert report["rates"]["in"][100] is None
        assert report["per_question"][0]["gold_in_pool"] == {100: None, 500: None}

    def test_embeds_only_the_probed_prefix(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [0])
        embed_fn = _make_embed_fn(mapping)
        probe_pool(rows, corpus, embed_fn, max_docs=3)

        assert embed_fn.calls[0] == ["doc0", "doc1", "doc2"]
        assert embed_fn.calls[1] == ["question 0"]


# ---------------------------------------------------------------------------
# Faithful embedding replication (spec R2)
# ---------------------------------------------------------------------------

class TestProbeEmbeddingFidelity:
    """The probe embeds exactly the serialized corpus and the question text."""

    def test_corpus_documents_are_embedded_verbatim(self):
        from pool_probe import probe_pool

        corpus = [
            "Year | Award | Result\n2013 | Drama Desk | Nominated",
            "Name | Age\nAlice | 30\nBob | 40",
        ]
        question = "Who won in 2013?"
        rows = [{"question": question, "feta_id": 7, "gold_doc_id": 0}]
        mapping = {text: [1.0, 0.0] for text in corpus}
        mapping[question] = [0.0, 1.0]
        embed_fn = _make_embed_fn(mapping)

        probe_pool(rows, corpus, embed_fn)

        # Multiline serialized content, untruncated and unmodified.
        assert embed_fn.calls[0] == corpus
        assert embed_fn.calls[1] == [question]

    def test_report_records_the_model_and_the_corpus_sizes(self):
        from pool_probe import DEFAULT_EMBEDDING_MODEL, probe_pool

        rows, corpus, mapping = _make_probe_inputs(5, [0])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=3)

        assert report["model"] == DEFAULT_EMBEDDING_MODEL == "text-embedding-3-small"
        assert report["corpus_size"] == 5
        assert report["effective_corpus_size"] == 3

    def test_per_question_entries_carry_the_row_identity(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(5, [0])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["per_question"][0]["feta_id"] == 2000
        assert report["per_question"][0]["question"] == "question 0"
        assert report["per_question"][0]["gold_doc_id"] == 0


# ---------------------------------------------------------------------------
# Determinism, cache reuse, and the replication caveat
# ---------------------------------------------------------------------------

class TestProbeDeterminismAndCaveat:
    """Spec: identical re-runs, cache-neutral results, mandatory caveat."""

    def test_two_runs_are_identical(self, tmp_path):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(700, [3, 611, None])
        first = probe_pool(rows, corpus, _make_embed_fn(mapping), cache_dir=tmp_path / "a")
        second = probe_pool(rows, corpus, _make_embed_fn(mapping), cache_dir=tmp_path / "b")

        assert first == second

    def test_warm_cache_does_not_change_the_report(self, tmp_path):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(700, [3, 611, None])
        cold = probe_pool(rows, corpus, _make_embed_fn(mapping), cache_dir=tmp_path)

        def exploding_embed_fn(texts):
            raise AssertionError(f"warm cache must not embed, got {texts}")

        warm = probe_pool(rows, corpus, exploding_embed_fn, cache_dir=tmp_path)
        assert warm == cold

    def test_report_carries_the_replication_caveat(self):
        from pool_probe import CAVEAT, probe_pool

        rows, corpus, mapping = _make_probe_inputs(5, [0])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["caveat"] == CAVEAT
        assert "PGVector" in report["caveat"]
        assert "ANN" in report["caveat"]
        assert "not sufficient" in report["caveat"]


# ---------------------------------------------------------------------------
# The real (network) embedder — SDK boundary mocked, no API call
# ---------------------------------------------------------------------------

class TestOpenAIEmbedFn:
    """`build_openai_embed_fn` is the only network path; the SDK is mocked."""

    def _fake_client(self):
        embeddings_api = MagicMock()
        embeddings_api.create.side_effect = lambda *, model, input: MagicMock(
            data=[MagicMock(embedding=[float(len(text)), 1.0]) for text in input])
        return MagicMock(embeddings=embeddings_api), embeddings_api

    def test_batches_requests_and_preserves_input_order(self):
        from pool_probe import build_openai_embed_fn

        client, embeddings_api = self._fake_client()
        with patch("pool_probe.OpenAI", return_value=client) as fake_openai:
            embed_fn = build_openai_embed_fn(
                model="text-embedding-3-small", api_key="test-key", batch_size=2)
            vectors = embed_fn(["a", "bb", "ccc"])

        assert fake_openai.call_args.kwargs["api_key"] == "test-key"
        requests = embeddings_api.create.call_args_list
        assert [call.kwargs["input"] for call in requests] == [["a", "bb"], ["ccc"]]
        assert all(call.kwargs["model"] == "text-embedding-3-small" for call in requests)
        assert vectors == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]]

    def test_empty_input_makes_no_request(self):
        from pool_probe import build_openai_embed_fn

        client, embeddings_api = self._fake_client()
        with patch("pool_probe.OpenAI", return_value=client):
            embed_fn = build_openai_embed_fn(model="text-embedding-3-small")
            assert embed_fn([]) == []

        assert embeddings_api.create.call_count == 0
