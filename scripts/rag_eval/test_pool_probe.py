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


def _make_probe_inputs(n_corpus: int, gold_doc_ids: list, *, row_id=None):
    """Build (rows, corpus, mapping) mirroring the loader's multi-gold linkage.

    ``gold_doc_ids`` carries one entry per question: the list of in-corpus gold
    document indices for that row (``[]`` mirrors a no-label row, a list whose
    ids all sit outside ``--max-docs`` mirrors an out-of-corpus row).
    ``row_id`` optionally pins explicit row identities; without it the rows carry
    a FeTaQA-style dataset identifier (``feta_id``), the loader fallback.
    """
    corpus = [f"doc{i}" for i in range(n_corpus)]
    mapping = dict(zip(corpus, _corpus_vectors(n_corpus)))
    rows = []
    for index, gold_ids in enumerate(gold_doc_ids):
        question = f"question {index}"
        mapping[question] = [1.0, 0.0]
        row = {
            "question": question,
            "gold_doc_ids": list(gold_ids),
            "gold_sentences": [],
            "gold_sentence_doc_ids": [],
            "no_gold_labels": not gold_ids,
        }
        if row_id is not None:
            row["row_id"] = row_id[index]
        else:
            row["feta_id"] = 2000 + index
        rows.append(row)
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
    """Spec: per-question flags at the configured depths plus the reported rank."""

    def test_gold_inside_top_50_flags_every_depth_from_50_up(self):
        """Spec scenario: gold ranks 37th → in-pool at 50/100/500, out-of-pool at 20."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [[36]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        question = report["per_question"][0]
        assert question["gold_rank"] == 37
        assert question["gold_in_pool"] == {20: False, 50: True, 100: True, 500: True}
        assert question["gold_out_of_corpus"] is False

    def test_gold_outside_every_depth_flags_all_false_and_reports_the_rank(self):
        """Spec scenario: gold ranks 612th → every depth flag false, rank reported."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(650, [[611]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        question = report["per_question"][0]
        assert question["gold_rank"] == 612
        assert question["gold_in_pool"] == {20: False, 50: False, 100: False, 500: False}

    @pytest.mark.parametrize("gold_index, expected", [
        (19, {20: True, 50: True, 100: True, 500: True}),      # rank 20 — last inside @20
        (20, {20: False, 50: True, 100: True, 500: True}),     # rank 21 — first outside @20
        (49, {20: False, 50: True, 100: True, 500: True}),     # rank 50 — last inside @50
        (50, {20: False, 50: False, 100: True, 500: True}),    # rank 51 — first outside @50
        (499, {20: False, 50: False, 100: False, 500: True}),  # rank 500 — last inside @500
        (500, {20: False, 50: False, 100: False, 500: False}),  # rank 501 — first outside @500
    ])
    def test_rank_boundaries_at_every_default_depth(self, gold_index, expected):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(520, [[gold_index]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["per_question"][0]["gold_rank"] == gold_index + 1
        assert report["per_question"][0]["gold_in_pool"] == expected

    def test_custom_depths_are_respected(self):
        """Spec scenario: an explicit depth list yields exactly those depths."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(20, [[4]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), depths=(3, 5))

        assert report["depths"] == [3, 5]
        assert report["per_question"][0]["gold_in_pool"] == {3: False, 5: True}
        assert report["rates"]["in"] == {3: 0.0, 5: 1.0}

    def test_default_depths_are_the_gate_depths(self):
        """Spec: the gate default depth list is 20, 50, 100 and 500."""
        from pool_probe import DEFAULT_DEPTHS, probe_pool

        rows, corpus, mapping = _make_probe_inputs(20, [[4]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert DEFAULT_DEPTHS == (20, 50, 100, 500)
        assert report["depths"] == [20, 50, 100, 500]
        assert set(report["rates"]["in"]) == {20, 50, 100, 500}
        assert set(report["per_question"][0]["gold_in_pool"]) == {20, 50, 100, 500}

    def test_flags_follow_the_injected_embedder(self):
        """Nothing is hardcoded: the vectors decide gold reachability."""
        from pool_probe import probe_pool

        rows, corpus, _ = _make_probe_inputs(10, [[7]])
        query = {row["question"]: [1.0, 0.0] for row in rows}
        gold_least_similar = dict(query, **{text: [0.0, 1.0] for text in corpus})
        gold_most_similar = dict(gold_least_similar, **{corpus[7]: [1.0, 0.0]})

        worst = probe_pool(rows, corpus, _make_embed_fn(gold_least_similar), depths=(1,))
        best = probe_pool(rows, corpus, _make_embed_fn(gold_most_similar), depths=(1,))

        assert worst["per_question"][0]["gold_rank"] == 8
        assert worst["per_question"][0]["gold_in_pool"] == {1: False}
        assert best["per_question"][0]["gold_rank"] == 1
        assert best["per_question"][0]["gold_in_pool"] == {1: True}


class TestProbeMultiGold:
    """Spec: gold_rank is the best rank across the in-prefix gold documents."""

    def test_best_rank_wins_and_every_rank_is_reported(self):
        """Spec scenario: gold documents ranking 340th and 12th → gold_rank 12."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(400, [[339, 11]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        question = report["per_question"][0]
        assert question["gold_rank"] == 12
        assert sorted(question["gold_ranks"]) == [12, 340]
        assert question["n_gold_docs"] == 2
        assert question["gold_doc_ids"] == [11, 339]

    def test_depth_flags_follow_the_best_rank(self):
        """The best-ranked gold document decides every depth flag."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [[550, 36]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        question = report["per_question"][0]
        assert question["gold_rank"] == 37
        assert question["gold_in_pool"] == {20: False, 50: True, 100: True, 500: True}

    def test_partial_coverage_scores_the_in_prefix_subset_only(self):
        """Spec scenario: one of two gold documents truncated → not out of corpus."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [[10, 550]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=500)

        question = report["per_question"][0]
        assert question["gold_out_of_corpus"] is False
        assert question["gold_doc_ids"] == [10]
        assert question["n_gold_docs"] == 1
        assert question["gold_rank"] == 11
        assert report["n_scored"] == 1
        assert report["n_out_of_corpus"] == 0

    def test_no_gold_inside_the_prefix_is_out_of_corpus_with_none_flags(self):
        """Spec scenario: --max-docs excludes every gold document → out of corpus."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [[550, 560]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=500)

        question = report["per_question"][0]
        assert question["gold_out_of_corpus"] is True
        assert question["gold_doc_ids"] == []
        assert question["gold_rank"] is None
        assert question["gold_ranks"] == []
        assert question["n_gold_docs"] == 0
        assert question["gold_in_pool"] == {20: None, 50: None, 100: None, 500: None}


# ---------------------------------------------------------------------------
# Dataset-level rates (the gate for C/D)
# ---------------------------------------------------------------------------

class TestProbeRates:
    """Spec: dataset-level gold-in-pool rates at every configured depth."""

    def test_rates_at_the_gate_depths_over_scored_questions(self):
        from pool_probe import probe_pool

        # ranks 37, 612, 99 → in@100 and in@500 are 2/3, in@20 is 0/3
        rows, corpus, mapping = _make_probe_inputs(650, [[36], [611], [98]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["n_questions"] == 3
        assert report["n_scored"] == 3
        assert report["rates"]["in"][20] == 0.0
        assert report["rates"]["in"][50] == pytest.approx(1 / 3)
        assert report["rates"]["in"][100] == pytest.approx(2 / 3)
        assert report["rates"]["in"][500] == pytest.approx(2 / 3)
        assert report["rates"]["out"][500] == pytest.approx(1 / 3)

    def test_all_reachable_reports_full_in_rates(self):
        from pool_probe import probe_pool

        # ranks 1 and 300 → in@100 = 1/2, in@500 = 2/2
        rows, corpus, mapping = _make_probe_inputs(300, [[0], [299]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["rates"]["in"][100] == pytest.approx(0.5)
        assert report["rates"]["in"][500] == pytest.approx(1.0)
        assert report["rates"]["out"][100] == pytest.approx(0.5)
        assert report["rates"]["out"][500] == pytest.approx(0.0)

    def test_rates_exist_for_exactly_the_configured_depths(self):
        """Spec scenario: `--depths 20 50` → flags and rates for exactly those."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(100, [[36]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), depths=(20, 50))

        assert report["depths"] == [20, 50]
        assert set(report["rates"]["in"]) == {20, 50}
        assert set(report["rates"]["out"]) == {20, 50}
        assert report["per_question"][0]["gold_in_pool"] == {20: False, 50: True}


# ---------------------------------------------------------------------------
# row_id identity — dataset-generic report vocabulary
# ---------------------------------------------------------------------------

class TestProbeRowId:
    """Spec: per-question entries carry `row_id`; `feta_id` never appears."""

    def test_explicit_row_id_wins(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(10, [[0]], row_id=[4242])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["per_question"][0]["row_id"] == 4242
        assert report["row_id_source"] == "loader"

    def test_dataset_identifier_is_the_loader_fallback(self):
        """FeTaQA compatibility: the numeric feta_id value reappears under row_id."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(10, [[0]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert rows[0]["feta_id"] == 2000
        assert report["per_question"][0]["row_id"] == 2000
        assert report["row_id_source"] == "loader"

    def test_position_is_the_last_resort(self):
        """No loader identity → the 0-based selection position is reported."""
        from pool_probe import probe_pool

        rows = [
            {"question": "q0", "gold_doc_ids": [0], "no_gold_labels": False},
            {"question": "q1", "gold_doc_ids": [1], "no_gold_labels": False},
        ]
        corpus = ["doc0", "doc1"]
        mapping = {"doc0": [1.0, 0.0], "doc1": [0.0, 1.0], "q0": [1.0, 0.0], "q1": [1.0, 0.0]}
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert [entry["row_id"] for entry in report["per_question"]] == [0, 1]
        assert report["row_id_source"] == "selection_index"

    def test_any_positional_fallback_marks_the_report_source(self):
        """Mixed identities report the weaker source, never a false 'loader'."""
        from pool_probe import probe_pool

        rows = [
            {"question": "q0", "row_id": 77, "gold_doc_ids": [0], "no_gold_labels": False},
            {"question": "q1", "gold_doc_ids": [1], "no_gold_labels": False},
        ]
        corpus = ["doc0", "doc1"]
        mapping = {"doc0": [1.0, 0.0], "doc1": [0.0, 1.0], "q0": [1.0, 0.0], "q1": [1.0, 0.0]}
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert [entry["row_id"] for entry in report["per_question"]] == [77, 1]
        assert report["row_id_source"] == "selection_index"

    def test_no_per_question_entry_carries_a_feta_id_key(self):
        """Spec scenario: `row_id` is present and no `feta_id` key exists."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(10, [[0], []])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        for entry in report["per_question"]:
            assert "row_id" in entry
            assert "feta_id" not in entry
            assert "gold_doc_id" not in entry


# ---------------------------------------------------------------------------
# No-label population — counted separately, never a miss
# ---------------------------------------------------------------------------

class TestProbeNoLabelPopulation:
    """Spec: rows without gold labels are their own population, excluded from rates."""

    def test_no_label_rows_are_counted_separately(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(650, [[36], [], [611]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["n_questions"] == 3
        assert report["n_scored"] == 2
        assert report["n_no_gold_labels"] == 1
        assert report["n_out_of_corpus"] == 0

    def test_no_label_rows_leave_the_rates_untouched(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [[36], []])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        # The single scorable question is in-pool at 50; the no-label row must
        # not dilute the rate and must not be counted as out-of-pool.
        assert report["rates"]["in"][50] == 1.0
        assert report["rates"]["out"][50] == 0.0

    def test_no_label_entry_carries_the_flag_and_no_rank(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(10, [[]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        entry = report["per_question"][0]
        assert entry["no_gold_labels"] is True
        assert entry["gold_out_of_corpus"] is False
        assert entry["gold_rank"] is None
        assert entry["gold_in_pool"] == {20: None, 50: None, 100: None, 500: None}

    def test_populations_account_for_every_question(self):
        """Invariant: n_questions == n_scored + n_out_of_corpus + n_no_gold_labels."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [[10], [550], []])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=500)

        assert report["n_questions"] == (
            report["n_scored"] + report["n_out_of_corpus"] + report["n_no_gold_labels"]
        )
        assert report["n_scored"] == 1
        assert report["n_out_of_corpus"] == 1
        assert report["n_no_gold_labels"] == 1

    def test_rows_without_any_gold_linkage_are_unscorable(self):
        """A dataset nobody labeled is counted as no-label, never out-of-corpus."""
        from pool_probe import probe_pool

        rows = [
            {"question": "scored?", "gold_doc_ids": [0], "no_gold_labels": False},
            {"question": "unlinked?"},
        ]
        corpus = ["doc0", "doc1"]
        mapping = {
            "doc0": [1.0, 0.0], "doc1": [0.0, 1.0],
            "scored?": [1.0, 0.0], "unlinked?": [1.0, 0.0],
        }
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["n_scored"] == 1
        assert report["n_no_gold_labels"] == 1
        assert report["n_out_of_corpus"] == 0
        assert report["per_question"][1]["no_gold_labels"] is True
        assert report["per_question"][1]["gold_rank"] is None


# ---------------------------------------------------------------------------
# Duplicate corpus documents — reported, semantics unchanged
# ---------------------------------------------------------------------------

class TestProbeDuplicateDocuments:
    """Spec/design: duplicate corpus documents are reported, not deduplicated."""

    def test_duplicate_content_is_counted(self):
        from pool_probe import probe_pool

        corpus = ["same document", "same document", "unique document"]
        mapping = {"same document": [1.0, 0.0], "unique document": [0.0, 1.0], "q?": [1.0, 0.0]}
        rows = [{"question": "q?", "gold_doc_ids": [0], "no_gold_labels": False}]

        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["n_duplicate_documents"] == 1
        # Identity semantics are unchanged: both copies stay in the corpus.
        assert report["corpus_size"] == 3
        assert report["effective_corpus_size"] == 3

    def test_unique_corpus_reports_zero_duplicates(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(5, [[0]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["n_duplicate_documents"] == 0


# ---------------------------------------------------------------------------
# Out-of-corpus ≠ out-of-pool
# ---------------------------------------------------------------------------

class TestProbeOutOfCorpusSeparation:
    """Spec: out-of-corpus questions are reported separately, never as misses."""

    def test_max_docs_truncation_marks_the_question_out_of_corpus(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [[550]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=500)

        question = report["per_question"][0]
        assert question["gold_out_of_corpus"] is True
        assert question["gold_rank"] is None
        assert question["gold_in_pool"] == {20: None, 50: None, 100: None, 500: None}
        assert report["n_out_of_corpus"] == 1
        assert report["n_scored"] == 0

    def test_out_of_corpus_is_excluded_from_the_pool_rates(self):
        from pool_probe import probe_pool

        # q0: gold at rank 501 → genuinely out of pool; q1: gold truncated → out of corpus
        rows, corpus, mapping = _make_probe_inputs(600, [[500], [550]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=520)

        assert report["n_scored"] == 1
        assert report["n_out_of_corpus"] == 1
        assert report["rates"]["in"][500] == 0.0
        assert report["rates"]["out"][500] == 1.0
        assert report["per_question"][0]["gold_out_of_corpus"] is False
        assert report["per_question"][1]["gold_out_of_corpus"] is True

    def test_gold_exactly_at_the_max_docs_boundary_is_out_of_corpus(self):
        """--max-docs keeps the first N documents, so index N is already outside."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [[500]])
        truncated = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=500)
        included = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=501)

        assert truncated["per_question"][0]["gold_out_of_corpus"] is True
        assert truncated["n_scored"] == 0
        assert included["per_question"][0]["gold_out_of_corpus"] is False
        assert included["per_question"][0]["gold_rank"] == 501
        assert included["rates"]["in"][500] == 0.0

    def test_max_docs_beyond_the_corpus_keeps_the_whole_corpus(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(5, [[4]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=99999)

        assert report["effective_corpus_size"] == 5
        assert report["per_question"][0]["gold_rank"] == 5
        assert report["rates"]["in"][100] == 1.0

    def test_empty_corpus_reports_none_rates_without_crashing(self):
        from pool_probe import probe_pool

        rows = [{"question": "who?", "row_id": 1, "gold_doc_ids": [0], "no_gold_labels": False}]
        report = probe_pool(rows, [], _make_embed_fn({"who?": [1.0, 0.0]}))

        assert report["corpus_size"] == 0
        assert report["n_out_of_corpus"] == 1
        assert report["rates"]["in"][100] is None
        assert report["per_question"][0]["gold_in_pool"] == {20: None, 50: None, 100: None, 500: None}

    def test_embeds_only_the_probed_prefix(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(600, [[0]])
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
        rows = [{"question": question, "row_id": 7, "gold_doc_ids": [0], "no_gold_labels": False}]
        mapping = {text: [1.0, 0.0] for text in corpus}
        mapping[question] = [0.0, 1.0]
        embed_fn = _make_embed_fn(mapping)

        probe_pool(rows, corpus, embed_fn)

        # Multiline serialized content, untruncated and unmodified.
        assert embed_fn.calls[0] == corpus
        assert embed_fn.calls[1] == [question]

    def test_report_records_the_model_and_the_corpus_sizes(self):
        from pool_probe import DEFAULT_EMBEDDING_MODEL, probe_pool

        rows, corpus, mapping = _make_probe_inputs(5, [[0]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping), max_docs=3)

        assert report["model"] == DEFAULT_EMBEDDING_MODEL == "text-embedding-3-small"
        assert report["corpus_size"] == 5
        assert report["effective_corpus_size"] == 3

    def test_per_question_entries_carry_the_row_identity(self):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(5, [[0], [1, 2]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["per_question"][0]["row_id"] == 2000
        assert report["per_question"][0]["question"] == "question 0"
        assert report["per_question"][0]["gold_doc_ids"] == [0]
        assert report["per_question"][1]["row_id"] == 2001
        assert report["per_question"][1]["gold_doc_ids"] == [1, 2]


# ---------------------------------------------------------------------------
# Determinism, cache reuse, and the replication caveat
# ---------------------------------------------------------------------------

class TestProbeDeterminismAndCaveat:
    """Spec: identical re-runs, cache-neutral results, mandatory caveat."""

    def test_two_runs_are_identical(self, tmp_path):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(700, [[3], [611], []])
        first = probe_pool(rows, corpus, _make_embed_fn(mapping), cache_dir=tmp_path / "a")
        second = probe_pool(rows, corpus, _make_embed_fn(mapping), cache_dir=tmp_path / "b")

        assert first == second

    def test_warm_cache_does_not_change_the_report(self, tmp_path):
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(700, [[3], [611], []])
        cold = probe_pool(rows, corpus, _make_embed_fn(mapping), cache_dir=tmp_path)

        def exploding_embed_fn(texts):
            raise AssertionError(f"warm cache must not embed, got {texts}")

        warm = probe_pool(rows, corpus, exploding_embed_fn, cache_dir=tmp_path)
        assert warm == cold

    def test_report_carries_the_replication_caveat(self):
        from pool_probe import CAVEAT, probe_pool

        rows, corpus, mapping = _make_probe_inputs(5, [[0]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        assert report["caveat"] == CAVEAT
        assert "PGVector" in report["caveat"]
        assert "ANN" in report["caveat"]
        assert "not sufficient" in report["caveat"]

    def test_caveat_states_the_full_document_approximation(self):
        """Spec scenario: no exact chunk-level parity is claimed for split documents."""
        from pool_probe import probe_pool

        rows, corpus, mapping = _make_probe_inputs(5, [[0]])
        report = probe_pool(rows, corpus, _make_embed_fn(mapping))

        caveat = report["caveat"]
        assert "whole" in caveat or "full document" in caveat
        assert "chunk" in caveat
        assert "approximation" in caveat


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
