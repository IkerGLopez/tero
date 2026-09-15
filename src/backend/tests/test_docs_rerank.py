"""Docs-tool reranker plumbing tests (fetaqa-retrieval-fix, phase 8, slice D1).

Spec: openspec/changes/fetaqa-retrieval-fix/specs/docs-tool-retrieval/spec.md
(R3 flag-gated reranker + invalid configuration, R5 post-rerank alignment).

The 0-10 LLM scorer is a seam in this slice: every test injects a deterministic
fake. The real batched/accounted scorer lands in slice D2, so no test here calls
an LLM or a database.
"""

from typing import Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore

from tero.core.env import Settings, env
from tero.tools.docs import tool as docs_tool_module
from tero.tools.docs.tool import (
    DOCS_TOOL_ID,
    DocsExecutionStep,
    DocsStatusUpdateCallbackHandler,
    DocsTool,
    DocumentUrlSolvingRetriever,
)


def _documents(count: int) -> list[Document]:
    return [
        Document(page_content=f"chunk {i}", metadata={"id": str(i)})
        for i in range(count)
    ]


def _configured_tool() -> DocsTool:
    tool = DocsTool()
    tool.configure(MagicMock(id=7), user_id=1, config={}, db=MagicMock())
    return tool


def _fake_vectorstore(documents: Sequence[Document]) -> MagicMock:
    vectorstore = MagicMock(spec=VectorStore)
    vectorstore.asimilarity_search = AsyncMock(return_value=list(documents))
    return vectorstore


class _RecordingScorer:
    """Deterministic async stand-in for the D2 LLM scorer (the injected seam)."""

    def __init__(self, scores: dict[str, float]):
        self._scores = scores
        self.calls: list[tuple[str, list[Document]]] = []

    async def __call__(self, query: str, documents: Sequence[Document]) -> list[float]:
        self.calls.append((query, list(documents)))
        return [self._scores[doc.metadata["id"]] for doc in documents]


def _retrieved_events(writer: MagicMock) -> list:
    return [
        call.args[0]
        for call in writer.call_args_list
        if call.args and call.args[0].step == DocsExecutionStep.RETRIEVED
    ]


# --- R3: settings ship disabled with defaults requiring no .env edit ---------

def test_rerank_setting_defaults_require_no_env_edit():
    """R3 defaults: off, fetch_k 20, top_n 5, scoring model unset (falls back)."""
    assert Settings.model_fields["docs_tool_rerank"].default is False
    assert Settings.model_fields["docs_tool_rerank_fetch_k"].default == 20
    assert Settings.model_fields["docs_tool_rerank_top_n"].default == 5
    assert Settings.model_fields["docs_tool_rerank_model"].default is None


def test_real_settings_resolve_rerank_defaults_without_env_edits():
    """R3 defaults: the loaded settings are off and default to the generator model.

    Relies on the change invariant that `.env` is not edited (phase 10.3).
    """
    assert env.docs_tool_rerank is False
    assert env.docs_tool_rerank_fetch_k == 20
    assert env.docs_tool_rerank_top_n == 5
    assert env.docs_tool_rerank_model == env.internal_generator_model


# --- R3/R5: disabled path stays behavior-equivalent to today -----------------

async def test_disabled_default_fetches_k5_and_emits_the_same_set():
    """R3/R5 disabled: plain retriever queries k=5 and the event equals that set."""
    tool = _configured_tool()
    documents = _documents(5)
    vectorstore = _fake_vectorstore(documents)
    handler = DocsStatusUpdateCallbackHandler(DOCS_TOOL_ID, "Docs")
    writer = MagicMock()

    with (
        patch.object(DocsTool, "_build_vectorstore", return_value=vectorstore),
        patch("tero.tools.docs.tool.get_stream_writer", return_value=writer),
    ):
        retriever = tool._build_retriever()
        returned = await retriever.ainvoke("emma routine", config={"callbacks": [handler]})

    assert type(retriever) is DocumentUrlSolvingRetriever
    assert retriever.search_kwargs == {"k": 5}
    assert vectorstore.asimilarity_search.await_args.kwargs == {"k": 5}
    assert [doc.page_content for doc in returned] == [doc.page_content for doc in documents]
    events = _retrieved_events(writer)
    assert len(events) == 1
    assert events[0].result == [doc.page_content for doc in returned]


def test_disabled_rerank_never_builds_a_scorer():
    """R3 disabled: no scoring stage is constructed at all."""
    tool = _configured_tool()

    with (
        patch.object(env, "docs_tool_rerank", False),
        patch.object(DocsTool, "_build_vectorstore", return_value=MagicMock(spec=VectorStore)),
        patch.object(
            DocsTool,
            "_build_rerank_scorer",
            side_effect=AssertionError("scorer must not be built while rerank is disabled"),
        ) as scorer_factory,
    ):
        retriever = tool._build_retriever()

    scorer_factory.assert_not_called()
    assert type(retriever) is DocumentUrlSolvingRetriever


# --- R3: enabled pipeline, fetch_k -> score -> top_n -------------------------

def test_enabled_rerank_builds_rerank_retriever_from_settings():
    """R3 enabled: fetch_k/top_n come from settings and the scorer is injected."""
    tool = _configured_tool()
    scorer = _RecordingScorer({})

    with (
        patch.object(env, "docs_tool_rerank", True),
        patch.object(env, "docs_tool_rerank_fetch_k", 20),
        patch.object(env, "docs_tool_rerank_top_n", 5),
        patch.object(DocsTool, "_build_vectorstore", return_value=MagicMock(spec=VectorStore)),
        patch.object(DocsTool, "_build_rerank_scorer", return_value=scorer) as scorer_factory,
    ):
        retriever = tool._build_retriever()

    assert type(retriever) is docs_tool_module.RerankRetriever
    assert (retriever.fetch_k, retriever.top_n) == (20, 5)
    assert retriever.scorer is scorer
    scorer_factory.assert_called_once_with()


async def test_enabled_pipeline_fetches_twenty_scores_all_and_returns_top_five():
    """R3 enabled: 20 candidates fetched, all scored, top 5 returned in score order."""
    candidates = _documents(20)
    scorer = _RecordingScorer({str(i): float(i) for i in range(20)})
    vectorstore = _fake_vectorstore(candidates)
    retriever = docs_tool_module.RerankRetriever(
        vectorstore=vectorstore,
        agent_id=1,
        tool_id=DOCS_TOOL_ID,
        fetch_k=20,
        top_n=5,
        scorer=scorer,
    )

    returned = await retriever.ainvoke("emma routine")

    assert vectorstore.asimilarity_search.await_args.kwargs == {"k": 20}
    assert len(scorer.calls) == 1
    assert scorer.calls[0][0] == "emma routine"
    assert [doc.metadata["id"] for doc in scorer.calls[0][1]] == [str(i) for i in range(20)]
    assert [doc.metadata["id"] for doc in returned] == ["19", "18", "17", "16", "15"]


async def test_rerank_ties_keep_stable_fetch_order():
    """R3 enabled: equal scores break ties by stable original fetch order."""
    candidates = _documents(20)
    scores = {str(i): 0.0 for i in range(20)}
    scores.update({"0": 9.0, "1": 8.0, "2": 7.0, "3": 6.0, "4": 5.0, "5": 5.0})
    scorer = _RecordingScorer(scores)
    retriever = docs_tool_module.RerankRetriever(
        vectorstore=_fake_vectorstore(candidates),
        agent_id=1,
        tool_id=DOCS_TOOL_ID,
        fetch_k=20,
        top_n=5,
        scorer=scorer,
    )

    returned = await retriever.ainvoke("emma routine")

    assert [doc.metadata["id"] for doc in returned] == ["0", "1", "2", "3", "4"]


# --- R5: the event exposes exactly the generation set -----------------------

async def test_retrieved_event_equals_post_rerank_generation_set():
    """R5: the emitted event carries exactly the post-rerank top_n; pool not disclosed."""
    tool = _configured_tool()
    candidates = _documents(20)
    scorer = _RecordingScorer({str(i): 10.0 - i for i in range(20)})
    vectorstore = _fake_vectorstore(candidates)
    handler = DocsStatusUpdateCallbackHandler(DOCS_TOOL_ID, "Docs")
    writer = MagicMock()

    with (
        patch.object(env, "docs_tool_rerank", True),
        patch.object(env, "docs_tool_rerank_fetch_k", 20),
        patch.object(env, "docs_tool_rerank_top_n", 5),
        patch.object(DocsTool, "_build_vectorstore", return_value=vectorstore),
        patch.object(DocsTool, "_build_rerank_scorer", return_value=scorer),
        patch("tero.tools.docs.tool.get_stream_writer", return_value=writer),
    ):
        retriever = tool._build_retriever()
        returned = await retriever.ainvoke("emma routine", config={"callbacks": [handler]})

    contents = [doc.page_content for doc in returned]
    assert contents == [f"chunk {i}" for i in range(5)]
    events = _retrieved_events(writer)
    assert len(events) == 1
    assert events[0].result == contents
    assert set(events[0].result or []) == {f"chunk {i}" for i in range(5)}
    assert all(f"chunk {i}" not in (events[0].result or []) for i in range(5, 20))


# --- R3: invalid configuration fails explicitly, no clamping -----------------

@pytest.mark.parametrize(
    "fetch_k,top_n,pattern",
    [
        (19, 5, r"fetch_k.*got 19"),
        (51, 5, r"fetch_k.*got 51"),
        (20, 21, r"top_n.*got 21"),
        (20, 0, r"top_n.*got 0"),
    ],
)
def test_invalid_rerank_config_raises_value_error_at_build_time(fetch_k, top_n, pattern):
    """R3 invalid: constructor reports the raw value; nothing is clamped."""
    with pytest.raises(ValueError, match=pattern):
        docs_tool_module.RerankRetriever(
            vectorstore=MagicMock(spec=VectorStore),
            agent_id=1,
            tool_id=DOCS_TOOL_ID,
            fetch_k=fetch_k,
            top_n=top_n,
            scorer=_RecordingScorer({}),
        )


@pytest.mark.parametrize("fetch_k,top_n", [(20, 5), (20, 20), (50, 5), (50, 50)])
def test_valid_rerank_config_boundaries_are_accepted(fetch_k, top_n):
    """R3: the 20-50 fetch_k range is inclusive and top_n may equal fetch_k."""
    retriever = docs_tool_module.RerankRetriever(
        vectorstore=MagicMock(spec=VectorStore),
        agent_id=1,
        tool_id=DOCS_TOOL_ID,
        fetch_k=fetch_k,
        top_n=top_n,
        scorer=_RecordingScorer({}),
    )

    assert (retriever.fetch_k, retriever.top_n) == (fetch_k, top_n)


def test_invalid_env_rerank_config_raises_before_any_retrieval():
    """R3 invalid: a bad setting fails while building the retriever, pre-query."""
    tool = _configured_tool()
    vectorstore = MagicMock(spec=VectorStore)

    with (
        patch.object(env, "docs_tool_rerank", True),
        patch.object(env, "docs_tool_rerank_fetch_k", 10),
        patch.object(env, "docs_tool_rerank_top_n", 5),
        patch.object(DocsTool, "_build_vectorstore", return_value=vectorstore),
        patch.object(DocsTool, "_build_rerank_scorer", return_value=_RecordingScorer({})),
    ):
        with pytest.raises(ValueError, match=r"fetch_k.*got 10"):
            tool._build_retriever()

    vectorstore.asimilarity_search.assert_not_called()


# --- pure ordering seam -----------------------------------------------------

def test_rank_candidates_orders_by_descending_score_with_stable_ties():
    candidates = _documents(5)

    ranked = docs_tool_module._rank_candidates(candidates, [3.0, 9.0, 5.0, 9.0, 1.0], top_n=3)

    assert [doc.metadata["id"] for doc in ranked] == ["1", "3", "2"]


def test_rank_candidates_keeps_the_first_top_n_of_an_all_ties_pool():
    candidates = _documents(20)

    ranked = docs_tool_module._rank_candidates(candidates, [7.0] * 20, top_n=5)

    assert [doc.metadata["id"] for doc in ranked] == ["0", "1", "2", "3", "4"]


def test_rank_candidates_rejects_a_score_count_mismatch():
    with pytest.raises(ValueError, match=r"2 scores.*3 candidates"):
        docs_tool_module._rank_candidates(_documents(3), [1.0, 2.0], top_n=2)


# --- TRIANGULATE: non-default parameters, short pools, float scores ---------

def test_rank_candidates_orders_fractional_and_negative_scores():
    candidates = _documents(5)

    ranked = docs_tool_module._rank_candidates(candidates, [0.5, -1.0, 7.5, 2.0, 7.5], top_n=3)

    assert [doc.metadata["id"] for doc in ranked] == ["2", "4", "3"]


async def test_enabled_pipeline_respects_a_non_default_fetch_and_top_n():
    """Triangulate: fetch_k=50/top_n=20 must work too, not just 20/5."""
    candidates = _documents(50)
    scorer = _RecordingScorer({str(i): float(i) for i in range(50)})
    vectorstore = _fake_vectorstore(candidates)
    retriever = docs_tool_module.RerankRetriever(
        vectorstore=vectorstore,
        agent_id=1,
        tool_id=DOCS_TOOL_ID,
        fetch_k=50,
        top_n=20,
        scorer=scorer,
    )

    returned = await retriever.ainvoke("emma routine")

    assert vectorstore.asimilarity_search.await_args.kwargs == {"k": 50}
    assert len(scorer.calls[0][1]) == 50
    assert [doc.metadata["id"] for doc in returned] == [str(i) for i in range(49, 29, -1)]


async def test_rerank_handles_a_pool_smaller_than_fetch_k():
    """Triangulate: a small corpus returns fewer candidates without error."""
    candidates = _documents(10)
    scorer = _RecordingScorer({str(i): float(i) for i in range(10)})
    retriever = docs_tool_module.RerankRetriever(
        vectorstore=_fake_vectorstore(candidates),
        agent_id=1,
        tool_id=DOCS_TOOL_ID,
        fetch_k=20,
        top_n=5,
        scorer=scorer,
    )

    returned = await retriever.ainvoke("emma routine")

    assert len(scorer.calls[0][1]) == 10
    assert [doc.metadata["id"] for doc in returned] == ["9", "8", "7", "6", "5"]


def test_invalid_env_top_n_raises_before_any_retrieval():
    """Triangulate: the top_n > fetch_k wiring path also fails pre-query."""
    tool = _configured_tool()
    vectorstore = MagicMock(spec=VectorStore)

    with (
        patch.object(env, "docs_tool_rerank", True),
        patch.object(env, "docs_tool_rerank_fetch_k", 20),
        patch.object(env, "docs_tool_rerank_top_n", 30),
        patch.object(DocsTool, "_build_vectorstore", return_value=vectorstore),
        patch.object(DocsTool, "_build_rerank_scorer", return_value=_RecordingScorer({})),
    ):
        with pytest.raises(ValueError, match=r"top_n.*got 30"):
            tool._build_retriever()

    vectorstore.asimilarity_search.assert_not_called()

