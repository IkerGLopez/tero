"""Docs-tool reranker tests (fetaqa-retrieval-fix, phases 8-9, slices D1/D2).

Spec: openspec/changes/fetaqa-retrieval-fix/specs/docs-tool-retrieval/spec.md
(R3 flag-gated reranker + invalid configuration, R4 accounting/batching,
R5 post-rerank alignment).

D1 injects a deterministic scorer to pin the retriever plumbing. D2 exercises
the real `DocsTool._build_rerank_scorer` (batched JSON 0-10 scoring, usage
accounting, fail-open) with the LLM client and both repositories replaced by
in-process fakes, so no test here performs an LLM, network, database or Docker
call.
"""

import contextlib
import json
import logging
from typing import Callable, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.callbacks.manager import AsyncCallbackManager
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.vectorstores import VectorStore

from tero.ai_models.domain import LlmModel, LlmModelType, LlmModelVendor
from tero.core.env import Settings, env
from tero.tools.docs import tool as docs_tool_module
from tero.tools.docs.tool import (
    DOCS_TOOL_ID,
    DocsExecutionStep,
    DocsStatusUpdateCallbackHandler,
    DocsTool,
    DocumentUrlSolvingRetriever,
)
from tero.usage.domain import MessageUsage


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


# ===========================================================================
# Slice D2: batched JSON 0-10 scorer, usage accounting and fail-open
# ===========================================================================

_SCORER_MODEL_ID = "scorer-model"
_INPUT_TOKENS_PER_CALL = 100
_OUTPUT_TOKENS_PER_CALL = 10
# Fallback token count when a test does not care about batch sizes. Fixed on
# purpose: the real GPT-2 fallback tokenizer reaches for huggingface.co, and
# every test in this file must stay offline.
_DEFAULT_FAKE_TOKENS = 3


def _candidate_line(index: int, position: int) -> str:
    """One numbered candidate line of the scoring prompt (`position` is batch-local)."""
    return f"[{position}] chunk {index}"


def _prompt_candidates(prompt: str) -> list[str]:
    """The numbered candidate lines of a rendered scoring prompt, in order.

    Candidate lines are the only ones starting with `[`, so this reads back the
    exact candidate set the model was given (nothing more, nothing less, no
    reordering) without falling for substring collisions like `chunk 1` in `chunk 14`.
    """
    return [line for line in prompt.splitlines() if line.startswith("[")]


def _legal_scores(count: int) -> list[int]:
    """`count` scores inside the 0-10 contract, varied enough to expose reordering."""
    return [10 - (index % 11) for index in range(count)]


def _llm_model(model_id: str = _SCORER_MODEL_ID) -> LlmModel:
    """In-memory LlmModel (no DB): supplies the per-1k-token costs usage needs."""
    return LlmModel(
        id=model_id,
        name="Rerank scorer",
        description="Deterministic scoring model for the offline rerank tests",
        model_type=LlmModelType.CHAT,
        model_vendor=LlmModelVendor.OPENAI,
        token_limit=128000,
        output_token_limit=8000,
        prompt_1k_token_usd=0.01,
        completion_1k_token_usd=0.02,
    )


def _prompt_text(messages: Sequence) -> str:
    content = messages[-1].content
    return content if isinstance(content, str) else str(content)


class _FakeChatModel(GenericFakeChatModel):
    """Deterministic chat model standing in for the LLM client seam.

    Records every rendered prompt (one entry per LLM call) and lets a test
    control the token count used for batch packing.
    """

    token_counter: Optional[Callable[[str], int]] = None
    prompts: list[str] = []

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(_prompt_text(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

    def get_num_tokens(self, text: str) -> int:
        return self.token_counter(text) if self.token_counter else _DEFAULT_FAKE_TOKENS


class _ExplodingChatModel(_FakeChatModel):
    """Generation model that fails, to prove `_run` still flushes usage in `finally`."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise RuntimeError("generation backend down")


def _fake_chat_model(
    responses: Sequence[str], token_counter: Optional[Callable[[str], int]] = None
) -> _FakeChatModel:
    return _FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    content=response,
                    usage_metadata={
                        "input_tokens": _INPUT_TOKENS_PER_CALL,
                        "output_tokens": _OUTPUT_TOKENS_PER_CALL,
                        "total_tokens": _INPUT_TOKENS_PER_CALL + _OUTPUT_TOKENS_PER_CALL,
                    },
                )
                for response in responses
            ]
        ),
        token_counter=token_counter,
    )


def _scorer_model_repository() -> MagicMock:
    repository = MagicMock()
    repository.return_value.find_by_id = AsyncMock(return_value=_llm_model())
    return repository


def _added_usages(usage_repo: MagicMock) -> list:
    return [call.args[0] for call in usage_repo.return_value.add.await_args_list]


@contextlib.contextmanager
def _offline_run(*, candidates: Sequence[Document], chat_models: Sequence, rerank: bool = True):
    """Patch every seam `DocsTool._run` touches: no LLM, DB, network or Docker.

    `chat_models` is the ordered `build_chat_model` side effect: the generation
    model first (`_run` builds it up front), the scoring model second (built
    lazily by the scorer on its first call).
    """
    usage_repo = MagicMock()
    usage_repo.return_value.add = AsyncMock()
    writer = MagicMock()
    with (
        patch.object(env, "docs_tool_rerank", rerank),
        patch.object(env, "docs_tool_rerank_fetch_k", 20),
        patch.object(env, "docs_tool_rerank_top_n", 5),
        patch.object(env, "docs_tool_rerank_model", _SCORER_MODEL_ID),
        patch.object(DocsTool, "_build_vectorstore", return_value=_fake_vectorstore(candidates)),
        patch.object(docs_tool_module.ai_factory, "build_chat_model", MagicMock(side_effect=list(chat_models))),
        patch.object(docs_tool_module, "AiModelRepository", _scorer_model_repository()),
        patch.object(docs_tool_module, "UsageRepository", usage_repo),
        patch("tero.tools.docs.tool.get_stream_writer", return_value=writer),
        patch("tero.tools.docs.tool.ensure_config", return_value={"callbacks": AsyncCallbackManager([])}),
    ):
        yield usage_repo, writer


# --- R4: the JSON 0-10 score contract --------------------------------------

def test_parse_rerank_scores_reads_a_json_array_in_candidate_order():
    """R4 scoring: scores come back as floats, in the order they were requested."""
    scores = docs_tool_module._parse_rerank_scores("[10, 0, 7.5, 3]", expected_count=4)

    assert scores == [10.0, 0.0, 7.5, 3.0]


def test_parse_rerank_scores_accepts_a_markdown_fenced_payload():
    """R4 scoring: models routinely wrap JSON in a fence; the parser strips it."""
    scores = docs_tool_module._parse_rerank_scores("```json\n[1, 2, 3]\n```", expected_count=3)

    assert scores == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    "raw,count,pattern",
    [
        ("[1, 2, 3]", 4, r"3 scores.*4 candidates"),
        ("scores: [1, 2, 3]", 3, r"not valid JSON"),
        ('{"scores": [1, 2, 3]}', 3, r"must be a JSON array"),
        ('[1, "high", 3]', 3, r"must be numbers"),
        ("[1, 11, 3]", 3, r"outside the 0-10 range"),
    ],
)
def test_parse_rerank_scores_rejects_contract_violations(raw, count, pattern):
    """R4 scoring: the count and range contract is strict so ordering cannot drift."""
    with pytest.raises(ValueError, match=pattern):
        docs_tool_module._parse_rerank_scores(raw, expected_count=count)


def test_batch_documents_by_tokens_packs_candidates_within_the_budget():
    """R4 batched: batches are consecutive, token-bounded and never drop a candidate."""
    tokens = {0: 7, 1: 7, 2: 7}
    counter = lambda text: tokens[int(text.removeprefix("chunk "))]
    documents = _documents(3)

    exact_fit = docs_tool_module._batch_documents_by_tokens(documents, counter, token_budget=21)
    overflow = docs_tool_module._batch_documents_by_tokens(documents, counter, token_budget=20)
    oversized = docs_tool_module._batch_documents_by_tokens(
        documents, lambda text: 30 if text == "chunk 0" else 7, token_budget=20
    )

    assert [[int(doc.metadata["id"]) for doc in batch] for batch in exact_fit] == [[0, 1, 2]]
    assert [[int(doc.metadata["id"]) for doc in batch] for batch in overflow] == [[0, 1], [2]]
    assert [[int(doc.metadata["id"]) for doc in batch] for batch in oversized] == [[0], [1, 2]]


# --- R4: the real scorer, batched through the LLM client seam ---------------

async def test_scorer_scores_the_whole_pool_in_one_token_bounded_call():
    """R4 batched: 20 small candidates cost ONE LLM call, not one call per candidate."""
    tool = _configured_tool()
    candidates = _documents(20)
    llm = _fake_chat_model([json.dumps(_legal_scores(20))], token_counter=lambda _text: 3)
    build_model = MagicMock(return_value=llm)
    model_repo = _scorer_model_repository()

    with (
        patch.object(env, "docs_tool_rerank_model", _SCORER_MODEL_ID),
        patch.object(docs_tool_module.ai_factory, "build_chat_model", build_model),
        patch.object(docs_tool_module, "AiModelRepository", model_repo),
    ):
        scorer = tool._build_rerank_scorer()
        scores = await scorer("emma routine", candidates)

    assert scores == [float(score) for score in _legal_scores(20)]
    assert len(llm.prompts) == 1
    assert build_model.call_args.args[0] == _SCORER_MODEL_ID
    assert model_repo.return_value.find_by_id.await_args.args[0] == _SCORER_MODEL_ID
    assert _prompt_candidates(llm.prompts[0]) == [
        _candidate_line(index, index) for index in range(20)
    ]


async def test_scorer_falls_back_to_the_configured_generator_model():
    """Triangulate R3: with no explicit scoring model, the configured generator is used."""
    tool = _configured_tool()
    llm = _fake_chat_model([json.dumps([5, 5])])
    build_model = MagicMock(return_value=llm)
    model_repo = _scorer_model_repository()

    with (
        patch.object(env, "docs_tool_rerank_model", None),
        patch.object(docs_tool_module.ai_factory, "build_chat_model", build_model),
        patch.object(docs_tool_module, "AiModelRepository", model_repo),
    ):
        scores = await tool._build_rerank_scorer()("emma routine", _documents(2))

    assert scores == [5.0, 5.0]
    assert build_model.call_args.args[0] == env.internal_generator_model
    assert model_repo.return_value.find_by_id.await_args.args[0] == env.internal_generator_model


async def test_scorer_splits_a_pool_across_token_bounded_batches():
    """R4 batched: a tight token budget produces one call per batch, all still scored."""
    tool = _configured_tool()
    candidates = _documents(20)
    groups = ((0, 7), (7, 14), (14, 20))
    llm = _fake_chat_model(
        [json.dumps(_legal_scores(20)[start:end]) for start, end in groups],
        token_counter=lambda _text: 7,
    )

    with (
        patch.object(env, "docs_tool_rerank_model", _SCORER_MODEL_ID),
        patch.object(docs_tool_module, "RERANK_BATCH_TOKEN_BUDGET", 50),
        patch.object(docs_tool_module.ai_factory, "build_chat_model", return_value=llm),
        patch.object(docs_tool_module, "AiModelRepository", _scorer_model_repository()),
    ):
        scorer = tool._build_rerank_scorer()
        scores = await scorer("emma routine", candidates)

    assert scores == [float(score) for score in _legal_scores(20)]
    assert len(llm.prompts) == 3
    assert [_prompt_candidates(prompt) for prompt in llm.prompts] == [
        [_candidate_line(index, position) for position, index in enumerate(range(start, end))]
        for start, end in groups
    ]


# --- R3: fail-open, the optional stage cannot break retrieval ---------------

async def test_fail_open_keeps_the_original_fetch_order_when_scoring_raises(caplog):
    """R3: a scorer failure keeps retrieval alive with the original fetch order."""
    candidates = _documents(20)
    writer = MagicMock()
    handler = DocsStatusUpdateCallbackHandler(DOCS_TOOL_ID, "Docs")

    async def failing_scorer(query: str, documents: Sequence[Document]) -> list[float]:
        raise RuntimeError("scoring backend down")

    retriever = docs_tool_module.RerankRetriever(
        vectorstore=_fake_vectorstore(candidates),
        agent_id=1,
        tool_id=DOCS_TOOL_ID,
        fetch_k=20,
        top_n=5,
        scorer=failing_scorer,
    )

    with (
        patch("tero.tools.docs.tool.get_stream_writer", return_value=writer),
        caplog.at_level(logging.WARNING, logger=docs_tool_module.__name__),
    ):
        returned = await retriever.ainvoke("emma routine", config={"callbacks": [handler]})

    assert [doc.metadata["id"] for doc in returned] == ["0", "1", "2", "3", "4"]
    assert "scoring backend down" in caplog.text
    events = _retrieved_events(writer)
    assert len(events) == 1
    assert events[0].result == [f"chunk {i}" for i in range(5)]


async def test_fail_open_covers_a_real_scorer_that_violates_the_score_contract():
    """R3/R4: the real scorer's own contract failures also fail open, not just a fake."""
    tool = _configured_tool()
    candidates = _documents(20)
    llm = _fake_chat_model([json.dumps(_legal_scores(19))])

    with (
        patch.object(env, "docs_tool_rerank", True),
        patch.object(env, "docs_tool_rerank_fetch_k", 20),
        patch.object(env, "docs_tool_rerank_top_n", 5),
        patch.object(env, "docs_tool_rerank_model", _SCORER_MODEL_ID),
        patch.object(DocsTool, "_build_vectorstore", return_value=_fake_vectorstore(candidates)),
        patch.object(docs_tool_module.ai_factory, "build_chat_model", return_value=llm),
        patch.object(docs_tool_module, "AiModelRepository", _scorer_model_repository()),
    ):
        retriever = tool._build_retriever()
        returned = await retriever.ainvoke("emma routine")

    assert [doc.metadata["id"] for doc in returned] == ["0", "1", "2", "3", "4"]


# --- R4: usage accounting through `_run` ------------------------------------

async def test_run_scores_across_batches_and_persists_the_scoring_usage():
    """R4 usage included + R5: `_run` accounts every scoring call and emits the top_n."""
    tool = _configured_tool()
    candidates = _documents(20)
    generation = _fake_chat_model(["Emma wakes up at 7:35."])
    scoring = _fake_chat_model(
        [json.dumps([0] * 10), json.dumps([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])],
        token_counter=lambda _text: 10,
    )

    with (
        _offline_run(candidates=candidates, chat_models=[generation, scoring]) as (usage_repo, writer),
        patch.object(docs_tool_module, "RERANK_BATCH_TOKEN_BUDGET", 100),
    ):
        answer = await tool._run("emma routine")

    assert answer == "Emma wakes up at 7:35."
    assert len(scoring.prompts) == 2
    events = _retrieved_events(writer)
    assert len(events) == 1
    assert events[0].result == [f"chunk {i}" for i in range(10, 15)]

    added = _added_usages(usage_repo)
    assert added[0] is tool.embedding_usage
    scoring_usage = added[1]
    assert isinstance(scoring_usage, MessageUsage)
    assert scoring_usage.model_id == _SCORER_MODEL_ID
    assert (scoring_usage.prompt_usage.quantity, scoring_usage.completion_usage.quantity) == (200, 20)
    assert scoring_usage.usd_cost == pytest.approx(0.0024)


async def test_run_still_persists_scoring_usage_when_generation_fails():
    """R4 usage included: `_run`'s finally accounts scoring even if generation breaks."""
    tool = _configured_tool()
    candidates = _documents(20)
    scoring = _fake_chat_model(
        [json.dumps(list(range(1, 11))), json.dumps(list(range(1, 11)))],
        token_counter=lambda _text: 10,
    )

    with (
        _offline_run(candidates=candidates, chat_models=[_ExplodingChatModel(messages=iter([])), scoring]) as (usage_repo, _),
        patch.object(docs_tool_module, "RERANK_BATCH_TOKEN_BUDGET", 100),
    ):
        with pytest.raises(RuntimeError, match="generation backend down"):
            await tool._run("emma routine")

    added = _added_usages(usage_repo)
    scoring_usages = [usage for usage in added if isinstance(usage, MessageUsage)]
    assert len(scoring_usages) == 1
    assert scoring_usages[0].prompt_usage.quantity == 200


async def test_run_does_not_re_account_a_previous_runs_scoring_usage():
    """R4: scoring usage is per run, so a reused tool instance never double-counts."""
    tool = _configured_tool()
    candidates = _documents(20)
    generation = _fake_chat_model(["First answer.", "Second answer."])
    scoring = _fake_chat_model([json.dumps(_legal_scores(20))], token_counter=lambda _text: 3)

    # Generation model -> scoring model (first run, rerank on) -> generation model again.
    with _offline_run(candidates=candidates, chat_models=[generation, scoring, generation]) as (usage_repo, _):
        first = await tool._run("emma routine")
        with patch.object(env, "docs_tool_rerank", False):
            second = await tool._run("emma routine again")

    assert (first, second) == ("First answer.", "Second answer.")
    added = _added_usages(usage_repo)
    assert added[0] is tool.embedding_usage
    assert isinstance(added[1], MessageUsage)
    # The rerank-off run re-accounts embedding usage only: no stale scoring usage.
    assert (added[2], added[3], len(added)) == (tool.embedding_usage, None, 4)

