"""RAGAS metric construction and citation-context pairing for the rag_eval runner."""

import config  # Bootstrap: repo .env + sys.path
import re

from ragas.metrics import DiscreteMetric
from ragas.metrics._context_precision import ContextPrecision
from ragas.metrics._context_recall import LLMContextRecall
from ragas.metrics._faithfulness import Faithfulness


def _build_metrics(llm):
    context_recall = LLMContextRecall(llm=llm)
    context_precision = ContextPrecision(llm=llm)
    faithfulness = Faithfulness(llm=llm)

    correctness = DiscreteMetric(
        name="correctness",
        prompt=(
            "Rate the response against the grading notes using this scale:\n"
            "0: incorrect, irrelevant, or covers less than 20% of the expected key claims.\n"
            "1: contains between 20% and 50% of the expected key claims.\n"
            "2: contains between 50% and 70% of the expected key claims.\n"
            "3: contains between 70% and 90% of the expected key claims, minor omissions.\n"
            "4: contains more than 90% of the expected key claims, no relevant errors.\n"
            "Return only the number (0, 1, 2, 3, or 4).\n"
            "Response: {response}\n"
            "Grading Notes: {grading_notes}"
        ),
        allowed_values=["0", "1", "2", "3", "4"],
    )

    citation_faithfulness = DiscreteMetric(
        name="citation_faithfulness",
        prompt=(
            "A RAG system answered a question by citing a document chunk. "
            "Determine if the cited chunk actually supports the claims made in the answer.\n"
            "Answer: {response}\n"
            "Cited chunk: {cited_chunk}\n"
            "Return 'supported' if the chunk backs the answer, 'unsupported' otherwise."
        ),
        allowed_values=["supported", "unsupported"],
    )

    return context_recall, context_precision, faithfulness, correctness, citation_faithfulness


def _pair_citations_with_contexts(
    citations: list[str],
    contexts: list[str],
) -> list[tuple[str, str]]:
    """Pair citations with retrieved contexts via chunk_N markers.

    Returns empty list when no valid chunk_N markers found.
    No positional fallback — only chunk_N index matching is used.
    """
    _CHUNK_IDX_RE = re.compile(r"chunk_(\d+)")
    pairs: list[tuple[str, str]] = []
    seen: set[int] = set()
    for cite in citations:
        m = _CHUNK_IDX_RE.search(cite)
        if m:
            idx = int(m.group(1)) - 1  # chunk_1 → contexts[0]
            if 0 <= idx < len(contexts) and idx not in seen:
                pairs.append((cite, contexts[idx]))
                seen.add(idx)
    return pairs
