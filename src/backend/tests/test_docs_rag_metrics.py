import json
import os
import re
import time
from pathlib import Path
from typing import Any

from .common import (
    AGENT_ID,
    add_message_to_thread,
    await_files_processed,
    configure_agent_tool,
    create_thread,
    find_agent_tool_config_files,
    find_asset_bytes,
    upload_agent_tool_config_file,
)
from httpx import AsyncClient
from tero.files.domain import FileStatus
from tero.tools.docs import DOCS_TOOL_ID

BASELINE_FILE = Path(__file__).with_name("rag_baseline_metrics.json")
EVALUATION_CASES_FILE = Path(__file__).with_name("rag_evaluation_cases.json")
ASSETS_DIR = Path(__file__).with_name("assets")
CITATION_PATTERN = re.compile(r"\[[^\]]+\]\(https?://[^)]+\)")
BASELINE_CASES = [
    {
        "question": "What time does Emma wake up according to the document? Output only the time in H:MM format. Don't use clock tool.",
        "expected_answer_fragment": "7:35",
    },
    {
        "question": "What is the title of the document?",
        "expected_answer_fragment": "Emma's routine",
    },
    {
        "question": "How much longer does Emma sleep at the weekends compared to weekdays?",
        "expected_answer_fragment": "an hour and a half more",
    },
    {
        "question": "What does Emma do first when she gets up?",
        "expected_answer_fragment": "go to the bathroom",
    },
]
BASELINE_VERSION = 1


async def _configure_docs_tool_with_file(client: AsyncClient, filename: str) -> int:
    await configure_agent_tool(AGENT_ID, DOCS_TOOL_ID, {}, client)
    file_id = await upload_agent_tool_config_file(
        AGENT_ID,
        DOCS_TOOL_ID,
        client,
        filename=filename,
        content=await find_asset_bytes(filename),
    )
    await await_files_processed(AGENT_ID, DOCS_TOOL_ID, file_id, client)
    return file_id


async def _configure_docs_tool_with_files(client: AsyncClient, filenames: list[str]) -> list[int]:
    await configure_agent_tool(AGENT_ID, DOCS_TOOL_ID, {}, client)
    file_ids: list[int] = []
    for filename in filenames:
        file_id = await upload_agent_tool_config_file(
            AGENT_ID,
            DOCS_TOOL_ID,
            client,
            filename=filename,
            content=await find_asset_bytes(filename),
        )
        await await_files_processed(AGENT_ID, DOCS_TOOL_ID, file_id, client)
        file_ids.append(file_id)
    return file_ids


def _load_evaluation_cases() -> list[dict[str, Any]]:
    if EVALUATION_CASES_FILE.exists():
        return json.loads(EVALUATION_CASES_FILE.read_text())
    return BASELINE_CASES


def _expected_answer_variants(case: dict[str, Any]) -> list[str]:
    if case.get("expected_answer_variants"):
        return case["expected_answer_variants"]
    if case.get("expected_answer_fragment"):
        return [case["expected_answer_fragment"]]
    return []


async def _answer_question(question: str, client: AsyncClient) -> str:
    resp = await create_thread(AGENT_ID, client)
    resp.raise_for_status()
    thread_id = resp.json()["id"]
    async with add_message_to_thread(client, thread_id, question) as resp:
        resp.raise_for_status()
        response = ""
        async for chunk in resp.aiter_text():
            if "event: error" in chunk:
                raise Exception(chunk)
            for event in chunk.replace("\r\n", "\n").split("\n\n"):
                if "event: error" in event:
                    raise Exception(event)
                for line in event.split("\n"):
                    if line.startswith("data: "):
                        response += line[len("data: "):]
        return response


def _extract_citations(response: str) -> list[str]:
    return CITATION_PATTERN.findall(response)


def _extract_answer_text(response: str) -> str:
    decoder = json.JSONDecoder()
    idx = 0
    length = len(response)
    while idx < length and response[idx].isspace():
        idx += 1
    while idx < length and response[idx] == "{":
        try:
            _, end = decoder.raw_decode(response, idx)
        except ValueError:
            break
        idx = end
        while idx < length and response[idx].isspace():
            idx += 1
    answer = response[idx:].strip()
    answer = re.sub(r"\s*\{\"answerMessageId\".*$", "", answer, flags=re.DOTALL)
    return answer.strip()


def _save_baseline_snapshot(metrics: list[dict[str, Any]]) -> None:
    baseline = {"version": BASELINE_VERSION, "cases": metrics}
    BASELINE_FILE.write_text(json.dumps(baseline, indent=2, ensure_ascii=False))


async def _collect_docs_rag_metrics(client: AsyncClient) -> list[dict[str, Any]]:
    cases = _load_evaluation_cases()
    filenames: set[str] = set()
    for case in cases:
        files = case.get("files") or ["Emma's routine.pdf"]
        filenames.update(files)
    await _configure_docs_tool_with_files(client, sorted(filenames))

    metrics = []
    for case in cases:
        start = time.perf_counter()
        response = await _answer_question(case["question"], client)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        answer_text = _extract_answer_text(response)
        citations = _extract_citations(response)
        expected_variants = _expected_answer_variants(case)
        metrics.append({
            "question": case["question"],
            "response": response,
            "answer_text": answer_text,
            "latency_ms": latency_ms,
            "has_citation": len(citations) > 0,
            "citation_count": len(citations),
            "citations": citations,
            "expected_answer_variants": expected_variants,
            "contains_expected_answer": any(
                variant.lower() in answer_text.lower() for variant in expected_variants
            ) if expected_variants else False,
        })
    if os.getenv("UPDATE_RAG_BASELINE"):
        _save_baseline_snapshot(metrics)
    return metrics


async def test_docs_rag_baseline_metrics_are_collectable(client: AsyncClient):
    cases = _load_evaluation_cases()
    metrics = await _collect_docs_rag_metrics(client)
    assert len(metrics) == len(cases)
    for case in metrics:
        assert case["latency_ms"] > 0
        assert isinstance(case["response"], str)
        assert isinstance(case["has_citation"], bool)


async def test_docs_rag_baseline_quality(client: AsyncClient):
    metrics = await _collect_docs_rag_metrics(client)
    assert all(case["contains_expected_answer"] for case in metrics)


async def test_docs_rag_baseline_citation_reporting(client: AsyncClient):
    metrics = await _collect_docs_rag_metrics(client)
    for case in metrics:
        assert "citation_count" in case
        assert isinstance(case["citation_count"], int)


async def test_docs_rag_baseline_snapshot_exists_or_can_be_created(client: AsyncClient):
    metrics = await _collect_docs_rag_metrics(client)
    if BASELINE_FILE.exists():
        baseline = json.loads(BASELINE_FILE.read_text())
        cases = baseline.get("cases") if isinstance(baseline, dict) else None
        if cases:
            assert len(cases) == len(metrics)
            assert all(case["question"] == metrics[i]["question"] for i, case in enumerate(cases))
        else:
            assert baseline["question"] == metrics[0]["question"]
            assert isinstance(baseline["latency_ms"], (int, float))
            assert "response" in baseline
    else:
        assert os.getenv("UPDATE_RAG_BASELINE") is not None, (
            "Baseline snapshot is missing. Run with UPDATE_RAG_BASELINE=1 to generate it."
        )


async def test_docs_tool_processes_all_pdf_assets(client: AsyncClient):
    pdf_files = sorted([path.name for path in ASSETS_DIR.glob("*.pdf")])
    assert pdf_files, "No PDF assets found in tests/assets"
    await _configure_docs_tool_with_files(client, pdf_files)
    resp = await find_agent_tool_config_files(AGENT_ID, DOCS_TOOL_ID, client)
    files = resp.json()
    assert len(files) >= len(pdf_files)
    assert all(file["status"] != FileStatus.PENDING.value for file in files)
