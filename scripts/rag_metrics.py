import argparse
import asyncio
import json
import os
import re
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import httpx

CITATION_PATTERN = re.compile(r"\[[^\]]+\]\(https?://[^)]+\)")
BASELINE_CASES = [
    {
        "question": "According to the uploaded document, what time does Emma wake up? Output only the time in H:MM format. Don't use clock tool.",
        "expected_answer_variants": ["7:35"],
    },
    {
        "question": "According to the uploaded document, what is the title of the document?",
        "expected_answer_variants": ["Emma's routine", "Emma’s routine"],
    },
    {
        "question": "According to the uploaded document, how much longer does Emma sleep at the weekends compared to weekdays?",
        "expected_answer_variants": ["an hour and a half more", "1 hour 30 minutes longer", "one hour thirty minutes longer"],
    },
    {
        "question": "According to the uploaded document, what does Emma do first when she gets up?",
        "expected_answer_variants": ["go to the bathroom", "goes to the bathroom", "bathroom first"],
    },
]
BASELINE_VERSION = 1
DEFAULT_BASELINE_PATH = Path("src/backend/tests/rag_baseline_metrics.json")
EVALUATION_CASES_FILE = Path("src/backend/tests/rag_evaluation_cases.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline and compare RAG metrics for the Docs tool.")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of the running Tero backend")
    parser.add_argument("--agent-id", type=int, default=1, help="Agent ID to use for the Docs tool")
    parser.add_argument("--tool-id", default="docs", help="Tool ID for the documents tool")
    parser.add_argument("--file", default="src/backend/tests/assets/Emma's routine.pdf", help="Local document file to upload if no evaluation dataset is present")
    parser.add_argument("--baseline-file", type=Path, default=DEFAULT_BASELINE_PATH, help="Path to baseline JSON file")
    parser.add_argument("--bearer-token", default=os.getenv("BEARER_TOKEN"), help="Raw bearer token for authenticated API access; do not include the 'Bearer ' prefix")
    parser.add_argument("--update-baseline", action="store_true", help="Update the baseline file with current metrics")
    parser.add_argument("--compare", action="store_true", help="Compare current metrics against the baseline file")
    parser.add_argument("--use-evaluation-cases", action="store_true", help="Use rag_evaluation_cases.json if present")
    return parser.parse_args()


async def configure_docs_tool(client: httpx.AsyncClient, base_url: str, agent_id: int, tool_id: str) -> None:
    url = f"{base_url}/api/agents/{agent_id}/tools"
    payload = {"toolId": tool_id, "config": {}}
    resp = await client.post(url, json=payload)
    if resp.status_code >= 400:
        print(f"Failed to configure tool: {resp.status_code} {resp.text}")
    resp.raise_for_status()


async def upload_document(client: httpx.AsyncClient, base_url: str, agent_id: int, tool_id: str, file_path: Path) -> int:
    url = f"{base_url}/api/agents/{agent_id}/tools/{tool_id}/files"
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f.read())}
        resp = await client.post(url, files=files)
    resp.raise_for_status()
    return resp.json()["id"]


async def wait_file_processed(client: httpx.AsyncClient, base_url: str, agent_id: int, tool_id: str, timeout: float = 300.0) -> None:
    url = f"{base_url}/api/agents/{agent_id}/tools/{tool_id}/files"
    deadline = time.time() + timeout
    while time.time() < deadline:
        resp = await client.get(url)
        resp.raise_for_status()
        files = resp.json()
        if all(file.get("status") != "PENDING" for file in files):
            return
        await asyncio.sleep(1)
    raise TimeoutError("Timed out waiting for document processing")


async def create_thread(client: httpx.AsyncClient, base_url: str, agent_id: int) -> int:
    url = f"{base_url}/api/threads"
    resp = await client.post(url, json={"agent_id": agent_id})
    if resp.status_code >= 400:
        print(f"create_thread failed: {resp.status_code}")
        print("response headers:", dict(resp.headers))
        print("response body:", resp.text)
    resp.raise_for_status()
    return resp.json()["id"]


async def ask_question(client: httpx.AsyncClient, base_url: str, thread_id: int, question: str) -> str:
    url = f"{base_url}/api/threads/{thread_id}/messages"
    data = {"text": question, "origin": "USER"}
    async with client.stream("POST", url, data=data, timeout=None) as resp:
        resp.raise_for_status()
        answer = ""
        async for line in resp.aiter_lines():
            if line.startswith("data: "):
                answer += line[len("data: "):]
    return answer


def extract_citations(response: str) -> list[str]:
    return CITATION_PATTERN.findall(response)


def extract_answer_text(response: str) -> str:
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


def matches_expected_answer(answer_text: str, expected_variants: list[str]) -> bool:
    answer_text_lower = answer_text.lower()
    return any(variant.lower() in answer_text_lower for variant in expected_variants)


def build_metrics(question: str, expected_answer_variants: list[str], response: str, latency_ms: float) -> dict[str, object]:
    answer_text = extract_answer_text(response)
    citations = extract_citations(response)
    return {
        "question": question,
        "response": response,
        "answer_text": answer_text,
        "latency_ms": latency_ms,
        "has_citation": len(citations) > 0,
        "citation_count": len(citations),
        "citations": citations,
        "expected_answer_variants": expected_answer_variants,
        "contains_expected_answer": matches_expected_answer(answer_text, expected_answer_variants),
    }


def print_summary(metrics: list[dict[str, object]]) -> None:
    print("\n=== RAG METRICS SUMMARY ===")
    for case in metrics:
        print(f"question: {case['question']}")
        print(f"  latency_ms: {case['latency_ms']}")
        print(f"  answer_text: {case['answer_text']}")
        print(f"  has_citation: {case['has_citation']}")
        print(f"  citation_count: {case['citation_count']}")
        print(f"  contains_expected_answer: {case['contains_expected_answer']}")
        print(f"  citations: {case['citations']}")
        print("  response sample: " + case['answer_text'][:80])
        print("")


def load_baseline_file(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict) and raw.get("cases"):
        return raw
    return {"version": BASELINE_VERSION, "cases": [raw]}


def compare_metrics(baseline: dict[str, Any], current: dict[str, Any]) -> None:
    baseline_by_question = {case['question']: case for case in baseline['cases']}
    print("\n=== COMPARISON VS BASELINE ===")
    matched = 0
    changed_answers = 0
    missing = 0
    for current_case in current['cases']:
        question = current_case['question']
        baseline_case = baseline_by_question.get(question)
        print(f"Question: {question}")
        if not baseline_case:
            print("  WARNING: baseline case missing")
            missing += 1
            continue
        matched += 1
        print(f"  baseline latency_ms: {baseline_case['latency_ms']}")
        print(f"  current latency_ms: {current_case['latency_ms']}")
        print(f"  delta latency_ms: {current_case['latency_ms'] - baseline_case['latency_ms']}")
        print(f"  baseline contains_expected_answer: {baseline_case['contains_expected_answer']}")
        print(f"  current contains_expected_answer: {current_case['contains_expected_answer']}")
        print(f"  baseline citation_count: {baseline_case['citation_count']}")
        print(f"  current citation_count: {current_case['citation_count']}")
        print(f"  baseline answer_text: {baseline_case.get('answer_text')}")
        print(f"  current answer_text: {current_case.get('answer_text')}")
        if baseline_case.get('answer_text') != current_case.get('answer_text'):
            print("  answer_text changed from baseline")
            changed_answers += 1
        elif baseline_case['response'] != current_case['response']:
            print("  response changed from baseline (likely metadata/noise)")
        print("")
    print(f"Summary: matched={matched}, missing={missing}, answer_text_changes={changed_answers}")


def _load_evaluation_cases() -> list[dict[str, Any]]:
    if EVALUATION_CASES_FILE.exists():
        return json.loads(EVALUATION_CASES_FILE.read_text(encoding='utf-8'))
    return BASELINE_CASES


def _collect_files_from_cases(cases: list[dict[str, Any]]) -> list[Path]:
    filenames = set()
    for case in cases:
        files = case.get("files")
        if files:
            filenames.update(files)
    return [Path("src/backend/tests/assets") / filename for filename in sorted(filenames)]


def save_baseline_file(path: Path, metrics: list[dict[str, object]]) -> None:
    baseline = {
        "version": BASELINE_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "cases": metrics,
    }
    path.write_text(json.dumps(baseline, indent=2, ensure_ascii=False))


def normalize_bearer_token(token: str | None) -> str | None:
    if not token:
        return None
    clean_token = token.strip()
    if clean_token.lower().startswith("bearer "):
        return clean_token.split(" ", 1)[1].strip()
    return clean_token


async def collect_metrics_for_cases(client: httpx.AsyncClient, base_url: str, agent_id: int, cases: list[dict[str, str]]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case in cases:
        thread_id = await create_thread(client, base_url, agent_id)
        start = time.perf_counter()
        response = await ask_question(client, base_url, thread_id, case['question'])
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        results.append(build_metrics(case['question'], case['expected_answer_variants'], response, latency_ms))
    return results


async def run(args: argparse.Namespace) -> None:
    headers = {}
    bearer_token = normalize_bearer_token(args.bearer_token)
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
        print(f"Authorization header set: Bearer <hidden> (token length {len(bearer_token)})")
    elif args.compare or args.update_baseline:
        print("Warning: no bearer token provided. Authenticated endpoints may reject the request with 401.")

    cases = _load_evaluation_cases() if args.use_evaluation_cases or EVALUATION_CASES_FILE.exists() else BASELINE_CASES
    file_paths = _collect_files_from_cases(cases)
    if not file_paths:
        file_paths = [Path(args.file)]
    print(f"Loaded {len(cases)} evaluation cases and {len(file_paths)} files to upload.")

    async with httpx.AsyncClient(headers=headers) as client:
        await configure_docs_tool(client, args.base_url, args.agent_id, args.tool_id)
        for file_path in file_paths:
            await upload_document(client, args.base_url, args.agent_id, args.tool_id, file_path)
        await wait_file_processed(client, args.base_url, args.agent_id, args.tool_id)
        results = await collect_metrics_for_cases(client, args.base_url, args.agent_id, cases)

    print_summary(results)

    if args.update_baseline:
        args.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        save_baseline_file(args.baseline_file, results)
        print(f"\nBaseline updated at {args.baseline_file}")

    if args.compare:
        if not args.baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {args.baseline_file}")
        baseline = load_baseline_file(args.baseline_file)
        compare_metrics(baseline, {"cases": results})


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
