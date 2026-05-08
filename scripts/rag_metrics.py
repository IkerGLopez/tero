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

USAGE_MODES = ("basic", "medium", "advanced")
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
DEFAULT_EVALUATION_CASES_FILE = Path("src/backend/tests/rag_evaluation_cases.json")


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
    parser.add_argument("--evaluation-cases-file", type=Path, default=DEFAULT_EVALUATION_CASES_FILE, help="Path to evaluation cases JSON file")
    parser.add_argument("--usage-mode", choices=USAGE_MODES, default=None, help="Docs usage mode to evaluate (optional). If omitted, keeps legacy tool config without usageMode.")
    parser.add_argument("--all-usage-modes", action="store_true", help="Run metrics for basic, medium and advanced in one execution")
    parser.add_argument("--force-docs", action="store_true", help="Invoke the Docs tool directly instead of asking the agent to decide whether to use it")
    parser.add_argument("--export-file", type=Path, default=None, help="Optional JSON output file for current run results")
    return parser.parse_args()


async def configure_docs_tool(client: httpx.AsyncClient, base_url: str, agent_id: int, tool_id: str, usage_mode: str | None) -> None:
    url = f"{base_url}/api/agents/{agent_id}/tools"
    if usage_mode is None and await is_tool_already_configured(client, base_url, agent_id, tool_id):
        print("Docs tool is already configured; using existing legacy configuration.")
        return

    configs_to_try: list[dict[str, str | bool]]
    if usage_mode:
        configs_to_try = [
            {"advancedFileProcessing": False, "usageMode": usage_mode},
            {"usageMode": usage_mode},
        ]
    else:
        # Legacy-first order: keep prior behavior, then try minimal fallback.
        configs_to_try = [
            {"advancedFileProcessing": False},
            {},
        ]

    failures: list[str] = []
    for config in configs_to_try:
        payload = {"toolId": tool_id, "config": config}
        resp = await client.post(url, json=payload)
        if resp.status_code < 400:
            return
        failures.append(f"config={config} -> {resp.status_code} {resp.text}")
        print(f"Failed to configure tool with {config}: {resp.status_code} {resp.text}")

    if usage_mode is None and await is_tool_already_configured(client, base_url, agent_id, tool_id):
        print("Docs tool is already configured; continuing with existing legacy configuration.")
        return

    raise RuntimeError(
        "Could not configure Docs tool with any compatible payload. "
        + " | ".join(failures)
    )


async def is_tool_already_configured(client: httpx.AsyncClient, base_url: str, agent_id: int, tool_id: str) -> bool:
    resp = await client.get(f"{base_url}/api/agents/{agent_id}/tools")
    if resp.status_code >= 400:
        print(f"Failed to inspect existing tools: {resp.status_code} {resp.text}")
        return False
    configured_tools = resp.json()
    return any(tool.get("toolId") == tool_id for tool in configured_tools)


async def upload_document(client: httpx.AsyncClient, base_url: str, agent_id: int, tool_id: str, file_path: Path) -> int:
    url = f"{base_url}/api/agents/{agent_id}/tools/{tool_id}/files"
    with file_path.open("rb") as f:
        files = {"file": (file_path.name, f.read())}
        resp = await client.post(url, files=files)
    resp.raise_for_status()
    return resp.json()["id"]


async def wait_file_processed(
    client: httpx.AsyncClient,
    base_url: str,
    agent_id: int,
    tool_id: str,
    uploaded_file_ids: list[int],
    timeout: float = 300.0,
) -> None:
    url = f"{base_url}/api/agents/{agent_id}/tools/{tool_id}/files"
    deadline = time.time() + timeout
    pending_ids = set(uploaded_file_ids)
    last_status_by_id: dict[int, str] = {}
    while time.time() < deadline:
        resp = await client.get(url)
        resp.raise_for_status()
        files = resp.json()
        files_by_id = {int(file["id"]): file for file in files if "id" in file}
        for file_id in uploaded_file_ids:
            file_data = files_by_id.get(file_id)
            if not file_data:
                last_status_by_id[file_id] = "MISSING"
                continue
            status = str(file_data.get("status", "UNKNOWN"))
            last_status_by_id[file_id] = status
            if status != "PENDING":
                pending_ids.discard(file_id)
        if not pending_ids:
            return
        await asyncio.sleep(1)
    status_summary = ", ".join(f"{file_id}:{last_status_by_id.get(file_id, 'UNKNOWN')}" for file_id in uploaded_file_ids)
    raise TimeoutError(f"Timed out waiting for document processing. File statuses: {status_summary}")


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


async def ask_docs_tool(client: httpx.AsyncClient, base_url: str, agent_id: int, tool_id: str, question: str) -> str:
    url = f"{base_url}/api/agents/{agent_id}/tools/{tool_id}/invoke"
    resp = await client.post(url, json={"userQuery": question}, timeout=None)
    if resp.status_code == 404:
        tools_url = f"{base_url}/api/agents/{agent_id}/tools"
        detail = resp.text
        try:
            tools_resp = await client.get(tools_url)
            if tools_resp.status_code < 400:
                configured = [tool.get("toolId") for tool in tools_resp.json()]
                detail = (
                    f"{detail}. Configured tools for agent {agent_id}: {configured}. "
                    "If 'docs' is present, restart the backend so the /invoke route is loaded."
                )
            else:
                detail = f"{detail}. Could not inspect configured tools: {tools_resp.status_code} {tools_resp.text}"
        except Exception as exc:
            detail = f"{detail}. Could not inspect configured tools: {exc}"
        raise RuntimeError(f"Forced Docs invocation endpoint not found or unavailable: {url}. {detail}") from None
    resp.raise_for_status()
    return resp.json()["answer"]


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


def print_summary(metrics: list[dict[str, object]], usage_mode: str | None) -> None:
    mode_label = usage_mode or "legacy-default"
    print(f"\n=== RAG METRICS SUMMARY ({mode_label}) ===")
    for case in metrics:
        print(f"question: {case['question']}")
        print(f"  latency_ms: {case['latency_ms']}")
        print(f"  answer_text: {case['answer_text']}")
        print(f"  has_citation: {case['has_citation']}")
        print(f"  citation_count: {case['citation_count']}")
        print(f"  docs_tool_invoked: {case.get('docs_tool_invoked')}")
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


def aggregate_mode_summary(metrics: list[dict[str, object]]) -> dict[str, float | int]:
    total = len(metrics)
    correct = sum(1 for case in metrics if case["contains_expected_answer"])
    with_citations = sum(1 for case in metrics if case["has_citation"])
    with_docs = sum(1 for case in metrics if case.get("docs_tool_invoked"))
    avg_latency = round(sum(float(case["latency_ms"]) for case in metrics) / total, 2) if total else 0.0
    avg_citations = round(sum(int(case["citation_count"]) for case in metrics) / total, 2) if total else 0.0
    return {
        "cases": total,
        "correct_answers": correct,
        "answer_accuracy_ratio": round(correct / total, 4) if total else 0.0,
        "cases_with_docs_invoked": with_docs,
        "docs_invocation_ratio": round(with_docs / total, 4) if total else 0.0,
        "cases_with_citation": with_citations,
        "citation_coverage_ratio": round(with_citations / total, 4) if total else 0.0,
        "avg_latency_ms": avg_latency,
        "avg_citation_count": avg_citations,
    }


def print_mode_comparison(results_by_mode: dict[str, list[dict[str, object]]]) -> None:
    print("\n=== USAGE MODE COMPARISON ===")
    for mode in USAGE_MODES:
        metrics = results_by_mode[mode]
        summary = aggregate_mode_summary(metrics)
        print(f"{mode}:")
        print(f"  cases: {summary['cases']}")
        print(f"  correct_answers: {summary['correct_answers']} ({summary['answer_accuracy_ratio']})")
        print(f"  cases_with_docs_invoked: {summary['cases_with_docs_invoked']} ({summary['docs_invocation_ratio']})")
        print(f"  cases_with_citation: {summary['cases_with_citation']} ({summary['citation_coverage_ratio']})")
        print(f"  avg_latency_ms: {summary['avg_latency_ms']}")
        print(f"  avg_citation_count: {summary['avg_citation_count']}")
        print("")


def _load_evaluation_cases(path: Path) -> list[dict[str, Any]]:
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
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


def save_results_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def normalize_bearer_token(token: str | None) -> str | None:
    if not token:
        return None
    clean_token = token.strip()
    if clean_token.lower().startswith("bearer "):
        return clean_token.split(" ", 1)[1].strip()
    return clean_token


async def collect_metrics_for_cases(
    client: httpx.AsyncClient,
    base_url: str,
    agent_id: int,
    tool_id: str,
    cases: list[dict[str, str]],
    force_docs: bool,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case in cases:
        start = time.perf_counter()
        if force_docs:
            response = await ask_docs_tool(client, base_url, agent_id, tool_id, case['question'])
        else:
            thread_id = await create_thread(client, base_url, agent_id)
            response = await ask_question(client, base_url, thread_id, case['question'])
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        metrics = build_metrics(case['question'], case['expected_answer_variants'], response, latency_ms)
        metrics["force_docs"] = force_docs
        metrics["docs_tool_invoked"] = force_docs or '"toolName": "docs"' in response or '"toolName":"docs"' in response
        results.append(metrics)
    return results


def _with_usage_mode(metrics: list[dict[str, object]], usage_mode: str | None) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for case in metrics:
        row = dict(case)
        row["usage_mode"] = usage_mode or "legacy-default"
        enriched.append(row)
    return enriched


def _mode_baseline_path(base: Path, usage_mode: str) -> Path:
    return base.with_name(f"{base.stem}_{usage_mode}{base.suffix}")


async def run_for_mode(args: argparse.Namespace, usage_mode: str | None, cases: list[dict[str, Any]], file_paths: list[Path], headers: dict[str, str]) -> list[dict[str, object]]:
    if usage_mode:
        print(f"\nRunning RAG metrics for usageMode={usage_mode}")
    else:
        print("\nRunning RAG metrics for legacy default mode (without explicit usageMode)")
    if args.force_docs:
        print("Docs tool invocation is forced via direct tool endpoint.")
    async with httpx.AsyncClient(headers=headers) as client:
        await configure_docs_tool(client, args.base_url, args.agent_id, args.tool_id, usage_mode)
        uploaded_file_ids: list[int] = []
        for file_path in file_paths:
            file_id = await upload_document(client, args.base_url, args.agent_id, args.tool_id, file_path)
            uploaded_file_ids.append(file_id)
        await wait_file_processed(client, args.base_url, args.agent_id, args.tool_id, uploaded_file_ids)
        results = await collect_metrics_for_cases(client, args.base_url, args.agent_id, args.tool_id, cases, args.force_docs)
    print_summary(results, usage_mode)
    return results


async def run(args: argparse.Namespace) -> None:
    headers = {}
    bearer_token = normalize_bearer_token(args.bearer_token)
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
        print(f"Authorization header set: Bearer <hidden> (token length {len(bearer_token)})")
    elif args.compare or args.update_baseline:
        print("Warning: no bearer token provided. Authenticated endpoints may reject the request with 401.")

    cases_file = args.evaluation_cases_file
    cases = _load_evaluation_cases(cases_file) if args.use_evaluation_cases or cases_file.exists() else BASELINE_CASES
    file_paths = _collect_files_from_cases(cases)
    if not file_paths:
        file_paths = [Path(args.file)]
    print(f"Loaded {len(cases)} evaluation cases and {len(file_paths)} files to upload.")

    if args.all_usage_modes:
        results_by_mode: dict[str, list[dict[str, object]]] = {}
        for mode in USAGE_MODES:
            mode_results = await run_for_mode(args, mode, cases, file_paths, headers)
            results_by_mode[mode] = mode_results
            mode_baseline_file = _mode_baseline_path(args.baseline_file, mode)
            if args.update_baseline:
                mode_baseline_file.parent.mkdir(parents=True, exist_ok=True)
                save_baseline_file(mode_baseline_file, _with_usage_mode(mode_results, mode))
                print(f"Baseline updated at {mode_baseline_file}")
            if args.compare:
                if not mode_baseline_file.exists():
                    raise FileNotFoundError(f"Baseline file not found for mode {mode}: {mode_baseline_file}")
                baseline = load_baseline_file(mode_baseline_file)
                compare_metrics(baseline, {"cases": _with_usage_mode(mode_results, mode)})
        print_mode_comparison(results_by_mode)
        if args.export_file:
            export_payload = {
                "generated_at": datetime.now(UTC).isoformat(),
                "base_url": args.base_url,
                "all_usage_modes": True,
                "results_by_mode": {
                    mode: _with_usage_mode(results_by_mode[mode], mode) for mode in USAGE_MODES
                },
                "summary_by_mode": {
                    mode: aggregate_mode_summary(results_by_mode[mode]) for mode in USAGE_MODES
                },
            }
            save_results_file(args.export_file, export_payload)
            print(f"Results exported to {args.export_file}")
        return

    results = await run_for_mode(args, args.usage_mode, cases, file_paths, headers)
    results = _with_usage_mode(results, args.usage_mode)
    if args.update_baseline:
        args.baseline_file.parent.mkdir(parents=True, exist_ok=True)
        save_baseline_file(args.baseline_file, results)
        print(f"\nBaseline updated at {args.baseline_file}")
    if args.compare:
        if not args.baseline_file.exists():
            raise FileNotFoundError(f"Baseline file not found: {args.baseline_file}")
        baseline = load_baseline_file(args.baseline_file)
        compare_metrics(baseline, {"cases": results})
    if args.export_file:
        export_payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "base_url": args.base_url,
            "all_usage_modes": False,
            "usage_mode": args.usage_mode or "legacy-default",
            "summary": aggregate_mode_summary(results),
            "cases": results,
        }
        save_results_file(args.export_file, export_payload)
        print(f"Results exported to {args.export_file}")


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
