"""
HTTP client for the Tero backend API.

Handles: Docs tool configuration, file upload, thread management, and
SSE stream parsing to extract answer text and retrieved contexts.
"""

import asyncio
import json
import re
import time
from pathlib import Path

import httpx


CITATION_PATTERN = re.compile(r"\[[^\]]+\]\(chunk_\d+\)")
_TOOL_ID = "docs"
_EVAL_AGENT_NAME = "RAG Evaluation Agent"


async def resolve_eval_agent(base_url: str, bearer_token: str, evals_dir: Path) -> int:
    """
    Return the ID of the dedicated evaluation agent.
    On first run: creates the agent and persists its ID to evals/eval_agent.json.
    On subsequent runs: reads the persisted ID directly.
    """
    state_file = evals_dir / "eval_agent.json"
    if state_file.exists():
        agent_id = json.loads(state_file.read_text())["agent_id"]
        print(f"Using existing eval agent (id={agent_id}).")
        return agent_id

    url = f"{base_url.rstrip('/')}/api/agents"
    headers = {"Authorization": f"Bearer {bearer_token}"}
    async with httpx.AsyncClient(headers=headers) as client:
        # Create agent
        resp = await client.post(url)
        resp.raise_for_status()
        agent_id = resp.json()["id"]
        # Rename it so it's identifiable in the UI
        update_url = f"{url}/{agent_id}"
        await client.put(update_url, json={"name": _EVAL_AGENT_NAME})

    evals_dir.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"agent_id": agent_id}, indent=2))
    print(f"Created eval agent (id={agent_id}). Saved to {state_file}.")
    return agent_id


class TeroClient:
    def __init__(self, base_url: str, agent_id: int, bearer_token: str):
        self._base_url = base_url.rstrip("/")
        self._agent_id = agent_id
        self._headers = {"Authorization": f"Bearer {bearer_token}"}

    # ------------------------------------------------------------------
    # Docs tool setup
    # ------------------------------------------------------------------

    async def configure_docs_tool(self, config: dict | None = None) -> None:
        """Configure (or reconfigure) the Docs tool on the agent via POST.

        Uses POST so the tool is overwritten if already present — no separate
        existence check needed.  Pass ``skipDescriptions: true`` in *config*
        to skip LLM description generation during file processing.
        """
        url = f"{self._base_url}/api/agents/{self._agent_id}/tools"
        payload = {"toolId": _TOOL_ID, "config": config if config is not None else {}}
        async with httpx.AsyncClient(headers=self._headers) as client:
            resp = await client.post(url, json=payload)
            if resp.is_error:
                print(f"  Server returned {resp.status_code}: {resp.text}", flush=True)
            resp.raise_for_status()

    async def delete_docs_tool(self) -> None:
        """Delete the Docs tool config and its vectors from the agent.

        Uses DELETE which also runs teardown (clears PGVector index).
        Safe to call even if the tool is not configured — 404 is ignored.
        """
        url = f"{self._base_url}/api/agents/{self._agent_id}/tools/{_TOOL_ID}"
        async with httpx.AsyncClient(headers=self._headers) as client:
            resp = await client.delete(url)
            print(f"  DELETE {url} → {resp.status_code}", flush=True)
            if resp.status_code == 404:
                return  # not configured yet, nothing to delete
            if resp.is_error:
                print(f"  DELETE error body: {resp.text}", flush=True)
            resp.raise_for_status()

    async def upload_document(self, filename: str, content: bytes) -> int:
        """Upload a file to the Docs tool. Returns the file ID."""
        url = f"{self._base_url}/api/agents/{self._agent_id}/tools/{_TOOL_ID}/files"
        async with httpx.AsyncClient(headers=self._headers, timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(url, files={"file": (filename, content)})
            resp.raise_for_status()
            return resp.json()["id"]

    async def wait_files_processed(self, file_ids: list[int], timeout: float = 1800.0) -> None:
        """Block until the given file IDs have finished processing."""
        url = f"{self._base_url}/api/agents/{self._agent_id}/tools/{_TOOL_ID}/files"
        pending = set(file_ids)
        deadline = time.monotonic() + timeout
        async with httpx.AsyncClient(headers=self._headers) as client:
            while time.monotonic() < deadline:
                resp = await client.get(url)
                resp.raise_for_status()
                files = {f["id"]: f for f in resp.json()}
                # MISSING sentinel: file IDs absent from API response stay pending
                pending = {
                    fid for fid in pending
                    if files.get(fid, {}).get("status", "MISSING") not in ("PROCESSED", "ERROR")
                }
                if not pending:
                    return
                statuses = {fid: files.get(fid, {}).get("status", "MISSING") for fid in pending}
                print(f"  Still processing: {len(pending)} file(s) — {statuses}")
                await asyncio.sleep(5)
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for files: {pending}"
        )

    async def delete_all_files(self, timeout: float = 600.0) -> int:
        """Delete all files from the Docs tool on this agent. Returns count of files deleted.

        Lists all files and deletes each one. Files in PROCESSING state are skipped
        with a warning (they must finish processing before they can be deleted).
        """
        url = f"{self._base_url}/api/agents/{self._agent_id}/tools/{_TOOL_ID}/files"
        async with httpx.AsyncClient(headers=self._headers, timeout=timeout) as client:
            # List all files
            resp = await client.get(url)
            resp.raise_for_status()
            files = resp.json()
            if not files:
                print("  No files to delete on agent.")
                return 0

            # Delete each file
            deleted = 0
            skipped = 0
            for f in files:
                file_id = f["id"]
                status = f.get("status", "")
                if status == "PROCESSING":
                    print(f"  Skipping file {file_id} (still PROCESSING)")
                    skipped += 1
                    continue
                try:
                    del_resp = await client.delete(f"{url}/{file_id}")
                    if del_resp.status_code == 404:
                        print(f"  File {file_id} already deleted (404).")
                    else:
                        del_resp.raise_for_status()
                    deleted += 1
                except httpx.HTTPStatusError as exc:
                    print(f"  WARNING: could not delete file {file_id}: {exc}")
                    skipped += 1

        print(f"  Deleted {deleted} file(s), skipped {skipped}.")
        return deleted

    # ------------------------------------------------------------------
    # Agent model management
    # ------------------------------------------------------------------

    async def list_models(self) -> list[dict]:
        """Return the list of models available in this Tero instance."""
        url = f"{self._base_url}/api/models"
        async with httpx.AsyncClient(headers=self._headers) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    async def set_agent_model(self, model_id: str) -> None:
        """Switch the agent to the given model ID."""
        url = f"{self._base_url}/api/agents/{self._agent_id}"
        async with httpx.AsyncClient(headers=self._headers) as client:
            resp = await client.put(url, json={"modelId": model_id})
            resp.raise_for_status()

    # ------------------------------------------------------------------
    # Threads and questions
    # ------------------------------------------------------------------

    async def create_thread(self) -> int:
        url = f"{self._base_url}/api/threads"
        async with httpx.AsyncClient(headers=self._headers) as client:
            resp = await client.post(url, json={"agentId": self._agent_id})
            resp.raise_for_status()
            return resp.json()["id"]

    async def ask_question(self, thread_id: int, question: str) -> dict:
        """
        Send a question and parse the SSE stream.

        Returns a dict with:
          - answer_text: str
          - retrieved_contexts: list[str]
          - citations: list[str]
          - latency_ms: float
          - error: str  — "" | "sse_tool_error" | "empty_answer"
          - parse_failures: int — count of swallowed JSONDecodeError/AttributeError
        """
        url = f"{self._base_url}/api/threads/{thread_id}/messages"
        retrieved_contexts: list[str] = []
        raw_response = ""
        start = time.monotonic()

        tool_error = False
        parse_failures = 0

        async with httpx.AsyncClient(headers=self._headers, timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)) as client:
            async with client.stream("POST", url, data={"text": question, "origin": "USER"}) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[len("data: "):]
                    raw_response += chunk
                    if chunk.startswith("{"):
                        try:
                            event = json.loads(chunk)
                            if event.get("action") == "toolError":          # REQ-001
                                tool_error = True
                            elif (
                                event.get("action") == "executingTool"
                                and event.get("step") == "retrieved"
                                and isinstance(event.get("result"), list)
                            ):
                                retrieved_contexts.extend(event["result"])
                        except (json.JSONDecodeError, AttributeError):
                            parse_failures += 1                              # REQ-003
                    # else: plain-text line — accumulate in raw_response only, not a parse failure

        latency_ms = round((time.monotonic() - start) * 1000, 2)
        answer_text = _extract_answer_text(raw_response)
        citations = CITATION_PATTERN.findall(answer_text)

        error = ""
        if tool_error:
            error = "sse_tool_error"
        elif not answer_text:
            error = "empty_answer"

        return {
            "answer_text": answer_text,
            "retrieved_contexts": retrieved_contexts,
            "citations": citations,
            "latency_ms": latency_ms,
            "error": error,
            "parse_failures": parse_failures,
        }


def _extract_answer_text(raw: str) -> str:
    """
    Strip leading JSON event blobs from the SSE stream and return only the
    human-readable answer text at the end.
    """
    decoder = json.JSONDecoder()
    idx = 0
    length = len(raw)
    while idx < length and raw[idx].isspace():
        idx += 1
    while idx < length and raw[idx] == "{":
        try:
            _, end = decoder.raw_decode(raw, idx)
        except ValueError:
            break
        idx = end
        while idx < length and raw[idx].isspace():
            idx += 1
    answer = raw[idx:].strip()
    # REQ-012: Only strip trailing {\"answerMessageId\":\"...\"} blob at end of string
    answer = re.sub(r'\s*\{\s*"answerMessageId"\s*:\s*"[^"]*"\s*\}\s*$', "", answer)
    return answer.strip()
