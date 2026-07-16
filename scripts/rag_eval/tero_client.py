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

    async def list_file_ids(self) -> list[int]:
        """Return IDs of all files currently indexed for this agent.

        Returns [] if the docs tool does not exist (404 — first run).
        Raises on any other non-2xx response.
        """
        url = f"{self._base_url}/api/agents/{self._agent_id}/tools/{_TOOL_ID}/files"
        async with httpx.AsyncClient(headers=self._headers) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            return [f["id"] for f in resp.json()]

    async def list_error_files(self) -> list[tuple[int, str]]:
        """Return (id, name) tuples for files in ERROR state for this agent."""
        url = f"{self._base_url}/api/agents/{self._agent_id}/tools/{_TOOL_ID}/files"
        async with httpx.AsyncClient(headers=self._headers) as client:
            resp = await client.get(url)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            return [(f["id"], f["name"]) for f in resp.json() if f.get("status") == "ERROR"]

    async def delete_files(self, file_ids: list[int]) -> int:
        """Delete specific files by ID. Returns count deleted."""
        url = f"{self._base_url}/api/agents/{self._agent_id}/tools/{_TOOL_ID}/files"
        deleted = 0
        async with httpx.AsyncClient(headers=self._headers) as client:
            for file_id in file_ids:
                resp = await client.delete(f"{url}/{file_id}")
                if resp.is_success or resp.status_code == 404:
                    deleted += 1
                else:
                    print(f"  WARNING: failed to delete file {file_id}: {resp.status_code}")
        return deleted

    async def delete_all_files(self) -> int:
        """Delete all files from the Docs tool on this agent. Returns count of files deleted.

        Lists files via list_file_ids(), then deletes each one with up to 4 attempts
        (1 initial + 3 retries) using exponential backoff (2s, 4s, 8s).
        Raises RuntimeError if all attempts for any file are exhausted.
        """
        _BACKOFF_DELAYS = (2, 4, 8)
        _MAX_ATTEMPTS = len(_BACKOFF_DELAYS) + 1  # 4 total

        url = f"{self._base_url}/api/agents/{self._agent_id}/tools/{_TOOL_ID}/files"
        file_ids = await self.list_file_ids()
        if not file_ids:
            print("  No files to delete on agent.")
            return 0

        async with httpx.AsyncClient(headers=self._headers) as client:
            deleted = 0
            for file_id in file_ids:
                for attempt in range(_MAX_ATTEMPTS):
                    if attempt > 0:
                        await asyncio.sleep(_BACKOFF_DELAYS[attempt - 1])
                    del_resp = await client.delete(f"{url}/{file_id}")
                    if del_resp.status_code == 404:
                        print(f"  File {file_id} already deleted (404).")
                        deleted += 1
                        break
                    elif del_resp.is_success:
                        deleted += 1
                        break
                    elif attempt < _MAX_ATTEMPTS - 1:
                        print(
                            f"  WARNING: DELETE file {file_id} returned "
                            f"{del_resp.status_code}, retrying "
                            f"({attempt + 2}/{_MAX_ATTEMPTS})..."
                        )
                    else:
                        raise RuntimeError(
                            f"DELETE file {file_id} failed after {_MAX_ATTEMPTS} attempts: "
                            f"{del_resp.status_code} — {del_resp.text}"
                        )

        print(f"  Deleted {deleted} file(s).")
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
        """
        url = f"{self._base_url}/api/threads/{thread_id}/messages"
        retrieved_contexts: list[str] = []
        answer_chunks: list[str] = []
        start = time.monotonic()

        async with httpx.AsyncClient(headers=self._headers, timeout=httpx.Timeout(connect=10.0, read=120.0, write=10.0, pool=10.0)) as client:
            async with client.stream("POST", url, data={"text": question, "origin": "USER"}) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[len("data: "):]

                    # Try to parse as JSON — if it fails, it's plain-text answer content.
                    try:
                        event = json.loads(chunk)
                    except json.JSONDecodeError:
                        # Plain text — LLM answer content
                        answer_chunks.append(chunk)
                        continue

                    # Non-dict JSON values (strings, numbers, arrays, null) are
                    # treated as answer content (e.g. the LLM outputs a JSON scalar).
                    if not isinstance(event, dict):
                        answer_chunks.append(chunk)
                        continue

                    # REQ-012: Skip trailing metadata blob emitted at end of stream.
                    if "answerMessageId" in event:
                        continue

                    # Extract retrieved contexts from tool execution events.
                    if (
                        event.get("action") == "executingTool"
                        and event.get("step") == "retrieved"
                    ):
                        if isinstance(event.get("result"), list):
                            retrieved_contexts.extend(event["result"])
                        else:
                            # Schema drift — result is not a list, contexts lost.
                            print(f"  WARNING: executingTool/retrieved event has "
                                  f"non-list result ({type(event.get('result')).__name__}) — "
                                  f"contexts not extracted.",
                                  flush=True)
                        # Always skip tool execution events — they are never
                        # answer text, regardless of result shape.
                        continue

                    # All other JSON dicts (status updates, thinking events,
                    # unknown future event types) — preserve in answer text so
                    # _extract_answer_text can handle them as defense-in-depth.
                    answer_chunks.append(chunk)

        latency_ms = round((time.monotonic() - start) * 1000, 2)
        answer_text = _extract_answer_text("".join(answer_chunks))
        citations = CITATION_PATTERN.findall(answer_text)

        return {
            "answer_text": answer_text,
            "retrieved_contexts": retrieved_contexts,
            "citations": citations,
            "latency_ms": latency_ms,
        }


def _extract_answer_text(raw: str) -> str:
    """
    Strip leading JSON event blobs from the SSE stream and return only the
    human-readable answer text at the end.

    Also strips the trailing metadata blob emitted by the Tero backend
    (``{"answerMessageId": N, "files": [...], ...}``) regardless of its
    exact shape — the function scans backward from the end of the string,
    tries to parse each ``{...}`` candidate as JSON, and removes it when
    it contains an ``"answerMessageId"`` key.
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
            # Malformed JSON blob — skip forward to the next potential event boundary
            next_boundary = raw.find("{", idx + 1)
            if next_boundary == -1:
                break  # No more JSON events — plain text starts at current position
            idx = next_boundary  # land on the next {
            continue
        idx = end
        while idx < length and raw[idx].isspace():
            idx += 1
    answer = raw[idx:].strip()

    # REQ-012: Strip trailing metadata blob.
    # The backend emits e.g. {"answerMessageId":1375,"files":[],"minutesSaved":0,"stopped":false}
    # Single-pass: parse the rightmost {…} suffix; strip only when it is a valid
    # JSON object anchored at the end of the string and contains an answerMessageId key.
    brace_pos = answer.rfind("{")
    if brace_pos != -1:
        suffix = answer[brace_pos:].strip()
        try:
            blob = json.loads(suffix)
        except (json.JSONDecodeError, ValueError):
            pass  # Unparseable trailing text — leave answer intact
        else:
            if isinstance(blob, dict) and "answerMessageId" in blob:
                answer = answer[:brace_pos].rstrip()

    return answer.strip()
