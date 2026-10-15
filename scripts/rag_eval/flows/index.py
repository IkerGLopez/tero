"""Index flow: load a dataset corpus and upload it into a Tero agent."""

import config  # Bootstrap: repo .env + sys.path

import argparse
import asyncio
import os
import sys
import time

import rag_datasets as ds_module
from judge import cost_report


async def do_index(args: argparse.Namespace) -> None:
    """Index a dataset's full corpus into an agent with skipDescriptions=true.

    1. Loads corpus from dataset (no questions)
    2. Configures agent with skipDescriptions=true
    3. Uploads all documents
    4. Reports embedding token count and estimated cost
    """
    import tiktoken
    from tero_client import TeroClient

    t0 = time.monotonic()

    agent_id = args.agent_id if args.agent_id is not None else int(os.environ.get("RAG_EVAL_AGENT_ID", "0"))
    if agent_id <= 0:
        print("ERROR: --agent-id is required for index (or set RAG_EVAL_AGENT_ID env var).", file=sys.stderr)
        sys.exit(1)

    bearer_token = (args.bearer_token or os.environ.get("BEARER_TOKEN", "")).strip()
    if not bearer_token:
        print("ERROR: --bearer-token is required (or set BEARER_TOKEN env var).", file=sys.stderr)
        sys.exit(1)

    tero = TeroClient(args.base_url, agent_id, bearer_token)

    # 1. Load full corpus (n=0 → no questions, full corpus)
    print(f"Loading corpus for dataset '{args.dataset}'...")
    _, corpus = ds_module.load_one(args.dataset, n=0)

    if args.max_docs and args.max_docs < len(corpus):
        corpus = corpus[:args.max_docs]
        print(f"Truncated to {args.max_docs} documents.")

    print(f"Corpus: {len(corpus)} documents loaded.")

    # 2. Estimate embedding tokens via cl100k_base (used by text-embedding-3-small/ada-002)
    enc = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(enc.encode(doc)) for doc in corpus)

    # 3. Clean teardown: list existing files, wait for in-flight processing,
    #    delete all files, then delete tool. Skip wait if no existing files (first run).
    print(f"\nResetting agent {agent_id} docs tool...")
    existing_ids = await tero.list_file_ids()
    if existing_ids:
        print(f"  Waiting for {len(existing_ids)} existing file(s) to settle...")
        await tero.wait_files_processed(existing_ids, timeout=120.0)
    await tero.delete_all_files()
    await tero.delete_docs_tool()

    print(f"Configuring agent {agent_id} with skipDescriptions=true...")
    config = {"skipDescriptions": True}
    await tero.configure_docs_tool(config)

    # 4. Upload all documents
    print(f"Uploading {len(corpus)} documents...")
    file_ids: list[int] = []
    for idx, doc in enumerate(corpus):
        filename = f"doc_{idx:06d}.txt"
        fid = await tero.upload_document(filename, doc.encode("utf-8"))
        file_ids.append(fid)
        if (idx + 1) % 100 == 0:
            print(f"  Uploaded {idx + 1}/{len(corpus)} documents...")
        await asyncio.sleep(0.05)  # throttle to avoid saturating JWT auth calls to Keycloak

    print(f"All {len(file_ids)} documents uploaded. Waiting for processing...")
    await tero.wait_files_processed(file_ids)
    print("All documents processed.")

    # 5. Retry failed files — transient errors (LangChain time-sync) resolve on re-upload.
    #     Re-uploads create new file IDs; we track them and wait on the new ones.
    seen_errors: set[int] = set()
    replaced: set[int] = set()  # old ERROR IDs that were successfully re-uploaded
    for attempt in range(1, 4):
        error_files = await tero.list_error_files()
        fresh = [(fid, name) for fid, name in error_files if fid not in seen_errors]
        if not fresh:
            break
        print(f"\nRetry attempt {attempt}: {len(fresh)} file(s) in ERROR. Re-uploading...")
        retry_ids: list[int] = []
        for fid, name in fresh:
            # Parse document index from filename (doc_NNNNNN.txt)
            try:
                idx = int(name.removeprefix("doc_").removesuffix(".txt"))
            except ValueError:
                print(f"  WARNING: skipping {name} — unexpected filename format")
                seen_errors.add(fid)
                continue
            try:
                new_fid = await tero.upload_document(name, corpus[idx].encode("utf-8"))
                retry_ids.append(new_fid)
                replaced.add(fid)
                seen_errors.add(fid)
            except Exception as exc:
                print(f"  WARNING: re-upload failed for {name}: {exc}")
            await asyncio.sleep(2)
        if retry_ids:
            await tero.wait_files_processed(retry_ids, timeout=120.0)
        remaining = [(fid, _) for fid, _ in await tero.list_error_files() if fid not in seen_errors]
        print(f"  After retry: {len(remaining)} new ERROR(s).")
    else:
        still = [(fid, _) for fid, _ in await tero.list_error_files() if fid not in seen_errors]
        if still:
            print(f"WARNING: {len(still)} file(s) still in ERROR after 3 retry attempts.")

    # Clean up orphaned ERROR records that were successfully replaced
    if replaced:
        deleted = await tero.delete_files(list(replaced))
        print(f"Cleaned up {deleted} orphaned ERROR record(s).")

    # 6. Cost report and wall-clock time
    elapsed = time.monotonic() - t0
    cost_report(total_tokens)
    print(f"Index wall-clock time: {elapsed:.1f} seconds")
