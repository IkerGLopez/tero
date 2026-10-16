# ODD Tasks — stream-registration-leak-fix (W1)

**Feature**: Close W1 — a client disconnect between response handoff and stream start must not leave a stale `active_streaming_connections` entry (permanent 409).
**Route**: delegated (writer trigger: 2+ non-trivial files — `threads/api.py` + `test_threads.py`).
**TDD**: strict — source: sdd-init/tero (`strict_tdd=true`); runner: `wsl -d Ubuntu bash -lc "cd /mnt/d/dev/tero/src/backend && timeout 900 ../../.venv/bin/python -m pytest tests/test_threads.py -q"`. NEVER the full backend suite (known hang at `tests/test_test_cases.py::test_run_test_suite_with_specific_test_cases`).

## Problem (W1 — Judgment Day round 1, both judges, WARNING)

Registration now happens pre-handoff (`56f0774`, `add_message` → `active_streaming_connections[thread.id]`). Release paths: endpoint `finally` only when `stream_handed_off` is false, and `_agent_response`'s `finally` — reachable only after the generator's first `yield`. If the body iterator never starts (client disconnects in the handoff→first-iteration window; Starlette cancels the stream task) or teardown is interrupted, the entry leaks → every later `POST /messages` on that thread returns 409 `threadAlreadyStreaming` until process restart. Ledger: `.judgment-day/ledger-fix-silent-failures-and-thread-race-round1.md`.

## Why

It is the only known introduced failure path of the shipped fix; narrow window but permanent per-thread lockout.

## Scope

- IN: `src/backend/tero/threads/api.py`, `src/backend/tests/test_threads.py`.
- OUT: TTL/watchdog reapers, multi-worker `/stop`, frontend 409 copy, release-on-first-sight heuristics.

## Approach (prescribed — refine only with justification)

Response-lifecycle ownership: a `StreamingResponse` subclass (in `threads/api.py`) bound to `(thread_id, stop_event)` whose `async __call__` wraps `super().__call__()` in `try/finally` → `_release_stream_registration(thread_id, stop_event)`. `add_message` returns it. The existing identity guard makes double-release safe; the endpoint's pre-handoff `finally` stays for pre-handoff failures; `/stop` semantics unchanged.

## Tasks

- [x] W1.0 RED: `response_lifecycle` tests — pre-fix leak reproduced (`registration_present=True` with never-started body + immediate disconnect); mutation check proved the tests pin the leak, not the import.
- [x] W1.1 Implemented `ThreadStreamingResponse` (identity-guarded release in `__call__` finally); endpoint pre-handoff `finally` and `/stop` untouched; double release no-op.
- [x] W1.2 Targeted: full `tests/test_threads.py` → 40 passed / 14 failed / 1 skipped (environmental baseline; known flake passed this run); focused selection (new lifecycle + pinned stream tests) → 7 passed.
- [x] W1.3 Commit `88e8f6e`; pushed to `origin/feat/upgrade-v1.6`.

## Acceptance

- W1.0–W1.3 checked with observed evidence; new tests RED before fix, GREEN after; pinned tests stay green. ✅

## Progress / evidence

- Independent verification (delegated-verification gate, tier HIGH — see below): **VERIFIED**. 0 CRITICAL / 0 WARNING. Confirmed against real ordering (generator cleanup precedes lifecycle release; identity guard no-ops) and the production ASGI branch (uvicorn 0.38.0 advertises spec 2.3 → Starlette task-group + disconnect-listener path exercised). Documented residual: spec ≥ 2.4 servers (future) would release at response-call end while the generator finalizes later — not this deployment.
- Native review (RDD): **UNAVAILABLE on this runtime** — same refusal as the converter unit (claude-code/codex only). Typed result preserved; switch is user-owned.
