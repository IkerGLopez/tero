# ODD Tasks — reranker-gpt6-luna-activation (W2)

**Feature**: Register GPT-6 Luna in the backend, make it the docs-tool reranker's default scoring model, and print the reranker ON/OFF state in eval runs.
**Route**: delegated (writer trigger: 2+ non-trivial files — alembic migration + `env.py` + `test_docs_rerank.py` + `runner.py` + `test_runner.py`; `.env`/`sample.env` mappings edited by the orchestrator).
**TDD**: strict — source: `sdd-init/tero` (`strict_tdd=true`); runners:
- backend: `wsl -d Ubuntu bash -lc "cd /mnt/d/dev/tero/src/backend && timeout 900 ../../.venv/bin/python -m pytest tests/test_docs_rerank.py -q"`
- eval: `wsl -d Ubuntu bash -lc "cd /mnt/d/dev/tero && timeout 900 .venv/bin/python -m pytest scripts/rag_eval/test_runner.py -q -k 'RerankerStateLine'"`
- NEVER the full backend suite (known hang at `tests/test_test_cases.py::test_run_test_suite_with_specific_test_cases`).

## Problem

`gpt-6-luna` exists only in the eval runner's judge pricing table (`scripts/rag_eval/runner.py`, commit `054b7cb`). The backend cannot resolve it: no `OPENAI_MODEL_ID_MAPPING` entry (provider lookup fails) and no `llm_model` row (cost-row lookup fails). Meanwhile `DOCS_TOOL_RERANK_MODEL` falls back to `INTERNAL_GENERATOR_MODEL` (gpt-4o-mini). Also, when the user runs an eval there is no console signal of whether the backend reranker is ON or OFF.

## Why

The user wants GPT-6 Luna as the reranker scorer default (cheaper, newer) and wants to see the reranker state per eval run without digging into backend logs or `.env`.

## Scope

- IN: alembic migration for `gpt-6-luna`; `OPENAI_MODEL_ID_MAPPING` in `sample.env` + local `.env`; `docs_tool_rerank_model` default in `env.py`; `test_docs_rerank.py` expectation; runner banner (`_reranker_state_line`) + tests.
- OUT: recreating the backend container; changes to `tool.py`'s defensive fallback; any other model registration. (The local `.env` already activates `DOCS_TOOL_RERANK` — user-owned; deployment runs through the normal compose flow.)

## Approach (prescribed — refine only with justification)

- Migration mirrors `add_claude_sonnet_5` (`ON CONFLICT DO NOTHING`), pricing from the eval judge table (0.0001/0.0005 per 1k), token limits confirmed by the user (1,050,000 context with ~922,000 max input / 128,000 output), `model_type='CHAT'`, `model_vendor='OPENAI'`.
- `Settings.set_defaults`: `self.docs_tool_rerank_model = self.docs_tool_rerank_model or "gpt-6-luna"` (single source of truth; `tool.py:577` fallback stays defensive-only).
- Runner: pure helper `_reranker_state_line()` reading `DOCS_TOOL_RERANK` / `_FETCH_K` / `_TOP_N` / `_MODEL` from `os.environ` (dotenv already loaded), printed in the live eval path and in offline CSV mode (with a "not applied" note).

## Tasks

- [x] W2.0 RED: test expectation flipped (isolated RED: `assert 'gpt-4o-mini' == 'gpt-6-luna'`); `TestRerankerStateLine` added (RED: `AttributeError: module 'runner' has no attribute '_reranker_state_line'`, 5 errors).
- [x] W2.1 Migration + `env.py` default; backend tests GREEN (40 passed).
- [x] W2.2 Runner banner + wiring; eval tests GREEN (5 passed, 257 deselected).
- [x] W2.3 Orchestrator: `.env` + `sample.env` mappings updated (count-guarded, 1 entry each).
- [x] W2.4 Verification + work-unit commits `bad88e4` (backend) and `6889335` (banner); RDD assess: risk medium, `review_due=false` (`under_budget`, 131 lines vs ~400) — no native review due.

## Acceptance

- W2.0–W2.4 checked with observed evidence; RED before implementation, GREEN after; `test_docs_rerank.py` fully green; new banner tests green.
- `gpt-6-luna` resolvable by the backend once the migration is applied and the container recreated.

## W3 — Runtime follow-up: reranker failed open on every call (found 2026-10-03)

**Problem**: the first post-activation eval ran with `[reranker] ON` (banner) and the backend had flag + mapping + row, but every scoring call returned `400: 'temperature' does not support 0.7 with this model` and `RerankRetriever` failed open — no effective rerank. Root cause: (a) `gpt-6-luna` is a fixed-temperature (reasoning-family) model but was registered as `CHAT`; (b) `_build_scoring_backend` sends `internal_generator_temperature` + `internal_generator_reasoning_effort` unconditionally instead of following the model type like `Agent.model_temperature` / `Agent.model_reasoning_effort`.
**Evidence**: app logs 2026-10-02 22:31/22:36 UTC — repeated `[docs] rerank scoring failed ... temperature`; live probe: `reasoning_effort=medium` OK, no-temperature OK, `temperature=1` OK, `0.7` rejected; DB row confirmed applied (`a3b4c5d6e7f8`, `CHAT`, 1050000/128000).
**Tasks**:
- [x] W3.0 RED: both new scorer tests failed against the old build (actual args `('scorer-model', 0.7, 'medium')`); GREEN after the fix — backend suite **42 passed**.
- [x] W3.1 Migration `b5c6d7e8f9a0` reclassifies `gpt-6-luna` as `REASONING` — commit `d9104de`.
- [x] W3.2 `_build_scoring_backend`: cost row fetched first; temperature only for CHAT, reasoning_effort only for REASONING — commit `9460541`.
- [x] W3.3 Verify + commits + RDD assess → native review **approved**; 8 advisory non-blocking findings; acknowledgement executed, authority burned.

## Progress / evidence

- **Discovery**: the local `.env` now sets `DOCS_TOOL_RERANK=true` (line 80, user-activated), so the fetaqa-era invariant "flag absent from `.env`" no longer holds. Two env-coupled tests were adapted so the suite is green under the activated flag: `test_disabled_default_fetches_k5_and_emits_the_same_set` pins the flag via `patch.object(env, ...)`, and the real-settings defaults test no longer asserts the flag off (its shipped default stays pinned by `test_rerank_setting_defaults_require_no_env_edit`).
- Writer evidence: baseline 2 failed/38 passed (env-coupled, pre-existing) → intended RED signatures → GREEN 40 passed with a `DOCS_TOOL_RERANK=false` process override; after the adaptations the exact command is 40 passed under the real activated `.env`.
- Backend (exact command): `wsl -d Ubuntu bash -lc "cd /mnt/d/dev/tero/src/backend && timeout 900 ../../.venv/bin/python -m pytest tests/test_docs_rerank.py -q"` → **40 passed, 2 warnings in 1.38s**.
- Eval (exact command): `wsl -d Ubuntu bash -lc "cd /mnt/d/dev/tero && timeout 900 .venv/bin/python -m pytest scripts/rag_eval/test_runner.py -q -k 'RerankerStateLine'"` → **5 passed, 257 deselected in 85.19s**.
- Migration `py_compile` → exit 0. Revision `a3b4c5d6e7f8` unique; `d9e8f7a6b5c4` was the head before it.
- Correction (user-provided specs): token limits set to 1,050,000/128,000 and the provisional note dropped before first apply — commit `0caba7c`.
- Commits: `bad88e4` `feat(db): register GPT-6 Luna and default the docs reranker to it`; `6889335` `feat(rag-eval): show the reranker state when an eval runs`.
- RDD assess (`--base-ref 054b7cb --committed-only --untracked-scope=exclude`): risk **medium** (`executable_change`: `scripts/rag_eval/runner.py`), 6 paths / 131 changed lines, `review_due=false` → `under_budget`. Boundary stays pending until the slice reaches the budget.
- Deployment (user-owned, pending): `docker compose up -d` applies the migration via the `migrations` service and recreates `app` for the updated `.env` (flag + mapping). The stack was down; no containers were started.

### W3 evidence (2026-10-03)

- Runtime root cause: app logs showed `400 'temperature' does not support 0.7 with this model` on every scoring call → fail-open; live probe confirmed luna accepts `reasoning_effort=medium`, no-temperature, and `temperature=1`, and rejects `0.7`.
- Fix commits: `d9104de` (reclassify REASONING) + `9460541` (type-aware scorer build). Backend tests: RED 2 failed/40 passed → GREEN **42 passed**; migration `py_compile` exit 0.
- RDD: assess **high_risk** (`hot_path` update) → START consented (user: granted) → 4 lenses admitted → **approved**; 8 advisory non-blocking findings (banner default drift R2/R4-C; default↔migration-row coupling R3-1/R4-B; downgrade asymmetry R4-A; unreachable else arms R3-2; env-dependent model assertion R3-3; reasoning-effort latency R4-D). Acknowledgement executed; authority burned (`review-acknowledged/v1`, revision `sha256:91b0d4c9…`).
- Deployed + verified (2026-10-03): `docker compose up -d --build` applied `b5c6d7e8f9a0` (row now `REASONING`) and rebuilt `app`; the next eval run (stratrag, agent 6, claude-sonnet-4-6, n=5) shows 13 scoring prompts invoked, 0 fail-open warnings, 0 tracebacks, and 9 persisted scoring usages (~26.8k prompt + 2.4k completion tokens, ~$0.0039). Rerank confirmed effective; metrics shifted vs the fail-open run (doc_coverage_5 0.6→0.7; context_recall 1.0→0.8; precision 0.67→0.63).

## Closure

- Cycle closed 2026-10-03. Archive report: `reranker-gpt6-luna-activation-cycle.md` (repo root) + Engram topics `odd/reranker-gpt6-luna-activation/tasks` / `.../cycle`. All 5 cycle commits pushed to `origin/feat/upgrade-v1.6` (`0fa6a54..9460541`).
