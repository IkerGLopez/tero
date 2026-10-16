# ODD Tasks — semicolon-converter-idempotency

**Feature**: Make `_csv_to_semicolon` never corrupt files; idempotent on already-converted semicolon CSVs.
**Route**: delegated (writer trigger: 2+ non-trivial files — `runner.py` + `test_runner.py`).
**TDD**: strict — source: sdd-init/tero (`strict_tdd=true`); runner: `wsl -d Ubuntu bash -lc "cd /mnt/d/dev/tero && timeout 600 .venv/bin/python -m pytest scripts/rag_eval/test_runner.py -q"`.

## Problem

On 2026-09-28 20:50 the end-of-run conversion pass rewrote `evals/experiments/ragbench/experiments/vibrant_dijkstra.csv` (already semicolon) into an unparseable single-column-quoted form — same class of damage previously observed on `boring_rivest.csv`. Root cause: `csv.Sniffer` fails to detect `;` on these ragged RAGAS CSVs, so Phase 1 `pd.read_csv(sep=',')` misparses a semicolon file and `to_csv(sep=';')` persists the (lossy) mangled result.

## Why

The pass protects real artifacts of past runs; today it destroys them. REQ-PIPELINE-HARDENING-009 (idempotency) is currently not satisfied for these file shapes.

## Scope

- IN: `scripts/rag_eval/runner.py` — `_csv_to_semicolon()`; tests in `scripts/rag_eval/test_runner.py`.
- OUT: salvage of already-damaged files; other runner behavior.

## Approach (prescribed — refine only with justification)

1. Deterministic already-converted detector BEFORE any rewrite: header heuristic — if the first line has more `;` than `,` (headers are simple, unquoted names), skip (bytes untouched).
2. Post-parse shape guard in Phase 1: after `pd.read_csv(sep=',')`, require a plausible results table (e.g. `question` column present and ≥ 10 columns); otherwise SKIP with a visible warning — never rewrite.
3. Keep the existing genuine-comma conversion path and Phase 2 fallback; convertible files must still convert.

## Tasks

- [x] C1. RED tests (`TestCsvToSemicolonHygiene`): pre-fix reproduced the exact incident (ragged fixture rewritten to single-column-quoted form); fixtures genuinely RED.
- [x] C2. Guards implemented: first-line heuristic (`;` > `,` → skip) + shape guard (`question` + ≥10 columns, else skip + visible warning). Genuine comma conversion and Phase-2 fallback preserved.
- [x] C3. Targeted: `-k TestCsvToSemicolonHygiene` → 4 passed; full `test_runner.py` → 236 passed / 16 failed (documented baseline). Real corpus (22 CSVs): every damaged-class file now skips byte-identical; pre-fix code rewrote `elegant_gates.csv` 613,761 → 624,638 B.
- [x] C4. Commit `730a5a5`; pushed to `origin/feat/upgrade-v1.6`.

## Acceptance

- C1–C4 checked with observed evidence; the exact damage scenario from 28/09 skips safely (bytes untouched); genuine comma CSVs still convert. ✅

## Progress / evidence

- Independent verification (delegated-verification gate, tier HIGH — see below): **VERIFIED**. 0 CRITICAL / 0 WARNING; 3 SUGGESTION (fyi, not actioned): (F1) skip-warning wording can misstate the reason when `question` exists but the table is narrow; (F2) contrived comma headers can cause a safe skip (no real corpus file affected); (F3) Phase-2 remains heuristic-unguarded in theory (no reachable counterexample found in bounded probing).
- Native review (RDD): **UNAVAILABLE on this runtime** — `review assess` refuses: "active runtime is not eligible for immutable receipt review; supported: claude-code, codex". Typed result preserved; not retried; switch is user-owned.
- Parent spot check: re-ran `-k TestCsvToSemicolonHygiene` → 4 passed (145.8s).
