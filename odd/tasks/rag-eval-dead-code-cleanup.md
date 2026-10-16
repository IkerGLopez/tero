# ODD Tasks — rag-eval-dead-code-cleanup

**Feature**: Eliminar código sin uso verificado del harness de evaluación RAG (`scripts/rag_eval`) sin cambio de comportamiento, más una clase de tests que prueba una interfaz inexistente.
**Route**: delegated por work unit (writer trigger: 2+ archivos por unidad; evidencia: auditoría 2026-10-08 con 2 exploradores read-only + spot check del orquestador).
**TDD**: strict — source: sdd-init/tero (`strict_tdd=true`). **NO APLICABLE en esta feature**: son eliminaciones/redundancia sin comportamiento nuevo; no existe RED posible. El check es paridad exacta con el baseline + greps de cero referencias. Runner canónico: `wsl -d Ubuntu bash -lc "cd /mnt/d/dev/tero && timeout 600 .venv/bin/python -m pytest scripts/rag_eval/ -q --tb=no"`.

## Problem

La auditoría encontró: símbolos sin una sola referencia repo-wide, imports muertos, parámetros sin uso, funciones llamadas solo por sus tests, ramas inalcanzables y una clase de tests (`TestCliArgs`) que arma su propio parser con flags que no existen en producción. Los tests del harness no corren en CI (`build.yml` y `devbox tests` ejecutan pytest solo en `src/backend`), así que el bitrot no lo detecta ningún pipeline.

## Why

Menos superficie de mantenimiento y eliminar falsa confianza de tests que no ejercitan la interfaz real.

## Scope

**IN**: ítems T1–T3 (superficies exactas en cada tarea).

**OUT (revisados; se conservan a propósito)**:
- `prompts/*.md` — documentation-only por diseño (banner explícito en cada archivo).
- `relevant_chunk_position` / `latency_ms` — persistidos por diseño; `analysis` no los agrega intencionalmente.
- Seams solo-tests `create_thread_timeout` / `configure_docs_tool(config=None)` — se conservan (pendiente preferencia del usuario).
- Rama `Sniffer → ";"` (runner.py:1012-1018) y loop monoclave `loaded_datasets` — defensivas/generalidad; no se tocan.
- Flags ignorados en modo CSV (`--update-baseline`, `--compare`, `--seed`, `--max-docs`, `--concurrency`) — usados por modo live; cambiarlo sería cambio de comportamiento, fuera de alcance.
- Guards de ausencia (`corpus_state_*`, `resolve_eval_agent`, `offset`, `PROBE_DATASET`) — se conservan (regression guards baratos).
- `TestNanFaithfulnessMapping` — se conserva (cubre mapeo real de `_process_question_result`; solo sus docstrings mencionan un campo `faithfulness_valid` inexistente).
- `evals/pool_probe_cache/embeddings.jsonl` — cache local regenerable; sin acción.

**HALLAZGO SEPARADO (NO se toca en esta limpieza)**: `test_malformed_json_logs_warning_and_returns_empty_with_error` contradice FALLBACK 2 (runner.py:580-584: texto plano → `([stripped], False)`), mientras el E2E `test_malformed_json_csv_yields_error_rows` (test_runner.py:1322) espera `error == "json_parse_error"`. Posible regresión enmascarada por fixtures ausentes; requiere decisión de producto sobre la semántica de `_parse_json_column`.

## Constraints / environment

- Rama `refactor/rag-eval-dead-code-cleanup` (creada desde `feat/upgrade-v1.6` @ `9460541`). Árbol sucio preservado: `M scripts/rag_eval/prompts/ragbench-agent-12-system-prompt.md` + untracked (docs, ciclos SDD, `odd/`) NO forman parte de este cambio y NUNCA se stagean.
- **Baseline canónico (WSL venv)**: `10 failed, 563 passed` (160s). Los 10: 9 por fixtures ausentes (`evals/test_fixtures/*` no existe en este checkout) + 1 el test stale de malformed JSON (hallazgo separado). Log: `/tmp/tero_baseline.txt` (WSL).
- Baseline host Windows (sin WSL): 41 failed / 213 errors — causado por `langchain-community 0.4.2` (ragas 0.4.3 importa `chat_models.vertexai`, removido en 0.4.x). Workaround no invasivo: `python C:\Users\ikerg\AppData\Local\Temp\opencode\rag_eval_pytest.py scripts/rag_eval/ -q --tb=no` (stub del módulo removido) → 10F/563P idéntico al canónico. Solo fallback.
- Aceptación: 0 fallos/errores nuevos vs baseline; la cantidad de tests baja EXACTAMENTE por lo removido intencionalmente; cada símbolo eliminado re-verificado con grep repo-wide por el writer (y spot check del orquestador).

## Tasks

- [x] **T1 — Higiene + símbolos sin referencias** (sin tests que los referencien; esperado: 10F/563P):
  - `TeroClient.list_models` (tero_client.py:193) — eliminar (única aparición repo-wide).
  - `JudgeCostTracker.unwrap` (runner.py:1514) — eliminar; actualizar comentario de :1463 que lo menciona.
  - `import sys` sin uso (sanity_checks.py:12).
  - `import tempfile` nivel módulo (test_runner.py:10) — usos re-importan local.
  - `key = q[:20]` (test_runner.py:1169) — nunca leída.
  - `call` en import (tests/test_tero_client.py:12) — sin uso.
  - `elif df is None: pass` (analysis.py:138-140) — rama inalcanzable.
  - Tuplas `except` redundantes + imports locales (runner.py:644-648, :784, :811-815, :841) — SOLO donde el handler genérico sea idéntico; si no, dejar.
  - Re-asignaciones redundantes post `_error_row` (runner.py:787-793) — SOLO si `_error_row` setea idénticos valores.
  - `Sequence` en anotación sin import (runner.py:~1998) — import mínimo.
  - ✅ Evidencia T1 (2026-10-08): commit `8cf647e` (6 archivos, +5/−44); suite WSL 10F/563P antes y después (mismos 10 nombres). Ítem "re-asignaciones post `_error_row`": SALTADO — `latency_ms` reasignado es portante, no redundante (premise falso, verificado).
- [x] **T2 — Símbolos solo-tests + call sites** (esperado real: 10F/554P; se remueven 9 items):
  - `agent_id` sin uso en `_upload_corpus` (smoke_test.py:88 + call site :131 + 3 tests en test_smoke_test.py).
  - `gate_depth` sin uso en `_probe_gate_verdict` (runner.py:2009 + ~8 call sites en tests).
  - `load_all` (rag_datasets.py:464) + `TestLoadAll` (test_rag_datasets.py:1680,1722-1767) = 2 tests.
  - `cosine_similarity` (pool_probe.py:273) + `TestCosineSimilarity` (test_pool_probe.py:259-271) = 3 tests.
  - NUEVOS (residuo de `unwrap`, verificado por orquestador): `self._provider = provider` (runner.py:1389; su único lector era `unwrap`) y `self._original_create = None` (runner.py:1401 y :1427; write-only tras quitar `unwrap`; la rama retorna antes de `_wrap_*`). Si algún test leyera `_original_create` de una instancia double-wrapped, saltar y reportar.
  - ✅ Evidencia T2 (2026-10-08): commit `83b3255` (8 archivos, +12/−117); suite WSL 10F/563P → 10F/554P (mismos 10 nombres; removidos 9 items reales según collect-only: `TestCosineSimilarity`=7 contando parametrize, `TestLoadAll`=2). Los 6 ítems aplicados; `gate_depth` tenía 7 call sites, no ~8.
- [x] **T3 — Test de interfaz falsa** (esperado: 10F/548P; se remueven 6 items):
  - `TestCliArgs` (test_runner.py:1081-1137) — arma su propio parser con `--only`/`--n` inexistentes y `seed` default 42 ≠ 14 real; nunca llama a `runner.build_parser()`.
  - ✅ Evidencia T3 (2026-10-08): commit `13abffb` (−62); WSL 10F/554P → 10F/548P (mismos 10 nombres). 3 chequeos previos verificados: parser propio sin `build_parser`, sin helpers huérfanos, sin referencias repo-wide.

## Acceptance

- Suite post-cambio = mismos 10 fallos, 0 nuevos; passed = 563 → 558 (T2) → 552 (T3).
- Cada eliminación con grep repo-wide: 1 hit (definición) o solo los call sites esperados.
- Sin cambios fuera de las superficies permitidas; sin staging de archivos ajenos.

## Progress / evidence

- Auditoría completa + baseline canónico + baseline host: hechos (2026-10-08).
- T1: commit `8cf647e`; suite WSL 10F/563P antes/después. Assess: medium / under_budget / review_due=false (base 9460541).
- T2: commit `83b3255`; WSL 10F/554P; assess medium / under_budget / review_due=false (178 líneas acumuladas, base 9460541).
- T3: commit `13abffb` (−62); WSL 10F/548P; assess medium / under_budget / review_due=false (240 líneas acumuladas, base 9460541).

## Next steps

- T1–T3 completados y commiteados (`8cf647e`, `83b3255`, `13abffb`) con assess medium/under_budget por commit (review_due=false).
- Spot check final OK (10F/548P); **pusheado a `origin/feat/upgrade-v1.6`** (`9460541..13abffb`, autorizado por el usuario); rama temporal `refactor/rag-eval-dead-code-cleanup` eliminada (mergeada).
