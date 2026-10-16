# ODD Tasks — rag-eval-module-split

**Feature**: Dividir `scripts/rag_eval/runner.py` (2.153 líneas, 47 defs + `JudgeCostTracker`) en módulos por responsabilidad dentro de `scripts/rag_eval/`, siguiendo patrones adaptados del backend (`config` central, verticales por flow, adaptadores de borde, CLI delgado), **sin cambio de comportamiento**.
**Route**: delegated por work unit (writer trigger: 2+ archivos por unidad; context trigger: el writer lee el código que mueve; el orquestador no re-lee la evidencia mapeada).
**TDD**: strict — source: sdd-init/tero (`strict_tdd=true`). **NO APLICABLE en esta feature**: movimiento estructural sin comportamiento nuevo; no existe RED posible. Verificación = paridad exacta de suite (mismos 10 fallos, mismo `passed`) + checks de import/collect/grep. Runner canónico: `wsl -d Ubuntu bash -lc "cd /mnt/d/dev/tero && timeout 600 .venv/bin/python -m pytest scripts/rag_eval/ -q --tb=no"`.

## Problem

`runner.py` es un monolito de 2.153 líneas con ~47 funciones que mezcla: bootstrap de paths, CLI, flujo index, flujo eval live, flujo CSV offline, métricas determinísticas, métricas RAGAS, cliente juez + costos, I/O CSV, baselines/comparabilidad y pool-probe. La auditoría (2 exploradores read-only + spot check, 2026-10-12) confirma acoplamientos concretos:

- `do_eval` es hub: llama a casi todos los clusters (runner.py:1778–1939).
- `_compute_metrics_from_sample` (runner.py:626) lo comparten ambos flujos (live :862, CSV :1207) y los tests lo monkeypatchean directo.
- Constantes de path ancladas en `__file__` (runner.py:51, 63–70) y parcheadas por tests vía atributos de `runner`.
- Un test inspecciona el **código fuente** de `_run_csv_mode` para exigir lazy import de TeroClient (REQ-OFFLINE-009) — landmine para cualquier movimiento.

Contexto de referencia: `docs/rag-eval-flow.md` (flujo real index → eval live). CI **no** corre esta suite (`build.yml` solo `src/backend`), así que la red de seguridad es la suite WSL local.

## Why

Menor superficie de mantenimiento y review; alinear el harness con la arquitectura real del backend (ver "Patrones del backend aplicables" abajo); habilitar evolución futura (tests nuevos, extensión de métricas) sin tocar un archivo de 2k líneas.

## Patrones del backend aplicables (catálogo verificado 2026-10-12)

Transfieren (adaptados, sin FastAPI/DB):
1. **Vertical por concern**: `users/` = api+domain+repos → aquí `flows/` (aplicación), `metrics/` (dominio de medición), `judge.py`/`tero_client.py` (adaptadores de borde).
2. **Config singleton tipado** (`core/env.py:18,164` `Settings` construido al import) → aquí `config.py`: única ancla `__file__`, carga de `.env`, paths y constantes compartidas.
3. **Boundary delgado / lógica adentro** (`users/api.py:20-23` vs `threads/engine.py`) → aquí `runner.py` = CLI parse+dispatch; los flows contienen la orquestación.
4. **Módulo de logging estructurado** — NO se adopta en esta feature (cambiaría el stdout del CLI; fuera de alcance).

NO transfieren (cargo-cult): routers/`Depends`, SQLModel/repos, `CamelCaseModel`, MCP/heartbeat, testcontainers.

## Scope

**IN**: T1–T7 (abajo). Únicamente `scripts/rag_eval/runner.py` se divide; los módulos ya existentes (`tero_client`, `rag_datasets`, `analysis`, `sanity_checks`, `retrieval_matching`, `smoke_test`, `pool_probe`, `export_datasets`) y `prompts/` no se tocan.

**OUT (revisados; decisión explícita)**:
- **Renames de símbolos** (`_doc_recall_5` → `doc_recall_5`, etc.): el movimiento preserva nombres exactos para un diff estructural limpio (`git diff --color-moved`). De-underscore como follow-up opcional.
- **Consolidación de tests** a un solo `tests/` + `conftest.py` central: churn ortogonal; feature separada si se quiere.
- **Conversión a logging** / cambio de formato de salida: sería cambio de comportamiento.
- **Arreglar los 10 fallos conocidos** (9 por fixtures ausentes + 1 test stale de malformed JSON ya escalado como hallazgo separado): fuera de alcance; deben permanecer idénticos.
- **Staging de archivos ajenos**: `M scripts/rag_eval/prompts/ragbench-agent-12-system-prompt.md` + untracked (`docs/rag-eval-flow.md`, `odd/`, ciclos SDD, `.engram/`, `.judgment-day/`, etc.) NUNCA se stagean.
- `docs/rag-eval-flow.md` está untracked: T7 lo actualiza en disco; **staging queda para decisión del usuario**.

## Estructura objetivo

```
scripts/rag_eval/
  runner.py            # CLI delgado: build_parser + main + dispatch (nada más)
  config.py            # paths (única ancla __file__), dotenv, EVAL_TOP_K, banner reranker
  judge.py             # pricing + modelos + JudgeCostTracker + cost_report + _build_judge_client
  metrics/
    __init__.py
    deterministic.py   # @5 determinísticas (doc/sentence/coverage/table/cell) + helpers
    gold.py            # anotación gold, alineación, población, relevant_chunk_index
    ragas.py           # _build_metrics + _pair_citations_with_contexts
    sample.py          # _compute_metrics_from_sample + _process_question_result + _sanitize_error
  reporting/
    __init__.py
    csv_io.py          # parse/validate CSV, separador ';', flags de tabla
    baseline.py        # comparabilidad, load/save baseline
  flows/
    __init__.py
    index.py           # do_index
    eval_live.py       # do_eval
    eval_csv.py        # _run_csv_mode
    probe.py           # gate consts + helpers + do_pool_probe
  [sin cambios] tero_client.py · rag_datasets.py · analysis.py · sanity_checks.py ·
                retrieval_matching.py · smoke_test.py · pool_probe.py · export_datasets.py · prompts/ · evals/
```

Reglas de la división (todas las tareas):
- **Sin shims de re-export**: los símbolos movidos dejan de existir en `runner.py` (excepto bindings necesarios para patch sites aún vigentes en la tarea). Un import viejo debe fallar ruidoso, no enmascarar un patch inefectivo.
- **Sin ciclos**: los módulos movidos no importan `runner`. Los imports flat existentes (`from tero_client import ...` dentro de funciones, `import pool_probe`, `import sanity_checks`) permanecen lazy donde ya lo eran.
- El bootstrap de `sys.path` vive en `config.py` (insert guardado, sin duplicados); el entry `python scripts/rag_eval/runner.py <cmd>` sigue funcionando idéntico (README/docs).
- Nombres de símbolos idénticos; docstrings preservadas (ajustar solo referencias a `runner.py` si mienten).

## Constraints / environment

- Rama `refactor/rag-eval-module-split` (creada desde `feat/upgrade-v1.6` @ `13abffb`). Árbol sucio preservado (ver OUT).
- **Baseline canónico (WSL venv)**: `10 failed, 548 passed` (149.57s), log `/tmp/tero_split_baseline.txt` — capturado 2026-10-12, ANTES de cualquier cambio. Los 10 nombres exactos:
  1. `test_runner.py::TestParseJsonColumn::test_malformed_json_logs_warning_and_returns_empty_with_error`
  2. `test_runner.py::TestCsvModeE2E::test_full_csv_pipeline_writes_output_and_computes_stats`
  3. `test_runner.py::TestCsvModeE2E::test_minimal_csv_produces_results`
  4. `test_runner.py::TestCsvModeE2E::test_multi_model_csv_groups_and_compares`
  5. `test_runner.py::TestCsvModeE2E::test_malformed_json_csv_yields_error_rows`
  6. `test_runner.py::TestCsvModeE2E::test_nan_question_skipped_with_error`
  7. `test_runner.py::TestEvalJudgeCostWiring::test_run_csv_mode_creates_judge_cost_tracker`
  8. `test_runner.py::TestSanityWiring::test_run_csv_mode_calls_run_sanity_checks`
  9. `test_runner.py::TestSanityWiring::test_run_csv_mode_passes_df_and_dataset_to_print_summary`
  10. `test_runner.py::TestSanityWiring::test_csv_rewrite_on_parametric_suspect_column`
- Test host Windows directo: NO fiable (`langchain-community 0.4.2` roto para ragas 0.4.3). Fallback host solo si se necesita: stub en `C:\Users\ikerg\AppData\Local\Temp\opencode\rag_eval_pytest.py`.
- **Landmines confirmados** (cada tarea relevante los lista):
  - Tests con `patch.object(runner, X)` (~55 sitios `import runner` en test_runner.py): al mover X y su consumidor, el patch target debe migrar al módulo consumidor o el patch queda mudo.
  - `test_runner.py:1175` inspecciona source de `_run_csv_mode` (lazy TeroClient) → migrar el `inspect` al módulo nuevo en T6.
  - `tests/test_tero_client.py:528` (`import runner as r`) usa internals de `do_index` → migrar en T6.
  - 4 sites parchean `runner.EVALS_DIR/EXPERIMENTS_DIR/BASELINE_DIR` (test_runner.py:1138, 2021, 3198, 3429) → migran cuando el lector se mueve (T5/T6).
  - `test_runner.py:1904-1905` parchea `runner.JUDGE_COST_PER_1K_*` que hoy solo existe como locals en `_run_csv_mode` — probablemente parte del fallo #7 conocido; NO arreglar, mantener comportamiento.
- Aceptación por tarea: `10 failed, 548 passed` con los MISMOS 10 nombres; `--collect-only` = 558 tests; entry smoke OK; grep sin defs residuales. Sin tests nuevos ni removidos.

## Tasks

- [x] **T1 — `config.py`**: mover `_REPO_ROOT` (:51) + `load_dotenv` (:52), `SCRIPT_DIR/EVALS_DIR/BASELINE_DIR/EXPERIMENTS_DIR/DEFAULT_PROBE_CACHE_DIR` (:63–70), `_RERANKER_TRUTHY` (:93), `_reranker_state_line` (:96–114), `EVAL_TOP_K` (:875). `runner.py` los importa (bindings intactos para patch sites actuales). `sys.path.insert` queda en `runner.py`. Verificación: paridad + import smoke + grep. Churn de tests esperado: 0 (los patch sites siguen resolviendo por binding en runner). ✅ Evidencia T1 (2026-10-12): commit `8c7545a` (2 archivos, +65/−51); suite WSL 10F/548P (mismos 10 nombres), collect 558, import smoke OK; churn de tests 0. Assess: medium / under_budget / review_due=false (base 9460541, 356 líneas acumuladas).
- [x] **T2 — `metrics/deterministic.py` + `metrics/gold.py`**: funciones puras. Candidatos: `_normalize_content` (:278), `_extract_cells` (:283), `_doc_recall_5` (:365), `_sentence_recall_5` (:384), `_doc_coverage_5` (:408), `_RETRIEVAL_METRIC_DEFAULTS` (:429), `_is_gold_linked` (:438), `_population_counts` (:450), `_print_alignment_report` (:479), `_table_recall_5` (:508), `_is_degenerate_gold_value` (:524), `_cell_recall_5` (:531); `_find_relevant_chunk_index` (:199), `_annotate_gold` (:308), `_in_prefix_gold_sentences` (:350). Asignación exacta la decide el writer según consumidores (sin ciclos; escritura del porqué en el reporte). Churn: imports de tests de privados. ✅ Evidencia T2 (2026-10-12): commit `f180d92` (5 archivos, +389/−339); paridad 10F/548P (mismos 10) + collect 558 + smokes A/B OK; churn de tests 0; AST byte-equality de los 16 símbolos movidos. Split sin ciclos: deterministic.py (métricas) ↔ gold.py (anotación; importa pandas). Bootstrap `sys.path` movido a config.py.
- [x] **T3 — `metrics/ragas.py` + `metrics/sample.py`**: `_build_metrics` (:121), `_pair_citations_with_contexts` (:161); `_sanitize_error` (:183), `_dedupe_first_occurrence` (:260), `_compute_metrics_from_sample` (:626), `_process_question_result` (:792). `_error_row` sigue siendo inner. Churn: ~15 import sites + patches directos (test_runner.py:1133, 1196). ✅ Evidencia T3 (2026-10-12): commit `b197e7e` (4 archivos, +384/−351); paridad 10F/548P (mismos 10) + collect 558 + smokes A/B OK; AST byte-identical de los 6 cuerpos movidos; auditoría de patch sites: 25 sites conservados (consumidor aún en runner), 1 retarget a `metrics.sample` (test_runner.py:~3602, path do_eval; verificado con sentinel de intercepción).
- [ ] **T4 — `judge.py`**: `JUDGE_PRICING_TABLE` (:1308), `BEDROCK_JUDGE_MODELS` (:1324), `NO_SAMPLING_PARAMS_MODELS` (:1335), `_resolve_judge_pricing` (:1339), `JudgeCostTracker` (:1361–1528), `cost_report` (:1529), `_build_judge_client` (:1548–1647, con lazy `from anthropic import ...` intacto). Churn alto: patches de `JudgeCostTracker` (test_runner.py:1505–2600 por tramos).
- [ ] **T5 — `reporting/csv_io.py` + `reporting/baseline.py`**: `_parse_json_column` (:555), `_validate_csv_columns` (:589), `_is_missing_value` (:601), `_move_model_id_last` (:614), `_RESULTS_TABLE_MIN_COLUMNS` (:946), `_is_already_semicolon_delimited` (:949), `_is_results_table` (:961), `_csv_to_semicolon` (:971); `_comparability_metadata` (:878), `_baseline_deviations` (:888), `_print_comparability_report` (:904), `_load_baseline` (:917), `_save_baseline` (:924). Patch sites `runner.EXPERIMENTS_DIR/BASELINE_DIR` migran si su lector se movió.
- [ ] **T6 — `flows/` + `runner.py` delgado**: `do_index` (:1648), `do_eval` (:1767), `_run_csv_mode` (:1063), gate consts (:1940–1963) + `_format_rate` (:1965) + `_probe_gate_depth` (:1970) + `_probe_gate_verdict` (:1981) + `_print_probe_gate_report` (:1996) + `do_pool_probe` (:2027). `runner.py` queda con `build_parser` (:2059) + `main` (:2141) + dispatch + bootstrap de path. Landmines: source-inspection test :1175, `test_tero_client.py:528`, patches de `do_eval/do_index/_run_csv_mode`.
- [x] **T7 — Sincronizar `docs/rag-eval-flow.md`** (en disco, sin stage): tabla de verificación con rangos `runner.py:NNNN` stale → nuevos módulos; nota de arquitectura. Staging = decisión del usuario. ✅ Evidencia T7 (2026-10-16): doc reescrito en disco (sección «Estructura de módulos» + tabla de verificación con refs a `flows/`/`metrics/`/`reporting/`/`judge.py` + nota de suite y baseline). Sin stage (untracked).

Rangos citados = runner.py original @ `13abffb` (2.153 líneas); se desplazan con cada tarea.

## Acceptance

- Cada tarea: suite WSL `10 failed, 548 passed` con los mismos 10 nombres; `558` collected; `import runner` smoke OK en WSL; grep sin definiciones residuales de lo movido.
- Cada eliminación/movimiento re-verificado: los imports de tests apuntan al módulo nuevo; ningún `patch.object(runner, X)` mudo queda donde el consumidor se movió.
- 0 cambios de comportamiento: mismo stdout/formatos/exit codes; entry `python scripts/rag_eval/runner.py` intacto.
- Sin staging de archivos ajenos; sin tests nuevos/removidos (movimiento estructural puro).
- Cada tarea cierra con 1 work-unit commit en la rama + assess (`review assess --committed-only`) registrado.

## Progress / evidence

- Exploración intensa (2 exploradores read-only + spot checks del orquestador): hecha 2026-10-12. Mapa de módulos, clusters y acoplamientos arriba.
- Baseline canónico capturado: `10 failed, 548 passed in 149.57s` → `/tmp/tero_split_baseline.txt`.
- T1 completado: commit `8c7545a`; suite WSL 10F/548P (mismos 10); assess medium / under_budget / review_due=false (base 9460541, 356 líneas).
- T2 completado: commit `f180d92`. Assess: medium / slice_budget_reached → review_due=true. Consent `granted` ejecutado por el usuario (terminal); review del slice `9460541..f180d92` con lente `review-reliability` → **APROBADO** (lineage `review-a926ccbcc32f04ee`, 16 archivos / 1074 líneas, revisión `sha256:8287447b…`). Acknowledgement exacto ejecutado: `review-acknowledged/v1`, authority: burned. Boundary revisado avanza a `f180d92`.
- T3 completado: commit `b197e7e`. Assess: medium / slice_budget_reached → review_due=true. Consent `granted` por el usuario; review del slice `f180d92..b197e7e` (lente `review-reliability`) → **APROBADO** (lineage `review-910301569f0d37ba`; revisión `sha256:f0441638…`). Acknowledgement: `review-acknowledged/v1`, authority: burned. Boundary revisado avanza a `b197e7e`.
- T4 completado: commit `62e4a89` (3 archivos, +423/−397): `judge.py` extraído (7 símbolos + 3 constantes env de costos); 23 patch targets + 2 monkeypatch retargeteados a `judge`; paridad 10F/548P, collect 558, smokes OK. Review del slice `b197e7e..62e4a89` (lineage `review-ad4e1cf015baf26d`) → **APROBADO** (revisión `sha256:3a6c69b5…`); ack: authority burned. Boundary avanza a `62e4a89`.
- T5 completado: commit `1c0d1b2` (5 archivos, +304/−263): paquete `reporting/` (`csv_io.py` + `baseline.py`, 13 símbolos); 4 patches de `BASELINE_DIR` retargeteados a `reporting.baseline` (lector movido); paridad 10F/548P, collect 558, smokes OK; runner 1110→878 líneas. Review del slice `62e4a89..1c0d1b2` (lineage `review-705c26f5a92c1792`) → **APROBADO** (revisión `sha256:951e7c21…`); ack: authority burned. Boundary avanza a `1c0d1b2`. Advisories: R3-1 (WARNING, namespaces `json`/`BASELINE_DIR`/`EVAL_TOP_K` — VERIFICADO 0 referencias residuales en runner) y R3-2 (SUGGESTION, tests vía shims — se atiende en T6 migrando imports a módulos canónicos).
- T6 completado: commit `0da412f` (8 archivos, +884/−948): paquete `flows/` (index/eval_live/eval_csv/probe); `runner.py` 878→146 (solo CLI); 76 patch retargets + 25 migraciones de imports + 1 source-inspection retarget (R3-2 atendido); paridad 10F/548P, collect 558, entry `--help` OK. Review del slice `1c0d1b2..0da412f` (lineage `review-955dad6bc7784fc7`): **4R completo + refuter → correction_required** — 3 findings corroborados sobre el mismo defecto real (R2-eval-live-missing-import CRITICAL, R3-1 BLOCKER, R4-1 BLOCKER): `_move_model_id_last` llamado sin import en `flows/eval_live.py` (NameError en el branch `parametric_suspect` del eval live; la suite no lo cubría). Corrección acotada: commit `9c5c03b` (+1/−1, plan 2/200 líneas) + paridad 10F/548P + targeted validation OK → **APROBADO** (target `sha256:83ef99bde533dc52a7ee7a72beb6c91d8533191f3511483864b3c3b2acea0043`, revisión `sha256:b272eb82…`); ack: authority burned. Boundary avanza a `9c5c03b`.
- **Follow-ups advisory ATENDIDOS (2026-10-16)**: commit `6610c94` (3 archivos, +156/−2) — (a) R1-1: bounds check `0 <= idx < len(corpus)` en el retry de `flows/index.py` + test RED→GREEN (antes: un filename negativo re-subía el documento equivocado en silencio); (b) R2-index-config-shadow: local `config` → `tool_config`; (c) R3-2: test `TestLiveSanityRewrite` cubre el branch live del sanity-rewrite completo (incluido el re-guardado con `model_id` al final). Suite: 10F/550P (mismos 10; +2 tests nuevos), collect 560. Assess: medium / under_budget / review_due=false (base 9c5c03b).
- **INCIDENTE DE TOOLING (2026-10-12)**: tras el renaming de tools del runtime (`bash`→`shell`, `task`→`subagent`), el tool `shell` falla en TODA invocación — «El sistema no puede encontrar la ruta especificada» (exit 1) — incluso con comandos triviales sin workdir; `read`/`glob`/`edit` funcionan. Sin shell no hay STATUS de re-entrada, captura del lens, ack ni git. Nada se mutó desde el `granted`; el estado del review queda preservado provider-side. **RESUELTO (2026-10-15)**: causa raíz = el shell tool tenía cacheado un pwsh del Store ya inexistente (`C:\Program Files\WindowsApps\Microsoft.PowerShell_7.6.6.0_x64__8wekyb3d8bbwe\pwsh.EXE` — verificado File not found); fix aplicado = clave top-level `shell` en `C:\Users\ikerg\.config\opencode\opencode.json` → `C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe` (hot-reload por watcher, verificado con `Write-Output` + `Get-Location`). Nota: los comandos ahora corren en Windows PowerShell 5.1 — sin `&&` (usar `;`).
- **Nota V2 (2026-10-15)**: con OpenCode 2.x, gentle-ai exige la *relay declaration* (`GENTLE_AI_OPENCODE_RELAY_CONTRACT=gentle-ai.opencode-relay/v2-staged`) en `review start`; el plugin V2 la inyecta solo en shells creadas por el runtime. Los `granted` del usuario deben prefijar `$env:GENTLE_AI_OPENCODE_RELAY_CONTRACT = "gentle-ai.opencode-relay/v2-staged"` en su terminal. Verificado empíricamente: shell del runtime → `consent/v3`; terminal sin la var → `immutable_review_transport_unsupported`. Plugins V2 cargan OK en el servicio (los fallos de carga del log eran de ayer, pre-actualización).
- Follow-ups advisory del review (NO bloqueantes; nunca re-correr el review del candidato): R3-METRIC-PATCH-SEAM (WARNING) — guía para futuros tests que mockeen helpers vía `runner` (verificado 2026-10-12: 0 patch sites reales afectados). R3-BOOTSTRAP-ORDER (WARNING) — hardening de orden del bootstrap candidato para T6. R3-CONFIG-IMPORT-SIDEEFFECT (SUGGESTION) — trade-off aceptado del bootstrap en módulos puros. R3-MOCK-RESOLUTION (SUGGESTION, review T3) — resolución de globals intra-paquete: cubierta por la auditoría de patch sites de cada tarea. R3-A1 (WARNING, re-export parcial de las 3 constantes de costos — sin consumidores residuales, aceptado) y R3-A2 (SUGGESTION, comentario contradictorio del binding block — se corrige en T6).
- Rama `refactor/rag-eval-module-split` creada desde `feat/upgrade-v1.6` @ `13abffb`.

## Next steps

- **Todo implementado** en `refactor/rag-eval-module-split`: 8 commits (`8c7545a` `f180d92` `b197e7e` `62e4a89` `1c0d1b2` `0da412f` `9c5c03b` `6610c94`; 5 slices revisados+ackeados, uno con corrección validada). Sin advisories abiertos. Pendiente de decisiones del usuario: staging de `docs/rag-eval-flow.md` (untracked), merge de la rama a `feat/upgrade-v1.6` y push.
- Al cierre: decisión del usuario sobre staging de `docs/rag-eval-flow.md` y merge de la rama.
