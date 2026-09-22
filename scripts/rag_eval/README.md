# RAG Eval

Framework de evaluación RAG (Retrieval-Augmented Generation) para agentes de Tero. Mide la calidad del retrieval y la generación usando datasets estándar de HuggingFace y métricas de [RAGAS](https://docs.ragas.io/).

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `runner.py` | CLI principal. Subcomandos `index`, `eval` y `pool-probe`. Construye métricas RAGAS, modo CSV offline, gestión de baselines, tracking de costos. |
| `tero_client.py` | Cliente HTTP asíncrono para la API de Tero (subir docs, crear hilos, parsear SSE streams). |
| `rag_datasets.py` | Carga datasets desde HuggingFace: RAGBench, FeTaQA, StratRAG. Resuelve el gold linkage de cada fila. |
| `retrieval_matching.py` | Primitivas stdlib-only compartidas: normalización de whitespace, decodificación de claves de oración y resolución de gold documents (la usan el loader, el runner y el probe). |
| `pool_probe.py` | Probe offline del pool denso: embeddea el corpus y rankea por coseno exacto para medir si el gold document es alcanzable a cada profundidad. Sin PostgreSQL ni backend. |
| `export_datasets.py` | Utilidad manual: exporta datasets de HF a archivos locales sin depender de Tero. |
| `analysis.py` | Cómputo de estadísticas (media, desvío), tablas comparativas, comparación contra baseline y análisis pareado (bootstrap + McNemar exacto). |
| `sanity_checks.py` | Chequeos post-evaluación sobre el DataFrame de resultados (detección de conocimiento paramétrico, faithfulness-cero-con-citas, divergencia correctness-grounded). |
| `smoke_test.py` | Test rápido de conectividad: verifica que el agente existe, la tool de docs está activa y el retrieval funciona. |
| `prompts/` | Prompts de referencia **documentation-only**: ningún pipeline los lee. |
| `test_*.py` | Tests unitarios con pytest + pytest-asyncio. |
| `tests/test_tero_client.py` | Tests del cliente HTTP. |

## Instalación

Desde la raíz del repo:

```bash
pip install -r requirements.txt -r requirements-eval.txt
```

Creá un archivo `.env` en la raíz del repo con las variables necesarias (ver sección siguiente).

## Variables de entorno

Cargadas desde `.env` en la raíz del repo.

| Variable | Requerida | Usada por | Descripción |
|---|---|---|---|
| `BEARER_TOKEN` | Sí (eval/index live) | `runner.py`, `smoke_test.py` | JWT de Tero |
| `GOOGLE_API_KEY` | Sí (juez Gemini) | `runner.py` | API key para Gemini |
| `OPENAI_API_KEY` | Sí | `runner.py` | API key para OpenAI |
| `RAG_EVAL_AGENT_ID` | No | `runner.py` | Agent ID por defecto si no se pasa `--agent-id` |
| `EMBEDDING_COST_PER_1K_TOKENS` | No (default: `0.00002`) | `runner.py` | Costo de embedding por 1K tokens |
| `JUDGE_COST_PER_1K_PROMPT_TOKENS` | No (default: `0.00015`) | `runner.py` | Costo de prompt del juez por 1K tokens |
| `JUDGE_COST_PER_1K_COMPLETION_TOKENS` | No (default: `0.00060`) | `runner.py` | Costo de completion del juez por 1K tokens |

Los costos del juez también se resuelven por tabla de precios interna si el modelo es `gemini-3.5-flash`, `gemini-2.5-flash` o `gpt-4o`. Las variables de entorno tienen precedencia sobre la tabla.

## Uso

Todos los comandos se ejecutan desde la raíz del repo.

### Indexar un dataset (`runner.py index`)

Sube el corpus completo de un dataset al agente de Tero. Sin preguntas, sin costo de LLM juez.

```bash
python scripts/rag_eval/runner.py index --dataset ragbench --agent-id 42 --bearer-token <TOKEN>
```

| Flag | Tipo | Default | Descripción |
|---|---|---|---|
| `--dataset` | `ragbench` / `fetaqa` / `stratrag` | requerido | Dataset a indexar |
| `--agent-id` | int | `RAG_EVAL_AGENT_ID` | ID del agente Tero |
| `--max-docs` | int | todos | Máximo de documentos a indexar |
| `--bearer-token` | str | `BEARER_TOKEN` | JWT de Tero |
| `--base-url` | str | `http://localhost:8000` | URL base de la API de Tero |

### Evaluar (`runner.py eval`)

Evalúa un agente pre-indexado con métricas RAGAS, o corre en modo offline contra un CSV.

#### Modo live (contra Tero)

```bash
python scripts/rag_eval/runner.py eval \
  --dataset ragbench \
  --agent-id 42 \
  --models gpt-5,claude-sonnet-4 \
  --judge-model gemini-3.5-flash \
  --bearer-token <TOKEN>
```

#### Modo offline (desde CSV)

Sin dependencia de Tero. El CSV debe tener las columnas `question`, `response`, `retrieved_contexts` (requeridas) y opcionalmente `citations`, `latency_ms`, `grading_notes`, `model_id`.

```bash
python scripts/rag_eval/runner.py eval \
  --dataset ragbench \
  --from-csv resultados.csv \
  --judge-model gpt-4o
```

#### Flags de `eval`

| Flag | Tipo | Default | Descripción |
|---|---|---|---|
| `--dataset` | `ragbench` / `fetaqa` / `stratrag` | requerido | Dataset a evaluar |
| `--agent-id` | int | `RAG_EVAL_AGENT_ID` | ID del agente (no necesario con `--from-csv`) |
| `--max-questions` | int | `1` | Cantidad de preguntas |
| `--models` | str (coma-separados) | — | IDs de modelos Tero (ej. `gpt-5,claude-sonnet-4`) |
| `--bearer-token` | str | `BEARER_TOKEN` | JWT de Tero |
| `--base-url` | str | `http://localhost:8000` | URL base de la API de Tero |
| `--from-csv` | str (path) | — | CSV para evaluación offline |
| `--update-baseline` | flag | `False` | Guarda resultados como nuevo baseline |
| `--compare` | flag | `False` | Compara resultados contra baseline existente |
| `--seed` | int | `14` | Semilla para selección de preguntas |
| `--judge-model` | str | `gemini-3.5-flash` | Modelo juez LLM. Prefijo `gpt*`/`o1*`/`o3*`/`o4*` → OpenAI; `gemini*` → Google |
| `--concurrency` | int | `5` | Máximo de preguntas concurrentes a Tero |

### Smoke test (`smoke_test.py`)

Test rápido de conectividad: verifica que el agente existe, la herramienta de documentos está activa y el retrieval de contexto funciona. No requiere RAGAS ni LLM juez.

```bash
python scripts/rag_eval/smoke_test.py --agent-id 42 --dataset ragbench --bearer-token <TOKEN>
```

| Flag | Tipo | Default | Descripción |
|---|---|---|---|
| `--bearer-token` | str | `BEARER_TOKEN` | JWT de Tero |
| `--base-url` | str | `http://localhost:8000` | URL base de la API |
| `--agent-id` | int | auto (desde `evals/eval_agent.json`) | Agente a testear |
| `--dataset` | str | `ragbench` | Dataset para el corpus |
| `--question` | str | — | Pregunta custom (pisa la default) |
| `--upload-corpus` | flag | `False` | Sube el corpus y sale |
| `--rebuild` | flag | `False` | Fuerza re-upload aunque ya esté subido |
| `--corpus-size` | int | `10` | Máximo de docs para el smoke test |

### Análisis (`analysis.py`)

Calcula estadísticas (media, desvío estándar) y tablas comparativas sobre los CSVs de resultados. Opcionalmente compara contra un baseline guardado.

```bash
python scripts/rag_eval/analysis.py --dataset ragbench --model-id gpt-5 --compare
```

| Flag | Tipo | Default | Descripción |
|---|---|---|---|
| `--dataset` | str | requerido (salvo modo pareado) | Dataset |
| `--model-id` | str | `""` | Model ID para lookup del baseline |
| `--compare` | flag | `False` | Compara contra baseline guardado |

#### Modo pareado (A/B)

Compara dos corridas pregunta a pregunta: delta puntual, intervalo de confianza bootstrap (BCa por defecto, fallback a percentil con datos degenerados) y McNemar exacto sobre la tabla 2×2 de `doc_recall_5`. `delta` es `media(métrica_A − métrica_B)`, así que un valor positivo favorece al brazo A.

```bash
python scripts/rag_eval/analysis.py \
  --paired-a arm_a.csv --paired-b arm_b.csv \
  --metric doc_recall_5 --rng-seed 14 --n-resamples 9999 \
  --out report.json
```

| Flag | Tipo | Default | Descripción |
|---|---|---|---|
| `--paired-a` / `--paired-b` | str (path) | — | CSVs de resultados de cada brazo (juntos habilitan el modo pareado) |
| `--metric` | str | `doc_recall_5` | Métrica a parear |
| `--rng-seed` | int | `14` | Semilla del bootstrap (reproducible) |
| `--n-resamples` | int | `9999` | Resamples del bootstrap |
| `--out` | str (path) | — | Escribe el reporte como JSON |

Regla de reclamo (pre-registrada): se declara una mejora **solo** si el IC 95% excluye 0 **y** el McNemar exacto da `p < 0.05`; en cualquier otro caso se reporta "no detectable difference at this sample size". Las preguntas sin métrica en alguno de los brazos (excluidas por out-of-corpus o sin labels) y las no apareadas se descartan con un warning y su conteo queda en el reporte.

### Pool probe (`runner.py pool-probe`)

Probe offline del pool denso: embeddea el corpus del dataset y las preguntas, rankea por coseno exacto y reporta si el gold document es alcanzable a cada profundidad. No usa PostgreSQL/PGVector ni llama al backend — sólo la API de embeddings de OpenAI.

```bash
python scripts/rag_eval/runner.py pool-probe \
  --dataset ragbench --questions 304 --seed 14 --depths 20 50 100 500
```

| Flag | Tipo | Default | Descripción |
|---|---|---|---|
| `--dataset` | `ragbench` / `fetaqa` / `stratrag` | **requerido** | Dataset a probear (se carga con el loader compartido) |
| `--questions` | int | **requerido** | Cantidad de preguntas a probear (tamaño de muestra del gate) |
| `--seed` | int | `14` | Semilla de la selección determinística de preguntas |
| `--embedding-model` | str | `text-embedding-3-small` | Modelo de embeddings — debe coincidir con el corpus indexado |
| `--max-docs` | int | corpus completo | Prefijo del corpus a probear (espeja la corrida indexada) |
| `--depths` | ints | `20 50 100 500` | Profundidades del pool a reportar |
| `--cache-dir` | str | `evals/pool_probe_cache` (gitignored) | Cache de embeddings content-addressed |

**Migración de CLI**: antes `pool-probe` era sólo FeTaQA y default `--depths 100 500`. Ahora `--dataset` es **requerido** y el default es `20 50 100 500`. Las invocaciones existentes deben agregar `--dataset fetaqa`; no hay formato de reporte persistido que migrar.

#### Regla del gate

La decisión se evalúa a la **profundidad de `fetch_k`**: la mayor profundidad configurada `≤ 50` (si no hay ninguna, la más superficial configurada), porque un reranker sólo reordena el pool que recibe:

| Tasa gold-in-pool en la profundidad del gate | Banda |
|---|---|
| `≥ 0.90` | **proceed** — el reranker sobre el pool es viable |
| `0.70 – 0.90` | **conditional** — sólo ayuda a las preguntas cuyo gold ya está en el pool; documentá la cobertura |
| `< 0.70` | **hold** — primero arreglá la cobertura del retrieval |
| `None` | **inconclusive** — no hay preguntas scoreables |

Los umbrales son defaults documentados y se calibran sobre el reporte real; la habilitación del reranker queda como decisión del operador (`.env` + restart) y define una nueva línea base de comparabilidad para **todos** los datasets.

**Diagnóstico del confound histórico**: corré además `--max-docs 500` para cuantificar cuántas preguntas quedan out-of-corpus con el índice viejo de 500 documentos; leé `n_out_of_corpus` antes de re-indexar.

#### Caveats del probe

- Replica **embedding + coseno exacto**, no los internals de ANN/índice de PGVector: es evidencia **necesaria pero no suficiente** para el retriever live.
- Embeddea cada documento **completo**, así que los documentos que el chunking indexado parte en varios pedazos son una **aproximación documentada** (no hay paridad exacta a nivel chunk).
- El corpus tiene **774 documentos duplicados** (contenido idéntico entre filas): el reporte lo informa como `n_duplicate_documents` y la semántica de dedupe no cambia.
- **No-label ≠ out-of-corpus ≠ miss**: son tres poblaciones distintas y el reporte las cuenta por separado (`n_scored` / `n_out_of_corpus` / `n_no_gold_labels`), con el invariante `n_questions == n_scored + n_out_of_corpus + n_no_gold_labels`.
- La identidad por pregunta es genérica: `row_id` (más `row_id_source`: `loader` o `selection_index`). Para FeTaQA el valor numérico de `feta_id` reaparece bajo `row_id`.

### Exportar datasets (`export_datasets.py`)

Exporta datasets de HuggingFace a archivos locales para carga manual por UI. Sin dependencia de Tero.

```bash
python scripts/rag_eval/export_datasets.py --dataset ragbench --n 20 --corpus-size 100 --manual
```

| Flag | Tipo | Default | Descripción |
|---|---|---|---|
| `--dataset` | str | — (los tres) | Dataset a exportar |
| `--n` | int | `10` | Cantidad de preguntas |
| `--corpus-size` | int | `200` | Máximo de docs del corpus (`0` = solo preguntas) |
| `--manual` | flag | `False` | **Requerido** para ejecutar (safety gate) |

## Datasets soportados

| Nombre | HuggingFace ID | Tipo |
|---|---|---|
| `ragbench` | `galileo-ai/ragbench` (techqa) | QA técnica con grounding labels |
| `fetaqa` | `DongfuJiang/FeTaQA` | QA libre basada en tablas |
| `stratrag` | `Aryanp088/StratRAG` | QA multi-hop con documentos distractores |

## Métricas

Cada evaluación produce estas columnas por pregunta:

| Columna | Descripción |
|---|---|
| `question` | Texto de la pregunta |
| `grading_notes` | Respuesta de referencia (gold answer del dataset) |
| `response` | Respuesta del agente |
| `retrieved_contexts` | Contextos recuperados, separados por `\|` |
| `citations` | Citas extraídas de la respuesta, separadas por `\|` |
| `latency_ms` | Tiempo de respuesta en milisegundos |
| `correctness` | Escala 0-4: qué tan bien coincide la respuesta con las grading notes |
| `faithfulness` | Escala 0-1: la respuesta está fundamentada en los contextos recuperados |
| `context_recall` | Escala 0-1: fracción del contexto relevante que fue recuperado |
| `context_precision` | Escala 0-1: fracción del contexto recuperado que es relevante |
| `citation_faithfulness` | Escala 0-1: las citas están respaldadas por sus chunks |
| `grounded_correctness` | Compuesto: `(correctness / 4) * faithfulness` |
| `relevant_chunk_position` | Índice (base 1) del contexto que mejor coincide con las grading notes |
| `table_recall_5` | Determinística (FeTaQA): 1 si el documento gold está entre los primeros 5 contextos únicos |
| `cell_recall_5` | Determinística (FeTaQA): fracción de celdas gold no degeneradas presentes en los primeros 5 contextos |
| `doc_recall_5` | Determinística (multi-gold): 1 si **cualquier** gold document in-prefix aparece entre los primeros 5 contextos únicos, 0 si no |
| `sentence_recall_5` | Determinística: fracción de oraciones gold usables (in-prefix) halladas como substring normalizado de esos contextos |
| `doc_coverage_5` | Determinística, secundaria: fracción de gold documents in-prefix alcanzados por esos contextos (con un solo gold degenera a `doc_recall_5`) |
| `error` | `None` en éxito; string de error en fallo |

Los CSVs de salida usan `;` como separador para compatibilidad con Excel.

### Semántica de exclusión (métricas determinísticas)

Las métricas determinísticas nunca cuentan una exclusión como un miss:

- **Out-of-corpus** (el gold document quedó fuera del prefijo indexado, p. ej. `--max-docs`) → `None`.
- **Sin gold labels** (la fila del dataset no trae anotaciones) → `None`, y es una población **distinta** de out-of-corpus.
- **Sin match** (pregunta etiquetada y en-corpus, pero el gold no está en el top-5) → `0`, y la pregunta **sí** queda en el denominador.

`None` viaja de punta a punta: celda vacía en el CSV → `pd.to_numeric(...).dropna()` → fuera de la media y del denominador. El run live imprime además un reporte de alineación con las poblaciones (`scorable` / `out-of-corpus` / `no-label` / sin linkage) y, cuando existen resultados, `measured` / `misses` / `unmeasured` — así el denominador de `doc_recall_5` se puede auditar contra `n_scorable`.

**Tamaño del CSV**: las filas anotan los campos gold (`gold_doc_ids`, `gold_sentences`, `gold_sentence_doc_ids`, `no_gold_labels`, `gold_doc_ids_in_prefix`, `gold_contents_in_prefix`, `gold_sentences_in_prefix`). Son columnas **aditivas**: ningún consumidor existente se rompe, y el peso dominante del CSV sigue siendo `retrieved_contexts` (5 chunks por fila).

## Tests

Desde la raíz del repo:

```bash
# Todos los tests
pytest scripts/rag_eval/

# Archivo específico
pytest scripts/rag_eval/test_runner.py

# Con verbose
pytest scripts/rag_eval/ -v

# Cliente HTTP (en subdirectorio)
pytest scripts/rag_eval/tests/
```
