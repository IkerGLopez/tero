# RAG Eval

Framework de evaluación RAG (Retrieval-Augmented Generation) para agentes de Tero. Mide la calidad del retrieval y la generación usando datasets estándar de HuggingFace y métricas de [RAGAS](https://docs.ragas.io/).

## Estructura de archivos

| Archivo | Responsabilidad |
|---|---|
| `runner.py` | CLI principal. Subcomandos `index` y `eval`. Construye métricas RAGAS, modo CSV offline, gestión de baselines, tracking de costos. |
| `tero_client.py` | Cliente HTTP asíncrono para la API de Tero (subir docs, crear hilos, parsear SSE streams). |
| `rag_datasets.py` | Carga datasets desde HuggingFace: RAGBench, FeTaQA, StratRAG. |
| `export_datasets.py` | Utilidad manual: exporta datasets de HF a archivos locales sin depender de Tero. |
| `analysis.py` | Cómputo de estadísticas (media, desvío), tablas comparativas, comparación contra baseline. |
| `sanity_checks.py` | Chequeos post-evaluación sobre el DataFrame de resultados (detección de conocimiento paramétrico, faithfulness-cero-con-citas, divergencia correctness-grounded). |
| `smoke_test.py` | Test rápido de conectividad: verifica que el agente existe, la tool de docs está activa y el retrieval funciona. |
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
| `--dataset` | str | requerido | Dataset |
| `--model-id` | str | `""` | Model ID para lookup del baseline |
| `--compare` | flag | `False` | Compara contra baseline guardado |

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
| `error` | `None` en éxito; string de error en fallo |

Los CSVs de salida usan `;` como separador para compatibilidad con Excel.

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
