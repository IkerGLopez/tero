# Bug: "Error al subir el archivo" — Indexación masiva de corpus

## Resumen

Al indexar corpus completos en los agentes 10, 11 y 12 (200–3000 archivos por agente), entre un 0.8% y 1% de los documentos quedaban en estado `ERROR` con el mensaje `"Error al subir el archivo"`. **El 100% de los errores aparecían en pares consecutivos**, firma inequívoca de un conflicto de escritura concurrente.

**Solución final: 0 errores en 5,521 documentos.**

---

## Diagnóstico

### Síntoma y patrón

| Agente | Documentos | Errores (antes) | Patrón |
|---|---|---|---|
| 10 | 1,001 | 8 (0.8%) | Pares consecutivos: 152-153, 384-385, 608-609, 819-820 |
| 11 | 3,000 | 29 (0.97%) | Pares consecutivos + un triplete |
| 12 | 1,520 | 15 (0.99%) | Pares consecutivos |

El patrón de **pares consecutivos revela que dos tareas de background fallan simultáneamente** — no es contenido de archivo, es timing.

### Hipótesis 1: Rate limiting del LLM → ❌ Descartada

Se pensó que 200 tareas concurrentes llamando a `gpt-4o-mini` saturaban el rate limit del provider. **Falso**: el script `runner.py` configura `skipDescriptions: true` durante la indexación de corpus. Las líneas que invocan al LLM (`_generate_file_description`, `_update_tool_description`) **nunca se ejecutan**:

```python
# tool.py — esta rama está muerta durante indexación
if not self.config.get("skipDescriptions"):
    await self._update_tool_description_with_file(...)  # nunca llamado
```

### Hipótesis 2: Deadlocks de PGVector → ⚠️ Existen, pero no son la causa principal

`aindex()` con `cleanup="incremental"` hace DELETE + INSERT sobre `langchain_pg_embedding`. Con 200 escrituras concurrentes, PostgreSQL genera deadlocks (40P01). Se agregó un semáforo y reintentos, pero **los errores persistían** incluso serializando completamente las escrituras.

### Causa raíz: `AssertionError("Time sync issue")` de LangChain

El verdadero culpable está **dentro de LangChain**, en `SQLRecordManager.aupdate()`:

```python
# langchain_core/indexing/_sql_record_manager.py
update_time = self.get_time()          # SELECT CURRENT_TIMESTAMP
if time_at_least and update_time < time_at_least:
    raise AssertionError(f"Time sync issue: {update_time} < {time_at_least}")
```

El flujo dentro de `aindex()` es:

1. `index_start_dt = get_time()` → captura timestamp al inicio de `aindex()`
2. Procesa documentos (hashing, splitting)
3. `aupdate(keys, time_at_least=index_start_dt)` → vuelve a llamar `get_time()`, compara

Si entre el paso 1 y el paso 3 **otra tarea concurrente** commiteó registros con timestamp > `index_start_dt`, el `get_time()` del paso 3 puede devolver un valor que el chequeo considera inconsistente.

**Este `aupdate()` se ejecuta SIEMPRE** dentro de `aindex()`, sin importar los parámetros `force_update` o `cleanup`. Es imposible evitarlo desde fuera de LangChain sin monkey-patching.

---

## Solución: tres capas de defensa

La solución ataca el problema en tres niveles: prevención (semáforos), mitigación (reintentos), y recuperación (retry loop del runner).

### Capa 1 — Prevención: semáforo por agente en `aindex()`

**Archivo**: `src/backend/tero/tools/docs/tool.py`

```python
# Diccionario module-level, keyed by agent_id.
# Module-level es CRÍTICO: ToolRepository() crea nuevas instancias de DocsTool
# por llamada, así que un semáforo de instancia (PrivateAttr) no funciona.
_index_semaphores: dict[int, asyncio.Semaphore] = {}

# En _handle_file(), cuando skipDescriptions=True:
sem = _index_semaphores.setdefault(self.agent.id, asyncio.Semaphore(1))
async with sem:
    await aindex(
        self._split_file_content(file_doc),
        self._build_record_manager(),
        self._build_vectorstore(),
        cleanup="incremental",
        source_id_key="id",
        key_encoder="sha256",
        force_update=True,   # evita el path de aupdate() para refrescar timestamps
    )
```

**Qué hace**: Solo un `aindex()` a la vez por agente. Sin concurrencia entre `index_start_dt` y `aupdate()`, el chequeo de time sync siempre pasa porque el tiempo solo avanza.

**Por qué `Semaphore(1)` y no 2 o 3**: Cualquier concurrencia >1 en `aindex()` reintroduce el time sync. Probado empíricamente: `Semaphore(2)` → ~5% errores, `Semaphore(1)` → 0%.

**Por qué `force_update=True`**: Con `force_update`, los documentos existentes se reindexan en vez de refrescar su timestamp vía `aupdate()`, evitando uno de los dos paths que ejecutan el chequeo. El otro path (el `aupdate()` que refresca TODOS los documentos del batch) sigue ejecutándose — no se puede evitar.

**Efecto**: Elimina la causa raíz del time sync. El costo es velocidad: ~1s por archivo secuencialmente → ~25 min para 1500 documentos.

---

### Capa 2 — Mitigación: semáforo de tareas

**Archivo**: `src/backend/tero/agents/tool_file.py`

```python
# Limita cuántas background tasks abren sesión de BD simultáneamente.
# El pool de PostgreSQL tiene DB_POOL_SIZE=20 + overflow=30 = 50 conexiones.
# Con 5 slots, nunca agotamos el pool incluso con otras requests concurrentes.
_TASK_SEMAPHORE = asyncio.Semaphore(5)

async def _add_tool_file(file_id, user_id, tool_id, agent_id, tool_config):
    async with _TASK_SEMAPHORE:
        async with AsyncSession(repos_module.engine, expire_on_commit=False) as db:
            # ... pipeline completo ...
```

**Qué hace**: Solo 5 tareas abren sesión de BD a la vez. Las otras 195 esperan **fuera** del `async with`, sin consumir conexiones del pool. Esto evita `QueuePool limit reached` y timeouts de conexión.

**Por qué 5**: Con `_index_semaphore(1)`, una tarea está en `aindex()` y 4 esperan en el semáforo de index, reteniendo conexiones. 5 conexiones ocupadas de 50 disponibles → margen holgado. Si se sube a 20, las 19 que esperan en el semáforo de index consumen 19 conexiones, dejando solo 31 para el resto del sistema — todavía seguro pero más ajustado.

**Efecto**: Previene agotamiento del pool de BD. Sin esto, 200 tareas intentando abrir `AsyncSession` simultáneamente saturan las 50 conexiones y el resto espera hasta 60s (pool timeout) antes de fallar.

---

### Capa 3 — Reintentos automáticos con `error_reason`

**Archivo**: `src/backend/tero/agents/tool_file.py`

Tres handlers de excepción en `_add_tool_file()`, cada uno con 1 reintento (5s backoff):

```python
try:
    await tool.add_file(f, user)
    f.status = FileStatus.PROCESSED

except QuotaExceededError:
    f.status = FileStatus.QUOTA_EXCEEDED

except SQLAlchemyError as e:
    if hasattr(e, '__cause__') and isinstance(e.__cause__, (DeadlockDetectedError, SerializationError)):
        if not retried:
            await asyncio.sleep(5); retried = True
            try:
                await tool.add_file(f, user)
                f.status = FileStatus.PROCESSED; f.error_reason = None
            except AssertionError:
                f.status = FileStatus.ERROR; f.error_reason = "TIME_SYNC"
            except Exception:
                f.status = FileStatus.ERROR; f.error_reason = "RETRY_FAILED"
        else:
            f.status = FileStatus.ERROR
            f.error_reason = "DB_DEADLOCK" or "DB_SERIALIZATION"
    else:
        f.status = FileStatus.ERROR; f.error_reason = "DB_ERROR"; raise

except AssertionError as e:
    if "Time sync" in str(e):
        [misma lógica de retry que arriba]
    else:
        f.status = FileStatus.ERROR; f.error_reason = "ASSERTION_ERROR"; raise

except Exception as e:
    f.status = FileStatus.ERROR
    if "Time sync" in str(e):
        f.error_reason = "TIME_SYNC"
```

**Qué hace cada handler**:

| Handler | Cuándo se dispara | Qué hace | `error_reason` |
|---|---|---|---|
| `SQLAlchemyError` + asyncpg cause | Deadlock (40P01) o serialization failure (40001) en PGVector | 1 retry tras 5s. Si el retry falla con `AssertionError` → `TIME_SYNC`. Si falla con otra cosa → `RETRY_FAILED`. | `DB_DEADLOCK`, `DB_SERIALIZATION`, `TIME_SYNC`, `RETRY_FAILED` |
| `AssertionError` + "Time sync" | LangChain time sync (la causa raíz) | 1 retry tras 5s. Si el retry falla con `SQLAlchemyError` → inspecciona `__cause__` para `DB_DEADLOCK`/`DB_SERIALIZATION`. | `TIME_SYNC`, `DB_DEADLOCK`, `DB_SERIALIZATION` |
| `Exception` genérico | Cualquier otra excepción | Sin retry. Si el mensaje contiene "Time sync" → `TIME_SYNC`. | `TIME_SYNC` o `None` |
| `raise` (errores no manejables) | `SQLAlchemyError` sin causa asyncpg, `AssertionError` sin "Time sync" | Setea `error_reason` y propaga. | `DB_ERROR`, `ASSERTION_ERROR` |

**Por qué 1 solo reintento**: El time sync es transitorio — ocurre por timing entre tareas concurrentes. Con el semáforo de index serializando, el reintento corre en aislamiento y casi siempre funciona. Múltiples reintentos rara vez ayudan y añaden latencia.

**Efecto**: Atrapa y recupera de errores transitorios que escapan al semáforo. El `error_reason` permite diagnosticar fallos desde la API sin acceder a logs del servidor.

---

### Capa 4 — Recuperación: retry loop en el runner

**Archivos**: `scripts/rag_eval/runner.py`, `scripts/rag_eval/tero_client.py`

Después de la indexación inicial, el runner detecta archivos en ERROR y los re-subí secuencialmente:

```python
# runner.py — do_index()
seen_errors: set[int] = set()   # dedup: no reintentar el mismo ID
replaced: set[int] = set()      # solo IDs con re-upload exitoso (seguro borrar)

for attempt in range(1, 4):
    error_files = await tero.list_error_files()          # GET /api/.../files → filtra ERROR
    fresh = [(fid, name) for fid, name in error_files if fid not in seen_errors]
    if not fresh:
        break
    for fid, name in fresh:
        idx = int(name.removeprefix("doc_").removesuffix(".txt"))  # extrae índice del nombre
        try:
            new_fid = await tero.upload_document(name, corpus[idx].encode("utf-8"))
            retry_ids.append(new_fid)
            replaced.add(fid)     # solo si el upload tuvo éxito
            seen_errors.add(fid)  # solo si el upload tuvo éxito
        except Exception:
            pass  # no se marca como visto → se reintenta en la siguiente iteración
        await asyncio.sleep(2)   # procesamiento aislado, sin concurrencia
    if retry_ids:
        await tero.wait_files_processed(retry_ids)

# Limpieza: borrar los ERROR huérfanos que fueron reemplazados exitosamente
if replaced:
    await tero.delete_files(list(replaced))
```

**Qué hace paso a paso**:

1. `list_error_files()` consulta la API y devuelve `[(id, name), ...]` de archivos en ERROR
2. `seen_errors` evita reintentar el mismo ID dos veces
3. El índice del documento se extrae del nombre (`doc_000037.txt` → 37)
4. Re-upload con `try/except`: si falla, NO se marca como visto → se reintenta en la siguiente iteración
5. `sleep(2)` entre re-uploads asegura procesamiento aislado (sin concurrencia = sin time sync)
6. `wait_files_processed` espera a que los nuevos uploads terminen de procesar
7. Al final, `delete_files(list(replaced))` borra los registros ERROR huérfanos

**Por qué 3 intentos con 2s de delay**: El time sync es transitorio. Un solo reintento aislado casi siempre funciona. 3 intentos cubren el caso raro de fallo en cascada. Los 2s dan tiempo al backend para procesar antes del siguiente upload.

**Por qué `replaced` y `seen_errors` separados**: `seen_errors` es solo para dedup. `replaced` solo contiene IDs cuyo re-upload **tuvo éxito** — seguro borrar el original. Si un re-upload falla, el ID original NO va a `replaced` y NO se borra. Esto evita pérdida de datos.

**Efecto**: Cualquier error residual que escape a las capas 1-3 se recupera automáticamente. La UI queda limpia (sin ERROR huérfanos). El runner es autosuficiente — no requiere intervención manual.

---

### Capa 5 — Throttle de uploads para Keycloak

**Archivo**: `scripts/rag_eval/runner.py`

```python
for idx, doc in enumerate(corpus):
    fid = await tero.upload_document(filename, doc.encode("utf-8"))
    file_ids.append(fid)
    await asyncio.sleep(0.05)  # 50ms entre uploads → 20 req/s
```

**Qué hace**: Inserta 50ms de pausa entre cada POST de upload.

**Por qué**: Cada `POST /api/agents/{id}/tools/docs/files` dispara validación JWT → llamada HTTP del backend a Keycloak. Con 3000 uploads sin pausa, el pool HTTP del módulo de auth se satura y Keycloak devuelve timeouts → 500 Internal Server Error.

**Efecto**: 20 requests/segundo → 3000 uploads en ~2.5 minutos. El auth respira. Sin esto, el upload falla aleatoriamente a mitad de la indexación.

---

### Soporte: `error_reason` en API

**Archivo**: `src/backend/tero/files/domain.py`

```python
# File model (DB)
error_reason: Optional[str] = Field(default=None)

# FileMetadata (API response)
error_reason: Optional[str] = None
```

**Qué hace**: Expone el motivo del error en la respuesta JSON de la API (`errorReason`). Permite al runner y a la UI diagnosticar fallos sin acceder a logs del servidor. Es nullable → no requiere migración de Alembic.

---

## Resultado final

| Agente | Documentos | Errores |
|---|---|---|
| 10 | 1,001 | 0 |
| 11 | 3,000 | 0 |
| 12 | 1,520 | 0 |
| **Total** | **5,521** | **0** |

---

## Archivos modificados

| Archivo | Cambio |
|---|---|
| `src/backend/tero/tools/docs/tool.py` | Per-agent `_index_semaphores` dict, `async with sem` en `aindex()`, `force_update=True` |
| `src/backend/tero/agents/tool_file.py` | `_TASK_SEMAPHORE(5)`, 3 handlers de retry con `error_reason`, clasificación correcta de excepciones |
| `src/backend/tero/files/domain.py` | `error_reason` en `File` y `FileMetadata` |
| `scripts/rag_eval/tero_client.py` | `list_error_files()`, `delete_files()` |
| `scripts/rag_eval/runner.py` | Retry loop post-indexación, limpieza de huérfanos, throttle de uploads |
| `src/backend/tests/assets/init_db.sql` | Seed `gpt-4o-mini` |
| `src/backend/tests/test_file_upload_errors.py` | 5 tests de integración (nuevo) |

---

## Lo que NO se cambió

- **LLM**: `_generate_file_description`, `_update_tool_description` — irrelevantes con `skipDescriptions`
- **Migración de Alembic**: `error_reason` es nullable, SQLModel la crea automáticamente
- **Frontend**: no se modificó la UI
- **Recovery al startup**: fuera del scope aprobado
- **LangChain**: no se monkey-patcheó — toda la solución es externa

---

## Lecciones aprendidas

- **`ToolRepository()` crea nuevas instancias por llamada** → los semáforos de instancia (`PrivateAttr`) no funcionan porque cada task tiene su propia copia. Deben ser module-level (`dict` o variable global).
- **`aindex()` de LangChain siempre ejecuta `aupdate()`** con chequeo de timestamp. Los parámetros `force_update` y `cleanup` no lo previenen. La única defensa externa es eliminar la concurrencia.
- **Judgment Day (5 rondas)** encontró 4 bugs CRITICAL que los tests de integración no cubrían: uso de IDs viejos en el retry, crash en `file_ids.index()`, clasificación errónea de excepciones, y pérdida de datos al borrar ERROR sin reemplazo exitoso.
- **El build de Docker estaba roto** (sass/pnpm en la fase de frontend) → se usó `docker compose cp` para desplegar cambios en el backend.
- **Keycloak se satura con requests de auth** durante uploads masivos sin throttle → `sleep(0.05)` entre uploads resolvió el problema.
- **El equilibrio de semáforos es delicado**: `Semaphore(1)` en `aindex()` = 0 errores garantizado; cualquier valor >1 reintroduce el time sync.
