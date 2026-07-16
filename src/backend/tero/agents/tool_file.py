import asyncio
import logging
from typing import cast

from asyncpg.exceptions import DeadlockDetectedError, SerializationError
from sqlalchemy.exc import SQLAlchemyError
from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi.background import BackgroundTasks

from ..core import repos as repos_module
from ..files.core import add_encoding_to_content_type, QuotaExceededError
from ..files.domain import File, FileStatus, FileMetadata
from ..files.repos import FileRepository
from ..tools.core import AgentTool
from ..tools.repos import ToolRepository
from ..users.domain import User
from ..users.repos import UserRepository
from .domain import AgentToolConfigFile, Agent
from .repos import AgentToolConfigFileRepository, AgentRepository


logger = logging.getLogger(__name__)

# Limit concurrent background tasks to avoid exhausting the DB connection pool.
# Each _add_tool_file() opens an AsyncSession and holds it for the entire
# pipeline.  With DB_POOL_SIZE=20 + DB_MAX_OVERFLOW=30 = 50 max connections,
# 20 concurrent tasks keeps the pool well within limits.
_TASK_SEMAPHORE = asyncio.Semaphore(20)


async def upload_tool_file(file: File, tool: AgentTool, agent_id: int, user: User, db: AsyncSession, background_tasks: BackgroundTasks) -> FileMetadata:
    file.content_type = add_encoding_to_content_type(file.content_type, file.content)
    file = await FileRepository(db).add(file)
    await AgentToolConfigFileRepository(db).add(AgentToolConfigFile(agent_id=agent_id, tool_id=tool.id, file_id=file.id))
    # Pass file_id instead of file object to avoid session conflicts
    # The background task will create its own session and re-fetch the file
    background_tasks.add_task(_add_tool_file, file.id, user.id, tool.id, agent_id, tool.config)
    logger.info(f"[tool_file] Upload scheduled file_id={file.id} name={file.name!r} tool_id={tool.id} agent_id={agent_id}")
    return FileMetadata.from_file(file)


async def _add_tool_file(file_id: int, user_id: int, tool_id: str, agent_id: int, tool_config: dict):
    logger.info(f"[tool_file] Processing started file_id={file_id} tool_id={tool_id} agent_id={agent_id}")
    async with _TASK_SEMAPHORE:
        async with AsyncSession(repos_module.engine, expire_on_commit=False) as db:
            f = cast(File, await FileRepository(db).find_by_id(file_id))
            user = cast(User, await UserRepository(db).find_by_id(user_id))
            agent = cast(Agent, await AgentRepository(db).find_by_id(agent_id))
            tool = cast(AgentTool, ToolRepository().find_by_id(tool_id))
            tool.configure(agent, user_id, tool_config, db)

            retried = False

            try:
                await tool.add_file(f, user)
                f.status = FileStatus.PROCESSED
                logger.info(f"[tool_file] Processing completed file_id={file_id} status=PROCESSED")
            except QuotaExceededError:
                f.status = FileStatus.QUOTA_EXCEEDED
                logger.error(f"Quota exceeded for user {user_id} when adding tool file {file_id} {f.name}")
            except SQLAlchemyError as e:
                if hasattr(e, '__cause__') and isinstance(e.__cause__, (DeadlockDetectedError, SerializationError)):
                    if not retried:
                        await asyncio.sleep(5)
                        retried = True
                        try:
                            await tool.add_file(f, user)
                            f.status = FileStatus.PROCESSED
                            f.error_reason = None
                            logger.info(f"[tool_file] Processing completed after retry file_id={file_id} status=PROCESSED")
                        except AssertionError:
                            f.status = FileStatus.ERROR
                            f.error_reason = "TIME_SYNC"
                            logger.error(f"Time sync on DB retry for file {file_id} {f.name}")
                        except Exception:
                            f.status = FileStatus.ERROR
                            f.error_reason = "RETRY_FAILED"
                            logger.error(f"Retry failed for file {file_id} {f.name} (original: {type(e.__cause__).__name__ if hasattr(e, '__cause__') and e.__cause__ else type(e).__name__})")
                    else:
                        f.status = FileStatus.ERROR
                        f.error_reason = "DB_DEADLOCK" if isinstance(e.__cause__, DeadlockDetectedError) else "DB_SERIALIZATION"
                        logger.error(f"DB conflict exhausted retries for file {file_id} {f.name}: {f.error_reason}")
                else:
                    f.status = FileStatus.ERROR
                    f.error_reason = "DB_ERROR"
                    raise
            except AssertionError as e:
                # LangChain's SQLRecordManager raises AssertionError("Time sync issue")
                # when concurrent aindex() calls with cleanup="incremental" cause
                # the DB clock to appear to move backwards relative to index_start_dt.
                if "Time sync" in str(e):
                    if not retried:
                        await asyncio.sleep(5)
                        retried = True
                        try:
                            await tool.add_file(f, user)
                            f.status = FileStatus.PROCESSED
                            f.error_reason = None
                            logger.info(f"[tool_file] Processing completed after time-sync retry file_id={file_id}")
                        except SQLAlchemyError as db_exc:
                            f.status = FileStatus.ERROR
                            f.error_reason = "DB_DEADLOCK" if (
                                hasattr(db_exc, '__cause__') and
                                isinstance(db_exc.__cause__, DeadlockDetectedError)
                            ) else "DB_SERIALIZATION"
                            logger.error(f"DB error on time-sync retry for file {file_id} {f.name}")
                        except Exception:
                            f.status = FileStatus.ERROR
                            f.error_reason = "TIME_SYNC"
                            logger.error(f"Time sync exhausted retries for file {file_id} {f.name}")
                    else:
                        f.status = FileStatus.ERROR
                        f.error_reason = "TIME_SYNC"
                        logger.error(f"Time sync exhausted retries for file {file_id} {f.name}")
                else:
                    f.status = FileStatus.ERROR
                    f.error_reason = "ASSERTION_ERROR"
                    raise
            except Exception as e:
                f.status = FileStatus.ERROR
                # LangChain's SQLRecordManager raises AssertionError("Time sync issue")
                # under concurrent aindex() calls with cleanup="incremental".
                if "Time sync" in str(e):
                    f.error_reason = "TIME_SYNC"
                logger.error(f"Error adding tool file {file_id} {f.name} {e}", exc_info=True)
            finally:
                await FileRepository(db).update(f)
