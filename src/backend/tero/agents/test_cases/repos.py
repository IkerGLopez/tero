from typing import List, Optional, cast

from sqlalchemy.orm import selectinload
from sqlmodel import select, delete, update, and_, col, func
from sqlmodel.ext.asyncio.session import AsyncSession

from ...core.repos import attr, scalar
from ...threads.domain import Thread, ThreadMessage
from ...threads.repos import ThreadRepository
from .domain import TestCase, TestCaseResult, TestSuiteRun, TestSuiteRunEvent, TestSuiteEventType


class TestCaseRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_by_id(self, test_case_id: int, agent_id: int) -> Optional[TestCase]:
        stmt = (
            select(TestCase)
            .join(Thread)
            .where(TestCase.thread_id == test_case_id, TestCase.agent_id == agent_id)
            .options(selectinload(attr(TestCase.thread))))
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def find_by_agent(self, agent_id: int) -> List[TestCase]:
        stmt = (
            select(TestCase)
            .join(Thread)
            .where(TestCase.agent_id == agent_id)
            .options(selectinload(attr(TestCase.thread)))
            .order_by(col(TestCase.thread_id)))
        ret = await self._db.exec(stmt)
        return list(ret.all())

    async def count_by_agent(self, agent_id: int) -> int:
        stmt = select(func.count()).select_from(TestCase).where(TestCase.agent_id == agent_id)
        ret = await self._db.exec(stmt)
        return ret.one() or 0

    async def find_empty_test_case(self, agent_id: int) -> Optional[TestCase]:
        stmt = (
            select(TestCase)
            .join(Thread)
            .outerjoin(ThreadMessage, and_(ThreadMessage.thread_id == Thread.id))
            .where(TestCase.agent_id == agent_id)
            .group_by(col(TestCase.thread_id), col(Thread.creation))
            .having(func.count(col(ThreadMessage.id)) == 0)
            .order_by(col(Thread.creation).asc())
            .limit(1)
            .options(selectinload(attr(TestCase.thread)))
        )
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def save(self, test_case: TestCase) -> TestCase:
        test_case = await self._db.merge(test_case)
        await self._db.commit()
        await self._db.refresh(test_case, ["thread"])
        return test_case

    async def delete(self, test_case: TestCase) -> None:
        await self._db.exec(scalar(
            update(TestCaseResult)
            .where(and_(TestCaseResult.test_case_id == test_case.thread_id))
            .values(test_case_id=None)
        ))

        await self._db.exec(scalar(
            delete(TestCase).where(and_(TestCase.thread_id == test_case.thread_id))
        ))

        await ThreadRepository(self._db).delete(test_case.thread)


class TestCaseResultRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_by_id_and_suite_run_id(self, result_id: int, suite_run_id: int) -> Optional[TestCaseResult]:
        stmt = (
            select(TestCaseResult)
            .where(TestCaseResult.id == result_id, TestCaseResult.test_suite_run_id == suite_run_id)
        )
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def save(self, result: TestCaseResult) -> TestCaseResult:
        result = await self._db.merge(result)
        await self._db.commit()
        await self._db.refresh(result)
        return result

    async def find_by_suite_run_id(self, suite_run_id: int) -> List[TestCaseResult]:
        stmt = (
            select(TestCaseResult)
            .where(TestCaseResult.test_suite_run_id == suite_run_id)
            .order_by(col(TestCaseResult.executed_at).asc(), col(TestCaseResult.id).asc())
        )
        ret = await self._db.exec(stmt)
        return list(ret.all())


class TestSuiteRunRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_by_id_and_agent_id(self, suite_run_id: int, agent_id: int) -> Optional[TestSuiteRun]:
        stmt = select(TestSuiteRun).where(TestSuiteRun.id == suite_run_id, TestSuiteRun.agent_id == agent_id)
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def find_by_agent_id(self, agent_id: int, limit: int = 20, offset: int = 0) -> List[TestSuiteRun]:
        stmt = (
            select(TestSuiteRun)
            .where(TestSuiteRun.agent_id == agent_id)
            .order_by(col(TestSuiteRun.executed_at).desc())
            .limit(limit)
            .offset(offset)
        )
        ret = await self._db.exec(stmt)
        return list(ret.all())

    async def find_latest_by_agent_id(self, agent_id: int) -> Optional[TestSuiteRun]:
        stmt = (
            select(TestSuiteRun)
            .where(TestSuiteRun.agent_id == agent_id)
            .order_by(col(TestSuiteRun.executed_at).desc())
            .limit(1)
        )
        ret = await self._db.exec(stmt)
        return ret.one_or_none()

    async def add(self, suite_run: TestSuiteRun) -> TestSuiteRun:
        self._db.add(suite_run)
        await self._db.commit()
        await self._db.refresh(suite_run)
        return suite_run

    async def save(self, suite_run: TestSuiteRun) -> TestSuiteRun:
        suite_run = await self._db.merge(suite_run)
        await self._db.commit()
        return suite_run

    async def delete(self, suite_run: TestSuiteRun) -> None:
        result_stmt = select(TestCaseResult).where(TestCaseResult.test_suite_run_id == suite_run.id)
        result = await self._db.exec(result_stmt)
        test_case_results = list(result.all())

        thread_repo = ThreadRepository(self._db)
        for test_case_result in test_case_results:
            if test_case_result.thread_id:
                execution_thread_stmt = select(Thread).where(Thread.id == test_case_result.thread_id)
                execution_thread_result = await self._db.exec(execution_thread_stmt)
                execution_thread = cast(Thread, execution_thread_result.one_or_none())
                await thread_repo.delete(execution_thread)

        delete_results_stmt = scalar(delete(TestCaseResult).where(and_(TestCaseResult.test_suite_run_id == suite_run.id)))
        await self._db.exec(delete_results_stmt)

        delete_suite_run_stmt = scalar(delete(TestSuiteRun).where(and_(TestSuiteRun.id == suite_run.id)))
        await self._db.exec(delete_suite_run_stmt)

        await self._db.commit()


class TestSuiteRunEventRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def add(self, event: TestSuiteRunEvent) -> TestSuiteRunEvent:
        self._db.add(event)
        await self._db.commit()
        await self._db.refresh(event)
        return event

    async def find_by_suite_run(self, suite_run_id: int, after_id: Optional[int] = None) -> List[TestSuiteRunEvent]:
        stmt = (
            select(TestSuiteRunEvent)
            .where(TestSuiteRunEvent.test_suite_run_id == suite_run_id)
        )
        if after_id:
            stmt = stmt.where(TestSuiteRunEvent.id > after_id)
        
        stmt = stmt.order_by(col(TestSuiteRunEvent.id).asc())
        ret = await self._db.exec(stmt)
        return list(ret.all())

    async def find_current_test_events(self, suite_run_id: int) -> List[TestSuiteRunEvent]:
        subq = (
            select(func.max(TestSuiteRunEvent.id))
            .where(TestSuiteRunEvent.test_suite_run_id == suite_run_id)
            .where(TestSuiteRunEvent.type == TestSuiteEventType.TEST_START.value)
        )
        last_start_id_result = await self._db.exec(subq)
        last_start_id = last_start_id_result.one_or_none()
        
        if not last_start_id:
            return []

        stmt = (
            select(TestSuiteRunEvent)
            .where(TestSuiteRunEvent.test_suite_run_id == suite_run_id)
            .where(TestSuiteRunEvent.id >= last_start_id)
            .order_by(col(TestSuiteRunEvent.id).asc())
        )
        ret = await self._db.exec(stmt)
        return list(ret.all())

    async def delete_by_suite_run_id(self, suite_run_id: int) -> None:
        stmt = scalar(delete(TestSuiteRunEvent).where(and_(TestSuiteRunEvent.test_suite_run_id == suite_run_id)))
        await self._db.exec(stmt)
        await self._db.commit()
