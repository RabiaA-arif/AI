from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession, create_async_engine, async_sessionmaker
    
)

"""
The @asynccontextmanager decorator is a powerful tool in FastAPI 
for managing resources that need setup and cleanup, such as database connections. It allows 
you to define dependencies that yield a resource and ensure proper cleanup after use.
"""

class DatabaseSessionManager:
    """ manage async database with proper lifecycle: 
    """
    def __init__(self):
        self._engine = None
        self._sessionmaker = None
        
    async def init(self, database_url: str):
        self._engine = create_async_engine(
            database_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
            echo=False,
        )
        
        self._sessionmaker = async_sessionmaker(
            bind=self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    
    async def close(self):
        if self._engine:
            await self._engine.dispose()
        
    @asynccontextmanager
    async def session(self):
        if self._sessionmaker is None:
            raise RuntimeError("DatabaseSessionManager is not initialized")
        
        async with self._sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            
sessionmanager = DatabaseSessionManager()


async def get_db_session():
    """FASTAPI dependency for database session"""
    async with sessionmanager.session() as session:
        yield session
        