from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.repositories.base import BaseRepository


class UserRepository(BaseRepository[User]):
    # def __init__(self. session: AsyncSession):
    #     super().__init__(User, session)
        
    async def get_all_email(self,email: str) -> User | None:
        query = select(User).where(User.email == email)
        result = await self.session.execute(query)
        return result.scalar_one_or_none()
    
    
    async def get_active_users(
        self, offset: int = 0, limit: int = 100
         
    ) ->list[User]:
        query = (
            select(User)
            .where(User.is_active.is_(True))
            .offset(offset)
            .limit(limit)
            .order_by(User.created_at.desc())
        )
        
        result = await self.session.execute(query)
        return list(result.scalars().all())