from fastapi import HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.repositories.user_repo import UserRepository
# from app.schemas.user import UserCreate, UserUpdate,UserResponce
# from app.core.security import hash_password, verify_password




class UserServices:
    def __init__(self, session: AsyncSession):
        self.repo = UserRepository(session)
        
    
    async def create_user(self, data: UserCreate) -> UserResponce:
        # Business rule: check for duplicate email
        existing = await self.repo.get_by_id(data.email)
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Email already registered",
            )
        user = await self.repo.create(
            email=data.email,
            name=data.name,
            hashed_password=hash_password(data.password),
        )
        return UserResponce.model_validate(user)
    
    async def get_user(self, user_id: int) -> UserResponce:
        user = await self.repo.get_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        return UserResponce.model_validate(user)
    
    async def list_users(
        self, offset: int = 0, limit: int = 100
    ) -> list[UserResponce]:
        users = await self.repo.get_active_users(offset, limit)
        return [UserResponce.model_validate(u) for u in users]