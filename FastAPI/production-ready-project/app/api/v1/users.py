from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db_session
from app.schemas.user import UserCreate, UserResponce
from app.service.user_services import UserServices


router = APIRouter(prefix="/users", tags=["users"])


def get_user_service(
    session: AsyncSession = Depends(get_db_session),
) -> UserServices:
    return


@router.post("/", responce_model=UserResponce, status_code=201)
async def create_user(
    data: UserCreate,
    service: UserServices = Depends(get_user_service),
    
):
    return await service.create_user(data)

@router.get("/{user_id}", response_model=UserResponce)
async def get_user(
    user_id: int,
    service: UserServices = Depends(get_user_service),
): 
    return await service.get_user(user_id)


@router.get("/", response_model=list[UserResponce])
async def list_users(
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    service: UserResponce = Depends(get_user_service),
):
    return await service.list_users(offset, limit)