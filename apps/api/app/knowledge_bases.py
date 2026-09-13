from datetime import UTC, datetime
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator
from sqlalchemy import DateTime, String, desc, select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Mapped, Session, mapped_column

from app.database import Base, get_session

MAX_NAME_LENGTH = 100


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id: Mapped[UUID] = mapped_column(primary_key=True, default=uuid4)
    name: Mapped[str] = mapped_column(String(MAX_NAME_LENGTH), nullable=False)
    normalized_name: Mapped[str] = mapped_column(
        String(MAX_NAME_LENGTH * 2), nullable=False, unique=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=lambda: datetime.now(UTC)
    )


class KnowledgeBaseCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("Name cannot be empty or only whitespace.")
        if len(name) > MAX_NAME_LENGTH:
            raise ValueError(f"Name must be at most {MAX_NAME_LENGTH} characters.")
        return name


class KnowledgeBaseResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    created_at: datetime

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


router = APIRouter(prefix="/knowledge-bases", tags=["knowledge bases"])


@router.post("", response_model=KnowledgeBaseResponse, status_code=status.HTTP_201_CREATED)
def create_knowledge_base(
    payload: KnowledgeBaseCreate, session: Session = Depends(get_session)
) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(name=payload.name, normalized_name=payload.name.casefold())
    session.add(knowledge_base)

    try:
        session.commit()
        session.refresh(knowledge_base)
    except IntegrityError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f'A knowledge base named "{payload.name}" already exists.',
        ) from None
    except SQLAlchemyError:
        session.rollback()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base storage is unavailable.",
        ) from None

    return knowledge_base


@router.get("", response_model=list[KnowledgeBaseResponse])
def list_knowledge_bases(session: Session = Depends(get_session)) -> list[KnowledgeBase]:
    try:
        statement = select(KnowledgeBase).order_by(
            desc(KnowledgeBase.created_at), desc(KnowledgeBase.id)
        )
        return list(session.scalars(statement))
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base storage is unavailable.",
        ) from None
