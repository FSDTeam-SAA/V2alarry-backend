import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, JSON, UUID
from sqlalchemy.sql import func

from app.db.base import Base


class CoachingWorkingState(Base):
    __tablename__ = "coaching_working_states"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    state = Column(JSON, nullable=False, default=dict)
    source_turn_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
