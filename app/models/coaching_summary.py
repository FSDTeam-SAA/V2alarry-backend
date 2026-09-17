import uuid

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, UUID
from sqlalchemy.sql import func

from app.db.base import Base


class CoachingSummary(Base):
    __tablename__ = "coaching_summaries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    presenting_focus = Column(Text, nullable=True)
    primary_discovery = Column(Text, nullable=True)
    developmental_theme = Column(Text, nullable=True)
    commitment = Column(Text, nullable=True)
    next_experiment = Column(Text, nullable=True)
    follow_up_question = Column(Text, nullable=True)
    coach_notes = Column(Text, nullable=True)
    prior_session_continuity = Column(Text, nullable=True)
    source_turn_count = Column(Integer, nullable=False, default=0)
    generation_status = Column(String(20), nullable=False, default="current")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
