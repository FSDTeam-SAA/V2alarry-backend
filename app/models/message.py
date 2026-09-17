# app/models/message.py
from sqlalchemy import Column, String, DateTime, Text, JSON, UUID, ForeignKey
from sqlalchemy.sql import func
from app.db.base import Base
import uuid

class Message(Base):
    __tablename__ = "messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )
    role = Column(String(50), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    doc_metadata = Column("metadata", JSON, default={})
