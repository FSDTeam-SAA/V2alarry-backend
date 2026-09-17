from sqlalchemy import Column, String, DateTime, Boolean, Integer, JSON, UUID, ForeignKey
from sqlalchemy.sql import func
from app.db.base import Base
import uuid

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    category = Column(String(100), nullable=False, default="general")
    file_size_bytes = Column(Integer, nullable=False, default=0)
    is_global = Column(Boolean, nullable=False, default=True)
    target_user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    uploaded_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default="processing")
    chunk_count = Column(Integer, default=0)
    doc_metadata = Column("metadata", JSON, default={})
    is_active = Column(Boolean, default=True)
