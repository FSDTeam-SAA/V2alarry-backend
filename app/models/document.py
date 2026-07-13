from sqlalchemy import Column, String, DateTime, Boolean, Integer, JSON, UUID
from sqlalchemy.sql import func
from app.db.base import Base
import uuid

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    file_type = Column(String(50), nullable=False)  # pdf, docx, txt
    file_path = Column(String(500), nullable=False)
    uploaded_by = Column(UUID(as_uuid=True), nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(String(50), default="processing")  # processing, completed, failed
    chunk_count = Column(Integer, default=0)
    doc_metadata = Column("metadata", JSON, default={})
    is_active = Column(Boolean, default=True)