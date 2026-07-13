from sqlalchemy import Column, String, DateTime, Integer, Text, UUID
from sqlalchemy.sql import func
from app.db.base import Base
import uuid

class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    vector_id = Column(String(255), nullable=False)  # ID in Qdrant
    created_at = Column(DateTime(timezone=True), server_default=func.now())