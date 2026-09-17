from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

class DocumentBase(BaseModel):
    title: str
    filename: str
    file_type: str

class DocumentResponse(DocumentBase):
    id: str
    status: str
    chunk_count: int
    uploaded_at: datetime
    is_active: bool
    category: str
    file_size_bytes: int
    is_global: bool
    target_user_id: int | None = None
    target_user_email: str | None = None

class DocumentUploadResponse(BaseModel):
    document_id: str
    chunks: int
    status: str
    message: str
    is_global: bool
    target_user_id: int | None = None
    target_user_email: str | None = None
    category: str
    file_size_bytes: int

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
