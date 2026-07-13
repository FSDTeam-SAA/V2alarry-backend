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

class DocumentUploadResponse(BaseModel):
    document_id: str
    chunks: int
    status: str
    message: str

class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int
