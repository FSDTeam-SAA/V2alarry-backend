from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from typing import List, Optional
from app.services.document_service import DocumentService
from app.api.dependencies.auth import require_admin
from app.models.user import User
from app.db.session import get_db
from app.repositories.user_repository import UserRepository
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from app.schemas.document import DocumentResponse, DocumentUploadResponse, DocumentListResponse
import uuid

router = APIRouter(prefix="/admin/documents", tags=["Admin Documents"])
user_repo = UserRepository()
logger = logging.getLogger(__name__)

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(default=None, max_length=255),
    category: str = Form(default="general", min_length=1, max_length=100),
    is_global: bool = Form(default=True),
    target_user_email: Optional[str] = Form(default=None),
    current_user: User = Depends(require_admin),
    document_service: DocumentService = Depends(),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a document to the knowledge base
    - Supports: PDF, DOCX, TXT, MD
    - Max file size: 10MB
    """
    # Validate file type
    allowed_types = ["pdf", "docx", "txt", "md"]
    if not file.filename:
        raise HTTPException(status_code=400, detail="A filename is required")
    safe_filename = file.filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    file_type = safe_filename.split('.')[-1].lower()
    
    if file_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Validate file size (10MB)
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")

    target_user = None
    normalized_target_email = target_user_email.strip().lower() if target_user_email else None
    if is_global and normalized_target_email:
        raise HTTPException(
            status_code=400,
            detail="Global documents cannot target a specific user",
        )
    if not is_global:
        if not normalized_target_email:
            raise HTTPException(
                status_code=400,
                detail="A target user email is required for user-specific documents",
            )
        target_user = await user_repo.get_by_email(db, normalized_target_email)
        if not target_user:
            raise HTTPException(status_code=400, detail="Target user was not found")
    
    try:
        result = await document_service.process_document(
            file_content=file_content,
            filename=safe_filename,
            uploaded_by=str(current_user.id),
            title=title,
            category=category.strip(),
            is_global=is_global,
            target_user_id=target_user.id if target_user else None,
            target_user_email=target_user.email if target_user else None,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.warning("Document processing failed for admin user %s", current_user.id)
        raise HTTPException(status_code=500, detail="Unable to process document")

@router.get("/", response_model=List[DocumentResponse])
async def get_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(require_admin),
    document_service: DocumentService = Depends()
):
    """Get all uploaded documents"""
    documents = await document_service.get_all_documents(skip=skip, limit=limit)
    return documents

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(require_admin),
    document_service: DocumentService = Depends()
):
    """Delete a document and its vectors"""
    try:
        result = await document_service.delete_document(document_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception:
        logger.warning("Document deletion failed for admin user %s", current_user.id)
        raise HTTPException(status_code=500, detail="Unable to delete document")

@router.get("/stats")
async def get_document_stats(
    current_user: User = Depends(require_admin),
    document_service: DocumentService = Depends()
):
    """Get document statistics"""
    documents = await document_service.get_all_documents()
    return {
        "total_documents": len(documents),
        "by_status": {
            "completed": len([d for d in documents if d["status"] == "completed"]),
            "processing": len([d for d in documents if d["status"] == "processing"]),
            "failed": len([d for d in documents if d["status"] == "failed"])
        },
        "by_type": {
            doc["file_type"]: len([d for d in documents if d["file_type"] == doc["file_type"]])
            for doc in documents
        }
    }
