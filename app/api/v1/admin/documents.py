from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from typing import List, Optional
from app.services.document_service import DocumentService
from app.api.dependencies.auth import get_current_user
from app.models.user import User
from app.schemas.document import DocumentResponse, DocumentUploadResponse, DocumentListResponse
import uuid

router = APIRouter(prefix="/admin/documents", tags=["Admin Documents"])

async def get_current_admin(current_user: User = Depends(get_current_user)):
    """Check if user is admin"""
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = None,
    current_user: User = Depends(get_current_admin),
    document_service: DocumentService = Depends()
):
    """
    Upload a document to the knowledge base
    - Supports: PDF, DOCX, TXT, MD
    - Max file size: 10MB
    """
    # Validate file type
    allowed_types = ["pdf", "docx", "txt", "md"]
    file_type = file.filename.split('.')[-1].lower()
    
    if file_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_types)}"
        )
    
    # Validate file size (10MB)
    file_content = await file.read()
    if len(file_content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
    
    try:
        result = await document_service.process_document(
            file_content=file_content,
            filename=file.filename,
            uploaded_by=str(current_user.id),
            title=title
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/", response_model=List[DocumentResponse])
async def get_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(get_current_admin),
    document_service: DocumentService = Depends()
):
    """Get all uploaded documents"""
    documents = await document_service.get_all_documents(skip=skip, limit=limit)
    return documents

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_admin),
    document_service: DocumentService = Depends()
):
    """Delete a document and its vectors"""
    try:
        result = await document_service.delete_document(document_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.get("/stats")
async def get_document_stats(
    current_user: User = Depends(get_current_admin),
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
