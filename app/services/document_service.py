import os
import uuid
from typing import List, Dict, Any
from pathlib import Path
import PyPDF2
import docx
from app.services.embedding_service import EmbeddingService
from app.core.vector_store import VectorStore
from app.core.config import settings
from app.db.session import SessionLocal
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from datetime import datetime
from functools import lru_cache

class DocumentService:
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.upload_dir = Path("uploads/documents")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
    
    async def process_document(
        self, 
        file_content: bytes, 
        filename: str, 
        uploaded_by: str,
        title: str = None
    ) -> Dict[str, Any]:
        """Process uploaded document"""
        
        # Save file
        file_path = self.upload_dir / f"{uuid.uuid4()}_{filename}"
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        # Extract text based on file type
        file_type = filename.split('.')[-1].lower()
        text = self._extract_text(str(file_path), file_type)
        
        if not text.strip():
            raise ValueError("No text could be extracted from the document")
        
        # Create document record in DB
        db = SessionLocal()
        try:
            document = Document(
                title=title or filename,
                filename=filename,
                file_type=file_type,
                file_path=str(file_path),
                uploaded_by=int(uploaded_by),
                status="processing"
            )
            db.add(document)
            db.commit()
            db.refresh(document)
            
            # Chunk the text
            chunks = self._chunk_text(text)
            
            # Generate embeddings
            embeddings = await self.embedding_service.embed_batch(chunks)
            
            # Store in vector DB
            metadata = [
                {
                    "document_id": str(document.id),
                    "chunk_index": i,
                    "content": chunk,
                    "filename": filename
                }
                for i, chunk in enumerate(chunks)
            ]
            
            vector_ids = await self.vector_store.add_vectors(embeddings, metadata)
            
            # Save chunks to DB
            for i, (chunk, vector_id) in enumerate(zip(chunks, vector_ids)):
                chunk_record = DocumentChunk(
                    document_id=document.id,
                    chunk_index=i,
                    content=chunk,
                    vector_id=vector_id
                )
                db.add(chunk_record)
            
            # Update document status
            document.status = "completed"
            document.chunk_count = len(chunks)
            db.commit()
            
            return {
                "document_id": str(document.id),
                "chunks": len(chunks),
                "status": "completed",
                "message": f"Document processed successfully"
            }
            
        except Exception as e:
            if 'document' in locals() and document is not None:
                document.status = "failed"
                db.commit()
            raise e
        finally:
            db.close()
    
    def _extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text from different file types"""
        
        if file_type == "pdf":
            return self._extract_pdf(file_path)
        elif file_type == "docx":
            return self._extract_docx(file_path)
        elif file_type in ["txt", "md"]:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    
    def _extract_pdf(self, file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def _extract_docx(self, file_path: str) -> str:
        """Extract text from DOCX"""
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        chunk_size = settings.CHUNK_SIZE
        overlap = settings.CHUNK_OVERLAP
        
        # Split by sentences or paragraphs
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                current_chunk = current_chunk[-overlap:] + sentence + ". " if overlap > 0 else sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def delete_document(self, document_id: str):
        """Delete document and its vectors"""
        db = SessionLocal()
        try:
            # Get document
            document = db.query(Document).filter(Document.id == uuid.UUID(document_id)).first()
            if not document:
                raise ValueError("Document not found")
            
            # Get all chunks
            chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == document.id).all()
            
            # Delete from vector DB
            vector_ids = [chunk.vector_id for chunk in chunks]
            if vector_ids:
                await self.vector_store.delete_vectors(vector_ids)
            
            # Delete chunks from DB
            for chunk in chunks:
                db.delete(chunk)
            
            # Delete document from DB
            db.delete(document)
            db.commit()
            
            # Delete physical file
            if os.path.exists(document.file_path):
                os.remove(document.file_path)
            
            return {"message": "Document deleted successfully"}
            
        finally:
            db.close()
    
    async def get_all_documents(self, skip: int = 0, limit: int = 100) -> List[Dict]:
        """Get all documents"""
        db = SessionLocal()
        try:
            documents = db.query(Document).offset(skip).limit(limit).all()
            return [
                {
                    "id": str(doc.id),
                    "title": doc.title,
                    "filename": doc.filename,
                    "file_type": doc.file_type,
                    "status": doc.status,
                    "chunk_count": doc.chunk_count,
                    "uploaded_at": doc.uploaded_at.isoformat(),
                    "is_active": doc.is_active
                }
                for doc in documents
            ]
        finally:
            db.close()

    @lru_cache()
    def get_document_service():
        return DocumentService()