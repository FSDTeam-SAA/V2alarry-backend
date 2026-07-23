from fastapi import APIRouter, Depends, HTTPException
from typing import Optional, List
from app.services.chat_history_service import ChatHistoryService
from app.workflows.chat_workflow import ChatWorkflow
from app.api.dependencies.auth import get_current_user
from app.models.user import User
from pydantic import BaseModel, field_validator
import uuid

router = APIRouter(prefix="/chat", tags=["User Chat"])

# Schemas
class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None

    @field_validator("conversation_id", mode="before")
    @classmethod
    def validate_conversation_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        try:
            return str(uuid.UUID(str(value)))
        except (ValueError, TypeError) as exc:
            raise ValueError("conversation_id must be a valid UUID") from exc

class ChatResponse(BaseModel):
    response: str
    conversation_id: str
    message_id: str
    metadata: dict

class ConversationResponse(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int

class MessageResponse(BaseModel):
    id: str
    role: str
    content: str
    created_at: str
    metadata: Optional[dict] = None

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    chat_workflow: ChatWorkflow = Depends()
):
    """
    Send a message and get AI response
    - Automatically creates new conversation if no ID provided
    - Uses user's history for personalization
    - Retrieves relevant documents from knowledge base
    """
    try:
        # Process message through workflow
        result = await chat_workflow.process_message(
            user_id=str(current_user.id),
            message=request.message,
            conversation_id=request.conversation_id
        )
        
        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            message_id=result.get("message_id", ""),
            metadata=result.get("metadata", {})
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    current_user: User = Depends(get_current_user),
    chat_history_service: ChatHistoryService = Depends()
):
    """Get all conversations for the current user"""
    try:
        conversations = await chat_history_service.get_user_conversations(
            user_id=str(current_user.id)
        )

        return [
            ConversationResponse(
                id=conv["id"],
                title=conv["title"],
                created_at=conv["created_at"],
                updated_at=conv["updated_at"],
                message_count=conv["message_count"]
            )
            for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")

@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    chat_history_service: ChatHistoryService = Depends()
):
    """Get all messages in a conversation"""
    try:
        messages = await chat_history_service.get_conversation_messages(
            conversation_id=conversation_id,
            user_id=str(current_user.id)
        )
        return messages
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching messages: {str(e)}")

@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(get_current_user),
    chat_history_service: ChatHistoryService = Depends()
):
    """Delete a conversation and all its messages"""
    try:
        deleted = await chat_history_service.delete_conversation(
            conversation_id=conversation_id,
            user_id=str(current_user.id)
        )

        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"message": "Conversation deleted successfully"}
    except ValueError:
        raise HTTPException(status_code=400, detail="conversation_id must be a valid UUID")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting conversation: {str(e)}")
