import json
import logging
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_validator

from app.api.dependencies.auth import require_current_agreement
from app.models.user import User
from app.services.chat_history_service import ChatHistoryService
from app.workflows.chat_workflow import ChatWorkflow, get_chat_workflow

router = APIRouter(prefix="/chat", tags=["User Chat"])
logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    stream: bool = False

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        message = value.strip()
        if not message:
            raise ValueError("message must not be blank")
        return message

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
    current_user: User = Depends(require_current_agreement),
):
    workflow = get_chat_workflow()

    if request.stream:
        return StreamingResponse(
            _stream_response(workflow, current_user, request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    try:
        result = await workflow.process_message(
            user_id=str(current_user.id),
            message=request.message,
            conversation_id=request.conversation_id,
        )

        return ChatResponse(
            response=result["response"],
            conversation_id=result["conversation_id"],
            message_id=result.get("metadata", {}).get("message_id", ""),
            metadata=result.get("metadata", {}),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Chat request failed for user %s: %s", current_user.id, e)
        raise HTTPException(
            status_code=500,
            detail="The chat service is temporarily unavailable. Please try again.",
        )


async def _stream_response(
    workflow: ChatWorkflow,
    current_user: User,
    request: ChatRequest,
):
    try:
        async for event in workflow.process_message_stream(
            user_id=str(current_user.id),
            message=request.message,
            conversation_id=request.conversation_id,
        ):
            yield f"data: {event}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.exception(
            "Chat stream failed while streaming a response for user %s in conversation %s: %s",
            current_user.id,
            request.conversation_id or "new",
            e,
        )
        yield (
            "data: "
            f"{json.dumps({'type': 'error', 'content': 'The chat service is temporarily unavailable. Please try again.'})}"
            "\n\n"
        )
        yield "data: [DONE]\n\n"


@router.get("/conversations", response_model=List[ConversationResponse])
async def get_conversations(
    current_user: User = Depends(require_current_agreement),
    chat_history_service: ChatHistoryService = Depends(),
):
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
                message_count=conv["message_count"],
            )
            for conv in conversations
        ]
    except Exception:
        logger.warning("Conversation listing failed for user %s", current_user.id)
        raise HTTPException(status_code=500, detail="Unable to fetch conversations")


@router.get("/conversations/{conversation_id}/messages", response_model=List[MessageResponse])
async def get_conversation_messages(
    conversation_id: str,
    current_user: User = Depends(require_current_agreement),
    chat_history_service: ChatHistoryService = Depends(),
):
    try:
        messages = await chat_history_service.get_conversation_messages(
            conversation_id=conversation_id,
            user_id=str(current_user.id),
        )
        return messages
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        logger.warning("Message listing failed for user %s", current_user.id)
        raise HTTPException(status_code=500, detail="Unable to fetch messages")


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: User = Depends(require_current_agreement),
    chat_history_service: ChatHistoryService = Depends(),
):
    try:
        deleted = await chat_history_service.delete_conversation(
            conversation_id=conversation_id,
            user_id=str(current_user.id),
        )

        if not deleted:
            raise HTTPException(status_code=404, detail="Conversation not found")

        return {"message": "Conversation deleted successfully"}
    except ValueError:
        raise HTTPException(status_code=400, detail="conversation_id must be a valid UUID")
    except HTTPException:
        raise
    except Exception:
        logger.warning("Conversation deletion failed for user %s", current_user.id)
        raise HTTPException(status_code=500, detail="Unable to delete conversation")
