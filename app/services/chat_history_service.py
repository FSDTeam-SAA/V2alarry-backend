from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import AsyncSessionLocal
from app.models.conversation import Conversation
from app.models.message import Message
from typing import List, Dict, Any, Optional
import uuid
from functools import lru_cache


class ChatHistoryService:
    def __init__(self):
        pass

    @staticmethod
    def _parse_conversation_id(conversation_id: Optional[str]) -> Optional[uuid.UUID]:
        if conversation_id is None:
            return None
        normalized_id = conversation_id.strip() if isinstance(conversation_id, str) else str(conversation_id)
        if not normalized_id:
            return None
        try:
            return uuid.UUID(normalized_id)
        except (ValueError, TypeError) as exc:
            raise ValueError("conversation_id must be a valid UUID") from exc

    async def get_or_create_conversation(
        self,
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        parsed_conversation_id = self._parse_conversation_id(conversation_id)
        db = AsyncSessionLocal()
        try:
            if parsed_conversation_id:
                result = await db.execute(
                    select(Conversation).where(
                        Conversation.id == parsed_conversation_id,
                        Conversation.user_id == int(user_id),
                    )
                )
                conv = result.scalar_one_or_none()
                if conv:
                    return {
                        "id": str(conv.id),
                        "title": conv.title,
                        "created_at": conv.created_at.isoformat(),
                    }

            conv = Conversation(
                user_id=int(user_id),
                title="New Conversation",
            )
            db.add(conv)
            await db.commit()
            await db.refresh(conv)

            return {
                "id": str(conv.id),
                "title": conv.title,
                "created_at": conv.created_at.isoformat(),
            }
        finally:
            await db.close()

    async def save_conversation_turn(
        self,
        user_id: str,
        conversation_id: Optional[str],
        user_message: str,
        assistant_response: str,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        db = AsyncSessionLocal()
        try:
            conv = await self.get_or_create_conversation(user_id, conversation_id)
            conv_uuid = uuid.UUID(conv["id"])

            user_msg = Message(
                conversation_id=conv_uuid,
                role="user",
                content=user_message,
                metadata=metadata or {},
            )
            db.add(user_msg)

            assistant_msg = Message(
                conversation_id=conv_uuid,
                role="assistant",
                content=assistant_response,
                metadata={
                    "retrieved_docs": (metadata or {}).get("docs_retrieved", 0),
                    "context_length": (metadata or {}).get("context_length", 0),
                },
            )
            db.add(assistant_msg)

            if conv["title"] == "New Conversation":
                result = await db.execute(
                    select(Conversation).where(Conversation.id == conv_uuid)
                )
                conv_obj = result.scalar_one_or_none()
                if conv_obj:
                    from app.core.config import settings
                    title = user_message[:settings.TITLE_TRUNCATE_LENGTH] + (
                        "..." if len(user_message) > settings.TITLE_TRUNCATE_LENGTH else ""
                    )
                    conv_obj.title = title

            await db.commit()
            await db.refresh(user_msg)

            return {
                "id": str(user_msg.id),
                "conversation_id": str(conv_uuid),
                "role": user_msg.role,
                "content": user_msg.content,
                "created_at": user_msg.created_at.isoformat(),
            }
        finally:
            await db.close()

    async def get_conversation_messages(
        self,
        conversation_id: str,
        user_id: str,
    ) -> List[Dict]:
        parsed_conversation_id = self._parse_conversation_id(conversation_id)
        if parsed_conversation_id is None:
            raise ValueError("conversation_id must be a valid UUID")

        db = AsyncSessionLocal()
        try:
            result = await db.execute(
                select(Message)
                .join(Conversation, Conversation.id == Message.conversation_id)
                .where(
                    Message.conversation_id == parsed_conversation_id,
                    Conversation.user_id == int(user_id),
                )
                .order_by(Message.created_at.asc())
            )
            messages = result.scalars().all()

            return [
                {
                    "id": str(msg.id),
                    "role": msg.role,
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat(),
                }
                for msg in messages
            ]
        finally:
            await db.close()

    async def get_conversation_history(
        self,
        conversation_id: str,
        user_id: str,
        limit: int = 10,
    ) -> List[Dict]:
        parsed_conversation_id = self._parse_conversation_id(conversation_id)
        if parsed_conversation_id is None:
            return []

        db = AsyncSessionLocal()
        try:
            result = await db.execute(
                select(Message)
                .join(Conversation, Conversation.id == Message.conversation_id)
                .where(
                    Message.conversation_id == parsed_conversation_id,
                    Conversation.user_id == int(user_id),
                )
                .order_by(Message.created_at.desc())
                .limit(limit)
            )
            messages = list(result.scalars().all())
            messages.reverse()

            return [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "conversation_id": str(msg.conversation_id),
                    "created_at": msg.created_at.isoformat(),
                }
                for msg in messages
            ]
        finally:
            await db.close()

    async def get_user_conversations(self, user_id: str) -> List[Dict[str, Any]]:
        db = AsyncSessionLocal()
        try:
            result = await db.execute(
                select(
                    Conversation,
                    func.count(Message.id).label("message_count"),
                )
                .outerjoin(Message, Message.conversation_id == Conversation.id)
                .where(Conversation.user_id == int(user_id))
                .group_by(Conversation.id)
                .order_by(Conversation.updated_at.desc())
            )
            rows = result.all()

            return [
                {
                    "id": str(conv.id),
                    "title": conv.title,
                    "created_at": conv.created_at.isoformat(),
                    "updated_at": conv.updated_at.isoformat(),
                    "message_count": count or 0,
                }
                for conv, count in rows
            ]
        finally:
            await db.close()

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        parsed_conversation_id = self._parse_conversation_id(conversation_id)
        if parsed_conversation_id is None:
            raise ValueError("conversation_id must be a valid UUID")

        db = AsyncSessionLocal()
        try:
            result = await db.execute(
                select(Conversation).where(
                    Conversation.id == parsed_conversation_id,
                    Conversation.user_id == int(user_id),
                )
            )
            conversation = result.scalar_one_or_none()

            if not conversation:
                return False

            await db.execute(
                delete(Message).where(Message.conversation_id == conversation.id)
            )
            await db.delete(conversation)
            await db.commit()
            return True
        finally:
            await db.close()

    @lru_cache()
    def get_chat_history_service():
        return ChatHistoryService()
