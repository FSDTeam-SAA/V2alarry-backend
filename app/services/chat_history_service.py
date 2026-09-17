from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import AsyncSessionLocal
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.coaching_summary import CoachingSummary
from app.models.coaching_working_state import CoachingWorkingState
from app.schemas.coaching import CoachingSummaryData
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
                raise ValueError("Conversation not found")

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
        working_state: Optional[Dict[str, Any]] = None,
        source_turn_count: int = 0,
    ) -> Dict[str, Any]:
        db = AsyncSessionLocal()
        try:
            parsed_conversation_id = self._parse_conversation_id(conversation_id)
            conv_obj = None
            if parsed_conversation_id:
                result = await db.execute(
                    select(Conversation).where(
                        Conversation.id == parsed_conversation_id,
                        Conversation.user_id == int(user_id),
                    )
                )
                conv_obj = result.scalar_one_or_none()
                if not conv_obj:
                    raise ValueError("Conversation not found")
            else:
                conv_obj = Conversation(
                    user_id=int(user_id),
                    title="New Conversation",
                )
                db.add(conv_obj)
                await db.flush()

            conv_uuid = conv_obj.id
            conv_obj.updated_at = func.now()

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

            if conv_obj.title == "New Conversation":
                from app.core.config import settings
                title = user_message[:settings.TITLE_TRUNCATE_LENGTH] + (
                    "..." if len(user_message) > settings.TITLE_TRUNCATE_LENGTH else ""
                )
                conv_obj.title = title

            if working_state is not None:
                state_result = await db.execute(
                    select(CoachingWorkingState).where(
                        CoachingWorkingState.conversation_id == conv_uuid
                    )
                )
                state_record = state_result.scalar_one_or_none()
                if state_record:
                    state_record.state = working_state
                    state_record.source_turn_count = source_turn_count
                else:
                    db.add(
                        CoachingWorkingState(
                            conversation_id=conv_uuid,
                            state=working_state,
                            source_turn_count=source_turn_count,
                        )
                    )

            await db.commit()
            await db.refresh(user_msg)
            await db.refresh(assistant_msg)

            return {
                "id": str(user_msg.id),
                "conversation_id": str(conv_uuid),
                "user_message_id": str(user_msg.id),
                "assistant_message_id": str(assistant_msg.id),
                "role": user_msg.role,
                "content": user_msg.content,
                "created_at": user_msg.created_at.isoformat(),
            }
        finally:
            await db.close()

    async def get_working_state(
        self,
        *,
        conversation_id: str,
        user_id: str,
    ) -> CoachingWorkingState | None:
        parsed_conversation_id = self._parse_conversation_id(conversation_id)
        if parsed_conversation_id is None:
            return None

        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(CoachingWorkingState)
                .join(
                    Conversation,
                    Conversation.id == CoachingWorkingState.conversation_id,
                )
                .where(
                    CoachingWorkingState.conversation_id == parsed_conversation_id,
                    Conversation.user_id == int(user_id),
                )
            )
            return result.scalar_one_or_none()

    async def get_latest_valid_summaries(
        self,
        *,
        user_id: str,
        limit: int = 3,
    ) -> List[CoachingSummary]:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(CoachingSummary)
                .where(
                    CoachingSummary.user_id == int(user_id),
                    CoachingSummary.generation_status == "current",
                )
                .order_by(CoachingSummary.updated_at.desc())
                .limit(limit)
            )
            return list(result.scalars().all())

    async def get_summary(
        self,
        *,
        conversation_id: str,
        user_id: str,
    ) -> CoachingSummary | None:
        parsed_conversation_id = self._parse_conversation_id(conversation_id)
        if parsed_conversation_id is None:
            return None
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(CoachingSummary).where(
                    CoachingSummary.conversation_id == parsed_conversation_id,
                    CoachingSummary.user_id == int(user_id),
                )
            )
            return result.scalar_one_or_none()

    async def save_summary(
        self,
        *,
        conversation_id: str,
        user_id: str,
        summary: CoachingSummaryData,
        source_turn_count: int,
    ) -> None:
        parsed_conversation_id = self._parse_conversation_id(conversation_id)
        if parsed_conversation_id is None:
            raise ValueError("conversation_id must be a valid UUID")
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(CoachingSummary).where(
                    CoachingSummary.conversation_id == parsed_conversation_id,
                    CoachingSummary.user_id == int(user_id),
                )
            )
            record = result.scalar_one_or_none()
            values = summary.model_dump()
            if record:
                for field, value in values.items():
                    setattr(record, field, value)
                record.source_turn_count = source_turn_count
                record.generation_status = "current"
            else:
                db.add(
                    CoachingSummary(
                        conversation_id=parsed_conversation_id,
                        user_id=int(user_id),
                        source_turn_count=source_turn_count,
                        generation_status="current",
                        **values,
                    )
                )
            await db.commit()

    async def mark_summary_stale(
        self,
        *,
        conversation_id: str,
        user_id: str,
        source_turn_count: int = 0,
    ) -> None:
        parsed_conversation_id = self._parse_conversation_id(conversation_id)
        if parsed_conversation_id is None:
            return
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(CoachingSummary).where(
                    CoachingSummary.conversation_id == parsed_conversation_id,
                    CoachingSummary.user_id == int(user_id),
                )
            )
            record = result.scalar_one_or_none()
            if record:
                record.generation_status = "stale"
            else:
                db.add(
                    CoachingSummary(
                        conversation_id=parsed_conversation_id,
                        user_id=int(user_id),
                        source_turn_count=source_turn_count,
                        generation_status="stale",
                    )
                )
            await db.commit()

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

            await db.delete(conversation)
            await db.commit()
            return True
        finally:
            await db.close()

    @lru_cache()
    def get_chat_history_service():
        return ChatHistoryService()
