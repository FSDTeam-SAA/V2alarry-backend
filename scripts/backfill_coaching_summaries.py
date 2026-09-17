"""Operator-run, resumable coaching-memory backfill.

Usage:
    .venv/bin/python -m scripts.backfill_coaching_summaries --limit 25
    .venv/bin/python -m scripts.backfill_coaching_summaries --after <conversation-uuid> --limit 25
"""

import argparse
import asyncio
import uuid

from sqlalchemy import select

from app.db.database import AsyncSessionLocal
from app.models.coaching_summary import CoachingSummary
from app.models.coaching_working_state import CoachingWorkingState as WorkingStateRecord
from app.models.conversation import Conversation
from app.models.message import Message
from app.schemas.coaching import CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService


async def backfill(*, after: uuid.UUID | None, limit: int) -> None:
    memory = CoachingMemoryService()
    async with AsyncSessionLocal() as db:
        query = select(Conversation).order_by(Conversation.id).limit(limit)
        if after:
            query = query.where(Conversation.id > after)
        conversations = list((await db.execute(query)).scalars().all())

    for conversation in conversations:
        async with AsyncSessionLocal() as db:
            existing = await db.scalar(
                select(CoachingSummary).where(
                    CoachingSummary.conversation_id == conversation.id,
                    CoachingSummary.generation_status == "current",
                )
            )
            if existing:
                print(f"skip conversation={conversation.id} status=current")
                continue

            messages = list(
                (
                    await db.execute(
                        select(Message)
                        .where(Message.conversation_id == conversation.id)
                        .order_by(Message.created_at.asc())
                    )
                ).scalars().all()
            )
            user_messages = [message for message in messages if message.role == "user"]
            if not user_messages:
                print(f"skip conversation={conversation.id} reason=no-user-turns")
                continue

            state = CoachingWorkingState()
            try:
                for message in user_messages:
                    state = await memory.update_working_state(
                        previous_state=state,
                        user_message=message.content,
                    )
                latest_assistant = next(
                    (message.content for message in reversed(messages) if message.role == "assistant"),
                    "",
                )
                summary = await memory.refresh_summary(
                    previous_summary=None,
                    working_state=state,
                    user_message=user_messages[-1].content,
                    assistant_response=latest_assistant,
                )

                state_record = await db.scalar(
                    select(WorkingStateRecord).where(
                        WorkingStateRecord.conversation_id == conversation.id
                    )
                )
                if state_record:
                    state_record.state = state.model_dump()
                    state_record.source_turn_count = len(user_messages)
                else:
                    db.add(
                        WorkingStateRecord(
                            conversation_id=conversation.id,
                            state=state.model_dump(),
                            source_turn_count=len(user_messages),
                        )
                    )

                summary_record = await db.scalar(
                    select(CoachingSummary).where(
                        CoachingSummary.conversation_id == conversation.id
                    )
                )
                if summary_record is None:
                    summary_record = CoachingSummary(
                        conversation_id=conversation.id,
                        user_id=conversation.user_id,
                    )
                    db.add(summary_record)
                for field, value in summary.model_dump().items():
                    setattr(summary_record, field, value)
                summary_record.source_turn_count = len(user_messages)
                summary_record.generation_status = "current"
                await db.commit()
                print(f"updated conversation={conversation.id} turns={len(user_messages)}")
            except Exception:
                await db.rollback()
                print(f"failed conversation={conversation.id}")

    if conversations:
        print(f"resume_after={conversations[-1].id}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--after", type=uuid.UUID)
    parser.add_argument("--limit", type=int, default=25)
    args = parser.parse_args()
    if args.limit < 1 or args.limit > 250:
        parser.error("--limit must be between 1 and 250")
    asyncio.run(backfill(after=args.after, limit=args.limit))


if __name__ == "__main__":
    main()
