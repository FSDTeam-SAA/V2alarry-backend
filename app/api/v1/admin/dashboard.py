from datetime import datetime, time, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import require_admin
from app.db.session import get_db
from app.models.coaching_summary import CoachingSummary
from app.models.conversation import Conversation
from app.models.document import Document
from app.models.message import Message
from app.models.user import User
from app.schemas.admin import (
    AdminCoachingSummaryResponse,
    AdminDashboardResponse,
    AdminUserResponse,
)


router = APIRouter(prefix="/admin", tags=["Admin"])


def utc_week_bounds(now: datetime | None = None) -> tuple[datetime, datetime, datetime]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    current_start = datetime.combine(
        current.date() - timedelta(days=current.weekday()),
        time.min,
        tzinfo=timezone.utc,
    )
    return current_start, current_start + timedelta(days=7), current_start - timedelta(days=7)


def _user_response(row) -> AdminUserResponse:
    user, last_activity, session_count, summary_count = row
    return AdminUserResponse(
        id=user.id,
        full_name=user.full_name,
        email=user.email,
        role="user" if user.role == "candidate" else user.role,
        registered_at=user.created_at,
        last_coaching_activity=last_activity,
        is_enabled=user.is_active,
        session_count=session_count or 0,
        summary_count=summary_count or 0,
    )


def _user_activity_query():
    last_activity = (
        select(
            Conversation.user_id.label("user_id"),
            func.max(Message.created_at).label("last_activity"),
        )
        .join(Message, Message.conversation_id == Conversation.id)
        .where(Message.role == "user")
        .group_by(Conversation.user_id)
        .subquery()
    )
    sessions = (
        select(
            Conversation.user_id.label("user_id"),
            func.count(Conversation.id).label("session_count"),
        )
        .group_by(Conversation.user_id)
        .subquery()
    )
    summaries = (
        select(
            CoachingSummary.user_id.label("user_id"),
            func.count(CoachingSummary.id).label("summary_count"),
        )
        .group_by(CoachingSummary.user_id)
        .subquery()
    )
    return (
        select(
            User,
            last_activity.c.last_activity,
            sessions.c.session_count,
            summaries.c.summary_count,
        )
        .outerjoin(last_activity, last_activity.c.user_id == User.id)
        .outerjoin(sessions, sessions.c.user_id == User.id)
        .outerjoin(summaries, summaries.c.user_id == User.id)
    )


@router.get("/dashboard", response_model=AdminDashboardResponse)
async def get_dashboard(
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    current_start, current_end, previous_start = utc_week_bounds()
    users_total = await db.scalar(select(func.count(User.id))) or 0
    conversations_total = await db.scalar(select(func.count(Conversation.id))) or 0
    documents_total = await db.scalar(
        select(func.count(Document.id)).where(Document.is_active.is_(True))
    ) or 0
    active_users = await db.scalar(
        select(func.count(distinct(Conversation.user_id)))
        .join(Message, Message.conversation_id == Conversation.id)
        .where(
            Message.role == "user",
            Message.created_at >= current_start,
            Message.created_at < current_end,
        )
    ) or 0

    activity_rows = await db.execute(
        select(
            func.date(Message.created_at).label("activity_date"),
            func.count(distinct(Conversation.user_id)).label("active_users"),
        )
        .join(Conversation, Conversation.id == Message.conversation_id)
        .where(
            Message.role == "user",
            Message.created_at >= previous_start,
            Message.created_at < current_end,
        )
        .group_by(func.date(Message.created_at))
    )
    activity_by_date = {row.activity_date: row.active_users for row in activity_rows}

    def week(start: datetime) -> list[dict]:
        return [
            {
                "date": (start + timedelta(days=offset)).date(),
                "active_users": activity_by_date.get(
                    (start + timedelta(days=offset)).date(), 0
                ),
            }
            for offset in range(7)
        ]

    recent_rows = await db.execute(
        _user_activity_query().order_by(User.created_at.desc()).limit(5)
    )
    return {
        "totals": {
            "users": users_total,
            "active_users": active_users,
            "coaching_sessions": conversations_total,
            "documents": documents_total,
        },
        "current_week": week(current_start),
        "previous_week": week(previous_start),
        "recent_users": [_user_response(row) for row in recent_rows.all()],
    }


@router.get("/users", response_model=list[AdminUserResponse])
async def get_users(
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await db.execute(_user_activity_query().order_by(User.created_at.desc()))
    return [_user_response(row) for row in rows.all()]


@router.get(
    "/users/{user_id}/coaching-summaries",
    response_model=list[AdminCoachingSummaryResponse],
)
async def get_user_coaching_summaries(
    user_id: int,
    _admin=Depends(require_admin),
    db: AsyncSession = Depends(get_db),
):
    target_exists = await db.scalar(select(User.id).where(User.id == user_id))
    if target_exists is None:
        raise HTTPException(status_code=404, detail="User not found")

    result = await db.execute(
        select(CoachingSummary)
        .where(CoachingSummary.user_id == user_id)
        .order_by(CoachingSummary.updated_at.desc())
    )
    return [
        AdminCoachingSummaryResponse(
            id=str(summary.id),
            conversation_id=str(summary.conversation_id),
            presenting_focus=summary.presenting_focus,
            primary_discovery=summary.primary_discovery,
            developmental_theme=summary.developmental_theme,
            commitment=summary.commitment,
            next_experiment=summary.next_experiment,
            follow_up_question=summary.follow_up_question,
            coach_notes=summary.coach_notes,
            prior_session_continuity=summary.prior_session_continuity,
            source_turn_count=summary.source_turn_count,
            generation_status=summary.generation_status,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
        )
        for summary in result.scalars().all()
    ]
