from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, EmailStr


class AdminUserResponse(BaseModel):
    id: int
    full_name: str
    email: EmailStr
    role: Literal["admin", "user"]
    registered_at: datetime
    last_coaching_activity: datetime | None
    is_enabled: bool
    session_count: int
    summary_count: int


class DailyActivity(BaseModel):
    date: date
    active_users: int


class DashboardTotals(BaseModel):
    users: int
    active_users: int
    coaching_sessions: int
    documents: int


class AdminDashboardResponse(BaseModel):
    totals: DashboardTotals
    current_week: list[DailyActivity]
    previous_week: list[DailyActivity]
    recent_users: list[AdminUserResponse]


class AdminCoachingSummaryResponse(BaseModel):
    id: str
    conversation_id: str
    presenting_focus: str | None
    primary_discovery: str | None
    developmental_theme: str | None
    commitment: str | None
    next_experiment: str | None
    follow_up_question: str | None
    coach_notes: str | None
    prior_session_continuity: str | None
    source_turn_count: int
    generation_status: str
    created_at: datetime
    updated_at: datetime
