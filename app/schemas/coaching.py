from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


CoachingStage = Literal["discovery", "reflection", "possibilities", "action"]
CoachingMove = Literal[
    "reflect",
    "clarify",
    "separate_observation_from_interpretation",
    "explore_desired_help",
    "explore_readiness",
    "explore_options",
    "deepen",
    "shift_focus",
    "challenge_tentatively",
    "test_understanding",
    "teach_selectively",
    "develop_experiment",
    "consolidate_commitment",
]
EssentialUnknown = Literal[
    "direct_evidence",
    "observable_pattern",
    "interpretation_separation",
    "desired_help",
    "prior_attempts",
    "user_owned_outcome",
    "tested_understanding",
    "readiness",
    "knowledge_gap",
]
TransitionBasis = Literal[
    "grounded_evidence",
    "observation_interpretation_distinction",
    "desired_help_known",
    "prior_attempts_known",
    "user_owned_outcome",
    "tested_understanding",
    "knowledge_gap_observed",
    "client_feedback_reopened_discovery",
    "policy_downgrade",
]
ClientResponseToLastMove = Literal[
    "not_applicable",
    "affirmed",
    "refined",
    "corrected",
    "new_evidence",
    "not_ready",
    "declined",
]
AnswerLevel = Literal[
    "unknown",
    "action",
    "feeling",
    "role",
    "contribution",
    "commitment",
]
ClientGeneration = Literal["unknown", "emerging", "productive"]


class HeldClue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=500)
    status: Literal["active", "strengthened", "weakened", "superseded"]


class CoachingWorkingState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    current_concern: str | None = Field(default=None, max_length=2000)
    reported_facts: list[str] = Field(default_factory=list, max_length=20)
    observations: list[str] = Field(default_factory=list, max_length=20)
    interpretations_feelings: list[str] = Field(default_factory=list, max_length=20)
    possible_anchors: list[str] = Field(default_factory=list, max_length=10)
    prior_attempts: list[str] = Field(default_factory=list, max_length=20)
    held_clues: list[HeldClue] = Field(default_factory=list, max_length=10)
    unresolved_items: list[str] = Field(default_factory=list, max_length=20)
    emerging_agency: list[str] = Field(default_factory=list, max_length=20)
    coaching_stage: CoachingStage = "discovery"
    next_coaching_move: CoachingMove = "clarify"
    last_coaching_move: CoachingMove | None = None
    client_response_to_last_move: ClientResponseToLastMove = "not_applicable"
    answer_level: AnswerLevel = "unknown"
    client_generation: ClientGeneration = "unknown"
    essential_unknowns: list[EssentialUnknown] = Field(default_factory=list, max_length=9)
    transition_basis: list[TransitionBasis] = Field(default_factory=list, max_length=9)


class RelevantSummarySelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_indices: list[int] = Field(default_factory=list, max_length=3)


class CoachingSummaryData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    presenting_focus: str | None = Field(default=None, max_length=2000)
    primary_discovery: str | None = Field(default=None, max_length=2000)
    developmental_theme: str | None = Field(default=None, max_length=2000)
    commitment: str | None = Field(default=None, max_length=2000)
    next_experiment: str | None = Field(default=None, max_length=2000)
    follow_up_question: str | None = Field(default=None, max_length=2000)
    coach_notes: str | None = Field(default=None, max_length=2000)
    prior_session_continuity: str | None = Field(default=None, max_length=2000)
