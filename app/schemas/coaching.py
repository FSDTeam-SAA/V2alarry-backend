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


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["current_message", "previous_facts", "previous_observations", "history"]
    excerpt: str = Field(min_length=1, max_length=500)
    index: int | None = Field(default=None, ge=0)


class FocusAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=500)
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)


class InteractionEvidenceReference(EvidenceReference):
    source: Literal["current_message"]


class InteractionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usefulness: Literal["unknown", "helpful", "possibly_unhelpful"] = "unknown"
    observation: str = Field(min_length=1, max_length=500)
    evidence: InteractionEvidenceReference


class InquiryAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    progress: Literal["unknown", "advancing", "stalled"]
    reason: str = Field(min_length=1, max_length=500)
    client_requested_pause: bool = Field(
        description="The client actually asks to pause or end the conversation; not an answer denying knowledge or declining one question"
    )
    client_requested_teaching: bool = Field(
        default=False,
        description="The client explicitly asks for a specific explanation or distinction; not merely uncertainty about the situation"
    )


class CoachingEvidenceState(BaseModel):
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


class CoachingWorkingState(CoachingEvidenceState):
    coaching_stage: CoachingStage = "discovery"
    next_coaching_move: CoachingMove = "clarify"
    last_coaching_move: CoachingMove | None = None
    client_response_to_last_move: ClientResponseToLastMove = "not_applicable"
    answer_level: AnswerLevel = "unknown"
    client_generation: ClientGeneration = "unknown"
    essential_unknowns: list[EssentialUnknown] = Field(default_factory=list, max_length=9)
    transition_basis: list[TransitionBasis] = Field(default_factory=list, max_length=9)
    focus_assessment: FocusAssessment | None = None
    interaction_assessment: InteractionAssessment | None = None
    inquiry_assessment: InquiryAssessment | None = None


class RelevantSummarySelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relevant_indices: list[int] = Field(default_factory=list, max_length=3)


class FocusSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    focus: str = Field(min_length=1, max_length=500)
    reason: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=3)


class InteractionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    usefulness: Literal["unknown", "helpful", "possibly_unhelpful"]
    observation: str = Field(min_length=1, max_length=500)


class CoachingTurnAssessment(BaseModel):
    """Bounded, evidence-grounded judgment made after factual state extraction."""

    model_config = ConfigDict(extra="forbid")

    focus_assessment: FocusSelection
    interaction_assessment: InteractionSelection | None = Field(...)
    inquiry_assessment: InquiryAssessment | None = None
    coaching_stage: CoachingStage
    next_coaching_move: CoachingMove
    client_response_to_last_move: ClientResponseToLastMove
    answer_level: AnswerLevel = "unknown"
    essential_unknowns: list[EssentialUnknown] = Field(..., max_length=9)
    transition_basis: list[TransitionBasis] = Field(..., max_length=9)
    client_generation: ClientGeneration
    knowledge_gap_observed: bool = Field(description="Specific relevant knowledge the client lacks; not uncertainty about their feelings, goals, or situation")
    user_owned_outcome: bool = Field(description="An outcome or experiment the client chooses in their own words; not acceptance of Jess's suggestion")
    tested_understanding: bool = Field(description="The client confirms a useful current understanding, or a prior confirmed understanding remains valid; not merely acknowledgment")
    transition_evidence_ids: list[str] = Field(max_length=3)


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
