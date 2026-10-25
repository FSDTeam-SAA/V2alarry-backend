from typing import Annotated, Literal

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
    "support_choice",
    "close_or_continue",
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

    source: Literal[
        "current_message", "previous_facts", "previous_observations", "history",
        "previous_attempts", "previous_feelings", "previous_agency",
        "action_context",
        "understanding_context",
    ]
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


ActionStatus = Literal["considering", "planned", "attempted", "ongoing", "completed", "abandoned", "unknown"]
ActionReadiness = Literal["unknown", "not_ready", "declined", "ready"]
RequestedSupport = Literal["none", "exploration", "conceptual_explanation", "action_wording", "action_planning", "rehearsal"]
BoundaryScope = Literal["action", "preparation", "topic", "unknown"]
EstablishedPoint = Annotated[str, Field(min_length=1, max_length=500)]


class UnderstandingContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    picture: str = Field(min_length=1, max_length=1500)
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)
    basis: Literal["explicit", "natural"]
    status: Literal["supported", "needs_revision"] = "supported"


class ProgressAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    established: list[EstablishedPoint] = Field(max_length=5)
    material_unknown: str | None = Field(default=None, max_length=500)
    discovery_useful: bool
    understanding_effect: Literal["unknown", "establish", "retain", "revise"]
    recommended_direction: Literal["discovery", "synthesis", "support", "consolidate"]
    reason: str = Field(min_length=1, max_length=500)
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)


class SupportAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["exploration", "preparation", "execution"]
    permitted: bool
    reason: str = Field(min_length=1, max_length=500)
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)
    boundary_scope: BoundaryScope | None = None


class ProgressSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    established: list[EstablishedPoint] = Field(max_length=5)
    material_unknown: str | None = Field(default=None, max_length=500)
    discovery_useful: bool
    understanding_effect: Literal["unknown", "establish", "retain", "revise"]
    recommended_direction: Literal["discovery", "synthesis", "support", "consolidate"]
    reason: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=3)


class UnderstandingSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    picture: str = Field(min_length=1, max_length=1500)
    basis: Literal["explicit", "natural"]
    evidence_ids: list[str] = Field(min_length=1, max_length=3,
        description="Client evidence demonstrating shared understanding, not silence, lack of objection, or a bare acknowledgment.")


class SupportSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["exploration", "preparation", "execution"]
    permitted: bool
    reason: str = Field(min_length=1, max_length=500)
    evidence_ids: list[str] = Field(min_length=1, max_length=3,
        description="Evidence for this help request or client-owned direction, separate from evidence for understanding or commitment.")
    boundary_scope: BoundaryScope | None = Field(default=None,
        description="Classify an existing relevant boundary from its original action_context evidence, including legacy scope unknown. This does not change readiness. Null when no boundary is relevant.")


class ActionContextUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, max_length=36, description="Reuse the exact existing action ID for wording, preparation, or starting that action. Null only for a distinct new action; omit unchanged actions entirely.")
    actor: str = Field(min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=500,
        description="The action the client actually describes, never an inferred opposite or goal. 'I keep reviewing their work' describes reviewing their work, NOT stopping it. Do not invent a contemplated action from a concern. For an existing record copy its exact action; wording/preparation refers to that underlying action.")
    status: ActionStatus | None = Field(default=None,
        description="Real action lifecycle only. Must be null for preparation/topic permission updates; practice permission cannot establish a planned action.")
    readiness: ActionReadiness | None = Field(default=None,
        description="Willingness/unreadiness/refusal within boundary_scope. For action/null scope this concerns performing the real action; preparation/topic changes only that permission. Awkwardness/anxiety alone and missing information/ability are not refusal. Ready requires readiness_confirmation and client evidence; intention or practice alone does not establish readiness to act.")
    basis: Literal["consideration", "intention", "reported_action", "readiness_boundary", "readiness_confirmation", "information_limit", "correction"] = Field(
        description="information_limit means missing information/examples/resources/ability, not unwillingness; these updates cannot establish action/readiness. readiness_boundary requires expressed unreadiness or refusal within the stated scope; discomfort alone is insufficient. EVERY ready update requires readiness_confirmation. Intention alone cannot support ready; do not invent a narrower action.")
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)
    boundary_scope: BoundaryScope | None = Field(default=None,
        description="action: willingness to do it, preparation: willingness for wording/practice, topic: willingness to discuss, unknown: ambiguous. For preparation/topic, readiness updates only that permission, never readiness to act. Ready with readiness_confirmation lifts that scoped refusal. Awkwardness alone is not unwillingness.")


class ClientActionContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=36)
    actor: str = Field(min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=500)
    status: ActionStatus = "unknown"
    readiness: ActionReadiness = "unknown"
    evidence: list[str] = Field(min_length=1, max_length=3)
    status_evidence: str | None = Field(default=None, max_length=500)
    readiness_evidence: str | None = Field(default=None, max_length=500)
    boundary_scope: BoundaryScope = "unknown"
    preparation_declined: bool = False
    discussion_declined: bool = False
    preparation_evidence: str | None = Field(default=None, max_length=500)
    discussion_evidence: str | None = Field(default=None, max_length=500)


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
    action_updates: list[ActionContextUpdate] = Field(default_factory=list, max_length=10,
        description="Include every newly stated action boundary, including on the first turn. An unchosen action still needs a record when the client declines it or is not comfortable doing it yet. Empty only when no action or boundary changes.")


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
    action_context: list[ClientActionContext] = Field(default_factory=list, max_length=10)
    requested_support: RequestedSupport = "none"
    support_action_id: str | None = Field(default=None, max_length=36)
    understanding_context: UnderstandingContext | None = None
    progress_assessment: ProgressAssessment | None = None
    support_assessment: SupportAssessment | None = None


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
    tested_understanding: bool = Field(description="Shared understanding supported explicitly, demonstrated naturally, or retained without material contradiction; never merely acknowledgment. No mandatory confirmation question.")
    transition_evidence_ids: list[str] = Field(max_length=3)
    requested_support: RequestedSupport = Field(default="none",
        description="Classify the client's request, not the response you intend to offer. 'How would I open that meeting?' is action_wording even when readiness is blocked, never exploration or conceptual_explanation.")
    support_action_id: str | None = Field(default=None, max_length=36,
        description="Existing underlying action ID for practical support, including opening, wording, preparation, or rehearsal of that action. Prefer the matching action with an active readiness boundary over a narrower paraphrase.")
    progress_assessment: ProgressSelection | None = None
    understanding_assessment: UnderstandingSelection | None = None
    support_assessment: SupportSelection | None = None
    ownership_evidence_ids: list[str] = Field(default_factory=list, max_length=3)
    knowledge_evidence_ids: list[str] = Field(default_factory=list, max_length=3)


class ContinuingTurnAssessment(CoachingTurnAssessment):
    inquiry_assessment: InquiryAssessment = Field(...)


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


class SummaryClientEvidence(EvidenceReference):
    source: Literal["current_message"] = "current_message"
    index: None = None
    excerpt: str = Field(min_length=1, max_length=500,
        description="Copy an exact contiguous quote from latest_turn.leader, not a paraphrase from working_state. Revision evidence is the exception to the prose summary's no-quote rule.")


class SummaryActionChange(BaseModel):
    model_config = ConfigDict(extra="forbid")

    operation: Literal["preserve", "replace", "clear"] = "preserve"
    previous_value: str | None = Field(default=None, max_length=2000)
    action_id: str | None = Field(default=None, max_length=36)
    evidence: list[SummaryClientEvidence] = Field(default_factory=list, max_length=3)


class CoachingSummaryRevision(CoachingSummaryData):
    commitment_change: SummaryActionChange = Field(default_factory=SummaryActionChange)
    next_experiment_change: SummaryActionChange = Field(default_factory=SummaryActionChange)
