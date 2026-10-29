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
    "invite_takeaways",
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
    material_unknown: str | None = Field(default=None, max_length=500,
        description="A specific unanswered question that changes the smallest useful next move; not a category label, missing commitment, or a future outcome nobody can know yet. Null when synthesis or scoped support can use what is established.")
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
        description="Null normally. This is the scope of an EXISTING boundary on support_action_id, NOT the kind of help requested (scope above). If supplied, match that action's saved boundary_scope and cite its original boundary evidence. Classify legacy unknown only from that proof. Never label requested practice as a preparation boundary. This does not change readiness.")


RawClientEvidenceId = Annotated[str, Field(pattern=r"^(current_message|history:[0-9]+)$")]


class ClientExcerptSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    evidence_id: RawClientEvidenceId
    excerpt: str = Field(min_length=1, max_length=500, pattern=r"\S",
        description="Copy the precise contiguous client words supporting this claim from the referenced raw message. Preserve punctuation; never paraphrase or quote Jess.")


class ClientClaritySelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["not_described", "choice_only", "insight"] = Field(default="not_described",
        description="Classify what the client actually describes, separately from choosing an action. insight means an understanding, distinction or perspective they gained; choice_only means a decision, intention or willingness without an account of what became clear; not_described means neither is stated. A desired outcome or action choice alone is not an insight.")
    evidence: list[ClientExcerptSelection] = Field(default_factory=list, max_length=3,
        description="For insight, quote the specific clause expressing the client's understanding, not a plan/readiness clause or an undifferentiated message. Ordinary spontaneous language counts; no special insight phrase or before/after formula is required. Empty when no account is available.")


class ClosureSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str | None = Field(default=None, min_length=1, max_length=36,
        description="Copy the exact matching planned/ongoing action_context id for the client's chosen step, including when their account of clarity is missing. Null only when no chosen action is supported; do not discard an available matching ID during recovery.")
    next_step_evidence_ids: list[RawClientEvidenceId] = Field(default_factory=list, max_length=3,
        description="Assess independently: raw client evidence describing what they intend to do with their clarity, matching action_id. Learning or research can be a step. Empty when absent.")
    willingness_evidence_ids: list[RawClientEvidenceId] = Field(default_factory=list, max_length=3,
        description="Assess independently: raw client evidence of willingness to try that step; not merely considering it, permission to rehearse, or agreement with Jess. Empty when absent.")


class ClosureAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str = Field(min_length=1, max_length=36)
    clarity_evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)
    next_step_evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)
    willingness_evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)


class ClosureReview(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["supported", "not_supported", "unavailable"]
    reason: str = Field(min_length=1, max_length=300)
    duration_ms: float = Field(ge=0)


class ActionContextUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str | None = Field(default=None, max_length=36, description="Reuse the exact existing action ID for wording, preparation, or starting that action. Null only for a distinct new action; omit unchanged actions entirely.")
    actor: str = Field(min_length=1, max_length=100)
    action: str = Field(min_length=1, max_length=500,
        description="The underlying action, without dates, lifecycle labels or readiness; retain timing in narrative facts and exact evidence. Never infer an opposite or goal. For existing records copy the exact action and ID. Recommitting or changing the date of the SAME task does not create a new underlying action, even if the client calls it a new commitment. Wording/preparation also refers to that underlying action.")
    status: ActionStatus | None = Field(default=None,
        description="Real action lifecycle only. Must be null for preparation/topic permission updates; practice permission cannot establish a planned action.")
    readiness: ActionReadiness | None = Field(default=None,
        description="Willingness/unreadiness/refusal within boundary_scope. For action/null scope this concerns performing the real action; preparation/topic changes only that permission. Awkwardness/anxiety alone and missing information/ability are not refusal. Ready requires readiness_confirmation and client evidence; intention or practice alone does not establish readiness to act.")
    basis: Literal["consideration", "intention", "reported_action", "readiness_boundary", "readiness_confirmation", "information_limit", "correction"] = Field(
        description="information_limit means missing information/examples/resources/ability, not unwillingness; these updates cannot establish action/readiness. readiness_boundary requires expressed unreadiness or refusal within the stated scope; discomfort alone is insufficient. EVERY ready update requires readiness_confirmation. Intention alone cannot support ready; do not invent a narrower action.")
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)
    boundary_scope: BoundaryScope | None = Field(default=None,
        description="Null unless recording or changing a genuine boundary with basis readiness_boundary, readiness_confirmation or correction. A practice request alone is not a boundary update; omit unchanged actions. action: doing it, preparation: wording/practice, topic: discussion, unknown: ambiguous. Preparation/topic changes only that permission, never readiness to act. Ready with readiness_confirmation lifts that scoped refusal. Awkwardness alone is not unwillingness.")
    status_basis: Literal["consideration", "intention", "reported_action", "correction"] | None = Field(default=None,
        description="Independent proof classification for a non-null status. Use this with status_evidence when one message changes both status and readiness. The shared basis is only the legacy fallback.")
    status_evidence: list[EvidenceReference] | None = Field(default=None, min_length=1, max_length=3,
        description="Exact client proof of the real action status, independently of willingness or refusal.")
    readiness_basis: Literal["readiness_boundary", "readiness_confirmation", "correction"] | None = Field(default=None,
        description="Independent proof classification for non-null readiness. Ready requires readiness_confirmation here or in legacy basis. A past attempt is not willingness to repeat it.")
    readiness_evidence: list[EvidenceReference] | None = Field(default=None, min_length=1, max_length=3,
        description="Exact client proof of current willingness, unreadiness or refusal within boundary_scope, independently of action status.")


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
    reported_facts: list[str] = Field(default_factory=list, max_length=20,
        description="Reported events, constraints, first-person experience and what was actually said, preserving attribution. Not every declarative client assertion is an established fact: inferred emotions, motives or broad judgments about another person belong in interpretations_feelings unless supported by reported words or concrete observations.")
    observations: list[str] = Field(default_factory=list, max_length=20,
        description="What the client actually saw, heard or received, including specific comments or behavior. Preserve the difference between this evidence and the meaning the client gives it.")
    interpretations_feelings: list[str] = Field(default_factory=list, max_length=20,
        description="Client feelings and inferred meanings, including conclusions about another person's emotion, intention or judgment. Keep these attributed even when the client states them confidently as fact. Categorizing a claim here does not mean the client has recognized it as an interpretation.")
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
    closure_assessment: ClosureAssessment | None = None
    closure_review: ClosureReview | None = None


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

    interpretation_gap: bool = Field(default=False,
        description="Does the client's own wording treat an inferred meaning as fact without direct evidence? Judge the raw client text, not the extractor's labels. Another person's inferred emotion, motive or judgment is not established merely because the client states it declaratively. True even if extraction already put it under interpretations_feelings. False for the client's own feeling or a meaning they explicitly recognize as tentative.")
    interpretation_evidence_ids: list[str] = Field(default_factory=list, max_length=3,
        description="For a true interpretation_gap, cite raw current_message/history evidence of the client's conclusion and the observation it rests on. Otherwise empty.")
    focus_assessment: FocusSelection
    interaction_assessment: InteractionSelection | None = Field(...)
    inquiry_assessment: InquiryAssessment | None = None
    requested_support: RequestedSupport = Field(default="none",
        description="Classify what the client ASKS, before selecting the response. A request for how to begin is action_wording even when readiness or understanding is incomplete; do not choose none merely because you plan discovery.")
    support_action_id: str | None = Field(default=None, max_length=36,
        description="Existing underlying action ID for the current help or chosen next step; do not attach the boundary on a different action.")
    closure_assessment: ClosureSelection = Field(default_factory=ClosureSelection,
        description="Select each available raw-client reference for the chosen next step and willingness independently, even if clarity is missing. Empty lists for missing elements, never null. A focused raw-client check verifies the account of understanding before the server permits closure. Unknown future results do not block closure.")
    progress_assessment: ProgressSelection | None = None
    understanding_assessment: UnderstandingSelection | None = None
    support_assessment: SupportSelection | None = None
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


class PreservedSummaryAction(SummaryActionChange):
    operation: Literal["preserve"] = "preserve"


class WithdrawableSummaryAction(SummaryActionChange):
    operation: Literal["preserve", "clear"] = "preserve"


class PreservedCoachingSummaryRevision(CoachingSummaryRevision):
    commitment_change: PreservedSummaryAction = Field(default_factory=PreservedSummaryAction)
    next_experiment_change: PreservedSummaryAction = Field(default_factory=PreservedSummaryAction)


class WithdrawableCoachingSummaryRevision(CoachingSummaryRevision):
    commitment_change: WithdrawableSummaryAction = Field(default_factory=WithdrawableSummaryAction)
    next_experiment_change: WithdrawableSummaryAction = Field(default_factory=WithdrawableSummaryAction)
