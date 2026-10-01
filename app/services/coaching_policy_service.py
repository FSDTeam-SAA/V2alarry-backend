from app.schemas.coaching import (
    CoachingMove,
    CoachingStage,
    CoachingWorkingState,
    EssentialUnknown,
    TransitionBasis,
)


class CoachingPolicyService:
    """Keeps coaching moves proportionate to the evidence in the working state."""

    _DISCOVERY_BLOCKING_UNKNOWNS = frozenset(
        {
            "direct_evidence",
            "observable_pattern",
            "interpretation_separation",
        }
    )
    _DISCOVERY_REOPENING_RESPONSES = frozenset(
        {"refined", "corrected", "new_evidence", "not_ready", "declined"}
    )

    @classmethod
    def enforce(cls, working_state: CoachingWorkingState) -> CoachingWorkingState:
        unknowns = cls._unique_unknowns(working_state.essential_unknowns)
        transition_basis = cls._unique_transition_basis(
            working_state.transition_basis
        )
        client_response = working_state.client_response_to_last_move

        if (
            working_state.last_coaching_move == "test_understanding"
            and client_response == "affirmed"
        ):
            transition_basis = cls._with_transition_basis(
                transition_basis,
                "tested_understanding",
            )
            unknowns = [item for item in unknowns if item != "tested_understanding"]

        normalized_state = working_state.model_copy(
            update={
                "essential_unknowns": unknowns,
                "transition_basis": transition_basis,
            }
        )

        if client_response in cls._DISCOVERY_REOPENING_RESPONSES:
            move: CoachingMove = "clarify"
            discarded_basis: tuple[TransitionBasis, ...] = ()
            if client_response in {"refined", "corrected", "new_evidence"}:
                unknowns = cls._with_unknowns(unknowns, "tested_understanding")
                discarded_basis = ("tested_understanding",)
            if client_response in {"not_ready", "declined"}:
                unknowns = cls._with_unknowns(unknowns, "readiness")
                move = "explore_readiness"
            return cls._downgrade(
                normalized_state,
                stage="discovery",
                move=move,
                unknowns=unknowns,
                transition_basis="client_feedback_reopened_discovery",
                discard_transition_basis=discarded_basis,
            )

        has_grounded_evidence = bool(
            normalized_state.reported_facts or normalized_state.observations
        )

        if not has_grounded_evidence:
            unknowns = cls._with_unknowns(
                unknowns,
                "direct_evidence",
                "observable_pattern",
            )
            return cls._downgrade(
                normalized_state,
                stage="discovery",
                move="clarify",
                unknowns=unknowns,
            )

        if (
            normalized_state.next_coaching_move == "teach_selectively"
            and "knowledge_gap_observed" not in transition_basis
        ):
            unknowns = cls._with_unknowns(unknowns, "knowledge_gap")
            return cls._downgrade(
                normalized_state,
                stage="discovery",
                move="clarify",
                unknowns=unknowns,
            )

        if (
            normalized_state.last_coaching_move == "teach_selectively"
            and normalized_state.next_coaching_move == "teach_selectively"
        ):
            unknowns = cls._with_unknowns(unknowns, "tested_understanding")
            return cls._downgrade(
                normalized_state,
                stage="reflection",
                move="test_understanding",
                unknowns=unknowns,
                discard_transition_basis=("knowledge_gap_observed",),
            )

        unknown_set = set(unknowns)
        if (
            normalized_state.coaching_stage in {"possibilities", "action"}
            and unknown_set & cls._DISCOVERY_BLOCKING_UNKNOWNS
        ):
            return cls._downgrade(
                normalized_state,
                stage="discovery",
                move="clarify",
                unknowns=unknowns,
            )

        if (
            normalized_state.coaching_stage in {"possibilities", "action"}
            and "desired_help" in unknown_set
        ):
            return cls._downgrade(
                normalized_state,
                stage="reflection",
                move="explore_desired_help",
                unknowns=unknowns,
            )

        if (
            normalized_state.coaching_stage == "action"
            and "user_owned_outcome" not in transition_basis
        ):
            unknowns = cls._with_unknowns(unknowns, "user_owned_outcome")
            return cls._downgrade(
                normalized_state,
                stage="possibilities",
                move="explore_options",
                unknowns=unknowns,
            )

        if (
            normalized_state.coaching_stage == "action"
            and "tested_understanding" not in transition_basis
        ):
            unknowns = cls._with_unknowns(unknowns, "tested_understanding")
            return cls._downgrade(
                normalized_state,
                stage="reflection",
                move="test_understanding",
                unknowns=unknowns,
            )

        if (
            normalized_state.coaching_stage == "action"
            and "user_owned_outcome" in unknown_set
        ):
            return cls._downgrade(
                normalized_state,
                stage="possibilities",
                move="explore_options",
                unknowns=unknowns,
            )

        return normalized_state

    @classmethod
    def build_response_policy(cls, working_state: CoachingWorkingState) -> str:
        decision = cls.enforce(working_state)
        move = decision.next_coaching_move.replace("_", " ")

        if decision.coaching_stage == "discovery":
            guidance = (
                "Use a brief, accurate reflection when useful and one focused "
                "question that helps the leader learn more. Do not offer advice, "
                "options, a script, a framework, a plan, or a recommended action."
            )
        elif decision.coaching_stage == "reflection":
            guidance = (
                "Help the leader examine a supported distinction or pattern. Do not "
                "prescribe what they should do or move to action before the remaining "
                "unknowns are resolved."
            )
        elif decision.coaching_stage == "possibilities":
            guidance = (
                "Help the leader explore choices and tradeoffs without selecting a "
                "course of action for them."
            )
        else:
            guidance = (
                "Help the leader choose a small, user-owned experiment or commitment. "
                "Do not present advice as the required or guaranteed solution."
            )

        knowledge_guidance = (
            "Do not introduce, cite, or teach a framework from knowledge sources "
            "unless the permitted next move is teach selectively."
        )
        if decision.next_coaching_move == "teach_selectively":
            knowledge_guidance = (
                "A demonstrated knowledge gap permits one concise, relevant framework "
                "or distinction. Return immediately to the leader's application and test "
                "what landed."
            )

        move_guidance = ""
        if decision.next_coaching_move == "deepen":
            move_guidance = (
                " Deepen only one level when doing so materially improves the leader's "
                "agency. Distinguish an action, feeling, role label, contribution, and "
                "commitment; do not assume deeper is always better."
            )
        elif decision.next_coaching_move == "shift_focus":
            move_guidance = (
                " Briefly name the shift, foreground one more useful thread, and leave "
                "other valid threads held rather than trying to resolve all of them."
            )
        elif decision.next_coaching_move == "challenge_tentatively":
            move_guidance = (
                " Offer one evidence-grounded challenge as a tentative observation and "
                "explicitly invite the leader to correct it."
            )

        generation_guidance = ""
        if decision.client_generation == "productive":
            generation_guidance = (
                " The leader is productively generating insight: reduce coach content, "
                "use a minimal prompt, and let them continue in their own language."
            )

        feedback_guidance = ""
        if decision.client_response_to_last_move in {"not_ready", "declined"}:
            feedback_guidance = (
                " Do not repeat or repackage a declined suggestion. Acknowledge the "
                "leader's feedback and explore readiness before proposing another action."
            )
        elif decision.client_response_to_last_move == "corrected":
            feedback_guidance = (
                " Acknowledge the correction and its impact without defending the prior "
                "approach. Thank the leader, update the working understanding, and reopen "
                "discovery with one precise question."
            )
        elif decision.client_response_to_last_move in {"refined", "new_evidence"}:
            feedback_guidance = (
                " Treat the leader's update as new evidence. Revise the working "
                "understanding instead of preserving the prior direction."
            )

        if decision.next_coaching_move == "test_understanding":
            feedback_guidance += (
                " Offer one tentative best-current understanding and invite the leader to "
                "affirm, refine, correct, or add. Do not move to action until their response "
                "supports it."
            )

        return (
            f"Current coaching stage: {decision.coaching_stage}. "
            f"Permitted next move: {move}. {guidance} {knowledge_guidance}"
            f"{move_guidance}{generation_guidance}{feedback_guidance}"
        )

    @classmethod
    def action_is_permitted(cls, working_state: CoachingWorkingState) -> bool:
        return cls.enforce(working_state).coaching_stage == "action"

    @staticmethod
    def _unique_unknowns(
        unknowns: list[EssentialUnknown],
    ) -> list[EssentialUnknown]:
        return list(dict.fromkeys(unknowns))

    @staticmethod
    def _with_unknowns(
        unknowns: list[EssentialUnknown],
        *additional: EssentialUnknown,
    ) -> list[EssentialUnknown]:
        return list(dict.fromkeys([*unknowns, *additional]))

    @staticmethod
    def _unique_transition_basis(
        transition_basis: list[TransitionBasis],
    ) -> list[TransitionBasis]:
        return list(dict.fromkeys(transition_basis))

    @staticmethod
    def _with_transition_basis(
        transition_basis: list[TransitionBasis],
        *additional: TransitionBasis,
    ) -> list[TransitionBasis]:
        return list(dict.fromkeys([*transition_basis, *additional]))

    @staticmethod
    def _downgrade(
        working_state: CoachingWorkingState,
        *,
        stage: CoachingStage,
        move: CoachingMove,
        unknowns: list[EssentialUnknown],
        transition_basis: TransitionBasis = "policy_downgrade",
        discard_transition_basis: tuple[TransitionBasis, ...] = (),
    ) -> CoachingWorkingState:
        retained_transition_basis = [
            item
            for item in working_state.transition_basis
            if item not in discard_transition_basis
        ]
        updated_transition_basis = list(
            dict.fromkeys([*retained_transition_basis, transition_basis])
        )
        return working_state.model_copy(
            update={
                "coaching_stage": stage,
                "next_coaching_move": move,
                "essential_unknowns": unknowns,
                "transition_basis": updated_transition_basis,
            }
        )
