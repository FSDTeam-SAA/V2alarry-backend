from app.schemas.coaching import (
    CoachingMove,
    CoachingStage,
    CoachingWorkingState,
    EssentialUnknown,
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

    @classmethod
    def enforce(cls, working_state: CoachingWorkingState) -> CoachingWorkingState:
        unknowns = cls._unique_unknowns(working_state.essential_unknowns)
        has_grounded_evidence = bool(
            working_state.reported_facts or working_state.observations
        )

        if not has_grounded_evidence:
            unknowns = cls._with_unknowns(
                unknowns,
                "direct_evidence",
                "observable_pattern",
            )
            return cls._downgrade(
                working_state,
                stage="discovery",
                move="clarify",
                unknowns=unknowns,
            )

        unknown_set = set(unknowns)
        if (
            working_state.coaching_stage in {"possibilities", "action"}
            and unknown_set & cls._DISCOVERY_BLOCKING_UNKNOWNS
        ):
            return cls._downgrade(
                working_state,
                stage="discovery",
                move="clarify",
                unknowns=unknowns,
            )

        if (
            working_state.coaching_stage in {"possibilities", "action"}
            and "desired_help" in unknown_set
        ):
            return cls._downgrade(
                working_state,
                stage="reflection",
                move="explore_desired_help",
                unknowns=unknowns,
            )

        if (
            working_state.coaching_stage == "action"
            and "user_owned_outcome" not in working_state.transition_basis
        ):
            unknowns = cls._with_unknowns(unknowns, "user_owned_outcome")
            return cls._downgrade(
                working_state,
                stage="possibilities",
                move="explore_options",
                unknowns=unknowns,
            )

        if (
            working_state.coaching_stage == "action"
            and "user_owned_outcome" in unknown_set
        ):
            return cls._downgrade(
                working_state,
                stage="possibilities",
                move="explore_options",
                unknowns=unknowns,
            )

        return working_state

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

        return (
            f"Current coaching stage: {decision.coaching_stage}. "
            f"Permitted next move: {move}. {guidance}"
        )

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
    def _downgrade(
        working_state: CoachingWorkingState,
        *,
        stage: CoachingStage,
        move: CoachingMove,
        unknowns: list[EssentialUnknown],
    ) -> CoachingWorkingState:
        transition_basis = list(
            dict.fromkeys([*working_state.transition_basis, "policy_downgrade"])
        )
        return working_state.model_copy(
            update={
                "coaching_stage": stage,
                "next_coaching_move": move,
                "essential_unknowns": unknowns,
                "transition_basis": transition_basis,
            }
        )
