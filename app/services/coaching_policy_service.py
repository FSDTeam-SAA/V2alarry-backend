from app.schemas.coaching import (
    CoachingMove,
    CoachingStage,
    CoachingWorkingState,
    EssentialUnknown,
    TransitionBasis,
)


class CoachingPolicyService:
    _DISCOVERY_MOVES = {
        "reflect", "clarify", "separate_observation_from_interpretation",
        "explore_desired_help", "explore_readiness", "deepen", "shift_focus",
        "challenge_tentatively", "test_understanding",
    }
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
        progress = working_state.progress_assessment
        understanding = working_state.understanding_context
        if progress and progress.understanding_effect == "revise":
            understanding = understanding.model_copy(update={"status": "needs_revision"}) if understanding else None
            transition_basis = [item for item in transition_basis if item != "tested_understanding"]
            unknowns = cls._with_unknowns(unknowns, "tested_understanding")
        shared = bool(understanding and understanding.status == "supported")
        if shared:
            transition_basis = cls._with_transition_basis(transition_basis, "tested_understanding")
            unknowns = [item for item in unknowns if item != "tested_understanding"]

        if (
            working_state.last_coaching_move == "test_understanding"
            and client_response == "affirmed"
            and progress is None
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
                "understanding_context": understanding,
            }
        )
        if normalized_state.next_coaching_move in {"invite_takeaways", "close_or_continue"}:
            normalized_state = normalized_state.model_copy(update={
                "coaching_stage": "possibilities",
                "support_action_id": normalized_state.support_action_id or (
                    normalized_state.closure_assessment.action_id if normalized_state.closure_assessment else None),
            })

        possibly_unhelpful = (normalized_state.interaction_assessment is not None
                              and normalized_state.interaction_assessment.usefulness == "possibly_unhelpful")
        inquiry = normalized_state.inquiry_assessment
        repairing_inquiry = bool(possibly_unhelpful and inquiry and inquiry.progress == "stalled"
                                 and normalized_state.client_generation != "productive")
        if inquiry and inquiry.client_requested_pause:
            return cls._downgrade(
                normalized_state, stage="discovery", move="explore_readiness",
                unknowns=unknowns,
            )
        repeated_understanding_check = (normalized_state.last_coaching_move == "test_understanding"
                and progress and not (progress.understanding_effect == "revise" and client_response == "corrected"
                                      and progress.recommended_direction == "synthesis"))
        if normalized_state.next_coaching_move == "test_understanding" and repeated_understanding_check:
            normalized_state = normalized_state.model_copy(update={
                "next_coaching_move": "shift_focus", "coaching_stage": "discovery",
            })
        # Teaching remains readable in older records, but is deferred in this calibration.
        if normalized_state.next_coaching_move == "teach_selectively":
            normalized_state = normalized_state.model_copy(update={
                "next_coaching_move": "explore_options" if shared else "clarify",
                "coaching_stage": "possibilities" if shared else "discovery",
            })
        relevant_actions = [item for item in normalized_state.action_context
                            if not normalized_state.support_action_id or item.id == normalized_state.support_action_id]
        if any(item.discussion_declined for item in relevant_actions):
            return cls._downgrade(normalized_state, stage="discovery", move="shift_focus", unknowns=unknowns)
        if "interpretation_separation" in unknowns and normalized_state.next_coaching_move not in {
                "test_understanding", "challenge_tentatively", "separate_observation_from_interpretation"}:
            return cls._downgrade(normalized_state, stage="discovery",
                                  move="separate_observation_from_interpretation", unknowns=unknowns)
        practical_help = normalized_state.requested_support in {"action_wording", "action_planning", "rehearsal"}
        practical_move = normalized_state.coaching_stage == "action" or normalized_state.next_coaching_move == "support_choice"
        support = normalized_state.support_assessment
        if ((normalized_state.coaching_stage == "action"
                or normalized_state.next_coaching_move in {"invite_takeaways", "close_or_continue"}) and support
                and support.scope == "execution" and not support.permitted):
            normalized_state = cls._downgrade(normalized_state, stage="possibilities",
                                             move="reflect", unknowns=unknowns)
            practical_move = False
        preparing = bool(support and support.permitted and support.scope == "preparation" and shared)
        nonexecuting_support = bool(support and support.permitted
                                    and support.scope in {"exploration", "preparation"} and shared)
        preparation_requested = preparing or (practical_help and (not support or support.scope != "execution"))
        blocked_actions = [item for item in normalized_state.action_context
                           if (item.readiness in {"not_ready", "declined"} or item.status == "abandoned"
                               or item.discussion_declined or (preparation_requested and item.preparation_declined))
                           and (not normalized_state.support_action_id or item.id == normalized_state.support_action_id)]
        if nonexecuting_support and normalized_state.coaching_stage == "action":
            # Exploration or preparation never authorizes execution/commitment.
            normalized_state = normalized_state.model_copy(update={"coaching_stage": "possibilities",
                                                                  "next_coaching_move": "support_choice"})
        blocking = [item for item in blocked_actions if not (
            nonexecuting_support and not (preparation_requested and item.preparation_declined)
            and not item.discussion_declined
            and (item.boundary_scope == "action" or (
                item.boundary_scope == "unknown" and support.boundary_scope == "action")) and item.status != "abandoned"
            and normalized_state.coaching_stage != "action")]
        if blocking and (practical_help or practical_move):
            if (not shared and support and support.permitted
                    and support.scope in {"exploration", "preparation"}
                    and all(item.status != "abandoned" and not item.discussion_declined
                            and not (preparation_requested and item.preparation_declined)
                            and (item.boundary_scope == "action" or (
                                item.boundary_scope == "unknown" and support.boundary_scope == "action"))
                            for item in blocking)):
                return cls._downgrade(normalized_state, stage="reflection", move="test_understanding",
                                      unknowns=cls._with_unknowns(unknowns, "tested_understanding"))
            feedback_repair = bool(inquiry and inquiry.progress == "stalled") or client_response in {
                "corrected", "not_ready", "declined",
            }
            return cls._downgrade(
                normalized_state, stage="discovery",
                move="shift_focus" if inquiry and inquiry.progress == "stalled" else "explore_readiness",
                unknowns=cls._with_unknowns(unknowns, "readiness"),
                transition_basis="client_feedback_reopened_discovery" if feedback_repair else "policy_downgrade",
            )
        if normalized_state.next_coaching_move in {"support_choice", "invite_takeaways", "close_or_continue"}:
            material_unknown = bool(progress and progress.material_unknown)
            if not shared or material_unknown or not (normalized_state.reported_facts or normalized_state.observations):
                correction = bool(progress and progress.understanding_effect == "revise")
                normalized_state = cls._downgrade(normalized_state,
                    stage="discovery" if material_unknown or correction else "reflection",
                    move="test_understanding" if not material_unknown else "clarify",
                    unknowns=cls._with_unknowns(unknowns, "tested_understanding") if not shared else unknowns,
                    transition_basis="client_feedback_reopened_discovery" if correction else "policy_downgrade")
            elif normalized_state.next_coaching_move == "support_choice" and (
                    not support or not support.permitted or support.scope == "execution"):
                normalized_state = cls._downgrade(normalized_state, stage="possibilities", move="explore_options", unknowns=unknowns)
            else:
                if normalized_state.next_coaching_move in {"invite_takeaways", "close_or_continue"}:
                    chosen = [item for item in relevant_actions
                              if item.status in {"planned", "ongoing"}
                              and item.readiness not in {"not_ready", "declined"}
                              and not item.discussion_declined]
                    if not chosen:
                        return cls._downgrade(normalized_state, stage="possibilities",
                                              move="explore_options", unknowns=unknowns)
                    closure = normalized_state.closure_assessment
                    if not closure or closure.action_id not in {item.id for item in chosen}:
                        return cls._downgrade(normalized_state, stage="possibilities",
                                              move="invite_takeaways", unknowns=unknowns)
                    normalized_state = normalized_state.model_copy(update={"next_coaching_move": "close_or_continue"})
                return normalized_state.model_copy(update={"coaching_stage": "possibilities"})
        synthesis_useful = normalized_state.next_coaching_move == "test_understanding" and (
            (normalized_state.last_coaching_move != "test_understanding"
             and client_response not in {"declined", "not_ready"})
            or bool(progress and progress.recommended_direction == "synthesis"
                    and progress.understanding_effect in {"establish", "revise"}))
        if (inquiry and inquiry.progress == "stalled"
                and normalized_state.client_generation != "productive"
                and client_response != "corrected"
                and not possibly_unhelpful
                and not synthesis_useful
                and bool(normalized_state.reported_facts or normalized_state.observations)):
            # Declining one unproductive question does not mean declining coaching.
            return cls._downgrade(
                normalized_state, stage="discovery", move="shift_focus",
                unknowns=cls._with_unknowns(unknowns, "tested_understanding"),
                transition_basis="client_feedback_reopened_discovery",
                discard_transition_basis=("tested_understanding",),
            )
        if (client_response in cls._DISCOVERY_REOPENING_RESPONSES
                and not repairing_inquiry
                and not (shared and progress and progress.understanding_effect in {"retain", "establish"}
                         and client_response in {"refined", "new_evidence"})
                and not (possibly_unhelpful and client_response in {"refined", "new_evidence", "corrected"})):
            move: CoachingMove = (normalized_state.next_coaching_move
                                  if normalized_state.next_coaching_move in cls._DISCOVERY_MOVES else "clarify")
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

        if possibly_unhelpful:
            if repeated_understanding_check:
                move = (normalized_state.next_coaching_move
                        if normalized_state.next_coaching_move in cls._DISCOVERY_MOVES
                        and normalized_state.next_coaching_move != "test_understanding" else "shift_focus")
            elif not repairing_inquiry and normalized_state.next_coaching_move in {"shift_focus", "explore_desired_help", "test_understanding"}:
                move = normalized_state.next_coaching_move
            else:
                move = "test_understanding"
            return cls._downgrade(
                normalized_state, stage="discovery", move=move,
                unknowns=cls._with_unknowns(unknowns, "tested_understanding"),
                transition_basis="client_feedback_reopened_discovery",
                discard_transition_basis=("tested_understanding",),
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

        if normalized_state.coaching_stage == "action" and progress and not any(
                item.status in {"planned", "ongoing"}
                and (not normalized_state.support_action_id or item.id == normalized_state.support_action_id)
                for item in normalized_state.action_context):
            return cls._downgrade(normalized_state, stage="possibilities", move="explore_options",
                                  unknowns=cls._with_unknowns(unknowns, "user_owned_outcome"))

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

        allowed = set(cls._DISCOVERY_MOVES)
        if normalized_state.coaching_stage in {"possibilities", "action"}:
            allowed.add("explore_options")
        if normalized_state.coaching_stage == "action":
            allowed |= {"develop_experiment", "consolidate_commitment"}
        if normalized_state.next_coaching_move not in allowed:
            return cls._downgrade(normalized_state, stage=normalized_state.coaching_stage,
                                  move="clarify", unknowns=unknowns)
        return normalized_state

    @classmethod
    def build_response_policy(cls, working_state: CoachingWorkingState) -> str:
        decision = cls.enforce(working_state)
        move = decision.next_coaching_move.replace("_", " ")

        if decision.coaching_stage == "discovery":
            guidance = (
                "Use one focused question or invitation that helps the leader learn more. "
                "Usually begin directly with that invitation; an opening reflection is optional. Do not offer advice, "
                "options, a script, a framework, a plan, or a recommended action."
            )
        elif decision.coaching_stage == "reflection":
            guidance = (
                "Help the leader examine a distinction or pattern supported by their own account. "
                "Reflect only what the client has supplied; do not add a general method, useful tip, "
                "possible opening or next step, even as an optional observation. If no useful reflection "
                "is supported, move directly to an invitation for their own thinking. Do not "
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
            "Teaching and coach-supplied examples are deferred in this calibration. "
            "Do not introduce a framework, script, recommended option or advice, even when asked. "
            "Help the client discover their own next thinking from their actual situation."
        )
        if decision.inquiry_assessment and decision.inquiry_assessment.client_requested_pause:
            return (
                f"Current coaching stage: {decision.coaching_stage}. Permitted next move: {move}. "
                "Respect the client's explicit request to pause or end. Acknowledge it briefly "
                "without another discovery question, pressure to continue, or an action assignment."
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
                " Follow the selected focus into a relevant unexplored part of the situation. "
                "Do not repeat the previous question in new words or make the client choose "
                "the next line of inquiry. A transition is optional; ask naturally without "
                "announcing or summarizing every shift. Hold other valid threads provisionally."
            )
        elif decision.next_coaching_move == "challenge_tentatively":
            move_guidance = (
                " Offer one evidence-grounded challenge as a tentative observation and "
                "explicitly invite the leader to correct it."
            )
        elif decision.next_coaching_move == "separate_observation_from_interpretation":
            move_guidance = (
                " Use the actual words or events already reported to distinguish what happened from "
                "the meaning the client gave it. Invite them to examine that meaning and another "
                "possible interpretation. Do not endorse a motive, lecture about terminology, "
                "or ask again for observations they already provided."
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
                "leader's boundary. Declining a question or suggestion is not a request "
                "to stop the conversation; do not infer unwillingness from missing information."
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
            guidance = (
                "Connect the accumulated purpose, relevant context and client-owned possibility in one "
                "concise tentative picture; distinguish a possibility from a decided plan. Invite review "
                "of that picture, not another question about desired outcomes, feelings or readiness. "
                "Do not supply scripts, wording, rehearsal or "
                "practical steps during this move."
                " Include only possibilities the client actually stated. If none exists, omit that part. "
                "Do not invent or suggest a next step inside the synthesis, even tentatively. "
                "A plausible method derived from their goal is still Jess's suggestion, not their idea."
            )
            feedback_guidance = (
                " Acknowledge any mistaken assumption without defending it. Invite the leader to "
                "affirm, refine, correct, or add to the picture before choosing a new inquiry. "
                "Do not infer that an unheld event is scheduled or that an unchosen possibility is blocked."
            )
        if decision.progress_assessment and decision.progress_assessment.understanding_effect == "retain":
            feedback_guidance = " Preserve the established picture; this detail extends it without requiring another confirmation."
        if decision.next_coaching_move == "support_choice":
            if decision.support_assessment and decision.support_assessment.scope == "preparation":
                guidance = (
                    "Support the client's own words and ideas for their possibility. Invite them to develop "
                    "how they might begin, using the purpose and understanding already established. "
                    "If they do not know, help them think through a manageable starting point without "
                    "supplying Jess's example or repeating an unsuccessful question. Preparation is not a commitment to act. "
                    "Use what is already understood instead of asking again for the outcome, feelings or readiness."
                )
            else:
                guidance = (
                    "Explore the coaching user's own possibility and what it means to them. "
                    "Exploration permission does not authorize wording, rehearsal or an action plan. "
                    "Keep any real action uncommitted unless the coaching user chooses it."
                )
            feedback_guidance = " Shared understanding is supported by the dialogue; no confirmation ritual is required."
        elif decision.next_coaching_move == "invite_takeaways":
            guidance = (
                "Invite the client to describe what became clearer in this conversation and what they "
                "will do with it, in their own words. Listen for their willingness to try their chosen "
                "step. Do not offer closure, supply the recap for them, or invent an assignment. "
                "If they cannot yet describe a useful takeaway or direction, help them develop it."
            )
            feedback_guidance = ""
        elif decision.next_coaching_move == "close_or_continue":
            guidance = (
                "The client has described their clarity, chosen next step and willingness to try it. "
                "Acknowledge their own account briefly and check naturally whether they wish to close "
                "or continue. Preserve the chosen action's status; do not add an assignment."
            )
            feedback_guidance = ""

        if (decision.interaction_assessment
                and decision.interaction_assessment.usefulness == "possibly_unhelpful"):
            feedback_guidance += (
                " The usefulness of Jess's last direction is uncertain. Check tentatively "
                "what would help now or invite correction; do not label the client disengaged "
                "or interpret acknowledgment as agreement or commitment."
            )
        elif (decision.interaction_assessment
                and decision.interaction_assessment.usefulness == "unknown"):
            feedback_guidance += (
                " It is uncertain whether the last invitation helped. If checking understanding, "
                "briefly invite the client to say what would help now; do not insist on the prior trail "
                "or label uncertainty as disengagement."
            )
        if (decision.inquiry_assessment
                and decision.inquiry_assessment.progress == "stalled"
                and decision.next_coaching_move == "shift_focus"):
            feedback_guidance = (
                " The previous inquiry is stalled, not evidence that the client wants to stop. "
                "Keep their established purpose and take responsibility for a fresh useful "
                "direction. Explore one relevant unknown about the situation, people, relationship, "
                "or client's experience. Do not ask again what help, outcome, or readiness they "
                "want, or request more clues they have already said they lack. Respect a declined "
                "topic by exploring another; do not recycle the same question or impose an action."
            )

        action_guidance = (
            " An action marked considering, planned, or unknown is not yet reported as done. "
            "Preserve actor, action status, and uncertainty from action_context. Do not turn an "
            "idea or a coach suggestion into an established practice, prior attempt, or commitment. "
            "An abandoned action must not be proposed again unless the client reopens it. "
            "Narrative facts alone do not establish implementation when action status is unconfirmed."
        )
        relevant_actions = [item for item in decision.action_context
                            if not decision.support_action_id or item.id == decision.support_action_id]
        if any(item.readiness in {"not_ready", "declined"} for item in relevant_actions):
            action_guidance += (
                " A wording request does not establish readiness to act. Respect the specific scope of each action boundary. Requested exploration or preparation may be permitted while "
                "the real action remains not ready; do not schedule, recommend carrying it out, or record a commitment. "
                "A preparation/topic refusal still blocks that help. If scope is unclear, ask one relevant clarification. "
                "Do not repeat a readiness question or propose a declined action in new words."
            )
        if any(item.preparation_declined or item.discussion_declined for item in decision.action_context):
            action_guidance += (
                " Respect refusals of preparation or discussion within their stated scope. "
                "Refusing rehearsal does not imply unreadiness for an independently chosen real action."
            )
        if any(item.discussion_declined for item in relevant_actions):
            guidance = (
                "Respect the refusal to discuss this topic. Acknowledge it briefly and follow a different "
                "coaching-user-owned concern if useful. Do not teach, rehearse, or ask why they are not ready "
                "to discuss the refused topic."
            )
        return (
            f"Current coaching stage: {decision.coaching_stage}. "
            f"Permitted next move: {move}. {guidance} {knowledge_guidance}"
            f"{move_guidance}{generation_guidance}{feedback_guidance}{action_guidance}"
            " Help the client see more clearly and discover their own available choices. "
            "Attend to role, relationship, authority, risk, prior attempts, and agency when relevant. "
            "Do not assume an ambitious goal implies a broken process. A question must not "
            "smuggle in a suggested solution. Reflect only when it builds trust or reveals "
            "a meaningful distinction, repairs a correction, or tests a useful synthesis. "
            "Update understanding internally on every turn; do not restate the latest answer "
            "as a routine preface. Demonstrate understanding through the relevance of the next move. "
            "Do not offer to pause or end merely because the client is uncertain or an inquiry "
            "stalls; respect an actual client request to stop. Avoid repetitive paraphrases and lists of questions. "
            "Select one useful invitation and leave room for the client's own thinking."
            " Keep coaching rules and internal restrictions out of the conversation. Do not announce "
            "that you cannot give advice or scripts; respond naturally with a useful client-owned invitation."
        )

    @classmethod
    def action_is_permitted(cls, working_state: CoachingWorkingState) -> bool:
        return cls.enforce(working_state).coaching_stage == "action"

    @classmethod
    def commitment_is_permitted(cls, working_state: CoachingWorkingState) -> bool:
        decision = cls.enforce(working_state)
        return decision.coaching_stage == "action" or (
            decision.next_coaching_move == "close_or_continue"
            and decision.closure_assessment is not None
        )

    @staticmethod
    def decision_reasons(proposed: CoachingWorkingState, decision: CoachingWorkingState) -> list[str]:
        if proposed.inquiry_assessment and proposed.inquiry_assessment.client_requested_pause:
            return ["explicit_stop"]
        reasons = []
        if decision.next_coaching_move == "support_choice":
            reasons.append("shared_understanding_and_scoped_support")
        if proposed.progress_assessment and proposed.progress_assessment.understanding_effect == "revise":
            reasons.append("material_revision")
        if proposed.next_coaching_move != decision.next_coaching_move or proposed.coaching_stage != decision.coaching_stage:
            reasons.append("permission_downgrade")
        return reasons or ["selected_move_permitted"]

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
        if working_state.understanding_context and working_state.understanding_context.status == "supported":
            # Repairing an invitation does not invalidate the supported picture.
            discard_transition_basis = tuple(item for item in discard_transition_basis
                                             if item != "tested_understanding")
            unknowns = [item for item in unknowns if item != "tested_understanding"]
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
