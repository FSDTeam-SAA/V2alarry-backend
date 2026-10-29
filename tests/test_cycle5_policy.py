import unittest

from app.schemas.coaching import CoachingWorkingState
from app.services.coaching_policy_service import CoachingPolicyService


def evidence(excerpt="I want to understand why he stopped trusting me."):
    return [{"source": "current_message", "excerpt": excerpt}]


def supported_understanding(**changes):
    return {
        "picture": "The coaching user wants to understand the change in trust before deciding what to do.",
        "evidence": evidence("Exactly; I want to understand what changed before choosing what to do."),
        "basis": "natural",
        "status": "supported",
        **changes,
    }


def progress(**changes):
    return {
        "established": ["The purpose is understanding the change in trust."],
        "material_unknown": None,
        "discovery_useful": False,
        "understanding_effect": "retain",
        "recommended_direction": "support",
        "reason": "The coaching user has established the purpose and now requests preparation.",
        "evidence": evidence("Can we practice how I might open that conversation?"),
        **changes,
    }


def support(**changes):
    return {
        "scope": "preparation",
        "permitted": True,
        "reason": "The coaching user explicitly requests practice, without committing to the conversation.",
        "evidence": evidence("Can we practice how I might open that conversation?"),
        **changes,
    }


def possibility_state(**changes):
    return CoachingWorkingState.model_validate({
        "reported_facts": ["The coaching user wants to understand a change in their boss's trust."],
        "coaching_stage": "possibilities",
        "next_coaching_move": "support_choice",
        "last_coaching_move": "clarify",
        "client_response_to_last_move": "not_applicable",
        "transition_basis": ["grounded_evidence", "desired_help_known"],
        "understanding_context": supported_understanding(),
        "progress_assessment": progress(),
        "support_assessment": support(),
        "requested_support": "rehearsal",
        **changes,
    })


def conversation_action(**changes):
    return {
        "id": "conversation-with-boss",
        "actor": "coaching user",
        "action": "talk with the boss about trust",
        "status": "considering",
        "readiness": "not_ready",
        "boundary_scope": "action",
        "evidence": ["I am not ready to have that conversation, but I would like to practice."],
        "readiness_evidence": "I am not ready to have that conversation, but I would like to practice.",
        **changes,
    }


class Cycle5PolicyTests(unittest.TestCase):
    def test_understanding_repair_does_not_also_require_a_new_discovery_question(self):
        state = CoachingWorkingState(reported_facts=["The review is only a possibility."],
            next_coaching_move="test_understanding", client_response_to_last_move="corrected")
        policy = CoachingPolicyService.build_response_policy(state)
        self.assertNotIn("reopen discovery with one precise question", policy)
        self.assertIn("invite", policy.lower())

    def test_stalled_unhelpful_inquiry_repairs_picture_before_more_discovery(self):
        for feedback in ("refined", "corrected", "not_ready", "declined"):
            with self.subTest(feedback=feedback):
                state = CoachingWorkingState.model_validate({
                    "reported_facts": ["The client has identified a possible review; it has not happened."],
                    "next_coaching_move": "shift_focus",
                    "client_response_to_last_move": feedback,
                    "inquiry_assessment": {"progress": "stalled", "reason": "The question assumed a result that does not exist.", "client_requested_pause": False},
                    "interaction_assessment": {"usefulness": "possibly_unhelpful", "observation": "The client corrected the premise of the question.", "evidence": evidence("We have not held that review.")[0]},
                })
                result = CoachingPolicyService.enforce(state)
                self.assertEqual(result.next_coaching_move, "test_understanding")
                self.assertEqual(CoachingPolicyService.enforce(result), result)

    def test_requested_preparation_without_shared_picture_tests_understanding_not_readiness(self):
        state = possibility_state(
            understanding_context=None,
            progress_assessment=progress(understanding_effect="unknown"),
            action_context=[conversation_action()], support_action_id="conversation-with-boss",
        )
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "test_understanding")
        self.assertEqual(result.action_context, state.action_context)
        self.assertFalse(CoachingPolicyService.action_is_permitted(result))
        self.assertEqual(CoachingPolicyService.enforce(result), result)

    def test_stalled_inquiry_preserves_selected_useful_synthesis(self):
        state = CoachingWorkingState.model_validate({
            "reported_facts": ["The possible team meeting has not happened; there are no results yet."],
            "coaching_stage": "reflection",
            "next_coaching_move": "test_understanding",
            "last_coaching_move": "clarify",
            "inquiry_assessment": {
                "progress": "stalled", "reason": "Further questions ask for results from a future meeting.",
                "client_requested_pause": False,
            },
        })
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "test_understanding")

    def test_natural_understanding_allows_preparation_without_commitment(self):
        result = CoachingPolicyService.enforce(possibility_state())
        self.assertEqual(result.coaching_stage, "possibilities")
        self.assertEqual(result.next_coaching_move, "support_choice")
        self.assertNotIn("user_owned_outcome", result.transition_basis)
        self.assertFalse(CoachingPolicyService.action_is_permitted(result))

    def test_action_only_unreadiness_allows_requested_practice_without_changing_action(self):
        state = possibility_state(
            action_context=[conversation_action()], support_action_id="conversation-with-boss",
            client_response_to_last_move="not_ready",
        )
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "support_choice")
        self.assertEqual(result.coaching_stage, "possibilities")
        self.assertEqual(result.action_context, state.action_context)
        self.assertEqual(result.action_context[0].readiness, "not_ready")
        self.assertEqual(result.action_context[0].status, "considering")

    def test_refused_preparation_blocks_support(self):
        for scope in ("preparation", "topic"):
            with self.subTest(scope=scope):
                result = CoachingPolicyService.enforce(possibility_state(
                    action_context=[conversation_action(boundary_scope=scope, readiness="declined")],
                    support_action_id="conversation-with-boss",
                ))
                self.assertNotEqual(result.next_coaching_move, "support_choice")
                self.assertFalse(CoachingPolicyService.action_is_permitted(result))

    def test_abandoned_direction_does_not_become_rehearsal(self):
        result = CoachingPolicyService.enforce(possibility_state(
            action_context=[conversation_action(status="abandoned")],
            support_action_id="conversation-with-boss",
        ))
        self.assertNotEqual(result.next_coaching_move, "support_choice")

    def test_execution_still_requires_readiness(self):
        result = CoachingPolicyService.enforce(possibility_state(
            coaching_stage="action", next_coaching_move="develop_experiment",
            transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"],
            support_assessment=support(scope="execution"), requested_support="action_planning",
            action_context=[conversation_action()], support_action_id="conversation-with-boss",
        ))
        self.assertNotEqual(result.coaching_stage, "action")

    def test_execution_still_requires_coaching_user_ownership(self):
        result = CoachingPolicyService.enforce(possibility_state(
            coaching_stage="action", next_coaching_move="develop_experiment",
            support_assessment=support(scope="execution"), requested_support="action_planning",
            action_context=[conversation_action(readiness="ready")],
            support_action_id="conversation-with-boss",
        ))
        self.assertNotEqual(result.coaching_stage, "action")

    def test_compatible_refinement_preserves_understanding_and_support(self):
        state = possibility_state(
            client_response_to_last_move="refined",
            transition_basis=["grounded_evidence", "desired_help_known", "tested_understanding"],
            progress_assessment=progress(understanding_effect="retain"),
        )
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.understanding_context, state.understanding_context)
        self.assertIn("tested_understanding", result.transition_basis)
        self.assertEqual(result.next_coaching_move, "support_choice")

    def test_material_correction_reopens_understanding_before_support(self):
        result = CoachingPolicyService.enforce(possibility_state(
            client_response_to_last_move="corrected",
            transition_basis=["grounded_evidence", "tested_understanding"],
            progress_assessment=progress(
                understanding_effect="revise", recommended_direction="discovery",
                discovery_useful=True, material_unknown="What actually changed in the working relationship?",
            ),
        ))
        self.assertNotIn("tested_understanding", result.transition_basis)
        self.assertNotEqual(result.next_coaching_move, "support_choice")
        self.assertTrue(result.understanding_context is None or result.understanding_context.status == "needs_revision")

    def test_support_requires_supported_understanding(self):
        for context in (None, supported_understanding(status="needs_revision")):
            with self.subTest(context=context):
                result = CoachingPolicyService.enforce(possibility_state(understanding_context=context))
                self.assertNotEqual(result.next_coaching_move, "support_choice")

    def test_support_requires_separate_permission_evidence(self):
        result = CoachingPolicyService.enforce(possibility_state(
            support_assessment=support(permitted=False, reason="No request or permission for preparation is established."),
        ))
        self.assertNotEqual(result.next_coaching_move, "support_choice")

    def test_explicit_stop_wins_over_understanding_and_support(self):
        result = CoachingPolicyService.enforce(possibility_state(inquiry_assessment={
            "progress": "advancing", "reason": "The coaching user asks to stop for today.",
            "client_requested_pause": True,
        }))
        self.assertNotEqual(result.next_coaching_move, "support_choice")
        policy = CoachingPolicyService.build_response_policy(result)
        self.assertIn("without another discovery question", policy)

    def test_repeated_policy_enforcement_preserves_a_valid_support_decision(self):
        first = CoachingPolicyService.enforce(possibility_state(
            action_context=[conversation_action()], support_action_id="conversation-with-boss",
        ))
        self.assertEqual(first.next_coaching_move, "support_choice")
        self.assertEqual(CoachingPolicyService.enforce(first), first)

    def test_clarity_without_a_chosen_step_continues_client_led_exploration(self):
        result = CoachingPolicyService.enforce(possibility_state(
            next_coaching_move="close_or_continue", requested_support="none", support_assessment=None,
            progress_assessment=progress(recommended_direction="consolidate"),
        ))
        self.assertEqual(result.next_coaching_move, "explore_options")
        self.assertNotIn("user_owned_outcome", result.transition_basis)

    def test_preparation_permission_cannot_authorize_an_action_stage(self):
        state = possibility_state(
            coaching_stage="action", next_coaching_move="develop_experiment",
            transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"],
            support_assessment=support(scope="preparation"), requested_support="rehearsal",
            action_context=[conversation_action(readiness="ready")],
            support_action_id="conversation-with-boss",
        )
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.coaching_stage, "possibilities")
        self.assertEqual(result.next_coaching_move, "support_choice")
        self.assertEqual(result.action_context, state.action_context)

    def test_new_progress_cannot_consolidate_a_merely_considered_action(self):
        state = possibility_state(
            coaching_stage="action", next_coaching_move="consolidate_commitment",
            transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"],
            support_assessment=support(scope="execution"), requested_support="action_planning",
            progress_assessment=progress(recommended_direction="consolidate"),
            action_context=[conversation_action(status="considering", readiness="ready")],
            support_action_id="conversation-with-boss",
        )
        result = CoachingPolicyService.enforce(state)
        self.assertNotEqual(result.coaching_stage, "action")
        self.assertNotEqual(result.next_coaching_move, "consolidate_commitment")
        for status in ("planned", "ongoing"):
            with self.subTest(status=status):
                chosen = state.model_copy(update={
                    "action_context": [state.action_context[0].model_copy(update={"status": status})],
                })
                self.assertEqual(CoachingPolicyService.enforce(chosen).next_coaching_move, "consolidate_commitment")
        legacy = state.model_copy(update={"progress_assessment": None})
        self.assertEqual(CoachingPolicyService.enforce(legacy).next_coaching_move, "consolidate_commitment")

    def test_preparation_and_discussion_refusals_block_support_even_when_action_ready(self):
        for boundary in ("preparation_declined", "discussion_declined"):
            with self.subTest(boundary=boundary):
                state = possibility_state(
                    action_context=[conversation_action(readiness="ready", **{boundary: True})],
                    support_action_id="conversation-with-boss",
                )
                result = CoachingPolicyService.enforce(state)
                self.assertNotEqual(result.next_coaching_move, "support_choice")
                self.assertEqual(result.action_context[0].readiness, "ready")

    def test_refusing_practice_does_not_block_a_chosen_ready_execution(self):
        state = possibility_state(
            coaching_stage="action", next_coaching_move="consolidate_commitment",
            transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"],
            support_assessment=support(scope="execution"), requested_support="action_planning",
            progress_assessment=progress(recommended_direction="consolidate"),
            action_context=[conversation_action(
                status="planned", readiness="ready", boundary_scope="preparation",
                preparation_declined=True, preparation_evidence="I will do it without rehearsal.",
            )],
            support_action_id="conversation-with-boss",
        )
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.coaching_stage, "action")
        self.assertEqual(result.next_coaching_move, "consolidate_commitment")
        self.assertEqual(result.action_context, state.action_context)

    def test_repeated_enforcement_keeps_material_correction_repair_consistent(self):
        state = possibility_state(
            client_response_to_last_move="corrected",
            progress_assessment=progress(
                understanding_effect="revise", recommended_direction="synthesis",
                reason="The user corrects the shared picture; offer the revised picture before preparation.",
            ),
        )
        first = CoachingPolicyService.enforce(state)
        second = CoachingPolicyService.enforce(first)
        self.assertEqual(first.next_coaching_move, "test_understanding")
        self.assertEqual(first.understanding_context.status, "needs_revision")
        self.assertNotEqual(first.next_coaching_move, "support_choice")
        self.assertEqual(second, first)
        self.assertEqual(CoachingPolicyService.build_response_policy(second),
                         CoachingPolicyService.build_response_policy(first))

    def test_stalled_question_retains_supported_picture_and_stable_repair(self):
        state = possibility_state(next_coaching_move="test_understanding",
            client_response_to_last_move="declined",
            progress_assessment=progress(understanding_effect="retain"),
            inquiry_assessment={"progress": "stalled", "reason": "The user declined this repeated question",
                                "client_requested_pause": False})
        decision = CoachingPolicyService.enforce(state)
        self.assertEqual(decision.next_coaching_move, "shift_focus")
        self.assertEqual(decision.understanding_context, state.understanding_context)
        self.assertIn("tested_understanding", decision.transition_basis)
        self.assertNotIn("tested_understanding", decision.essential_unknowns)
        self.assertEqual(CoachingPolicyService.enforce(decision), decision)

    def test_exploration_label_cannot_bypass_a_wording_refusal(self):
        state = possibility_state(requested_support="action_wording",
            support_assessment=support(scope="exploration"),
            action_context=[conversation_action(readiness="ready", preparation_declined=True)],
            support_action_id="conversation-with-boss")
        self.assertNotEqual(CoachingPolicyService.enforce(state).next_coaching_move, "support_choice")

    def test_denied_support_applies_stalled_inquiry_repair_on_first_enforcement(self):
        state = possibility_state(support_assessment=support(permitted=False),
            inquiry_assessment={"progress": "stalled", "reason": "This question has yielded no new understanding",
                                "client_requested_pause": False})
        decision = CoachingPolicyService.enforce(state)
        self.assertEqual(decision.next_coaching_move, "shift_focus")
        self.assertEqual(CoachingPolicyService.enforce(decision), decision)

    def test_ready_execution_guidance_does_not_confuse_a_refusal_of_rehearsal(self):
        state = possibility_state(coaching_stage="action", next_coaching_move="consolidate_commitment",
            transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"],
            support_assessment=support(scope="execution"), requested_support="action_planning",
            progress_assessment=progress(recommended_direction="consolidate"),
            action_context=[conversation_action(status="planned", readiness="ready",
                preparation_declined=True, preparation_evidence="I will do it without rehearsal.")],
            support_action_id="conversation-with-boss")
        policy = CoachingPolicyService.build_response_policy(state)
        self.assertIn("consolidate commitment", policy)
        self.assertNotIn("the real action remains not ready", policy)
        self.assertIn("Respect refusals of preparation", policy)


if __name__ == "__main__":
    unittest.main()
