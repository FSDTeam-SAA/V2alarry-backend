import unittest
from itertools import permutations
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.coaching import CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService, WorkingStateExtractionError
from app.services.coaching_policy_service import CoachingPolicyService
from tests.test_cycle5_policy import conversation_action, possibility_state, progress, support


class Cycle45BoundaryTests(unittest.IsolatedAsyncioTestCase):
    def previous(self, **changes):
        return possibility_state(
            action_context=[conversation_action(**changes)],
            support_action_id="conversation-with-boss",
        )

    def update(self, state, message, **changes):
        action = state.action_context[0]
        return {
            "id": action.id, "actor": action.actor, "action": action.action,
            "status": None, "readiness": None, "basis": "readiness_boundary",
            "evidence": [{"source": "current_message", "excerpt": message}],
            **changes,
        }

    async def extract(self, previous, message, updates):
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
            "reported_facts": [message], "action_updates": updates,
        }))
        return await CoachingMemoryService(llm).update_working_state(
            previous_state=previous, user_message=message,
            latest_assistant_response="What support would be useful?", history=[],
        )

    async def test_lifted_preparation_refusal_allows_practice_without_changing_action_unreadiness(self):
        previous = self.previous(
            readiness="not_ready", boundary_scope="action", preparation_declined=True,
            preparation_evidence="I do not want to rehearse this conversation.",
        )
        message = "I would like to rehearse now, but I am still not ready for the real conversation."
        extracted = await self.extract(previous, message, [self.update(
            previous, message, readiness="ready", basis="readiness_confirmation", boundary_scope="preparation",
        )])
        action = extracted.action_context[0]
        self.assertFalse(action.preparation_declined)
        self.assertEqual(action.readiness, "not_ready")
        self.assertEqual(action.readiness_evidence, previous.action_context[0].readiness_evidence)
        assessed = self.previous().model_copy(update={"action_context": extracted.action_context})
        decision = CoachingPolicyService.enforce(assessed)
        self.assertEqual(decision.next_coaching_move, "support_choice")
        self.assertEqual(decision.coaching_stage, "possibilities")
        self.assertEqual(CoachingPolicyService.enforce(decision), decision)

    async def test_independent_action_and_preparation_refusals_merge_on_same_action(self):
        previous = self.previous(readiness="unknown", readiness_evidence=None)
        message = "I am not ready to have that conversation, and I do not want to rehearse it either."
        action_refusal = self.update(previous, "I am not ready to have that conversation",
                                     readiness="not_ready", boundary_scope="action")
        preparation_refusal = self.update(previous, "I do not want to rehearse it either",
                                          readiness="declined", boundary_scope="preparation")
        for updates in ([action_refusal, preparation_refusal], [preparation_refusal, action_refusal]):
            with self.subTest(order=[update["boundary_scope"] for update in updates]):
                state = await self.extract(previous, message, updates)
                self.assertEqual(len(state.action_context), 1)
                action = state.action_context[0]
                self.assertEqual(action.id, previous.action_context[0].id)
                self.assertEqual(action.readiness, "not_ready")
                self.assertTrue(action.preparation_declined)
                self.assertEqual(action.readiness_evidence, "I am not ready to have that conversation")
                self.assertEqual(action.preparation_evidence, "I do not want to rehearse it either")

    async def test_distinct_scope_readiness_updates_can_lift_practice_and_preserve_action_refusal(self):
        previous = self.previous(preparation_declined=True,
                                 preparation_evidence="No rehearsal for me.")
        message = "I am still not ready to do it, but I am happy to practice now."
        action_refusal = self.update(previous, "I am still not ready to do it",
                                     readiness="not_ready", boundary_scope="action")
        preparation_permission = self.update(previous, "I am happy to practice now",
            readiness="ready", basis="readiness_confirmation", boundary_scope="preparation")
        for updates in ([action_refusal, preparation_permission], [preparation_permission, action_refusal]):
            with self.subTest(order=[update["boundary_scope"] for update in updates]):
                state = await self.extract(previous, message, updates)
                self.assertEqual(state.action_context[0].readiness, "not_ready")
                self.assertFalse(state.action_context[0].preparation_declined)
                self.assertEqual(state.action_context[0].status, "considering")

    async def test_two_conflicting_updates_to_same_permission_are_rejected(self):
        previous = self.previous()
        message = "I do not want to rehearse. Actually, yes, I want to rehearse."
        updates = [
            self.update(previous, "I do not want to rehearse", readiness="declined", boundary_scope="preparation"),
            self.update(previous, "yes, I want to rehearse", readiness="ready",
                        basis="readiness_confirmation", boundary_scope="preparation"),
        ]
        with self.assertRaises(WorkingStateExtractionError):
            await self.extract(previous, message, updates)

    def test_topic_refusal_blocks_conceptual_help_on_that_topic(self):
        state = self.previous(
            readiness="ready", discussion_declined=True,
            discussion_evidence="I do not want to discuss or hear more explanations about this trust conversation.",
        ).model_copy(update={
            "coaching_stage": "discovery", "next_coaching_move": "teach_selectively",
            "requested_support": "conceptual_explanation", "last_coaching_move": "clarify",
            "transition_basis": ["grounded_evidence", "tested_understanding", "knowledge_gap_observed"],
        })
        result = CoachingPolicyService.enforce(state)
        self.assertNotEqual(result.next_coaching_move, "teach_selectively")
        self.assertTrue(result.action_context[0].discussion_declined)

    def test_topic_refusal_does_not_block_conceptual_help_on_a_different_topic(self):
        state = self.previous(readiness="ready", discussion_declined=True)
        other_action = state.action_context[0].model_copy(update={
            "id": "another-topic", "action": "delegate a report", "discussion_declined": False,
            "evidence": ["I want to understand delegation."],
        })
        state = state.model_copy(update={
            "action_context": [*state.action_context, other_action],
            "coaching_stage": "discovery", "next_coaching_move": "teach_selectively",
            "requested_support": "conceptual_explanation", "support_action_id": "another-topic",
            "last_coaching_move": "clarify",
            "transition_basis": ["grounded_evidence", "knowledge_gap_observed"],
        })
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "teach_selectively")

    def test_denied_execution_permission_cannot_consolidate_commitment(self):
        state = possibility_state(
            coaching_stage="action", next_coaching_move="consolidate_commitment",
            transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"],
            support_assessment=support(scope="execution", permitted=False,
                reason="The coaching user explicitly declines discussing or settling an action commitment now."),
            requested_support="action_planning",
            progress_assessment=progress(recommended_direction="consolidate"),
            action_context=[conversation_action(status="planned", readiness="ready")],
            support_action_id="conversation-with-boss",
        )
        result = CoachingPolicyService.enforce(state)
        self.assertNotEqual(result.coaching_stage, "action")
        self.assertNotEqual(result.next_coaching_move, "consolidate_commitment")
        self.assertEqual(result.action_context, state.action_context)

    async def test_first_turn_independent_scopes_share_one_server_assigned_action_id(self):
        message = "I am not ready to talk with the boss; I do not want to rehearse it or discuss it."
        template = self.previous()
        updates = [
            self.update(template, "I am not ready to talk with the boss", id=None,
                        readiness="not_ready", boundary_scope="action"),
            self.update(template, "I do not want to rehearse it", id=None,
                        readiness="declined", boundary_scope="preparation"),
            self.update(template, "or discuss it", id=None,
                        readiness="declined", boundary_scope="topic"),
        ]
        for ordered in permutations(updates):
            with self.subTest(order=[update["boundary_scope"] for update in ordered]):
                llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
                    "reported_facts": [message], "action_updates": list(ordered),
                }))
                state = await CoachingMemoryService(llm).update_working_state(
                    previous_state=CoachingWorkingState(), user_message=message,
                    latest_assistant_response=None, history=[],
                )
                self.assertEqual(len(state.action_context), 1)
                action = state.action_context[0]
                self.assertTrue(action.id)
                self.assertEqual(action.actor, template.action_context[0].actor)
                self.assertEqual(action.action, template.action_context[0].action)
                self.assertEqual(action.readiness, "not_ready")
                self.assertEqual(action.boundary_scope, "action")
                self.assertTrue(action.preparation_declined)
                self.assertTrue(action.discussion_declined)
                self.assertEqual(action.readiness_evidence, updates[0]["evidence"][0]["excerpt"])
                self.assertEqual(action.preparation_evidence, updates[1]["evidence"][0]["excerpt"])
                self.assertEqual(action.discussion_evidence, updates[2]["evidence"][0]["excerpt"])

    def test_denied_execution_repair_is_stable_across_feedback_and_inquiry_states(self):
        for feedback in ("not_applicable", "refined", "corrected", "not_ready", "declined"):
            for inquiry in ("advancing", "stalled"):
                with self.subTest(feedback=feedback, inquiry=inquiry):
                    state = possibility_state(
                        coaching_stage="action", next_coaching_move="consolidate_commitment",
                        client_response_to_last_move=feedback,
                        transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"],
                        support_assessment=support(scope="execution", permitted=False),
                        requested_support="action_planning",
                        progress_assessment=progress(recommended_direction="consolidate"),
                        inquiry_assessment={"progress": inquiry, "reason": "The current turn needs a useful response.",
                                            "client_requested_pause": False},
                        action_context=[conversation_action(status="planned", readiness="ready")],
                        support_action_id="conversation-with-boss",
                    )
                    first = CoachingPolicyService.enforce(state)
                    self.assertNotEqual(first.coaching_stage, "action")
                    self.assertEqual(CoachingPolicyService.enforce(first), first)

    def test_topic_refusal_repair_is_stable_even_when_teaching_would_otherwise_be_earned(self):
        state = self.previous(readiness="ready", discussion_declined=True).model_copy(update={
            "coaching_stage": "discovery", "next_coaching_move": "teach_selectively",
            "requested_support": "conceptual_explanation",
            "transition_basis": ["grounded_evidence", "tested_understanding", "knowledge_gap_observed"],
        })
        first = CoachingPolicyService.enforce(state)
        self.assertNotEqual(first.next_coaching_move, "teach_selectively")
        self.assertEqual(CoachingPolicyService.enforce(first), first)
        self.assertEqual(first.action_context, state.action_context)

    def test_execution_remains_permitted_after_independent_rehearsal_refusal(self):
        state = possibility_state(
            coaching_stage="action", next_coaching_move="consolidate_commitment",
            transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"],
            support_assessment=support(scope="execution", permitted=True),
            requested_support="action_planning",
            progress_assessment=progress(recommended_direction="consolidate"),
            action_context=[conversation_action(status="planned", readiness="ready",
                preparation_declined=True, preparation_evidence="I will speak to him without rehearsing.")],
            support_action_id="conversation-with-boss",
        )
        decision = CoachingPolicyService.enforce(state)
        self.assertEqual(decision.coaching_stage, "action")
        self.assertEqual(decision.next_coaching_move, "consolidate_commitment")
        self.assertEqual(CoachingPolicyService.enforce(decision), decision)
        self.assertNotIn("the real action remains not ready", CoachingPolicyService.build_response_policy(decision))


if __name__ == "__main__":
    unittest.main()
