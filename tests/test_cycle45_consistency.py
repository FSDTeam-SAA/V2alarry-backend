import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.coaching import CoachingSummaryData, CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService, WorkingStateExtractionError, WORKING_STATE_PROMPT
from app.services.coaching_policy_service import CoachingPolicyService
from tests.test_cycle5_policy import conversation_action, possibility_state, progress, support


class CombinedCalibrationConsistencyTests(unittest.IsolatedAsyncioTestCase):
    async def test_permission_to_practice_cannot_change_real_action_status(self):
        previous = possibility_state(action_context=[conversation_action()])
        message = "We can practice now, but I have not chosen to have that conversation."
        invalid = {"reported_facts": [message], "action_updates": [{
            "id": "conversation-with-boss", "actor": "coaching user", "action": "talk with the boss about trust",
            "status": "planned", "readiness": "ready", "boundary_scope": "preparation",
            "basis": "readiness_confirmation", "evidence": [{"source": "current_message", "excerpt": message}],
        }]}
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=invalid))
        with self.assertRaises(WorkingStateExtractionError):
            await CoachingMemoryService(llm).update_working_state(previous_state=previous,
                user_message=message, latest_assistant_response="Would practice help?", history=[])
        self.assertEqual(llm.generate_structured.await_count, 2)

    def test_exploration_permission_does_not_instruct_wording_or_rehearsal(self):
        state = possibility_state(requested_support="exploration", support_assessment=support(scope="exploration"))
        policy = CoachingPolicyService.build_response_policy(state)
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "support_choice")
        self.assertNotIn("offer one brief adaptable example or rehearse", policy)
        self.assertIn("Exploration permission does not authorize wording", policy)

    def test_unrelated_unready_action_does_not_forbid_a_chosen_ready_commitment(self):
        chosen = conversation_action(status="planned", readiness="ready")
        other = conversation_action(id="other-action", action="present to the board", readiness="not_ready")
        state = possibility_state(coaching_stage="action", next_coaching_move="consolidate_commitment",
            action_context=[chosen, other], support_action_id=chosen["id"], requested_support="action_planning",
            support_assessment=support(scope="execution"), progress_assessment=progress(recommended_direction="consolidate"),
            transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"])
        policy = CoachingPolicyService.build_response_policy(state)
        self.assertEqual(CoachingPolicyService.enforce(state).coaching_stage, "action")
        self.assertNotIn("the real action remains not ready; do not schedule", policy)

    def test_legacy_prompt_does_not_require_explicit_confirmation_only(self):
        self.assertNotIn("Record tested_understanding only when the client has confirmed", WORKING_STATE_PROMPT)

    async def test_summary_cannot_borrow_permission_from_another_chosen_action(self):
        chosen = conversation_action(status="planned", readiness="ready")
        practice_only = conversation_action(id="board-presentation", action="present to the board")
        state = possibility_state(coaching_stage="action", next_coaching_move="consolidate_commitment",
            action_context=[chosen, practice_only], support_action_id=chosen["id"],
            support_assessment=support(scope="execution"), requested_support="action_planning",
            transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"])
        message = "I choose to speak with my boss. The board presentation is only something I might practice."
        for action_id in (None, "board-presentation"):
            with self.subTest(action_id=action_id):
                llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
                    "commitment": "Present to the board.", "commitment_change": {
                        "operation": "replace", "previous_value": None, "action_id": action_id,
                        "evidence": [{"source": "current_message", "excerpt": message}]}}))
                with self.assertRaises(WorkingStateExtractionError):
                    await CoachingMemoryService(llm).refresh_summary(previous_summary=None,
                        working_state=state, user_message=message, assistant_response="Let's retain your choice.")

    async def test_summary_can_store_the_matching_chosen_action(self):
        chosen = conversation_action(status="planned", readiness="ready")
        state = possibility_state(coaching_stage="action", next_coaching_move="consolidate_commitment",
            action_context=[chosen], support_action_id=chosen["id"], support_assessment=support(scope="execution"),
            requested_support="action_planning", transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"])
        message = "I choose to speak with my boss tomorrow."
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
            "commitment": "Speak with my boss tomorrow.", "commitment_change": {
                "operation": "replace", "previous_value": None, "action_id": chosen["id"],
                "evidence": [{"source": "current_message", "excerpt": message}]}}))
        result = await CoachingMemoryService(llm).refresh_summary(previous_summary=CoachingSummaryData(),
            working_state=state, user_message=message, assistant_response="That's your choice.")
        self.assertEqual(result.commitment, "Speak with my boss tomorrow.")

    def test_requested_exploration_can_continue_without_readiness_or_practice_permission(self):
        action = conversation_action(preparation_declined=True)
        state = possibility_state(action_context=[action], support_action_id=action["id"],
            requested_support="exploration", support_assessment=support(scope="exploration",
                reason="The user wants to explore what the conversation might mean, without wording or a plan."))
        decision = CoachingPolicyService.enforce(state)
        self.assertEqual(decision.next_coaching_move, "support_choice")
        self.assertEqual(decision.coaching_stage, "possibilities")
        self.assertEqual(decision.action_context, state.action_context)
        self.assertEqual(CoachingPolicyService.enforce(decision), decision)
        self.assertIn("Exploration permission does not authorize wording", CoachingPolicyService.build_response_policy(decision))

    def test_exploration_permission_cannot_authorize_execution(self):
        state = possibility_state(coaching_stage="action", next_coaching_move="consolidate_commitment",
            action_context=[conversation_action(status="planned", readiness="ready")],
            support_action_id="conversation-with-boss", requested_support="exploration",
            support_assessment=support(scope="exploration"),
            transition_basis=["grounded_evidence", "tested_understanding", "user_owned_outcome"])
        decision = CoachingPolicyService.enforce(state)
        self.assertEqual(decision.coaching_stage, "possibilities")
        self.assertEqual(decision.next_coaching_move, "support_choice")

    def test_abandoned_direction_repair_keeps_diagnostic_basis_stable(self):
        for feedback in ("not_applicable", "declined", "corrected", "refined"):
            for inquiry in ("advancing", "stalled"):
                with self.subTest(feedback=feedback, inquiry=inquiry):
                    state = possibility_state(action_context=[conversation_action(status="abandoned")],
                        support_action_id="conversation-with-boss", requested_support="exploration",
                        support_assessment=support(scope="exploration"), client_response_to_last_move=feedback,
                        inquiry_assessment={"progress": inquiry, "reason": "Assess the declined direction",
                                            "client_requested_pause": False})
                    first = CoachingPolicyService.enforce(state)
                    self.assertEqual(CoachingPolicyService.enforce(first), first)


if __name__ == "__main__":
    unittest.main()
