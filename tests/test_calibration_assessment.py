import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.coaching import CoachingTurnAssessment, CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService
from app.services.coaching_policy_service import CoachingPolicyService
from app.workflows.chat_workflow import WorkflowNodes


def assessment(message, **overrides):
    return CoachingTurnAssessment(
        focus_assessment={"focus": "Client clarity", "reason": "The client chooses what matters",
            "evidence_ids": ["current_message"] if message != "An invented quote" else ["invented"]},
        interaction_assessment=None, coaching_stage="discovery", next_coaching_move="explore_desired_help",
        client_response_to_last_move="not_applicable", client_generation="emerging",
        essential_unknowns=["desired_help"], transition_basis=["grounded_evidence"], **overrides,
        knowledge_gap_observed=False, user_owned_outcome=False, tested_understanding=False, transition_evidence_ids=[],
    )


class TurnAssessmentTests(unittest.IsolatedAsyncioTestCase):
    async def test_assessment_preserves_the_existing_answer_level_signal(self):
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=assessment("I choose this experiment", answer_level="commitment")))
        result = await CoachingMemoryService(llm).assess_turn(
            working_state=CoachingWorkingState(reported_facts=["Client chooses an experiment"]),
            user_message="I choose this experiment", latest_assistant_response="What do you choose?", history=[],
        )
        self.assertEqual(result.answer_level, "commitment")

    async def test_fresh_assessment_selects_focus_without_rewriting_client_facts(self):
        message = "I can negotiate priorities"
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=assessment(message)))
        state = CoachingWorkingState(reported_facts=[message], last_coaching_move="test_understanding")
        result = await CoachingMemoryService(llm).assess_turn(
            working_state=state, user_message=message, latest_assistant_response="What influence is available?", history=[],
        )
        self.assertEqual(result.reported_facts, state.reported_facts)
        self.assertEqual(result.last_coaching_move, "test_understanding")
        self.assertEqual(result.next_coaching_move, "explore_desired_help")
        self.assertEqual(result.focus_assessment.evidence[0].excerpt, message)

    async def test_bad_assessment_recovers_once_from_original_evidence(self):
        invalid = assessment("An invented quote")
        valid = assessment("I can negotiate priorities")
        llm = SimpleNamespace(generate_structured=AsyncMock(side_effect=[invalid, valid]))
        result = await CoachingMemoryService(llm).assess_turn(
            working_state=CoachingWorkingState(), user_message="I can negotiate priorities",
            latest_assistant_response=None, history=[],
        )
        self.assertEqual(llm.generate_structured.await_count, 2)
        self.assertEqual(result.focus_assessment.evidence[0].excerpt, "I can negotiate priorities")

    async def test_assessment_outage_preserves_new_facts_and_actual_last_move(self):
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.coaching_memory_service = SimpleNamespace(assess_turn=AsyncMock(side_effect=ConnectionError("private detail")))
        result = await nodes.assess_coaching_move({
            "user_id": "7", "message": "That misses the point", "metadata": {"working_state_status": "updated"},
            "working_state": CoachingWorkingState(reported_facts=["One colleague is new"],
                last_coaching_move="test_understanding", coaching_stage="action", next_coaching_move="develop_experiment").model_dump(),
        })
        self.assertEqual(result["working_state"]["reported_facts"], ["One colleague is new"])
        self.assertEqual(result["working_state"]["last_coaching_move"], "test_understanding")
        self.assertEqual(result["working_state"]["coaching_stage"], "discovery")
        self.assertNotIn("private detail", str(result))

    def test_material_refinement_can_earn_teaching_without_an_action_permission(self):
        state = CoachingWorkingState(reported_facts=["I have never learned the distinction"],
            next_coaching_move="teach_selectively", client_response_to_last_move="refined",
            transition_basis=["knowledge_gap_observed", "tested_understanding"])
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "teach_selectively")
        self.assertNotIn("tested_understanding", result.transition_basis)
        self.assertFalse(CoachingPolicyService.action_is_permitted(result))

    def test_explicit_corrective_request_can_earn_one_concise_explanation(self):
        state = CoachingWorkingState(reported_facts=["The client asks to learn a specific relevant distinction"],
            client_response_to_last_move="corrected", next_coaching_move="teach_selectively",
            transition_basis=["knowledge_gap_observed"], last_coaching_move="clarify")
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "teach_selectively")

    def test_unhelpful_question_can_be_repaired_by_earned_teaching(self):
        state = CoachingWorkingState.model_validate({
            "reported_facts": ["The client asks to learn a specific relevant distinction"],
            "client_response_to_last_move": "refined", "next_coaching_move": "teach_selectively",
            "transition_basis": ["knowledge_gap_observed", "tested_understanding"], "last_coaching_move": "clarify",
            "interaction_assessment": {"usefulness": "possibly_unhelpful", "observation": "Question missed the knowledge need",
                "evidence": {"source": "current_message", "excerpt": "Please explain that distinction"}},
        })
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "teach_selectively")
        self.assertNotIn("tested_understanding", result.transition_basis)
        self.assertEqual(CoachingPolicyService.enforce(result), result)

    async def test_ownership_is_checked_before_extracting_or_generating(self):
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.chat_history_service = SimpleNamespace(get_or_create_conversation=AsyncMock(side_effect=ValueError("Conversation not found")))
        nodes._fetch_history = AsyncMock()
        with self.assertRaises(ValueError):
            await nodes.retrieve_data({"user_id": "7", "message": "Attempt", "conversation_id": "foreign", "metadata": {}})
        nodes._fetch_history.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
