import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.coaching import CoachingTurnAssessment, CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService
from app.services.coaching_policy_service import CoachingPolicyService
from app.workflows.chat_workflow import WorkflowNodes


def stalled_state(**overrides):
    return CoachingWorkingState.model_validate({
        "reported_facts": ["The client wants to understand a change in a trusted colleague"],
        "next_coaching_move": "explore_readiness",
        "last_coaching_move": "explore_desired_help",
        "client_response_to_last_move": "declined",
        "inquiry_assessment": {
            "progress": "stalled",
            "reason": "The client has no further answer to this question, but has not asked to stop",
            "client_requested_pause": False,
        },
        **overrides,
    })


class ConversationStewardshipTests(unittest.TestCase):
    def test_stalled_question_is_not_a_refusal_of_coaching(self):
        decision = CoachingPolicyService.enforce(stalled_state())
        self.assertEqual(decision.next_coaching_move, "shift_focus")
        self.assertEqual(decision.coaching_stage, "discovery")
        self.assertEqual(CoachingPolicyService.enforce(decision), decision)
        self.assertFalse(CoachingPolicyService.action_is_permitted(decision))

    def test_stalled_reflection_shifts_without_repeating_understanding_check(self):
        decision = CoachingPolicyService.enforce(stalled_state(
            next_coaching_move="test_understanding", last_coaching_move="test_understanding",
            client_response_to_last_move="not_applicable",
            transition_basis=["tested_understanding", "user_owned_outcome"],
        ))
        self.assertEqual(decision.next_coaching_move, "shift_focus")
        self.assertNotIn("tested_understanding", decision.transition_basis)

    def test_explicit_pause_is_respected(self):
        state = stalled_state(inquiry_assessment={
            "progress": "stalled", "reason": "Client explicitly asks to stop for today",
            "client_requested_pause": True,
        })
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "explore_readiness")

    def test_uncertain_usefulness_alone_does_not_force_a_reflection(self):
        state = CoachingWorkingState.model_validate({
            "reported_facts": ["The client has not noticed any other changes"],
            "next_coaching_move": "clarify",
            "interaction_assessment": {
                "usefulness": "unknown", "observation": "The reply gives little new information",
                "evidence": {"source": "current_message", "excerpt": "Nothing else"},
            },
        })
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "clarify")

    def test_stalled_question_shifts_focus_while_teaching_is_deferred(self):
        state = stalled_state(
            next_coaching_move="teach_selectively", client_response_to_last_move="refined",
            transition_basis=["knowledge_gap_observed"],
        )
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "shift_focus")

    def test_requested_explanation_does_not_reopen_declined_direction(self):
        state = stalled_state(
            next_coaching_move="teach_selectively", client_response_to_last_move="not_ready",
            transition_basis=["knowledge_gap_observed"],
            inquiry_assessment={"progress": "stalled", "reason": "The client asks for a distinction before exploring",
                                "client_requested_pause": False, "client_requested_teaching": True},
        )
        decision = CoachingPolicyService.enforce(state)
        self.assertEqual(decision.next_coaching_move, "shift_focus")
        self.assertEqual(CoachingPolicyService.enforce(decision), decision)

    def test_unrequested_teaching_does_not_bypass_a_declined_direction(self):
        state = stalled_state(next_coaching_move="teach_selectively",
                              transition_basis=["knowledge_gap_observed"])
        decision = CoachingPolicyService.enforce(state)
        self.assertEqual(decision.next_coaching_move, "shift_focus")
        self.assertEqual(CoachingPolicyService.enforce(decision), decision)

    def test_pause_wins_even_when_teaching_is_also_requested(self):
        state = stalled_state(
            next_coaching_move="teach_selectively", transition_basis=["knowledge_gap_observed"],
            inquiry_assessment={"progress": "stalled", "reason": "Client asks to end this session",
                                "client_requested_pause": True, "client_requested_teaching": True},
        )
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "explore_readiness")

    def test_new_turn_clears_inquiry_assessment(self):
        self.assertIsNone(CoachingMemoryService.prepare_state_for_turn(stalled_state()).inquiry_assessment)

    def test_productive_client_insight_is_not_interrupted_for_variety(self):
        state = stalled_state(client_generation="productive",
                              client_response_to_last_move="not_applicable",
                              next_coaching_move="reflect")
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "reflect")

    def test_stalled_nonanswer_is_not_treated_as_unwillingness(self):
        state = stalled_state(client_response_to_last_move="not_ready")
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "shift_focus")

    def test_unreadiness_without_a_stalled_inquiry_keeps_its_boundary(self):
        state = stalled_state(client_response_to_last_move="not_ready", inquiry_assessment=None)
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "explore_readiness")

    def test_correction_keeps_its_selected_repair(self):
        state = stalled_state(client_response_to_last_move="corrected", next_coaching_move="clarify")
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "clarify")

    def test_no_grounded_evidence_does_not_earn_a_hypothetical_focus_shift(self):
        state = stalled_state(reported_facts=[], client_response_to_last_move="not_applicable")
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "clarify")


class InquiryAssessmentBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_per_turn_trace_records_stall_without_public_metadata_leakage(self):
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.coaching_memory_service = SimpleNamespace(assess_turn=AsyncMock(return_value=stalled_state()))
        result = await nodes.assess_coaching_move({
            "user_id": "7", "message": "Not really", "working_state": stalled_state().model_dump(),
            "metadata": {"working_state_status": "updated"},
        })
        self.assertEqual(result["decision_trace"]["version"], "calibration-5")
        self.assertEqual(result["decision_trace"]["inquiry_assessment"]["progress"], "stalled")
        self.assertEqual(result["decision_trace"]["proposed_move"], "explore_readiness")
        self.assertEqual(result["decision_trace"]["enforced_move"], "shift_focus")
        self.assertNotIn("inquiry_assessment", result["metadata"])

    async def test_assessment_reaches_policy_without_rewriting_client_facts(self):
        candidate = CoachingTurnAssessment(
            focus_assessment={"focus": "The working relationship", "reason": "It remains unexplored",
                              "evidence_ids": ["previous_facts:0"]},
            interaction_assessment=None, inquiry_assessment=stalled_state().inquiry_assessment,
            coaching_stage="discovery", next_coaching_move="explore_readiness",
            client_response_to_last_move="declined", client_generation="emerging",
            essential_unknowns=[], transition_basis=[], knowledge_gap_observed=False,
            user_owned_outcome=False, tested_understanding=False, transition_evidence_ids=[],
        )
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=candidate))
        state = stalled_state(inquiry_assessment=None)
        result = await CoachingMemoryService(llm).assess_turn(
            working_state=state, user_message="Not really", latest_assistant_response="Any other clues?",
            history=[], previous_state=state,
        )
        self.assertEqual(result.reported_facts, state.reported_facts)
        self.assertEqual(result.inquiry_assessment.progress, "stalled")
        self.assertEqual(CoachingPolicyService.enforce(result).next_coaching_move, "shift_focus")

    async def test_summary_does_not_retain_inquiry_judgments(self):
        llm = SimpleNamespace(generate=AsyncMock(return_value="{}"))
        await CoachingMemoryService(llm).refresh_summary(
            previous_summary=None, working_state=stalled_state(),
            user_message="Not really", assistant_response="How has the relationship developed?",
        )
        request = json.loads(llm.generate.call_args.kwargs["user_message"])
        self.assertNotIn("inquiry_assessment", request["working_state"])

    def test_fresh_conversation_cannot_inherit_a_pause_request(self):
        state = stalled_state(inquiry_assessment={
            "progress": "stalled", "reason": "An invalid fresh-turn assessment",
            "client_requested_pause": True,
        })
        result = CoachingMemoryService.validate_assessments(state, {
            "new_user_message": "I want to understand this", "previous_state": {},
        })
        self.assertIsNone(result.inquiry_assessment)


if __name__ == "__main__":
    unittest.main()
