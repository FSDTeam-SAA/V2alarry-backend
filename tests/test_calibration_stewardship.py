import unittest

from app.schemas.coaching import CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService
from app.services.coaching_policy_service import CoachingPolicyService


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

    def test_earned_teaching_still_repairs_a_stalled_question(self):
        state = stalled_state(
            next_coaching_move="teach_selectively", client_response_to_last_move="refined",
            transition_basis=["knowledge_gap_observed"],
        )
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "teach_selectively")

    def test_new_turn_clears_inquiry_assessment(self):
        self.assertIsNone(CoachingMemoryService.prepare_state_for_turn(stalled_state()).inquiry_assessment)


if __name__ == "__main__":
    unittest.main()
