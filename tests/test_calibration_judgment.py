import unittest

from pydantic import ValidationError

from app.schemas.coaching import CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService
from app.services.coaching_policy_service import CoachingPolicyService


class ContextualJudgmentTests(unittest.TestCase):
    def test_legacy_state_defaults_assessments_to_none(self):
        state = CoachingWorkingState.model_validate({"reported_facts": ["One employee is new"]})
        self.assertIsNone(state.focus_assessment)
        self.assertIsNone(state.interaction_assessment)

    def test_unsupported_evidence_is_rejected(self):
        state = CoachingWorkingState.model_validate({"focus_assessment": {
            "focus": "Client's influence", "reason": "Role changes the available choices",
            "evidence": [{"source": "current_message", "excerpt": "I am the owner"}],
        }})
        with self.assertRaises(Exception) as error:
            CoachingMemoryService.validate_assessments(state, {
                "new_user_message": "I report to the director", "previous_state": {},
            })
        self.assertEqual(getattr(error.exception, "category", None), "unsupported_evidence")

    def test_assessment_is_bounded(self):
        with self.assertRaises(ValidationError):
            CoachingWorkingState.model_validate({"focus_assessment": {
                "focus": "x" * 501, "reason": "Relevant", "evidence": [],
            }})

    def test_possible_loss_of_usefulness_reopens_discovery_without_a_phrase_rule(self):
        state = CoachingWorkingState.model_validate({
            "reported_facts": ["Client is unsure where to begin after several suggestions"],
            "coaching_stage": "action", "next_coaching_move": "develop_experiment",
            "client_response_to_last_move": "refined",
            "transition_basis": ["tested_understanding", "user_owned_outcome"],
            "interaction_assessment": {"usefulness": "possibly_unhelpful",
                "observation": "The client is uncertain after Jess supplied several directions",
                "evidence": {"source": "current_message", "excerpt": "Where do I start?"}},
        })
        decision = CoachingPolicyService.enforce(state)
        self.assertEqual(decision.coaching_stage, "discovery")
        self.assertEqual(decision.next_coaching_move, "test_understanding")
        self.assertNotIn("tested_understanding", decision.transition_basis)

    def test_correction_preserves_a_useful_selected_shift(self):
        decision = CoachingPolicyService.enforce(CoachingWorkingState(
            reported_facts=["Only one employee is new"], client_response_to_last_move="corrected",
            next_coaching_move="shift_focus", transition_basis=["tested_understanding"],
        ))
        self.assertEqual(decision.next_coaching_move, "shift_focus")
        self.assertEqual(decision.coaching_stage, "discovery")

    def test_discovery_cannot_execute_an_action_move(self):
        decision = CoachingPolicyService.enforce(CoachingWorkingState(
            reported_facts=["The team is exceeding its targets"], next_coaching_move="explore_options",
        ))
        self.assertEqual(decision.next_coaching_move, "clarify")

    def test_new_turn_clears_provisional_assessments(self):
        state = CoachingWorkingState.model_validate({"interaction_assessment": {
            "usefulness": "possibly_unhelpful", "observation": "Tentative",
            "evidence": {"source": "current_message", "excerpt": "Hmm"}},
        })
        prepared = CoachingMemoryService.prepare_state_for_turn(state)
        self.assertIsNone(prepared.interaction_assessment)


if __name__ == "__main__":
    unittest.main()
