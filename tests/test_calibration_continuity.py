import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.coaching import CoachingSummaryData, CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService


class ProvisionalContinuityTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_input_excludes_provisional_client_characterizations(self):
        llm = SimpleNamespace(generate=AsyncMock(return_value="{}"))
        state = CoachingWorkingState.model_validate({
            "reported_facts": ["The client paused after a long question"],
            "held_clues": [{"text": "May be losing trust", "status": "active"}],
            "interaction_assessment": {"usefulness": "possibly_unhelpful", "observation": "May be uncertain",
                "evidence": {"source": "current_message", "excerpt": "Hmm"}},
        })
        await CoachingMemoryService(llm).refresh_summary(
            previous_summary=None, working_state=state, user_message="Hmm", assistant_response="What would help now?",
        )
        request = json.loads(llm.generate.call_args.kwargs["user_message"])
        self.assertNotIn("interaction_assessment", request["working_state"])
        self.assertNotIn("focus_assessment", request["working_state"])
        self.assertNotIn("held_clues", request["working_state"])

    def test_tentative_feedback_cannot_promote_a_new_permanent_theme(self):
        state = CoachingWorkingState.model_validate({
            "reported_facts": ["The client acknowledged the question"], "transition_basis": ["tested_understanding"],
            "interaction_assessment": {"usefulness": "possibly_unhelpful", "observation": "Uncertain usefulness",
                "evidence": {"source": "current_message", "excerpt": "Hmm"}},
        })
        result = CoachingMemoryService._apply_retention_gate(
            previous_summary=None, candidate=CoachingSummaryData(developmental_theme="The client avoids conflict"),
            working_state=state,
        )
        self.assertIsNone(result.developmental_theme)


if __name__ == "__main__":
    unittest.main()
