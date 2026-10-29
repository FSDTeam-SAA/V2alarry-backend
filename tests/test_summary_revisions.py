import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from pydantic import ValidationError

from app.schemas.coaching import CoachingSummaryData, CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService, WorkingStateExtractionError
from tests.test_action_context import action_update


class SummaryRevisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_legacy_generation_receives_the_revision_schema(self):
        llm = SimpleNamespace(generate=AsyncMock(return_value="{}"))
        await CoachingMemoryService(llm).refresh_summary(previous_summary=None,
            working_state=CoachingWorkingState(), user_message="I am reflecting.", assistant_response="Go on.")
        import json
        request = json.loads(llm.generate.call_args.kwargs["user_message"])
        self.assertIn("commitment_change", request["schema"]["properties"])

    async def test_explicit_abandonment_clears_only_matching_commitment(self):
        message = "I have decided not to arrange that meeting."
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
            "reported_facts": [message], "action_updates": [action_update(message,
                actor="client", action="meet the boss", status="abandoned", basis="reported_action")],
        }))
        service = CoachingMemoryService(llm)
        state = await service.update_working_state(previous_state=CoachingWorkingState(), user_message=message)
        previous = CoachingSummaryData(commitment="Arrange a meeting with the boss.",
                                       next_experiment="Notice when I feel ignored.")
        llm.generate_structured.return_value = {
            "commitment_change": {"operation": "clear", "previous_value": previous.commitment,
                "action_id": state.action_context[0].id,
                "evidence": [{"source": "current_message", "excerpt": message}]},
        }
        result = await service.refresh_summary(previous_summary=previous, working_state=state,
            user_message=message, assistant_response="What matters more to you now?")
        self.assertIsNone(result.commitment)
        self.assertEqual(result.next_experiment, previous.next_experiment)

    async def test_a_mismatched_clear_fails_instead_of_erasing_another_commitment(self):
        message = "I have decided to stop observing my reactions."
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
            "commitment_change": {"operation": "clear", "previous_value": "A different commitment",
                "action_id": "observe-reactions", "evidence": [{"source": "current_message", "excerpt": message}]},
        }))
        state = CoachingWorkingState(action_context=[{"id": "observe-reactions", "actor": "client",
            "action": "observe my reactions", "status": "abandoned", "status_evidence": message, "evidence": [message]}])
        with self.assertRaises(WorkingStateExtractionError):
            await CoachingMemoryService(llm).refresh_summary(
                previous_summary=CoachingSummaryData(commitment="Observe my reactions."),
                working_state=state, user_message=message, assistant_response="Understood.")

    async def test_replace_requires_client_evidence_and_existing_action_gate(self):
        message = "I choose to observe one reaction tomorrow."
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
            "commitment": "Observe one reaction tomorrow.",
            "commitment_change": {"operation": "replace", "previous_value": None,
                "action_id": None, "evidence": [{"source": "current_message", "excerpt": message}]},
        }))
        service = CoachingMemoryService(llm)
        for action_allowed in (False, True):
            state = CoachingWorkingState(reported_facts=[message],
                coaching_stage="action" if action_allowed else "discovery",
                next_coaching_move="develop_experiment" if action_allowed else "clarify",
                transition_basis=["tested_understanding", "user_owned_outcome"] if action_allowed else [])
            if not action_allowed:
                with self.assertRaises(ValidationError):
                    await service.refresh_summary(previous_summary=None, working_state=state,
                        user_message=message, assistant_response="What will you notice?")
            else:
                result = await service.refresh_summary(previous_summary=None, working_state=state,
                    user_message=message, assistant_response="What will you notice?")
                self.assertEqual(result.commitment, "Observe one reaction tomorrow.")
