import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.coaching import CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService
from app.core.llm import LLMService


class ExtractionRecoveryTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_native_structure_is_used_and_validated(self):
        state = CoachingWorkingState(reported_facts=["I own priorities"])
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=state))
        result = await CoachingMemoryService(llm).update_working_state(
            previous_state=CoachingWorkingState(), user_message="I own priorities",
        )
        self.assertEqual(result.reported_facts, state.reported_facts)
        self.assertEqual(llm.generate_structured.call_args.kwargs["schema"], CoachingWorkingState)

    async def test_malformed_json_has_one_bounded_recovery(self):
        llm = SimpleNamespace(generate=AsyncMock(side_effect=["not json", '{"reported_facts":["I own the priorities"]}']))
        result = await CoachingMemoryService(llm).update_working_state(
            previous_state=CoachingWorkingState(), user_message="I own the priorities",
        )
        self.assertEqual(result.reported_facts, ["I own the priorities"])
        self.assertEqual(llm.generate.await_count, 2)

    async def test_semantically_empty_state_cannot_silently_erase_evidence(self):
        llm = SimpleNamespace(generate=AsyncMock(return_value="{}"))
        with self.assertRaises(Exception) as error:
            await CoachingMemoryService(llm).update_working_state(
                previous_state=CoachingWorkingState(reported_facts=["I report to the director"]),
                user_message="She trusts my judgment",
            )
        self.assertEqual(getattr(error.exception, "category", None), "semantic_empty")
        self.assertEqual(llm.generate.await_count, 2)

    async def test_provider_failure_does_not_trigger_another_application_call(self):
        llm = SimpleNamespace(generate=AsyncMock(side_effect=ConnectionError("private provider error")))
        with self.assertRaises(Exception) as error:
            await CoachingMemoryService(llm).update_working_state(
                previous_state=CoachingWorkingState(), user_message="Help me think",
            )
        self.assertEqual(getattr(error.exception, "category", None), "provider")
        self.assertNotIn("private", str(error.exception))
        self.assertEqual(llm.generate.await_count, 1)

    def test_failed_update_retains_evidence_but_clears_stale_action_permission(self):
        previous = CoachingWorkingState(
            reported_facts=["I own staffing decisions"], coaching_stage="action",
            next_coaching_move="develop_experiment", transition_basis=["tested_understanding", "user_owned_outcome"],
        )
        result = CoachingMemoryService.conservative_state(previous, "That advice made things worse")
        self.assertEqual(result.reported_facts, previous.reported_facts)
        self.assertEqual(result.coaching_stage, "discovery")
        self.assertNotIn("tested_understanding", result.transition_basis)
        self.assertEqual(result.last_coaching_move, "develop_experiment")


if __name__ == "__main__":
    unittest.main()
