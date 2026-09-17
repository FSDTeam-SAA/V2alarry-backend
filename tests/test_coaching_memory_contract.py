import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from app.schemas.coaching import CoachingSummaryData, CoachingWorkingState, HeldClue
from app.services.coaching_memory_service import WORKING_STATE_PROMPT
from app.workflows.chat_workflow import LEADERSHIP_COACH_SYSTEM_PROMPT
from app.workflows.chat_workflow import WorkflowNodes


class CoachingMemoryContractTests(unittest.TestCase):
    def test_working_state_keeps_observation_and_interpretation_separate(self):
        state = CoachingWorkingState(
            current_concern="My boss thinks I am lazy",
            reported_facts=["The boss asked about two missed deadlines"],
            observations=["Two deadlines were missed"],
            interpretations_feelings=["I feel judged as lazy"],
            possible_anchors=["Fear of being seen as unreliable"],
            prior_attempts=[],
            held_clues=[HeldClue(text="The label may be inferred", status="active")],
            unresolved_items=["What words did the boss actually use?"],
            emerging_agency=["Can ask for specific examples"],
        )

        self.assertNotEqual(state.observations, state.interpretations_feelings)
        self.assertEqual(state.held_clues[0].status, "active")

    def test_held_clues_have_a_bounded_lifecycle(self):
        with self.assertRaises(ValidationError):
            HeldClue(text="Unsupported hypothesis", status="permanent")

    def test_summary_contract_is_structured_and_does_not_contain_transcripts(self):
        fields = CoachingSummaryData.model_fields
        self.assertNotIn("messages", fields)
        self.assertNotIn("transcript", fields)
        for field in (
            "presenting_focus",
            "primary_discovery",
            "developmental_theme",
            "commitment",
            "next_experiment",
            "follow_up_question",
            "coach_notes",
            "prior_session_continuity",
        ):
            self.assertIn(field, fields)

    def test_prompts_encode_calibration_without_hard_coding_one_scenario(self):
        extraction_prompt = WORKING_STATE_PROMPT.lower()
        coaching_prompt = LEADERSHIP_COACH_SYSTEM_PROMPT.lower()

        self.assertIn("observation", extraction_prompt)
        self.assertIn("interpretation", extraction_prompt)
        self.assertIn("strengthened", extraction_prompt)
        self.assertNotIn("boss thinks i'm lazy", extraction_prompt)
        self.assertIn("one tentative hypothesis", coaching_prompt)
        self.assertIn("vary", coaching_prompt)


class CoachingMemoryWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_summary_failure_keeps_the_saved_chat_turn_and_marks_summary_stale(self):
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.chat_history_service = SimpleNamespace(
            save_conversation_turn=AsyncMock(
                return_value={
                    "id": "user-message",
                    "conversation_id": "conversation-id",
                    "user_message_id": "user-message",
                    "assistant_message_id": "assistant-message",
                }
            ),
            get_summary=AsyncMock(return_value=None),
            save_summary=AsyncMock(),
            mark_summary_stale=AsyncMock(),
        )
        nodes.coaching_memory_service = SimpleNamespace(
            refresh_summary=AsyncMock(side_effect=ValueError("invalid model output"))
        )
        state = {
            "user_id": "7",
            "message": "I will ask for a concrete example.",
            "response": "That is a useful next experiment.",
            "conversation_id": None,
            "working_state": CoachingWorkingState().model_dump(),
            "source_turn_count": 1,
            "metadata": {},
        }

        with patch.object(nodes, "_invalidate_history_cache", new=AsyncMock()):
            result = await nodes.save_conversation(state)

        self.assertTrue(result["metadata"]["saved"])
        self.assertEqual(result["metadata"]["summary_status"], "stale")
        nodes.chat_history_service.mark_summary_stale.assert_awaited_once_with(
            conversation_id="conversation-id",
            user_id="7",
            source_turn_count=1,
        )

    async def test_new_session_requests_only_three_valid_summaries(self):
        summary_values = {
            field: None for field in CoachingSummaryData.model_fields
        }
        history = SimpleNamespace(
            latest_limit=None,
            get_latest_valid_summaries=AsyncMock(
                return_value=[SimpleNamespace(**summary_values)]
            ),
            get_conversation_history=AsyncMock(return_value=[]),
        )
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.chat_history_service = history
        nodes.coaching_memory_service = SimpleNamespace(
            update_working_state=AsyncMock(return_value=CoachingWorkingState())
        )
        nodes.document_service = SimpleNamespace(
            get_accessible_document_ids=AsyncMock(return_value=[])
        )
        nodes.embedding_service = SimpleNamespace(embed=AsyncMock(return_value=[0.1]))
        nodes.vector_store = SimpleNamespace(search=AsyncMock(return_value=[]))
        state = {
            "user_id": "7",
            "message": "What should I focus on today?",
            "conversation_id": None,
            "user_history": [],
            "metadata": {},
        }

        with (
            patch("app.workflows.chat_workflow.cache_get", new=AsyncMock(return_value=None)),
            patch("app.workflows.chat_workflow.cache_set", new=AsyncMock()),
        ):
            await nodes.retrieve_data(state)

        history.get_latest_valid_summaries.assert_awaited_once_with(
            user_id="7",
            limit=3,
        )


if __name__ == "__main__":
    unittest.main()
