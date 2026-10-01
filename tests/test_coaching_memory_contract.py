import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from app.schemas.coaching import CoachingSummaryData, CoachingWorkingState, HeldClue
from app.services.coaching_memory_service import (
    WORKING_STATE_PROMPT,
    CoachingMemoryService,
)
from app.services.coaching_policy_service import CoachingPolicyService
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

    def test_legacy_working_state_defaults_to_discovery(self):
        state = CoachingWorkingState.model_validate(
            {
                "current_concern": "I am unsure how my manager sees me.",
                "interpretations_feelings": ["I feel judged."],
            }
        )

        self.assertEqual(state.coaching_stage, "discovery")
        self.assertEqual(state.next_coaching_move, "clarify")
        self.assertEqual(state.essential_unknowns, [])

    def test_policy_downgrades_an_ungrounded_interpretation_to_discovery(self):
        state = CoachingWorkingState(
            current_concern="My boss thinks I am lazy.",
            interpretations_feelings=["I feel my boss sees me as lazy."],
            coaching_stage="action",
            next_coaching_move="develop_experiment",
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.coaching_stage, "discovery")
        self.assertEqual(decision.next_coaching_move, "clarify")
        self.assertIn("direct_evidence", decision.essential_unknowns)
        self.assertIn("observable_pattern", decision.essential_unknowns)

    def test_policy_keeps_a_grounded_user_owned_action(self):
        state = CoachingWorkingState(
            current_concern="I need to address missed deadlines.",
            reported_facts=["My manager named two missed deadlines."],
            observations=["Two deadlines were missed."],
            coaching_stage="action",
            next_coaching_move="develop_experiment",
            client_response_to_last_move="affirmed",
            transition_basis=[
                "grounded_evidence",
                "user_owned_outcome",
                "tested_understanding",
            ],
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.coaching_stage, "action")
        self.assertEqual(decision.next_coaching_move, "develop_experiment")

    def test_policy_requires_a_user_owned_outcome_before_action(self):
        state = CoachingWorkingState(
            current_concern="I need to address missed deadlines.",
            reported_facts=["My manager named two missed deadlines."],
            observations=["Two deadlines were missed."],
            coaching_stage="action",
            next_coaching_move="develop_experiment",
            transition_basis=["grounded_evidence"],
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.coaching_stage, "possibilities")
        self.assertEqual(decision.next_coaching_move, "explore_options")
        self.assertIn("user_owned_outcome", decision.essential_unknowns)

    def test_policy_requires_tested_understanding_before_action(self):
        state = CoachingWorkingState(
            current_concern="I need to address missed deadlines.",
            reported_facts=["My manager named two missed deadlines."],
            observations=["Two deadlines were missed."],
            coaching_stage="action",
            next_coaching_move="develop_experiment",
            client_response_to_last_move="affirmed",
            transition_basis=["grounded_evidence", "user_owned_outcome"],
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.coaching_stage, "reflection")
        self.assertEqual(decision.next_coaching_move, "test_understanding")
        self.assertIn("tested_understanding", decision.essential_unknowns)

    def test_policy_reopens_discovery_when_client_is_not_ready_for_the_last_move(self):
        state = CoachingWorkingState(
            current_concern="I want feedback from my boss.",
            reported_facts=["My boss asked about recent progress."],
            observations=["I have been improving."],
            coaching_stage="action",
            next_coaching_move="develop_experiment",
            client_response_to_last_move="not_ready",
            transition_basis=[
                "grounded_evidence",
                "user_owned_outcome",
                "tested_understanding",
            ],
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.coaching_stage, "discovery")
        self.assertEqual(decision.next_coaching_move, "explore_readiness")
        self.assertIn("readiness", decision.essential_unknowns)
        self.assertIn("client_feedback_reopened_discovery", decision.transition_basis)

    def test_discovery_policy_does_not_repeat_a_declined_suggestion(self):
        state = CoachingWorkingState(
            current_concern="I want feedback from my boss.",
            reported_facts=["My boss asked about recent progress."],
            client_response_to_last_move="declined",
            coaching_stage="action",
            next_coaching_move="develop_experiment",
        )

        policy = CoachingPolicyService.build_response_policy(state)

        self.assertIn("Do not repeat or repackage a declined suggestion", policy)
        self.assertIn("readiness", policy.lower())

    def test_policy_requires_repair_when_the_client_corrects_jess(self):
        policy = CoachingPolicyService.build_response_policy(
            CoachingWorkingState(
                current_concern="I need help with a difficult conversation.",
                reported_facts=["The earlier suggestion did not fit the situation."],
                client_response_to_last_move="corrected",
                coaching_stage="action",
                next_coaching_move="develop_experiment",
            )
        )

        self.assertIn("Acknowledge the correction", policy)
        self.assertIn("without defending the prior approach", policy)

    def test_policy_blocks_knowledge_frameworks_until_teaching_is_earned(self):
        discovery_policy = CoachingPolicyService.build_response_policy(
            CoachingWorkingState(
                current_concern="I feel stuck with a colleague.",
                reported_facts=["We disagreed in the last meeting."],
                coaching_stage="discovery",
                next_coaching_move="clarify",
            )
        )
        teaching_policy = CoachingPolicyService.build_response_policy(
            CoachingWorkingState(
                current_concern="I need a way to set a boundary.",
                reported_facts=["I have delayed the conversation twice."],
                observations=["Avoidance is making the issue harder."],
                coaching_stage="reflection",
                next_coaching_move="teach_selectively",
                transition_basis=["grounded_evidence", "knowledge_gap_observed"],
            )
        )

        self.assertIn("Do not introduce, cite, or teach a framework", discovery_policy)
        self.assertIn("demonstrated knowledge gap permits", teaching_policy)

    def test_summary_does_not_replace_confirmed_commitment_with_a_coach_suggestion(self):
        class _SummaryLLM:
            async def generate(self, **_kwargs):
                return CoachingSummaryData(
                    commitment="Schedule a meeting with the boss.",
                    next_experiment="Ask for feedback this week.",
                ).model_dump_json()

        previous_summary = CoachingSummaryData(
            commitment="Observe what makes feedback conversations feel difficult.",
            next_experiment="Notice the moment discomfort appears.",
        )
        working_state = CoachingWorkingState(
            current_concern="I want feedback from my boss.",
            reported_facts=["My boss asked about recent progress."],
            client_response_to_last_move="not_ready",
            coaching_stage="action",
            next_coaching_move="develop_experiment",
            transition_basis=[
                "grounded_evidence",
                "user_owned_outcome",
                "tested_understanding",
            ],
        )

        result = self.asyncio_run(
            CoachingMemoryService(_SummaryLLM()).refresh_summary(
                previous_summary=previous_summary,
                working_state=working_state,
                user_message="Honestly, I would not feel comfortable doing that yet.",
                assistant_response="What would make a conversation feel more manageable?",
            )
        )

        self.assertEqual(result.commitment, previous_summary.commitment)
        self.assertEqual(result.next_experiment, previous_summary.next_experiment)

    def test_summary_keeps_a_new_experiment_after_client_affirms_an_action(self):
        class _SummaryLLM:
            async def generate(self, **_kwargs):
                return CoachingSummaryData(
                    commitment="Ask for one concrete example in the next check-in.",
                    next_experiment="Practice that opening on Thursday.",
                ).model_dump_json()

        working_state = CoachingWorkingState(
            current_concern="I need feedback on my progress.",
            reported_facts=["My manager named two missed deadlines."],
            observations=["I have been improving my delivery pace."],
            client_response_to_last_move="affirmed",
            coaching_stage="action",
            next_coaching_move="develop_experiment",
            transition_basis=[
                "grounded_evidence",
                "user_owned_outcome",
                "tested_understanding",
            ],
        )

        result = self.asyncio_run(
            CoachingMemoryService(_SummaryLLM()).refresh_summary(
                previous_summary=None,
                working_state=working_state,
                user_message="I will ask for one concrete example on Thursday.",
                assistant_response="That is a focused experiment.",
            )
        )

        self.assertEqual(
            result.commitment,
            "Ask for one concrete example in the next check-in.",
        )

    def test_working_state_extractor_receives_the_last_assistant_move_as_untrusted_context(self):
        class _StateExtractor:
            async def generate(self, **kwargs):
                self.request = kwargs["user_message"]
                return CoachingWorkingState().model_dump_json()

        extractor = _StateExtractor()
        service = CoachingMemoryService(extractor)

        result = self.asyncio_run(
            service.update_working_state(
                previous_state=CoachingWorkingState(),
                user_message="Honestly, I would not feel comfortable doing that yet.",
                latest_assistant_response="Would you feel comfortable scheduling a meeting?",
            )
        )

        self.assertEqual(result.coaching_stage, "discovery")
        self.assertIn("latest_assistant_response", extractor.request)
        self.assertIn("Would you feel comfortable", extractor.request)

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
        self.assertIn("coaching stage", extraction_prompt)
        self.assertIn("essential unknown", extraction_prompt)
        self.assertIn("immediately preceding assistant response", extraction_prompt)
        self.assertIn("declines the prior move", extraction_prompt)
        self.assertIn("one foreground focus", extraction_prompt)
        self.assertNotIn("boss thinks i'm lazy", extraction_prompt)
        self.assertIn("one tentative hypothesis", coaching_prompt)
        self.assertIn("vary", coaching_prompt)

    @staticmethod
    def asyncio_run(coroutine):
        import asyncio

        return asyncio.run(coroutine)


class CoachingMemoryWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_assessment_records_the_enforced_stage_and_move(self):
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        state = {
            "working_state": CoachingWorkingState(
                current_concern="My boss thinks I am lazy.",
                interpretations_feelings=["I feel judged as lazy."],
                coaching_stage="action",
                next_coaching_move="develop_experiment",
            ).model_dump(),
            "metadata": {},
        }

        result = await nodes.assess_coaching_move(state)

        self.assertEqual(result["working_state"]["coaching_stage"], "discovery")
        self.assertEqual(result["working_state"]["next_coaching_move"], "clarify")
        self.assertEqual(result["metadata"]["coaching_stage"], "discovery")
        self.assertEqual(result["metadata"]["next_coaching_move"], "clarify")

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

    async def test_working_state_update_receives_the_last_assistant_response(self):
        history = SimpleNamespace(
            get_conversation_history=AsyncMock(
                return_value=[
                    {"role": "assistant", "content": "Would you schedule a meeting?"}
                ]
            ),
            get_working_state=AsyncMock(return_value=None),
        )
        memory_service = SimpleNamespace(
            update_working_state=AsyncMock(return_value=CoachingWorkingState())
        )
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.chat_history_service = history
        nodes.coaching_memory_service = memory_service
        nodes.document_service = SimpleNamespace(
            get_accessible_document_ids=AsyncMock(return_value=[])
        )
        nodes.embedding_service = SimpleNamespace(embed=AsyncMock(return_value=[0.1]))
        nodes.vector_store = SimpleNamespace(search=AsyncMock(return_value=[]))
        nodes.llm_service = SimpleNamespace(generate=AsyncMock(return_value="not ready"))
        state = {
            "user_id": "7",
            "message": "Honestly, I would not feel comfortable yet doing that.",
            "conversation_id": "conversation-1",
            "user_history": [],
            "metadata": {},
        }

        with (
            patch("app.workflows.chat_workflow.cache_get", new=AsyncMock(return_value=None)),
            patch("app.workflows.chat_workflow.cache_set", new=AsyncMock()),
        ):
            await nodes.retrieve_data(state)

        memory_service.update_working_state.assert_awaited_once_with(
            previous_state=CoachingWorkingState(last_coaching_move="clarify"),
            user_message="Honestly, I would not feel comfortable yet doing that.",
            latest_assistant_response="Would you schedule a meeting?",
            relevant_summaries=[],
        )


if __name__ == "__main__":
    unittest.main()
