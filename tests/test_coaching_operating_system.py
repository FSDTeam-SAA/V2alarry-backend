import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.schemas.coaching import CoachingSummaryData, CoachingWorkingState
from app.services.coaching_memory_service import (
    SUMMARY_RETRIEVAL_PROMPT,
    WORKING_STATE_PROMPT,
    CoachingMemoryService,
)
from app.services.coaching_policy_service import CoachingPolicyService
from app.workflows.chat_workflow import LEADERSHIP_COACH_SYSTEM_PROMPT, WorkflowNodes


class CoachingPolicyInvariantTests(unittest.TestCase):
    def test_teaching_without_an_observed_knowledge_gap_is_downgraded(self):
        state = CoachingWorkingState(
            reported_facts=["The team missed a deadline."],
            coaching_stage="reflection",
            next_coaching_move="teach_selectively",
            transition_basis=["grounded_evidence"],
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.coaching_stage, "discovery")
        self.assertEqual(decision.next_coaching_move, "clarify")
        self.assertIn("knowledge_gap", decision.essential_unknowns)

    def test_teaching_with_an_observed_knowledge_gap_is_allowed(self):
        state = CoachingWorkingState(
            reported_facts=["The leader only knows outcome goals."],
            coaching_stage="reflection",
            next_coaching_move="teach_selectively",
            transition_basis=["grounded_evidence", "knowledge_gap_observed"],
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.next_coaching_move, "teach_selectively")

    def test_consecutive_teaching_returns_ownership_to_the_client(self):
        state = CoachingWorkingState(
            reported_facts=["The leader applied the framework to a live example."],
            coaching_stage="reflection",
            last_coaching_move="teach_selectively",
            next_coaching_move="teach_selectively",
            transition_basis=["grounded_evidence", "knowledge_gap_observed"],
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.coaching_stage, "reflection")
        self.assertEqual(decision.next_coaching_move, "test_understanding")

    def test_correction_invalidates_the_previous_understanding_test(self):
        state = CoachingWorkingState(
            reported_facts=["The client says the earlier picture missed a key fact."],
            coaching_stage="action",
            last_coaching_move="test_understanding",
            next_coaching_move="develop_experiment",
            client_response_to_last_move="corrected",
            transition_basis=[
                "grounded_evidence",
                "user_owned_outcome",
                "tested_understanding",
            ],
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.coaching_stage, "discovery")
        self.assertNotIn("tested_understanding", decision.transition_basis)
        self.assertIn("tested_understanding", decision.essential_unknowns)

    def test_affirming_a_test_records_reusable_understanding_evidence(self):
        state = CoachingWorkingState(
            reported_facts=["The leader named the observed behavior."],
            coaching_stage="action",
            last_coaching_move="test_understanding",
            next_coaching_move="develop_experiment",
            client_response_to_last_move="affirmed",
            transition_basis=["grounded_evidence", "user_owned_outcome"],
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.coaching_stage, "action")
        self.assertIn("tested_understanding", decision.transition_basis)

    def test_action_remains_available_after_turn_scoped_affirmation_resets(self):
        state = CoachingWorkingState(
            reported_facts=["The leader named the observed behavior."],
            coaching_stage="action",
            last_coaching_move="explore_options",
            next_coaching_move="develop_experiment",
            client_response_to_last_move="not_applicable",
            transition_basis=[
                "grounded_evidence",
                "user_owned_outcome",
                "tested_understanding",
            ],
        )

        decision = CoachingPolicyService.enforce(state)

        self.assertEqual(decision.coaching_stage, "action")

    def test_all_declared_unknowns_fit_in_one_complex_working_state(self):
        state = CoachingWorkingState(
            essential_unknowns=[
                "direct_evidence",
                "observable_pattern",
                "interpretation_separation",
                "desired_help",
                "prior_attempts",
                "user_owned_outcome",
                "tested_understanding",
                "readiness",
                "knowledge_gap",
            ]
        )

        self.assertEqual(len(state.essential_unknowns), 9)

    def test_advanced_moves_and_answer_signals_are_representable(self):
        deepen = CoachingWorkingState(
            next_coaching_move="deepen",
            answer_level="role",
            client_generation="emerging",
        )
        shift = CoachingWorkingState(next_coaching_move="shift_focus")
        challenge = CoachingWorkingState(next_coaching_move="challenge_tentatively")

        self.assertEqual(deepen.answer_level, "role")
        self.assertEqual(shift.next_coaching_move, "shift_focus")
        self.assertEqual(challenge.next_coaching_move, "challenge_tentatively")

    def test_productive_client_generation_reduces_coach_content(self):
        policy = CoachingPolicyService.build_response_policy(
            CoachingWorkingState(
                reported_facts=["The leader is articulating a commitment."],
                coaching_stage="reflection",
                next_coaching_move="deepen",
                answer_level="contribution",
                client_generation="productive",
            )
        )

        self.assertIn("reduce coach content", policy.lower())
        self.assertIn("continue in their own language", policy.lower())


class CoachingMemoryLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_turn_resets_ephemeral_feedback_and_records_the_last_move(self):
        previous = CoachingWorkingState(
            reported_facts=["A deadline was missed."],
            next_coaching_move="explore_readiness",
            client_response_to_last_move="not_ready",
            transition_basis=["grounded_evidence"],
        )

        prepared = CoachingMemoryService.prepare_state_for_turn(previous)

        self.assertEqual(prepared.client_response_to_last_move, "not_applicable")
        self.assertEqual(prepared.last_coaching_move, "explore_readiness")

    async def test_turn_after_teaching_consumes_the_old_knowledge_gap(self):
        previous = CoachingWorkingState(
            reported_facts=["A knowledge gap was identified."],
            next_coaching_move="teach_selectively",
            transition_basis=["grounded_evidence", "knowledge_gap_observed"],
        )

        prepared = CoachingMemoryService.prepare_state_for_turn(previous)

        self.assertNotIn("knowledge_gap_observed", prepared.transition_basis)
        self.assertEqual(prepared.last_coaching_move, "teach_selectively")

    async def test_retention_gate_rejects_untested_long_term_claims(self):
        class _SummaryLLM:
            async def generate(self, **_kwargs):
                return CoachingSummaryData(
                    presenting_focus="Invented focus",
                    primary_discovery="Invented discovery",
                    developmental_theme="Invented theme",
                    commitment="Schedule the meeting",
                    next_experiment="Send the message",
                    follow_up_question="Did the meeting work?",
                    coach_notes="The client always avoids authority.",
                    prior_session_continuity="Continue the authority work.",
                ).model_dump_json()

        summary = await CoachingMemoryService(_SummaryLLM()).refresh_summary(
            previous_summary=None,
            working_state=CoachingWorkingState(
                current_concern="My boss thinks I am lazy.",
                interpretations_feelings=["I feel judged."],
                coaching_stage="discovery",
                next_coaching_move="clarify",
            ),
            user_message="My boss thinks I am lazy.",
            assistant_response="What happened that left you with that impression?",
        )

        self.assertEqual(summary, CoachingSummaryData())

    async def test_retention_gate_keeps_grounded_tested_discovery(self):
        class _SummaryLLM:
            async def generate(self, **_kwargs):
                return CoachingSummaryData(
                    presenting_focus="Speaking up when expectations are unclear.",
                    primary_discovery="The concern came from silence, not direct feedback.",
                    developmental_theme="Curiosity before judgment.",
                    follow_up_question="What changed when you asked one question?",
                    coach_notes="The client recognized the interpretation.",
                    prior_session_continuity="Revisit the curiosity practice.",
                ).model_dump_json()

        summary = await CoachingMemoryService(_SummaryLLM()).refresh_summary(
            previous_summary=None,
            working_state=CoachingWorkingState(
                current_concern="Speaking up when expectations are unclear.",
                reported_facts=["The boss did not use the word lazy."],
                observations=["The boss spoke more often with two colleagues."],
                coaching_stage="reflection",
                next_coaching_move="explore_options",
                transition_basis=["grounded_evidence", "tested_understanding"],
            ),
            user_message="Yes, I was interpreting the silence.",
            assistant_response="What might curiosity look like next time?",
        )

        self.assertEqual(summary.primary_discovery, "The concern came from silence, not direct feedback.")
        self.assertEqual(summary.developmental_theme, "Curiosity before judgment.")
        self.assertIsNone(summary.commitment)

    async def test_correction_removes_stale_understanding_from_continuity(self):
        class _SummaryLLM:
            async def generate(self, **_kwargs):
                return CoachingSummaryData(
                    primary_discovery="The old explanation was correct.",
                    developmental_theme="The old theme.",
                    coach_notes="Keep the old diagnosis.",
                    prior_session_continuity="Continue the old direction.",
                ).model_dump_json()

        previous = CoachingSummaryData(
            primary_discovery="Old discovery",
            developmental_theme="Old theme",
            coach_notes="Old notes",
            prior_session_continuity="Old direction",
        )
        summary = await CoachingMemoryService(_SummaryLLM()).refresh_summary(
            previous_summary=previous,
            working_state=CoachingWorkingState(
                reported_facts=["The client says the old interpretation was wrong."],
                coaching_stage="discovery",
                next_coaching_move="clarify",
                client_response_to_last_move="corrected",
                transition_basis=["grounded_evidence", "tested_understanding"],
            ),
            user_message="That is not what I meant.",
            assistant_response="Thank you for correcting me.",
        )

        self.assertIsNone(summary.primary_discovery)
        self.assertIsNone(summary.developmental_theme)
        self.assertIsNone(summary.coach_notes)
        self.assertIsNone(summary.prior_session_continuity)

    async def test_retrieval_gate_returns_only_selected_summaries(self):
        class _SelectionLLM:
            async def generate(self, **_kwargs):
                return '{"relevant_indices":[1]}'

        summaries = [
            CoachingSummaryData(presenting_focus="Hiring a new manager"),
            CoachingSummaryData(presenting_focus="Delegating technical reviews"),
            CoachingSummaryData(presenting_focus="Budget planning"),
        ]

        selected = await CoachingMemoryService(_SelectionLLM()).select_relevant_summaries(
            user_message="I need help delegating code reviews.",
            summaries=summaries,
        )

        self.assertEqual(selected, [summaries[1]])

    async def test_retrieval_gate_ignores_duplicate_and_out_of_range_indices(self):
        class _SelectionLLM:
            async def generate(self, **_kwargs):
                return '{"relevant_indices":[1,1,99]}'

        summaries = [
            CoachingSummaryData(presenting_focus="Hiring"),
            CoachingSummaryData(presenting_focus="Delegation"),
        ]

        selected = await CoachingMemoryService(_SelectionLLM()).select_relevant_summaries(
            user_message="Help me delegate.",
            summaries=summaries,
        )

        self.assertEqual(selected, [summaries[1]])


class CoachingPromptContractTests(unittest.TestCase):
    def test_prompts_cover_advanced_selection_and_continuity_gates(self):
        working_prompt = WORKING_STATE_PROMPT.lower()
        response_prompt = LEADERSHIP_COACH_SYSTEM_PROMPT.lower()
        retrieval_prompt = SUMMARY_RETRIEVAL_PROMPT.lower()

        self.assertIn("turn-scoped", working_prompt)
        self.assertIn("answer level", working_prompt)
        self.assertIn("productive", working_prompt)
        self.assertIn("deepen", response_prompt)
        self.assertIn("shift focus", response_prompt)
        self.assertIn("challenge tentatively", response_prompt)
        self.assertIn("reduce coach content", response_prompt)
        self.assertIn("return ownership", response_prompt)
        self.assertIn("directly useful", retrieval_prompt)
        self.assertIn("empty list", retrieval_prompt)


class ClientDocumentScenarioTests(unittest.TestCase):
    def test_hannah_trace_clarifies_before_a_user_owned_experiment(self):
        opening = CoachingPolicyService.enforce(
            CoachingWorkingState(
                current_concern="The leader believes they are too trusting.",
                interpretations_feelings=["Other people may be unreliable."],
                coaching_stage="action",
                next_coaching_move="develop_experiment",
            )
        )
        action = CoachingPolicyService.enforce(
            CoachingWorkingState(
                current_concern="Pause before judgment and ask one curiosity question.",
                reported_facts=["The leader notices an internal doubt signal."],
                observations=["The leader reaches a judgment before asking a question."],
                coaching_stage="action",
                next_coaching_move="develop_experiment",
                transition_basis=[
                    "grounded_evidence",
                    "user_owned_outcome",
                    "tested_understanding",
                ],
            )
        )

        self.assertEqual(opening.next_coaching_move, "clarify")
        self.assertEqual(action.next_coaching_move, "develop_experiment")

    def test_mario_trace_earns_teaching_and_then_returns_to_application(self):
        teaching = CoachingPolicyService.enforce(
            CoachingWorkingState(
                reported_facts=["The leader only recognizes outcome goals."],
                coaching_stage="reflection",
                next_coaching_move="teach_selectively",
                transition_basis=["grounded_evidence", "knowledge_gap_observed"],
            )
        )
        after_teaching = CoachingPolicyService.enforce(
            CoachingWorkingState(
                reported_facts=["The leader applied the distinction to a live initiative."],
                coaching_stage="reflection",
                last_coaching_move="teach_selectively",
                next_coaching_move="teach_selectively",
                transition_basis=["grounded_evidence", "knowledge_gap_observed"],
            )
        )

        self.assertEqual(teaching.next_coaching_move, "teach_selectively")
        self.assertEqual(after_teaching.next_coaching_move, "test_understanding")

    def test_judy_trace_preserves_generation_and_repairs_correction(self):
        generation_policy = CoachingPolicyService.build_response_policy(
            CoachingWorkingState(
                reported_facts=["The leader is naming the value she wants to create."],
                coaching_stage="reflection",
                next_coaching_move="deepen",
                answer_level="contribution",
                client_generation="productive",
            )
        )
        repair_policy = CoachingPolicyService.build_response_policy(
            CoachingWorkingState(
                reported_facts=["The client says Jess's earlier framing caused harm."],
                coaching_stage="reflection",
                next_coaching_move="reflect",
                client_response_to_last_move="corrected",
                transition_basis=["grounded_evidence", "tested_understanding"],
            )
        )

        self.assertIn("reduce coach content", generation_policy.lower())
        self.assertIn("acknowledge the correction", repair_policy.lower())
        self.assertIn("without defending", repair_policy.lower())


class CoachingRetrievalGateWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_new_session_injects_only_relevant_cce_into_state_and_response_context(self):
        summary_values = {
            field: None for field in CoachingSummaryData.model_fields
        }
        candidates = [
            SimpleNamespace(**{**summary_values, "presenting_focus": "Hiring"}),
            SimpleNamespace(**{**summary_values, "presenting_focus": "Delegation"}),
            SimpleNamespace(**{**summary_values, "presenting_focus": "Budgeting"}),
        ]
        relevant = CoachingSummaryData(presenting_focus="Delegation")
        history = SimpleNamespace(
            get_latest_valid_summaries=AsyncMock(return_value=candidates),
            get_conversation_history=AsyncMock(return_value=[]),
        )
        memory = SimpleNamespace(
            select_relevant_summaries=AsyncMock(return_value=[relevant]),
            prepare_state_for_turn=CoachingMemoryService.prepare_state_for_turn,
            update_working_state=AsyncMock(return_value=CoachingWorkingState()),
        )
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.chat_history_service = history
        nodes.coaching_memory_service = memory
        nodes.document_service = SimpleNamespace(
            get_accessible_document_ids=AsyncMock(return_value=[])
        )
        nodes.embedding_service = SimpleNamespace(embed=AsyncMock(return_value=[0.1]))
        nodes.vector_store = SimpleNamespace(search=AsyncMock(return_value=[]))
        state = {
            "user_id": "7",
            "message": "I need help delegating code reviews.",
            "conversation_id": None,
            "user_history": [],
            "metadata": {},
        }

        with (
            patch("app.workflows.chat_workflow.cache_get", new=AsyncMock(return_value=None)),
            patch("app.workflows.chat_workflow.cache_set", new=AsyncMock()),
        ):
            result = await nodes.retrieve_data(state)

        self.assertIn("Delegation", result["coaching_context"])
        self.assertNotIn("Hiring", result["coaching_context"])
        self.assertNotIn("Budgeting", result["coaching_context"])
        memory.update_working_state.assert_awaited_once_with(
            previous_state=CoachingWorkingState(last_coaching_move="clarify"),
            user_message="I need help delegating code reviews.",
            latest_assistant_response=None,
            relevant_summaries=[relevant],
        )

    async def test_retrieval_gate_failure_injects_no_cross_session_cce(self):
        summary_values = {
            field: None for field in CoachingSummaryData.model_fields
        }
        history = SimpleNamespace(
            get_latest_valid_summaries=AsyncMock(
                return_value=[SimpleNamespace(**{**summary_values, "presenting_focus": "Hiring"})]
            ),
            get_conversation_history=AsyncMock(return_value=[]),
        )
        memory = SimpleNamespace(
            select_relevant_summaries=AsyncMock(side_effect=ValueError("bad selector output")),
            prepare_state_for_turn=CoachingMemoryService.prepare_state_for_turn,
            update_working_state=AsyncMock(return_value=CoachingWorkingState()),
        )
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.chat_history_service = history
        nodes.coaching_memory_service = memory
        nodes.document_service = SimpleNamespace(
            get_accessible_document_ids=AsyncMock(return_value=[])
        )
        nodes.embedding_service = SimpleNamespace(embed=AsyncMock(return_value=[0.1]))
        nodes.vector_store = SimpleNamespace(search=AsyncMock(return_value=[]))
        state = {
            "user_id": "7",
            "message": "I need help delegating code reviews.",
            "conversation_id": None,
            "user_history": [],
            "metadata": {},
        }

        with (
            patch("app.workflows.chat_workflow.cache_get", new=AsyncMock(return_value=None)),
            patch("app.workflows.chat_workflow.cache_set", new=AsyncMock()),
        ):
            result = await nodes.retrieve_data(state)

        self.assertNotIn("Hiring", result["coaching_context"])
        self.assertEqual(result["metadata"]["cce_retrieval_status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
