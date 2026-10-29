import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from pydantic import ValidationError

from app.schemas.coaching import CoachingSummaryData, CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService, WorkingStateExtractionError
from tests.test_cycle5_policy import possibility_state


def selection(**changes):
    return {
        "focus_assessment": {"focus": "Prepare an opening", "reason": "The purpose is established", "evidence_ids": ["current_message"]},
        "interaction_assessment": None,
        "inquiry_assessment": {"progress": "advancing", "reason": "The user applies the distinction", "client_requested_pause": False},
        "coaching_stage": "possibilities", "next_coaching_move": "support_choice",
        "client_response_to_last_move": "not_applicable", "client_generation": "emerging",
        "essential_unknowns": [], "transition_basis": [], "knowledge_gap_observed": False,
        "user_owned_outcome": False, "tested_understanding": False, "transition_evidence_ids": [],
        "progress_assessment": {"established": ["The issue is priority, not ability"], "material_unknown": None,
            "discovery_useful": False, "understanding_effect": "establish", "recommended_direction": "support",
            "reason": "The user applies the understanding and requests preparation", "evidence_ids": ["current_message"]},
        "understanding_assessment": {"picture": "The user wants a priority decision, not reassurance about ability",
            "basis": "natural", "evidence_ids": ["current_message"]},
        "support_assessment": {"scope": "preparation", "permitted": True,
            "reason": "The user requests an opening", "evidence_ids": ["current_message"]},
        "requested_support": "action_wording", **changes,
    }


class Cycle5AssessmentTests(unittest.IsolatedAsyncioTestCase):
    async def assess(self, payload, previous=None, message="The issue is priority, not ability. Can we practice an opening?"):
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=payload))
        return await CoachingMemoryService(llm).assess_turn(
            working_state=CoachingWorkingState(reported_facts=[message]), user_message=message,
            latest_assistant_response="What is becoming clear?", history=[], previous_state=previous)

    async def test_natural_evidence_establishes_understanding_without_a_confirmation_question(self):
        state = await self.assess(selection())
        self.assertEqual(state.understanding_context.basis, "natural")
        self.assertIn("tested_understanding", state.transition_basis)
        self.assertNotIn("user_owned_outcome", state.transition_basis)

    async def test_ownership_cannot_borrow_the_shared_transition_reference(self):
        with self.assertRaises(WorkingStateExtractionError):
            await self.assess(selection(user_owned_outcome=True, transition_evidence_ids=["current_message"]))

    async def test_support_cannot_borrow_the_understanding_reference(self):
        payload = selection()
        payload["support_assessment"]["evidence_ids"] = ["invented"]
        with self.assertRaises(WorkingStateExtractionError):
            await self.assess(payload)

    async def test_retention_preserves_the_original_picture_and_proof(self):
        previous = await self.assess(selection())
        payload = selection(understanding_assessment=None)
        payload["progress_assessment"]["understanding_effect"] = "retain"
        result = await self.assess(payload, previous=previous, message="One detail: the deadline is Friday.")
        self.assertEqual(result.understanding_context, previous.understanding_context)

    async def test_legacy_boolean_does_not_supply_a_missing_shared_picture(self):
        payload = selection(understanding_assessment=None)
        payload["progress_assessment"]["understanding_effect"] = "retain"
        with self.assertRaises(WorkingStateExtractionError):
            await self.assess(payload, previous=CoachingWorkingState(transition_basis=["tested_understanding"]))

    def test_preparing_a_turn_keeps_understanding_but_clears_temporary_judgments(self):
        previous = possibility_state()
        prepared = CoachingMemoryService.prepare_state_for_turn(previous)
        self.assertEqual(prepared.understanding_context, previous.understanding_context)
        self.assertIsNone(prepared.progress_assessment)
        self.assertIsNone(prepared.support_assessment)

    async def test_preparation_does_not_become_a_commitment_even_if_summary_model_proposes_it(self):
        message = "Could we practice an opening? I have not chosen to use it."
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
            "commitment": "Speak to the boss tomorrow.",
            "commitment_change": {"operation": "replace", "previous_value": None,
                "evidence": [{"source": "current_message", "excerpt": message}]}}))
        with self.assertRaises(ValidationError):
            await CoachingMemoryService(llm).refresh_summary(previous_summary=None,
                working_state=possibility_state(), user_message=message, assistant_response="What would you say?")
        self.assertEqual(llm.generate_structured.await_count, 2)

    async def test_compatible_refinement_preserves_confirmed_summary_learning(self):
        previous = CoachingSummaryData(primary_discovery="I need a priority decision, not reassurance.")
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={}))
        result = await CoachingMemoryService(llm).refresh_summary(previous_summary=previous,
            working_state=possibility_state(client_response_to_last_move="refined"),
            user_message="One detail: Friday is the deadline.", assistant_response="Let's prepare.")
        self.assertEqual(result.primary_discovery, previous.primary_discovery)

    async def test_summary_prompt_excludes_session_judgments_and_understanding_proof(self):
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={}))
        await CoachingMemoryService(llm).refresh_summary(previous_summary=None, working_state=possibility_state(
            closure_review={"status": "supported", "reason": "private diagnostic", "duration_ms": 1}),
            user_message="Just preparing", assistant_response="Let's try it.")
        payload = json.loads(llm.generate_structured.call_args.kwargs["user_message"])
        self.assertFalse({"progress_assessment", "support_assessment", "understanding_context", "closure_review"} & payload["working_state"].keys())

    async def test_json_and_streaming_use_identical_validated_preparation_context(self):
        from copy import deepcopy
        from unittest.mock import Mock
        from app.workflows.chat_workflow import WorkflowNodes

        async def stream(**kwargs):
            yield "We can practice."

        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.llm_service = SimpleNamespace(generate=AsyncMock(return_value="We can practice."),
                                           generate_stream=Mock(side_effect=stream))
        validated = possibility_state()
        context = nodes._build_coaching_context(relevant_summaries=[], working_state=validated)
        state = {"message": "Could we practice?", "working_state": validated.model_dump(),
                 "coaching_context": context, "user_history": [], "metadata": {}}
        await nodes.generate_response(deepcopy(state))
        events = [event async for event in nodes.generate_response_stream(deepcopy(state))]
        json_input = nodes.llm_service.generate.call_args.kwargs
        self.assertEqual(json_input, nodes.llm_service.generate_stream.call_args.kwargs)
        self.assertIn("understanding_context", json_input["coaching_context"])
        self.assertIn("Preparation is not a commitment", json_input["coaching_policy"])
        self.assertEqual(json.loads(events[0]), {"type": "token", "content": "We can practice."})


class Cycle5BoundaryAndRecoveryTests(unittest.IsolatedAsyncioTestCase):
    def previous_state(self, **action_changes):
        from tests.test_cycle5_policy import conversation_action

        return possibility_state(
            action_context=[conversation_action(**action_changes)],
            support_action_id="conversation-with-boss",
        )

    async def extract(self, previous, message, responses):
        llm = SimpleNamespace(generate_structured=AsyncMock(side_effect=responses))
        state = await CoachingMemoryService(llm).update_working_state(
            previous_state=previous, user_message=message,
            latest_assistant_response="What would help now?", history=[],
        )
        return state, llm

    def action_update(self, previous, message, **changes):
        action = previous.action_context[0]
        return {
            "id": action.id, "actor": action.actor, "action": action.action,
            "status": None, "readiness": None, "basis": "correction",
            "evidence": [{"source": "current_message", "excerpt": message}],
            **changes,
        }

    async def assess_state(self, state, payload, message):
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=payload))
        result = await CoachingMemoryService(llm).assess_turn(
            working_state=state, user_message=message,
            latest_assistant_response="What would help now?", history=[], previous_state=state,
        )
        return result, llm

    async def test_omitted_boundary_scope_preserves_action_boundary_and_original_proof(self):
        previous = self.previous_state(boundary_scope="action")
        result, _ = await self.extract(previous, "My deadline is Friday.", [{
            "reported_facts": ["My deadline is Friday."], "action_updates": [],
        }])
        self.assertEqual(result.action_context, previous.action_context)
        self.assertEqual(result.action_context[0].boundary_scope, "action")

    async def test_scope_correction_preserves_independent_readiness_and_action_identity(self):
        previous = self.previous_state(boundary_scope="unknown")
        message = "I meant I am not ready to do it; practicing is fine."
        result, _ = await self.extract(previous, message, [{
            "reported_facts": [message],
            "action_updates": [self.action_update(previous, message, boundary_scope="action")],
        }])
        self.assertEqual(result.action_context[0].id, previous.action_context[0].id)
        self.assertEqual(result.action_context[0].boundary_scope, "action")
        self.assertEqual(result.action_context[0].readiness, "not_ready")
        self.assertEqual(result.action_context[0].readiness_evidence, previous.action_context[0].readiness_evidence)

    async def test_preparation_request_leaves_action_readiness_status_and_boundary_unchanged(self):
        previous = self.previous_state()
        message = "Can we practice an opening? I still have not decided to have the conversation."
        result, _ = await self.extract(previous, message, [{
            "reported_facts": [message], "action_updates": [],
        }])
        self.assertEqual(result.action_context, previous.action_context)
        self.assertIsNone(result.support_assessment)
        self.assertEqual(len(result.action_context), 1)

    async def test_invalid_scope_change_basis_recovers_once_without_erasing_boundary(self):
        previous = self.previous_state(boundary_scope="action")
        message = "I would like to practice."
        invalid = {"reported_facts": [message], "action_updates": [self.action_update(
            previous, message, basis="consideration", boundary_scope="preparation",
        )]}
        valid = {"reported_facts": [message], "action_updates": []}
        result, llm = await self.extract(previous, message, [invalid, valid])
        self.assertEqual(llm.generate_structured.await_count, 2)
        self.assertEqual(result.action_context, previous.action_context)

    async def test_invented_scope_evidence_is_rejected_after_bounded_recovery(self):
        previous = self.previous_state(boundary_scope="unknown")
        invalid = {"reported_facts": ["I want to practice."], "action_updates": [self.action_update(
            previous, "I refuse any preparation.", boundary_scope="preparation",
        )]}
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=invalid))
        with self.assertRaises(WorkingStateExtractionError):
            await CoachingMemoryService(llm).update_working_state(
                previous_state=previous, user_message="I want to practice.",
                latest_assistant_response="What would help now?", history=[],
            )
        self.assertEqual(llm.generate_structured.await_count, 2)
        self.assertEqual(previous.action_context[0].boundary_scope, "unknown")

    async def test_material_revision_marks_shared_picture_needing_revision(self):
        previous = self.previous_state()
        payload = selection(understanding_assessment=None, support_assessment=None,
            coaching_stage="discovery", next_coaching_move="clarify",
            client_response_to_last_move="corrected", requested_support="none")
        payload["progress_assessment"].update({
            "understanding_effect": "revise", "recommended_direction": "discovery",
            "discovery_useful": True, "material_unknown": "What changed in the underlying purpose?",
        })
        result, _ = await self.assess_state(previous, payload, "No, I am concerned about fairness, not trust.")
        self.assertEqual(result.understanding_context.status, "needs_revision")
        self.assertEqual(result.understanding_context.picture, previous.understanding_context.picture)
        self.assertNotIn("tested_understanding", result.transition_basis)

    async def test_assessment_failure_preserves_facts_and_removes_support_permissions(self):
        from app.services.coaching_policy_service import CoachingPolicyService
        from app.workflows.chat_workflow import WorkflowNodes

        previous = self.previous_state()
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.coaching_memory_service = SimpleNamespace(
            assess_turn=AsyncMock(side_effect=WorkingStateExtractionError("unsupported_evidence")))
        result = await nodes.assess_coaching_move({
            "user_id": "7", "message": "Could we practice?",
            "metadata": {"working_state_status": "updated"},
            "working_state": previous.model_dump(), "previous_working_state": previous.model_dump(),
        })
        state = CoachingWorkingState.model_validate(result["working_state"])
        self.assertEqual(state.reported_facts, previous.reported_facts)
        self.assertEqual(state.action_context, previous.action_context)
        self.assertIsNone(state.support_assessment)
        self.assertNotEqual(state.next_coaching_move, "support_choice")
        self.assertFalse(CoachingPolicyService.action_is_permitted(state))
        self.assertEqual(result["metadata"]["assessment_status"], "conservative_fallback")

    async def test_extraction_failure_preserves_facts_and_cannot_authorize_support(self):
        from app.workflows.chat_workflow import WorkflowNodes

        previous = self.previous_state()
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.chat_history_service = SimpleNamespace(
            get_or_create_conversation=AsyncMock(),
            get_working_state=AsyncMock(return_value=SimpleNamespace(state=previous.model_dump(), source_turn_count=2)),
        )
        nodes.coaching_memory_service = SimpleNamespace(
            update_working_state=AsyncMock(side_effect=WorkingStateExtractionError("unsupported_boundary_scope")),
            assess_turn=AsyncMock(),
        )
        nodes._fetch_history = AsyncMock(return_value=[])
        nodes._fetch_documents = AsyncMock(return_value=[])
        async def rewrite(state):
            return {**state, "rewritten_query": state["message"]}
        nodes.query_rewriting = rewrite
        result = await nodes.retrieve_data({
            "user_id": "7", "message": "Could we practice?", "conversation_id": "synthetic",
            "metadata": {},
        })
        result = await nodes.assess_coaching_move(result)
        state = CoachingWorkingState.model_validate(result["working_state"])
        self.assertEqual(state.reported_facts, previous.reported_facts)
        self.assertEqual(state.action_context, previous.action_context)
        self.assertIsNone(state.support_assessment)
        self.assertNotEqual(state.next_coaching_move, "support_choice")
        self.assertEqual(result["metadata"]["working_state_status"], "conservative_fallback")
        nodes.coaching_memory_service.assess_turn.assert_not_awaited()

    async def test_legacy_scope_classification_cannot_borrow_only_new_practice_request(self):
        state = self.previous_state(boundary_scope="unknown")
        payload = selection(support_action_id=state.action_context[0].id)
        payload["support_assessment"]["boundary_scope"] = "action"
        with self.assertRaises(WorkingStateExtractionError):
            await self.assess_state(state, payload, "Can we practice an opening?")

    async def test_legacy_scope_classification_uses_original_action_proof_without_mutating_it(self):
        from app.services.coaching_policy_service import CoachingPolicyService

        state = self.previous_state(boundary_scope="unknown")
        payload = selection(support_action_id=state.action_context[0].id)
        payload["support_assessment"].update({
            "boundary_scope": "action", "evidence_ids": ["current_message", "action_context:0"],
        })
        result, _ = await self.assess_state(state, payload, "Can we practice an opening?")
        self.assertEqual(result.action_context, state.action_context)
        self.assertEqual(result.support_assessment.boundary_scope, "action")
        self.assertEqual(CoachingPolicyService.enforce(result).next_coaching_move, "support_choice")

    async def test_legacy_scope_classification_cannot_borrow_another_actions_boundary(self):
        state = self.previous_state(boundary_scope="unknown")
        unrelated = state.action_context[0].model_copy(update={
            "id": "unrelated-action", "action": "present a proposal to the board",
            "evidence": ["I am not ready to present to the board, but I can practice it."],
            "readiness_evidence": "I am not ready to present to the board, but I can practice it.",
        })
        state = state.model_copy(update={"action_context": [*state.action_context, unrelated]})
        payload = selection(support_action_id=state.action_context[0].id)
        payload["support_assessment"].update({
            "boundary_scope": "action", "evidence_ids": ["current_message", "action_context:1"],
        })
        with self.assertRaises(WorkingStateExtractionError):
            await self.assess_state(state, payload, "Can we practice an opening?")

    async def test_preparation_refusal_does_not_change_readiness_to_perform_the_action(self):
        for refusal in ("declined", "not_ready"):
            with self.subTest(refusal=refusal):
                previous = self.previous_state(
                    readiness="ready", readiness_evidence="I am ready to have the conversation.",
                )
                message = "I do not want to practice wording for it."
                result, _ = await self.extract(previous, message, [{
                    "reported_facts": [message], "action_updates": [self.action_update(
                        previous, message, basis="readiness_boundary", readiness=refusal,
                        boundary_scope="preparation",
                    )],
                }])
                action = result.action_context[0]
                self.assertEqual(action.readiness, "ready")
                self.assertEqual(action.readiness_evidence, previous.action_context[0].readiness_evidence)
                self.assertTrue(action.preparation_declined)
                self.assertEqual(action.preparation_evidence, message)
                self.assertFalse(action.discussion_declined)

    async def test_lifting_preparation_refusal_preserves_unreadiness_for_the_actual_action(self):
        previous = self.previous_state(
            boundary_scope="preparation", preparation_declined=True,
            preparation_evidence="I do not want to practice wording.",
        )
        message = "We can practice now, but I am still not ready to do it."
        result, _ = await self.extract(previous, message, [{
            "reported_facts": [message], "action_updates": [self.action_update(
                previous, message, basis="readiness_confirmation", readiness="ready",
                boundary_scope="preparation",
            )],
        }])
        action = result.action_context[0]
        self.assertFalse(action.preparation_declined)
        self.assertEqual(action.preparation_evidence, message)
        self.assertEqual(action.readiness, "not_ready")
        self.assertEqual(action.readiness_evidence, previous.action_context[0].readiness_evidence)
        self.assertEqual(action.status, previous.action_context[0].status)

    async def test_topic_refusal_has_its_own_boundary_and_proof(self):
        previous = self.previous_state(
            readiness="ready", readiness_evidence="I am ready to have the conversation.",
        )
        message = "I do not want to discuss this topic anymore."
        result, _ = await self.extract(previous, message, [{
            "reported_facts": [message], "action_updates": [self.action_update(
                previous, message, basis="readiness_boundary", readiness="declined", boundary_scope="topic",
            )],
        }])
        action = result.action_context[0]
        self.assertEqual(action.readiness, "ready")
        self.assertTrue(action.discussion_declined)
        self.assertEqual(action.discussion_evidence, message)
        self.assertEqual(action.readiness_evidence, previous.action_context[0].readiness_evidence)
