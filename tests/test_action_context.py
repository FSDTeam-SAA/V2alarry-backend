import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.coaching import CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService, WorkingStateExtractionError
from app.services.coaching_policy_service import CoachingPolicyService


def action_update(message, **changes):
    return {
        "id": None, "actor": "client and team", "action": "use team huddles",
        "status": "considering", "readiness": None, "basis": "consideration",
        "evidence": [{"source": "current_message", "excerpt": message}], **changes,
    }


class ActionContextTests(unittest.IsolatedAsyncioTestCase):
    async def extract(self, message, updates, previous=None):
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
            "reported_facts": ["The team is performing well."], "action_updates": updates,
        }))
        return await CoachingMemoryService(llm).update_working_state(
            previous_state=previous or CoachingWorkingState(), user_message=message,
            latest_assistant_response="What might that look like?", history=[],
        )

    async def test_proposal_receives_server_id_and_stays_prospective(self):
        message = "We could use huddles to compare experiments."
        state = await self.extract(message, [action_update(message)])
        self.assertTrue(state.action_context[0].id)
        self.assertEqual(state.action_context[0].status, "considering")
        self.assertIn("not yet reported as done", CoachingPolicyService.build_response_policy(state))

    async def test_information_limit_cannot_create_a_readiness_boundary(self):
        message = "I cannot find another process problem to analyze."
        state = await self.extract(message, [action_update(message,
            action="analyze another process problem", readiness="not_ready", basis="information_limit")])
        self.assertEqual(state.action_context, [])

    async def test_information_limit_does_not_erase_an_existing_boundary(self):
        boundary = "I am not comfortable with that meeting yet."
        state = await self.extract(boundary, [action_update(boundary, readiness="not_ready", basis="readiness_boundary")])
        message = "I cannot find another example."
        result = await self.extract(message, [action_update(message, id=state.action_context[0].id,
            readiness="unknown", basis="information_limit")], state)
        self.assertEqual(result.action_context, state.action_context)

    async def test_omission_preserves_boundary_beyond_history_window_and_resume(self):
        message = "I am not comfortable with that meeting yet."
        state = await self.extract(message, [action_update(message, actor="client", action="meet the boss",
            status=None, readiness="not_ready", basis="readiness_boundary")])
        resumed = CoachingWorkingState.model_validate(state.model_dump())
        result = await self.extract("My calls are getting quicker.", [], resumed)
        self.assertEqual(result.action_context, state.action_context)
        self.assertEqual(result.client_response_to_last_move, "not_applicable")

    async def test_implementation_status_requires_reported_action_basis(self):
        message = "We could use huddles."
        with self.assertRaises(WorkingStateExtractionError):
            await self.extract(message, [action_update(message, status="ongoing")])

    async def test_wording_request_cannot_reopen_readiness(self):
        message = "I am not comfortable with that meeting yet."
        state = await self.extract(message, [action_update(message, readiness="not_ready", basis="readiness_boundary")])
        with self.assertRaises(WorkingStateExtractionError):
            await self.extract("How would I open it?", [action_update("How would I open it?",
                id=state.action_context[0].id, status=None, readiness="ready", basis="consideration")], state)

    async def test_new_confirmation_can_reopen_same_action_without_losing_id(self):
        message = "I am not comfortable yet."
        state = await self.extract(message, [action_update(message, readiness="not_ready", basis="readiness_boundary")])
        action_id = state.action_context[0].id
        confirmed = "I now feel ready to try this."
        result = await self.extract(confirmed, [action_update(confirmed, id=action_id,
            status="planned", readiness="ready", basis="readiness_confirmation")], state)
        self.assertEqual(result.action_context[0].id, action_id)
        self.assertEqual(result.action_context[0].readiness, "ready")

    async def test_coach_or_invented_evidence_cannot_update_client_action(self):
        for source, excerpt in (("history", "We already do this"), ("current_message", "Invented")):
            with self.subTest(source=source), self.assertRaises(WorkingStateExtractionError):
                await self.extract("Maybe.", [action_update(excerpt, evidence=[{
                    "source": source, "excerpt": excerpt, "index": 0,
                }])])

    async def test_correction_can_establish_an_existing_practice(self):
        first = "We could use huddles."
        state = await self.extract(first, [action_update(first)])
        correction = "I was unclear: we have actually used those huddles for months."
        result = await self.extract(correction, [action_update(correction,
            id=state.action_context[0].id, status="ongoing", basis="correction")], state)
        self.assertEqual(result.action_context[0].status, "ongoing")
        self.assertEqual(result.action_context[0].status_evidence, correction)

    async def test_existing_id_is_required_instead_of_creating_a_duplicate(self):
        message = "We could use huddles."
        state = await self.extract(message, [action_update(message)])
        with self.assertRaises(WorkingStateExtractionError):
            await self.extract(message, [action_update(message)], state)

    async def test_invalid_update_retries_once_and_can_recover(self):
        message = "We could use huddles."
        bad = {"reported_facts": [message], "action_updates": [action_update(message, status="ongoing")]}
        good = {"reported_facts": [message], "action_updates": [action_update(message)]}
        llm = SimpleNamespace(generate_structured=AsyncMock(side_effect=[bad, good]))
        result = await CoachingMemoryService(llm).update_working_state(
            previous_state=CoachingWorkingState(), user_message=message)
        self.assertEqual(result.action_context[0].status, "considering")
        self.assertEqual(llm.generate_structured.await_count, 2)

    async def test_oversized_update_is_rejected_with_bounded_recovery(self):
        message = "We could try something."
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
            "reported_facts": [message], "action_updates": [action_update(message)] * 11,
        }))
        with self.assertRaises(WorkingStateExtractionError):
            await CoachingMemoryService(llm).update_working_state(
                previous_state=CoachingWorkingState(), user_message=message)
        self.assertEqual(llm.generate_structured.await_count, 2)

    async def test_boundary_proof_survives_later_status_updates(self):
        boundary = "I am not comfortable yet."
        state = await self.extract(boundary, [action_update(boundary, readiness="not_ready", basis="readiness_boundary")])
        for index in range(4):
            message = f"I am still only considering this, reflection {index}."
            state = await self.extract(message, [action_update(message, id=state.action_context[0].id)], state)
        self.assertEqual(state.action_context[0].readiness_evidence, boundary)
        self.assertEqual(state.action_context[0].readiness, "not_ready")

    async def test_redundant_unchanged_update_keeps_original_proof(self):
        message = "I am not comfortable yet."
        state = await self.extract(message, [action_update(message, readiness="not_ready", basis="readiness_boundary")])
        unchanged = action_update(message, id=state.action_context[0].id, readiness="not_ready",
            basis="readiness_boundary", evidence=[{"source": "previous_facts", "excerpt": message, "index": 0}])
        result = await self.extract("Preparing one factual question helped me before.", [unchanged], state)
        self.assertEqual(result.action_context, state.action_context)

    def test_legacy_state_and_fallback_keep_safe_defaults(self):
        legacy = CoachingWorkingState.model_validate({"reported_facts": ["I want clarity."]})
        self.assertEqual(legacy.action_context, [])
        fallback = CoachingMemoryService.conservative_state(legacy, "A new concern.")
        self.assertEqual(fallback.coaching_stage, "discovery")
        self.assertEqual(fallback.requested_support, "none")


class ReadinessFirstTests(unittest.IsolatedAsyncioTestCase):
    async def test_practical_teaching_cannot_bypass_an_active_boundary(self):
        message = "I am not comfortable with that meeting yet."
        state = await ActionContextTests().extract(message, [action_update(message,
            readiness="not_ready", basis="readiness_boundary")])
        state = state.model_copy(update={"next_coaching_move": "teach_selectively",
            "transition_basis": ["knowledge_gap_observed"], "requested_support": "action_wording",
            "support_action_id": state.action_context[0].id})
        decision = CoachingPolicyService.enforce(state)
        self.assertEqual(decision.next_coaching_move, "explore_readiness")
        self.assertEqual(CoachingPolicyService.enforce(decision), decision)
        self.assertIn("wording request does not establish readiness", CoachingPolicyService.build_response_policy(decision))

    async def test_conceptual_explanation_remains_available(self):
        message = "I am not comfortable yet."
        state = await ActionContextTests().extract(message, [action_update(message,
            readiness="not_ready", basis="readiness_boundary")])
        state = state.model_copy(update={"next_coaching_move": "teach_selectively",
            "transition_basis": ["knowledge_gap_observed"], "requested_support": "conceptual_explanation"})
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "teach_selectively")

    async def test_boundary_is_scoped_to_its_action(self):
        message = "I am not comfortable yet."
        state = await ActionContextTests().extract(message, [action_update(message,
            readiness="not_ready", basis="readiness_boundary")])
        ready = "I choose to make a private note and feel ready to do that."
        state = await ActionContextTests().extract(ready, [action_update(ready,
            action="make a private note", status="planned", readiness="ready", basis="readiness_confirmation")], state)
        state = state.model_copy(update={"coaching_stage": "action", "next_coaching_move": "develop_experiment",
            "transition_basis": ["tested_understanding", "user_owned_outcome"], "requested_support": "action_planning",
            "support_action_id": state.action_context[1].id})
        self.assertEqual(CoachingPolicyService.enforce(state).coaching_stage, "action")

    async def test_stalled_readiness_shifts_without_repeating_the_same_question(self):
        message = "I am not comfortable yet."
        state = await ActionContextTests().extract(message, [action_update(message,
            readiness="not_ready", basis="readiness_boundary")])
        state = CoachingWorkingState.model_validate({**state.model_dump(), "requested_support": "action_wording", "inquiry_assessment": {
            "progress": "stalled", "reason": "The client has already described the discomfort", "client_requested_pause": False}})
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "shift_focus")
        self.assertEqual(CoachingPolicyService.enforce(result), result)
