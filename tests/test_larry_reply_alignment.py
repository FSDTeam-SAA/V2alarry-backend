import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock
from pydantic import ValidationError

from app.schemas.coaching import CoachingWorkingState, CoachingTurnAssessment, ClientClaritySelection
from app.services.coaching_memory_service import CoachingMemoryService, WorkingStateExtractionError
from app.services.coaching_policy_service import CoachingPolicyService
from test_cycle5_policy import conversation_action, possibility_state
from test_cycle5_assessment import selection


class LarryReplyPolicyTests(unittest.TestCase):
    def test_material_observation_distinction_precedes_more_desired_help_questions(self):
        state = CoachingWorkingState(reported_facts=["The manager asked for revised cost assumptions."],
            interpretations_feelings=["The client takes that to mean the manager is unhappy with their performance."],
            essential_unknowns=["interpretation_separation"], next_coaching_move="explore_desired_help")
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "separate_observation_from_interpretation")
        self.assertEqual(CoachingPolicyService.enforce(result), result)

    def test_confirmation_is_not_repeated_without_a_material_revision(self):
        state = possibility_state(next_coaching_move="test_understanding", last_coaching_move="test_understanding",
            understanding_context=None, client_response_to_last_move="not_applicable",
            progress_assessment={"established": ["The client wants a priority decision"],
                "material_unknown": None, "discovery_useful": False, "understanding_effect": "unknown",
                "recommended_direction": "synthesis", "reason": "The client does not know where to start",
                "evidence": [{"source": "current_message", "excerpt": "I do not know where to start"}]})
        result = CoachingPolicyService.enforce(state)
        self.assertNotEqual(result.next_coaching_move, "test_understanding")
        self.assertEqual(CoachingPolicyService.enforce(result), result)
        unhelpful = CoachingWorkingState.model_validate({**state.model_dump(), "interaction_assessment": {
            "usefulness": "possibly_unhelpful", "observation": "The previous synthesis did not help the client start",
            "evidence": {"source": "current_message", "excerpt": "I do not know where to start"}}})
        repaired = CoachingPolicyService.enforce(unhelpful)
        self.assertEqual(repaired.next_coaching_move, "shift_focus")
        self.assertEqual(CoachingPolicyService.enforce(repaired), repaired)
        unresolved_start = repaired.model_copy(update={
            "next_coaching_move": "test_understanding", "client_response_to_last_move": "corrected",
            "progress_assessment": repaired.progress_assessment.model_copy(update={
                "material_unknown": "How might the client start?", "understanding_effect": "retain"})})
        result = CoachingPolicyService.enforce(unresolved_start)
        self.assertEqual(result.next_coaching_move, "shift_focus")
        self.assertEqual(CoachingPolicyService.enforce(result), result)

    def test_requested_wording_draws_out_clients_language(self):
        state = possibility_state(action_context=[conversation_action()])
        policy = CoachingPolicyService.build_response_policy(state)
        self.assertNotIn("offer one brief", policy)
        self.assertIn("client's own words", policy)
        self.assertEqual(CoachingPolicyService.enforce(state).action_context, state.action_context)

    def test_knowledge_request_does_not_activate_deferred_teaching(self):
        state = CoachingWorkingState(reported_facts=["Feedback has been limited."],
            next_coaching_move="teach_selectively", transition_basis=["knowledge_gap_observed"])
        result = CoachingPolicyService.enforce(state)
        self.assertNotEqual(result.next_coaching_move, "teach_selectively")

    def test_coach_does_not_offer_closure_without_client_owned_step(self):
        result = CoachingPolicyService.enforce(possibility_state(
            next_coaching_move="close_or_continue", requested_support="none", support_assessment=None))
        self.assertEqual(result.next_coaching_move, "explore_options")

    def test_chosen_step_invites_client_takeaways_before_closure(self):
        state = possibility_state(next_coaching_move="close_or_continue", requested_support="none",
            support_assessment=None,
            action_context=[conversation_action(status="planned", readiness="ready")])
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "invite_takeaways")
        self.assertEqual(CoachingPolicyService.enforce(result), result)
        policy = CoachingPolicyService.build_response_policy(result)
        self.assertIn("what became clearer", policy)
        self.assertNotIn("anything else would help", policy)

    def test_stop_remains_unconditional_without_takeaways_or_action(self):
        state = possibility_state(next_coaching_move="close_or_continue", inquiry_assessment={
            "progress": "unknown", "reason": "The client asks to stop", "client_requested_pause": True})
        policy = CoachingPolicyService.build_response_policy(state)
        self.assertIn("without another discovery question", policy)
        self.assertNotIn("what became clearer", policy)


class LarryClosureEvidenceTests(unittest.IsolatedAsyncioTestCase):
    def test_closure_evidence_is_assessed_independently_instead_of_skipped(self):
        with self.assertRaises(ValidationError):
            CoachingTurnAssessment.model_validate(selection(closure_assessment=None))
        for field in ("next_step_evidence_ids", "willingness_evidence_ids"):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                CoachingTurnAssessment.model_validate(selection(closure_assessment={field: ["action_context:0"]}))
        with self.assertRaises(ValidationError):
            ClientClaritySelection.model_validate({"kind": "insight",
                "evidence": [{"evidence_id": "action_context:0", "excerpt": "A stored action"}]})

    async def test_incomplete_client_account_does_not_authorize_closure(self):
        payload = selection(requested_support="none", support_assessment=None,
            closure_assessment={"action_id": "conversation-with-boss",
                "next_step_evidence_ids": ["current_message"], "willingness_evidence_ids": ["current_message"]})
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=payload))
        state = await CoachingMemoryService(llm).assess_turn(working_state=possibility_state(
            action_context=[conversation_action(status="planned", readiness="ready")]),
            user_message="I will ask the boss tomorrow and I am ready to try.", latest_assistant_response=None, history=[])
        self.assertIsNone(state.closure_assessment)
        self.assertNotEqual(CoachingPolicyService.enforce(state).next_coaching_move, "close_or_continue")

    async def test_practice_summary_schema_cannot_replace_or_clear_commitments(self):
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={}))
        await CoachingMemoryService(llm).refresh_summary(previous_summary=None,
            working_state=possibility_state(), user_message="Could we practice?", assistant_response="What would you say?")
        schema = llm.generate_structured.call_args.kwargs["schema"]
        for field in ("commitment_change", "next_experiment_change"):
            for operation in ("replace", "clear"):
                with self.subTest(field=field, operation=operation), self.assertRaises(ValidationError):
                    schema.model_validate({field: {"operation": operation}})

    async def test_current_withdrawal_summary_schema_allows_clearing_but_not_replacing(self):
        message = "I have decided to abandon that meeting request."
        state = possibility_state(action_context=[conversation_action(status="abandoned",
            status_evidence=message, evidence=[message])])
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={}))
        await CoachingMemoryService(llm).refresh_summary(previous_summary=None,
            working_state=state, user_message=message, assistant_response="Understood.")
        schema = llm.generate_structured.call_args.kwargs["schema"]
        self.assertEqual(schema.model_validate({"commitment_change": {"operation": "clear"}}).commitment_change.operation, "clear")
        with self.assertRaises(ValidationError):
            schema.model_validate({"commitment_change": {"operation": "replace"}})
        await CoachingMemoryService(llm).refresh_summary(previous_summary=None,
            working_state=state, user_message="Could we explore a different direction?", assistant_response="What matters now?")
        with self.assertRaises(ValidationError):
            llm.generate_structured.call_args.kwargs["schema"].model_validate({"commitment_change": {"operation": "clear"}})

    async def test_explicit_stop_needs_no_substantive_progress_assessment(self):
        payload = selection(next_coaching_move="close_or_continue", requested_support="none",
            progress_assessment=None, understanding_assessment=None, support_assessment=None,
            inquiry_assessment={"progress": "unknown", "reason": "The client asks to stop",
                "client_requested_pause": True})
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=payload))
        state = await CoachingMemoryService(llm).assess_turn(working_state=possibility_state(),
            previous_state=possibility_state(), user_message="I would like to stop here for today.",
            latest_assistant_response="Would you like to close here?", history=[])
        self.assertTrue(state.inquiry_assessment.client_requested_pause)
        self.assertIn("without another discovery question", CoachingPolicyService.build_response_policy(state))
        self.assertEqual(llm.generate_structured.await_count, 1)

    async def test_explicit_stop_does_not_validate_an_unneeded_closure_account(self):
        payload = selection(next_coaching_move="close_or_continue", requested_support="none",
            progress_assessment=None, understanding_assessment=None, support_assessment=None,
            inquiry_assessment={"progress": "unknown", "reason": "The client asks to stop", "client_requested_pause": True},
            closure_assessment={"action_id": "unrelated",
                "next_step_evidence_ids": ["history:0"], "willingness_evidence_ids": ["history:0"]})
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=payload))
        state = await CoachingMemoryService(llm).assess_turn(working_state=possibility_state(),
            user_message="I would like to stop here for today.", latest_assistant_response="Would you like to close here?",
            history=[{"role": "user", "content": "An earlier client account."}])
        self.assertTrue(state.inquiry_assessment.client_requested_pause)
        self.assertIsNone(state.closure_assessment)
        self.assertEqual(llm.generate_structured.await_count, 1)

    async def test_evidenced_interpretation_gap_selects_the_natural_distinction(self):
        message = "My manager is unhappy with my performance. They only wrote please revise the cost assumptions."
        payload = selection(interpretation_gap=True, interpretation_evidence_ids=["current_message"])
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=payload))
        state = await CoachingMemoryService(llm).assess_turn(working_state=CoachingWorkingState(reported_facts=[message]),
            user_message=message, latest_assistant_response=None, history=[])
        self.assertIn("interpretation_separation", state.essential_unknowns)
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move,
                         "separate_observation_from_interpretation")

    async def test_interpretation_gap_cannot_borrow_paraphrased_state_as_client_proof(self):
        payload = selection(interpretation_gap=True, interpretation_evidence_ids=["previous_facts:0"])
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=payload))
        state = CoachingWorkingState(reported_facts=["The client assumes the manager is unhappy."])
        with self.assertRaises(WorkingStateExtractionError):
            await CoachingMemoryService(llm).assess_turn(working_state=state, previous_state=state,
                user_message="What should I make of this?", latest_assistant_response=None, history=[])

    async def assess(self, *, evidence_id="current_message", action_id="conversation-with-boss", status="planned"):
        message = "I see that I assumed distrust. I will ask what feedback would help, and I am willing to try it."
        state = possibility_state(action_context=[conversation_action(status=status, readiness="ready")])
        payload = selection(next_coaching_move="close_or_continue", support_assessment=None,
            requested_support="none", understanding_assessment=None, closure_assessment={
                "action_id": action_id,
                "next_step_evidence_ids": [evidence_id],
                "willingness_evidence_ids": [evidence_id]})
        payload["progress_assessment"]["understanding_effect"] = "retain"
        async def generate_structured(**kwargs):
            if kwargs["schema"] is ClientClaritySelection:
                return {"kind": "insight", "evidence": [{"evidence_id": evidence_id,
                    "excerpt": "I see that I assumed distrust."}]}
            return payload
        llm = SimpleNamespace(generate_structured=AsyncMock(side_effect=generate_structured))
        return await CoachingMemoryService(llm).assess_turn(working_state=state,
            previous_state=state, user_message=message, history=[{"role": "assistant", "content": message}],
            latest_assistant_response="What became clear and what will you do with it?")

    async def test_clients_account_can_support_closure_without_another_confirmation(self):
        state = await self.assess()
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "close_or_continue")
        self.assertIsNotNone(state.closure_assessment)
        self.assertIsNone(CoachingMemoryService.prepare_state_for_turn(state).closure_assessment)
        self.assertEqual(state.closure_review.status, "supported")
        self.assertIsNone(CoachingMemoryService.prepare_state_for_turn(state).closure_review)
        self.assertFalse(CoachingPolicyService.action_is_permitted(state))
        self.assertTrue(CoachingPolicyService.commitment_is_permitted(state))

    async def test_completed_takeaways_are_not_requested_again(self):
        state = (await self.assess()).model_copy(update={"next_coaching_move": "invite_takeaways"})
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "close_or_continue")
        self.assertEqual(CoachingPolicyService.enforce(result), result)

    async def test_closure_uses_its_own_action_despite_an_unready_alternative(self):
        state = await self.assess()
        state = CoachingWorkingState.model_validate({**state.model_dump(), "coaching_stage": "action",
            "action_context": [*state.model_dump()["action_context"], conversation_action(
                id="team-meeting", action="hold a team meeting")]})
        result = CoachingPolicyService.enforce(state)
        self.assertEqual(result.next_coaching_move, "close_or_continue")
        self.assertEqual(result.coaching_stage, "possibilities")
        self.assertEqual(result.support_action_id, "conversation-with-boss")
        self.assertEqual(CoachingPolicyService.enforce(result), result)

    async def test_closure_keeps_the_clients_chosen_step_in_continuity(self):
        state = await self.assess()
        message = state.closure_assessment.next_step_evidence[0].excerpt
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value={
            "commitment": "Ask the boss what feedback would help.",
            "commitment_change": {"operation": "replace", "previous_value": None,
                "action_id": "conversation-with-boss", "evidence": [{"source": "current_message", "excerpt": message}]}}))
        summary = await CoachingMemoryService(llm).refresh_summary(previous_summary=None,
            working_state=state, user_message=message, assistant_response="Would you like to close here?")
        self.assertEqual(summary.commitment, "Ask the boss what feedback would help.")
        import json
        request = json.loads(llm.generate_structured.call_args.kwargs["user_message"])
        self.assertNotIn("closure_assessment", request["working_state"])
        self.assertTrue(request["commitment_recording_permitted"])

    async def test_summary_recovers_once_when_history_is_mislabeled_as_current_proof(self):
        state = await self.assess()
        message = state.closure_assessment.next_step_evidence[0].excerpt
        change = {"operation": "replace", "previous_value": None, "action_id": "conversation-with-boss",
            "evidence": [{"source": "current_message", "excerpt": "An earlier client quote"}]}
        bad = {"commitment": "Ask the boss what feedback would help.", "commitment_change": change}
        good = {**bad, "commitment_change": {**change,
            "evidence": [{"source": "current_message", "excerpt": message}]}}
        llm = SimpleNamespace(generate_structured=AsyncMock(side_effect=[bad, good]))
        summary = await CoachingMemoryService(llm).refresh_summary(previous_summary=None,
            working_state=state, user_message=message, assistant_response="Would you like to close here?")
        self.assertEqual(summary.commitment, good["commitment"])
        self.assertEqual(llm.generate_structured.await_count, 2)

    async def test_closure_rejects_coach_evidence_and_state_paraphrases(self):
        for evidence_id in ("history:0", "previous_facts:0", "understanding_context:0", "invented"):
            with self.subTest(evidence_id=evidence_id), self.assertRaises(WorkingStateExtractionError):
                await self.assess(evidence_id=evidence_id)

    async def test_closure_rejects_unrelated_and_unselected_actions(self):
        with self.assertRaises(WorkingStateExtractionError):
            await self.assess(action_id="unrelated")
        for status in ("considering", "abandoned"):
            with self.subTest(status=status):
                state = await self.assess(status=status)
                self.assertIsNone(state.closure_assessment)
                self.assertIsNone(state.closure_review)
                self.assertEqual(state.action_context[0].status, status)
                self.assertNotEqual(CoachingPolicyService.enforce(state).next_coaching_move, "close_or_continue")


class LarryIndependentActionProofTests(unittest.TestCase):
    def test_past_attempt_and_current_refusal_keep_their_separate_proof(self):
        message = "I tried confronting my manager last month. I will not confront them again."
        attempted = {"source": "current_message", "excerpt": "I tried confronting my manager last month."}
        refused = {"source": "current_message", "excerpt": "I will not confront them again."}
        extracted = CoachingWorkingState.model_validate({"action_updates": [{
            "actor": "client", "action": "confronting the manager", "status": "attempted",
            "readiness": "declined", "boundary_scope": "action", "basis": "reported_action",
            "evidence": [attempted, refused], "status_basis": "reported_action", "status_evidence": [attempted],
            "readiness_basis": "readiness_boundary", "readiness_evidence": [refused]}]})
        actions = CoachingMemoryService._merge_action_context(CoachingWorkingState(), extracted, message, [])
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0].status, "attempted")
        self.assertEqual(actions[0].readiness, "declined")
        self.assertEqual(actions[0].status_evidence, attempted["excerpt"])
        self.assertEqual(actions[0].readiness_evidence, refused["excerpt"])

    def test_independent_proof_cannot_bypass_evidence_or_status_checks(self):
        ref = {"source": "current_message", "excerpt": "I tried it and will not repeat it."}
        base = {"actor": "client", "action": "confront the manager", "status": "attempted",
            "readiness": "declined", "basis": "reported_action", "boundary_scope": "action",
            "evidence": [ref], "status_basis": "reported_action", "status_evidence": [ref],
            "readiness_basis": "readiness_boundary", "readiness_evidence": [ref]}
        for changes in ({"status_basis": "intention"}, {"boundary_scope": "preparation"},
                {"readiness_evidence": [{"source": "history", "index": 0, "excerpt": ref["excerpt"]}]}):
            with self.subTest(changes=changes), self.assertRaises(WorkingStateExtractionError):
                extracted = CoachingWorkingState.model_validate({"action_updates": [{**base, **changes}]})
                CoachingMemoryService._merge_action_context(CoachingWorkingState(), extracted, ref["excerpt"],
                    [{"role": "assistant", "content": ref["excerpt"]}])


if __name__ == "__main__":
    unittest.main()
