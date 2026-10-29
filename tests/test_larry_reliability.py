import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.coaching import ClientClaritySelection, CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService
from app.services.coaching_policy_service import CoachingPolicyService
from test_cycle5_assessment import selection


class LarryRepetitionReliabilityTests(unittest.TestCase):
    def starting_point_correction(self):
        message = "I do not know where to start. That is why I asked."
        proof = [{"source": "current_message", "excerpt": message}]
        return CoachingWorkingState.model_validate({
            "reported_facts": [
                "The client wants a priority decision from their manager.",
                "The client knows which deadlines can move.",
                "The client does not know how to begin the conversation.",
            ],
            "coaching_stage": "discovery",
            "next_coaching_move": "clarify",
            "last_coaching_move": "test_understanding",
            "client_response_to_last_move": "corrected",
            "requested_support": "action_wording",
            "essential_unknowns": ["desired_help"],
            "interaction_assessment": {
                "usefulness": "possibly_unhelpful",
                "observation": "The summary repeated a known goal rather than helping the client begin.",
                "evidence": proof[0],
            },
            "inquiry_assessment": {
                "progress": "stalled",
                "reason": "The client says the repeated confirmation did not help them start.",
                "client_requested_pause": False,
            },
            "progress_assessment": {
                "established": ["The client wants a priority decision and knows which deadlines can move."],
                "material_unknown": "What starting point could the client see themselves trying?",
                "discovery_useful": True,
                "understanding_effect": "revise",
                "recommended_direction": "discovery",
                "reason": "The client clarifies the help they need, without changing the established goal.",
                "evidence": proof,
            },
            "support_assessment": {
                "scope": "exploration",
                "permitted": True,
                "reason": "The client asks for help finding their own starting point.",
                "evidence": proof,
            },
        })

    def test_clarifying_help_after_repeated_confirmation_preserves_useful_move(self):
        for inquiry_progress in ("advancing", "stalled"):
            with self.subTest(inquiry_progress=inquiry_progress):
                state = self.starting_point_correction()
                state = state.model_copy(update={
                    "inquiry_assessment": state.inquiry_assessment.model_copy(update={"progress": inquiry_progress}),
                })
                result = CoachingPolicyService.enforce(state)

                self.assertEqual(result.next_coaching_move, "clarify")
                self.assertEqual(CoachingPolicyService.enforce(result), result)

    def test_genuine_factual_correction_can_still_repair_the_picture(self):
        state = self.starting_point_correction()
        message = "No, we have not met. I am only considering asking for a meeting."
        proof = [{"source": "current_message", "excerpt": message}]
        state = CoachingWorkingState.model_validate({
            **state.model_dump(),
            "reported_facts": ["The client has not met with the manager."],
            "next_coaching_move": "test_understanding",
            "progress_assessment": {
                "established": ["The meeting has not occurred; asking for one remains a possibility."],
                "material_unknown": None,
                "discovery_useful": False,
                "understanding_effect": "revise",
                "recommended_direction": "synthesis",
                "reason": "The previous picture incorrectly treated an unheld meeting as completed.",
                "evidence": proof,
            },
            "interaction_assessment": {
                "usefulness": "possibly_unhelpful",
                "observation": "The previous synthesis contained a mistaken factual assumption.",
                "evidence": proof[0],
            },
            "inquiry_assessment": {
                "progress": "advancing",
                "reason": "The client corrects an important factual assumption.",
                "client_requested_pause": False,
            },
        })
        result = CoachingPolicyService.enforce(state)

        self.assertEqual(result.next_coaching_move, "test_understanding")
        self.assertEqual(CoachingPolicyService.enforce(result), result)


class LarryClarityReliabilityTests(unittest.IsolatedAsyncioTestCase):
    insight = "I was treating an absence as proof of what the volunteers felt."
    plan = "I will compare the rota versions tomorrow, and I am willing to do that."
    primary_reason = "The primary assessment recommends closing; this is not client testimony."

    async def assess(self, message, clarity, *, history=None, stopping=False, missing_proof=None, readiness="ready"):
        previous = CoachingWorkingState.model_validate({
            "reported_facts": [message],
            "coaching_stage": "possibilities",
            "next_coaching_move": "close_or_continue",
            "understanding_context": {
                "picture": "The client wants to check what happened before judging volunteer motives.",
                "basis": "natural",
                "status": "supported",
                "evidence": [{"source": "current_message", "excerpt": message}],
            },
            "action_context": [{
                "id": "compare-rota",
                "actor": "client",
                "action": "compare the rota versions",
                "status": "planned",
                "readiness": readiness,
                "evidence": [self.plan],
                "status_evidence": self.plan,
                "readiness_evidence": self.plan,
            }],
        })
        payload = selection(
            next_coaching_move="close_or_continue", requested_support="none",
            support_assessment=None, understanding_assessment=None,
            closure_assessment={
                "action_id": "compare-rota",
                "next_step_evidence_ids": ["current_message"],
                "willingness_evidence_ids": ["current_message"],
            },
        )
        payload["progress_assessment"]["understanding_effect"] = "retain"
        payload["progress_assessment"]["reason"] = self.primary_reason
        payload["inquiry_assessment"]["client_requested_pause"] = stopping
        if missing_proof:
            payload["closure_assessment"][missing_proof] = []

        async def generate_structured(*, schema, **kwargs):
            if schema is ClientClaritySelection:
                if isinstance(clarity, Exception):
                    raise clarity
                return clarity
            return payload

        self.llm = SimpleNamespace(generate_structured=AsyncMock(side_effect=generate_structured))
        return await CoachingMemoryService(self.llm).assess_turn(
            working_state=previous, previous_state=previous,
            user_message=message, latest_assistant_response="What is becoming clearer?",
            history=history or [],
        )

    async def test_a_chosen_step_and_willingness_do_not_supply_a_missing_insight(self):
        for clarity in (
            {"kind": "not_described", "evidence": []},
            {"kind": "choice_only", "evidence": [{"evidence_id": "current_message", "excerpt": self.plan}]},
        ):
            with self.subTest(clarity=clarity):
                state = await self.assess(self.plan, clarity)
                self.assertIsNone(state.closure_assessment)
                self.assertEqual(state.closure_review.status, "not_supported")
                self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "invite_takeaways")
                self.assertEqual(self.llm.generate_structured.await_count, 2)

    async def test_insight_classification_without_a_raw_account_does_not_authorize_closure(self):
        state = await self.assess(self.plan, {"kind": "insight", "evidence": []})
        self.assertIsNone(state.closure_assessment)
        self.assertNotEqual(CoachingPolicyService.enforce(state).next_coaching_move, "close_or_continue")
        self.assertEqual(self.llm.generate_structured.await_count, 2)

    async def test_spontaneous_account_uses_the_precise_client_insight_without_another_recap(self):
        message = f"{self.insight} {self.plan}"
        state = await self.assess(message, {
            "kind": "insight",
            "evidence": [{"evidence_id": "current_message", "excerpt": self.insight}],
        })
        self.assertEqual(state.closure_assessment.clarity_evidence[0].excerpt, self.insight)
        self.assertEqual(state.closure_review.status, "supported")
        self.assertEqual(self.llm.generate_structured.await_count, 2)
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "close_or_continue")

    async def test_earlier_client_insight_and_current_plan_do_not_require_another_recap(self):
        state = await self.assess(self.plan, {
            "kind": "insight",
            "evidence": [{"evidence_id": "history:0", "excerpt": self.insight}],
        }, history=[{"role": "user", "content": self.insight}])
        self.assertEqual(state.closure_assessment.clarity_evidence[0].source, "history")
        self.assertEqual(state.closure_assessment.clarity_evidence[0].index, 0)
        self.assertEqual(state.closure_assessment.clarity_evidence[0].excerpt, self.insight)
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "close_or_continue")

    async def test_independent_review_uses_client_words_without_assessor_or_coach_claims(self):
        coach_claim = "Synthetic coach-only claim that the client has already learned enough."
        await self.assess(self.plan, {
            "kind": "insight",
            "evidence": [{"evidence_id": "history:0", "excerpt": self.insight}],
        }, history=[{"role": "user", "content": self.insight}, {"role": "assistant", "content": coach_claim}])
        request = json.dumps(json.loads(self.llm.generate_structured.call_args.kwargs["user_message"]))
        self.assertIn(self.insight, request)
        self.assertIn(self.plan, request)
        self.assertNotIn(self.primary_reason, request)
        self.assertNotIn(coach_claim, request)

    async def test_a_compound_client_account_can_support_all_three_dimensions(self):
        message = f"{self.insight} {self.plan}"
        state = await self.assess(message, {
            "kind": "insight",
            "evidence": [{"evidence_id": "current_message", "excerpt": message}],
        })
        self.assertEqual(state.closure_assessment.clarity_evidence, state.closure_assessment.next_step_evidence)
        self.assertEqual(state.closure_assessment.clarity_evidence, state.closure_assessment.willingness_evidence)
        self.assertEqual(CoachingPolicyService.enforce(state).next_coaching_move, "close_or_continue")

    async def test_insight_quote_must_be_exact_and_belong_to_the_selected_client_source(self):
        message = f"{self.insight} {self.plan}"
        for quote in (
            {"evidence_id": "current_message", "excerpt": "Absence is not proof of how volunteers feel."},
            {"evidence_id": "history:0", "excerpt": self.insight},
        ):
            with self.subTest(quote=quote):
                state = await self.assess(message, {"kind": "insight", "evidence": [quote]},
                    history=[{"role": "user", "content": "The rota was updated yesterday."}])
                self.assertIsNone(state.closure_assessment)
                self.assertEqual(state.closure_review.status, "unavailable")
                self.assertEqual(state.reported_facts, [message])

    async def test_coach_words_cannot_be_the_clients_insight(self):
        state = await self.assess(self.plan, {
            "kind": "insight",
            "evidence": [{"evidence_id": "history:0", "excerpt": self.insight}],
        }, history=[{"role": "assistant", "content": self.insight}])
        self.assertIsNone(state.closure_assessment)
        self.assertEqual(state.closure_review.status, "unavailable")

    async def test_failed_or_malformed_clarity_review_preserves_valid_state_without_closing(self):
        for response in (
            RuntimeError("Synthetic provider failure"),
            TimeoutError("Synthetic provider timeout"),
            {"kind": "insight", "evidence": [{"evidence_id": "current_message", "excerpt": " "}]},
            {"not_the_requested_schema": True},
        ):
            with self.subTest(response=response):
                state = await self.assess(self.plan, response)
                self.assertIsNone(state.closure_assessment)
                self.assertEqual(state.closure_review.status, "unavailable")
                self.assertEqual(state.reported_facts, [self.plan])
                self.assertEqual(state.action_context[0].status, "planned")
                self.assertEqual(self.llm.generate_structured.await_count, 2)
                self.assertNotEqual(CoachingPolicyService.enforce(state).next_coaching_move, "close_or_continue")

    async def test_incomplete_step_or_willingness_does_not_run_a_clarity_review(self):
        for missing_proof in ("next_step_evidence_ids", "willingness_evidence_ids"):
            with self.subTest(missing_proof=missing_proof):
                state = await self.assess(self.plan, RuntimeError("Review must not be invoked"), missing_proof=missing_proof)
                self.assertEqual(self.llm.generate_structured.await_count, 1)
                self.assertIsNone(state.closure_review)
                self.assertIsNone(state.closure_assessment)

    async def test_action_readiness_boundary_does_not_run_a_clarity_review(self):
        state = await self.assess(self.plan, RuntimeError("Review must not be invoked"), readiness="not_ready")
        self.assertEqual(self.llm.generate_structured.await_count, 1)
        self.assertIsNone(state.closure_review)
        self.assertIsNone(state.closure_assessment)
        self.assertNotEqual(CoachingPolicyService.enforce(state).next_coaching_move, "close_or_continue")

    async def test_stopping_does_not_require_a_clarity_account(self):
        state = await self.assess("I would like to stop now.", {
            "kind": "insight",
            "evidence": [{"evidence_id": "history:99", "excerpt": "An unused closure claim."}],
        }, stopping=True)
        self.assertTrue(state.inquiry_assessment.client_requested_pause)
        self.assertIsNone(state.closure_assessment)
        self.assertIsNone(state.closure_review)
        self.assertEqual(self.llm.generate_structured.await_count, 1)
        self.assertIn("without another discovery question", CoachingPolicyService.build_response_policy(state))
