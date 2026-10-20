import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.schemas.coaching import CoachingTurnAssessment, CoachingWorkingState
from app.services.coaching_memory_service import CoachingMemoryService, WorkingStateExtractionError


def focus_on(evidence_id):
    return CoachingTurnAssessment(
        focus_assessment={
            "focus": "Understand the client's relevant experience",
            "reason": "Use the client's existing context before choosing another direction",
            "evidence_ids": [evidence_id],
        },
        interaction_assessment=None,
        coaching_stage="discovery",
        next_coaching_move="deepen",
        client_response_to_last_move="not_applicable",
        client_generation="emerging",
        essential_unknowns=["prior_attempts"],
        transition_basis=["grounded_evidence"],
        knowledge_gap_observed=False,
        user_owned_outcome=False,
        tested_understanding=False,
        transition_evidence_ids=[],
    )


class ClientEvidenceCatalogTests(unittest.IsolatedAsyncioTestCase):
    async def assert_older_evidence_resolves(self, field, source, excerpt, history):
        previous = CoachingWorkingState(**{field: [excerpt]})
        current = previous.model_copy(deep=True)
        original = current.model_dump()
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=focus_on(f"{source}:0")))

        result = await CoachingMemoryService(llm).assess_turn(
            working_state=current,
            previous_state=previous,
            user_message="I would like to understand what matters here.",
            latest_assistant_response=None,
            history=history,
        )

        self.assertEqual(
            [reference.model_dump() for reference in result.focus_assessment.evidence],
            [{"source": source, "excerpt": excerpt, "index": 0}],
        )
        self.assertEqual(getattr(result, field), [excerpt])
        self.assertEqual(current.model_dump(), original)
        self.assertEqual(result.reported_facts, [])
        self.assertEqual(result.observations, [])

    async def test_prior_attempt_is_available_without_recent_history(self):
        await self.assert_older_evidence_resolves(
            "prior_attempts", "previous_attempts",
            "I tried asking her privately last month; she listened more openly.", [],
        )

    async def test_feeling_is_available_after_original_turn_leaves_history_window(self):
        excerpt = "I feel nervous because this relationship matters to me."
        history = [{"role": "user", "content": excerpt}] + [
            {"role": "assistant", "content": f"An unrelated later exchange {index}."}
            for index in range(12)
        ]
        await self.assert_older_evidence_resolves(
            "interpretations_feelings", "previous_feelings", excerpt, history,
        )

    async def test_emerging_agency_is_available_without_recent_history(self):
        await self.assert_older_evidence_resolves(
            "emerging_agency", "previous_agency",
            "I can decide which priorities to renegotiate with my manager.", [],
        )

    async def test_reference_cannot_borrow_evidence_from_a_different_category(self):
        llm = SimpleNamespace(generate_structured=AsyncMock(return_value=focus_on("previous_attempts:0")))
        with self.assertRaises(WorkingStateExtractionError) as raised:
            await CoachingMemoryService(llm).assess_turn(
                working_state=CoachingWorkingState(
                    interpretations_feelings=["I feel nervous discussing this."],
                ),
                user_message="I would like to understand what matters here.",
                latest_assistant_response=None,
                history=[],
            )
        self.assertEqual(raised.exception.category, "unsupported_evidence")


if __name__ == "__main__":
    unittest.main()
