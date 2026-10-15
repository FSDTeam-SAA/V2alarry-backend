import unittest
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.models.message import Message
from app.models.coaching_working_state import CoachingWorkingState as StateRecord
from app.services.chat_history_service import ChatHistoryService
from app.workflows.chat_workflow import WorkflowNodes


class TurnPersistenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_saved_pair_contains_mapped_metadata_and_linked_decision(self):
        added = []
        conversation = SimpleNamespace(id=uuid.uuid4(), title="Existing", updated_at=None)
        results = iter([
            SimpleNamespace(scalar_one_or_none=lambda: conversation),
            SimpleNamespace(scalar_one_or_none=lambda: None),
        ])
        db = SimpleNamespace(
            execute=AsyncMock(side_effect=lambda *_: next(results)),
            add=added.append, commit=AsyncMock(), close=AsyncMock(),
        )

        async def refresh(message):
            message.id = message.id or uuid.uuid4()
            message.created_at = message.created_at or datetime.now(timezone.utc)

        db.refresh = AsyncMock(side_effect=refresh)
        with patch("app.services.chat_history_service.AsyncSessionLocal", return_value=db):
            result = await ChatHistoryService().save_conversation_turn(
                user_id="7", conversation_id=str(conversation.id),
                user_message="I can influence the priorities.", assistant_response="What matters?",
                metadata={"docs_retrieved": 2, "coaching_trace": {"version": "calibration-2"}},
                working_state={"reported_facts": ["I can influence priorities"],
                               "interaction_assessment": {"usefulness": "possibly_unhelpful"},
                               "inquiry_assessment": {"progress": "stalled"}}, source_turn_count=3,
            )
        user, assistant = [item for item in added if isinstance(item, Message)]
        self.assertEqual(user.doc_metadata["coaching_trace"]["version"], "calibration-2")
        self.assertEqual(assistant.doc_metadata["coaching_trace"], user.doc_metadata["coaching_trace"])
        self.assertEqual(assistant.doc_metadata["user_message_id"], result["user_message_id"])
        self.assertEqual(user.doc_metadata["assistant_message_id"], result["assistant_message_id"])
        self.assertLess(user.created_at, assistant.created_at)
        saved_state = next(item for item in added if isinstance(item, StateRecord))
        self.assertNotIn("interaction_assessment", saved_state.state)
        self.assertNotIn("inquiry_assessment", saved_state.state)

    async def test_internal_trace_is_saved_without_entering_public_metadata(self):
        nodes = WorkflowNodes.__new__(WorkflowNodes)
        nodes.chat_history_service = SimpleNamespace(save_conversation_turn=AsyncMock(return_value={
            "conversation_id": "conversation", "id": "user", "user_message_id": "user",
            "assistant_message_id": "assistant",
        }))
        nodes._invalidate_history_cache = AsyncMock()
        state = {"user_id": "7", "message": "Hello", "response": "Hello", "metadata": {},
                 "decision_trace": {"focus": "private diagnostic"}}
        result = await nodes.save_conversation(state)
        stored = nodes.chat_history_service.save_conversation_turn.call_args.kwargs["metadata"]
        self.assertEqual(stored["coaching_trace"], state["decision_trace"])
        self.assertNotIn("coaching_trace", result["metadata"])


if __name__ == "__main__":
    unittest.main()
