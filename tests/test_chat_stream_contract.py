import json
import unittest
from types import SimpleNamespace

from app.api.v1.user.chat import ChatRequest, _stream_response
from app.workflows.chat_workflow import ChatWorkflow


class _WorkflowNodes:
    async def retrieve_data(self, state):
        return state

    async def generate_context(self, state):
        return state

    async def generate_response_stream(self, state):
        state["response"] = "Hello"
        yield json.dumps({"type": "token", "content": "Hello"})

    async def save_conversation(self, state):
        state["conversation_id"] = "conversation-1"
        state["metadata"] = {
            "user_message_id": "user-message-1",
            "assistant_message_id": "assistant-message-1",
        }
        return state


class _FailingWorkflow:
    async def process_message_stream(self, **_kwargs):
        raise RuntimeError("database password should never reach the browser")
        yield ""


class ChatStreamContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_done_event_is_emitted_after_persisting_both_messages(self):
        workflow = ChatWorkflow.__new__(ChatWorkflow)
        workflow.nodes = _WorkflowNodes()

        events = [
            json.loads(event)
            async for event in workflow.process_message_stream(
                user_id="1",
                message="Hello",
            )
        ]

        self.assertEqual(events[0], {"type": "status", "status": "accepted"})
        self.assertEqual(
            events[1], {"type": "status", "status": "retrieving_context"}
        )
        self.assertEqual(
            events[2], {"type": "status", "status": "building_context"}
        )
        self.assertEqual(
            events[3], {"type": "status", "status": "generating_response"}
        )
        self.assertEqual(events[4], {"type": "token", "content": "Hello"})
        self.assertEqual(
            events[5],
            {
                "type": "done",
                "conversation_id": "conversation-1",
                "user_message_id": "user-message-1",
                "assistant_message_id": "assistant-message-1",
                "persisted": True,
            },
        )

    async def test_stream_errors_are_safe_for_the_browser(self):
        events = [
            event
            async for event in _stream_response(
                _FailingWorkflow(),
                current_user=SimpleNamespace(id="user-1"),
                request=ChatRequest(message="Hello", stream=True),
            )
        ]

        error_event = json.loads(events[0].removeprefix("data: ").strip())
        self.assertEqual(error_event["type"], "error")
        self.assertEqual(
            error_event["content"],
            "The chat service is temporarily unavailable. Please try again.",
        )
        self.assertNotIn("password", error_event["content"])


if __name__ == "__main__":
    unittest.main()
