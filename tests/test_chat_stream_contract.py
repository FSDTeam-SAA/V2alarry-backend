import json
import unittest

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

        self.assertEqual(events[0], {"type": "token", "content": "Hello"})
        self.assertEqual(
            events[1],
            {
                "type": "done",
                "conversation_id": "conversation-1",
                "user_message_id": "user-message-1",
                "assistant_message_id": "assistant-message-1",
                "persisted": True,
            },
        )


if __name__ == "__main__":
    unittest.main()
