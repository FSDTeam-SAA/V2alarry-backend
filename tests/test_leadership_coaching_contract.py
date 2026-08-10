import unittest

from pydantic import ValidationError

from app.api.v1.user.chat import ChatRequest
from app.workflows.chat_workflow import LEADERSHIP_COACH_SYSTEM_PROMPT


class LeadershipCoachingContractTests(unittest.TestCase):
    def test_chat_request_trims_messages_and_rejects_blank_input(self):
        self.assertEqual(ChatRequest(message="  I need help delegating.  ").message, "I need help delegating.")

        with self.assertRaises(ValidationError):
            ChatRequest(message=" \n\t ")

    def test_jess_prompt_defines_hybrid_coaching_and_safety_boundaries(self):
        prompt = LEADERSHIP_COACH_SYSTEM_PROMPT.lower()

        self.assertIn("jess", prompt)
        self.assertIn("one focused question", prompt)
        self.assertIn("practical next step", prompt)
        self.assertIn("discrimination", prompt)
        self.assertIn("harassment", prompt)
        self.assertIn("not a therapist", prompt)
        self.assertIn("untrusted", prompt)


if __name__ == "__main__":
    unittest.main()
