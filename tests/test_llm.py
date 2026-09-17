from __future__ import annotations

import unittest
from unittest.mock import patch

from maniskill_backend.llm import gen_code, gen_text


class LLMTests(unittest.TestCase):
    @patch("maniskill_backend.llm.has_api_key", return_value=True)
    @patch("maniskill_backend.llm.make_client")
    @patch("maniskill_backend.llm.chat_with_metadata")
    def test_text_request_failure_becomes_structured_result(
        self, chat, make_client, has_key
    ) -> None:
        chat.side_effect = RuntimeError("provider unavailable")

        result = gen_text(
            system="system",
            prompt="prompt",
            fallback_text="fallback",
        )

        self.assertFalse(result.used_llm)
        self.assertEqual(result.text, "fallback")
        self.assertIn("llm_request_failed:RuntimeError", result.reason)

    @patch("maniskill_backend.llm.has_api_key", return_value=True)
    @patch("maniskill_backend.llm.make_client")
    @patch("maniskill_backend.llm.chat_with_metadata")
    def test_code_request_failure_becomes_structured_result(
        self, chat, make_client, has_key
    ) -> None:
        chat.side_effect = RuntimeError("provider unavailable")

        result = gen_code(prompt="prompt", fallback_code="pass\n")

        self.assertFalse(result.used_llm)
        self.assertEqual(result.code, "pass\n")
        self.assertIn("llm_request_failed:RuntimeError", result.reason)


if __name__ == "__main__":
    unittest.main()
