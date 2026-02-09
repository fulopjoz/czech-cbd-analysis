import unittest
from unittest.mock import patch, MagicMock

from czech_cbd_analysis.llm_integration import _post, chat_completion, translate, summarise, embed


class TestLLMIntegration(unittest.TestCase):
    """Test suite for the llm_integration module."""

    def setUp(self) -> None:
        self.api_key = "test-key"

    @patch("czech_cbd_analysis.llm_integration.requests.post")
    def test_chat_completion(self, mock_post):
        """chat_completion should return the content of the first choice."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "Hello"}}]})
        result = chat_completion([{"role": "user", "content": "Hi"}], api_key=self.api_key)
        self.assertEqual(result, "Hello")
        mock_post.assert_called_once()

    @patch("czech_cbd_analysis.llm_integration.requests.post")
    def test_translate(self, mock_post):
        """translate should pass a system prompt and return the translated text."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "Hello"}}]})
        result = translate("Ahoj", api_key=self.api_key)
        self.assertEqual(result, "Hello")

    @patch("czech_cbd_analysis.llm_integration.requests.post")
    def test_summarise(self, mock_post):
        """summarise should produce a concise summary."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"choices": [{"message": {"content": "Summary"}}]})
        result = summarise("Long text", api_key=self.api_key)
        self.assertEqual(result, "Summary")

    @patch("czech_cbd_analysis.llm_integration.requests.post")
    def test_embed(self, mock_post):
        """embed should return a list of embeddings."""
        mock_post.return_value = MagicMock(status_code=200, json=lambda: {"data": [{"embedding": [0.1, 0.2], "text": "test"}]})
        result = embed(["test"], api_key=self.api_key)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIn("embedding", result[0])

    def test_post_no_api_key(self):
        """_post should raise ValueError when api_key is empty."""
        with self.assertRaises(ValueError):
            _post("/v1/chat/completions", {}, api_key="")


if __name__ == "__main__":
    unittest.main()
