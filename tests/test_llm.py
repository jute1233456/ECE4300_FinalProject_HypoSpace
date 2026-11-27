import json
import unittest
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from llm import ArkCodingPlanClient, LLMClient, LLMError
from llm.openai_compatible import OpenAICompatibleClient


class ClientTests(unittest.TestCase):
    def test_file_configuration_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"ark_coding_plan": {
                "api_key": "file-key", "model": "file-model", "timeout": 15}}))
            with patch.dict("os.environ", {"ARK_API_KEY": "env-key"}):
                client = ArkCodingPlanClient(config_path=path)
                self.assertEqual(client._api_key, "file-key")
                self.assertEqual(client.model, "file-model")
                self.assertEqual(client.timeout, 15)
                override = ArkCodingPlanClient(config_path=path, api_key="override")
                self.assertEqual(override._api_key, "override")

    def test_invalid_config_does_not_expose_content(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"api_key": "private-test-key", broken}')
            with self.assertRaises(ValueError) as caught:
                ArkCodingPlanClient(config_path=path)
            self.assertNotIn("private-test-key", str(caught.exception))

    @patch("llm.openai_compatible.urlopen")
    def test_request_and_usage(self, open_url):
        response = MagicMock()
        response.read.return_value = json.dumps({"model": "test-model", "choices": [
            {"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 12}}).encode()
        open_url.return_value.__enter__.return_value = response
        client: LLMClient = ArkCodingPlanClient(api_key="test-key")
        result = client.generate("question", system="system", max_tokens=50)
        request = open_url.call_args.args[0]
        self.assertEqual(request.full_url, ArkCodingPlanClient.BASE_URL + "/chat/completions")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        body = json.loads(request.data)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertEqual(body["max_tokens"], 50)
        self.assertEqual(result.text, "answer")
        self.assertEqual(result.total_tokens, 12)
        self.assertIsNone(result.prompt_tokens)

    @patch("llm.openai_compatible.urlopen")
    def test_http_error_hides_response_and_does_not_retry(self, open_url):
        open_url.side_effect = HTTPError("https://example.com", 401, "test-key", {}, None)
        with self.assertRaises(LLMError) as caught:
            ArkCodingPlanClient(api_key="test-key").generate("question")
        self.assertNotIn("test-key", str(caught.exception))
        self.assertIn("401", str(caught.exception))
        self.assertEqual(open_url.call_count, 1)

    @patch("llm.openai_compatible.urlopen")
    def test_missing_answer_fails(self, open_url):
        open_url.return_value.__enter__.return_value.read.return_value = b'{"choices": []}'
        with self.assertRaises(LLMError):
            ArkCodingPlanClient(api_key="test-key").generate("question")

    @patch("llm.openai_compatible.urlopen")
    def test_provider_limit_and_reasoning_usage_on_truncation(self, open_url):
        response = open_url.return_value.__enter__.return_value
        response.read.return_value = json.dumps({
            "choices": [{"message": {"content": None}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 42, "completion_tokens_details": {"reasoning_tokens": 42}},
        }).encode()
        with self.assertRaises(LLMError) as caught:
            ArkCodingPlanClient(api_key="test-key").generate("question", max_tokens=None)
        self.assertNotIn("max_tokens", json.loads(open_url.call_args.args[0].data))
        self.assertEqual(caught.exception.details["finish_reason"], "length")
        self.assertEqual(caught.exception.details["reasoning_tokens"], 42)

    def test_empty_key_rejected(self):
        with self.assertRaises(ValueError):
            ArkCodingPlanClient(api_key="")

    def test_local_http_endpoint_is_allowed(self):
        client = OpenAICompatibleClient(
            api_key="local", model="local-model", base_url="http://127.0.0.1:8000/v1",
            temperature=0.8)
        self.assertEqual(client.temperature, 0.8)
        with self.assertRaises(ValueError):
            OpenAICompatibleClient(api_key="local", model="local-model", base_url="http://example.com/v1")


if __name__ == "__main__":
    unittest.main()
