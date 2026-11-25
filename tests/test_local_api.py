import json
import unittest
from unittest.mock import patch
from llm import OpenAICompatibleClient


class LocalAPITests(unittest.TestCase):
    def test_external_plain_http_rejected(self):
        with self.assertRaises(ValueError):
            OpenAICompatibleClient(api_key='local', model='qwen', base_url='http://10.1.3.100:8000/v1')

    @patch('llm.openai_compatible.urlopen')
    def test_loopback_generation_options_are_sent(self, urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value = json.dumps({
            'choices': [{'message': {'content': 'Expression: x'}, 'finish_reason': 'stop'}]
        }).encode()
        client = OpenAICompatibleClient(api_key='local', model='qwen', base_url='http://127.0.0.1:8000/v1',
            generation_options={'temperature': 0.7, 'chat_template_kwargs': {'enable_thinking': False}})
        client.generate('question', max_tokens=4096)
        body = json.loads(urlopen.call_args.args[0].data)
        self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
        self.assertEqual(body['max_tokens'], 4096)
        self.assertEqual(body['temperature'], 0.7)
