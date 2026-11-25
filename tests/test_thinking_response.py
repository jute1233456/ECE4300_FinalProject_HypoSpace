import json,unittest
from unittest.mock import patch
from types import SimpleNamespace
from llm import OpenAICompatibleClient
from llm import LLMError
from llm.local_thinking import LocalThinkingClient


class ThinkingResponseTests(unittest.TestCase):
    @patch('llm.openai_compatible.urlopen')
    def test_intermediate_candidate_never_enters_final_answer(self,urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value=json.dumps({
            'choices':[{'message':{'content':'<think>Expression: WRONG\nCheck it.</think>\nExpression: x'},'finish_reason':'stop'}]}).encode()
        c=OpenAICompatibleClient(api_key='local',model='qwen',base_url='http://127.0.0.1:8000/v1',separate_reasoning=True)
        result=c.generate('Question')
        self.assertEqual(result.text,'Expression: x')
        self.assertIn('WRONG',result.reasoning_text)
        self.assertFalse(result.final_answer_missing)

    @patch('llm.openai_compatible.urlopen')
    def test_truncated_thought_is_an_empty_candidate_not_a_connection_failure(self,urlopen):
        urlopen.return_value.__enter__.return_value.read.return_value=json.dumps({
            'choices':[{'message':{'content':'<think>Expression: unfinished'},'finish_reason':'length'}],
            'usage':{'completion_tokens':100}}).encode()
        c=OpenAICompatibleClient(api_key='local',model='qwen',base_url='http://127.0.0.1:8000/v1',separate_reasoning=True)
        result=c.generate('Question',max_tokens=100)
        self.assertEqual(result.text,'')
        self.assertTrue(result.final_answer_missing)
        self.assertEqual(result.finish_reason,'length')
        self.assertEqual(result.completion_tokens,100)

    @patch('llm.openai_compatible.urlopen')
    def test_output_budget_respects_context_without_truncating_input(self,urlopen):
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:list(range(20)),
                                  encode=lambda *a,**kw:[1,2])
        fake=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a:tokenizer))
        urlopen.return_value.__enter__.return_value.read.return_value=json.dumps({
            'choices':[{'message':{'content':'<think>reason here</think>Expression: x'},'finish_reason':'stop'}]}).encode()
        with patch.dict('sys.modules',{'transformers':fake}):
            client=LocalThinkingClient(tokenizer_path='unused',context_window=64,api_key='local',
                model='qwen',base_url='http://127.0.0.1:8000/v1')
        result=client.generate('Original complete question',max_tokens=100)
        body=json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(body['max_tokens'],36)
        self.assertEqual(body['messages'][0]['content'],'Original complete question')
        self.assertEqual(result.reasoning_tokens,2)
        self.assertEqual(result.effective_max_tokens,36)

    @patch('llm.openai_compatible.urlopen')
    def test_oversized_input_is_reported_without_sending_a_bad_request(self,urlopen):
        tokenizer=SimpleNamespace(apply_chat_template=lambda *a,**kw:list(range(100)))
        fake=SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a:tokenizer))
        with patch.dict('sys.modules',{'transformers':fake}):
            client=LocalThinkingClient(tokenizer_path='unused',context_window=64,api_key='local',
                model='qwen',base_url='http://127.0.0.1:8000/v1')
        with self.assertRaises(LLMError) as caught:client.generate('Question')
        self.assertTrue(caught.exception.details['context_overflow'])
        urlopen.assert_not_called()
