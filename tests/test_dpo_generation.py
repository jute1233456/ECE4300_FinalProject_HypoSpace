import ast
import json
from pathlib import Path
import tempfile
import unittest
import threading
from unittest.mock import patch
from urllib.error import HTTPError

from dpo.common import atomic,parse_response,read_jsonl,write_jsonl
from dpo.prepare import REFLECTION_INSTRUCTION
from dpo.generate import spare_cards,sample_once,priority_load,collect_state,task_status,should_yield
from dpo.export import export_records


class DPOGenerationTests(unittest.TestCase):
    def test_training_request_yields_only_requested_cards(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'training.json';atomic(path,dict(active=True,gpus=['a']))
            cfg=dict(training_requests=[str(path)])
            self.assertTrue(should_yield(cfg,dict(id='a')))
            self.assertFalse(should_yield(cfg,dict(id='b')))
            atomic(path,dict(active=False,gpus=['a']))
            self.assertFalse(should_yield(cfg,dict(id='a')))

    def test_higher_priority_work_yields_between_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            task=dict(samples=[dict(status='pending',sample_index=i,attempts=0) for i in range(4)])
            with patch('dpo.generate.should_yield',side_effect=[False,True]),patch('dpo.generate.sample_once',return_value=('saved',dict(response_status='received'))) as sample,patch('dpo.generate.release_when_idle') as release:
                collect_state(dict(output=d),{},task,{},'test',100,10,threading.Event())
            self.assertEqual(sample.call_count,1)
            self.assertEqual(task['samples'][0]['status'],'received')
            self.assertEqual(task['samples'][1]['status'],'pending')
            release.assert_called_once()
    def test_state_task_collects_missing_candidates_without_repeating_saved_ones(self):
        with tempfile.TemporaryDirectory() as d:
            task=dict(samples=[dict(status='received',sample_index=0,attempts=1),dict(status='pending',sample_index=1,attempts=0),dict(status='pending',sample_index=2,attempts=0)])
            def sample(cfg,state,item,*args,**kwargs):
                self.assertFalse(kwargs['release_card']);return 'record-'+str(item['sample_index']),dict(response_status='received')
            with patch('dpo.generate.sample_once',side_effect=sample) as calls,patch('dpo.generate.release_when_idle') as release:
                collect_state(dict(output=d),{},task,{},'test',100,10,threading.Event())
            self.assertEqual(calls.call_count,2);release.assert_called_once()
            self.assertEqual(task_status(task),'received')

    def test_review_contract_matches_deployed_protocol_source(self):
        path=Path(__file__).resolve().parents[1]/'llm/qwen_stage_protocol.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenStageProtocolClient')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='review')
        assignment=next(n for n in fn.body if isinstance(n,ast.Assign) and n.targets[0].id=='instruction')
        self.assertEqual(ast.literal_eval(assignment.value),REFLECTION_INSTRUCTION)

    def test_spare_admission_respects_higher_priority_and_model_capability(self):
        def card(name,**kw):return dict(id=name,role='shared',models=['ep1'],inflight=0,observed=0,owner=None,**kw)
        health=dict(capacity=dict(validation=3),backends=[card('a'),card('b'),card('c')])
        self.assertEqual(spare_cards(health,'ep1','gen',3,0,9),[])
        self.assertEqual(len(spare_cards(health,'ep1','gen',2,0,9)),1)
        health['backends'][0]['models']=['other'];health['backends'][1]['observed']=1
        self.assertEqual([c['id'] for c in spare_cards(health,'ep1','gen',0,0,9)],['c'])

    def test_completed_priority_queue_is_not_blocked_by_old_heartbeat(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);atomic(r/'state.json',dict(phase='completed',running=[]));atomic(r/'tasks.json',[])
            with patch('dpo.generate.time.time',return_value=10**12):
                self.assertEqual(priority_load([dict(state=str(r/'state.json'),tasks=str(r/'tasks.json'))]),(0,0))

    def test_raw_response_saved_before_gpu_is_released_and_no_rating_created(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=dict(output=d,model='ep1',checkpoint_sha256='sha',gateway='pool',owner='gen')
            state=dict(state_id='s',stage='reflection',case_id='case',family_id='family',input_sha256='hash',messages=[dict(role='user',content='fixed')])
            task=dict(sample_id='12345678abcdef',sample_index=0);entry=dict(id='gpu',url='http://backend',role='shared')
            class Metrics:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read(self):return b'vllm:num_requests_running 0\nvllm:num_requests_waiting 0\n'
            def api(url,*args,**kwargs):
                if url.endswith('/health'):
                    self.assertEqual(len(list((Path(d)/'responses').glob('*.json'))),1)
                    return dict(backends=[dict(id='gpu',owner='gen')])
                return dict(model='ep1',choices=[dict(message=dict(content='<think>thought</think>Review: check the rule'),finish_reason='stop')],usage=dict(completion_tokens=12))
            with patch('dpo.generate.call',side_effect=api),patch('urllib.request.urlopen',return_value=Metrics()),patch('dpo.generate.change_role') as release:
                path,row=sample_once(cfg,state,task,entry,'test',100,20,1)
            self.assertEqual(row['response']['final_text'],'Review: check the rule')
            self.assertIn('raw_response',json.loads(Path(path).read_text()))
            self.assertEqual(row['review_status'],'unreviewed');self.assertNotIn('score',row)
            release.assert_called_once_with('pool','test','gpu','shared',None)

    def test_export_keeps_ds_answers_failed_attempts_and_truncation_without_labels(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_jsonl(root/'states.jsonl',[dict(state_id='a',stage='reflection',messages=[],ds_answers=[dict(text='DS history')])])
            atomic(root/'responses/a.json',dict(state_id='a',sample_id='one',sample_index=0,attempt=1,response_status='received',finished_utc=1,response=dict(finish_reason='length',final_text='')))
            atomic(root/'responses/b.json',dict(state_id='a',sample_id='two',sample_index=1,attempt=1,response_status='error',finished_utc=1))
            counts=export_records(root);self.assertEqual(counts['target_attempts'],2);self.assertEqual(counts['ds_answers'],1)
            row=read_jsonl(root/'records.jsonl')[0]
            self.assertEqual(row['target_responses'][0]['response']['finish_reason'],'length')
            self.assertEqual(row['ds_answers'][0]['text'],'DS history')
            self.assertNotIn('split',row);self.assertNotIn('chosen',row);self.assertNotIn('rejected',row)


if __name__=='__main__':unittest.main()
