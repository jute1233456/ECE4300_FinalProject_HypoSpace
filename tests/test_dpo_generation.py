import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from dpo.common import atomic,parse_response,read_jsonl,write_jsonl
from dpo.prepare import assign_splits,REFLECTION_INSTRUCTION
from dpo.generate import spare_cards,sample_once,priority_load
from dpo.export import export_review


class DPOGenerationTests(unittest.TestCase):
    def test_review_contract_matches_deployed_protocol_source(self):
        path=Path(__file__).resolve().parents[1]/'llm/qwen_stage_protocol.py'
        tree=ast.parse(path.read_text(encoding='utf-8'))
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='QwenStageProtocolClient')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='review')
        assignment=next(n for n in fn.body if isinstance(n,ast.Assign) and n.targets[0].id=='instruction')
        self.assertEqual(ast.literal_eval(assignment.value),REFLECTION_INSTRUCTION)

    def test_family_holdout_never_splits_equivalent_questions_or_samples(self):
        states=[dict(domain=d,family_id=d+f,case_id=c) for d in ['causal','3d','boolean'] for f in ['a','b'] for c in ['first','renamed']]
        assign_splits(states,.1)
        for family in {s['family_id'] for s in states}:
            self.assertEqual(len({s['split'] for s in states if s['family_id']==family}),1)
        self.assertEqual({s['domain'] for s in states if s['split']=='validation'},{'causal','3d','boolean'})

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
            state=dict(state_id='s',split='train',stage='reflection',case_id='case',family_id='family',input_sha256='hash',messages=[dict(role='user',content='fixed')])
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

    def test_export_keeps_truncation_and_never_pairs_or_mixes_splits(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write_jsonl(root/'states.jsonl',[dict(state_id='a',split='train',stage='reflection',messages=[]),dict(state_id='b',split='validation',stage='candidate',messages=[])])
            atomic(root/'responses/a.json',dict(state_id='a',sample_id='one',sample_index=0,response_status='received',finished_utc=1,response=dict(finish_reason='length',final_text='')))
            atomic(root/'responses/b.json',dict(state_id='b',sample_id='two',sample_index=0,response_status='received',finished_utc=1,response=dict(finish_reason='stop',final_text='answer')))
            counts=export_review(root);self.assertEqual(counts['train']['responses'],1);self.assertEqual(counts['validation']['responses'],1)
            row=read_jsonl(root/'review/train.unscored.jsonl')[0]
            self.assertEqual(row['outputs'][0]['response']['finish_reason'],'length')
            self.assertNotIn('chosen',row);self.assertNotIn('rejected',row)


if __name__=='__main__':unittest.main()
