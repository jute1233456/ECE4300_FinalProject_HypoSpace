import unittest
from types import SimpleNamespace
from llm.stage_routing import route_reflection
from server.hybrid_worker_python import prepare


class FakeClient:
    model = 'sft-pruned-v1-ep1'
    def review(self, prompt, *, max_tokens):
        return SimpleNamespace(model=self.model, text=prompt)


class RoutingTests(unittest.TestCase):
    def test_only_reflection_changes_and_source_is_unchanged(self):
        client = FakeClient()
        roles = route_reflection(client, 'dpo-reflection-v3-ep1')
        self.assertEqual(client.review('state',max_tokens=32768).model, 'dpo-reflection-v3-ep1')
        self.assertEqual(client.model, 'sft-pruned-v1-ep1')
        self.assertTrue(all(roles[k]==client.model for k in ['read_tree','selection','expansion','candidate']))

    def test_wrong_backend_model_is_rejected(self):
        class Wrong(FakeClient):
            def review(self, *args, **kwargs): return SimpleNamespace(model='base')
        client = Wrong();route_reflection(client,'v3')
        with self.assertRaises(Exception):client.review('state',max_tokens=32768)

    def test_queue_only_reroutes_registered_v3(self):
        cfg={'routes':{'v3':dict(search_model='sft',reflection_model='v3',code='/frozen')}}
        args=['-u','-m','experiments.test_hard_balanced','--model','v3']
        actual,cwd=prepare(args,cfg)
        self.assertEqual(actual[-2:],['--reflection-model','v3'])
        self.assertEqual(actual[actual.index('--model')+1],'sft')
        self.assertEqual(cwd,'/frozen')
        other=args[:-1]+['old']
        self.assertEqual(prepare(other,cfg),(other,None))


if __name__ == '__main__':unittest.main()
