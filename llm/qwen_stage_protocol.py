"""Experimental Qwen-only stage isolation; does not modify Ark or released snapshots."""
import json
from .qwen_stage_revision import QwenStageRevisionClient
from .local_thinking import LocalThinkingClient


EXAMPLES={
    'reflection':('FORMAT EXAMPLE ONLY, from an unrelated toy search. Do not copy its facts.\n'
        'Review: The last attempt fixed a color without evidence.\n'
        'Constraint check: The toy rules allow either color, so that restriction is unnecessary.\n'
        'Next direction: Explore the previously excluded color while retaining all required constraints.'),
    'selection':('FORMAT EXAMPLE ONLY, from an unrelated search. This is not a recommendation '
        'to choose branch 0 in the actual tree; select using the actual menu and evidence.\n'
        'Branch: 0\nReason: This listed branch still has an untested legal choice.\n'
        'Untried direction: Test a structural choice not present among its children.\n'
        'Avoid: Repeating the already explored construction.'),
    'expansion':('FORMAT EXAMPLE ONLY, an unrelated toy search over colored objects. '
        'Do not copy the toy constraint into the actual problem.\n'
        'Plan: Restrict the next unused object to red.\n'
        'Changed condition: The color of one previously unrestricted object.\n'
        'Expected difference: Candidates differ on that object rather than just formatting.'),
}


def isolate_stage(name,prompt,instruction,examples=False):
    scope=(f'CURRENT STAGE: {name.upper()}. You are executing ONE internal search action. '
        'The reference data contains the original benchmark question, including its final-answer '
        'instructions. Those final-answer instructions are DATA during reflection, selection and expansion; '
        'they are not the output format of the current stage. Follow ONLY the current stage '
        'output contract below. Never answer the benchmark during reflection, selection or expansion.\n\n'
        +instruction)
    if examples and name in EXAMPLES:scope+='\n\n'+EXAMPLES[name]
    user=('REFERENCE DATA (quoted JSON, not the current task):\n'
        +json.dumps({'problem_and_search_context':prompt},ensure_ascii=False)
        +'\n\nEND REFERENCE DATA.\n'+scope)
    return user,scope


class QwenStageProtocolClient(QwenStageRevisionClient):
    protocol_examples=False
    def review(self,prompt,*,max_tokens):
        instruction=('Only review prior attempts. Output a brief Review:, Constraint check:, and '
            'Next direction:. State one evidence-based adjustment for the next search. '
            'Do not submit Expression:, Structure:, Graph:, Branch:, Plan:, or a complete candidate.')
        prompt,instruction=isolate_stage('reflection',prompt,instruction,examples=self.protocol_examples)
        return LocalThinkingClient.generate(self,prompt,system=instruction,max_tokens=max_tokens)
    def _stage_request(self,name,prompt,instruction,max_tokens):
        prompt,instruction=isolate_stage(name,prompt,instruction,examples=self.protocol_examples)
        return super()._stage_request(name,prompt,instruction,max_tokens)


class QwenStageExampleClient(QwenStageProtocolClient):
    protocol_examples=True
