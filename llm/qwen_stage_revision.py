"""Self-hosted Qwen implementation of the Stage-2.2 decision-feedback revision."""
from .split_tree_tools import SplitTreeToolClient


class QwenStageRevisionClient(SplitTreeToolClient):
    def generate(self,*args,**kwargs):
        self.stage_decisions=[]
        return super().generate(*args,**kwargs)

    def _stage(self,name,prompt,instruction,max_tokens):
        prompt+='\n\nDecisions from earlier stages in this round:\n'+'\n'.join(self.stage_decisions)
        if name=='selection':
            instruction=(
                'Choose one listed branch using the tree and history. Output the first line '
                'Branch: <number>, followed by Reason:, Untried direction:, and Avoid: lines. '
                'Explain what legal choice remains and what repeated construction to avoid. '
                'Do not solve the problem or declare exhaustion.')
        elif name=='expansion':
            instruction=(
                'Propose exactly one concrete next constraint for this branch. First line '
                'Plan: <step>, followed by Changed condition: and Expected difference:. '
                'Compare existing children and answers; rewording an existing step is not new. '
                'If no distinct step is available output Plan: NONE and explain why. '
                'Do not output EXHAUSTED: inability to find a step is not proof of exhaustion. '
                'Check all explicit problem constraints, including expression depth. '
                'Do not submit a complete candidate.')
        elif name=='candidate':
            instruction+=(
                ' Check the original structural limits and selected path before answering. '
                'Use the previous decision rationale and avoid the identified repeated construction. '
                'Submit any complete candidate you obtain, including a repeated one.')
        text=self._stage_request(name,prompt,instruction,max_tokens)
        self.stage_decisions.append(name+': '+text)
        return text.splitlines()[0] if name in ('selection','expansion') else text

    def _stage_request(self,name,prompt,instruction,max_tokens):
        return super()._stage(name,prompt,instruction,max_tokens)
