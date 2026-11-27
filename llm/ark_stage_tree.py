"""Stage-MCTS 2.2 using Ark native tool calls and separate stage requests."""
from dataclasses import asdict
from types import SimpleNamespace
from time import monotonic

from .ark_coding_plan import ArkCodingPlanClient
from .deepseek_tree_tools import DeepseekTreeToolClient
from .split_tree_tools import SplitTreeToolClient


class ArkStageTreeClient(SplitTreeToolClient, DeepseekTreeToolClient):
    def __init__(self, provider="ark", **kwargs):
        if provider == "deepseek":
            from .deepseek_official import DeepSeekOfficialClient
            DeepSeekOfficialClient.__init__(self, **kwargs)
        else:
            ArkCodingPlanClient.__init__(self, **kwargs)
        self.separate_reasoning = True
        self.agent = self.reader = None
        self.tokenizer = None
        self.tool_model = SimpleNamespace(chat=self._inspect)
        self.revision = False
        self.stage_decisions = []

    def _inspect(self, *, messages, **kwargs):
        self.stage_decisions = []
        response = DeepseekTreeToolClient._chat(self, messages, self.output_limit)
        self.requests.append(dict(stage="read_tree", response=response))
        calls = response["choices"][0]["message"].get("tool_calls") or []
        return [dict(function_call=call["function"]) for call in calls]

    def _stage(self, name, prompt, instruction, max_tokens):
        if self.revision:
            prompt += "\n\nDecisions from earlier stages in this round:\n" + "\n".join(self.stage_decisions)
            if name == "selection":
                instruction = (
                    "Choose one listed branch using the tree and history. Output the first line "
                    "Branch: <number>, followed by Reason:, Untried direction:, and Avoid: lines. "
                    "Explain what legal choice remains and what repeated construction to avoid. "
                    "Do not solve the problem or declare exhaustion."
                )
            elif name == "expansion":
                instruction = (
                    "Propose exactly one concrete next constraint for this branch. First line "
                    "Plan: <step>, followed by Changed condition: and Expected difference:. "
                    "Compare existing children and answers; rewording an existing step is not new. "
                    "If no distinct step is available output Plan: NONE and explain why. "
                    "Do not output EXHAUSTED: inability to find a step is not proof of exhaustion. "
                    "Check all explicit problem constraints, including expression depth. "
                    "Do not submit a complete candidate."
                )
            elif name == "candidate":
                instruction += (
                    " Check the original structural limits and selected path before answering. "
                    "Use the previous decision rationale and avoid the identified repeated construction. "
                    "Submit any complete candidate you obtain, including a repeated one."
                )
        started = monotonic()
        response = ArkCodingPlanClient.generate(
            self, prompt, system=instruction, max_tokens=max_tokens)
        self.requests.append(dict(stage=name, stage_response=asdict(response),
            seconds=monotonic()-started, response=dict(
                usage={key: getattr(response, key) for key in
                       ("prompt_tokens", "completion_tokens", "total_tokens")},
                choices=[dict(finish_reason=response.finish_reason,
                              message=dict(content=response.text,
                                           reasoning_content=response.reasoning_text))])))
        if response.finish_reason == "length" or not response.text.strip():
            raise ValueError(f"{name}: incomplete generation")
        text = response.text.strip()
        if self.revision:
            self.stage_decisions.append(name + ": " + text)
            if name in ("selection", "expansion"):
                return text.splitlines()[0]
        return text

    def review(self, prompt, *, max_tokens):
        return ArkCodingPlanClient.generate(self, prompt, max_tokens=max_tokens)
