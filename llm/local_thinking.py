"""Thinking inference with final-answer isolation and explicit context budgeting."""
from dataclasses import replace
from .openai_compatible import OpenAICompatibleClient
from .base import LLMError


class LocalThinkingClient(OpenAICompatibleClient):
    def __init__(self, *, tokenizer_path, context_window=32768, **kwargs):
        super().__init__(separate_reasoning=True, **kwargs)
        from transformers import AutoTokenizer
        self.tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_path))
        self.context_window=context_window

    def generate(self,prompt,*,system=None,max_tokens=16384):
        messages=[]
        if system:messages.append({'role':'system','content':system})
        messages.append({'role':'user','content':prompt})
        input_ids=self.tokenizer.apply_chat_template(messages,tokenize=True,
            add_generation_prompt=True,enable_thinking=True)
        available=self.context_window-len(input_ids)-8
        if available<1:
            raise LLMError('Input exceeds local model context window',details={'context_overflow':True,'input_tokens':len(input_ids)})
        limit=min(max_tokens or available,available)
        response=super().generate(prompt,system=system,max_tokens=limit)
        reasoning_tokens=response.reasoning_tokens
        if reasoning_tokens is None and response.reasoning_text is not None:
            reasoning_tokens=len(self.tokenizer.encode(response.reasoning_text,add_special_tokens=False))
        return replace(response,reasoning_tokens=reasoning_tokens)
