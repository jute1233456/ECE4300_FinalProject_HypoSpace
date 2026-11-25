"""Abstract client contract and shared response types."""
from abc import ABC, abstractmethod
from dataclasses import dataclass


class LLMError(RuntimeError):
    """A provider request failed; messages never include credentials or bodies."""

    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details or {}


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None
    reasoning_tokens: int | None = None
    reasoning_text: str | None = None
    final_answer_missing: bool = False
    effective_max_tokens: int | None = None


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int | None = 1024) -> LLMResponse:
        """Return one response, or raise LLMError. Never treat errors as answers."""
        raise NotImplementedError
