"""HTTP adapter for OpenAI-compatible chat completions."""
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from .base import LLMClient, LLMError, LLMResponse


class OpenAICompatibleClient(LLMClient):
    """Adapter for providers implementing the chat/completions wire format."""

    def __init__(self, *, api_key: str, model: str, base_url: str,
                 timeout: float = 120, generation_options: dict | None = None,
                 separate_reasoning: bool = False):
        if not api_key or not api_key.strip():
            raise ValueError("API key is required")
        if not model or not model.strip():
            raise ValueError("Model is required")
        parsed = urlparse(base_url)
        loopback_http = parsed.scheme == "http" and parsed.hostname in ("127.0.0.1", "localhost", "::1")
        if ((parsed.scheme != "https" and not loopback_http) or not parsed.netloc or parsed.username
                or parsed.password or parsed.query or parsed.fragment):
            raise ValueError("Base URL must be an HTTPS URL without credentials/query/fragment")
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self._api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.generation_options = dict(generation_options or {})
        self.separate_reasoning = separate_reasoning
        if set(self.generation_options) - {"temperature", "top_p", "top_k", "seed", "chat_template_kwargs"}:
            raise ValueError("Unsupported generation option")

    def generate(self, prompt: str, *, system: str | None = None,
                 max_tokens: int | None = 1024) -> LLMResponse:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be nonempty text")
        if max_tokens is not None and (type(max_tokens) is not int or max_tokens <= 0):
            raise ValueError("max_tokens must be a positive integer")
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {"model": self.model, "messages": messages,
                   "stream": False, **self.generation_options}
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        request = Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + self._api_key,
                     "Content-Type": "application/json"}, method="POST")
        stage = "connect_or_headers"
        try:
            # No automatic retries: repeated requests can consume extra quota.
            with urlopen(request, timeout=self.timeout) as response:
                stage = "response_body"
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            status = error.code
            error.close()
            raise LLMError(f"Model service returned HTTP {status}; check credentials, model and quota") from None
        except (URLError, TimeoutError, OSError) as error:
            cause = error.reason if isinstance(error, URLError) else error
            raise LLMError("Model connection failed or timed out", details={
                "network_stage": stage,
                "network_exception": type(cause).__name__,
                "network_errno": getattr(cause, "errno", None),
                "network_winerror": getattr(cause, "winerror", None),
            }) from None
        except (ValueError, UnicodeError):
            raise LLMError("Model service returned invalid JSON") from None
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"]
            reasoning_text = None
            final_answer_missing = False
            if self.separate_reasoning:
                separate = choice['message'].get('reasoning_content')
                if isinstance(separate, str):
                    reasoning_text = separate
                    content = content or ''
                elif isinstance(content, str) and '</think>' in content:
                    reasoning_text, content = content.rsplit('</think>', 1)
                    reasoning_text = reasoning_text.removeprefix('<think>').strip()
                    content = content.strip()
                else:
                    reasoning_text = content if isinstance(content, str) else ''
                    content = ''
                final_answer_missing = not content.strip()
            usage = data.get("usage") or {}
            reasoning_tokens = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
            if not isinstance(content, str) or (not content.strip() and not self.separate_reasoning):
                raise LLMError(
                    "Model response has no usable text or an unexpected structure",
                    details={
                        "finish_reason": choice.get("finish_reason"),
                        "reasoning_tokens": reasoning_tokens,
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                    },
                )
            return LLMResponse(
                text=content, model=data.get("model") or self.model,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                finish_reason=choice.get("finish_reason"), reasoning_tokens=reasoning_tokens,
                reasoning_text=reasoning_text, final_answer_missing=final_answer_missing,
                effective_max_tokens=max_tokens)
        except (KeyError, IndexError, TypeError, ValueError, AttributeError):
            raise LLMError("Model response has no usable text or an unexpected structure") from None
