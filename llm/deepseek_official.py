"""DeepSeek official API configuration, independent of Ark credentials."""
import json
import os
from pathlib import Path

from .openai_compatible import OpenAICompatibleClient


class DeepSeekOfficialClient(OpenAICompatibleClient):
    BASE_URL = "https://api.deepseek.com"
    MODEL = "deepseek-flash"

    def __init__(self, *, api_key=None, model=None, timeout=None, config_path=None):
        path = Path(config_path) if config_path else Path(__file__).resolve().parents[1] / "config/llm.json"
        config = {}
        if path.exists():
            config = json.loads(path.read_text(encoding="utf-8-sig")).get("deepseek_official", {})
        key = api_key if api_key is not None else config.get("api_key") or os.getenv("DEEPSEEK_API_KEY", "")
        if not key.strip():
            raise ValueError("Set DEEPSEEK_API_KEY or config/llm.json deepseek_official.api_key")
        OpenAICompatibleClient.__init__(
            self, api_key=key, model=model or DeepSeekOfficialClient.MODEL,
            base_url=DeepSeekOfficialClient.BASE_URL,
            timeout=timeout if timeout is not None else config.get("timeout", 600),
            separate_reasoning=True)
