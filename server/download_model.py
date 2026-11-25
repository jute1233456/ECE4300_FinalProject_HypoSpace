"""Download official weights; ModelScope fallback for restricted HF connectivity."""
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
target = root / 'models/Qwen3-8B'
try:
    from modelscope import snapshot_download as ms_download
    ms_download('Qwen/Qwen3-8B', local_dir=str(target), max_workers=4)
    source = 'https://modelscope.cn/models/Qwen/Qwen3-8B'
except Exception as error:
    print('ModelScope download unavailable:', type(error).__name__, flush=True)
    from huggingface_hub import snapshot_download
    snapshot_download('Qwen/Qwen3-8B', local_dir=str(target), max_workers=4,
                      allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', 'README.md'])
    source = 'https://huggingface.co/Qwen/Qwen3-8B'
(root / 'logs/model_source.json').write_text(json.dumps({'model': 'Qwen/Qwen3-8B', 'source': source}, indent=2))
print('MODEL_DOWNLOAD_COMPLETE', target, flush=True)
