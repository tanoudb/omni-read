from pathlib import Path
from huggingface_hub import snapshot_download

CACHE_DIR = Path("assets/cache/translation_models").resolve()
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
    "Qwen/Qwen2.5-3B-Instruct",
]

ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.model",
    "*.tiktoken",
    "*.txt",
    "tokenizer*",
    "generation_config*",
    "special_tokens_map*",
    "merges.txt",
    "vocab*",
]

print(f"Cache dir: {CACHE_DIR}")

last_error = None
for model_id in CANDIDATES:
    print(f"\nTrying: {model_id}")
    try:
        local_dir = snapshot_download(
            repo_id=model_id,
            cache_dir=str(CACHE_DIR),
            resume_download=True,
            allow_patterns=ALLOW_PATTERNS,
        )
        print(f"\nSUCCESS: {model_id}")
        print(f"Local path: {local_dir}")
        break
    except Exception as exc:
        last_error = exc
        print(f"FAILED: {model_id} -> {exc}")
else:
    raise SystemExit(f"No model could be downloaded. Last error: {last_error}")
