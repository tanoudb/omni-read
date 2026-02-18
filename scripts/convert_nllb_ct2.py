"""
One-time conversion script:
- Download facebook/nllb-200-3.3B from Hugging Face
- Convert to CTranslate2 int8
- Save in assets/models/nllb-200-3.3b-ct2/
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import MODEL_DIR, TRANSLATION_CACHE_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert NLLB HF model to CTranslate2")
    parser.add_argument("--model", default="facebook/nllb-200-3.3B", help="HuggingFace model id")
    parser.add_argument(
        "--output",
        type=Path,
        default=MODEL_DIR / "nllb-200-3.3b-ct2",
        help="Output directory for CT2 model",
    )
    parser.add_argument("--quantization", default="int8", help="CT2 quantization (int8, int8_float16, float16)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("[NLLB] Conversion -> CT2")
    print(f"   model: {args.model}")
    print(f"   output: {args.output}")
    print(f"   quantization: {args.quantization}")

    try:
        from huggingface_hub import snapshot_download
        from ctranslate2.converters import TransformersConverter
    except Exception as exc:
        raise RuntimeError(f"Dépendances manquantes (ctranslate2/huggingface_hub): {exc}")

    class SafeTransformersConverter(TransformersConverter):
        @staticmethod
        def load_model(model_class, model_name_or_path, **kwargs):
            if "dtype" in kwargs and "torch_dtype" not in kwargs:
                kwargs["torch_dtype"] = kwargs.pop("dtype")
            return model_class.from_pretrained(model_name_or_path, **kwargs)

    args.output.mkdir(parents=True, exist_ok=True)

    try:
        print("[NLLB] Download snapshot HuggingFace...")
        snapshot_path = snapshot_download(
            repo_id=args.model,
            cache_dir=str(TRANSLATION_CACHE_DIR),
            local_dir=None,
            local_dir_use_symlinks=False,
        )
        snapshot_path = Path(snapshot_path)
        print(f"[NLLB] Snapshot: {snapshot_path}")

        print("[NLLB] Conversion CT2...")
        converter = SafeTransformersConverter(str(snapshot_path))
        converter.convert(
            output_dir=str(args.output),
            quantization=args.quantization,
            force=True,
        )

        print("[NLLB] Copy tokenizer files...")
        for filename in [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "sentencepiece.bpe.model",
            "spm.model",
        ]:
            src = snapshot_path / filename
            dst = args.output / filename
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)

        print("[NLLB] Conversion complete")
        print(f"   CT2 model path: {args.output}")
    except Exception as exc:
        print(f"[NLLB] Conversion failed: {exc}")
        raise


if __name__ == "__main__":
    main()
