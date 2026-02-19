"""
Script de setup NLLB CT2 — à lancer une seule fois.
Télécharge facebook/nllb-200-3.3B depuis HuggingFace
et le convertit en format CTranslate2 int8.

Usage : python setup_nllb_ct2.py
"""

from __future__ import annotations

import shutil
from pathlib import Path
import sys


MODEL_ID = "facebook/nllb-200-3.3B"
MIN_FREE_BYTES = 10 * 1024**3  # 10 Go


def _human_gb(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} Go"


def main() -> int:
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "assets" / "models" / "nllb-200-3.3b-ct2"
    model_bin = output_dir / "model.bin"

    # 1) ctranslate2 importable
    try:
        import ctranslate2
        from ctranslate2.converters import TransformersConverter
    except Exception as exc:
        print(f"❌ Échec : ctranslate2 non importable ({exc})")
        return 1

    # 2) Déjà converti
    if model_bin.exists() and model_bin.is_file() and model_bin.stat().st_size > 0:
        print("Déjà converti, rien à faire")
        print(f"model.bin: {model_bin} ({_human_gb(model_bin.stat().st_size)})")
        return 0

    # 3) Conversion partielle -> nettoyage
    if output_dir.exists():
        print(f"⚠️  Conversion partielle détectée: {output_dir}")
        print("🧹 Suppression du dossier incomplet...")
        shutil.rmtree(output_dir, ignore_errors=True)

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # 4) Vérification espace disque
    usage = shutil.disk_usage(output_dir.parent)
    free_bytes = int(usage.free)
    print(f"Espace libre: {_human_gb(free_bytes)}")
    if free_bytes < MIN_FREE_BYTES:
        print("⚠️  Moins de 10 Go libres. La conversion peut échouer.")
        answer = input("Continuer quand même ? (o/n) ").strip().lower()
        if answer != "o":
            print("Annulé.")
            return 1

    # 5) Conversion
    try:
        print(f"⏳ Conversion {MODEL_ID} -> CT2 int8...")
        print("ℹ️  Le modèle est téléchargé dans le cache HuggingFace standard.")

        converter = TransformersConverter(MODEL_ID)
        converter.convert(str(output_dir), quantization="int8", force=True)

        if not model_bin.exists() or model_bin.stat().st_size <= 0:
            raise RuntimeError("model.bin absent après conversion")

        size_bytes = model_bin.stat().st_size
        print(f"model.bin créé: {model_bin} ({_human_gb(size_bytes)})")

        # Log device capability info (compatibilité CUDA visible)
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            torch_cuda = str(getattr(torch.version, "cuda", None))
            ct2_cuda_types = ctranslate2.get_supported_compute_types("cuda") if cuda_available else set()
            print(f"torch.cuda.is_available: {cuda_available}")
            print(f"torch CUDA version: {torch_cuda}")
            if cuda_available:
                print(f"CT2 compute types CUDA: {ct2_cuda_types}")
        except Exception as exc:
            print(f"⚠️  Info compatibilité CUDA indisponible: {exc}")

        print("✅ NLLB CT2 prêt")
        return 0

    except Exception as exc:
        print(f"❌ Échec : {exc}")
        # Nettoyage d'un éventuel dossier incomplet
        if output_dir.exists() and not model_bin.exists():
            shutil.rmtree(output_dir, ignore_errors=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
