
from core.translation.translator_nllb import NLLBCT2Translator
from config import config


def main() -> None:
    tests = [
        "I still want to reach the ultimate heights of martial arts.",
        "Are you telling me this is the fruit of immortality?",
        "UGH... I can still fight!",
        "This watermark should remain unchanged: reset-scan.co",
        "Please help me, Yunho!",
    ]

    print("[NLLB] Test CT2")
    print(f"   source_model: {config.translation.nllb_source_model}")
    print(f"   model_name:   {config.translation.model_name}")
    print(f"   ct2_dir:      {config.translation.nllb_ct2_model_dir}")
    translator = NLLBCT2Translator(device="cuda")
    try:
        outputs = translator.translate_batch(tests, src_lang="eng_Latn", tgt_lang="fra_Latn")
    except Exception as exc:
        print(f"[NLLB] Test impossible: {exc}")
        print("[NLLB] Installe les dépendances puis convertis le modèle:")
        print("   pip install -r requirements.txt")
        print("   python scripts/convert_nllb_ct2.py")
        return

    for idx, (src, out) in enumerate(zip(tests, outputs), start=1):
        print(f"\n[{idx}] EN: {src}")
        print(f"[{idx}] FR: {out}")


if __name__ == "__main__":
    main()
