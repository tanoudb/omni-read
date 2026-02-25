import os
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.translator_gemini import GeminiTranslator
from utils.gemini_prompt import PromptBank, FONT_MAP


def main():
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("GEMINI_API_KEY not set in environment. Aborting live test.")
        return 2

    tr = GeminiTranslator(series_name="live_test", source_lang="en")

    texts = [
        "I... I can't breathe... What is this terrifying pressure?",
        "[CRITICAL WARNING: MANA STABILIZATION AT 12%]",
        "DIE! YOU FILTHY MONSTER!",
    ]

    clean = texts
    numbered = "\n".join(f"{i}: {t}" for i, t in enumerate(clean))
    ctx = tr._build_context()

    prompt = (
        f"{ctx}\n\n"
        f"{PromptBank.LANG_RULES.get(tr.source_lang, '')}\n\n"
        f"{PromptBank.TYPO_RULES}\n\n"
        f"TEXTES A TRADUIRE (id: texte source) :\n{numbered}\n\n"
        "Pour chaque item, renvoie un objet {id, fr, font_key} où `font_key` est l'une des clés suivantes: "
        f"{', '.join(list(FONT_MAP.keys()))}.\n"
        "Renvoie un JSON au format exact demandé. Inclut aussi un objet `state_update`\n"
        "contenant `summary_update`, `relationship_changes` et `entity_discovery` si pertinent.\n"
        "Ne répète pas l'ancien résumé; fournis seulement les faits nouveaux dans `summary_update`.\n"
    )

    print("Sending prompt to Gemini... (this may take a few seconds)")
    parsed = tr._call(prompt)
    if not parsed:
        print("No response or parse failed.")
        return 3

    print("\nTranslations returned:")
    for item in parsed.get("traductions", []):
        if not isinstance(item, dict):
            continue
        idv = item.get("id")
        fr = item.get("fr")
        fk = item.get("font_key") or "STANDARD"
        if fk not in FONT_MAP:
            print(f" - id={idv} fr={fr!r} font_key={fk} -> UNKNOWN, fallback to STANDARD")
        else:
            print(f" - id={idv} fr={fr!r} font_key={fk}")

    print("\nstate_update:")
    su = parsed.get("state_update") or {}
    print(json_pretty(su))

    print("\nSaving merged state and intrigue files (if any changes were returned)...")
    # Apply same merges as translate_page_json would
    summary_update = su.get("summary_update")
    if summary_update:
        tr._update_intrigue(summary_update)
    entity_disc = su.get("entity_discovery") or {}
    if isinstance(entity_disc, dict):
        for k, v in entity_disc.items():
            if isinstance(v, dict):
                tr._state.setdefault(k, {}).update(v)
    rel = su.get("relationship_changes") or []
    if isinstance(rel, list) and rel:
        lst = tr._state.setdefault("relationship_changes", [])
        lst.extend(r for r in rel if isinstance(r, str))
    tr._save_state()

    print("\nLive test complete.")
    return 0


def json_pretty(obj):
    try:
        import json
        return json.dumps(obj, ensure_ascii=False, indent=2)
    except Exception:
        return str(obj)


if __name__ == '__main__':
    sys.exit(main())
