import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so sibling packages `utils` and `core` import correctly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import gemini_prompt
import core.translator_gemini as tg


def check_font_map():
    print("Checking FONT_MAP keys and paths...")
    for k, p in gemini_prompt.FONT_MAP.items():
        ok = p.startswith("A:\\") or p.startswith(r"A:/")
        print(f" - {k}: {p} -> startswith A:\\ : {ok}")


def check_response_schema():
    print("\nChecking RESPONSE_SCHEMA for font_key enum...")
    schema = tg.RESPONSE_SCHEMA
    try:
        enums = schema["properties"]["traductions"]["items"]["properties"]["font_key"]["enum"]
        print(f" - Found font_key enum with keys: {enums}")
        expected = list(gemini_prompt.FONT_MAP.keys())
        print(f" - Matches FONT_MAP keys: {enums == expected}")
    except Exception as e:
        print(" - ERROR reading schema:", e)


def simulate_parse_and_fallback():
    print("\nSimulating Gemini JSON parse and font fallback...")
    # Simulated Gemini response (for two Korean source bubbles)
    simulated = {
        "traductions": [
            {"id": "0", "fr": "Qui êtes-vous ?", "font_key": "STANDARD"},
            {"id": "1", "fr": "[DANGER : niveau trop élevé]", "font_key": "SYSTEM"}
        ],
        "state_update": {
            "summary_update": "Un personnage pose une question; une alerte système apparaît.",
            "relationship_changes": [],
            "entity_discovery": {}
        }
    }

    raw = json.dumps(simulated, ensure_ascii=False)
    # feed through the module parser
    parsed = tg.GeminiTranslator._parse_json(raw)
    print(" - Parsed OK:", isinstance(parsed, dict))
    if parsed:
        for item in parsed.get("traductions", []):
            fk = item.get("font_key", "STANDARD")
            if fk not in gemini_prompt.FONT_MAP:
                print(f"   * Unknown font_key '{fk}' -> fallback to STANDARD")
                fk = "STANDARD"
            else:
                print(f"   * id={item.get('id')} fr={item.get('fr')!r} font_key={fk}")


def main():
    check_font_map()
    check_response_schema()
    simulate_parse_and_fallback()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print("Test failed:", e)
        sys.exit(2)
    print("\nSanity test complete.")
