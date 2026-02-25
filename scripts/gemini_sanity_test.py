"""Sanity test for Gemini prompt/schema/font handling.

Runs these checks:
- FONT_MAP paths begin with 'A:\\omni read\\'
- Try to build a google-genai GenerateContentConfig with RESPONSE_SCHEMA (if installed)
- Simulate two Korean bubbles and verify font_key selection and fallback
"""
import json
import traceback
from utils.gemini_prompt import FONT_MAP
import core.translator_gemini as tg


def check_font_map():
    print("FONT_MAP keys:")
    ok = True
    for k, p in FONT_MAP.items():
        print(f" - {k}: {p}")
        lp = p.replace('/', '\\')
        if not lp.lower().startswith(r"a:\\omni read\\"):
            print(f"   -> ERROR: path for {k} does not start with A:\\omni read\\")
            ok = False
    return ok


def try_validate_schema():
    try:
        from google.genai import types
        cfg = types.GenerateContentConfig(response_json_schema=tg.RESPONSE_SCHEMA)
        print("Schema accepted by google.genai GenerateContentConfig (instantiation OK)")
        return True
    except Exception as e:
        print("Schema validation with google.genai failed or google-genai not installed:")
        traceback.print_exc()
        return False


def simulate_translation():
    bubbles = ["당신은 누구십니까?", "[위험: 레벨이 너무 높습니다]"]
    # Simulate a Gemini parsed response we expect
    parsed = {
        "traductions": [
            {"id": "0", "fr": "Qui êtes-vous ?", "font_key": "STANDARD"},
            {"id": "1", "fr": "[DANGER : le niveau est trop élevé]", "font_key": "SYSTEM"}
        ],
        "state_update": {
            "summary_update": "Une alerte a été détectée; niveau critique.",
            "relationship_changes": [],
            "entity_discovery": {}
        }
    }

    # Emulate translator logic for font_key fallback
    results = {}
    font_choices = {}
    for item in parsed["traductions"]:
        idx = item.get("id")
        fr = item.get("fr")
        fk = item.get("font_key") or "STANDARD"
        if fk not in FONT_MAP:
            print(f"Font key '{fk}' unknown, falling back to STANDARD")
            fk = "STANDARD"
        results[str(idx)] = fr
        font_choices[str(idx)] = fk

    print("Simulated translation results:")
    print(json.dumps({"results": results, "fonts": font_choices}, ensure_ascii=False, indent=2))
    return results, font_choices


if __name__ == '__main__':
    print("Running Gemini sanity test...")
    fonts_ok = check_font_map()
    print(f"FONT_MAP paths OK: {fonts_ok}")
    schema_ok = try_validate_schema()
    print(f"Schema validated by client: {schema_ok}")
    res, fonts = simulate_translation()
    # Expected: 0 -> STANDARD, 1 -> SYSTEM
    print("Expectation: bubble 0 -> STANDARD, bubble 1 -> SYSTEM")
    print(f"Observed: bubble0={fonts.get('0')}, bubble1={fonts.get('1')}")