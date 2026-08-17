# -*- coding: utf-8 -*-
"""Banc PaddleOCR-VL : meme verite terrain que scratch/ocr_bench.py.

A executer DANS le venv qui contient paddleocr >= 3.7 :
    .venv_paddle_next\\Scripts\\python.exe scratch/bench_vl.py
"""
import json
import sys
from pathlib import Path

import cv2

# On importe le module par son CHEMIN, sans passer par `core/` : ce venv
# n'a que paddle, pas torch.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_bench", r"A:\omni read\scratch\ocr_bench.py")
_m = _ilu.module_from_spec(_spec)
import types as _t
_fake = _t.ModuleType('core'); _fake.OCREngine = None
sys.modules['core'] = _fake
_spec.loader.exec_module(_m)
CASES, levenshtein = _m.CASES, _m.levenshtein


def main():
    root = Path("scratch/render_out")
    try:
        from paddleocr import PaddleOCRVL
        pipe = PaddleOCRVL()
        mode = "PaddleOCRVL"
    except Exception as exc:
        print(f"PaddleOCRVL indisponible: {exc}")
        return

    total_ed = total_len = exact = 0
    rows = []
    for run, idx, expected in CASES:
        p = root / run / "bubbles" / f"{idx:02d}_before.png"
        if not p.exists():
            continue
        try:
            out = pipe.predict(str(p))
        except Exception as exc:
            print(f"  [ERR] {run}#{idx:02d}: {exc}")
            continue

        texts = []
        for res in out:
            rd = res.json if hasattr(res, "json") else res
            if isinstance(rd, dict) and "res" in rd:
                rd = rd["res"]
            if not isinstance(rd, dict):
                continue
            # PaddleOCR-VL ne rend PAS `rec_texts` : il rend une liste de
            # blocs de mise en page, chacun avec son contenu deja assemble.
            blocks = rd.get("parsing_res_list") or []
            if blocks:
                for b in blocks:
                    c = (b or {}).get("block_content")
                    if c:
                        texts.append(str(c))
            else:
                for key in ("rec_texts", "texts"):
                    if rd.get(key):
                        texts.extend(str(t) for t in rd[key])
                        break
        got = " ".join(" ".join(texts).split()).strip()

        ed = levenshtein(got.upper(), expected.upper())
        total_ed += ed
        total_len += len(expected)
        ok = got.upper() == expected.upper()
        exact += int(ok)
        rows.append({"case": f"{run}#{idx:02d}", "attendu": expected,
                     "obtenu": got, "distance": ed})
        print(f"  [{'OK ' if ok else f'{ed:3d}'}] {run}#{idx:02d} {got[:60]!r}")

    cer = total_ed / max(1, total_len)
    print(f"\n=== {mode} ===")
    print(f"exact: {exact}/{len(rows)}   CER: {cer:.4f}  ({total_ed}/{total_len})")

    out_path = Path("scratch/ocr_bench_results.json")
    payload = {}
    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    payload["paddleocr_vl"] = {"scale": 1.0, "exact": exact, "n": len(rows),
                               "cer": round(cer, 4), "edits": total_ed, "rows": rows}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
