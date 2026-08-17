# -*- coding: utf-8 -*-
"""Banc de mesure OCR : verite terrain vs sortie moteur, sur des crops reels.

Sert a comparer objectivement des configurations (agrandissement, parametres
PaddleOCR) au lieu de juger a l'oeil bulle par bulle.

Usage:
    python scratch/ocr_bench.py [--scale 2.0] [--label baseline]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import numpy as np

from core import OCREngine

# (dossier_run, index, texte attendu) — verite terrain relevee a l'oeil sur
# les crops `NN_before.png`, en incluant la ponctuation.
CASES = [
    ("FRONTIER_FINAL", 27, "JONAS! JONAS!"),
    ("FRONTIER_FINAL", 37, "DID YOU HAVE A NIGHTMARE?"),
    ("FRONTIER_FINAL", 42, "THAT WAS A LONG TIME AGO."),
    ("FRONTIER_FINAL", 54, "AS YOU KNOW,"),
    ("FRONTIER_FINAL", 56, "MANA CIRCULATES AT A SPEED THAT MATCHES THE SIZE OF THAT VESSEL."),
    ("FRONTIER_FINAL", 65, "MANY WERE KILLED OR INJURED."),
    ("FRONTIER_FINAL", 66, "IT WOULD GO QUIET, ONLY TO ERUPT AGAIN WITHOUT WARNING."),
    ("FRONTIER_FINAL", 67, "IT KEPT EVERYONE IN THE ESTATE ON EDGE."),
    ("FRONTIER_FINAL", 69, "PLEASE, LOCK ME AWAY."),
    ("FRONTIER_FINAL", 72, "I THINK I WAS FIFTEEN THEN."),
    ("FRONTIER_FINAL", 77, "IT'S BECAUSE HE STAYED BY MY SIDE FOR TOO LONG."),
    ("POV_V9", 2, "DAMN IT... I CAN'T LAND A SINGLE HIT."),
    ("POV_V9", 20, "IT MIGHT JUST BE A NORMAL RUN, BUT..."),
    ("POV_V9", 21, "MAKE SURE YOU COME BACK SAFE, OKAY?"),
    ("DRAGON_FINAL", 6, "MY HOME, THE ROSNOVA FAMILY, WAS MY ONLY TARGET."),
    ("DRAGON_FINAL", 23, "FERDA, ENEMY OF THE CONTINENT! I WILL BRING YOU TO JUSTICE!!"),
]


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="agrandissement applique au crop AVANT l'OCR")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--out", type=Path, default=Path("scratch/ocr_bench_results.json"))
    a = ap.parse_args()

    root = Path("scratch/render_out")
    crops, refs, names = [], [], []
    for run, idx, expected in CASES:
        p = root / run / "bubbles" / f"{idx:02d}_before.png"
        img = cv2.imread(str(p))
        if img is None:
            print(f"  (absent) {p}")
            continue
        if a.scale != 1.0:
            img = cv2.resize(img, None, fx=a.scale, fy=a.scale,
                             interpolation=cv2.INTER_LANCZOS4)
        crops.append(img)
        refs.append(expected)
        names.append(f"{run}#{idx:02d}")

    engine = OCREngine(device="cuda")
    results = engine.extract_batch(crops)

    total_ed = total_len = exact = 0
    rows = []
    for name, expected, res in zip(names, refs, results):
        got = (res[0] or "").strip()
        ed = levenshtein(got.upper(), expected.upper())
        total_ed += ed
        total_len += len(expected)
        ok = got.upper() == expected.upper()
        exact += int(ok)
        rows.append({"case": name, "attendu": expected, "obtenu": got, "distance": ed})
        flag = "OK " if ok else f"{ed:3d}"
        print(f"  [{flag}] {name:22s} {got[:56]!r}")
        if not ok:
            print(f"          attendu {expected[:56]!r}")

    cer = total_ed / max(1, total_len)
    print(f"\n=== {a.label} (scale={a.scale}) ===")
    print(f"exact: {exact}/{len(rows)}   CER: {cer:.4f}  ({total_ed} erreurs / {total_len} car.)")

    payload = {}
    if a.out.exists():
        try:
            payload = json.loads(a.out.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    payload[a.label] = {
        "scale": a.scale, "exact": exact, "n": len(rows),
        "cer": round(cer, 4), "edits": total_ed, "rows": rows,
    }
    a.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
