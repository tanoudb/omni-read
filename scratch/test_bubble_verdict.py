# -*- coding: utf-8 -*-
"""Banc du VERDICT « y a-t-il un ballon ? », sur quatre series.

La question n'est pas « quelle forme ? » mais « une forme fermee existe-t-elle
autour de ce texte ? ». C'est cette reponse, et elle seule, qui aiguille
`insert_text` entre `_draw_exact_lines` (texte libre, on rejoue les lignes
d'origine) et le wrap sur polygone de surface (ballon).

Verite terrain : l'etiquette de classe (`bulle` vs `out_text`/`System`). Elle
est imparfaite, mais c'est la seule disponible, et les desaccords sont
justement ce qu'on veut lire.

Usage:  python scratch/test_bubble_verdict.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import numpy as np

from core.bubble_shape import grow_from_ink, ink_mask_from_regions
from core.renderer import TextRenderer

RUNS = [
    ("path-of-vengeance", "scratch/render_out/POV_BBOX"),
    ("i-married-the-dragon", "scratch/render_out/DRAGON_FINAL"),
    ("frontier-count", "scratch/render_out/FRONTIER_FINAL"),
    ("hellogin", "scratch/render_out/hellogin_ch1_iter9"),
]
LIBRE = ("out_text", "system")


def main():
    r = TextRenderer.__new__(TextRenderer)
    total = {"n": 0, "t": 0, "c": 0}
    desaccords = []

    for nom, run in RUNS:
        p = Path(run)
        meta = json.loads((p / "bubbles_meta.json").read_text(encoding="utf-8"))
        orig = cv2.imread(str(p / "page_before.png"))
        era = cv2.imread(str(p / "page_erased.png"))
        if orig is None or era is None:
            print(f"{nom:<22s} run illisible")
            continue
        H, W = orig.shape[:2]

        n = nt = nc = 0
        for d in meta:
            x1, y1, x2, y2 = d["bbox"]
            if not (0 <= x1 < x2 <= W and 0 <= y1 < y2 <= H):
                continue
            c_o, c_e = orig[y1:y2, x1:x2], era[y1:y2, x1:x2]
            if c_o.size == 0 or c_e.shape[:2] != c_o.shape[:2]:
                continue
            cls = str(d.get("class", "")).lower().strip()
            libre = cls in LIBRE
            bw, bh = x2 - x1, y2 - y1

            t = r._bubble_shape_mask(None, c_e, bw, bh, is_bubble=(cls == "bulle"))
            t_ballon = t is not None and r._is_non_rectangular(t)

            ink = ink_mask_from_regions(c_o, d.get("text_regions"))
            m, diag = grow_from_ink(c_e, ink)
            c_ballon = m is not None

            n += 1
            nt += (not t_ballon) == libre
            nc += (not c_ballon) == libre
            if (not c_ballon) != libre:
                desaccords.append((nom, d["index"], cls, diag["mode"],
                                   round(diag["fill"], 3), bw, bh))

        total["n"] += n; total["t"] += nt; total["c"] += nc
        pt = 100.0 * nt / n if n else 0.0
        pc = 100.0 * nc / n if n else 0.0
        print(f"{nom:<22s} {n:3d} cas   TEINTE {nt:3d} ({pt:5.1f}%)   "
              f"CROISSANCE {nc:3d} ({pc:5.1f}%)")

    n = total["n"]
    print("-" * 74)
    print(f"{'TOTAL':<22s} {n:3d} cas   TEINTE {total['t']:3d} "
          f"({100.0*total['t']/n:5.1f}%)   CROISSANCE {total['c']:3d} "
          f"({100.0*total['c']/n:5.1f}%)")

    if desaccords:
        print(f"\nDesaccords de la CROISSANCE avec l'etiquette ({len(desaccords)}) :")
        print(f"  {'serie':<22s} {'#':>3s} {'classe':<9s} {'mode':<14s} "
              f"{'fill':>5s} {'taille':>10s}")
        for s, i, cls, mode, fill, bw, bh in desaccords:
            print(f"  {s:<22s} {i:3d} {cls:<9s} {mode:<14s} {fill:5.2f} "
                  f"{bw:4d}x{bh:<4d}")


if __name__ == "__main__":
    main()
