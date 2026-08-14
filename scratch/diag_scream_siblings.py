# -*- coding: utf-8 -*-
"""Balaie toutes les paires de bulles adjacentes de POV_V2 pour voir sur
combien de paires `_shrink_zone_away_from_siblings` (le code REEL, importe
tel quel, pas une reimplementation) declenche un rognage, et quel axe il
choisit — histoire de jauger le rayon d'impact d'un correctif sur cette
fonction. N'appelle que la methode statique existante, aucune modif.
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"A:\omni read")))

from core import TextRenderer

meta = json.load(open(r"A:\omni read\scratch\render_out\POV_V2\bubbles_meta.json", encoding="utf-8"))
bulles = [m for m in meta if m["class"] == "bulle"]

# Approxime la zone utile de _get_inner_zone par un simple shrink de bbox a
# 0.08 (sans recentrage sur le masque, indisponible ici) : suffisant pour
# reperer QUELLES paires provoquent un rognage et sur QUEL axe.
def approx_zone(bbox, ratio=0.08):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    sx, sy = max(3, int(bw * ratio)), max(3, int(bh * ratio))
    return (x1 + sx, y1 + sy, x2 - sx, y2 - sy)


n_triggered = 0
n_x_axis = 0
n_y_axis = 0
n_degenerate = 0
for i, b in enumerate(bulles):
    zone = approx_zone(tuple(b["bbox"]))
    siblings = [tuple(o["bbox"]) for j, o in enumerate(bulles) if j != i]
    # Ne garde que les voisins qui chevauchent reellement la bbox (pas la zone
    # deja retrecie) pour rester proche du sibling_boxes reel passe par
    # render_iterate.py (bbox brutes de toutes les AUTRES detections).
    out = TextRenderer._shrink_zone_away_from_siblings(zone, siblings)
    if out != zone:
        n_triggered += 1
        zw_in, zh_in = zone[2] - zone[0], zone[3] - zone[1]
        zw_out, zh_out = out[2] - out[0], out[3] - out[1]
        kept = (zw_out * zh_out) / max(1, zw_in * zh_in)
        axis = "X" if zw_out != zw_in else ("Y" if zh_out != zh_in else "?")
        if zw_out <= 0 or zh_out <= 0:
            n_degenerate += 1
        if axis == "X":
            n_x_axis += 1
        elif axis == "Y":
            n_y_axis += 1
        print(f"#{b['index']:02d} {b['ocr_text'][:30]!r:33s} zone_in={zone} -> zone_out={out} "
              f"axis={axis} kept={kept:.2f}")

print(f"\nTotal bulles: {len(bulles)}  paires avec rognage: {n_triggered}  "
      f"axe X: {n_x_axis}  axe Y: {n_y_axis}  degenere/fallback: {n_degenerate}")
