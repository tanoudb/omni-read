# -*- coding: utf-8 -*-
"""Pour les 4 paires POV_V2 ou _shrink_zone_away_from_siblings choisit
aujourd'hui l'axe X, calcule ce que donnerait l'axe Y alternatif, pour
voir si le correctif 'garder l'aire max' changerait leur resultat."""
import json
from pathlib import Path

meta = json.load(open(r"A:\omni read\scratch\render_out\POV_V2\bubbles_meta.json", encoding="utf-8"))
bulles = [m for m in meta if m["class"] == "bulle"]


def approx_zone(bbox, ratio=0.08):
    x1, y1, x2, y2 = bbox
    bw, bh = x2 - x1, y2 - y1
    sx, sy = max(3, int(bw * ratio)), max(3, int(bh * ratio))
    return (x1 + sx, y1 + sy, x2 - sx, y2 - sy)


targets = {23, 24, 26, 27}
for i, b in enumerate(bulles):
    if b["index"] not in targets:
        continue
    zone = approx_zone(tuple(b["bbox"]))
    siblings = [tuple(o["bbox"]) for j, o in enumerate(bulles) if j != i]
    zx1, zy1, zx2, zy2 = zone
    for sx1, sy1, sx2, sy2 in siblings:
        ox1, oy1 = max(zx1, sx1), max(zy1, sy1)
        ox2, oy2 = min(zx2, sx2), min(zy2, sy2)
        if ox2 <= ox1 or oy2 <= oy1:
            continue
        zcx, zcy = (zx1 + zx2) / 2.0, (zy1 + zy2) / 2.0
        scx, scy = (sx1 + sx2) / 2.0, (sy1 + sy2) / 2.0
        cand_x = (zx1, zy1, min(zx2, ox1), zy2) if zcx <= scx else (max(zx1, ox2), zy1, zx2, zy2)
        cand_y = (zx1, zy1, zx2, min(zy2, oy1)) if zcy <= scy else (zx1, max(zy1, oy2), zx2, zy2)
        area = lambda z: max(0, z[2]-z[0]) * max(0, z[3]-z[1])
        print(f"#{b['index']:02d} {b['ocr_text'][:25]!r:28s} sibling={tuple(round(v) for v in (sx1,sy1,sx2,sy2))} "
              f"cand_X={cand_x} area={area(cand_x)}  cand_Y={cand_y} area={area(cand_y)}  "
              f"-> fix_pick={'X (unchanged)' if area(cand_x)>=area(cand_y) else 'Y (CHANGED)'}")
