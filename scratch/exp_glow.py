# -*- coding: utf-8 -*-
"""Compare masque-glyphe vs masque-bloc (glyphe + lueur) sur les cartouches
out_text, avec un gros contexte pour LaMa.

Usage: python scratch/exp_glow.py <cache_dir> <out_dir>
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image


def glyph_mask(item, shape, origin, dil):
    x1, y1, x2, y2 = item["bbox"]
    cx1, cy1 = origin
    ch, cw = shape[:2]
    dh, dw = max(1, y2 - y1), max(1, x2 - x1)
    cm = item["chirurgical_mask"]
    if cm is None:
        return None
    if cm.shape[:2] != (dh, dw):
        cm = cv2.resize(cm, (dw, dh), interpolation=cv2.INTER_NEAREST)
    m = np.zeros((ch, cw), np.uint8)
    ox, oy = x1 - cx1, y1 - cy1
    m[oy:oy + dh, ox:ox + dw] = (cm > 0).astype(np.uint8) * 255
    if dil > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil, dil)))
    return m


def block_mask(item, shape, origin, grow):
    """Masque = polygones de LIGNE OCR dilates de `grow` px (glyphe + lueur)."""
    x1, y1, x2, y2 = item["bbox"]
    cx1, cy1 = origin
    ch, cw = shape[:2]
    m = np.zeros((ch, cw), np.uint8)
    for reg in item["text_regions"] or []:
        pts = reg.get("bbox") if isinstance(reg, dict) else None
        if not pts:
            continue
        arr = np.array([[int(p[0]) + x1 - cx1, int(p[1]) + y1 - cy1] for p in pts], np.int32)
        if arr.shape[0] >= 3:
            cv2.fillPoly(m, [arr], 255)
    if grow > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
        m = cv2.dilate(m, k)
    return m


def lama_run(lama, crop, mask):
    p = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    res = cv2.cvtColor(np.array(lama(p, Image.fromarray(mask).convert("L"))), cv2.COLOR_RGB2BGR)
    if res.shape[:2] != crop.shape[:2]:
        res = cv2.resize(res, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(a.cache_dir / "page.png"))
    items = pickle.load(open(a.cache_dir / "dets.pkl", "rb"))
    H, W = img.shape[:2]
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()

    labels = ["ORIGINAL", "glyphe d5", "bloc +0.15lh", "bloc +0.30lh", "bloc +0.50lh"]
    rows = []
    for idx, it in enumerate(items):
        x1, y1, x2, y2 = it["bbox"]
        bh = y2 - y1
        lh = it.get("source_line_height") or 30.0
        margin = max(60, 2 * bh)
        cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
        cx2, cy2 = min(W, x2 + margin), min(H, y2 + margin)
        crop = img[cy1:cy2, cx1:cx2].copy()
        vx, vy = max(0, x1 - 40) - cx1, max(0, y1 - 40) - cy1
        vw = min(W, x2 + 40) - max(0, x1 - 40)
        vh = min(H, y2 + 40) - max(0, y1 - 40)

        cells = [crop[vy:vy + vh, vx:vx + vw].copy()]
        variants = [("glyph", glyph_mask(it, crop.shape, (cx1, cy1), 5)),
                    ("b15", block_mask(it, crop.shape, (cx1, cy1), int(0.15 * lh))),
                    ("b30", block_mask(it, crop.shape, (cx1, cy1), int(0.30 * lh))),
                    ("b50", block_mask(it, crop.shape, (cx1, cy1), int(0.50 * lh)))]
        covs = []
        for name, m in variants:
            if m is None or m.sum() == 0:
                cells.append(np.zeros((vh, vw, 3), np.uint8))
                covs.append(0)
                continue
            res = lama_run(lama, crop, m)
            o = crop.copy()
            o[m > 0] = res[m > 0]
            cells.append(o[vy:vy + vh, vx:vx + vw])
            covs.append(np.count_nonzero(m) / m.size)
        rows.append((idx, cells, covs))
        print(f"{idx}: lh={lh} " + " ".join(f"{n}={c:.1%}" for (n, _), c in zip(variants, covs)))

    ch = max(c.shape[0] for _, cs, _ in rows for c in cs)
    cw = max(c.shape[1] for _, cs, _ in rows for c in cs)
    lab = 20
    sheet = np.full((len(rows) * (ch + lab), len(labels) * cw, 3), 25, np.uint8)
    for ri, (idx, cs, covs) in enumerate(rows):
        y0 = ri * (ch + lab)
        for ci, c in enumerate(cs):
            sheet[y0 + lab:y0 + lab + c.shape[0], ci * cw:ci * cw + c.shape[1]] = c
            t = labels[ci] if ci == 0 else f"{labels[ci]} {covs[ci-1]:.0%}"
            cv2.putText(sheet, f"#{idx} {t}", (ci * cw + 4, y0 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
    p = a.out_dir / "glow_compare.png"
    cv2.imwrite(str(p), sheet)
    print("->", p)


if __name__ == "__main__":
    main()
