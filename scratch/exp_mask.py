# -*- coding: utf-8 -*-
"""Teste le remplissage de trous du masque de glyphes (outline -> glyphe plein)
puis l'inpainting, et compare aux variantes precedentes.

Usage: python scratch/exp_mask.py <cache_dir> <out_dir> --idx N
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image


def fill_glyph_holes(mask: np.ndarray, max_hole_area: int) -> np.ndarray:
    """Bouche les trous FERMES du masque (l'interieur d'un contour de lettre).

    Le seuillage par Otsu ne retient qu'un cote: sur du texte d'impact
    (remplissage noir + contour blanc epais) il attrape le contour et laisse le
    corps de la lettre en place. Les trous ainsi laisses sont fermes et petits;
    les combler reconstitue le glyphe entier.
    """
    inv = (mask == 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, 4)
    filled = mask.copy()
    h, w = mask.shape[:2]
    for lab in range(1, n):
        x, y, bw, bh, area = stats[lab]
        touches_border = (x == 0 or y == 0 or x + bw >= w or y + bh >= h)
        if touches_border:
            continue
        if area <= max_hole_area:
            filled[labels == lab] = 255
    return filled


def local_mask(item, crop_shape, origin, dilate_k, fill_holes, line_h):
    x1, y1, x2, y2 = item["bbox"]
    cx1, cy1 = origin
    ch, cw = crop_shape[:2]
    det_h, det_w = max(1, y2 - y1), max(1, x2 - x1)
    cm = item["chirurgical_mask"]
    if cm is None:
        return None
    if cm.shape[:2] != (det_h, det_w):
        cm = cv2.resize(cm, (det_w, det_h), interpolation=cv2.INTER_NEAREST)
    if fill_holes:
        cm = fill_glyph_holes(cm, int((line_h or 30) ** 2 * 1.2))
    m = np.zeros((ch, cw), dtype=np.uint8)
    ox, oy = x1 - cx1, y1 - cy1
    dx1, dy1 = max(0, ox), max(0, oy)
    dx2, dy2 = min(cw, ox + det_w), min(ch, oy + det_h)
    sx1, sy1 = dx1 - ox, dy1 - oy
    m[dy1:dy2, dx1:dx2] = (cm[sy1:sy1 + (dy2 - dy1), sx1:sx1 + (dx2 - dx1)] > 0).astype(np.uint8) * 255
    if dilate_k > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)))
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
    ap.add_argument("--idx", type=int, default=1)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(a.cache_dir / "page.png"))
    items = pickle.load(open(a.cache_dir / "dets.pkl", "rb"))
    it = items[a.idx]
    x1, y1, x2, y2 = it["bbox"]
    H, W = img.shape[:2]
    box_h = y2 - y1
    line_h = it.get("source_line_height") or 30

    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()

    cells = [("ORIGINAL", img[max(0, y1 - 40):min(H, y2 + 40), max(0, x1 - 40):min(W, x2 + 40)])]

    def add(name, margin, dil, fill):
        cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
        cx2, cy2 = min(W, x2 + margin), min(H, y2 + margin)
        crop = img[cy1:cy2, cx1:cx2].copy()
        m = local_mask(it, crop.shape, (cx1, cy1), dil, fill, line_h)
        if m is None or m.sum() == 0:
            return
        res = lama_run(lama, crop, m)
        outc = crop.copy()
        outc[m > 0] = res[m > 0]
        vx1, vy1 = max(0, x1 - 40) - cx1, max(0, y1 - 40) - cy1
        vw = min(W, x2 + 40) - max(0, x1 - 40)
        vh = min(H, y2 + 40) - max(0, y1 - 40)
        cov = np.count_nonzero(m) / m.size
        cells.append((f"{name} mask{cov:.0%}", outc[vy1:vy1 + vh, vx1:vx1 + vw]))
        # visu du masque
        mv = crop.copy()
        mv[m > 0] = (0, 0, 255)
        cells.append((f"^masque", cv2.addWeighted(crop, 0.4, mv, 0.6, 0)[vy1:vy1 + vh, vx1:vx1 + vw]))
        print(f"{name}: mask={cov:.1%}")

    add("A fill d3 m=2h", max(60, 2 * box_h), 3, True)
    add("B fill d5 m=2h", max(60, 2 * box_h), 5, True)
    add("C fill d7 m=2h", max(60, 2 * box_h), 7, True)

    ch = max(c.shape[0] for _, c in cells)
    cw = max(c.shape[1] for _, c in cells)
    cols = 3
    rows = (len(cells) + cols - 1) // cols
    lab = 22
    sheet = np.full((rows * (ch + lab), cols * cw, 3), 25, dtype=np.uint8)
    for i, (name, c) in enumerate(cells):
        r, cc = divmod(i, cols)
        y0 = r * (ch + lab)
        sheet[y0 + lab:y0 + lab + c.shape[0], cc * cw:cc * cw + c.shape[1]] = c
        cv2.putText(sheet, name, (cc * cw + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
    p = a.out_dir / f"mask_idx{a.idx}.png"
    cv2.imwrite(str(p), sheet)
    print("->", p)


if __name__ == "__main__":
    main()
