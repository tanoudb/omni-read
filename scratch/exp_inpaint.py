# -*- coding: utf-8 -*-
"""Compare plusieurs strategies d'inpainting sur les zones out_text mises en
cache par erase_lab.py --build. Produit une planche de comparaison.

Usage: python scratch/exp_inpaint.py <cache_dir> <out_dir> [--idx 1]
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image


def local_mask_for(item, crop_shape, crop_origin, dilate_k):
    x1, y1, x2, y2 = item["bbox"]
    cx1, cy1 = crop_origin
    ch, cw = crop_shape[:2]
    det_h, det_w = max(1, y2 - y1), max(1, x2 - x1)
    cm = item["chirurgical_mask"]
    if cm is None:
        return None
    if cm.shape[:2] != (det_h, det_w):
        cm = cv2.resize(cm, (det_w, det_h), interpolation=cv2.INTER_NEAREST)
    m = np.zeros((ch, cw), dtype=np.uint8)
    ox, oy = x1 - cx1, y1 - cy1
    dx1, dy1 = max(0, ox), max(0, oy)
    dx2, dy2 = min(cw, ox + det_w), min(ch, oy + det_h)
    sx1, sy1 = dx1 - ox, dy1 - oy
    m[dy1:dy2, dx1:dx2] = (cm[sy1:sy1 + (dy2 - dy1), sx1:sx1 + (dx2 - dx1)] > 0).astype(np.uint8) * 255
    if dilate_k > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)))
    return m


def blend(crop, result, mask):
    o = crop.copy()
    o[mask > 0] = result[mask > 0]
    return o


def lama_run(lama, crop, mask):
    p = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    mp = Image.fromarray(mask).convert("L")
    res = cv2.cvtColor(np.array(lama(p, mp)), cv2.COLOR_RGB2BGR)
    if res.shape[:2] != crop.shape[:2]:
        res = cv2.resize(res, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    return res


def lama_downscaled(lama, crop, mask, cap=512):
    h, w = crop.shape[:2]
    s = min(1.0, cap / float(max(h, w)))
    if s >= 1.0:
        return lama_run(lama, crop, mask)
    small = cv2.resize(crop, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    smask = cv2.resize(mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
    res = lama_run(lama, small, smask)
    return cv2.resize(res, (w, h), interpolation=cv2.INTER_LANCZOS4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--idx", type=int, default=1)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(args.cache_dir / "page.png"))
    items = pickle.load(open(args.cache_dir / "dets.pkl", "rb"))
    it = items[args.idx]
    x1, y1, x2, y2 = it["bbox"]
    H, W = img.shape[:2]
    box_h = y2 - y1

    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()

    variants = []

    def add(name, margin, dilate_k, fn):
        cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
        cx2, cy2 = min(W, x2 + margin), min(H, y2 + margin)
        crop = img[cy1:cy2, cx1:cx2].copy()
        m = local_mask_for(it, crop.shape, (cx1, cy1), dilate_k)
        if m is None or m.sum() == 0:
            return
        res = fn(crop, m)
        outc = blend(crop, res, m)
        # recadre sur la bbox + 40px pour comparer a taille egale
        vx1, vy1 = max(0, x1 - 40) - cx1, max(0, y1 - 40) - cy1
        vx2, vy2 = vx1 + (min(W, x2 + 40) - max(0, x1 - 40)), vy1 + (min(H, y2 + 40) - max(0, y1 - 40))
        view = outc[vy1:vy2, vx1:vx2]
        cov = np.count_nonzero(m) / m.size
        variants.append((f"{name} (mask {cov:.0%})", view))
        print(f"{name}: crop={crop.shape[1]}x{crop.shape[0]} mask={cov:.1%}")

    add("V0 actuel d11 m60", 60, 11, lambda c, m: lama_run(lama, c, m))
    add("V1 d3 m60", 60, 3, lambda c, m: lama_run(lama, c, m))
    add("V2 d3 m=2*h", max(60, 2 * box_h), 3, lambda c, m: lama_run(lama, c, m))
    add("V3 d3 m=2*h ds512", max(60, 2 * box_h), 3, lambda c, m: lama_downscaled(lama, c, m, 512))
    add("V4 d3 m=2*h ds768", max(60, 2 * box_h), 3, lambda c, m: lama_downscaled(lama, c, m, 768))
    add("V5 d5 m=3*h", max(60, 3 * box_h), 5, lambda c, m: lama_run(lama, c, m))

    orig = img[max(0, y1 - 40):min(H, y2 + 40), max(0, x1 - 40):min(W, x2 + 40)]
    cells = [("ORIGINAL", orig)] + variants

    ch = max(c.shape[0] for _, c in cells)
    cw = max(c.shape[1] for _, c in cells)
    cols = 4
    rows = (len(cells) + cols - 1) // cols
    lab = 22
    sheet = np.full((rows * (ch + lab), cols * cw, 3), 25, dtype=np.uint8)
    for i, (name, c) in enumerate(cells):
        r, cc = divmod(i, cols)
        y0 = r * (ch + lab)
        sheet[y0 + lab:y0 + lab + c.shape[0], cc * cw:cc * cw + c.shape[1]] = c
        cv2.putText(sheet, name, (cc * cw + 4, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
    p = args.out_dir / f"compare_idx{args.idx}.png"
    cv2.imwrite(str(p), sheet)
    print("->", p)


if __name__ == "__main__":
    main()
