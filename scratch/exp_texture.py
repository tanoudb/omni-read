# -*- coding: utf-8 -*-
"""Compare, sur une bulle a fond TEXTURE, les strategies de secours quand
l'effacement echoue : diffusion Navier-Stokes (actuel) vs LaMa seul, avec
plusieurs marges de contexte.

Usage: python scratch/exp_texture.py <cache_dir> <out.png>
"""
import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from PIL import Image


def build_local_mask(item, crop_shape, origin, dilate_k):
    x1, y1, x2, y2 = item["bbox"]
    cx1, cy1 = origin
    ch, cw = crop_shape[:2]
    dh, dw = max(1, y2 - y1), max(1, x2 - x1)
    cm = item["chirurgical_mask"]
    if cm is None:
        return None
    if cm.shape[:2] != (dh, dw):
        cm = cv2.resize(cm, (dw, dh), interpolation=cv2.INTER_NEAREST)
    m = np.zeros((ch, cw), np.uint8)
    ox, oy = x1 - cx1, y1 - cy1
    m[oy:oy + dh, ox:ox + dw] = (cm > 0).astype(np.uint8) * 255
    if dilate_k > 1:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_k, dilate_k)))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dir", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()

    img = cv2.imread(str(a.cache_dir / "page.png"))
    items = pickle.load(open(a.cache_dir / "dets.pkl", "rb"))
    H, W = img.shape[:2]
    from simple_lama_inpainting import SimpleLama
    lama = SimpleLama()

    it = items[0]
    x1, y1, x2, y2 = it["bbox"]
    bh = y2 - y1

    cells = [("ORIGINAL", img[max(0, y1 - 20):min(H, y2 + 20), max(0, x1 - 20):min(W, x2 + 20)])]

    def view(o, cx1, cy1):
        vx, vy = max(0, x1 - 20) - cx1, max(0, y1 - 20) - cy1
        return o[vy:vy + (min(H, y2 + 20) - max(0, y1 - 20)),
                 vx:vx + (min(W, x2 + 20) - max(0, x1 - 20))]

    for name, margin, mode in [
        ("NS diffusion m=30", 30, "ns"),
        ("LaMa m=30", 30, "lama"),
        ("LaMa m=1x", max(30, bh), "lama"),
        ("LaMa m=2x", max(30, 2 * bh), "lama"),
    ]:
        cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
        cx2, cy2 = min(W, x2 + margin), min(H, y2 + margin)
        crop = img[cy1:cy2, cx1:cx2].copy()
        m = build_local_mask(it, crop.shape, (cx1, cy1), 7)
        if m is None:
            continue
        if mode == "ns":
            extra = max(6, int(round(min(m.shape[:2]) * 0.02)))
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * extra + 1, 2 * extra + 1))
            wide = cv2.dilate(m, k)
            res = cv2.inpaint(crop, wide, 20, cv2.INPAINT_NS)
            o = crop.copy()
            o[wide > 0] = res[wide > 0]
        else:
            p = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            r = cv2.cvtColor(np.array(lama(p, Image.fromarray(m).convert("L"))), cv2.COLOR_RGB2BGR)
            if r.shape[:2] != crop.shape[:2]:
                r = cv2.resize(r, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_LANCZOS4)
            o = crop.copy()
            o[m > 0] = r[m > 0]
        cells.append((name, view(o, cx1, cy1)))
        print(name, "mask", f"{np.count_nonzero(m)/m.size:.1%}")

    ch = max(c.shape[0] for _, c in cells)
    cw = max(c.shape[1] for _, c in cells)
    lab = 20
    sheet = np.full((ch + lab, len(cells) * cw, 3), 25, np.uint8)
    for i, (n, c) in enumerate(cells):
        sheet[lab:lab + c.shape[0], i * cw:i * cw + c.shape[1]] = c
        cv2.putText(sheet, n, (i * cw + 4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(a.out), sheet)
    print("->", a.out)


if __name__ == "__main__":
    main()
