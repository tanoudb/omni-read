# -*- coding: utf-8 -*-
"""Genere des vues AVANT | APRES decoupees en tuiles lisibles.

Usage: python scratch/side_by_side.py <run_dir> [--tile 2000] [--scale 1.0]
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--tile", type=int, default=2000)
    ap.add_argument("--scale", type=float, default=1.0)
    args = ap.parse_args()

    before = cv2.imread(str(args.run_dir / "page_before.png"))
    after = cv2.imread(str(args.run_dir / "page_after.png"))
    h, w = before.shape[:2]
    out_dir = args.run_dir / "side_by_side"
    out_dir.mkdir(parents=True, exist_ok=True)

    sep = np.full((args.tile, 8, 3), (30, 30, 30), dtype=np.uint8)
    n = 0
    for y in range(0, h, args.tile):
        y2 = min(h, y + args.tile)
        b = before[y:y2]
        a = after[y:y2]
        s = sep[: y2 - y]
        pair = np.hstack([b, s, a])
        if args.scale != 1.0:
            pair = cv2.resize(pair, None, fx=args.scale, fy=args.scale,
                              interpolation=cv2.INTER_AREA)
        p = out_dir / f"pair_{n:02d}_y{y}.png"
        cv2.imwrite(str(p), pair)
        print(p)
        n += 1


if __name__ == "__main__":
    main()
