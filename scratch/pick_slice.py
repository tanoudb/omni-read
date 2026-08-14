# -*- coding: utf-8 -*-
"""Choisit une tranche representative (densite de bulles + variete de tailles)
dans un strip webtoon tres haut, et l'ecrit sur disque.

Usage: python scratch/pick_slice.py <strip.jpg> <out.png> [--height 6000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from config import config
from core import YOLODetector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strip", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--height", type=int, default=6000)
    args = ap.parse_args()

    img = cv2.imread(str(args.strip))
    h, w = img.shape[:2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    det = YOLODetector(config.YOLO_MODEL_PATH, device)
    dets = det.detect(img)
    dets = det.get_translatable_detections(dets)
    dets = [d for d in dets if str(d.class_name).lower() != "sfx"]
    print(f"{len(dets)} zones traduisibles sur {w}x{h}")

    best = (None, -1e9)
    step = 500
    for y in range(0, max(1, h - args.height), step):
        y2 = y + args.height
        inside = [d for d in dets if d.y1 >= y and d.y2 <= y2]
        if len(inside) < 3:
            continue
        areas = np.array([(d.x2 - d.x1) * (d.y2 - d.y1) for d in inside], dtype=float)
        heights = np.array([d.y2 - d.y1 for d in inside], dtype=float)
        variety = float(np.std(heights) / (np.mean(heights) + 1e-6))
        score = len(inside) + 6.0 * variety
        if score > best[1]:
            best = ((y, y2, len(inside), variety), score)

    if best[0] is None:
        raise SystemExit("Aucune fenetre trouvee")
    y, y2, n, variety = best[0]
    print(f"Fenetre choisie: y={y}..{y2}  bulles={n}  variete={variety:.2f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), img[y:y2])
    print(f"Ecrit {args.out}")


if __name__ == "__main__":
    main()
