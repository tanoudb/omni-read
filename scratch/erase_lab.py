# -*- coding: utf-8 -*-
"""Banc d'essai EFFACEMENT.

Phase 1 (--build) : detection + OCR + segmentation sur une image, puis mise en
cache sur disque (crop original, chirurgical_mask, bubble_mask, bbox, classe).
Phase 2 (par defaut) : recharge le cache et rejoue UNIQUEMENT l'effacement,
ce qui permet d'iterer en quelques secondes sur l'algorithme sans repasser par
YOLO/PaddleOCR/SAM.

Usage:
    python scratch/erase_lab.py --build <image> <cache_dir>
    python scratch/erase_lab.py <cache_dir> <out_dir>
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np


def build(image_path: Path, cache_dir: Path):
    import torch
    from config import config
    from core import OCREngine, SmartSegmenter, YOLODetector
    from pipeline import TranslationPipeline
    from utils import WebtoonLogger

    cache_dir.mkdir(parents=True, exist_ok=True)
    logger = WebtoonLogger("erase-lab")
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.logger = logger
    p.debug = False
    p.device = "cuda" if torch.cuda.is_available() else "cpu"
    p.detector = YOLODetector(config.YOLO_MODEL_PATH, p.device)
    p.detector_secondary = None
    p.segmenter = SmartSegmenter(logger=logger)
    p.ocr_engine = OCREngine(device=p.device)

    img = cv2.imread(str(image_path))
    dets = p._detect_ensemble(img)
    dets = [d for d in p.detector.get_translatable_detections(dets)
            if str(d.class_name).lower() != "sfx"]
    dets = p._sort_detections_reading_order(dets)
    crops = [img[d.y1:d.y2, d.x1:d.x2] for d in dets]
    results = p.ocr_engine.extract_batch(crops) if crops else []
    sibling = [(d.x1, d.y1, d.x2, d.y2) for d in dets]

    items = []
    for d, res in zip(dets, results):
        reason, _ = p._apply_ocr_result(img, d, res, sibling)
        if not reason and TranslationPipeline._is_render_noise_text(d.text_original, d.ocr_confidence):
            reason = "noise"
        if reason:
            continue
        items.append({
            "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
            "class_name": str(d.class_name),
            "text": getattr(d, "text_original", "") or "",
            "text_regions": getattr(d, "text_regions", None),
            "chirurgical_mask": getattr(d, "chirurgical_mask", None),
            "mask_binary": getattr(d, "mask_binary", None),
            "source_line_height": TranslationPipeline._measure_source_line_height(
                getattr(d, "text_regions", None)),
        })

    cv2.imwrite(str(cache_dir / "page.png"), img)
    with open(cache_dir / "dets.pkl", "wb") as f:
        pickle.dump(items, f)
    print(f"Cache ecrit: {len(items)} detections -> {cache_dir}")


def replay(cache_dir: Path, out_dir: Path, margin: int = 40):
    from core import TextRenderer

    out_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = out_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(cache_dir / "page.png"))
    with open(cache_dir / "dets.pkl", "rb") as f:
        items = pickle.load(f)

    r = TextRenderer()
    out = img.copy()
    h, w = img.shape[:2]
    for it in items:
        x1, y1, x2, y2 = it["bbox"]
        out = r.inpaint_region(
            out, x1, y1, x2, y2,
            text_regions=it["text_regions"],
            class_name=it["class_name"],
            chirurgical_mask=it["chirurgical_mask"],
            bubble_mask=it["mask_binary"],
        )

    cv2.imwrite(str(out_dir / "page_erased.png"), out)
    meta = []
    for i, it in enumerate(items):
        x1, y1, x2, y2 = it["bbox"]
        cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
        cx2, cy2 = min(w, x2 + margin), min(h, y2 + margin)
        cv2.imwrite(str(crops_dir / f"{i:02d}_orig.png"), img[cy1:cy2, cx1:cx2])
        cv2.imwrite(str(crops_dir / f"{i:02d}_erased.png"), out[cy1:cy2, cx1:cx2])
        meta.append({"index": i, "class": it["class_name"], "bbox": it["bbox"],
                     "text": it["text"][:60]})
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"{len(items)} zones effacees -> {out_dir}")
    for f in r.qcheck_flags:
        print("  flag:", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    args = ap.parse_args()
    if args.build:
        build(args.a, args.b)
    else:
        replay(args.a, args.b)
