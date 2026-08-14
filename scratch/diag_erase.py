# -*- coding: utf-8 -*-
"""Diagnostic de l'EFFACEMENT: pour chaque detection, dump du masque
chirurgical, du taux de couverture, et de la branche reellement empruntee
dans TextRenderer.inpaint_region (flat / lama / anime / diffuse).

Usage: python scratch/diag_erase.py <image> <out_dir>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch

from config import config
from core import OCREngine, TextRenderer, SmartSegmenter, YOLODetector
from core import renderer as renderer_mod
from pipeline import TranslationPipeline
from utils import WebtoonLogger

TRACE = []


def instrument(r: TextRenderer):
    """Enveloppe les methodes de decision pour tracer la branche prise."""
    orig_flat = r._flat_fill_color
    orig_lama = r._inpaint_lama
    orig_diffuse = TextRenderer._diffuse_fill
    orig_failed = TextRenderer._erasure_failed
    orig_diffusable = TextRenderer._background_is_diffusable

    def flat(crop, mask, **kw):
        res = orig_flat(crop, mask, **kw)
        TRACE[-1]["flat_fill"] = None if res is None else [int(v) for v in res]
        return res

    def lama(crop, mask):
        TRACE[-1]["lama"] = True
        return orig_lama(crop, mask)

    def diffuse(crop, mask):
        TRACE[-1].setdefault("diffuse_calls", 0)
        TRACE[-1]["diffuse_calls"] += 1
        TRACE[-1].setdefault("diffuse_px", []).append(int(np.count_nonzero(mask)))
        return orig_diffuse(crop, mask)

    def failed(crop, result, mask):
        v = orig_failed(crop, result, mask)
        TRACE[-1].setdefault("erasure_failed", []).append(bool(v))
        return v

    def diffusable(crop, mask, margin=25, class_name=""):
        v = orig_diffusable(crop, mask, margin, class_name)
        TRACE[-1].setdefault("diffusable", []).append(bool(v))
        return v

    r._flat_fill_color = flat
    r._inpaint_lama = lama
    TextRenderer._diffuse_fill = staticmethod(diffuse)
    TextRenderer._erasure_failed = staticmethod(failed)
    TextRenderer._background_is_diffusable = staticmethod(diffusable)


def build_pipeline():
    logger = WebtoonLogger("diag-erase")
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.logger = logger
    p.debug = False
    p.device = "cuda" if torch.cuda.is_available() else "cpu"
    p.detector = YOLODetector(config.YOLO_MODEL_PATH, p.device)
    p.detector_secondary = None
    p.segmenter = SmartSegmenter(logger=logger)
    p.ocr_engine = OCREngine(device=p.device)
    return p


def run(image_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = out_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    p = build_pipeline()
    r = TextRenderer()
    instrument(r)

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    dets = p._detect_ensemble(img)
    dets = [d for d in p.detector.get_translatable_detections(dets)
            if str(d.class_name).lower() != "sfx"]
    dets = p._sort_detections_reading_order(dets)

    crops = [img[d.y1:d.y2, d.x1:d.x2] for d in dets]
    results = p.ocr_engine.extract_batch(crops) if crops else []
    sibling = [(d.x1, d.y1, d.x2, d.y2) for d in dets]
    keep = []
    for d, res in zip(dets, results):
        reason, _ = p._apply_ocr_result(img, d, res, sibling)
        if not reason and TranslationPipeline._is_render_noise_text(d.text_original, d.ocr_confidence):
            reason = "noise"
        if not reason:
            keep.append(d)

    out = img.copy()
    report = []
    for i, d in enumerate(keep):
        TRACE.append({"index": i, "class": str(d.class_name)})
        cm = getattr(d, "chirurgical_mask", None)
        det_area = max(1, (d.y2 - d.y1) * (d.x2 - d.x1))
        if cm is not None:
            cov = float(np.count_nonzero(cm)) / det_area
            cv2.imwrite(str(masks_dir / f"{i:02d}_mask.png"), cm)
            # visualisation masque en rouge sur le crop
            crop = img[d.y1:d.y2, d.x1:d.x2].copy()
            m = cm
            if m.shape[:2] != crop.shape[:2]:
                m = cv2.resize(m, (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST)
            ov = crop.copy()
            ov[m > 0] = (0, 0, 255)
            cv2.imwrite(str(masks_dir / f"{i:02d}_overlay.png"),
                        cv2.addWeighted(crop, 0.45, ov, 0.55, 0))
        else:
            cov = 0.0
        TRACE[-1]["mask_coverage"] = round(cov, 4)
        TRACE[-1]["bbox"] = [int(d.x1), int(d.y1), int(d.x2), int(d.y2)]
        TRACE[-1]["ocr"] = (getattr(d, "text_original", "") or "")[:60]
        TRACE[-1]["n_regions"] = len(getattr(d, "text_regions", None) or [])

        out = r.inpaint_region(
            out, d.x1, d.y1, d.x2, d.y2,
            text_regions=getattr(d, "text_regions", None) or getattr(d, "mask_regions", None),
            class_name=str(d.class_name),
            chirurgical_mask=cm,
            bubble_mask=getattr(d, "mask_binary", None),
        )
        report.append(TRACE[-1])

    cv2.imwrite(str(out_dir / "page_erased.png"), out)
    for i, d in enumerate(keep):
        m = 40
        x1, y1 = max(0, d.x1 - m), max(0, d.y1 - m)
        x2, y2 = min(w, d.x2 + m), min(h, d.y2 + m)
        cv2.imwrite(str(masks_dir / f"{i:02d}_erased.png"), out[y1:y2, x1:x2])
        cv2.imwrite(str(masks_dir / f"{i:02d}_orig.png"), img[y1:y2, x1:x2])

    (out_dir / "trace.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for t in report:
        print(json.dumps(t, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    run(a.image, a.out_dir)
