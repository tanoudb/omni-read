# -*- coding: utf-8 -*-
"""Diagnostic cible: bulles de cri dentelees qui se chevauchent
(POV_V2 #05 "TOO SLOW, KAZUKI!" / #06 "IF YOU LEAVE YOURSELF OPEN...").

Enveloppe TextRenderer._bubble_shape_mask, _get_inner_zone,
_shrink_zone_away_from_siblings, _mask_row_span/_mask_row_center pour
imprimer les valeurs REELLES prises a chaque etape, sur la tranche isolee
scratch/render_out/scream_slice.png.

Usage: python scratch/diag_scream.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import numpy as np
import torch

from config import config
from core import OCREngine, TextRenderer, SmartSegmenter, YOLODetector
from pipeline import TranslationPipeline
from utils import WebtoonLogger

IMG_PATH = Path(r"A:\omni read\scratch\render_out\scream_slice.png")
OUT_DIR = Path(r"A:\omni read\scratch\render_out\scream_diag")

CALLS = []


def instrument(r: TextRenderer):
    orig_shape_mask = r._bubble_shape_mask
    orig_inner_zone = r._get_inner_zone
    orig_shrink_siblings = TextRenderer._shrink_zone_away_from_siblings
    orig_row_span = TextRenderer._mask_row_span
    orig_row_center = TextRenderer._mask_row_center

    def shape_mask(raw_mask, crop_bgr, box_w, box_h, is_bubble):
        res = orig_shape_mask(raw_mask, crop_bgr, box_w, box_h, is_bubble)
        rec = {
            "call": "_bubble_shape_mask",
            "raw_mask_is_none": raw_mask is None,
            "raw_mask_shape": None if raw_mask is None else list(np.asarray(raw_mask).shape),
            "raw_mask_nnz": None if raw_mask is None else int(np.count_nonzero(raw_mask)),
            "box_w": box_w, "box_h": box_h, "is_bubble": is_bubble,
            "result_is_none": res is None,
        }
        if res is not None:
            nnz = int(np.count_nonzero(res))
            rec["result_shape"] = list(res.shape)
            rec["result_nnz"] = nnz
            rec["result_coverage_of_bbox"] = round(nnz / float(box_w * box_h), 4)
            ys, xs = np.nonzero(res)
            if xs.size:
                rec["result_xrange"] = [int(xs.min()), int(xs.max())]
                rec["result_yrange"] = [int(ys.min()), int(ys.max())]
                rec["result_centroid_xy"] = [round(float(xs.mean()), 1), round(float(ys.mean()), 1)]
                rec["bbox_center_xy"] = [box_w / 2.0, box_h / 2.0]
        CALLS.append(rec)
        return res

    def inner_zone(x1, y1, x2, y2, img_shape, bubble_mask=None, shrink=None):
        res = orig_inner_zone(x1, y1, x2, y2, img_shape, bubble_mask=bubble_mask, shrink=shrink)
        rec = {
            "call": "_get_inner_zone",
            "bbox_in": [x1, y1, x2, y2],
            "shrink": shrink,
            "bubble_mask_given": bubble_mask is not None,
            "zone_out": list(res),
        }
        bw, bh = x2 - x1, y2 - y1
        zw, zh = res[2] - res[0], res[3] - res[1]
        rec["bbox_wh"] = [bw, bh]
        rec["zone_wh"] = [zw, zh]
        rec["zone_center"] = [(res[0] + res[2]) / 2.0, (res[1] + res[3]) / 2.0]
        rec["bbox_center"] = [x1 + bw / 2.0, y1 + bh / 2.0]
        CALLS.append(rec)
        return res

    def shrink_siblings(zone, sibling_boxes):
        res = orig_shrink_siblings(zone, sibling_boxes)
        rec = {
            "call": "_shrink_zone_away_from_siblings",
            "zone_in": list(zone),
            "siblings": [list(b) for b in (sibling_boxes or [])],
            "zone_out": list(res),
        }
        zw_in, zh_in = zone[2] - zone[0], zone[3] - zone[1]
        zw_out, zh_out = res[2] - res[0], res[3] - res[1]
        rec["area_in"] = zw_in * zh_in
        rec["area_out"] = zw_out * zh_out
        rec["area_kept_ratio"] = round((zw_out * zh_out) / max(1, zw_in * zh_in), 4)
        CALLS.append(rec)
        return res

    row_span_calls = {"n": 0, "vals": []}

    def row_span(mask, y0, y1):
        v = orig_row_span(mask, y0, y1)
        row_span_calls["n"] += 1
        row_span_calls["vals"].append((round(y0, 1), round(y1, 1), round(v, 1)))
        return v

    r._bubble_shape_mask = shape_mask
    r._get_inner_zone = inner_zone
    TextRenderer._shrink_zone_away_from_siblings = staticmethod(shrink_siblings)
    TextRenderer._mask_row_span = staticmethod(row_span)

    return row_span_calls


def build_pipeline() -> TranslationPipeline:
    logger = WebtoonLogger("diag-scream")
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.logger = logger
    p.debug = False
    p.device = "cuda" if torch.cuda.is_available() else "cpu"
    p.detector = YOLODetector(config.YOLO_MODEL_PATH, p.device)
    p.detector_secondary = None
    p.segmenter = SmartSegmenter(logger=logger)
    p.ocr_engine = OCREngine(device=p.device)
    return p


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = build_pipeline()
    renderer = TextRenderer()
    row_span_calls = instrument(renderer)

    img = cv2.imread(str(IMG_PATH))
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}px")

    dets = p._detect_ensemble(img)
    dets = [d for d in p.detector.get_translatable_detections(dets)
            if str(d.class_name).lower() != "sfx"]
    dets = p._sort_detections_reading_order(dets)
    print(f"Detections: {len(dets)}")
    for d in dets:
        print("  bbox", d.x1, d.y1, d.x2, d.y2, "class", d.class_name, "score", d.score)

    crops = [img[d.y1:d.y2, d.x1:d.x2] for d in dets]
    results = p.ocr_engine.extract_batch(crops) if crops else []
    sibling_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]
    keep = []
    for d, res in zip(dets, results):
        reason, _ = p._apply_ocr_result(img, d, res, sibling_boxes)
        if not reason and TranslationPipeline._is_render_noise_text(d.text_original, d.ocr_confidence):
            reason = "noise"
        if not reason:
            keep.append(d)
    print(f"OCR valides: {len(keep)}")

    out, _, ghost_idx = p._run_pre_inpainting(img, keep, renderer)

    for i, d in enumerate(keep):
        print(f"\n=== BULLE {i}: {d.text_original!r} bbox=({d.x1},{d.y1},{d.x2},{d.y2}) mask_binary={'yes' if getattr(d,'mask_binary',None) is not None else 'no'} ===")
        d.text_translated = str(d.text_original)
        p._prepare_render_style(renderer, img, d)
        print(f"  source_line_height={getattr(d,'source_line_height',None)} text_style={getattr(d,'text_style',None)} font_hint={getattr(d,'font_hint',None)}")
        n_calls_before = len(CALLS)
        row_span_calls["vals"].clear()
        out = renderer.insert_text(
            out,
            d.text_translated,
            int(d.x1), int(d.y1), int(d.x2), int(d.y2),
            text_regions=getattr(d, 'text_regions', None),
            text_color_rgb=getattr(d, 'text_color_rgb', None),
            text_style=getattr(d, 'text_style', 'dialogue'),
            font_hint=getattr(d, 'font_hint', 'regular'),
            class_name=str(d.class_name),
            bubble_mask=getattr(d, 'mask_binary', None),
            font_key=getattr(d, 'font_key', None),
            source_line_height=getattr(d, 'source_line_height', None),
            sibling_boxes=[(o.x1, o.y1, o.x2, o.y2) for o in keep if o is not d],
        )
        for rec in CALLS[n_calls_before:]:
            print("  ", json.dumps(rec, ensure_ascii=False))
        print(f"  row_span samples (n={row_span_calls['n']}): {row_span_calls['vals'][:12]}")
        dbg = renderer.last_layout_debug
        if dbg:
            print("  last_layout_debug:")
            for k, v in dbg.items():
                if k == "lines":
                    continue
                print(f"    {k} = {v}")
            print(f"    lines = {dbg.get('lines')}")

    cv2.imwrite(str(OUT_DIR / "page_after.png"), out)
    print(f"\nOK -> {OUT_DIR / 'page_after.png'}")


if __name__ == "__main__":
    run()
