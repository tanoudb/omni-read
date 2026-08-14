# -*- coding: utf-8 -*-
"""
Instrumentation du pipeline de rendu (SANS traduction, texte source réinjecté
tel quel) pour QUANTIFIER les défauts de placement/dimensionnement de
`TextRenderer.insert_text`.

Réutilise exactement le montage de `scratch/render_iterate.py` (détection +
OCR + pré-inpainting identiques) puis, au lieu de se contenter d'écrire les
images, lit `renderer.last_layout_debug` (hook additif posé dans
`core/renderer.py::insert_text`) après CHAQUE appel pour produire un tableau
chiffré par détection :

  - hauteur de ligne SOURCE (médiane OCR, `_measure_source_line_height`)
  - taille de police effectivement choisie, nb de lignes, interligne (px)
  - centre du bloc rendu vs centre de la zone utilisable -> dx, dy (px)
  - taux de remplissage (aire encre / aire utilisable)

Usage:
    python scratch/measure_layout.py <image.jpg> [--out scratch/layout_measurements.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
from config import config
from core import OCREngine, TextRenderer, SmartSegmenter, YOLODetector
from pipeline import TranslationPipeline
from utils import WebtoonLogger


def build_pipeline() -> TranslationPipeline:
    logger = WebtoonLogger("measure-layout")
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.logger = logger
    p.debug = False
    import torch
    p.device = "cuda" if torch.cuda.is_available() else "cpu"
    p.detector = YOLODetector(config.YOLO_MODEL_PATH, p.device)
    p.detector_secondary = None
    sec = getattr(config, "YOLO_MODEL_PATH_SECONDARY", None)
    if sec and Path(sec).exists():
        p.detector_secondary = YOLODetector(sec, p.device)
    p.segmenter = SmartSegmenter(logger=logger)
    p.ocr_engine = OCREngine(device=p.device)
    return p


def run(image_path: Path, out_json: Path):
    p = build_pipeline()
    renderer = TextRenderer()

    img = cv2.imread(str(image_path))
    if img is None:
        raise SystemExit(f"Impossible de charger {image_path}")
    h, w = img.shape[:2]
    print(f"Image: {w}x{h}px")

    # ── Détection ──
    max_h = int(getattr(config.detection, "max_height", 0) or 0)
    det_img, scale = img, 1.0
    if max_h > 0 and h > max_h:
        scale = h / float(max_h)
        det_img = cv2.resize(img, (max(1, int(w / scale)), max_h), interpolation=cv2.INTER_AREA)
    dets = p._detect_ensemble(det_img)
    if scale != 1.0:
        for d in dets:
            d.bbox = [float(v * scale) for v in d.bbox]
    dets = [d for d in p.detector.get_translatable_detections(dets)
            if str(d.class_name).lower() != "sfx"]
    dets = p._sort_detections_reading_order(dets)
    print(f"Détections traduisibles: {len(dets)}")

    # ── OCR + masques ──
    crops, idxs = [], []
    for i, d in enumerate(dets):
        c = img[d.y1:d.y2, d.x1:d.x2]
        crops.append(c)
        idxs.append(i)

    results = p.ocr_engine.extract_batch(crops) if crops else []
    sibling_boxes = [(d.x1, d.y1, d.x2, d.y2) for d in dets]
    keep = []
    for i, res in zip(idxs, results):
        reason, _ = p._apply_ocr_result(img, dets[i], res, sibling_boxes)
        if not reason and TranslationPipeline._is_render_noise_text(dets[i].text_original, dets[i].ocr_confidence):
            reason = "render_noise_or_watermark"
        if not reason:
            keep.append(dets[i])
    print(f"Textes OCR valides: {len(keep)}")

    # ── Effacement (inpainting) ── (identique à render_iterate.py)
    out, _, ghost_idx = p._run_pre_inpainting(img, keep, renderer)

    # ── Réinjection texte source + capture du hook de mesure ──
    rows = []
    for i, d in enumerate(keep):
        d.text_translated = str(d.text_original)
        p._prepare_render_style(renderer, img, d)
        renderer.last_layout_debug = None
        try:
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
        except Exception as e:
            print(f"[{i}] Exception rendu: {e}")

        dbg = renderer.last_layout_debug
        row = {
            "index": i,
            "class": str(d.class_name),
            "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
            "ocr_text": getattr(d, 'text_original', '') or '',
            "source_line_height": getattr(d, 'source_line_height', None),
        }
        if dbg is None:
            row["debug"] = "NO_HOOK_CAPTURED (autre chemin de rendu, ex: _draw_exact_lines hors-bulle)"
        else:
            row.update({
                "render_bail": dbg.get("bail"),
                "is_bubble": dbg.get("is_bubble"),
                "has_mask_wrap": dbg.get("has_mask_wrap"),
                "container": dbg.get("container"),
                "use_locked_mode": dbg.get("use_locked_mode"),
                "angle_deg": dbg.get("angle"),
                "usable_zone": dbg.get("usable_zone"),
                "inner_w": dbg.get("inner_w"),
                "inner_h": dbg.get("inner_h"),
                "fs_estimate": dbg.get("fs_estimate"),
                "font_size_final": dbg.get("font_size_final"),
                "n_lines": dbg.get("n_lines"),
                "lines": dbg.get("lines"),
                "line_h_px": dbg.get("line_h_px"),
                "spacing_px": dbg.get("spacing_px"),
                "total_h_px": dbg.get("total_h_px"),
                "zone_center": dbg.get("zone_center"),
                "block_center": dbg.get("block_center"),
                "dx_px": dbg.get("dx"),
                "dy_px": dbg.get("dy"),
                "fill_ratio": dbg.get("fill_ratio"),
            })
        rows.append(row)

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"OK: {len(rows)} lignes de mesure -> {out_json}")

    # ── Tableau condensé lisible en console ──
    header = f"{'idx':>3} {'cls':<7} {'src_lh':>7} {'fs':>4} {'nL':>3} {'line_h':>7} {'sp':>4} {'dx':>7} {'dy':>7} {'fill%':>6}  text"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("debug"):
            print(f"{r['index']:>3} {r['class']:<7} {'--':>7} {'--':>4} {'--':>3} {'--':>7} {'--':>4} {'--':>7} {'--':>7} {'--':>6}  {r['ocr_text'][:30]!r} [{r['debug']}]")
            continue
        src_lh = r.get("source_line_height")
        src_lh_s = f"{src_lh:.1f}" if isinstance(src_lh, (int, float)) else "None"
        dx = r.get("dx_px"); dy = r.get("dy_px"); fr = r.get("fill_ratio")
        dx_s = f"{dx:+.1f}" if isinstance(dx, (int, float)) else "--"
        dy_s = f"{dy:+.1f}" if isinstance(dy, (int, float)) else "--"
        fr_s = f"{fr*100:.1f}" if isinstance(fr, (int, float)) else "--"
        print(
            f"{r['index']:>3} {r['class']:<7} {src_lh_s:>7} "
            f"{str(r.get('font_size_final')):>4} {str(r.get('n_lines')):>3} "
            f"{str(r.get('line_h_px')):>7} {str(r.get('spacing_px')):>4} "
            f"{dx_s:>7} {dy_s:>7} {fr_s:>6}  {r['ocr_text'][:30]!r}"
        )

    del img, det_img, dets, keep, out, crops, idxs
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=Path("scratch/layout_measurements.json"))
    args = ap.parse_args()
    run(args.image, args.out)
