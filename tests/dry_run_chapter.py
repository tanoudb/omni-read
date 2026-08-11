"""
Passe à blanc sur une planche entière — SANS appel API.

Le texte OCRisé est réinjecté tel quel à la place de la traduction. Tout le
reste du pipeline tourne pour de vrai : détection, dédoublonnage, OCR,
segmentation, effacement, mise en page, rendu.

But : exercer le pipeline sur ~120 zones au lieu des quelques-unes qu'on
inspecte à la main, et sortir des chiffres comparables d'une version à l'autre
(utile pour arbitrer v4 seul vs ensemble v3+v4).

    python tests/dry_run_chapter.py [chemin_image] [--limit N]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from config import config
from core import OCREngine, TextRenderer, SmartSegmenter, YOLODetector
from pipeline import TranslationPipeline
from utils import ImageUtils, WebtoonLogger

DEFAULT_IMAGE = (
    "manhwa/rise_of_the_dragon_overlord/Chapitre 001/Chapitre 001_merged_part01.jpg"
)


def build_pipeline() -> TranslationPipeline:
    logger = WebtoonLogger("dry-run")
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.logger = logger
    p.debug = False
    p.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    p.detector = YOLODetector(config.YOLO_MODEL_PATH, p.device)
    p.detector_secondary = None
    sec = getattr(config, "YOLO_MODEL_PATH_SECONDARY", None)
    if sec and Path(sec).exists():
        p.detector_secondary = YOLODetector(sec, p.device)
    p.segmenter = SmartSegmenter(logger=logger)
    p.ocr_engine = OCREngine(device=p.device)
    return p


def erase_residue(after_crop: np.ndarray, mask: np.ndarray) -> float:
    """% de pixels encore écartés du fond, dans la zone qui portait du texte."""
    if mask is None or not mask.any():
        return 0.0
    zone = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
    if not zone.any():
        return 0.0
    gray = cv2.cvtColor(after_crop, cv2.COLOR_BGR2GRAY).astype(np.int16)
    bg = float(np.median(gray[zone]))
    return float(np.mean(np.abs(gray[zone] - bg) > 10)) * 100


def overlapping_pairs(dets, iou_min: float = 0.30) -> int:
    """Zones qui se recouvrent encore après dédoublonnage = doublons de rendu."""
    n = 0
    for i in range(len(dets)):
        for j in range(i + 1, len(dets)):
            a, b = dets[i], dets[j]
            iw = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
            ih = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
            inter = iw * ih
            if inter <= 0:
                continue
            union = ((a.x2 - a.x1) * (a.y2 - a.y1)
                     + (b.x2 - b.x1) * (b.y2 - b.y1) - inter)
            if union > 0 and inter / union >= iou_min:
                n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", default=DEFAULT_IMAGE)
    ap.add_argument("--limit", type=int, default=0, help="n'traiter que les N premières zones")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "dry_run_out"))
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"introuvable: {img_path}")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = build_pipeline()
    renderer = TextRenderer()
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"illisible: {img_path}")
        return 1
    h, w = img.shape[:2]
    print(f"\nPlanche : {img_path.name}  ({w}x{h})")
    print(f"Modèles : {Path(config.YOLO_MODEL_PATH).name}"
          + (f" + {Path(config.YOLO_MODEL_PATH_SECONDARY).name}"
             if p.detector_secondary else " (mono-modèle)"))

    # ── Détection ──
    t0 = time.time()
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
    if args.limit:
        dets = dets[: args.limit]
    t_det = time.time() - t0

    by_class: dict = {}
    for d in dets:
        by_class[d.class_name] = by_class.get(d.class_name, 0) + 1
    print(f"\nDétection      : {len(dets)} zones en {t_det:.1f}s  {by_class}")
    print(f"  recouvrements restants (doublons de rendu) : {overlapping_pairs(dets)}")

    # ── OCR + masques ──
    t0 = time.time()
    crops, idxs = [], []
    for i, d in enumerate(dets):
        c = img[d.y1:d.y2, d.x1:d.x2]
        if c.size:
            crops.append(c)
            idxs.append(i)
    results = p.ocr_engine.extract_batch(crops) if crops else []
    skips: dict = {}
    for i, res in zip(idxs, results):
        reason, _ = p._apply_ocr_result(img, dets[i], res)
        if reason:
            skips[reason] = skips.get(reason, 0) + 1
    keep = [d for d in dets if getattr(d, "text_original", "")]
    t_ocr = time.time() - t0
    print(f"OCR            : {len(keep)}/{len(dets)} textes en {t_ocr:.1f}s   ignorés: {skips}")

    glued = [d.text_original for d in keep
             if any(len(tok) > 14 for tok in str(d.text_original).split())]
    print(f"  textes gardant un mot de +14 car. (collage non résolu) : {len(glued)}")

    # ── Effacement ──
    t0 = time.time()
    out, _ = p._run_pre_inpainting(img, keep, renderer)
    t_erase = time.time() - t0
    residues = []
    for d in keep:
        if getattr(d, "chirurgical_mask", None) is None:
            continue
        residues.append((erase_residue(out[d.y1:d.y2, d.x1:d.x2], d.chirurgical_mask), d))
    residues.sort(key=lambda t: -t[0])
    vals = [r for r, _ in residues]
    if vals:
        print(f"Effacement     : {t_erase:.1f}s   résidu médian {np.median(vals):.2f}% "
              f"| p90 {np.percentile(vals, 90):.2f}% | max {max(vals):.2f}%")
        print(f"  zones à résidu > 2% : {sum(1 for v in vals if v > 2)}")

    # ── Rendu (texte source réinjecté) ──
    t0 = time.time()
    floored = 0
    min_fs = int(config.rendering.min_font_size)
    real_fit = renderer._fit_font_hard

    def counting_fit(*a, **kw):
        nonlocal floored
        layout = real_fit(*a, **kw)
        if layout is not None and layout["size"] <= min_fs:
            floored += 1
        return layout

    renderer._fit_font_hard = counting_fit
    for d in keep:
        d.text_translated = d.text_original
        p._prepare_render_style(renderer, img, d)
        out = renderer.insert_text(
            out, d.text_translated, d.x1, d.y1, d.x2, d.y2,
            text_regions=d.text_regions,
            text_color_rgb=getattr(d, "text_color_rgb", None),
            text_style=getattr(d, "text_style", "dialogue"),
            font_hint=getattr(d, "font_hint", "regular"),
            class_name=d.class_name,
            bubble_mask=getattr(d, "mask_binary", None),
            font_key=getattr(d, "font_key", None),
            source_line_height=getattr(d, "source_line_height", None),
        )
    renderer._fit_font_hard = real_fit
    t_render = time.time() - t0
    print(f"Rendu          : {t_render:.1f}s   zones au plancher de police ({min_fs}px) : "
          f"{floored}/{len(keep)}")

    # ── Sorties ──
    p._write_output_image(out_dir, img_path.stem + "_dryrun", out)
    worst = residues[:6]
    if worst:
        tiles = []
        for _, d in worst:
            a = img[d.y1:d.y2, d.x1:d.x2]
            b = out[d.y1:d.y2, d.x1:d.x2]
            tile = np.hstack([a, np.zeros((a.shape[0], 4, 3), np.uint8), b])
            tiles.append(cv2.resize(tile, (600, max(1, int(600 * tile.shape[0] / tile.shape[1])))))
        height = sum(t.shape[0] + 6 for t in tiles)
        sheet = np.full((height, 600, 3), 30, np.uint8)
        y = 0
        for t in tiles:
            sheet[y:y + t.shape[0]] = t
            y += t.shape[0] + 6
        cv2.imwrite(str(out_dir / "pires_effacements.png"), sheet)

    print(f"\nSorties dans {out_dir}")
    print(f"Total : {t_det + t_ocr + t_erase + t_render:.1f}s pour {len(keep)} zones")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
