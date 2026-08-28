# -*- coding: utf-8 -*-
"""
BARÈME DE RENDU — note chiffrée de la qualité de rendu, sur corpus multi-séries.

Pourquoi
────────
Les 24 correctifs de `RENDU_CHANGELOG.md` ont été validés à l'œil, une série à
la fois. Rien ne garantit qu'un correctif de la série 2 n'a pas défait un
correctif de la série 1, et « professionnel quel que soit le manhwa » n'est pas
vérifiable sans un chiffre. Ce module produit ce chiffre.

Principe
────────
Comme `render_iterate.py`, la traduction est DÉSACTIVÉE : on réinjecte le texte
OCR d'origine. Le texte rendu est donc la MÊME chaîne que le texte source, dans
la MÊME bulle. Toute différence mesurée entre l'encre source et l'encre rendue
est imputable au rendu, jamais à la traduction. La planche d'origine sert de
référence : c'est le geste du letterer du studio qu'on cherche à retrouver.

Trois images par planche :
    before  — l'originale
    erased  — après effacement, AVANT réinjection du texte
    after   — le rendu final
Les métriques d'effacement se lisent sur (before, erased), celles de mise en
page sur (before, erased, after) + `renderer.last_layout_debug`.

Usage
─────
    python scratch/bareme.py build                 # cache détection+OCR (lent, 1 fois)
    python scratch/bareme.py score --run baseline  # rejoue effacement+rendu, note
    python scratch/bareme.py report --run baseline
    python scratch/bareme.py compare baseline apres-rag

`build` est le seul passage par YOLO/PaddleOCR. `score` repart du cache : une
modification de `core/renderer.py` se re-note sans repasser par la détection.
"""
import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import cv2
import numpy as np

# Corpus d'images : hors dépôt (gitignoré), lu en LECTURE SEULE.
CORPUS_ROOT = Path(r"A:\omni read\manhwa")
CACHE_ROOT = REPO / "scratch" / "bareme" / "cache"
RUNS_ROOT = REPO / "scratch" / "bareme" / "runs"

# Seuils de conformité. Volontairement dans le code et non dans un YAML : ils se
# calibrent sur la première mesure et se relisent avec les résultats.
SEUILS = {
    "erase_spill_pct":      0.30,  # % de l'aire bbox repeinte hors encre source
    "ghost_contrast":       12.0,  # niveaux de gris entre zone effacée et voisinage
    "residu_pct":           12.0,  # % de l'encre source encore visible après effacement
    "footprint_ratio_min":  0.60,  # boîte de l'encre rendue / boîte de l'encre source
    "footprint_ratio_max":  1.60,
    "cap_ratio_min":        0.80,  # corps rendu / corps source, mesure en pixels
    "cap_ratio_max":        1.25,
    "centroid_dx_norm":     0.06,  # décentrage, en fraction de la dimension bbox
    "centroid_dy_norm":     0.06,
    "bubble_overflow_pct":  3.0,   # % d'encre rendue hors du ballon
    "n_lines_delta_max":    1,     # lignes rendues - lignes source
    "orphelin_compte":      True,  # dernière ligne réduite à un mot court
}


# ─────────────────────────────────────────────────────────────────────────────
# CORPUS
# ─────────────────────────────────────────────────────────────────────────────

# Planches retenues par série. La `part01` porte le titre et les crédits, donc
# elle sur-représente les cartouches `out_text` ; la `part02` est du corps de
# chapitre, avec le dialogue dense sous-représenté sans elle.
PLANCHES = ("_merged_part01", "_merged_part02")


def discover_corpus(root: Path = CORPUS_ROOT):
    """Les planches de `PLANCHES` dans le premier chapitre de chaque série.

    Rend des triplets (série, étiquette de planche, chemin). Le choix est
    déterministe : deux corpus construits à deux moments contiennent les mêmes
    images, donc deux runs restent comparables.
    """
    pages = []
    if not root.exists():
        return pages
    for series_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        chapters = sorted(p for p in series_dir.iterdir() if p.is_dir())
        if not chapters:
            continue
        trouve = False
        for i, motif in enumerate(PLANCHES):
            imgs = sorted(chapters[0].glob("*%s.*" % motif))
            if imgs:
                pages.append((series_dir.name, "p%02d" % (i + 1), imgs[0]))
                trouve = True
        if not trouve:
            imgs = sorted(q for q in chapters[0].iterdir()
                          if q.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
            if imgs:
                pages.append((series_dir.name, "p01", imgs[0]))
    return pages


def _cache_dir(series, page):
    """Un répertoire de cache par PLANCHE : les index de bulle sont locaux à une
    planche, les mélanger ferait collisionner deux bulles différentes."""
    return CACHE_ROOT / ("%s__%s" % (_slug(series), page))


def _slug(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in name).strip("-").lower()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — CACHE (YOLO + PaddleOCR, lent)
# ─────────────────────────────────────────────────────────────────────────────

def build(only=None, force=False):
    import torch
    from config import config
    from core import OCREngine, SmartSegmenter, YOLODetector
    from pipeline import TranslationPipeline
    from utils import WebtoonLogger

    pages = discover_corpus()
    if only:
        pages = [x for x in pages if only.lower() in x[0].lower()]
    if not pages:
        raise SystemExit("Aucune planche trouvee sous %s" % CORPUS_ROOT)

    todo = [x for x in pages
            if force or not (_cache_dir(x[0], x[1]) / "dets.pkl").exists()]
    print("%d planches au corpus, %d a construire" % (len(pages), len(todo)))
    if not todo:
        return

    logger = WebtoonLogger("bareme-build")
    p = TranslationPipeline.__new__(TranslationPipeline)
    p.logger = logger
    p.debug = False
    p.device = "cuda" if torch.cuda.is_available() else "cpu"
    p.detector = YOLODetector(config.YOLO_MODEL_PATH, p.device)
    p.detector_secondary = None
    sec = getattr(config, "YOLO_MODEL_PATH_SECONDARY", None)
    if sec and Path(sec).exists():
        p.detector_secondary = YOLODetector(sec, p.device)
    p.segmenter = SmartSegmenter(logger=logger)
    p.ocr_engine = OCREngine(device=p.device)

    for series, page, img_path in todo:
        t0 = time.perf_counter()
        cache_dir = _cache_dir(series, page)
        cache_dir.mkdir(parents=True, exist_ok=True)

        img = cv2.imread(str(img_path))
        if img is None:
            print("  !! illisible: %s" % img_path)
            continue
        h, w = img.shape[:2]

        # Detection sur image reduite (les planches font ~45 000 px de haut).
        max_h = int(getattr(config.detection, "max_height", 0) or 0)
        det_img, scale = img, 1.0
        if max_h > 0 and h > max_h:
            scale = h / float(max_h)
            det_img = cv2.resize(img, (max(1, int(w / scale)), max_h),
                                 interpolation=cv2.INTER_AREA)
        dets = p._detect_ensemble(det_img)
        if scale != 1.0:
            for d in dets:
                d.bbox = [float(v * scale) for v in d.bbox]
        dets = [d for d in p.detector.get_translatable_detections(dets)
                if str(d.class_name).lower() != "sfx"]
        dets = p._sort_detections_reading_order(dets)

        crops = [img[d.y1:d.y2, d.x1:d.x2] for d in dets]
        results = p.ocr_engine.extract_batch(crops) if crops else []
        sibling = [(d.x1, d.y1, d.x2, d.y2) for d in dets]

        items = []
        for d, res in zip(dets, results):
            reason, _ = p._apply_ocr_result(img, d, res, sibling)
            if not reason and TranslationPipeline._is_render_noise_text(
                    d.text_original, d.ocr_confidence):
                reason = "render_noise_or_watermark"
            if reason:
                continue
            items.append({
                "bbox": [int(d.x1), int(d.y1), int(d.x2), int(d.y2)],
                "class_name": str(d.class_name),
                "score": float(getattr(d, "score", 0.0)),
                "text": getattr(d, "text_original", "") or "",
                "ocr_confidence": float(getattr(d, "ocr_confidence", 0.0)),
                "text_regions": getattr(d, "text_regions", None),
                "mask_regions": getattr(d, "mask_regions", None),
                "chirurgical_mask": getattr(d, "chirurgical_mask", None),
                "mask_binary": getattr(d, "mask_binary", None),
            })

        cv2.imwrite(str(cache_dir / "page.png"), img)
        with open(cache_dir / "dets.pkl", "wb") as f:
            pickle.dump({"series": series, "page": page, "image": str(img_path),
                         "size": [w, h], "items": items}, f)
        print("  %s %s: %d zones, %dx%dpx, %.0fs"
              % (series, page, len(items), w, h, time.perf_counter() - t0))

        del img, det_img, dets, crops, results, items
        import gc
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRIQUES
# ─────────────────────────────────────────────────────────────────────────────

def _kernel(n):
    n = max(1, int(n) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def _centroid(mask):
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.mean()), float(ys.mean())


def _changed(a, b, thr=20):
    """Pixels ou deux images different visiblement."""
    d = cv2.absdiff(a, b)
    if d.ndim == 3:
        d = cv2.cvtColor(d, cv2.COLOR_BGR2GRAY)
    return (d > thr).astype(np.uint8) * 255


def _hull_mask(src_ink, margin):
    """Emprise du texte SOURCE, dilatee. Indicateur d'ecart d'empreinte."""
    pts = cv2.findNonZero(src_ink)
    if pts is None or len(pts) < 3:
        return None
    hull = cv2.convexHull(pts)
    m = np.zeros(src_ink.shape, dtype=np.uint8)
    cv2.fillConvexPoly(m, hull, 255)
    if margin >= 1:
        m = cv2.dilate(m, _kernel(2 * int(margin) + 1))
    return m


def _band_heights(mask):
    """Hauteur MEDIANE d'une bande de texte, mesuree en pixels.

    On projette l'encre sur les lignes de l'image : une bande est une suite de
    lignes qui portent de l'encre, c'est-a-dire une ligne de texte. Sa hauteur
    est la hauteur des glyphes (capitale + jambage), pas le pas entre lignes.

    Applique des deux cotes (encre source et encre rendue), ce rapport est le
    seul controle du corps de texte qui ne partage AUCUNE constante avec le
    renderer : ni le 0,75 em de `calculate_optimal_font_size`, ni la hauteur des
    polygones OCR.
    """
    if mask is None or mask.size == 0:
        return None
    rows = np.count_nonzero(mask, axis=1)
    if rows.max() == 0:
        return None
    on = rows > max(1.0, 0.10 * float(rows.max()))
    heights, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            heights.append(i - start)
            start = None
    if start is not None:
        heights.append(len(on) - start)
    heights = sorted(h for h in heights if h >= 3)
    if not heights:
        return None
    return float(heights[len(heights) // 2])


def _ink_box(mask):
    """(largeur, hauteur, aire) de la boite englobante de l'encre."""
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) == 0:
        return None
    x, y, w, h = cv2.boundingRect(pts)
    return float(w), float(h), float(w * h)


def _zone_effacement_legitime(page_ink_c, item, dbg, forme):
    """Ce que l'effacement a le DROIT de repeindre, selon la classe.

    Une bavure se juge par rapport a ce que l'algorithme vise, pas par rapport
    aux glyphes. Pour une `bulle`, il vise l'encre dilatee de quelques pixels
    (anticrenelage). Pour un cartouche `out_text`, E3 lui donne deliberement un
    masque au BLOC — polygones de ligne dilates de 0,30 x hauteur de ligne —
    parce que le texte d'impact porte une LUEUR externe qu'un masque au glyphe
    laisse visible. Compter ce debordement voulu comme une bavure faisait
    ressortir 29 `out_text` sur 41 signales.
    """
    base = cv2.dilate(page_ink_c, _kernel(11)) if page_ink_c is not None else None
    classe = str(item.get("class_name") or "").lower()
    if classe != "out_text":
        return base
    # Enveloppe au BLOC : polygones de ligne remplis, dilates comme en E3.
    slh = (dbg or {}).get("source_line_height") or 0
    bloc = np.zeros(forme, dtype=np.uint8)
    for region in (item.get("text_regions") or []):
        pts = region.get("bbox") if isinstance(region, dict) else None
        if not pts or len(pts) < 3:
            continue
        arr = np.array(pts, dtype=np.int32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        arr[:, 0] = np.clip(arr[:, 0], 0, forme[1] - 1)
        arr[:, 1] = np.clip(arr[:, 1], 0, forme[0] - 1)
        cv2.fillPoly(bloc, [arr], 255)
    if np.count_nonzero(bloc) == 0:
        return base
    marge = max(5, int(0.30 * float(slh)) + 11)
    bloc = cv2.dilate(bloc, _kernel(2 * marge + 1))
    return bloc if base is None else cv2.bitwise_or(base, bloc)


def _bubble_interior(erased_pad, src_ink_pad):
    """Interieur du ballon, mesure sur l'image EFFACEE.

    Sur l'originale, le texte remplit le ballon et brouille toute detection de
    forme — le changelog (E5) l'avait constate et abandonne pour cette raison.
    Apres effacement la contrainte tombe : l'interieur est lisse. Sa couleur est
    celle de la couronne qui entourait les lettres, et on l'etend par connexite
    depuis l'emplacement du texte.

    Rend None quand la detection n'est pas fiable (out_text sans ballon, ballon
    sombre fondu dans son decor) : mieux vaut ne pas mesurer que mal mesurer.
    """
    if src_ink_pad is None or np.count_nonzero(src_ink_pad) < 30:
        return None
    ring = cv2.bitwise_and(cv2.dilate(src_ink_pad, _kernel(9)),
                           cv2.bitwise_not(cv2.dilate(src_ink_pad, _kernel(3))))
    if np.count_nonzero(ring) < 30:
        return None
    g = cv2.cvtColor(erased_pad, cv2.COLOR_BGR2GRAY).astype(np.int16)
    vals = g[ring > 0]
    ref = float(np.median(vals))
    # Écart ROBUSTE (MAD), pas l'écart-type : la couronne autour des lettres
    # contient le trait du ballon, très sombre, qui faisait exploser sigma —
    # donc la tolérance, donc la bande fuyait à travers le contour et couvrait
    # tout le crop (mesuré sur rise-of-the-dragon p01 #16 : « intérieur »
    # s'étendant sur 653 x 753 px, c'est-à-dire le décor entier).
    mad = float(np.median(np.abs(vals - ref)))
    tol = max(18.0, 3.0 * 1.4826 * mad)
    band = (np.abs(g - ref) <= tol).astype(np.uint8)
    n, lab = cv2.connectedComponents(band, connectivity=8)
    labs = lab[src_ink_pad > 0]
    labs = labs[labs > 0]
    if labs.size == 0:
        return None
    main = int(np.bincount(labs).argmax())
    interior = (lab == main).astype(np.uint8) * 255
    cov = np.count_nonzero(interior) / float(interior.size)
    # Trop petit : la detection a echoue. Trop grand : la bande a fui dans le
    # decor, il n'y a pas de ballon refermable dans ce crop.
    if not (0.10 <= cov <= 0.80):
        return None
    return interior


def measure_bubble(before_c, erased_c, after_c, src_ink, page_ink_c, dbg, item,
                   pad=None, voisines_pad=None):
    """Metriques d'une bulle. Coordonnees locales a la bbox.

    `pad` = (erased_pad, after_pad, src_ink_pad) sur un crop ELARGI. La bbox de
    detection est serree sur le texte, donc le ballon deborde d'elle : c'est
    seulement sur le crop elargi qu'on peut voir le texte sortir du ballon.

    `voisines_pad` marque, dans ce crop elargi, les zones des AUTRES detections.
    Sans lui, le texte redessine dans les bulles voisines etait compte comme de
    l'encre de CETTE bulle tombee hors de son ballon — 74 % de faux debordement
    mesure sur rise-of-the-dragon p01 #16.
    """
    h, w = before_c.shape[:2]
    area = float(max(1, h * w))
    m = {"bbox_w": w, "bbox_h": h}

    src_n = int(np.count_nonzero(src_ink)) if src_ink is not None else 0
    m["src_ink_px"] = src_n

    # ── Effacement ──────────────────────────────────────────────────────────
    erase_changed = _changed(before_c, erased_c)
    # L'union PLEINE PAGE de l'encre source sert de base : deux bulles qui se
    # chevauchent (bulles de cri) partagent de la bbox, et effacer la voisine
    # n'est pas une bavure. La zone legitime est ensuite elargie selon la
    # classe (cf. `_zone_effacement_legitime`).
    legit = _zone_effacement_legitime(page_ink_c, item, dbg, before_c.shape[:2])
    if legit is not None:
        spill = cv2.bitwise_and(erase_changed, cv2.bitwise_not(legit))
        m["erase_spill_pct"] = 100.0 * np.count_nonzero(spill) / area
    else:
        m["erase_spill_pct"] = None

    if src_n > 0:
        # Résidu d'effacement : là où l'encre était, l'image doit désormais
        # ressembler à son voisinage immédiat.
        #
        # L'ancienne mesure comparait deux MÉDIANES — insensible à un résidu
        # partiel. Mesuré sur 30-years p01 #14 : 26 860 pixels sombres restants
        # sur 38 183, soit 70 % du texte anglais encore visible, et
        # `ghost_contrast` restait sous son seuil parce que la médiane de la
        # zone basculait du côté effacé.
        #
        # On COMPTE désormais les pixels : quelle part de l'encre source
        # s'écarte encore nettement du fond local après effacement.
        core = cv2.erode(src_ink, _kernel(3))
        if np.count_nonzero(core) < 10:
            core = src_ink
        ring = cv2.bitwise_and(cv2.dilate(src_ink, _kernel(15)),
                               cv2.bitwise_not(cv2.dilate(src_ink, _kernel(5))))
        g_er = cv2.cvtColor(erased_c, cv2.COLOR_BGR2GRAY).astype(np.int16)
        if np.count_nonzero(ring) >= 10:
            fond = float(np.median(g_er[ring > 0]))
            # `ghost_contrast` : sur l'EFFACÉE, indicateur de qualité brute de
            # l'effacement (conservé, non décisif).
            m["ghost_contrast"] = float(abs(np.median(g_er[core > 0]) - fond))

            # `residu_pct` : résidu VISIBLE dans le résultat final. Un texte
            # `out_text` est réinjecté à la même position, donc un résidu
            # d'effacement y est RECOUVERT par le nouveau texte et ne se voit
            # pas — mesuré sur path-of-vengeance p02 #6, résidu brut 86 % mais
            # rendu final impeccable. On ne compte donc que l'encre source qui,
            # sur l'image FINALE, reste sombre HORS du texte rendu.
            rendered_ink = _changed(erased_c, after_c)
            decouvert = cv2.bitwise_and(core, cv2.bitwise_not(
                cv2.dilate(rendered_ink, _kernel(5))))
            n_dec = int(np.count_nonzero(decouvert))
            if n_dec >= 10:
                g_af = cv2.cvtColor(after_c, cv2.COLOR_BGR2GRAY).astype(np.int16)
                vals = g_af[decouvert > 0]
                # Part de l'encre source DÉCOUVERTE encore visible, rapportée à
                # toute l'encre source : « 40 % de résidu » = 40 % des lettres
                # d'origine transparaissent hors du nouveau texte.
                visibles = float(np.sum(np.abs(vals - fond) > 35))
                m["residu_pct"] = 100.0 * visibles / float(np.count_nonzero(core))
            else:
                m["residu_pct"] = 0.0
        else:
            m["ghost_contrast"] = None
            m["residu_pct"] = None
    else:
        m["ghost_contrast"] = None
        m["residu_pct"] = None

    # ── Corps et placement du texte rendu ───────────────────────────────────
    rendered_ink = _changed(erased_c, after_c)
    ren_n = int(np.count_nonzero(rendered_ink))
    m["rendered_ink_px"] = ren_n
    # Rapport d'aire d'encre BRUTE : contamine par le contour ajoute au rendu
    # (R9), donc conserve comme indicateur mais PAS comme critere.
    m["ink_px_ratio"] = (ren_n / src_n) if src_n > 0 else None

    if src_n > 0 and ren_n > 0:
        cs, cr = _centroid(src_ink), _centroid(rendered_ink)
        if cs and cr:
            m["centroid_dx"] = cr[0] - cs[0]
            m["centroid_dy"] = cr[1] - cs[1]
            m["centroid_dx_norm"] = abs(cr[0] - cs[0]) / max(1.0, float(w))
            m["centroid_dy_norm"] = abs(cr[1] - cs[1]) / max(1.0, float(h))

        # Empreinte : le pave rendu occupe-t-il la meme surface que l'original ?
        # Mesure sur la boite englobante de l'encre, insensible a l'epaisseur du
        # contour, contrairement au comptage de pixels.
        box_src, box_ren = _ink_box(src_ink), _ink_box(rendered_ink)
        if box_src and box_ren:
            m["footprint_ratio"] = box_ren[2] / box_src[2]
            m["src_ink_h"] = box_src[1]
            m["rendered_ink_h"] = box_ren[1]
            # PAS entre lignes (hauteur du pave / nb de lignes) : dit si le bloc
            # a la meme densite que l'original, PAS si les glyphes ont la bonne
            # taille — les deux peuvent diverger via l'interligne.
            n_ren = (dbg or {}).get("n_lines")
            slh = (dbg or {}).get("source_line_height")
            if n_ren and slh and float(slh) > 4:
                m["px_pitch_ratio"] = (box_ren[1] / float(n_ren)) / float(slh)

        # CORPS DE TEXTE mesure en pixels des deux cotes : c'est le controle
        # independant du renderer.
        cap_src = _band_heights(src_ink)
        cap_ren = _band_heights(rendered_ink)
        m["cap_src_px"] = cap_src
        m["cap_rendered_px"] = cap_ren
        if cap_src and cap_ren:
            m["cap_ratio"] = cap_ren / cap_src

        slh = dbg.get("source_line_height") if dbg else None
        hull = _hull_mask(src_ink, 0.5 * float(slh) if slh else 8)
        if hull is not None:
            out = cv2.bitwise_and(rendered_ink, cv2.bitwise_not(hull))
            m["footprint_out_pct"] = 100.0 * np.count_nonzero(out) / float(ren_n)

    # ── Debordement hors du ballon (crop elargi) ────────────────────────────
    if pad is not None:
        erased_pad, after_pad, src_ink_pad = pad
        interior = _bubble_interior(erased_pad, src_ink_pad)
        if interior is not None:
            ren_pad = _changed(erased_pad, after_pad)
            if voisines_pad is not None:
                ren_pad = cv2.bitwise_and(ren_pad, cv2.bitwise_not(voisines_pad))
            n_pad = int(np.count_nonzero(ren_pad))
            if n_pad > 0:
                # 2 px de tolerance : l'anticrenelage du texte mord legitimement
                # sur le trait interieur du ballon.
                allow = cv2.dilate(interior, _kernel(5))
                out = cv2.bitwise_and(ren_pad, cv2.bitwise_not(allow))
                m["bubble_overflow_pct"] = 100.0 * np.count_nonzero(out) / float(n_pad)

    # ── Lecture du hook de mise en page ─────────────────────────────────────
    if dbg and not dbg.get("bail"):
        slh = dbg.get("source_line_height")
        fs = dbg.get("font_size_final")
        # `calculate_optimal_font_size` plafonne a em_source = slh / 0.75 ; le
        # ratio dit donc directement de combien on est SOUS le corps d'origine.
        if slh and fs:
            m["line_h_ratio"] = (float(fs) * 0.75) / float(slh)
        m["font_size"] = fs
        m["source_line_height"] = slh
        m["mode"] = dbg.get("mode", "wrap")
        m["route_costs"] = dbg.get("route_costs") or {}
        m["blocage"] = dbg.get("blocage") or {}
        # Lignes effectivement rendues : sans elles, impossible d'analyser un
        # orphelin ou un mauvais découpage après coup.
        m["lignes"] = list(dbg.get("lines") or [])[:12]
        m["n_lines"] = dbg.get("n_lines")
        m["fill_ratio"] = dbg.get("fill_ratio")
        m["zone_dx"] = dbg.get("dx")
        m["zone_dy"] = dbg.get("dy")

        n_src = len(item.get("text_regions") or [])
        m["n_lines_src"] = n_src
        if n_src and dbg.get("n_lines"):
            m["n_lines_delta"] = int(dbg["n_lines"]) - n_src

        # ── Qualite du pave (rag) — indicateur pour le levier suivant ──────
        lines = [l for l in (dbg.get("lines") or []) if l.strip()]
        if len(lines) >= 2:
            # Largeurs en PIXELS quand le renderer les expose : c'est ce que
            # l'oeil voit, et c'est ce que `_rebalance_lines` optimise. Repli
            # sur le nombre de caracteres pour les runs anterieurs au hook.
            px = [w for w in (dbg.get("line_widths_px") or []) if w]
            lens = px if len(px) == len(lines) else [len(l) for l in lines]
            mean = sum(lens) / len(lens)
            var = sum((x - mean) ** 2 for x in lens) / len(lens)
            m["rag_cv"] = (math.sqrt(var) / mean) if mean else None
            m["orphan"] = bool(len(lines[-1].split()) == 1 and len(lines[-1]) <= 6)
    else:
        m["bail"] = (dbg or {}).get("bail", "no_debug")

    return m


def verdict(m):
    """Liste des seuils franchis. Vide = bulle conforme."""
    bad = []
    S = SEUILS
    if m.get("bail"):
        return ["bail:" + str(m["bail"])]
    v = m.get("erase_spill_pct")
    if v is not None and v > S["erase_spill_pct"]:
        bad.append("erase_spill")
    v = m.get("ghost_contrast")
    if v is not None and v > S["ghost_contrast"]:
        bad.append("ghost")
    # Résidu COMPTÉ, insensible au basculement de la médiane.
    v = m.get("residu_pct")
    if v is not None and v > S["residu_pct"]:
        bad.append("residu")
    # `empreinte` et `trop_de_lignes` ont été RETIRÉS du verdict.
    #
    # Tous deux demandent au pavé rendu de reproduire la géométrie du pavé
    # SOURCE : même aire, même nombre de lignes. C'est impossible à corps
    # correct dès que la police diffère — celle du studio fait 0,72 de largeur
    # par hauteur de casse, la nôtre 0,80 (mesuré, cf. changelog). À `cap_ratio`
    # égal, le même texte prend donc plus de largeur, donc plus de lignes, donc
    # plus de surface. Ces deux critères pénalisent la justesse.
    #
    # Falsification directe, mesurée sur 5 bulles où `empreinte` a doublé :
    # `cap_ratio` 0,67-0,86 -> 0,91-1,12 (entré dans la cible) et le rendu est
    # visiblement meilleur — texte lisible au lieu de minuscule. Voir
    # `planche_ref16_vs_budget-cout_footprint_ratio.png`.
    #
    # Ils restent MESURÉS et affichés : un écart énorme reste un signal. Ils ne
    # décident simplement plus de la conformité, qui repose désormais sur des
    # critères insensibles à la police : effacement, corps, centrage,
    # débordement hors ballon, orphelin.
    # Le corps se juge sur `cap_ratio` (pixels des deux cotes) et JAMAIS sur
    # `line_h_ratio` : celui-ci a `source_line_height` au denominateur, donc il
    # change de sens des qu'on touche a la facon de mesurer la source — il
    # n'est pas comparable d'un run a l'autre.
    v = m.get("cap_ratio")
    if v is not None and v < S["cap_ratio_min"]:
        bad.append("corps_petit")
    if v is not None and v > S["cap_ratio_max"]:
        bad.append("corps_gros")
    if (m.get("centroid_dx_norm") or 0) > S["centroid_dx_norm"]:
        bad.append("decentre_x")
    if (m.get("centroid_dy_norm") or 0) > S["centroid_dy_norm"]:
        bad.append("decentre_y")
    v = m.get("bubble_overflow_pct")
    if v is not None and v > S["bubble_overflow_pct"]:
        bad.append("debordement")

    # Le mot seul sur la dernière ligne est le défaut de lettrage le plus
    # visible, et il n'était compté nulle part — seulement affiché en bas de
    # rapport. Un pavé qui finit sur un orphelin n'est pas « conforme ».
    if S.get("orphelin_compte") and m.get("orphan"):
        bad.append("orphelin")
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — NOTATION (repart du cache, rapide)
# ─────────────────────────────────────────────────────────────────────────────

def score(run, only=None, save_pages=False, save_crops=False):
    from core import TextRenderer
    from core.bubble_shape import has_closed_bubble
    from core.detector import Detection
    from pipeline import TranslationPipeline

    run_dir = RUNS_ROOT / run
    run_dir.mkdir(parents=True, exist_ok=True)

    caches = (sorted(d for d in CACHE_ROOT.iterdir() if (d / "dets.pkl").exists())
              if CACHE_ROOT.exists() else [])
    if only:
        caches = [d for d in caches if only.lower() in d.name.lower()]
    if not caches:
        raise SystemExit("Cache vide — lancer `python scratch/bareme.py build` d'abord.")

    renderer = TextRenderer()
    all_rows = []

    for cache_dir in caches:
        t0 = time.perf_counter()
        with open(cache_dir / "dets.pkl", "rb") as f:
            blob = pickle.load(f)
        series, items = blob["series"], blob["items"]
        page = blob.get("page", "p01")
        img = cv2.imread(str(cache_dir / "page.png"))
        if img is None or not items:
            print("  %s %s: rien a noter" % (series, page))
            continue
        H, W = img.shape[:2]

        # Reconstruction des detections (pas de YOLO/OCR ici).
        dets = []
        for it in items:
            d = Detection(it["class_name"], [float(v) for v in it["bbox"]], it["score"])
            d.text_original = it["text"]
            d.text_translated = it["text"]          # traduction desactivee
            d.ocr_confidence = it["ocr_confidence"]
            d.text_regions = it["text_regions"]
            d.mask_regions = it.get("mask_regions")
            d.chirurgical_mask = it.get("chirurgical_mask")
            d.mask_binary = it.get("mask_binary")
            dets.append(d)

        # Masque d'encre SOURCE, par bulle puis en union pleine page.
        src_inks = []
        page_ink = np.zeros((H, W), dtype=np.uint8)
        for it in items:
            x1, y1, x2, y2 = it["bbox"]
            crop = img[y1:y2, x1:x2]
            ink = TranslationPipeline._ocr_mask_from_regions(
                it["text_regions"], y2 - y1, x2 - x1, crop, dilate=3)
            src_inks.append(ink)
            if ink is not None:
                page_ink[y1:y2, x1:x2] = np.maximum(page_ink[y1:y2, x1:x2], ink)

        # Effacement, puis reinjection du texte SOURCE.
        erased, _, _ = TranslationPipeline._run_pre_inpainting(img, dets, renderer)
        after = erased.copy()
        debugs = []
        for d in dets:
            try:
                TranslationPipeline._prepare_render_style(renderer, img, d)
                renderer.last_layout_debug = None
                after = renderer.insert_text(
                    after, d.text_translated,
                    int(d.x1), int(d.y1), int(d.x2), int(d.y2),
                    text_regions=getattr(d, "text_regions", None),
                    text_color_rgb=getattr(d, "text_color_rgb", None),
                    outline_width_px=getattr(d, "measured_outline_px", None),
                    source_text=getattr(d, "text_original", None),
                    text_style=getattr(d, "text_style", "dialogue"),
                    font_hint=getattr(d, "font_hint", "regular"),
                    class_name=str(d.class_name),
                    bubble_mask=getattr(d, "mask_binary", None),
                    font_key=getattr(d, "font_key", None),
                    source_line_height=getattr(d, "source_line_height", None),
                    # Le barème doit passer CHAQUE argument que `pipeline.py`
                    # passe, sinon il note un chemin que la production
                    # n'emprunte pas — le piège déjà rencontré sur
                    # `render_iterate.py`. Verdict calculé comme en production :
                    # l'encre se lit sur l'ORIGINE, le trait du ballon sur
                    # l'image en cours de rendu.
                    bubble_present=has_closed_bubble(
                        img, after, (int(d.x1), int(d.y1), int(d.x2), int(d.y2)),
                        getattr(d, "text_regions", None),
                    ),
                    sibling_boxes=[(o.x1, o.y1, o.x2, o.y2) for o in dets if o is not d],
                )
                debugs.append(renderer.last_layout_debug)
            except Exception as e:
                debugs.append({"bail": "exception:%s:%s" % (type(e).__name__, e)})

        # Mesure.
        rows = []
        for i, (it, ink, dbg) in enumerate(zip(items, src_inks, debugs)):
            x1, y1, x2, y2 = it["bbox"]
            b = img[y1:y2, x1:x2]
            e = erased[y1:y2, x1:x2]
            a = after[y1:y2, x1:x2]
            if b.size == 0:
                continue

            # Crop elargi de 60 % : la bbox est serree sur le texte, le ballon
            # deborde d'elle. Sans marge, un texte qui sort du ballon reste
            # invisible a la mesure.
            bh, bw = b.shape[:2]
            mx, my = int(0.6 * bw), int(0.6 * bh)
            px1, py1 = max(0, x1 - mx), max(0, y1 - my)
            px2, py2 = min(W, x2 + mx), min(H, y2 + my)
            src_pad = None
            if ink is not None:
                src_pad = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
                ih, iw = ink.shape[:2]
                src_pad[y1 - py1:y1 - py1 + ih, x1 - px1:x1 - px1 + iw] = ink
            pad = (erased[py1:py2, px1:px2], after[py1:py2, px1:px2], src_pad)

            # Zones des AUTRES detections presentes dans le crop elargi : leur
            # texte redessine n'appartient pas a cette bulle.
            vois = np.zeros((py2 - py1, px2 - px1), dtype=np.uint8)
            for j, other in enumerate(items):
                if j == i:
                    continue
                ox1, oy1, ox2, oy2 = other["bbox"]
                cx1, cy1 = max(px1, ox1), max(py1, oy1)
                cx2, cy2 = min(px2, ox2), min(py2, oy2)
                if cx2 > cx1 and cy2 > cy1:
                    vois[cy1 - py1:cy2 - py1, cx1 - px1:cx2 - px1] = 255

            m = measure_bubble(b, e, a, ink, page_ink[y1:y2, x1:x2], dbg, it, pad, vois)
            m.update({"series": series, "page": page,
                      "index": i, "class": it["class_name"],
                      "text": (it["text"] or "")[:70], "bbox": it["bbox"]})
            m["defauts"] = verdict(m)
            rows.append(m)

            if save_crops:
                cd = run_dir / "crops" / ("%s__%s" % (_slug(series), page))
                cd.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cd / ("%03d_before.png" % i)), img[py1:py2, px1:px2])
                cv2.imwrite(str(cd / ("%03d_after.png" % i)), after[py1:py2, px1:px2])

        all_rows.extend(rows)
        clean = sum(1 for r in rows if not r["defauts"])
        print("  %s %s: %d/%d bulles conformes (%.0fs)"
              % (series, page, clean, len(rows), time.perf_counter() - t0))

        if save_pages:
            out = run_dir / "pages" / ("%s__%s" % (_slug(series), page))
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / "erased.png"), erased)
            cv2.imwrite(str(out / "after.png"), after)

        del img, erased, after, page_ink, src_inks
        import gc
        gc.collect()

    payload = {"run": run, "seuils": SEUILS, "n_bulles": len(all_rows),
               "rows": all_rows}
    (run_dir / "scores.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print("\n%d bulles -> %s" % (len(all_rows), run_dir / "scores.json"))
    report(run)


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT
# ─────────────────────────────────────────────────────────────────────────────

def _pcts(vals):
    v = sorted(x for x in vals
               if isinstance(x, (int, float)) and not isinstance(x, bool))
    if not v:
        return None, None, None
    def q(p):
        return v[min(len(v) - 1, int(p * len(v)))]
    return q(0.50), q(0.75), q(0.90)


DEFAUTS = ["erase_spill", "ghost", "residu", "corps_petit", "corps_gros",
           "decentre_x", "decentre_y", "debordement", "orphelin"]
# Mesurés et affichés, mais NON décisifs (cf. `verdict`).
INDICATEURS = ["footprint_ratio", "n_lines_delta"]


def cle(r):
    """Identite d'une bulle. La PLANCHE en fait partie : les index sont locaux
    a une planche, donc deux bulles differentes partagent l'index 0."""
    return (r["series"], r.get("page", "p01"), r["index"])


def load_run(run):
    p = RUNS_ROOT / run / "scores.json"
    if not p.exists():
        raise SystemExit("Run introuvable: %s" % p)
    data = json.loads(p.read_text(encoding="utf-8"))
    # Rejuge chaque bulle avec les seuils COURANTS. Sans ca, comparer deux runs
    # notes a des epoques differentes compare deux barremes, pas deux rendus.
    for r in data.get("rows", []):
        r["defauts"] = verdict(r)
    return data


def report(run):
    data = load_run(run)
    rows = data["rows"]
    if not rows:
        print("Aucune bulle.")
        return

    series = sorted({r["series"] for r in rows})
    print("\n" + "=" * 78)
    print("BAREME << %s >> — %d bulles, %d series" % (run, len(rows), len(series)))
    print("=" * 78)

    print("\n%-40s %7s %10s %6s" % ("Serie", "bulles", "conformes", "%"))
    print("-" * 66)
    for s in series:
        rs = [r for r in rows if r["series"] == s]
        ok = sum(1 for r in rs if not r["defauts"])
        print("%-40s %7d %10d %5.0f%%" % (s[:40], len(rs), ok, 100.0 * ok / len(rs)))
    ok = sum(1 for r in rows if not r["defauts"])
    print("-" * 66)
    print("%-40s %7d %10d %5.0f%%" % ("TOTAL", len(rows), ok, 100.0 * ok / len(rows)))

    print("\n%-18s %7s %6s   %s" % ("Defaut", "bulles", "%", "par serie"))
    print("-" * 78)
    for d in DEFAUTS:
        hit = [r for r in rows if d in r["defauts"]]
        if not hit:
            continue
        per = {}
        for r in hit:
            per[r["series"]] = per.get(r["series"], 0) + 1
        top = ", ".join("%s:%d" % (k[:22], v) for k, v in
                        sorted(per.items(), key=lambda kv: -kv[1])[:3])
        print("%-18s %7d %5.0f%%   %s" % (d, len(hit), 100.0 * len(hit) / len(rows), top))
    bails = [r for r in rows if any(str(x).startswith("bail:") for x in r["defauts"])]
    if bails:
        print("%-18s %7d %5.0f%%" % ("bail (non rendu)", len(bails),
                                     100.0 * len(bails) / len(rows)))

    print("\n%-20s %9s %9s %9s   %s" % ("Metrique", "p50", "p75", "p90", "cible"))
    print("-" * 78)
    cibles = [
        ("erase_spill_pct", "< %s" % SEUILS["erase_spill_pct"]),
        ("ghost_contrast", "< %s" % SEUILS["ghost_contrast"]),
        ("residu_pct", "< %s  (encre source encore visible)" % SEUILS["residu_pct"]),
        ("cap_ratio", "%s a %s  (CORPS, pixels des deux cotes)"
         % (SEUILS["cap_ratio_min"], SEUILS["cap_ratio_max"])),
        ("line_h_ratio", "(indicatif — depend de source_line_height)"),
        ("px_pitch_ratio", "~ 1.00  (pas entre lignes)"),
        ("footprint_ratio", "~ 1.00"),
        ("centroid_dx_norm", "< %s" % SEUILS["centroid_dx_norm"]),
        ("centroid_dy_norm", "< %s" % SEUILS["centroid_dy_norm"]),
        ("bubble_overflow_pct", "< %s" % SEUILS["bubble_overflow_pct"]),
        ("ink_px_ratio", "(indicatif, contour inclus)"),
        ("footprint_out_pct", "(indicatif, ecart d'empreinte)"),
        ("fill_ratio", "(indicatif)"),
        ("rag_cv", "plus bas = mieux"),
    ]
    for k, cible in cibles:
        p50, p75, p90 = _pcts([r.get(k) for r in rows])
        if p50 is None:
            continue
        n = sum(1 for r in rows if isinstance(r.get(k), (int, float))
                and not isinstance(r.get(k), bool))
        suffix = "" if n == len(rows) else "   [mesure sur %d/%d]" % (n, len(rows))
        print("%-20s %9.3f %9.3f %9.3f   %s%s" % (k, p50, p75, p90, cible, suffix))
    orph = [r for r in rows if r.get("orphan")]
    multi = [r for r in rows if r.get("n_lines") and r["n_lines"] >= 2]
    if multi:
        print("%-20s %9d %9s %9s   %.0f%% des blocs multi-lignes"
              % ("orphelins", len(orph), "", "", 100.0 * len(orph) / len(multi)))


def sheet_vs(run_a, run_b, n=10, tile_h=230, par="cap_ratio"):
    """Planche ORIGINAL | RENDU `run_a` | RENDU `run_b`, sur les bulles qui ont
    le plus bouge entre les deux runs.

    C'est la seule vue qui permette de VALIDER un correctif : le tableau de
    chiffres dit qu'il a bouge, celle-ci dit dans quel sens.
    """
    a, b = load_run(run_a), load_run(run_b)
    ra = {cle(r): r for r in a["rows"]}
    rb = {cle(r): r for r in b["rows"]}
    da, db = RUNS_ROOT / run_a, RUNS_ROOT / run_b
    for d, nm in ((da, run_a), (db, run_b)):
        if not (d / "crops").exists():
            raise SystemExit("Crops absents pour %s — relancer: score --run %s --crops" % (nm, nm))

    # Les plus gros ecarts de corps, un par serie au moins.
    common = [k for k in (set(ra) & set(rb))
              if isinstance(ra[k].get(par), (int, float))
              and isinstance(rb[k].get(par), (int, float))]
    common.sort(key=lambda k: -abs(rb[k][par] - ra[k][par]))
    vus, picks = set(), []
    for k in common:                      # d'abord un cas par serie
        if k[0] not in vus:
            vus.add(k[0]); picks.append(k)
    picks += [k for k in common if k not in picks][:max(0, n - len(picks))]

    tiles = []
    for k in picks:
        s, pg, i = k
        sous = "%s__%s" % (_slug(s), pg)
        orig = cv2.imread(str(da / "crops" / sous / ("%03d_before.png" % i)))
        av = cv2.imread(str(da / "crops" / sous / ("%03d_after.png" % i)))
        ap = cv2.imread(str(db / "crops" / sous / ("%03d_after.png" % i)))
        if orig is None or av is None or ap is None:
            continue
        sc = tile_h / float(max(1, orig.shape[0]))
        dim = (max(1, int(orig.shape[1] * sc)), tile_h)
        cells = [cv2.resize(x, dim, interpolation=cv2.INTER_AREA) for x in (orig, av, ap)]
        sep = np.full((tile_h, 3, 3), 40, dtype=np.uint8)
        body = np.hstack([cells[0], sep, cells[1], sep, cells[2]])
        bar = np.full((26, body.shape[1], 3), 250, dtype=np.uint8)
        cv2.putText(bar, "%s %s #%d   ORIGINAL | %s %s=%.2f | %s %s=%.2f"
                    % (s[:20], pg, i, run_a[:10], par[:7], ra[k][par],
                       run_b[:10], par[:7], rb[k][par]),
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, body]))

    if not tiles:
        raise SystemExit("Aucun crop lisible.")
    width = max(t.shape[1] for t in tiles)
    out_rows = []
    for t_ in tiles:
        if t_.shape[1] < width:
            t_ = np.hstack([t_, np.full((t_.shape[0], width - t_.shape[1], 3), 250, np.uint8)])
        out_rows.append(t_)
        out_rows.append(np.full((6, width, 3), 250, np.uint8))
    out = RUNS_ROOT / ("planche_%s_vs_%s_%s.png" % (run_a, run_b, par))
    cv2.imwrite(str(out), np.vstack(out_rows))
    print("Planche ORIGINAL | %s | %s -> %s" % (run_a, run_b, out))
    return out


def sheet(run, metric="cap_ratio", n=3, tile_h=230):
    """Planche-contact ORIGINAL | RENDU des cas extremes de `metric`.

    Un barreme qui ne renvoie que des nombres se calibre a l'aveugle. Cette
    planche met en regard ce que le chiffre dit et ce que l'oeil voit, serie par
    serie, aux deux extremites de la distribution.
    """
    data = load_run(run)
    run_dir = RUNS_ROOT / run
    rows = [r for r in data["rows"] if isinstance(r.get(metric), (int, float))]
    if not rows:
        raise SystemExit("Metrique %s absente du run %s" % (metric, run))
    if not (run_dir / "crops").exists():
        raise SystemExit("Crops absents — relancer: score --run %s --crops" % run)

    def tile(rec, tag):
        cd = run_dir / "crops" / ("%s__%s" % (_slug(rec["series"]),
                                              rec.get("page", "p01")))
        b = cv2.imread(str(cd / ("%03d_before.png" % rec["index"])))
        a = cv2.imread(str(cd / ("%03d_after.png" % rec["index"])))
        if b is None or a is None:
            return None
        sc = tile_h / float(max(1, b.shape[0]))
        dim = (max(1, int(b.shape[1] * sc)), tile_h)
        b = cv2.resize(b, dim, interpolation=cv2.INTER_AREA)
        a = cv2.resize(a, dim, interpolation=cv2.INTER_AREA)
        sep = np.full((tile_h, 3, 3), 40, dtype=np.uint8)
        body = np.hstack([b, sep, a])
        bar = np.full((26, body.shape[1], 3), 250, dtype=np.uint8)
        cv2.putText(bar, "%s %s #%d  %s=%.2f  [%s]"
                    % (rec["series"][:24], rec.get("page", "p01"),
                       rec["index"], metric, rec[metric], tag),
                    (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1, cv2.LINE_AA)
        return np.vstack([bar, body])

    tiles = []
    for s in sorted({r["series"] for r in rows}):
        rs = sorted((r for r in rows if r["series"] == s), key=lambda r: r[metric])
        for r in rs[:n]:
            t = tile(r, "le plus BAS")
            if t is not None:
                tiles.append(t)
        for r in rs[-n:]:
            t = tile(r, "le plus HAUT")
            if t is not None:
                tiles.append(t)

    if not tiles:
        raise SystemExit("Aucun crop lisible.")
    width = max(t.shape[1] for t in tiles)
    padded = []
    for t in tiles:
        if t.shape[1] < width:
            t = np.hstack([t, np.full((t.shape[0], width - t.shape[1], 3), 250, np.uint8)])
        padded.append(t)
        padded.append(np.full((6, width, 3), 250, np.uint8))
    out = run_dir / ("planche_%s.png" % metric)
    cv2.imwrite(str(out), np.vstack(padded))
    print("Planche-contact (gauche = ORIGINAL, droite = RENDU) -> %s" % out)
    return out


def compare(run_a, run_b):
    a, b = load_run(run_a), load_run(run_b)
    ra = {cle(r): r for r in a["rows"]}
    rb = {cle(r): r for r in b["rows"]}
    common = sorted(set(ra) & set(rb))
    print("\n" + "=" * 78)
    print("%s  ->  %s   (%d bulles communes)" % (run_a, run_b, len(common)))
    print("=" * 78)

    oka = sum(1 for k in common if not ra[k]["defauts"])
    okb = sum(1 for k in common if not rb[k]["defauts"])
    n = max(1, len(common))
    print("\nBulles conformes : %d (%.0f%%)  ->  %d (%.0f%%)   [%+d]"
          % (oka, 100.0 * oka / n, okb, 100.0 * okb / n, okb - oka))

    print("\n%-18s %12s %12s %7s" % ("Defaut", run_a[:12], run_b[:12], "delta"))
    print("-" * 60)
    for d in DEFAUTS:
        ca = sum(1 for k in common if d in ra[k]["defauts"])
        cb = sum(1 for k in common if d in rb[k]["defauts"])
        if ca or cb:
            flag = "  OK" if cb < ca else ("  << REGRESSION" if cb > ca else "")
            print("%-18s %12d %12d %+7d%s" % (d, ca, cb, cb - ca, flag))

    print("\n%-20s %16s %16s" % ("Metrique", "p50 " + run_a[:8], "p50 " + run_b[:8]))
    print("-" * 56)
    for k in ("cap_ratio", "footprint_ratio", "erase_spill_pct", "ghost_contrast",
              "centroid_dy_norm", "bubble_overflow_pct", "rag_cv"):
        pa = _pcts([ra[x].get(k) for x in common])[0]
        pb = _pcts([rb[x].get(k) for x in common])[0]
        if pa is None or pb is None:
            continue
        print("%-20s %16.3f %16.3f" % (k, pa, pb))

    # Les bulles qui basculent : c'est la qu'on regarde les crops.
    casse = [k for k in common if not ra[k]["defauts"] and rb[k]["defauts"]]
    repare = [k for k in common if ra[k]["defauts"] and not rb[k]["defauts"]]
    print("\nReparees : %d   |   Cassees : %d" % (len(repare), len(casse)))
    for k in casse[:12]:
        print("  X %-22s %s #%-3d %-24s %s"
              % (k[0][:22], k[1], k[2], ",".join(rb[k]["defauts"])[:24],
                 (rb[k].get("text") or "")[:30]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Bareme de rendu multi-series")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="cache detection+OCR (lent, une fois)")
    b.add_argument("--only", help="filtre sur le nom de serie")
    b.add_argument("--force", action="store_true")

    s = sub.add_parser("score", help="rejoue effacement+rendu et note")
    s.add_argument("--run", required=True, help="nom du run (ex: baseline)")
    s.add_argument("--only", help="filtre sur le nom de serie")
    s.add_argument("--save-pages", action="store_true", help="ecrit erased/after pleine page")
    s.add_argument("--crops", action="store_true", help="ecrit un crop avant/apres par bulle")

    r = sub.add_parser("report", help="tableau d'un run")
    r.add_argument("--run", required=True)

    h = sub.add_parser("sheet", help="planche-contact des cas extremes")
    h.add_argument("--run", required=True)
    h.add_argument("--metric", default="cap_ratio")
    h.add_argument("--n", type=int, default=3)
    h.add_argument("--vs", help="second run : planche ORIGINAL | run | vs")

    c = sub.add_parser("compare", help="diff entre deux runs")
    c.add_argument("run_a")
    c.add_argument("run_b")

    args = ap.parse_args()
    if args.cmd == "build":
        build(args.only, args.force)
    elif args.cmd == "score":
        score(args.run, args.only, args.save_pages, args.crops)
    elif args.cmd == "report":
        report(args.run)
    elif args.cmd == "sheet":
        if args.vs:
            sheet_vs(args.vs, args.run, max(args.n, 8), par=args.metric)
        else:
            sheet(args.run, args.metric, args.n)
    elif args.cmd == "compare":
        compare(args.run_a, args.run_b)
