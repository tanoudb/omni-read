# -*- coding: utf-8 -*-
"""Banc de validation de la forme du ballon : TEINTE (actuel) vs CROISSANCE.

Compare les deux méthodes sur trois familles de cas, et chiffre ce qui compte
pour la mise en page :

  - remplissage      : aire du masque / aire de la bbox. Un cercle inscrit fait
                       78,5 %, une ellipse inscrite aussi. Trop bas = le masque
                       rate le ballon ; trop haut = il déborde sur le décor.
  - dérive du centre : écart entre le centre le plus à gauche et le plus à
                       droite, mesuré bande par bande. C'est LA métrique qui
                       compte : une dérive pousse les lignes hors du ballon.
                       Un ballon symétrique donne une dérive quasi nulle.
  - asymétrie        : écart moyen entre la demi-largeur gauche et la demi-
                       largeur droite, rapporté à la largeur. Complète la
                       dérive : un masque peut avoir un centre stable et être
                       tordu.
  - IoU              : recouvrement entre les deux méthodes, pour voir où elles
                       divergent réellement.

Les cas sont choisis PAR MESURE, pas codés en dur : la bulle de cri est celle
dont la solidité (aire / aire de l'enveloppe convexe) est la plus basse, les
bulles classiques celles dont la solidité est la plus haute. Le banc reste donc
valable si les runs sont régénérés.

Usage:
    python scratch/test_mask_growth.py
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import numpy as np

from core.bubble_shape import grow_from_ink, ink_mask_from_regions
from core.renderer import TextRenderer

RUN_SOMBRE = "scratch/render_out/DRAG_BIG"
RUN_CLAIR = "scratch/render_out/POV_BBOX"
N_CLAIRES = 10


# ── métriques ────────────────────────────────────────────────────────────────

def bandes(mask, n=9):
    """[(largeur, centre, x_min, x_max)] sur n bandes horizontales."""
    rows = np.nonzero(mask.any(axis=1))[0]
    if rows.size < 2:
        return []
    out = []
    for r in np.linspace(rows.min(), rows.max(), n).astype(int):
        cols = np.nonzero(mask[r])[0]
        if cols.size:
            out.append((int(cols.max() - cols.min() + 1),
                        (cols.min() + cols.max()) / 2.0,
                        int(cols.min()), int(cols.max())))
    return out


def metriques(mask):
    if mask is None:
        return None
    h, w = mask.shape[:2]
    b = bandes(mask)
    if len(b) < 3:
        return None
    centres = [c for _, c, _, _ in b]
    derive = max(centres) - min(centres)

    # Asymétrie : autour du centre GLOBAL du masque, pas du centre de bande.
    cols_all = np.nonzero(mask.any(axis=0))[0]
    cx = (cols_all.min() + cols_all.max()) / 2.0
    ecarts = [abs((cx - xmin) - (xmax - cx)) for _, _, xmin, xmax in b]

    return {
        "fill": float(np.count_nonzero(mask)) / float(h * w),
        "derive_px": float(derive),
        "derive_pct": 100.0 * derive / float(w),
        "asym_pct": 100.0 * statistics.mean(ecarts) / float(w),
        "largeurs": [l for l, _, _, _ in b],
    }


def iou(a, b):
    if a is None or b is None:
        return None
    A, B = a > 0, b > 0
    u = int((A | B).sum())
    return float((A & B).sum()) / u if u else 0.0


def solidite(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 1.0
    c = max(cnts, key=cv2.contourArea)
    a = cv2.contourArea(c)
    ah = cv2.contourArea(cv2.convexHull(c))
    return float(a / ah) if ah > 0 else 1.0


# ── chargement des cas ───────────────────────────────────────────────────────

def charge_run(run):
    meta = json.loads(Path(f"{run}/bubbles_meta.json").read_text(encoding="utf-8"))
    orig = cv2.imread(f"{run}/page_before.png")
    era = cv2.imread(f"{run}/page_erased.png")
    return meta, orig, era


def prepare(det, orig, era):
    x1, y1, x2, y2 = det["bbox"]
    c_o = orig[max(0, y1):y2, max(0, x1):x2]
    c_e = era[max(0, y1):y2, max(0, x1):x2]
    if c_o.size == 0 or c_e.size == 0 or c_o.shape[:2] != c_e.shape[:2]:
        return None
    ink = ink_mask_from_regions(c_o, det.get("text_regions"))
    return c_o, c_e, ink


def choisir_cas():
    """(nom, det, crop_orig, crop_erased, ink) pour chaque cas du banc."""
    cas = []

    meta, orig, era = charge_run(RUN_SOMBRE)
    for d in meta:
        p = prepare(d, orig, era)
        if p:
            cas.append(("SOMBRE  dragon", d, *p))

    meta, orig, era = charge_run(RUN_CLAIR)
    candidats = []
    for d in meta:
        if d.get("class") != "bulle":
            continue
        p = prepare(d, orig, era)
        if not p:
            continue
        c_o, c_e, ink = p
        ancien = TextRenderer._bubble_mask_from_image(c_e)
        s = solidite(ancien) if ancien is not None else 1.0
        candidats.append((s, d, c_o, c_e, ink))

    candidats.sort(key=lambda t: t[0])
    # La plus DENTELÉE : solidité minimale.
    if candidats:
        s, d, c_o, c_e, ink = candidats[0]
        cas.append((f"CRI     pov#{d['index']:02d} (solidite {s:.3f})", d, c_o, c_e, ink))
    # Les plus RONDES : solidité maximale.
    for s, d, c_o, c_e, ink in list(reversed(candidats))[:N_CLAIRES]:
        cas.append((f"CLAIRE  pov#{d['index']:02d}", d, c_o, c_e, ink))
    return cas


# ── exécution ────────────────────────────────────────────────────────────────

def main():
    cas = choisir_cas()
    print(f"{len(cas)} cas\n")
    entete = (f"{'cas':>28s} | {'methode':10s} {'fill%':>6s} {'derive px':>9s} "
              f"{'derive%':>8s} {'asym%':>6s} {'mode':>18s}")
    print(entete)
    print("-" * len(entete))

    lignes = []
    for nom, det, c_o, c_e, ink in cas:
        ancien = TextRenderer._bubble_mask_from_image(c_e)
        nouveau, diag = grow_from_ink(c_e, ink)

        m_a, m_n = metriques(ancien), metriques(nouveau)
        rec = {"cas": nom, "bbox": det["bbox"], "mode": diag["mode"],
               "steps": diag["steps"], "iou": iou(ancien, nouveau)}

        for lbl, m, mode in (("TEINTE", m_a, "-"), ("CROISSANCE", m_n, diag["mode"])):
            if m is None:
                print(f"{nom:>28s} | {lbl:10s} {'--':>6s} {'--':>9s} {'--':>8s} "
                      f"{'--':>6s} {mode:>18s}   (aucun masque)")
                continue
            print(f"{nom:>28s} | {lbl:10s} {100*m['fill']:6.1f} {m['derive_px']:9.1f} "
                  f"{m['derive_pct']:8.1f} {m['asym_pct']:6.1f} {mode:>18s}")
            rec[lbl.lower()] = m
        if rec.get("iou") is not None:
            print(f"{'':>28s} | IoU entre les deux : {rec['iou']:.3f}")
        lignes.append(rec)
        print()

    # ── synthèse ─────────────────────────────────────────────────────────────
    def agg(key, champ):
        vals = [l[key][champ] for l in lignes if l.get(key)]
        return (statistics.median(vals), statistics.mean(vals), max(vals)) if vals else None

    print("=" * 78)
    print("SYNTHESE (mediane / moyenne / pire)")
    for champ, lbl in (("derive_pct", "derive du centre %"),
                       ("asym_pct", "asymetrie %"),
                       ("fill", "remplissage")):
        a, n = agg("teinte", champ), agg("croissance", champ)
        f = (lambda t: "  --  " if t is None else
             f"{t[0]:6.3f} / {t[1]:6.3f} / {t[2]:6.3f}")
        print(f"  {lbl:22s} TEINTE {f(a)}   CROISSANCE {f(n)}")

    modes = {}
    for l in lignes:
        modes[l["mode"]] = modes.get(l["mode"], 0) + 1
    print(f"\n  modes de croissance : {modes}")
    sans_a = sum(1 for l in lignes if not l.get("teinte"))
    sans_n = sum(1 for l in lignes if not l.get("croissance"))
    print(f"  cas sans masque : TEINTE {sans_a}/{len(lignes)}   "
          f"CROISSANCE {sans_n}/{len(lignes)}")

    out = Path("scratch/mask_growth_report.json")
    out.write_text(json.dumps(lignes, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
