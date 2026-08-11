"""Vérifie que le texte reste dans la bulle et que les lignes ne se chevauchent pas."""
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent))
import cv2, numpy as np
from core.renderer import TextRenderer

r = TextRenderer()
out_dir = str(__import__('pathlib').Path(__file__).resolve().parent)

CASES = [
    # (nom, largeur bbox, hauteur bbox, texte, classe)
    ("bulle_ronde_longue", 300, 300,
     "Pour survivre dans ce monde, et pour offrir une vie meilleure a ma tante...", "bulle"),
    ("bulle_ronde_courte", 300, 300, "Enfin...", "bulle"),
    ("petite_boite", 215, 90, "Des royaumes secrets...", "out_text"),
    ("boite_narration", 470, 130,
     "Tout le monde chasse des monstres pour gagner des points d'experience et des materiaux "
     "afin de fabriquer de nouveaux equipements et armes.", "out_text"),
    ("accents", 300, 300, "Meme le monstre le plus feroce pale en comparaison de cet etre.", "bulle"),
]

panels = []
for name, bw, bh, text, cls in CASES:
    pad = 40
    img = np.full((bh + 2 * pad, bw + 2 * pad, 3), 40, np.uint8)
    x1, y1, x2, y2 = pad, pad, pad + bw, pad + bh

    mask = np.zeros((bh, bw), np.uint8)
    if cls == "bulle":
        cv2.ellipse(mask, (bw // 2, bh // 2), (bw // 2 - 2, bh // 2 - 2), 0, 0, 360, 255, -1)
    else:
        mask[:] = 255
    img[y1:y2, x1:x2][mask > 0] = (255, 255, 255)

    before = img.copy()
    out = r.insert_text(img.copy(), text, x1, y1, x2, y2,
                        text_regions=None, class_name=cls, bubble_mask=mask,
                        source_line_height=None)

    # pixels d'encre = pixels modifies
    diff = np.any(np.abs(out.astype(int) - before.astype(int)) > 30, axis=2)
    ys, xs = np.nonzero(diff)
    if xs.size == 0:
        print(f"{name:22s} AUCUN TEXTE RENDU")
        panels.append(out); continue

    # combien d'encre tombe hors de la bulle ?
    full_mask = np.zeros(img.shape[:2], np.uint8)
    full_mask[y1:y2, x1:x2] = mask
    outside = int(np.sum(diff & (full_mask == 0)))
    total = int(np.sum(diff))
    print(f"{name:22s} encre={total:6d}  hors-bulle={outside:5d} ({100*outside/total:5.1f}%)  "
          f"bbox_texte=({xs.min()},{ys.min()})-({xs.max()},{ys.max()})")
    panels.append(out)

h = max(p.shape[0] for p in panels)
row = []
for p in panels:
    canvas = np.full((h, p.shape[1], 3), 20, np.uint8)
    canvas[:p.shape[0]] = p
    row.append(canvas)
    row.append(np.full((h, 6, 3), 0, np.uint8))
cv2.imwrite(out_dir + r'\test_render.png', np.hstack(row))
print('-> test_render.png')
