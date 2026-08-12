"""
Le rendu ne doit pas dépendre de l'étiquette de classe donnée par YOLO.

Mesuré sur une planche : selon le cadrage des fenêtres glissantes, la même
image donne « 32 bulle + 3 out_text + 1 System » en tranches et « 36 bulle »
en pleine hauteur. Composer d'après le libellé revenait donc à traiter une
boîte de narration comme un ovale une fois sur deux.

La mise en page se décide sur la forme MESURÉE (contenant uni englobant,
remplissage du rectangle englobant). `System` fait exception : sa police et son
ancrage sont un choix de style, pas une déduction de forme.

    python tests/test_label_independence.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from core import TextRenderer

PLANCHE = Path("manhwa/rise-of-the-dragon-overlord/Chapitre 001/Chapitre 001_merged_part01.jpg")

# (nom, x1, y1, x2, y2) — zones réelles de la planche
CAS = [
    ("boite narration", 122, 8850 + 249, 413, 8850 + 369),
    ("bulle ronde", 42, 33050 + 817, 627, 33050 + 1272),
    ("bulle de cri", 53, 24350 + 200, 442, 24350 + 522),
]

TEXTE = "Tout le monde chasse des monstres pour gagner des points d'experience."


def main() -> int:
    if not PLANCHE.exists():
        print(f"planche absente ({PLANCHE}) — test ignoré")
        return 0

    img = cv2.imread(str(PLANCHE))
    if img is None:
        print("planche illisible — test ignoré")
        return 0

    renderer = TextRenderer()
    echecs = 0

    for nom, x1, y1, x2, y2 in CAS:
        rendus = {}
        for cls in ("bulle", "out_text"):
            out = renderer.insert_text(
                img.copy(), TEXTE, x1, y1, x2, y2,
                class_name=cls, source_line_height=22.0,
            )
            rendus[cls] = out[max(0, y1 - 200):y2 + 200, :]

        ecart = int(np.count_nonzero(np.any(rendus["bulle"] != rendus["out_text"], axis=2)))
        ok = ecart == 0
        echecs += 0 if ok else 1
        print(f"  {'OK ' if ok else 'ECHEC'} {nom:18s} ecart bulle/out_text : {ecart} pixels")

    print(f"\n{len(CAS) - echecs} OK / {echecs} echecs")
    return 1 if echecs else 0


if __name__ == "__main__":
    raise SystemExit(main())
