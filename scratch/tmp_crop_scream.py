# -*- coding: utf-8 -*-
"""Decoupe une tranche autour des bulles #05/#06 (POV_V2) pour repro isolee."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"A:\omni read")))
import cv2

SRC = Path(r"A:\omni read\manhwa\path-of-vengeance\Chapitre 001\Chapitre 001_merged_part01.jpg")
OUT = Path(r"A:\omni read\scratch\render_out\scream_slice.png")

img = cv2.imread(str(SRC))
h, w = img.shape[:2]
print("source size", w, h)

# bbox #05: [497,5252,682,5546]  bbox #06: [399,5440,654,5793]
x1, y1, x2, y2 = 399, 5252, 682, 5793
margin = 300
cx1, cy1 = max(0, x1 - margin), max(0, y1 - margin)
cx2, cy2 = min(w, x2 + margin), min(h, y2 + margin)
crop = img[cy1:cy2, cx1:cx2]
print("crop box (orig coords):", cx1, cy1, cx2, cy2, "-> size", crop.shape[1], crop.shape[0])
cv2.imwrite(str(OUT), crop)
print("wrote", OUT)
