# -*- coding: utf-8 -*-
"""Inspecte directement les polygones OCR bruts pour un bbox donné,
sans refaire tout le pipeline (debug rapide)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(r"A:\omni read")))

import cv2
import json
from config import config
from core import OCREngine

img_path = r"A:\omni read\manhwa\path-of-vengeance\Chapitre 001\Chapitre 001_merged_part01.jpg"
img = cv2.imread(img_path)

# bbox de "EIGHT YEARS AGO" (index 29 dans le run precedent)
x1, y1, x2, y2 = 254, 29865, 449, 30163
crop = img[y1:y2, x1:x2]
cv2.imwrite(str(Path(r"A:\omni read\scratch\render_out\debug_eight_years_crop_raw.png")), crop)
print("crop shape:", crop.shape)

ocr = OCREngine(device="cuda")
results = ocr.extract_batch([crop])
text, confidence, is_valid, skip_reason, text_regions, upscale_factor = results[0]
print("text:", text)
print("confidence:", confidence)
print("is_valid:", is_valid)
print("upscale_factor:", upscale_factor)
print("n_regions:", len(text_regions or []))
for i, r in enumerate(text_regions or []):
    print(f"region {i}: text={r.get('text')!r} bbox={r.get('bbox')}")
