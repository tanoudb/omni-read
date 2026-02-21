from pathlib import Path
import cv2
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.ocr import OCREngine

base = Path(r"a:/omni read/output/debug/Chapitre 001_merged_part01_pipeline")
crop_paths = sorted(base.glob("*_crop.png"))[:12]

engine = OCREngine(device="cuda")

images = []
for p in crop_paths:
    img = cv2.imread(str(p))
    if img is not None:
        images.append((p, img))

print(f"loaded_crops={len(images)}")

results = engine.extract_batch([img for _, img in images], debug_hook=lambda msg: print(msg))

for (path, _), (text, conf, ok, reason, _regions, up) in zip(images, results):
    name = path.name
    txt = (text or "").replace("\n", " ")
    print(f"{name}\tok={ok}\tconf={conf:.3f}\tup={up:.2f}\treason={reason}\ttext={txt[:120]}")
