import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
from core.renderer import TextRenderer
from core.detector import YOLODetector
from config import config

def test_image(img_path):
    print(f'\n--- Testing {img_path} ---')
    img = cv2.imread(img_path)
    detector = YOLODetector(config.YOLO_MODEL_PATH, 'cpu')
    dets = detector.get_translatable_detections(detector.detect(img))
    
    # We need the actual mask. We can simulate it by finding dark pixels inside the bbox.
    for i, d in enumerate(dets):
        x1, y1, x2, y2 = map(int, d.bbox)
        m = TextRenderer.CROP_MARGIN
        crop_x1 = max(0, x1 - m)
        crop_y1 = max(0, y1 - m)
        crop_x2 = min(img.shape[1], x2 + m)
        crop_y2 = min(img.shape[0], y2 + m)
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
        
        # Simple text mask approximation: thresholding
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # text is usually dark
        _, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        # dilate a bit to simulate OCR mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, kernel)
        
        # Now run flat fill color logic from original
        try:
            ring = (cv2.dilate(mask, kernel, iterations=1) > 0) & (mask == 0)
            samples = crop[ring]
            if samples.shape[0] < 64:
                continue
            samples = samples.reshape(-1, 3)
            reference = np.median(samples.astype(np.float32), axis=0)
            deviation = np.abs(samples.astype(np.float32) - reference).max(axis=1)
            
            pass12 = float(np.mean(deviation <= 12.0)) >= 0.85
            pass3 = float(np.mean(deviation <= 3.0)) >= 0.85
            
            inliers = samples[deviation <= 12.0]
            if inliers.shape[0] < 32:
                inliers = samples
            mode = np.array([
                int(np.bincount(inliers[:, c], minlength=256).argmax())
                for c in range(3)
            ], dtype=np.uint8)
            
            print(f"Box {i} | text: '{d.text[:15]}' | mode: {mode} | pass12: {pass12} | pass3: {pass3}")
        except Exception as e:
            pass

test_image(r'manhwa\hellogin\Chapitre 001\Chapitre 001_merged_part01.jpg')
test_image(r'manhwa\path-of-vengeance\Chapitre 001\Chapitre 001_merged_part01.jpg')
