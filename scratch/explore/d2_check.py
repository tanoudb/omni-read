import sys, pickle, os
sys.path.insert(0, '.')
import numpy as np, cv2
from scratch.explore.d2_ink import ink_mask, cov
from pipeline import TranslationPipeline as TP

OUT = 'scratch/explore/d2_vis'
os.makedirs(OUT, exist_ok=True)

CASES = [
    ('30-years-have-passed-since-the-prologue__p01', 14),
]
if len(sys.argv) > 1:
    CASES = [(sys.argv[i], int(sys.argv[i+1])) for i in range(1, len(sys.argv), 2)]

for key, idx in CASES:
    d = pickle.load(open(f'scratch/bareme/cache/{key}/dets.pkl', 'rb'))
    page = cv2.imread(f'scratch/bareme/cache/{key}/page.png', cv2.IMREAD_COLOR)
    it = d['items'][idx]
    x1, y1, x2, y2 = it['bbox']
    crop = page[max(0, y1):y2, max(0, x1):x2]
    h, w = crop.shape[:2]
    dbg = {}
    ink, env = ink_mask(crop, it['text_regions'], h, w, dbg)
    ocr = TP._ocr_mask_from_regions(it['text_regions'], h, w, crop_bgr=crop, dilate=3)
    chir = it.get('chirurgical_mask')
    print(f'{key} #{idx} {it["class_name"]} {w}x{h} lineh={dbg["line_h"]:.0f} k={dbg["k"]} thr={dbg["thr"]:.1f}')
    print('  ink px', int(np.count_nonzero(ink)), 'env px', int(np.count_nonzero(env)),
          'ink/env %.3f' % (np.count_nonzero(ink)/max(1, np.count_nonzero(env))))
    print('  cov_ocr', cov(ink, ocr), ' cov_chir', cov(ink, chir))

    def ov(base, m, color):
        o = base.copy()
        if m is not None:
            o[m > 0] = (0.45*o[m > 0] + 0.55*np.array(color)).astype(np.uint8)
        return o
    panels = [crop, ov(crop, ink, (0,0,255)), ov(crop, ocr, (0,255,0)), ov(crop, chir, (255,0,0))]
    sep = np.full((h, 6, 3), 255, np.uint8)
    strip = panels[0]
    for p in panels[1:]:
        strip = np.hstack([strip, sep, p])
    cv2.imwrite(f'{OUT}/{key}_{idx}.png', strip)
    print('  ->', f'{OUT}/{key}_{idx}.png')
