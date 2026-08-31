import sys, pickle, numpy as np, cv2
sys.path.insert(0, '.')
from pipeline import TranslationPipeline as TP
from core.renderer import TextRenderer

CACHE = 'scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01'
d = pickle.load(open(CACHE + '/dets.pkl', 'rb'))
it = d['items'][14]
page = cv2.imread(CACHE + '/page.png')
x1, y1, x2, y2 = it['bbox']
crop = page[y1:y2, x1:x2]
h, w = y2 - y1, x2 - x1
print('crop', crop.shape, 'class', it['class_name'])

def prof(m, label, bands=5):
    if m is None:
        print(f'{label:34s} None'); return
    b = w / bands
    cols = [int(np.count_nonzero(m[:, int(i*b):int((i+1)*b)])) for i in range(bands)]
    print(f'{label:34s} total={int(np.count_nonzero(m)):6d}  bands={cols}')

# --- etape 0 : masque OCR complet (reference) ---
ocr = TP._ocr_mask_from_regions(it['text_regions'], h, w, crop_bgr=crop, dilate=3)
prof(ocr, '0. ocr_mask (polygones+encre)')

cm = it['chirurgical_mask']
prof(cm, 'REF chirurgical_mask (cache)')
mb = it['mask_binary']
if mb is not None and mb.ndim == 3: mb = mb[:, :, 0]
prof(mb, 'REF mask_binary (segmenter)')

# --- etape 1 : garde-fou interior (_bubble_mask_from_image) ---
interior = TextRenderer._bubble_mask_from_image(crop)
prof(interior, '1a. interior brut')
if interior is not None and interior.shape[:2] == (h, w):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inte = cv2.erode(interior, k, 1)
    prof(inte, '1b. interior erode 7x7')
    bounded = cv2.bitwise_and(ocr, inte)
    prof(bounded, '1c. ocr & interior')
    ratio = np.count_nonzero(bounded) / max(1, np.count_nonzero(ocr))
    print(f'    garde-fou 0.5 : ratio={ratio:.3f} -> {"BOUNDED RETENU" if ratio>=0.5 else "rejete"}')
    if ratio >= 0.5:
        ocr = bounded
prof(ocr, '1. apres interior')

# --- etape 2 : intersection mask_binary ---
bubble = mb
if bubble is not None:
    if bubble.shape[:2] != (h, w):
        bubble = cv2.resize(bubble, (w, h), interpolation=cv2.INTER_NEAREST)
    bubble = (bubble > 0).astype(np.uint8) * 255
    stroke = max(3, int(round(min(h, w) * 0.025)))
    ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*stroke+1, 2*stroke+1))
    er = cv2.erode(bubble, ks, 1)
    print(f'    stroke={stroke} erode keep={np.count_nonzero(er)/max(1,np.count_nonzero(bubble)):.3f}')
    if np.count_nonzero(er) >= 0.35 * np.count_nonzero(bubble):
        bubble = er
    inter = cv2.bitwise_and(ocr, bubble)
    r2 = np.sum(inter) / max(1, np.sum(ocr))
    print(f'    garde-fou 0.30 : ratio={r2:.3f} -> {"INTER RETENU" if r2>0.30 else "rejete"}')
    if r2 > 0.30:
        ocr = inter
prof(ocr, '2. apres mask_binary')

kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
final = cv2.morphologyEx(ocr, cv2.MORPH_CLOSE, kc)
prof(final, '3. final (close 3x3)')
print('identique au cache ?', bool(np.array_equal(final, cm)))

cv2.imwrite('scratch/explore/d1_crop.png', crop)
for name, m in [('ocr0', TP._ocr_mask_from_regions(it['text_regions'], h, w, crop_bgr=crop, dilate=3)),
                ('interior', interior), ('maskbin', mb), ('cm', cm), ('final', final)]:
    if m is not None:
        cv2.imwrite(f'scratch/explore/d1_{name}.png', m)
