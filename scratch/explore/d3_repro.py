"""D3 — reproduire chirurgical_mask a partir des polygones + identifier l'etape
qui tronque. Aucun GPU : uniquement les pickles + deux @staticmethod."""
import sys, pickle, os
sys.path.insert(0, '.')
import numpy as np
import cv2
from pipeline import TranslationPipeline
from core.renderer import TextRenderer

CACHE = 'scratch/bareme/cache'


def load(page):
    d = pickle.load(open(os.path.join(CACHE, page, 'dets.pkl'), 'rb'))
    img = cv2.imread(os.path.join(CACHE, page, 'page.png'))
    return d, img


def nz(m):
    return 0 if m is None else int(np.count_nonzero(m))


def rebuild_steps(img, it):
    """Rejoue _build_masks_for_detection etape par etape."""
    x1, y1, x2, y2 = it['bbox']
    h, w = max(1, y2 - y1), max(1, x2 - x1)
    crop = img[max(0, y1):y2, max(0, x1):x2]
    ocr = TranslationPipeline._ocr_mask_from_regions(
        it['text_regions'], h, w, crop_bgr=crop)
    steps = {'ocr': ocr}
    if ocr is None:
        return steps
    cur = ocr
    # garde-fou 1 : interieur deduit de l'image
    if str(it['class_name']).lower() != 'out_text':
        try:
            interior = TextRenderer._bubble_mask_from_image(crop)
        except Exception:
            interior = None
        if interior is not None and interior.shape[:2] == (h, w):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            interior = cv2.erode(interior, k, iterations=1)
            bounded = cv2.bitwise_and(cur, interior)
            steps['interior'] = interior
            steps['after_interior_candidate'] = bounded
            steps['interior_applied'] = nz(bounded) >= 0.5 * nz(cur)
            if steps['interior_applied']:
                cur = bounded
    # garde-fou 2 : mask_binary erode
    bubble = it.get('mask_binary')
    if bubble is not None:
        if getattr(bubble, 'ndim', 0) == 3:
            bubble = bubble[:, :, 0]
        if bubble.shape[:2] != (h, w):
            bubble = cv2.resize(bubble, (w, h), interpolation=cv2.INTER_NEAREST)
        bubble = (bubble > 0).astype(np.uint8) * 255
        stroke = max(3, int(round(min(h, w) * 0.025)))
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * stroke + 1, 2 * stroke + 1))
        eroded = cv2.erode(bubble, kk, iterations=1)
        if nz(eroded) >= 0.35 * nz(bubble):
            bubble = eroded
        inter = cv2.bitwise_and(cur, bubble)
        steps['bubble'] = bubble
        steps['after_bubble_candidate'] = inter
        steps['bubble_applied'] = int(np.sum(inter)) > 0.30 * int(np.sum(cur))
        if steps['bubble_applied']:
            cur = inter
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    steps['final'] = cv2.morphologyEx(cur, cv2.MORPH_CLOSE, kc)
    return steps


if __name__ == '__main__':
    page = '30-years-have-passed-since-the-prologue__p01'
    d, img = load(page)
    it = d['items'][14]
    x1, y1, x2, y2 = it['bbox']
    print('bbox', it['bbox'], 'class', it['class_name'])
    st = rebuild_steps(img, it)
    cached = it['chirurgical_mask']
    print('ocr           nz=%d' % nz(st['ocr']))
    print('interior_applied', st.get('interior_applied'),
          'candidate nz=%d (%.0f%% de ocr)' % (
              nz(st.get('after_interior_candidate')),
              100.0 * nz(st.get('after_interior_candidate')) / max(1, nz(st['ocr']))))
    if 'after_bubble_candidate' in st:
        base = nz(st['after_interior_candidate']) if st.get('interior_applied') else nz(st['ocr'])
        print('bubble_applied  ', st.get('bubble_applied'),
              'candidate nz=%d (%.0f%% du courant)' % (
                  nz(st['after_bubble_candidate']),
                  100.0 * nz(st['after_bubble_candidate']) / max(1, base)))
    print('final rebuilt nz=%d   cache nz=%d' % (nz(st['final']), nz(cached)))
    same = st['final'].shape == cached.shape and np.array_equal(st['final'] > 0, cached > 0)
    print('IDENTIQUE AU CACHE :', same)

    # profil en x
    def prof(m, name):
        cols = (m > 0).sum(axis=0)
        chunks = [int(cols[i:i + 100].sum()) for i in range(0, m.shape[1], 100)]
        xs = np.where(cols > 0)[0]
        print('%-22s xmax=%s  profil/100px %s' % (name, xs.max() if xs.size else None, chunks))
    prof(st['ocr'], 'ocr(polygones)')
    if 'interior' in st:
        prof(st['interior'], 'interior(erode)')
        prof(st['after_interior_candidate'], 'ocr&interior')
    if 'bubble' in st:
        prof(st['bubble'], 'mask_binary(erode)')
        prof(st['after_bubble_candidate'], 'apres bubble')
    prof(cached, 'chirurgical(cache)')
