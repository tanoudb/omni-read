import sys, os, glob, pickle, json, time
sys.path.insert(0, '.')
import numpy as np, cv2
from scratch.explore.d2_ink import ink_mask
from pipeline import TranslationPipeline as TP

OPEN_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

def cover(ink, mask):
    tot = int(np.count_nonzero(ink))
    if mask is None:
        miss = ink
    else:
        m = mask
        if m.shape[:2] != ink.shape[:2]:
            m = cv2.resize(m, (ink.shape[1], ink.shape[0]), interpolation=cv2.INTER_NEAREST)
        miss = ((ink > 0) & ~(m > 0)).astype(np.uint8) * 255
    covd = 1.0 - np.count_nonzero(miss) / tot
    # residu LISIBLE : ce qui survit a une ouverture 5x5 = des corps de
    # glyphes, pas la frange d'anticrenelage de 1 px.
    res = cv2.morphologyEx(miss, cv2.MORPH_OPEN, OPEN_K)
    return covd, float(np.count_nonzero(res)) / tot

rows = []
t0 = time.time()
for pkl in sorted(glob.glob('scratch/bareme/cache/*/dets.pkl')):
    key = os.path.basename(os.path.dirname(pkl))
    d = pickle.load(open(pkl, 'rb'))
    page = cv2.imread(os.path.join(os.path.dirname(pkl), 'page.png'), cv2.IMREAD_COLOR)
    for i, it in enumerate(d['items']):
        x1, y1, x2, y2 = [int(v) for v in it['bbox']]
        crop = page[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue
        h, w = crop.shape[:2]
        dbg = {}
        try:
            ink, env = ink_mask(crop, it['text_regions'], h, w, dbg)
        except Exception as e:
            rows.append(dict(key=key, i=i, err=repr(e))); continue
        if ink is None or int(np.count_nonzero(ink)) < 30:
            rows.append(dict(key=key, i=i, skip='ink<30',
                             ink=int(np.count_nonzero(ink)) if ink is not None else 0)); continue
        try:
            ocr = TP._ocr_mask_from_regions(it['text_regions'], h, w, crop_bgr=crop, dilate=3)
        except Exception:
            ocr = None
        c_ocr, r_ocr = cover(ink, ocr)
        c_chir, r_chir = cover(ink, it.get('chirurgical_mask'))
        rows.append(dict(
            key=key, i=i, cls=it['class_name'], text=(it.get('text') or '')[:90],
            w=w, h=h, ink=int(np.count_nonzero(ink)),
            env=int(np.count_nonzero(env)), line_h=round(dbg['line_h'], 1),
            thr=round(dbg['thr'], 1), nlines=len(it['text_regions'] or []),
            cov_ocr=round(c_ocr, 4), cov_chir=round(c_chir, 4),
            res_ocr=round(r_ocr, 4), res_chir=round(r_chir, 4),
            chir_px=int(np.count_nonzero(it['chirurgical_mask'])) if it.get('chirurgical_mask') is not None else 0,
        ))
    del page
    print(key, len(rows), '%.0fs' % (time.time() - t0), flush=True)

json.dump(rows, open('scratch/explore/d2_rows.json', 'w'), ensure_ascii=False)
print('rows', len(rows))
