import sys, os, pickle, glob, numpy as np, cv2
sys.path.insert(0,'.')
from pipeline import TranslationPipeline as TP
from core.renderer import TextRenderer

rows = []
mismatch = 0
for cdir in sorted(glob.glob('scratch/bareme/cache/*')):
    pkl = os.path.join(cdir,'dets.pkl')
    if not os.path.exists(pkl): continue
    d = pickle.load(open(pkl,'rb'))
    page = cv2.imread(os.path.join(cdir,'page.png'))
    for idx, it in enumerate(d['items']):
        x1,y1,x2,y2 = it['bbox']
        crop = page[max(0,y1):y2, max(0,x1):x2]
        h,w = max(1,y2-y1), max(1,x2-x1)
        if crop.shape[:2] != (h,w): continue
        ocr = TP._ocr_mask_from_regions(it['text_regions'], h, w, crop_bgr=crop, dilate=3)
        if ocr is None: continue
        n0 = int(np.count_nonzero(ocr))
        m = ocr
        r1 = None; used1 = False
        if str(it.get('class_name','')).lower() != 'out_text':
            try: interior = TextRenderer._bubble_mask_from_image(crop)
            except Exception: interior = None
            if interior is not None and interior.shape[:2]==(h,w):
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
                inte = cv2.erode(interior,k,1)
                b = cv2.bitwise_and(m,inte)
                r1 = int(np.count_nonzero(b))/max(1,n0)
                if int(np.count_nonzero(b)) >= 0.5*n0:
                    m = b; used1 = True
        n1 = int(np.count_nonzero(m))
        bubble = it.get('mask_binary'); used2=False; r2=None
        if bubble is not None:
            if getattr(bubble,'ndim',0)==3: bubble = bubble[:,:,0]
            if bubble.shape[:2]!=(h,w):
                bubble = cv2.resize(bubble,(w,h),interpolation=cv2.INTER_NEAREST)
            bubble = (bubble>0).astype(np.uint8)*255
            stroke = max(3,int(round(min(h,w)*0.025)))
            ks = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*stroke+1,2*stroke+1))
            er = cv2.erode(bubble,ks,1)
            if int(np.count_nonzero(er)) >= 0.35*int(np.count_nonzero(bubble)): bubble = er
            inter = cv2.bitwise_and(m,bubble)
            r2 = float(np.sum(inter))/max(1.0,float(np.sum(m)))
            if float(np.sum(inter)) > 0.30*float(np.sum(m)): m = inter; used2=True
        n2 = int(np.count_nonzero(m))
        final = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        cm = it.get('chirurgical_mask')
        if cm is not None and not np.array_equal(final, cm): mismatch += 1
        cov = int(np.count_nonzero(cv2.bitwise_and(ocr, final)))/max(1,n0)
        rows.append(dict(series=os.path.basename(cdir), idx=idx, cls=it.get('class_name'),
                         n0=n0, r1=r1, used1=used1, n1=n1, r2=r2, used2=used2, n2=n2, cov=cov))

print('items analyses :', len(rows), ' mismatch vs cache :', mismatch)
covs = np.array([r['cov'] for r in rows])
print('couverture de l encre OCR par chirurgical_mask :')
for t in (0.99,0.95,0.90,0.80,0.70,0.60,0.50):
    print(f'   >= {t:.2f} : {int((covs>=t).sum()):4d} / {len(covs)}   (< {t:.2f} : {int((covs<t).sum())})')
# attribution des pertes
bad = [r for r in rows if r['cov'] < 0.90]
print('\n', len(bad), 'detections ou > 10% de l encre n est PAS effacee')
from collections import Counter
cause = Counter()
for r in bad:
    if r['used1'] and r['n1'] < 0.9*r['n0']: cause['garde-fou INTERIOR (pipeline.py:711-726)'] += 1
    elif r['used2'] and r['n2'] < 0.9*r['n1']: cause['intersection mask_binary (pipeline.py:744-762)'] += 1
    else: cause['autre'] += 1
for k,v in cause.most_common(): print('   ', k, v)
print('\n classes concernees :', Counter(r['cls'] for r in bad).most_common())
# marge du garde-fou 0.5
near = [r for r in rows if r['r1'] is not None and 0.5 <= r['r1'] < 0.65]
print(f'\n detections dont le ratio interior tombe dans [0.50, 0.65] (guard applique de justesse) : {len(near)}')
for r in sorted(near, key=lambda r: r['r1'])[:15]:
    print(f"   {r['series'][:42]:44s} #{r['idx']:3d} {str(r['cls']):8s} ratio={r['r1']:.3f} cov={r['cov']:.3f}")
