import sys, pickle, numpy as np, cv2
sys.path.insert(0,'.')
from pipeline import TranslationPipeline as TP
CACHE='scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01'
d=pickle.load(open(CACHE+'/dets.pkl','rb')); it=d['items'][14]
page=cv2.imread(CACHE+'/page.png'); x1,y1,x2,y2=it['bbox']
crop=page[y1:y2,x1:x2]; h,w=crop.shape[:2]
gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
sim=(np.abs(gray.astype(np.int16)-255.0)<40).astype(np.uint8)
ocr=TP._ocr_mask_from_regions(it['text_regions'],h,w,crop_bgr=crop,dilate=3)
dist=cv2.distanceTransform(sim,cv2.DIST_L2,5)
# halo blanc autour des lettres, uniquement la ou l encre est perdue (x>460)
ring=cv2.dilate((ocr>0).astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))-(ocr>0).astype(np.uint8)
for lo,hi,lbl in [(120,300,'gauche (encre conservee)'),(460,587,'droite (encre perdue)')]:
    sel=(ring[:,lo:hi]>0)&(sim[:,lo:hi]>0)
    v=dist[:,lo:hi][sel]
    if v.size:
        print(f'{lbl:28s} : epaisseur du blanc autour des lettres  mediane={np.median(v):.1f} px  p90={np.percentile(v,90):.1f} px  n={v.size}')
