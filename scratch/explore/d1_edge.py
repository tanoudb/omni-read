import sys, os, pickle, glob, numpy as np, cv2
sys.path.insert(0,'.')
from pipeline import TranslationPipeline as TP
from core.renderer import TextRenderer
CACHE='scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01'
d=pickle.load(open(CACHE+'/dets.pkl','rb')); it=d['items'][14]
page=cv2.imread(CACHE+'/page.png'); x1,y1,x2,y2=it['bbox']
crop=page[y1:y2,x1:x2]; h,w=crop.shape[:2]
gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
ocr=TP._ocr_mask_from_regions(it['text_regions'],h,w,crop_bgr=crop,dilate=3)
sim=(np.abs(gray.astype(np.int16)-255.0)<40)
print("fond (p75 de la colonne, = luminance du fond) et part de colonne 'similaire au ref' :")
for xx in range(0,w,25):
    col=gray[:,xx:xx+25]
    print(f'  x={xx:3d} fond_p75={int(np.percentile(col,75)):3d}  frac_similar={sim[:,xx:xx+25].mean():.2f}')
# derniere colonne ou le fond reste dans la fenetre +/-40
bg=np.percentile(gray,75,axis=0)
inwin=np.where(bg>=215)[0]
print('\nderniere colonne dont le fond est >=215 (255-40) :', inwin.max() if len(inwin) else None)
print('derniere colonne non vide du chirurgical_mask :', np.nonzero(it["chirurgical_mask"])[1].max())
print('derniere colonne non vide de l encre OCR      :', np.nonzero(ocr)[1].max())
