import sys, pickle, numpy as np, cv2
sys.path.insert(0,'.')
from pipeline import TranslationPipeline as TP
CACHE='scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01'
d=pickle.load(open(CACHE+'/dets.pkl','rb')); it=d['items'][14]
page=cv2.imread(CACHE+'/page.png'); x1,y1,x2,y2=it['bbox']
crop=page[y1:y2,x1:x2]; h,w=crop.shape[:2]
gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
ocr=TP._ocr_mask_from_regions(it['text_regions'],h,w,crop_bgr=crop,dilate=3); ink=ocr>0
sim=(np.abs(gray.astype(np.int16)-255.0)<40).astype(np.uint8)*255
def welded(white):
    inv=(white==0).astype(np.uint8)
    nn,ll,ss,_=cv2.connectedComponentsWithStats(inv,4)
    lb=[l for l in range(1,nn) if ss[l,0]==0 or ss[l,1]==0 or ss[l,0]+ss[l,2]>=w or ss[l,1]+ss[l,3]>=h]
    return int((ink & np.isin(ll,lb)).sum())
n=int(ink.sum())
print("taille du noyau d'erosion elliptique -> encre soudee a l'exterieur")
for ks in (1,3,5,7,9,11):
    m = sim if ks==1 else cv2.erode(sim,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ks,ks)),1)
    print(f'   {ks}x{ks} (retire ~{ks//2} px de blanc de chaque cote) : {welded(m):6d} px  ({welded(m)/n:5.1%})')
