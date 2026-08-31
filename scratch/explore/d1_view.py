import sys, pickle, numpy as np, cv2
sys.path.insert(0,'.')
from pipeline import TranslationPipeline as TP
from core.renderer import TextRenderer
CACHE='scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01'
d=pickle.load(open(CACHE+'/dets.pkl','rb')); it=d['items'][14]
page=cv2.imread(CACHE+'/page.png'); x1,y1,x2,y2=it['bbox']
crop=page[y1:y2,x1:x2]; h,w=crop.shape[:2]
gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
ocr=TP._ocr_mask_from_regions(it['text_regions'],h,w,crop_bgr=crop,dilate=3)
sim=(np.abs(gray.astype(np.int16)-255.0)<40).astype(np.uint8)*255
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
er=cv2.erode(sim,k,1)
n,lab,st,_=cv2.connectedComponentsWithStats(er,8)
big=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
cc=cv2.dilate((lab==big).astype(np.uint8)*255,k,1)
cnts,_=cv2.findContours(cc,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
filled=np.zeros_like(cc); cv2.drawContours(filled,cnts,-1,255,-1)
inte=cv2.erode(filled,k,1)
lost=cv2.bitwise_and(ocr,cv2.bitwise_not(inte))
kept=cv2.bitwise_and(ocr,inte)
def bgr(m,c):
    o=np.zeros((h,w,3),np.uint8); o[m>0]=c; return o
panels=[crop, cv2.cvtColor(sim,cv2.COLOR_GRAY2BGR), cv2.cvtColor(cc,cv2.COLOR_GRAY2BGR),
        cv2.cvtColor(filled,cv2.COLOR_GRAY2BGR), bgr(kept,(0,255,0))+bgr(lost,(0,0,255))]
labels=['crop','similar(|g-255|<40)','plus grande CC (dilatee)','contours RETR_EXTERNAL remplis','vert=efface / rouge=survit']
out=[]
for p,l in zip(panels,labels):
    p=p.copy(); cv2.putText(p,l,(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    out.append(p)
cv2.imwrite('scratch/explore/d1_overview.png', np.vstack(out))
print('nb contours externes =', len(cnts), 'aires =', sorted(int(cv2.contourArea(c)) for c in cnts)[-3:])
print('ecrit')
