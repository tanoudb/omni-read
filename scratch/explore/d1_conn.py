import sys, pickle, numpy as np, cv2
sys.path.insert(0,'.')
from pipeline import TranslationPipeline as TP
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
# le complement du ballon : composantes NOIRES
inv=(cc==0).astype(np.uint8)
nn,ll,ss,_=cv2.connectedComponentsWithStats(inv,4)
border=set()
for l2 in range(1,nn):
    x,y,bw,bh,a=ss[l2]
    if x==0 or y==0 or x+bw>=w or y+bh>=h: border.add(l2)
touch=np.isin(ll,list(border))
ink=ocr>0
n_ink=int(ink.sum())
print(f'encre totale                                  = {n_ink}')
print(f'encre dans un TROU FERME du ballon (=remplie) = {int((ink & ~touch & (cc==0)).sum())}')
print(f'encre dans une composante noire OUVERTE sur le bord (=perdue) = {int((ink & touch).sum())}  ({(ink&touch).sum()/n_ink:.1%})')
print(f'encre deja dans le blanc du ballon            = {int((ink & (cc>0)).sum())}')
# la grande composante noire ouverte : est-ce l arche + les lettres soudees ?
areas=[(int(ss[l2,4]),l2) for l2 in border]
areas.sort(reverse=True)
print('\ncomposantes noires ouvertes sur le bord, 3 plus grandes (aire, bbox x,y,w,h) :')
for a,l2 in areas[:3]:
    print('   aire=%7d  bbox=%s  encre dedans=%d' % (a, tuple(ss[l2,:4]), int((ink & (ll==l2)).sum())))
