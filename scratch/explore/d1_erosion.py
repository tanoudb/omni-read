import sys, os, pickle, glob, numpy as np, cv2
sys.path.insert(0,'.')
from pipeline import TranslationPipeline as TP
from core.renderer import TextRenderer
k7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))

CACHE='scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01'
d=pickle.load(open(CACHE+'/dets.pkl','rb')); it=d['items'][14]
page=cv2.imread(CACHE+'/page.png'); x1,y1,x2,y2=it['bbox']
crop=page[y1:y2,x1:x2]; h,w=crop.shape[:2]
gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
ocr=TP._ocr_mask_from_regions(it['text_regions'],h,w,crop_bgr=crop,dilate=3)
ink=ocr>0; n_ink=int(ink.sum())
sim=(np.abs(gray.astype(np.int16)-255.0)<40).astype(np.uint8)*255

def open_to_border(white):
    inv=(white==0).astype(np.uint8)
    nn,ll,ss,_=cv2.connectedComponentsWithStats(inv,4)
    lab_border=[l for l in range(1,nn) if ss[l,0]==0 or ss[l,1]==0 or ss[l,0]+ss[l,2]>=w or ss[l,1]+ss[l,3]>=h]
    return np.isin(ll,lab_border)

for name,white in [('similar BRUT (avant erosion)', sim), ('similar EROSION 7x7 (ligne 2877)', cv2.erode(sim,k7,1))]:
    t=open_to_border(white)
    print(f'{name:36s} : encre soudee a l exterieur = {int((ink&t).sum()):6d} / {n_ink}  ({(ink&t).sum()/n_ink:5.1%})')

# largeur du pont blanc entre les lettres de droite et l arche, sur similar brut
dist=cv2.distanceTransform((sim>0).astype(np.uint8),cv2.DIST_L2,5)
t_er=open_to_border(cv2.erode(sim,k7,1)); t_br=open_to_border(sim)
lost=ink & t_er & ~t_br     # encre qui bascule a cause de la seule erosion
if lost.sum():
    ys,xs=np.nonzero(lost)
    print(f'\nencre qui bascule DU FAIT DE L EROSION seule : {int(lost.sum())} px, x {xs.min()}..{xs.max()}')
# largeur mini du blanc separant l encre perdue de l exterieur
print('largeur du blanc (distance transform) le long du bord droit du ballon :')
for xx in range(380, w, 20):
    col=dist[:,xx]
    print(f'   x={xx:3d} max_dist_blanc={col.max():5.1f} px  (erosion 7x7 retire ~3.5 px de chaque cote)')

# flotte : effet de la suppression de l erosion
def bubble(crop_bgr, erode=True):
    hh,ww=crop_bgr.shape[:2]
    if hh<24 or ww<24: return None
    g=cv2.cvtColor(crop_bgr,cv2.COLOR_BGR2GRAY)
    ref=float(np.median(g[hh//3:2*hh//3, ww//3:2*ww//3]))
    s=(np.abs(g.astype(np.int16)-ref)<40).astype(np.uint8)*255
    m=cv2.erode(s,k7,1) if erode else s
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    if n<2: return None
    big=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
    mm=(lab==big).astype(np.uint8)*255
    if erode: mm=cv2.dilate(mm,k7,1)
    c,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not c: return None
    f=np.zeros_like(mm); cv2.drawContours(f,c,-1,255,-1)
    fill=float(np.count_nonzero(f))/(ww*hh)
    if not (0.20<=fill<=0.98): return None
    return f

a_cur=[];a_noer=[]
for cdir in sorted(glob.glob('scratch/bareme/cache/*')):
    p=os.path.join(cdir,'dets.pkl')
    if not os.path.exists(p): continue
    dd=pickle.load(open(p,'rb')); pg=cv2.imread(os.path.join(cdir,'page.png'))
    for i,jt in enumerate(dd['items']):
        if str(jt.get('class_name','')).lower()=='out_text': continue
        a,b,c2,d2=jt['bbox']; cr=pg[max(0,b):d2,max(0,a):c2]
        hh,ww=max(1,d2-b),max(1,c2-a)
        if cr.shape[:2]!=(hh,ww): continue
        om=TP._ocr_mask_from_regions(jt['text_regions'],hh,ww,crop_bgr=cr,dilate=3)
        if om is None: continue
        n0=int(np.count_nonzero(om))
        for lst,er in ((a_cur,True),(a_noer,False)):
            f=bubble(cr,er)
            if f is None or f.shape[:2]!=(hh,ww): lst.append(1.0); continue
            e=cv2.erode(f,k7,1)
            lst.append(int(np.count_nonzero(cv2.bitwise_and(om,e)))/max(1,n0))
a_cur=np.array(a_cur); a_noer=np.array(a_noer)
print(f'\nflotte ({len(a_cur)} bulles) : part de l encre conservee apres le garde-fou interior')
for nm,arr in (('erosion 7x7 (code actuel)',a_cur),('sans erosion',a_noer)):
    print(f'   {nm:26s} p5={np.percentile(arr,5):.3f} p10={np.percentile(arr,10):.3f} mediane={np.median(arr):.3f} | <0.90 : {int((arr<0.90).sum()):3d} | <0.70 : {int((arr<0.70).sum()):3d}')
