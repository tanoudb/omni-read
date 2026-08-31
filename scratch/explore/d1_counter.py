import sys, os, pickle, glob, numpy as np, cv2
sys.path.insert(0,'.')
from pipeline import TranslationPipeline as TP
from core.renderer import TextRenderer
CACHE='scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01'
d=pickle.load(open(CACHE+'/dets.pkl','rb')); it=d['items'][14]
page=cv2.imread(CACHE+'/page.png'); x1,y1,x2,y2=it['bbox']
crop=page[y1:y2,x1:x2]; h,w=crop.shape[:2]
ocr=TP._ocr_mask_from_regions(it['text_regions'],h,w,crop_bgr=crop,dilate=3)
interior=TextRenderer._bubble_mask_from_image(crop)
k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
inte=cv2.erode(interior,k,1)
dropped=cv2.bitwise_and(ocr,cv2.bitwise_not(inte))
# bande "trait de contour" = couronne autour de la frontiere du ballon
band=cv2.bitwise_xor(cv2.dilate(interior,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)),1),
                     cv2.erode(interior,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)),1))
nd=int(np.count_nonzero(dropped))
print(f'#14 : encre retiree par le garde-fou interior = {nd} px')
print(f'      dont dans la couronne de +/-7px autour du bord du ballon (=trait a proteger) = {int(np.count_nonzero(cv2.bitwise_and(dropped,band)))}')
print(f'      soit {int(np.count_nonzero(cv2.bitwise_and(dropped,band)))/nd:.1%} : le reste ({nd-int(np.count_nonzero(cv2.bitwise_and(dropped,band)))} px) est du GLYPHE pur')

# effet des variantes de _bubble_mask_from_image sur la couverture de l encre
gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
def variant(tol=40, erode=True, closing=0):
    ref=float(np.median(gray[h//3:2*h//3, w//3:2*w//3]))
    sim=(np.abs(gray.astype(np.int16)-ref)<tol).astype(np.uint8)*255
    if closing: sim=cv2.morphologyEx(sim,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(closing,closing)))
    m=cv2.erode(sim,k,1) if erode else sim
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    if n<2: return None
    big=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
    mm=(lab==big).astype(np.uint8)*255
    if erode: mm=cv2.dilate(mm,k,1)
    c,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    f=np.zeros_like(mm); cv2.drawContours(f,c,-1,255,-1)
    fill=np.count_nonzero(f)/(w*h)
    cov=np.count_nonzero(cv2.bitwise_and(ocr,cv2.erode(f,k,1)))/np.count_nonzero(ocr)
    return fill,cov
print('\nvariantes de _bubble_mask_from_image (couverture de l encre) :')
for name,kw in [('code actuel',{}), ('sans erosion',{'erode':False}), ('tol=80',{'tol':80}),
                ('fermeture 31 avant CC',{'closing':31}), ('fermeture 61 avant CC',{'closing':61}),
                ('tol=80 + fermeture 61',{'tol':80,'closing':61})]:
    r=variant(**kw)
    print(f'   {name:24s} fill={r[0]:.3f} couverture_encre={r[1]:.3f}' if r else f'   {name:24s} None')

# balayage du seuil 0.5 sur la flotte
rows=[]
for cdir in sorted(glob.glob('scratch/bareme/cache/*')):
    p=os.path.join(cdir,'dets.pkl')
    if not os.path.exists(p): continue
    dd=pickle.load(open(p,'rb')); pg=cv2.imread(os.path.join(cdir,'page.png'))
    for i,jt in enumerate(dd['items']):
        a,b,c2,d2=jt['bbox']; cr=pg[max(0,b):d2, max(0,a):c2]
        hh,ww=max(1,d2-b),max(1,c2-a)
        if cr.shape[:2]!=(hh,ww): continue
        if str(jt.get('class_name','')).lower()=='out_text': continue
        om=TP._ocr_mask_from_regions(jt['text_regions'],hh,ww,crop_bgr=cr,dilate=3)
        if om is None: continue
        try: inr=TextRenderer._bubble_mask_from_image(cr)
        except Exception: inr=None
        if inr is None or inr.shape[:2]!=(hh,ww): continue
        e=cv2.erode(inr,k,1); bb=cv2.bitwise_and(om,e)
        rows.append(int(np.count_nonzero(bb))/max(1,int(np.count_nonzero(om))))
r=np.array(rows)
print(f'\n{len(r)} detections non-out_text avec un interior exploitable')
print('   ratio bounded/ocr : ', ' '.join(f'p{q}={np.percentile(r,q):.3f}' for q in (5,10,25,50,75)))
for t in (0.50,0.70,0.80,0.90,0.95,0.98):
    print(f'   seuil {t:.2f} -> garde-fou applique sur {int((r>=t).sum()):4d}/{len(r)} ; encre sauvee sur {int(((r>=0.5)&(r<t)).sum()):3d} detections')
