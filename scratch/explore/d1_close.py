import sys, os, pickle, glob, numpy as np, cv2
sys.path.insert(0,'.')
from pipeline import TranslationPipeline as TP
k7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
def bubble(crop_bgr, mode='current'):
    hh,ww=crop_bgr.shape[:2]
    if hh<24 or ww<24: return None
    g=cv2.cvtColor(crop_bgr,cv2.COLOR_BGR2GRAY)
    ref=float(np.median(g[hh//3:2*hh//3, ww//3:2*ww//3]))
    s=(np.abs(g.astype(np.int16)-ref)<40).astype(np.uint8)*255
    if mode=='current':
        m=cv2.erode(s,k7,1)
    elif mode=='close':
        m=cv2.morphologyEx(s,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(31,31)))
    n,lab,st,_=cv2.connectedComponentsWithStats(m,8)
    if n<2: return None
    big=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
    mm=(lab==big).astype(np.uint8)*255
    if mode=='current': mm=cv2.dilate(mm,k7,1)
    c,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not c: return None
    f=np.zeros_like(mm); cv2.drawContours(f,c,-1,255,-1)
    fill=float(np.count_nonzero(f))/(ww*hh)
    if not (0.20<=fill<=0.98): return None
    return f
res={'current':[], 'close':[]}; nones={'current':0,'close':0}; fills={'current':[],'close':[]}
for cdir in sorted(glob.glob('scratch/bareme/cache/*')):
    p=os.path.join(cdir,'dets.pkl')
    if not os.path.exists(p): continue
    dd=pickle.load(open(p,'rb')); pg=cv2.imread(os.path.join(cdir,'page.png'))
    for jt in dd['items']:
        if str(jt.get('class_name','')).lower()=='out_text': continue
        a,b,c2,d2=jt['bbox']; cr=pg[max(0,b):d2,max(0,a):c2]
        hh,ww=max(1,d2-b),max(1,c2-a)
        if cr.shape[:2]!=(hh,ww): continue
        om=TP._ocr_mask_from_regions(jt['text_regions'],hh,ww,crop_bgr=cr,dilate=3)
        if om is None: continue
        n0=max(1,int(np.count_nonzero(om)))
        for mode in res:
            f=bubble(cr,mode)
            if f is None: nones[mode]+=1; res[mode].append(1.0); continue
            fills[mode].append(float(np.count_nonzero(f))/(ww*hh))
            e=cv2.erode(f,k7,1)
            res[mode].append(int(np.count_nonzero(cv2.bitwise_and(om,e)))/n0)
for mode in res:
    arr=np.array(res[mode]); fl=np.array(fills[mode])
    print(f'{mode:8s} : interior=None sur {nones[mode]:3d} bulles | encre conservee p5={np.percentile(arr,5):.3f} <0.90:{int((arr<0.90).sum()):3d} <0.70:{int((arr<0.70).sum()):3d} | fill median={np.median(fl):.3f} p95={np.percentile(fl,95):.3f}')
