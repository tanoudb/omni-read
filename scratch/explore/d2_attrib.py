"""Attribue la perte du chirurgical_mask a l'un des deux garde-fous."""
import sys, os, glob, pickle, json
sys.path.insert(0,'.')
import numpy as np, cv2
from scratch.explore.d2_ink import ink_mask
from pipeline import TranslationPipeline as TP
from core.renderer import TextRenderer

def cov(ink,m):
    t=int(np.count_nonzero(ink))
    if t==0: return None
    if m is None: return 0.0
    return float(np.count_nonzero((m>0)&(ink>0)))/t

out=[]
for pkl in sorted(glob.glob('scratch/bareme/cache/*/dets.pkl')):
    key=os.path.basename(os.path.dirname(pkl))
    d=pickle.load(open(pkl,'rb'))
    page=cv2.imread(os.path.join(os.path.dirname(pkl),'page.png'),cv2.IMREAD_COLOR)
    for i,it in enumerate(d['items']):
        x1,y1,x2,y2=[int(v) for v in it['bbox']]
        crop=page[max(0,y1):y2,max(0,x1):x2]
        if crop.size==0: continue
        h,w=crop.shape[:2]
        ink,_=ink_mask(crop,it['text_regions'],h,w,{})
        if ink is None or np.count_nonzero(ink)<30: continue
        m0=TP._ocr_mask_from_regions(it['text_regions'],h,w,crop_bgr=crop,dilate=3)
        if m0 is None: continue
        rec=dict(key=key,i=i,cls=it['class_name'],text=(it.get('text') or '')[:60],
                 cov0=round(cov(ink,m0),4))
        m=m0
        g1=None
        if str(it['class_name']).lower()!='out_text':
            try: interior=TextRenderer._bubble_mask_from_image(crop)
            except Exception: interior=None
            if interior is not None and interior.shape[:2]==(h,w):
                k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
                interior=cv2.erode(interior,k,1)
                b=cv2.bitwise_and(m,interior)
                applied=int(np.count_nonzero(b))>=0.5*int(np.count_nonzero(m))
                g1=dict(applied=applied,cov=round(cov(ink,b),4),
                        keep=round(int(np.count_nonzero(b))/max(1,int(np.count_nonzero(m))),3))
                if applied: m=b
        rec['g1_interieur']=g1
        rec['cov_apres_g1']=round(cov(ink,m),4)
        bub=it.get('mask_binary'); g2=None
        if bub is not None:
            if bub.ndim==3: bub=bub[:,:,0]
            if bub.shape[:2]!=(h,w): bub=cv2.resize(bub,(w,h),interpolation=cv2.INTER_NEAREST)
            bub=(bub>0).astype(np.uint8)*255
            st=max(3,int(round(min(h,w)*0.025)))
            ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*st+1,2*st+1))
            er=cv2.erode(bub,ker,1)
            if int(np.count_nonzero(er))>=0.35*int(np.count_nonzero(bub)): bub=er
            inter=cv2.bitwise_and(m,bub)
            applied=int(np.sum(inter))>0.30*int(np.sum(m))
            g2=dict(applied=applied,cov=round(cov(ink,inter),4),
                    keep=round(int(np.count_nonzero(inter))/max(1,int(np.count_nonzero(m))),3))
            if applied: m=inter
        rec['g2_mask_binary']=g2
        m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
        rec['cov_recalc']=round(cov(ink,m),4)
        rec['cov_cache']=round(cov(ink,it['chirurgical_mask']),4)
        out.append(rec)
    del page
    print(key,len(out),flush=True)
json.dump(out,open('scratch/explore/d2_attrib.json','w',encoding='utf-8'),ensure_ascii=False)
