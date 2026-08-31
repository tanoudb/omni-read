# -*- coding: utf-8 -*-
"""Sweep DIRECT de _uniform_bg_erase (sans inpaint_region complet). Reconstruit
crop + local_mask + region_box comme inpaint_region, appelle la méthode, mesure
la surface peinte hors encre source."""
import sys, os, glob, pickle; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection

def build_local(img, it, d):
    x1,y1,x2,y2=it['bbox']; H,W=img.shape[:2]
    m=max(30,y2-y1)
    cx1,cy1=max(0,x1-m),max(0,y1-m); cx2,cy2=min(W,x2+m),min(H,y2+m)
    crop=img[cy1:cy2,cx1:cx2].copy(); ch_,cw_=crop.shape[:2]
    cls=str(it['class_name']).lower()
    block=TextRenderer._block_mask_from_regions(cw_,ch_,it['text_regions'],x1-cx1,y1-cy1)
    if cls in ("out_text","system") and block is not None and int((block>0).sum())>0:
        lm=block.copy()
    else:
        lm=np.zeros((ch_,cw_),np.uint8)
        cm=d.chirurgical_mask
        if cm is not None:
            ox,oy=x1-cx1,y1-cy1
            lm[oy:oy+cm.shape[0],ox:ox+cm.shape[1]]=(cm>0).astype(np.uint8)*255
        kd=7 if cls=='bulle' else 5
        lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kd,kd)))
    # local_bubble
    lb=None
    mb=it.get('mask_binary')
    if mb is not None:
        mb=np.asarray(mb)
        if mb.ndim==3: mb=mb[:,:,0]
        dh,dw=y2-y1,x2-x1
        if mb.shape[:2]!=(dh,dw): mb=cv2.resize(mb,(dw,dh),interpolation=cv2.INTER_NEAREST)
        lb=np.zeros((ch_,cw_),np.uint8); ox,oy=x1-cx1,y1-cy1
        lb[oy:oy+dh,ox:ox+dw]=(mb>0).astype(np.uint8)*255
    rb=(max(0,x1-cx1),max(0,y1-cy1),min(cw_,x2-cx1),min(ch_,y2-cy1))
    return crop,lm,lb,rb,(cx1,cy1)

rows=[]
for cd in sorted(glob.glob('scratch/bareme/cache/*')):
    f=os.path.join(cd,'dets.pkl')
    if not os.path.exists(f): continue
    blob=pickle.load(open(f,'rb')); img=cv2.imread(os.path.join(cd,'page.png'))
    if img is None: continue
    series=blob['series']; page=blob.get('page','p01')
    for i,it in enumerate(blob['items']):
        x1,y1,x2,y2=it['bbox']
        if x2<=x1 or y2<=y1: continue
        d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
        d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
        TranslationPipeline._assemble_chirurgical_mask(img,d)
        crop,lm,lb,rb,off=build_local(img,it,d)
        res=TextRenderer._uniform_bg_erase(crop,lm,lb,4,rb,it["class_name"])
        if res is None: continue
        ch=(res.astype(int)-crop.astype(int)); changed=(ch!=0).any(axis=2)
        # encre source dans le crop
        src=TranslationPipeline._ocr_mask_from_regions(it['text_regions'],y2-y1,x2-x1,img[y1:y2,x1:x2],dilate=3)
        srcc=np.zeros(crop.shape[:2],np.uint8)
        if src is not None:
            ox,oy=x1-off[0],y1-off[1]; srcc[oy:oy+src.shape[0],ox:ox+src.shape[1]]=(src>0).astype(np.uint8)*255
        srcd=cv2.dilate(srcc,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
        spill=int((changed&(srcd==0)).sum())
        area=(y2-y1)*(x2-x1)
        rows.append({'series':series,'page':page,'idx':i,'cls':it['class_name'],
                     'painted':int(changed.sum()),'spill':spill,'spill_pct':100.0*spill/max(1,area),
                     'text':(it['text'] or '')[:28]})
    del img
from collections import Counter
print("FIRED %d zones ; by class %s"%(len(rows),dict(Counter(x['cls'] for x in rows))))
rows.sort(key=lambda x:-x['spill_pct'])
print("\n-- worst spill_pct (decor damage suspects) --")
for x in rows[:22]:
    print("  %-22s %s #%-3d %-8s spill=%5d %5.1f%%  %r"%(x['series'][:22],x['page'],x['idx'],x['cls'],x['spill'],x['spill_pct'],x['text']))
import json,io; json.dump(rows,io.open('scratch/explore/ube_fired.json','w',encoding='utf-8'),ensure_ascii=False)
