# -*- coding: utf-8 -*-
"""Sweep RAPIDE (LaMa désactivé) : sur quelles zones `_uniform_bg_erase` se
déclenche-t-il, et combien peint-il HORS encre source (proxy erase_spill) ?
LaMa=None => les zones non déclenchées tombent sur des fallbacks rapides ; les
zones déclenchées renvoient AVANT LaMa donc résultat identique à la prod."""
import sys, os, glob, pickle; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection

r=TextRenderer(); r.lama=None  # force fallbacks rapides pour les non-déclenchées
fired=[]
orig=TextRenderer._uniform_bg_erase.__get__(None, TextRenderer) if False else None
UBE=TextRenderer._uniform_bg_erase
rec={}
def spy(crop,mask,local_bubble=None,grow=4,region_box=None):
    res=UBE(crop,mask,local_bubble,grow,region_box)
    if res is not None:
        ch=(res.astype(int)-crop.astype(int)); painted=int((ch!=0).any(axis=2).sum())
        rec['painted']=painted; rec['fired']=True
    else:
        rec['fired']=False
    return res
TextRenderer._uniform_bg_erase=staticmethod(spy)

rows=[]
for cd in sorted(glob.glob('scratch/bareme/cache/*')):
    f=os.path.join(cd,'dets.pkl')
    if not os.path.exists(f): continue
    blob=pickle.load(open(f,'rb')); img=cv2.imread(os.path.join(cd,'page.png'))
    if img is None: continue
    H,W=img.shape[:2]; series=blob['series']; page=blob.get('page','p01')
    for i,it in enumerate(blob['items']):
        x1,y1,x2,y2=it['bbox']
        if x2<=x1 or y2<=y1: continue
        d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
        d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
        TranslationPipeline._assemble_chirurgical_mask(img,d)
        rec.clear()
        out=img.copy()
        try:
            out=r.inpaint_region(out,d.x1,d.y1,d.x2,d.y2,text_regions=it['text_regions'],
                class_name=str(d.class_name),chirurgical_mask=d.chirurgical_mask,bubble_mask=d.mask_binary)
        except Exception as e:
            pass
        if rec.get('fired'):
            # spill proxy : peint hors encre source (ocr dilate 3)
            crop_o=img[max(0,y1-max(30,y2-y1)):min(H,y2+max(30,y2-y1)),
                       max(0,x1-max(30,y2-y1)):min(W,x2+max(30,y2-y1))]
            src=TranslationPipeline._ocr_mask_from_regions(it['text_regions'],y2-y1,x2-x1,img[y1:y2,x1:x2],dilate=3)
            ch=(out[max(0,y1-max(30,y2-y1)):min(H,y2+max(30,y2-y1)),
                    max(0,x1-max(30,y2-y1)):min(W,x2+max(30,y2-y1))].astype(int)-crop_o.astype(int))
            changed=(ch!=0).any(axis=2)
            area=(y2-y1)*(x2-x1)
            rows.append({'series':series,'page':page,'idx':i,'cls':it['class_name'],
                         'painted':rec['painted'],'painted_pct':100.0*rec['painted']/max(1,area),
                         'text':(it['text'] or '')[:30]})
    del img
print("FIRED zones: %d / total"%len(rows))
from collections import Counter
print("by class:",dict(Counter(x['cls'] for x in rows)))
rows.sort(key=lambda x:-x['painted_pct'])
print("\n-- worst painted_pct (decor damage suspects) --")
for x in rows[:25]:
    print("  %-24s %s #%-3d %-8s painted=%5d %5.0f%%  %r"%(x['series'][:24],x['page'],x['idx'],x['cls'],x['painted'],x['painted_pct'],x['text']))
import json,io
json.dump(rows,io.open('scratch/explore/ube_fired.json','w',encoding='utf-8'),ensure_ascii=False)
