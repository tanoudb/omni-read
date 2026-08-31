# -*- coding: utf-8 -*-
import sys,pickle,os; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection

ZONES=[
 ("30-years-have-passed-since-the-prologue","p01",14,"#14 cartouche arche/fenetre (out_text)"),
 ("the-wandering-knight's-survival-manual","p01",17,"WK #17 (regression residu)"),
 ("the-returnee's-hidden-strategy-stream","p01",17,"RET #17 DING (regression spill)"),
]
def slug(s): return "".join(c if c.isalnum() or c in "-_" else "-" for c in s).strip("-").lower()

r=TextRenderer()
os.makedirs('scratch/explore/deverse_out',exist_ok=True)
for serie,page,idx,lbl in ZONES:
    cd='scratch/bareme/cache/%s__%s'%(slug(serie),page)
    if not os.path.exists(cd): cd='scratch/bareme/cache/%s__%s'%(serie,page)
    blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    it=blob['items'][idx]
    x1,y1,x2,y2=it['bbox']
    print("\n==",lbl,"class=",it['class_name'],"bbox",it['bbox'])
    for variant in ("ON","OFF"):
        d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
        d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
        TranslationPipeline._assemble_chirurgical_mask(img,d)
        # monkeypatch deverse OFF
        orig=TextRenderer._flat_deverse
        if variant=="OFF":
            TextRenderer._flat_deverse=staticmethod(lambda *a,**k: False)
        out=img.copy()
        out=r.inpaint_region(out,d.x1,d.y1,d.x2,d.y2,
             text_regions=it['text_regions'], class_name=str(d.class_name),
             chirurgical_mask=d.chirurgical_mask, bubble_mask=d.mask_binary)
        TextRenderer._flat_deverse=orig
        m=max(30,y2-y1)
        px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
        crop=out[py1:py2,px1:px2]
        fn='scratch/explore/deverse_out/%s_%03d_%s.png'%(slug(serie)[:16],idx,variant)
        cv2.imwrite(fn,crop)
        print("   saved",variant,fn,"crop",crop.shape)
