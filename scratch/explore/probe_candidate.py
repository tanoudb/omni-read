# -*- coding: utf-8 -*-
"""Candidat FINAL : masque fidele (_assemble_chirurgical_mask) + inpaint_region
avec split_flatten bake. Sauve before/erased pour inspection visuelle."""
import sys,pickle,os; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
def slug(s): return "".join(c if c.isalnum() or c in "-_" else "-" for c in s).strip("-").lower()
ZONES=[
 ("30-years-have-passed-since-the-prologue","p01",14,"14_arche_blanc"),
 ("archmage-curriculum","p01",11,"archmage_goldenage_ghost163"),
 ("the-apocalypse-needs-a-pro","p02",16,"apoc_secondapoc_ghost180"),
 ("rise-of-the-dragon-overlord","p02",22,"rise_dingskill_System"),
 ("i-married-the-dragon-i-killed","p02",12,"imarried_justkill_radiant_guard"),
 ("turning-my-life-around-with-crypto-uncensored","p01",0,"turning_skin_guard"),
]
r=TextRenderer(); os.makedirs('scratch/explore/cand_out',exist_ok=True)
for serie,page,idx,tag in ZONES:
    cd='scratch/bareme/cache/%s__%s'%(slug(serie),page)
    if not os.path.exists(cd):
        print("MISS",tag,cd); continue
    blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    if idx>=len(blob['items']): print("IDX oob",tag,len(blob['items'])); continue
    it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    TranslationPipeline._assemble_chirurgical_mask(img,d)
    out=img.copy()
    out=r.inpaint_region(out,d.x1,d.y1,d.x2,d.y2,text_regions=it['text_regions'],
        class_name=str(d.class_name),chirurgical_mask=d.chirurgical_mask,bubble_mask=d.mask_binary)
    m=max(30,y2-y1); px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    cv2.imwrite('scratch/explore/cand_out/%s_before.png'%tag,img[py1:py2,px1:px2])
    cv2.imwrite('scratch/explore/cand_out/%s_erased.png'%tag,out[py1:py2,px1:px2])
    print("OK",tag,it['class_name'],"bbox",it['bbox'])
print("done")
