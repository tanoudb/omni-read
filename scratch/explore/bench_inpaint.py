# -*- coding: utf-8 -*-
"""A/B inpainting : sur les pires zones TEXTURÉES (fantômes/bavures LaMa), compare
le LaMa actuel (SimpleLama) aux modèles lama_cleaner (mat/zits/fcf/manga/ldm) sur
le MÊME masque. Sauve un montage before|current|<candidats> par zone."""
import sys,pickle,os,glob,time; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
import torch
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config as LamaConfig

CAND = sys.argv[1].split(',') if len(sys.argv)>1 else ['mat','zits','fcf','manga','ldm']
ZONES=[  # (series_substr, page, idx, tag)  -- fonds texturés
 ("hellogin","p02",6,"hel_bricks"),
 ("the_cleaner","p01",29,"cleaner_photo"),
 ("hellogin","p01",4,"hel_forest"),
 ("path-of-vengeance","p02",8,"pov_rays"),
 ("the-apocalypse","p02",16,"apoc_sky"),
 ("i-married","p01",11,"imarried_ghost"),
 ("path-of-vengeance","p02",0,"pov_ghost"),
 ("spend-more","p02",26,"spend_screen"),
]
def find_cd(sub,page):
    for d in glob.glob('scratch/bareme/cache/*__%s'%page):
        if sub.replace("'","").lower() in os.path.basename(d).replace("'","").lower(): return d

cfg=LamaConfig(hd_strategy="Original", ldm_steps=20, hd_strategy_crop_margin=64,
               hd_strategy_crop_trigger_size=1024, hd_strategy_resize_limit=2048)
dev=torch.device("cuda")
# baseline SimpleLama (= production)
r=TextRenderer()
def build(sub,page,idx):
    cd=find_cd(sub,page); blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    TranslationPipeline._assemble_chirurgical_mask(img,d)
    m=max(30,y2-y1); px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    crop=img[py1:py2,px1:px2].copy()
    cls=str(it['class_name']).lower()
    bm=TextRenderer._block_mask_from_regions(crop.shape[1],crop.shape[0],it['text_regions'],x1-px1,y1-py1)
    if cls in ("out_text","system") and bm is not None and int((bm>0).sum())>0:
        lm=bm.copy()
    else:
        lm=np.zeros(crop.shape[:2],np.uint8); ch=d.chirurgical_mask; ox,oy=x1-px1,y1-py1
        if ch is not None: lm[oy:oy+ch.shape[0],ox:ox+ch.shape[1]]=(ch>0).astype(np.uint8)*255
        kd=7 if cls=='bulle' else 5
        lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kd,kd)))
    return crop,lm

os.makedirs('scratch/explore/bench',exist_ok=True)
# précharge les modèles une fois
models={}
for name in CAND:
    try: models[name]=ModelManager(name=name, device=dev); print("loaded",name,flush=True)
    except Exception as e: print("skip",name,repr(e)[:80],flush=True)

for sub,page,idx,tag in ZONES:
    try:
        crop,lm=build(sub,page,idx)
    except Exception as e:
        print("build FAIL",tag,e); continue
    panels=[crop]; labels=['before']
    # current SimpleLama
    try:
        cur=r._inpaint_lama(crop,lm); panels.append(cur); labels.append('current')
    except Exception as e: print("cur fail",tag,e)
    for name,mm in models.items():
        try:
            rgb=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)          # lama_cleaner attend du RGB
            out=mm(rgb, lm, cfg)
            if out.shape[:2]!=crop.shape[:2]: out=cv2.resize(out,(crop.shape[1],crop.shape[0]))
            out=out.astype(np.uint8)
            # normalise la sortie en BGR : on détecte l'ordre en recalant la zone
            # NON masquée sur la source (l'inpainting ne doit toucher que le masque).
            um=(lm==0)
            if int(um.sum())>200:
                e_bgr=float(np.abs(out[um].astype(int)-crop[um].astype(int)).mean())
                e_rgb=float(np.abs(out[um].astype(int)-rgb[um].astype(int)).mean())
                if e_rgb<e_bgr: out=cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            panels.append(out); labels.append(name)
        except Exception as e:
            print("run fail",tag,name,repr(e)[:80]); panels.append(np.zeros_like(crop)); labels.append(name+"!")
    # montage horizontal avec labels
    H=max(p.shape[0] for p in panels); W=sum(p.shape[1] for p in panels)+8*len(panels)
    canvas=np.full((H+22,W,3),40,np.uint8); x=0
    for p,l in zip(panels,labels):
        canvas[22:22+p.shape[0],x:x+p.shape[1]]=p
        cv2.putText(canvas,l,(x+2,16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
        x+=p.shape[1]+8
    cv2.imwrite('scratch/explore/bench/%s.png'%tag,canvas)
    print("done",tag,labels,flush=True)
print("BENCHDONE",flush=True)
