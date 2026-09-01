# -*- coding: utf-8 -*-
"""Un discriminateur peut-il distinguer un GAIN texture de mat d'une HALLUCINATION
(blob) ? Vérité terrain connue : mat GAGNE sur bricks/forest/photo/rays, PERD sur
pov_ghost (blob bleu), NUL sur imarried/apoc. Métriques testées dans la zone où
mat s'écarte de current :
 - anom_sat   : plus gros amas où mat est SATURÉ et loin de la médiane du voisinage
 - tex_ratio  : énergie de texture (gradient) de mat / du voisinage (blob = bas)
 - gamut_out  : plus gros amas hors gamut du voisinage
"""
import sys,pickle,os,glob; sys.path.insert(0,'.')
import numpy as np, cv2, torch
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config as LamaConfig
ZONES=[("hellogin","p02",6,"bricks","WIN"),("the_cleaner","p01",29,"photo","WIN"),
 ("hellogin","p01",4,"forest","WIN"),("path-of-vengeance","p02",8,"rays","WIN"),
 ("i-married","p01",11,"imar_ghost","TIE"),("the-apocalypse","p02",16,"apoc","TIE"),
 ("path-of-vengeance","p02",0,"pov_ghost","LOSS")]
def find_cd(s,p):
    for d in glob.glob('scratch/bareme/cache/*__%s'%p):
        if s.replace("'","").lower() in os.path.basename(d).replace("'","").lower(): return d
def build(sub,page,idx):
    cd=find_cd(sub,page); blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    TranslationPipeline._assemble_chirurgical_mask(img,d)
    m=max(30,y2-y1); px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    crop=img[py1:py2,px1:px2].copy(); cls=str(it['class_name']).lower()
    bm=TextRenderer._block_mask_from_regions(crop.shape[1],crop.shape[0],it['text_regions'],x1-px1,y1-py1)
    if cls in ("out_text","system") and bm is not None and int((bm>0).sum())>0: lm=bm.copy()
    else:
        lm=np.zeros(crop.shape[:2],np.uint8); ch=d.chirurgical_mask; ox,oy=x1-px1,y1-py1
        if ch is not None: lm[oy:oy+ch.shape[0],ox:ox+ch.shape[1]]=(ch>0).astype(np.uint8)*255
        lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7 if cls=='bulle' else 5,)*2))
    return crop,lm
cfg=LamaConfig(hd_strategy="Original",ldm_steps=20,hd_strategy_crop_margin=64,hd_strategy_crop_trigger_size=1024,hd_strategy_resize_limit=2048)
mm=ModelManager(name='mat',device=torch.device('cuda')); r=TextRenderer()
def texenergy(gray,m):
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    g=np.sqrt(gx*gx+gy*gy); return float(g[m].mean()) if m.sum()>0 else 0.0
print("%-11s %-5s %8s %8s"%("zone","truth","anom_sat","anom/mask"))
for sub,page,idx,tag,truth in ZONES:
    crop,lm=build(sub,page,idx)
    cur=r._inpaint_lama(crop,lm)
    rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB); out=mm(rgb,lm,cfg)
    if out.shape[:2]!=crop.shape[:2]: out=cv2.resize(out,(crop.shape[1],crop.shape[0]))
    out=out.astype(np.uint8); um=(lm==0)
    e_bgr=np.abs(out[um].astype(int)-crop[um].astype(int)).mean(); e_rgb=np.abs(out[um].astype(int)-rgb[um].astype(int)).mean()
    matb=cv2.cvtColor(out,cv2.COLOR_RGB2BGR) if e_rgb<e_bgr else out
    md=(lm>0)
    # voisinage (ring 15px)
    ring=(cv2.dilate(lm,np.ones((31,31),np.uint8))>0)&(lm==0)
    surr=crop[ring].reshape(-1,3).astype(np.float32); smed=np.median(surr,axis=0); sstd=surr.std(0).mean()
    # anom_sat : mat, dans le masque, loin de la mediane voisinage ET saturé
    hsv=cv2.cvtColor(matb,cv2.COLOR_BGR2HSV); sat=hsv[:,:,1].astype(np.float32)
    dev=np.abs(matb.astype(np.float32)-smed).max(2)
    anom=((dev>60)&(sat>90)&md).astype(np.uint8)
    n,_,st,_=cv2.connectedComponentsWithStats(anom,8); anom_sat=int(st[1:,4].max()) if n>1 else 0
    # tex ratio dans la zone de désaccord mat/current
    dis=(np.abs(matb.astype(int)-cur.astype(int)).max(2)>25)&md
    gm=cv2.cvtColor(matb,cv2.COLOR_BGR2GRAY); tr=texenergy(gm,dis)/max(1e-3,texenergy(cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY),ring))
    # gamut : plus gros amas hors gamut (dev>2.2*sstd)
    go=((dev>max(35,2.2*sstd))&md).astype(np.uint8)
    n2,_,st2,_=cv2.connectedComponentsWithStats(go,8); gamut_cc=int(st2[1:,4].max()) if n2>1 else 0
    print("%-11s %-5s %8d %8.3f"%(tag,truth,anom_sat,anom_sat/max(1,int(md.sum()))))
