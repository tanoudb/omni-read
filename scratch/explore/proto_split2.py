# -*- coding: utf-8 -*-
"""SPLIT v2 : sur les sous-regions a fond UNIFORME, poser la couleur mediane
locale EXACTE et l'etendre au-dela du masque pour engloutir le halo d'anti-
aliasing (invisible car le fond y est deja de cette couleur). LaMa garde les
sous-regions texturees. On teste plusieurs cas: #14 (arche/blanc)."""
import sys,pickle,os; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection

def slug(s): return "".join(c if c.isalnum() or c in "-_" else "-" for c in s).strip("-").lower()

def local_bg_stats(crop, mask, ksize=41):
    known=(mask==0).astype(np.float32); img=crop.astype(np.float32); K=(ksize,ksize)
    cnt=cv2.boxFilter(known,-1,K,normalize=False)+1e-6
    mean=np.zeros_like(img); sq=np.zeros_like(img)
    for c in range(3):
        mean[:,:,c]=cv2.boxFilter(img[:,:,c]*known,-1,K,normalize=False)/cnt
        sq[:,:,c]=cv2.boxFilter((img[:,:,c]**2)*known,-1,K,normalize=False)/cnt
    std=np.sqrt(np.clip(sq-mean**2,0,None)).max(axis=2)
    return mean,std

def split_erase(crop, lm, lama, T=8, grow=6, halo_tol=45):
    mean,std=local_bg_stats(crop,lm,41)
    uniform=(std<T)
    final=lama.copy().astype(np.float32)
    sel=(lm>0)&uniform
    # 1) poser la mediane locale sur la partie uniforme du masque
    final[sel]=mean[sel]
    # 2) etendre pour engloutir le halo : dilater sel, ne garder que les pixels
    #    dont la couleur ORIGINALE est proche de la mediane locale (meme fond)
    #    OU deja dans le masque. On reste dans la zone uniforme.
    sel_u=np.zeros(crop.shape[:2],np.uint8); sel_u[sel]=255
    grown=cv2.dilate(sel_u,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*grow+1,2*grow+1)))
    diff=np.abs(crop.astype(np.float32)-mean).max(axis=2)
    add=(grown>0)&uniform&(diff<halo_tol)&(sel==False)
    final[add]=mean[add]
    return np.clip(final,0,255).astype(np.uint8), sel, add

CASES=[("30-years-have-passed-since-the-prologue","p01",14)]
r=TextRenderer()
os.makedirs('scratch/explore/split_out',exist_ok=True)
for serie,page,idx in CASES:
    cd='scratch/bareme/cache/%s__%s'%(slug(serie),page)
    blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    TranslationPipeline._assemble_chirurgical_mask(img,d)
    h,w=y2-y1,x2-x1; m=max(30,h)
    px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    crop=img[py1:py2,px1:px2].copy(); ch=d.chirurgical_mask
    lm=np.zeros(crop.shape[:2],np.uint8); oy,ox=y1-py1,x1-px1
    lm[oy:oy+ch.shape[0],ox:ox+ch.shape[1]]=(ch>0).astype(np.uint8)*255
    lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    lama=r._inpaint_lama(crop,lm)
    out,sel,add=split_erase(crop,lm,lama,T=8,grow=6,halo_tol=45)
    cv2.imwrite('scratch/explore/split_out/v2_%s_%03d.png'%(slug(serie)[:12],idx),out)
    print("uniforme=%d halo_ajoute=%d masque=%d"%(int(sel.sum()),int(add.sum()),int((lm>0).sum())))
print("done")
