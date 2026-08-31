# -*- coding: utf-8 -*-
"""Prototype effacement SPLIT : aplat local la ou le fond est uniforme,
LaMa la ou il est texture. But : tuer le fantome gris sur le blanc de #14
sans toucher l'arche beige reconstruite par LaMa."""
import sys,pickle,os; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection

def slug(s): return "".join(c if c.isalnum() or c in "-_" else "-" for c in s).strip("-").lower()

def local_bg_stats(crop, mask, ksize=41):
    """Moyenne et ecart-type du FOND (pixels non masques) dans un voisinage
    ksize autour de chaque pixel. Aux pixels masques, ne compte que le fond."""
    known = (mask==0).astype(np.float32)            # 1 = fond connu
    img = crop.astype(np.float32)
    K=(ksize,ksize)
    cnt = cv2.boxFilter(known, -1, K, normalize=False)+1e-6
    mean = np.zeros_like(img); sq=np.zeros_like(img)
    for c in range(3):
        ch=img[:,:,c]*known
        s=cv2.boxFilter(ch,-1,K,normalize=False)
        s2=cv2.boxFilter((img[:,:,c]**2)*known,-1,K,normalize=False)
        mean[:,:,c]=s/cnt
        sq[:,:,c]=s2/cnt
    var=np.clip(sq-mean**2,0,None)
    std=np.sqrt(var).max(axis=2)                     # pire canal
    return mean, std

serie,page,idx="30-years-have-passed-since-the-prologue","p01",14
cd='scratch/bareme/cache/%s__%s'%(slug(serie),page)
blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
TranslationPipeline._assemble_chirurgical_mask(img,d)

# reconstruire crop + local_mask comme inpaint_region (marge = max(30,h))
h,w=y2-y1,x2-x1; m=max(30,h)
px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
crop=img[py1:py2,px1:px2].copy()
ch=d.chirurgical_mask
lm=np.zeros(crop.shape[:2],np.uint8); oy,ox=y1-py1,x1-px1
lm[oy:oy+ch.shape[0],ox:ox+ch.shape[1]]=(ch>0).astype(np.uint8)*255
lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))

r=TextRenderer()
lama=r._inpaint_lama(crop,lm)

mean,std=local_bg_stats(crop,lm,41)
os.makedirs('scratch/explore/split_out',exist_ok=True)
cv2.imwrite('scratch/explore/split_out/00_lama.png',lama)
# carte d'uniformite (visualisation)
uni=(std<8).astype(np.uint8)*255
cv2.imwrite('scratch/explore/split_out/01_uniform.png',cv2.bitwise_and(uni,lm))

for T in (6,8,12):
    uniform=(std<T)
    final=lama.copy().astype(np.float32)
    sel=(lm>0)&uniform
    final[sel]=mean[sel]
    final=np.clip(final,0,255).astype(np.uint8)
    cv2.imwrite('scratch/explore/split_out/split_T%02d.png'%T,final)
    print("T=%d  pixels aplati=%d / masque=%d (%.0f%%)"%(T,int(sel.sum()),int((lm>0).sum()),100*sel.sum()/max(1,(lm>0).sum())))
print("done")
