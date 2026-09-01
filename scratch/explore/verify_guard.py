import sys,pickle,os,glob; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
def find_cd(s,p):
    for d in glob.glob('scratch/bareme/cache/*__%s'%p):
        if s.replace("'","").lower() in os.path.basename(d).replace("'","").lower(): return d
def build(sub,page,idx,r):
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
r=TextRenderer()
print("mat loaded:", r.mat_inpainter is not None)
os.makedirs('scratch/explore/guard',exist_ok=True)
for sub,page,idx,tag in [("hellogin","p02",6,"bricks"),("path-of-vengeance","p02",0,"povghost")]:
    crop,lm=build(sub,page,idx,r)
    mat=r._inpaint_mat(crop,lm); anom=r._mat_anomaly(mat,crop,lm)
    guarded=r._inpaint_lama(crop,lm)
    used = "SimpleLama(rejected mat)" if anom else "MAT"
    print("%-9s anom=%s -> used %s"%(tag,anom,used))
    cv2.imwrite('scratch/explore/guard/%s.png'%tag, np.hstack([crop, np.full((crop.shape[0],5,3),128,np.uint8), guarded]))
