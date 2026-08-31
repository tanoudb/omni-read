import sys,pickle,os,glob; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
def find_cd(sub,page):
    for d in glob.glob('scratch/bareme/cache/*__%s'%page):
        if sub.lower() in os.path.basename(d).lower(): return d
r=TextRenderer(); os.makedirs('scratch/explore/diff_out',exist_ok=True)
for sub,page,idx,tag in [("a-mountain-of-corpses","p01",8,"mountain8_murim"),("archmage","p01",11,"archmage11b")]:
    cd=find_cd(sub,page); blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    TranslationPipeline._assemble_chirurgical_mask(img,d)
    out=img.copy()
    out=r.inpaint_region(out,d.x1,d.y1,d.x2,d.y2,text_regions=it['text_regions'],
        class_name=str(d.class_name),chirurgical_mask=d.chirurgical_mask,bubble_mask=d.mask_binary)
    m=max(30,y2-y1); px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    bef=img[py1:py2,px1:px2].copy(); era=out[py1:py2,px1:px2].copy()
    changed=(np.abs(bef.astype(int)-era.astype(int)).max(axis=2)>8)
    ov=bef.copy(); ov[changed]=(0,0,255)  # rouge = pixels modifiés
    blend=cv2.addWeighted(bef,0.5,ov,0.5,0)
    sep=np.full((bef.shape[0],6,3),200,np.uint8)
    cv2.imwrite('scratch/explore/diff_out/%s.png'%tag,np.hstack([bef,sep,blend,sep,era]))
    print("OK",tag,"changed_px",int(changed.sum()),"bbox",it['bbox'])
print("done")
