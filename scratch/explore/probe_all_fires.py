import sys,pickle,os,glob; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
# (series_substr, page, idx, tag)
ZONES=[
 ("the-marquis","p01",29,"marquis_29_RIDICULOUS_62pct"),
 ("the-marquis","p02",43,"marquis_43_ILLIA_58pct"),
 ("reincarnation-of-the-v","p01",26,"reincarn_26_53pct"),
 ("the-returnee","p02",28,"returnee_28_LIVE_38pct"),
 ("archmage","p01",11,"archmage_11_38pct"),
 ("30-years","p02",22,"z30p2_22_YEAR10_33pct"),
 ("savior","p01",3,"savior_3_CHOSEN_17pct"),
 ("30-years","p01",43,"z30_43_System_15pct"),
 ("30-years","p02",10,"z30p2_10_System_14pct"),
 ("savior","p01",4,"savior_4_12pct"),
]
r=TextRenderer(); os.makedirs('scratch/explore/allfires',exist_ok=True)
def find_cd(sub,page):
    for d in glob.glob('scratch/bareme/cache/*__%s'%page):
        if sub.lower() in os.path.basename(d).lower(): return d
    return None
for sub,page,idx,tag in ZONES:
    cd=find_cd(sub,page)
    if not cd: print("MISS",tag,sub,page); continue
    blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    if idx>=len(blob['items']): print("OOB",tag); continue
    it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    TranslationPipeline._assemble_chirurgical_mask(img,d)
    out=img.copy()
    out=r.inpaint_region(out,d.x1,d.y1,d.x2,d.y2,text_regions=it['text_regions'],
        class_name=str(d.class_name),chirurgical_mask=d.chirurgical_mask,bubble_mask=d.mask_binary)
    m=max(30,y2-y1); px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    # side-by-side before|erased
    bef=img[py1:py2,px1:px2]; era=out[py1:py2,px1:px2]
    h=max(bef.shape[0],era.shape[0]); sep=np.full((h,6,3),128,np.uint8)
    canvas=np.hstack([bef,sep,era])
    cv2.imwrite('scratch/explore/allfires/%s.png'%tag,canvas)
    print("OK",tag,it['class_name'])
print("done")
