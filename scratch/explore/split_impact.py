import sys,pickle,os,glob; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
def find_cd(sub,page):
    for d in glob.glob('scratch/bareme/cache/*__%s'%page):
        if sub.replace("'","").lower() in os.path.basename(d).replace("'","").lower(): return d
ZONES=[("solo-ex-rank","p01",10,"solo10"),("30-years","p01",14,"z14"),
       ("the_cleaner","p01",29,"cleaner29"),("hellogin","p02",6,"hel6")]
o_split=TextRenderer._split_flatten
STD=[8.0,4.0]
for std in STD:
    print("\n===== std_thr=%.1f ====="%std)
    def split(crop,mask,erased,std_thr=8.0,**k):
        res=o_split(crop,mask,erased,std_thr=std,**k)
        ch=int((res.astype(int)-erased.astype(int)).any(axis=2).sum())
        split.last=ch; return res
    TextRenderer._split_flatten=staticmethod(split)
    r=TextRenderer()
    for sub,page,idx,tag in ZONES:
        cd=find_cd(sub,page); blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
        it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
        d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
        d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
        TranslationPipeline._assemble_chirurgical_mask(img,d)
        split.last=0
        out=img.copy()
        out=r.inpaint_region(out,d.x1,d.y1,d.x2,d.y2,text_regions=it['text_regions'],
            class_name=str(d.class_name),chirurgical_mask=d.chirurgical_mask,bubble_mask=d.mask_binary)
        m=max(30,y2-y1); px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
        cv2.imwrite('scratch/explore/split_%s_std%d.png'%(tag,int(std)),out[py1:py2,px1:px2])
        print("  %-10s split_changed=%d"%(tag,split.last))
TextRenderer._split_flatten=staticmethod(o_split)
