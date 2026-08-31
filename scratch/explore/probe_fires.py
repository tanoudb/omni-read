import sys,pickle,os; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
def slug(s): return "".join(c if c.isalnum() or c in "-_" else "-" for c in s).strip("-").lower()
ZONES=[
 ("archmage-curriculum","p01",11,"archmage_out_text"),
 ("savior-of-divine-blood%3A-draw-out-0.00000001percent-to-become-the-strongest","p01",3,"savior3_out_text"),
 ("30-years-have-passed-since-the-prologue","p01",43,"z30_43_System"),
 ("30-years-have-passed-since-the-prologue","p02",10,"z30p2_10_System"),
]
ube=TextRenderer._uniform_bg_erase
def spy(crop,mask,lb=None,grow=4,region_box=None,class_name=""):
    res=ube(crop,mask,lb,grow,region_box,class_name); print("   fired=%s"%(res is not None)); return res
TextRenderer._uniform_bg_erase=staticmethod(spy)
r=TextRenderer(); os.makedirs('scratch/explore/fires_out',exist_ok=True)
for serie,page,idx,tag in ZONES:
    cd='scratch/bareme/cache/%s__%s'%(slug(serie),page)
    if not os.path.exists(cd): print("MISS",tag); continue
    blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    TranslationPipeline._assemble_chirurgical_mask(img,d)
    print("==",tag,it['class_name'],"src=",repr((it['text'] or '')[:30]))
    out=img.copy()
    out=r.inpaint_region(out,d.x1,d.y1,d.x2,d.y2,text_regions=it['text_regions'],
        class_name=str(d.class_name),chirurgical_mask=d.chirurgical_mask,bubble_mask=d.mask_binary)
    m=max(30,y2-y1); px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    cv2.imwrite('scratch/explore/fires_out/%s_before.png'%tag,img[py1:py2,px1:px2])
    cv2.imwrite('scratch/explore/fires_out/%s_erased.png'%tag,out[py1:py2,px1:px2])
print("done")
