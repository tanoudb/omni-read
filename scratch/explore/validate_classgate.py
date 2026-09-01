import sys,pickle,os,glob; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
def find_cd(s,p):
    for d in glob.glob('scratch/bareme/cache/*__%s'%p):
        if s.replace("'","").lower()[:12] in os.path.basename(d).replace("'","").lower(): return d
r=TextRenderer()
matcalls={'n':0}
orig=r._inpaint_mat
def spy(crop,mask):
    matcalls['n']+=1; return orig(crop,mask)
r._inpaint_mat=spy
os.makedirs('scratch/explore/classgate',exist_ok=True)
for s,p,i,tag in [("the-marquis","p01",25,"marquis25_bulle"),("the-apocalypse","p01",19,"apoc19_bulle"),("hellogin","p02",6,"bricks_out_text")]:
    cd=find_cd(s,p); b=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    it=b['items'][i]; x1,y1,x2,y2=it['bbox']
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    TranslationPipeline._assemble_chirurgical_mask(img,d)
    matcalls['n']=0
    out=img.copy()
    out=r.inpaint_region(out,d.x1,d.y1,d.x2,d.y2,text_regions=it['text_regions'],
        class_name=str(d.class_name),chirurgical_mask=d.chirurgical_mask,bubble_mask=d.mask_binary)
    m=max(30,y2-y1); px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    cv2.imwrite('scratch/explore/classgate/%s.png'%tag,out[py1:py2,px1:px2])
    print("%-18s class=%-8s mat_calls=%d %s"%(tag,it['class_name'],matcalls['n'],"MAT USED" if matcalls['n']>0 else "SimpleLama"))
