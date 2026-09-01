import sys,pickle,os,glob; sys.path.insert(0,'.')
import numpy as np, cv2, torch
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config as LamaConfig
ZONES=[("30-years-have-passed-since-the-prologue","p01",1),
 ("archmage-curriculum","p02",67),("reincarnation-of-the-veteran-soldier","p02",34),
 ("spend-more-earn-more","p02",38),("the-marquis","p01",3),("the-returnee","p02",20)]
def find_cd(s,p):
    for d in glob.glob('scratch/bareme/cache/*__%s'%p):
        if s.replace("'","").lower()[:14] in os.path.basename(d).replace("'","").lower(): return d
def build(sub,page,idx):
    cd=find_cd(sub,page); blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    TranslationPipeline._assemble_chirurgical_mask(img,d)
    m=max(30,y2-y1); px1,py1=max(0,x1-m),max(0,y1-m); px2,py2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    crop=img[py1:py2,px1:px2].copy(); cls=str(it['class_name']).lower()
    lm=np.zeros(crop.shape[:2],np.uint8); ch=d.chirurgical_mask; ox,oy=x1-px1,y1-py1
    if ch is not None: lm[oy:oy+ch.shape[0],ox:ox+ch.shape[1]]=(ch>0).astype(np.uint8)*255
    lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    return crop,lm,(it['text'] or '')[:20]
cfg=LamaConfig(hd_strategy="Original",ldm_steps=20,hd_strategy_crop_margin=64,hd_strategy_crop_trigger_size=1024,hd_strategy_resize_limit=2048)
mm=ModelManager(name='mat',device=torch.device('cuda')); r=TextRenderer()
def anom_sat(matb,crop,lm):
    md=lm>0; ring=(cv2.dilate(lm,np.ones((31,31),np.uint8))>0)&(lm==0)
    if ring.sum()<50: return 0,0
    smed=np.median(crop[ring].reshape(-1,3).astype(np.float32),0)
    sat=cv2.cvtColor(matb,cv2.COLOR_BGR2HSV)[:,:,1].astype(np.float32)
    dev=np.abs(matb.astype(np.float32)-smed).max(2)
    anom=((dev>60)&(sat>90)&md).astype(np.uint8)
    n,_,st,_=cv2.connectedComponentsWithStats(anom,8); a=int(st[1:,4].max()) if n>1 else 0
    return a, a/max(1,int(md.sum()))
os.makedirs('scratch/explore/matsimple',exist_ok=True)
print("%-26s %8s %8s"%("zone","anom","anom/mask"))
for sub,page,idx in ZONES:
    try: crop,lm,txt=build(sub,page,idx)
    except Exception as e: print("skip",sub,e); continue
    cur=r._inpaint_lama(crop,lm)
    rgb=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB); out=mm(rgb,lm,cfg)
    if out.shape[:2]!=crop.shape[:2]: out=cv2.resize(out,(crop.shape[1],crop.shape[0]))
    out=out.astype(np.uint8); um=lm==0
    matb=cv2.cvtColor(out,cv2.COLOR_RGB2BGR) if np.abs(out[um].astype(int)-rgb[um].astype(int)).mean()<np.abs(out[um].astype(int)-crop[um].astype(int)).mean() else out
    a,af=anom_sat(matb,crop,lm)
    print("%-26s %8d %8.3f"%(sub[:26],a,af))
    H=max(crop.shape[0],cur.shape[0]); sep=np.full((H,5,3),128,np.uint8)
    cv2.imwrite('scratch/explore/matsimple/%s_%d.png'%(sub[:12],idx),np.hstack([crop,sep,cur,sep,matb]))
