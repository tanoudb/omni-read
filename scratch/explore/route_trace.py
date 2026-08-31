import sys,pickle,os,glob; sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core import TextRenderer
from core.detector import Detection
def find_cd(sub,page):
    for d in glob.glob('scratch/bareme/cache/*__%s'%page):
        if sub.replace("'","").lower() in os.path.basename(d).replace("'","").lower(): return d
ZONES=[  # (series_substr,page,idx)
 ("hellogin","p01",4),("hellogin","p02",6),("hellogin","p02",13),("hellogin","p02",27),
 ("i-married","p02",13),("path-of-vengeance","p02",8),("rise-of-the-dragon","p02",26),
 ("spend-more","p02",11),("spend-more","p02",26),("the-apocalypse","p02",29),
 ("the-returnee","p01",17),("the-wandering","p02",32),("the-wandering","p02",39),
 ("the_cleaner","p01",29),("solo-ex-rank","p01",10),("the-returnee","p01",1),
]
# instrument route
route={}
r=TextRenderer()
import core.renderer as RM
o_uni=TextRenderer._uniform_bg_erase; o_flat=TextRenderer._flat_fill_color; o_smooth=TextRenderer._smooth_fill; o_split=TextRenderer._split_flatten
def wrap():
    def uni(crop,mask,lb=None,grow=4,region_box=None,class_name=""):
        res=o_uni(crop,mask,lb,grow,region_box,class_name)
        if res is not None: route['r']='UNIFORM'
        return res
    def flat(crop,mask,max_std=12.0,local_bubble_mask=None,class_name=""):
        res=o_flat(crop,mask,max_std,local_bubble_mask,class_name)
        if res is not None and 'r' not in route: route['r']='FLAT'
        return res
    def smooth(crop,mask):
        res=o_smooth(crop,mask)
        if res is not None and 'r' not in route: route['r']='SMOOTH'
        return res
    def split(crop,mask,erased,**k):
        route.setdefault('r','LAMA'); route['split']=True
        return o_split(crop,mask,erased,**k)
    TextRenderer._uniform_bg_erase=staticmethod(uni)
    TextRenderer._flat_fill_color=staticmethod(flat)
    TextRenderer._smooth_fill=staticmethod(smooth)
    TextRenderer._split_flatten=staticmethod(split)
wrap()
for sub,page,idx in ZONES:
    cd=find_cd(sub,page)
    if not cd: print("MISS",sub,page,idx); continue
    blob=pickle.load(open(cd+'/dets.pkl','rb')); img=cv2.imread(cd+'/page.png')
    if idx>=len(blob['items']): print("OOB",sub,idx); continue
    it=blob['items'][idx]; x1,y1,x2,y2=it['bbox']
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    TranslationPipeline._assemble_chirurgical_mask(img,d)
    route.clear()
    out=img.copy()
    try:
        out=r.inpaint_region(out,d.x1,d.y1,d.x2,d.y2,text_regions=it['text_regions'],
            class_name=str(d.class_name),chirurgical_mask=d.chirurgical_mask,bubble_mask=d.mask_binary)
    except Exception as e: route['r']='ERR:%s'%e
    print("%-20s %s #%-3d %-8s route=%-7s split=%s  src=%r"%(sub[:20],page,idx,it['class_name'],route.get('r','?'),route.get('split',False),(it['text'] or '')[:22]))
