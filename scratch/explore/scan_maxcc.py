# Discriminant final : basculer vers Telea SEULEMENT si l'aplat differe de Telea
# sur une COMPOSANTE CONNEXE notable sous le masque (l'aplat va y deverser sa
# couleur sur un fond different). Sinon, garder l'aplat (parfait sur fond uni,
# sans le fantome que Telea laisse sur les bulles rayonnantes/degradees).
import sys, pickle, glob, os, json
sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core.detector import Detection
from core.renderer import TextRenderer
def chir(img,it):
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_original=d.text_translated=it['text']; d.ocr_confidence=it['ocr_confidence']
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    p=TranslationPipeline.__new__(TranslationPipeline); p.segmenter=None;p.logger=None;p.debug=False
    p._build_masks_for_detection(img,d); return d.chirurgical_mask
rows=[]
for cd in sorted(glob.glob('scratch/bareme/cache/*')):
    if not os.path.exists(cd+"/dets.pkl"): continue
    blob=pickle.load(open(cd+"/dets.pkl","rb")); img=cv2.imread(cd+"/page.png")
    if img is None: continue
    for i,it in enumerate(blob['items']):
        if str(it['class_name']).lower()=='out_text' or not it.get('text_regions'): continue
        x1,y1,x2,y2=it['bbox']; h,w=y2-y1,x2-x1; m=max(30,h)
        cx1,cy1=max(0,x1-m),max(0,y1-m); cx2,cy2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
        crop=img[cy1:cy2,cx1:cx2].copy()
        try:
            ch=chir(img,it)
            lm=np.zeros(crop.shape[:2],np.uint8); oy,ox=y1-cy1,x1-cx1
            lm[oy:oy+ch.shape[0],ox:ox+ch.shape[1]]=(ch>0).astype(np.uint8)*255
            lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
            flat=TextRenderer._flat_fill_color(crop,lm,local_bubble_mask=None,class_name=it['class_name'])
            if flat is None: continue
            tel=cv2.inpaint(crop,lm,3,cv2.INPAINT_TELEA).astype(np.int16)
            fl=np.array([int(c) for c in flat],np.int16)
            dev=np.abs(tel-fl).max(axis=2)
            ne=((dev>30)&(lm>0)).astype(np.uint8)
            n,_,st,_=cv2.connectedComponentsWithStats(ne,8)
            maxcc=int(st[1:,cv2.CC_STAT_AREA].max()) if n>1 else 0
        except Exception: continue
        rows.append([maxcc,blob['series'],blob.get('page','p01'),i,str(it['class_name']).lower(),(it['text'] or '')[:20]])
    del img
json.dump(rows,open('scratch/explore/scan_maxcc.json','w'))
sv=sorted(r[0] for r in rows); q=lambda p:sv[min(len(sv)-1,int(p*len(sv)))]
print("zones aplat: %d"%len(rows))
print("maxCC (aplat vs telea) : p50=%d p90=%d p95=%d p99=%d max=%d"%(q(.5),q(.9),q(.95),q(.99),sv[-1]))
for s in (500,800,1000,1500,2000):
    print("  maxCC>%4d : %d zones -> Telea"%(s,sum(1 for r in rows if r[0]>s)))
print("\nzones maxCC>800 (candidates Telea), triees :")
for r in sorted([r for r in rows if r[0]>800],reverse=True)[:25]:
    print("  %6d  %-24s %s #%-3d %-6s %r"%(r[0],r[1][:24],r[2],r[3],r[4],r[5]))
