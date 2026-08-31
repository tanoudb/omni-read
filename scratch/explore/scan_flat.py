import sys, pickle, glob, os, json
sys.path.insert(0,'.')
import numpy as np, cv2
from pipeline import TranslationPipeline
from core.detector import Detection
from core.renderer import TextRenderer
def masks(img,it):
    d=Detection(it['class_name'],[float(v) for v in it['bbox']],it['score'])
    d.text_original=d.text_translated=it['text']; d.ocr_confidence=it['ocr_confidence']
    d.text_regions=it['text_regions']; d.mask_regions=it.get('mask_regions'); d.mask_binary=it.get('mask_binary')
    p=TranslationPipeline.__new__(TranslationPipeline); p.segmenter=None;p.logger=None;p.debug=False
    p._build_masks_for_detection(img,d); return d.chirurgical_mask
def local_mask(img,it,ch):
    x1,y1,x2,y2=it['bbox']; h=y2-y1; m=max(30,h)
    cx1,cy1=max(0,x1-m),max(0,y1-m); cx2,cy2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    crop=img[cy1:cy2,cx1:cx2].copy()
    lm=np.zeros(crop.shape[:2],np.uint8); oy,ox=y1-cy1,x1-cx1
    lm[oy:oy+ch.shape[0],ox:ox+ch.shape[1]]=(ch>0).astype(np.uint8)*255
    lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    return crop, lm, h
def spread(crop,lm,h,nreg):
    k=max(9,min(151,(max(1,h//max(1,nreg)))|1))
    lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)
    bg=np.dstack([cv2.medianBlur(lab[...,c],k) for c in range(3)]).astype(np.float32)
    sel=lm>0
    if sel.sum()<50: return None
    vals=bg[sel]; return float(np.linalg.norm(np.percentile(vals,90,axis=0)-np.percentile(vals,10,axis=0)))
rows=[]
for cd in sorted(glob.glob('scratch/bareme/cache/*')):
    if not os.path.exists(cd+"/dets.pkl"): continue
    blob=pickle.load(open(cd+"/dets.pkl","rb")); img=cv2.imread(cd+"/page.png")
    if img is None: continue
    for i,it in enumerate(blob['items']):
        cl=str(it['class_name']).lower()
        if cl=='out_text' or not it.get('text_regions'): continue
        try:
            ch=masks(img,it); crop,lm,h=local_mask(img,it,ch)
            v=TextRenderer._flat_fill_color(crop,lm,local_bubble_mask=None,class_name=it['class_name'])
            fa=v is not None; s=spread(crop,lm,h,len(it['text_regions']))
        except Exception: continue
        if s is not None:
            rows.append([round(s,1),bool(fa),None if v is None else [int(c) for c in v],blob['series'],blob.get('page','p01'),i,cl,(it['text'] or '')[:22]])
    del img
json.dump(rows,open('scratch/explore/scan_flat.json','w'))
fa=[r for r in rows if r[1]]
sv=sorted(r[0] for r in fa); q=lambda p:sv[min(len(sv)-1,int(p*len(sv)))] if sv else 0
print("zones bulle/System: %d | aplat applique: %d" % (len(rows),len(fa)))
print("etendue des zones OU L'APLAT S'APPLIQUE : p50=%.1f p90=%.1f p95=%.1f p99=%.1f"%(q(.5),q(.9),q(.95),q(.99)))
for s in (6,8,10,15,20,30):
    print("  aplat applique AVEC etendue>%2d : %d"%(s,sum(1 for r in fa if r[0]>s)))
print("\n15 pires (aplat sur fond etendu) :")
for r in sorted(fa,reverse=True)[:15]:
    print("  %6.1f aplat=%s %-24s %s #%-3d %-6s %r"%(r[0],r[2],r[3][:24],r[4],r[5],r[6],r[7]))
