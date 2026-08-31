# Telea peut-il REMPLACER l'aplat partout ? On mesure le RESIDU (fantome) que
# Telea laisse, sur toutes les zones ou l'aplat s'applique aujourd'hui.
# Fantome = pixels de l'encre source qui, apres Telea, s'ecartent encore du
# fond local (celui qu'on voulait). Compare a l'aplat sur les memes zones.
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
def resid(after_crop, lm, encmask):
    # ecart entre l'apres et le fond local median hors masque, sous l'encre
    g=cv2.cvtColor(after_crop,cv2.COLOR_BGR2GRAY).astype(np.int16)
    # fond de reference = median des pixels NON masques dans un anneau
    ring=(cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))>0)&(lm==0)
    if ring.sum()<30: return 0.0
    ref=float(np.median(g[ring]))
    core=encmask>0
    if core.sum()<20: return 0.0
    return float(np.mean(np.abs(g[core]-ref)>35))  # part de l'encre encore visible
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
            enc=np.zeros(crop.shape[:2],np.uint8); enc[oy:oy+ch.shape[0],ox:ox+ch.shape[1]]=(ch>0).astype(np.uint8)*255
            # aplat
            ca=crop.copy(); ca[lm>0]=flat
            # telea
            ct=cv2.inpaint(crop,lm,5,cv2.INPAINT_TELEA)
            ra=resid(ca,lm,enc); rt=resid(ct,lm,enc)
        except Exception: continue
        rows.append([round(ra,3),round(rt,3),blob['series'],blob.get('page','p01'),i,(it['text'] or '')[:18]])
    del img
json.dump(rows,open('scratch/explore/scan_telea_ghost.json','w'))
import statistics
ra=[r[0] for r in rows]; rt=[r[1] for r in rows]
def q(v,p): v=sorted(v);return v[min(len(v)-1,int(p*len(v)))]
print("zones: %d"%len(rows))
print("residu APLAT  : p50=%.3f p90=%.3f p95=%.3f p99=%.3f"%(q(ra,.5),q(ra,.9),q(ra,.95),q(ra,.99)))
print("residu TELEA  : p50=%.3f p90=%.3f p95=%.3f p99=%.3f"%(q(rt,.5),q(rt,.9),q(rt,.95),q(rt,.99)))
pire=[r for r in rows if r[1]-r[0]>0.10]
print("\nzones ou TELEA laisse NETTEMENT plus de residu que l'aplat (>0.10) : %d"%len(pire))
for r in sorted(pire,key=lambda r:-(r[1]-r[0]))[:12]:
    print("  aplat=%.2f telea=%.2f  %-24s %s #%-3d %r"%(r[0],r[1],r[2][:24],r[3],r[4],r[5]))
