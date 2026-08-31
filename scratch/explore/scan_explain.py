# Discriminant : l'aplat choisi EXPLIQUE-t-il le fond LOCAL sous tout le masque ?
# Auto-valide, comme _smooth_fill : on compare la couleur d'aplat au fond local
# (median-blur qui efface le texte) SOUS le masque. Si une part notable du fond
# sous le masque s'ecarte de l'aplat, l'aplat va deverser sa couleur ailleurs.
# #14 : aplat blanc, mais fond beige a droite -> forte fraction non expliquee.
# DING/cartouche uni : aplat = fond partout -> fraction ~0.
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
def analyse(img,it):
    x1,y1,x2,y2=it['bbox']; h,w=y2-y1,x2-x1; m=max(30,h)
    cx1,cy1=max(0,x1-m),max(0,y1-m); cx2,cy2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
    crop=img[cy1:cy2,cx1:cx2].copy()
    ch=chir(img,it)
    lm=np.zeros(crop.shape[:2],np.uint8); oy,ox=y1-cy1,x1-cx1
    lm[oy:oy+ch.shape[0],ox:ox+ch.shape[1]]=(ch>0).astype(np.uint8)*255
    lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    flat=TextRenderer._flat_fill_color(crop,lm,local_bubble_mask=None,class_name=it['class_name'])
    if flat is None: return None
    # fond local (median-blur assez large pour effacer le texte)
    nreg=len(it['text_regions']) if it['text_regions'] else 1
    k=max(9,min(151,(max(1,h//max(1,nreg)))|1))
    bgc=np.dstack([cv2.medianBlur(crop[...,c],k) for c in range(3)]).astype(np.int16)
    fl=np.array([int(c) for c in flat],np.int16)
    sel=lm>0
    if int(sel.sum())<50: return None
    dev=np.abs(bgc[sel]-fl).max(axis=1)
    # fraction du fond sous le masque qui s'ecarte de l'aplat de plus de 25
    frac=float(np.mean(dev>25))
    return round(frac,3),[int(c) for c in flat]
rows=[]
for cd in sorted(glob.glob('scratch/bareme/cache/*')):
    if not os.path.exists(cd+"/dets.pkl"): continue
    blob=pickle.load(open(cd+"/dets.pkl","rb")); img=cv2.imread(cd+"/page.png")
    if img is None: continue
    for i,it in enumerate(blob['items']):
        if str(it['class_name']).lower()=='out_text' or not it.get('text_regions'): continue
        try: r=analyse(img,it)
        except Exception: r=None
        if r: rows.append([r[0],r[1],blob['series'],blob.get('page','p01'),i,str(it['class_name']).lower(),(it['text'] or '')[:20]])
    del img
json.dump(rows,open('scratch/explore/scan_explain.json','w'))
sv=sorted(r[0] for r in rows); q=lambda p:sv[min(len(sv)-1,int(p*len(sv)))]
print("zones aplat: %d"%len(rows))
print("fraction fond NON expliquee par l'aplat : p50=%.2f p75=%.2f p90=%.2f p95=%.2f"%(q(.5),q(.75),q(.9),q(.95)))
for s in (0.10,0.15,0.20,0.30):
    print("  au-dessus de %.2f : %d zones -> LaMa"%(s,sum(1 for r in rows if r[0]>s)))
print("\nZones > 0.15 (candidates LaMa), triees :")
for r in sorted([r for r in rows if r[0]>0.15],reverse=True)[:30]:
    print("  %.2f aplat=%s %-24s %s #%-3d %r"%(r[0],r[1],r[2][:24],r[3],r[4],r[6]))
