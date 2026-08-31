# Discriminant v2 : l'aplat ne doit peindre que si la zone REELLEMENT peinte
# (extension du masque, bornee a l'interieur du ballon fill_limit) est de
# couleur uniforme. On reproduit fill_limit exactement comme inpaint_region.
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
    crop=img[cy1:cy2,cx1:cx2].copy(); ch,cw=crop.shape[:2]
    chm=chir(img,it)
    lm=np.zeros((ch,cw),np.uint8); oy,ox=y1-cy1,x1-cx1
    lm[oy:oy+chm.shape[0],ox:ox+chm.shape[1]]=(chm>0).astype(np.uint8)*255
    lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    if str(it['class_name']).lower()=='out_text': return None
    flat=TextRenderer._flat_fill_color(crop,lm,local_bubble_mask=None,class_name=it['class_name'])
    if flat is None: return None  # deja LaMa
    # fill_limit = interieur du ballon deduit de la bbox, erode 9, garde si >=90% de l'encre
    interior=TextRenderer._bubble_mask_from_image(img[max(0,y1):y2,max(0,x1):x2])
    fill_limit=None
    if interior is not None and interior.shape[:2]==(h,w):
        interior=cv2.erode(interior,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
        lim=np.zeros((ch,cw),np.uint8)
        y0=oy; x0=ox
        lim[y0:y0+h,x0:x0+w]=interior
        if int(np.count_nonzero(cv2.bitwise_and(lm,lim)))>=0.9*int(np.count_nonzero(lm)):
            fill_limit=lim
    # zone reellement peinte : extension du masque
    ext=TextRenderer._extend_fill_mask(crop,lm,flat,inside_box=None)
    if fill_limit is not None:
        ext=cv2.bitwise_or(cv2.bitwise_and(ext,fill_limit),lm)
    # etendue de la couleur du FOND LOCAL sous la zone peinte
    nreg=len(it['text_regions']) if it['text_regions'] else 1
    k=max(9,min(151,(max(1,h//max(1,nreg)))|1))
    lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)
    bg=np.dstack([cv2.medianBlur(lab[...,c],k) for c in range(3)]).astype(np.float32)
    sel=ext>0
    if sel.sum()<50: return None
    vals=bg[sel]
    etendue=float(np.linalg.norm(np.percentile(vals,90,axis=0)-np.percentile(vals,10,axis=0)))
    borne = fill_limit is not None
    return etendue, borne, [int(c) for c in flat]

rows=[]
for cd in sorted(glob.glob('scratch/bareme/cache/*')):
    if not os.path.exists(cd+"/dets.pkl"): continue
    blob=pickle.load(open(cd+"/dets.pkl","rb")); img=cv2.imread(cd+"/page.png")
    if img is None: continue
    for i,it in enumerate(blob['items']):
        if str(it['class_name']).lower()=='out_text' or not it.get('text_regions'): continue
        try: r=analyse(img,it)
        except Exception: r=None
        if r: rows.append([round(r[0],1),r[1],r[2],blob['series'],blob.get('page','p01'),i,str(it['class_name']).lower(),(it['text'] or '')[:22]])
    del img
json.dump(rows,open('scratch/explore/scan_flat2.json','w'))
sv=sorted(r[0] for r in rows); q=lambda p:sv[min(len(sv)-1,int(p*len(sv)))]
print("zones aplat (v2, bornee ballon): %d"%len(rows))
print("etendue zone peinte : p50=%.1f p90=%.1f p95=%.1f p99=%.1f"%(q(.5),q(.9),q(.95),q(.99)))
for s in (5,6,8,10,12):
    hi=[r for r in rows if r[0]>s]
    print("  >%2d : %d zones  (dont bornees ballon: %d)"%(s,len(hi),sum(1 for r in hi if r[1])))
print("\nEtendue > 6 (candidates a LaMa) :")
for r in sorted([r for r in rows if r[0]>6],reverse=True)[:25]:
    print("  %6.1f borne=%s aplat=%s %-24s %s #%-3d %r"%(r[0],r[1],r[2],r[3][:24],r[4],r[5],r[7]))
