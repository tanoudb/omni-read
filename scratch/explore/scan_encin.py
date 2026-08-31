# Discriminant final : fraction de l'ENCRE couverte par l'INTERIEUR du ballon
# deduit (_bubble_mask_from_image, erode 9 comme le garde-fou fill_limit).
# #14 : 0.53 (moitie hors ballon) ; vraies bulles : ~1.00.
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
        x1,y1,x2,y2=it['bbox']; h,w=y2-y1,x2-x1
        crop=img[max(0,y1):y2,max(0,x1):x2]
        try:
            # aplat s'applique-t-il aujourd'hui ?
            m=max(30,h); cx1,cy1=max(0,x1-m),max(0,y1-m); cx2,cy2=min(img.shape[1],x2+m),min(img.shape[0],y2+m)
            bigcrop=img[cy1:cy2,cx1:cx2].copy(); ch=chir(img,it)
            lm=np.zeros(bigcrop.shape[:2],np.uint8); oy,ox=y1-cy1,x1-cx1
            lm[oy:oy+ch.shape[0],ox:ox+ch.shape[1]]=(ch>0).astype(np.uint8)*255
            lm=cv2.dilate(lm,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
            flat=TextRenderer._flat_fill_color(bigcrop,lm,local_bubble_mask=None,class_name=it['class_name'])
            if flat is None: continue
            interior=TextRenderer._bubble_mask_from_image(crop)
            chm=(ch>0)
            if interior is None:
                encin=0.0
            elif interior.shape[:2]==chm.shape:
                it9=cv2.erode(interior,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
                encin=float(np.count_nonzero(it9[chm]))/max(1,np.count_nonzero(chm))
            else:
                continue
        except Exception: continue
        rows.append([round(encin,3),blob['series'],blob.get('page','p01'),i,str(it['class_name']).lower(),(it['text'] or '')[:22]])
    del img
json.dump(rows,open('scratch/explore/scan_encin.json','w'))
sv=sorted(r[0] for r in rows); q=lambda p:sv[min(len(sv)-1,int(p*len(sv)))]
print("zones aplat : %d"%len(rows))
print("encre_dans_interieur : p05=%.2f p10=%.2f p25=%.2f p50=%.2f"%(q(.05),q(.10),q(.25),q(.50)))
for s in (0.80,0.85,0.90,0.95):
    lo=[r for r in rows if r[0]<s]
    print("  sous %.2f : %d zones basculeraient vers LaMa"%(s,len(lo)))
print("\nZones sous 0.90 (candidates LaMa) :")
for r in sorted([r for r in rows if r[0]<0.90]):
    print("  %.2f  %-26s %s #%-3d %-6s %r"%(r[0],r[1][:26],r[2],r[3],r[4],r[5]))
