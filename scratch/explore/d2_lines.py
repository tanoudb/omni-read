import sys, os, glob, pickle, json
sys.path.insert(0,'.')
import numpy as np, cv2
from scratch.explore.d2_ink import ink_mask, polygons_and_lines
from pipeline import TranslationPipeline as TP

out=[]
for pkl in sorted(glob.glob('scratch/bareme/cache/*/dets.pkl')):
    key=os.path.basename(os.path.dirname(pkl))
    d=pickle.load(open(pkl,'rb'))
    page=cv2.imread(os.path.join(os.path.dirname(pkl),'page.png'),cv2.IMREAD_COLOR)
    for i,it in enumerate(d['items']):
        x1,y1,x2,y2=[int(v) for v in it['bbox']]
        crop=page[max(0,y1):y2,max(0,x1):x2]
        if crop.size==0: continue
        h,w=crop.shape[:2]
        ink,_=ink_mask(crop,it['text_regions'],h,w,{})
        if ink is None or np.count_nonzero(ink)<30: continue
        ch=it.get('chirurgical_mask')
        ocr=TP._ocr_mask_from_regions(it['text_regions'],h,w,crop_bgr=crop,dilate=3)
        env,per_line=polygons_and_lines(it['text_regions'],h,w)
        inter=lambda a,b: int(np.count_nonzero((a>0)&(b>0)))
        u=int(np.count_nonzero((ocr>0)|(ch>0))) if (ocr is not None and ch is not None) else 0
        iou=inter(ocr,ch)/u if u else None
        lines=[]
        for li,ln in enumerate(per_line):
            li_ink=((ink>0)&(ln>0)).astype(np.uint8)
            t=int(np.count_nonzero(li_ink))
            if t<20: continue
            lines.append(dict(n=t,
                c_ocr=round(inter(li_ink,ocr)/t,3) if ocr is not None else 0.0,
                c_chir=round(inter(li_ink,ch)/t,3) if ch is not None else 0.0,
                txt=(it['text_regions'][li].get('text') or '')[:40]))
        out.append(dict(key=key,i=i,cls=it['class_name'],w=w,h=h,
                        text=(it.get('text') or '')[:70],
                        iou_ocr_chir=round(iou,3) if iou is not None else None,
                        ocr_px=int(np.count_nonzero(ocr)) if ocr is not None else 0,
                        chir_px=int(np.count_nonzero(ch)) if ch is not None else 0,
                        lines=lines))
    del page
    print(key,len(out),flush=True)
json.dump(out,open('scratch/explore/d2_lines.json','w'),ensure_ascii=False)
