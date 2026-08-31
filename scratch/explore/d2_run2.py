import sys, os, glob, pickle, json, time
sys.path.insert(0,'.')
import numpy as np, cv2
from scratch.explore.d2_ink import ink_mask, polygons_and_lines
from pipeline import TranslationPipeline as TP
OPEN_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))

def cover(ink, mask):
    tot=int(np.count_nonzero(ink))
    if tot==0: return None,None
    if mask is None: miss=ink
    else:
        m=mask
        if m.shape[:2]!=ink.shape[:2]:
            m=cv2.resize(m,(ink.shape[1],ink.shape[0]),interpolation=cv2.INTER_NEAREST)
        miss=((ink>0)&~(m>0)).astype(np.uint8)*255
    res=cv2.morphologyEx(miss,cv2.MORPH_OPEN,OPEN_K)
    return 1.0-np.count_nonzero(miss)/tot, float(np.count_nonzero(res))/tot

rows=[]; t0=time.time()
for pkl in sorted(glob.glob('scratch/bareme/cache/*/dets.pkl')):
    key=os.path.basename(os.path.dirname(pkl))
    d=pickle.load(open(pkl,'rb'))
    page=cv2.imread(os.path.join(os.path.dirname(pkl),'page.png'),cv2.IMREAD_COLOR)
    for i,it in enumerate(d['items']):
        x1,y1,x2,y2=[int(v) for v in it['bbox']]
        crop=page[max(0,y1):y2,max(0,x1):x2]
        if crop.size==0: continue
        h,w=crop.shape[:2]; dbg={}
        ink,env=ink_mask(crop,it['text_regions'],h,w,dbg)
        if ink is None or int(np.count_nonzero(ink))<30:
            rows.append(dict(key=key,i=i,skip='ink<30')); continue
        ocr=TP._ocr_mask_from_regions(it['text_regions'],h,w,crop_bgr=crop,dilate=3)
        ch=it.get('chirurgical_mask')
        c_o,r_o=cover(ink,ocr); c_c,r_c=cover(ink,ch)
        nb=dbg['ink_nb']; nbn=int(np.count_nonzero(nb))
        c_o2,_=cover(nb,ocr) if nbn>=30 else (None,None)
        c_c2,_=cover(nb,ch) if nbn>=30 else (None,None)
        _,per_line=polygons_and_lines(it['text_regions'],h,w)
        lines=[]
        for li,ln in enumerate(per_line):
            m=((ink>0)&(ln>0)).astype(np.uint8); t=int(np.count_nonzero(m))
            if t<20: continue
            lines.append([t,
                round(int(np.count_nonzero((m>0)&(ocr>0)))/t,3) if ocr is not None else 0.0,
                round(int(np.count_nonzero((m>0)&(ch>0)))/t,3) if ch is not None else 0.0,
                (it['text_regions'][li].get('text') or '')[:35]])
        rows.append(dict(key=key,i=i,cls=it['class_name'],text=(it.get('text') or '')[:80],
            w=w,h=h,line_h=round(dbg['line_h'],1),thr=round(dbg['thr'],1),
            ink=int(np.count_nonzero(ink)),env=int(np.count_nonzero(env)),ink_nb=nbn,
            cov_ocr=round(c_o,4),cov_chir=round(c_c,4),res_ocr=round(r_o,4),res_chir=round(r_c,4),
            cov_ocr_nb=round(c_o2,4) if c_o2 is not None else None,
            cov_chir_nb=round(c_c2,4) if c_c2 is not None else None,
            ocr_px=int(np.count_nonzero(ocr)) if ocr is not None else 0,
            chir_px=int(np.count_nonzero(ch)) if ch is not None else 0,
            iou=round(int(np.count_nonzero((ocr>0)&(ch>0)))/max(1,int(np.count_nonzero((ocr>0)|(ch>0)))),3) if ocr is not None and ch is not None else None,
            lines=lines))
    del page
    print(key,len(rows),'%.0fs'%(time.time()-t0),flush=True)
json.dump(rows,open('scratch/explore/d2_final.json','w',encoding='utf-8'),ensure_ascii=False)
print('done',len(rows))
