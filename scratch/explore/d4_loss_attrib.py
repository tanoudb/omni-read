import sys, pickle, csv; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np, cv2
from pipeline import TranslationPipeline as TP
from core.renderer import TextRenderer as TR
nz=lambda m:int(np.count_nonzero(m))
rows=[]
for d in sorted(Path('scratch/bareme/cache').iterdir()):
    data=pickle.load(open(d/'dets.pkl','rb')); page=cv2.imread(str(d/'page.png'))
    for i,it in enumerate(data['items']):
        x1,y1,x2,y2=it['bbox']; crop=page[max(0,y1):y2,max(0,x1):x2]
        if crop.size==0: continue
        h,w=crop.shape[:2]; regions=it.get('text_regions') or []; cn=str(it.get('class_name','')).lower()
        ocr=TP._ocr_mask_from_regions(regions,h,w,crop_bgr=crop)
        if ocr is None: rows.append(dict(key=f'{d.name}#{i}',cls=cn,ocr_none=1)); continue
        m=ocr; r_int=1.0; applied_int=False
        if cn!='out_text':
            interior=TR._bubble_mask_from_image(crop)
            if interior is not None and interior.shape[:2]==(h,w):
                interior=cv2.erode(interior,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),1)
                b=cv2.bitwise_and(m,interior); ratio=nz(b)/max(1,nz(m))
                if ratio>=0.5: m=b; applied_int=True; r_int=ratio
                else: r_int=ratio
        after_int=m.copy()
        bub=it.get('mask_binary'); r_bub=1.0; applied_bub=False
        if bub is not None:
            if bub.ndim==3: bub=bub[:,:,0]
            if bub.shape[:2]!=(h,w): bub=cv2.resize(bub,(w,h),interpolation=cv2.INTER_NEAREST)
            bub=(bub>0).astype(np.uint8)*255
            s=max(3,int(round(min(h,w)*0.025)))
            er=cv2.erode(bub,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*s+1,2*s+1)),1)
            if nz(er)>=0.35*nz(bub): bub=er
            inter=cv2.bitwise_and(m,bub); ratio=nz(inter)/max(1,nz(m))
            if int(np.sum(inter))>0.30*int(np.sum(m)): m=inter; applied_bub=True
            r_bub=ratio
        rows.append(dict(key=f'{d.name}#{i}',cls=cn,ocr_none=0,
            r_int=round(r_int,3),app_int=int(applied_int),r_bub=round(r_bub,3),app_bub=int(applied_bub),
            final=round(nz(m)/max(1,nz(ocr)),3)))
with open('scratch/explore/d4_loss_attrib.csv','w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=['key','cls','ocr_none','r_int','app_int','r_bub','app_bub','final'],extrasaction='ignore')
    w.writeheader(); w.writerows(rows)
import statistics as st
ok=[r for r in rows if not r.get('ocr_none')]
print('n',len(rows),'ocr_mask None:',sum(r.get('ocr_none',0) for r in rows))
ints=[r for r in ok if r['app_int']]
print('interior applique:',len(ints),' garde-fou 0.5 declenche (rejet):',sum(1 for r in ok if r['cls']!='out_text' and not r['app_int'] and r['r_int']<1.0))
print('  perte due a interior seul, mediane %.3f  p10 %.3f  <0.7: %d  <0.6: %d'%(
    st.median([r['r_int'] for r in ints]), sorted(r['r_int'] for r in ints)[len(ints)//10],
    sum(1 for r in ints if r['r_int']<0.7), sum(1 for r in ints if r['r_int']<0.6)))
bubs=[r for r in ok if r['app_bub']]
print('bubble applique:',len(bubs),' rejete:',sum(1 for r in ok if not r['app_bub']))
print('  perte due a bubble seul, mediane %.3f  <0.9: %d  <0.7: %d'%(
    st.median([r['r_bub'] for r in bubs]), sum(1 for r in bubs if r['r_bub']<0.9), sum(1 for r in bubs if r['r_bub']<0.7)))
print()
print('cas ou interior passe JUSTE le garde-fou (0.50 <= r_int < 0.60):')
for r in sorted(ok,key=lambda r:r['r_int'])[:12]:
    print('  ',r)
