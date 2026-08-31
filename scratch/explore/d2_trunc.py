import sys, os, glob, pickle, json
sys.path.insert(0,'.')
import numpy as np, cv2
from scratch.explore.d2_ink import ink_mask
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
        mb=it.get('mask_binary')
        rec=dict(key=key,i=i,cls=it['class_name'],w=w,h=h,
                 text=(it.get('text') or '')[:70])
        ys,xs=np.nonzero(ink)
        rec['ink_x']=[int(xs.min()),int(xs.max())]; rec['ink_y']=[int(ys.min()),int(ys.max())]
        for nm,m in (('chir',ch),('mb',mb)):
            if m is None or np.count_nonzero(m)==0:
                rec[nm+'_x']=None; continue
            ys2,xs2=np.nonzero(m)
            rec[nm+'_x']=[int(xs2.min()),int(xs2.max())]
            rec[nm+'_y']=[int(ys2.min()),int(ys2.max())]
            rec[nm+'_shape']=list(m.shape)
        # part de l'encre au-dela du dernier x du masque chir
        if rec.get('chir_x'):
            xm=rec['chir_x'][1]; ym=rec['chir_y'][1]
            rec['ink_beyond_x']=round(float(np.mean(xs>xm)),4)
            rec['ink_beyond_y']=round(float(np.mean(ys>ym)),4)
        out.append(rec)
    del page
    print(key,len(out),flush=True)
json.dump(out,open('scratch/explore/d2_trunc.json','w'),ensure_ascii=False)
