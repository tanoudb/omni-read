# Recensement RAPIDE : premiere sortie de l'arbre (flat_fill) seulement.
import sys, pickle, csv, time; sys.path.insert(0,'.')
from pathlib import Path
import numpy as np, cv2
from core.renderer import TextRenderer as TR
from config import config
nz=lambda m:int(np.count_nonzero(m))
rows=[]
for d in sorted(Path('scratch/bareme/cache').iterdir()):
    data=pickle.load(open(d/'dets.pkl','rb')); page=cv2.imread(str(d/'page.png'))
    H,W=page.shape[:2]
    for i,it in enumerate(data['items']):
        x1,y1,x2,y2=it['bbox']; cn=str(it.get('class_name','')).lower(); bh=y2-y1
        regions=it.get('text_regions') or []; chir=it.get('chirurgical_mask')
        if bh<TR.INPAINT_MIN_HEIGHT: rows.append(dict(key=f'{d.name}#{i}',cls=cn,branch='skip_petit')); continue
        m=max(TR.CROP_MARGIN*2,2*bh) if cn in ('out_text','system') else max(TR.CROP_MARGIN,bh)
        cx1,cy1,cx2,cy2=max(0,x1-m),max(0,y1-m),min(W,x2+m),min(H,y2+m)
        crop=page[cy1:cy2,cx1:cx2].copy(); ch,cw=crop.shape[:2]; ox,oy=x1-cx1,y1-cy1
        blk=TR._block_mask_from_regions(cw,ch,regions,ox,oy) if regions else None
        if cn in ('out_text','system') and blk is not None and nz(blk)>0:
            local,src=blk,'block_mask'
        elif chir is not None and chir.size>0:
            dh,dw=max(1,y2-y1),max(1,x2-x1)
            cm=chir if chir.shape[:2]==(dh,dw) else cv2.resize(chir,(dw,dh),interpolation=cv2.INTER_NEAREST)
            local=np.zeros((ch,cw),np.uint8); local[oy:oy+dh,ox:ox+dw]=(cm>0).astype(np.uint8)*255
            k=config.rendering.out_text_mask_dilate_kernel if cn in ('out_text','system') else config.rendering.inpaint_mask_dilate_kernel
            local=cv2.dilate(local,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),1); src='chirurgical'
        else: rows.append(dict(key=f'{d.name}#{i}',cls=cn,branch='skip')); continue
        lb=None
        if it.get('mask_binary') is not None:
            dh,dw=max(1,y2-y1),max(1,x2-x1); bm=it['mask_binary']
            if bm.shape[:2]!=(dh,dw): bm=cv2.resize(bm,(dw,dh),interpolation=cv2.INTER_NEAREST)
            lb=np.zeros((ch,cw),np.uint8); lb[oy:oy+dh,ox:ox+dw]=(bm>0).astype(np.uint8)*255
        flat=TR._flat_fill_color(crop,local,local_bubble_mask=lb,class_name=cn)
        rows.append(dict(key=f'{d.name}#{i}',cls=cn,src=src,branch='flat_fill' if flat is not None else 'smooth_ou_lama',
                         mask_frac=round(nz(local)/(ch*cw),4)))
    print('  ',d.name,len(rows),flush=True)
with open('scratch/explore/d4_flat_census.csv','w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=['key','cls','src','branch','mask_frac'],extrasaction='ignore'); w.writeheader(); w.writerows(rows)
from collections import Counter
print('n=',len(rows)); print(Counter(r['branch'] for r in rows).most_common()); print(Counter(r.get('src') for r in rows).most_common())
for cl in ('bulle','out_text','system'):
    sub=[r for r in rows if r['cls']==cl]; print(cl,len(sub),Counter(r['branch'] for r in sub).most_common(),Counter(r.get('src') for r in sub).most_common())
