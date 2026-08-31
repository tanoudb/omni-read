import sys, pickle
sys.path.insert(0,'.')
import numpy as np, cv2
from core.renderer import TextRenderer
key,idx=sys.argv[1],int(sys.argv[2])
d=pickle.load(open(f'scratch/bareme/cache/{key}/dets.pkl','rb'))
page=cv2.imread(f'scratch/bareme/cache/{key}/page.png',cv2.IMREAD_COLOR)
it=d['items'][idx]; x1,y1,x2,y2=[int(v) for v in it['bbox']]
crop=page[max(0,y1):y2,max(0,x1):x2]; h,w=crop.shape[:2]
inter=TextRenderer._bubble_mask_from_image(crop)
er=cv2.erode(inter,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)),1)
def ov(m,c):
    o=crop.copy(); o[m>0]=(0.45*o[m>0]+0.55*np.array(c)).astype(np.uint8); return o
sep=np.full((h,6,3),255,np.uint8)
cv2.imwrite(f'scratch/explore/d2_vis/{key}_{idx}_interior.png',
            np.hstack([crop,sep,ov(er,(0,200,255)),sep,ov(it['chirurgical_mask'],(255,0,0))]))
print('interieur non vide colonnes', np.nonzero(er.any(axis=0))[0].min(), np.nonzero(er.any(axis=0))[0].max(), '/ largeur', w)
