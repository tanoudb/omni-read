import sys, pickle, os
sys.path.insert(0,'.')
import numpy as np, cv2
from scratch.explore.d2_ink import ink_mask
from pipeline import TranslationPipeline as TP
key, idx = sys.argv[1], int(sys.argv[2])
d = pickle.load(open(f'scratch/bareme/cache/{key}/dets.pkl','rb'))
page = cv2.imread(f'scratch/bareme/cache/{key}/page.png', cv2.IMREAD_COLOR)
it = d['items'][idx]; x1,y1,x2,y2 = it['bbox']
crop = page[max(0,y1):y2, max(0,x1):x2]; h,w = crop.shape[:2]
dbg={}; ink,env = ink_mask(crop, it['text_regions'], h, w, dbg)
ocr = TP._ocr_mask_from_regions(it['text_regions'], h, w, crop_bgr=crop, dilate=3)
chir = it.get('chirurgical_mask')
miss_ocr = ((ink>0) & ~(ocr>0)).astype(np.uint8)*255
miss_chir = ((ink>0) & ~(chir>0)).astype(np.uint8)*255
def ov(m,c):
    o=crop.copy(); o[m>0]=(0.2*o[m>0]+0.8*np.array(c)).astype(np.uint8); return o
sep=np.full((h,6,3),255,np.uint8)
cv2.imwrite(f'scratch/explore/d2_vis/{key}_{idx}_miss.png', np.hstack([ov(miss_ocr,(0,0,255)),sep,ov(miss_chir,(255,0,0))]))
print('ink',int(np.count_nonzero(ink)),'miss_ocr',int(np.count_nonzero(miss_ocr)),'miss_chir',int(np.count_nonzero(miss_chir)))
