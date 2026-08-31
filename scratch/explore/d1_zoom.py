import sys, pickle, numpy as np, cv2
sys.path.insert(0,'.')
CACHE='scratch/bareme/cache/30-years-have-passed-since-the-prologue__p01'
d=pickle.load(open(CACHE+'/dets.pkl','rb')); it=d['items'][14]
page=cv2.imread(CACHE+'/page.png'); x1,y1,x2,y2=it['bbox']
crop=page[y1:y2,x1:x2]; h,w=crop.shape[:2]
gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
sim=(np.abs(gray.astype(np.int16)-255.0)<40).astype(np.uint8)*255
z=(slice(90,200), slice(400,587))
a=cv2.resize(crop[z], None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
b=cv2.resize(cv2.cvtColor(sim[z],cv2.COLOR_GRAY2BGR), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
cv2.imwrite('scratch/explore/d1_zoom.png', np.vstack([a,b]))
print('ok')
