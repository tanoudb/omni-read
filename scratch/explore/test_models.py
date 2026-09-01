import torch,numpy as np,time,sys
from lama_cleaner.model_manager import ModelManager
from lama_cleaner.schema import Config as LamaConfig
cfg=LamaConfig(hd_strategy="Original", ldm_steps=20, hd_strategy_crop_margin=64,
               hd_strategy_crop_trigger_size=1024, hd_strategy_resize_limit=2048)
img=(np.random.rand(320,320,3)*255).astype('uint8')
mask=np.zeros((320,320),'uint8'); mask[120:170,90:230]=255
for name in ['manga','mat','zits','fcf','ldm']:
    try:
        t=time.time(); mm=ModelManager(name=name, device=torch.device("cuda"))
        out=mm(img, mask, cfg)
        print("OK   %-6s shape=%s dt=%.1fs"%(name,out.shape,time.time()-t),flush=True)
        del mm; torch.cuda.empty_cache()
    except Exception as e:
        print("FAIL %-6s %s"%(name,repr(e)[:160]),flush=True)
print("ALLDONE",flush=True)
