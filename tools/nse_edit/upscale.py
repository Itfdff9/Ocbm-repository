# Tiled Real-ESRGAN x4 via ncnn (CPU). Usage: upscale.py model in.png out.png  (RGB only; alpha handled separately)
import sys, numpy as np, ncnn, time
from PIL import Image
M='/home/claude/nse_edit/models/rg/models/'
def up(img, model='realesrgan-x4plus', tile=192, pad=12):
    net=ncnn.Net(); net.opt.use_vulkan_compute=False; net.opt.num_threads=2
    net.load_param(M+model+'.param'); net.load_model(M+model+'.bin')
    a=np.asarray(img.convert('RGB')).astype(np.float32)/255.
    H,W,_=a.shape; out=np.zeros((H*4,W*4,3),np.float32)
    ap=np.pad(a,((pad,pad),(pad,pad),(0,0)),mode='reflect')
    t0=time.time(); n=0; tot=((H+tile-1)//tile)*((W+tile-1)//tile)
    for y in range(0,H,tile):
        for x in range(0,W,tile):
            h=min(tile,H-y); w=min(tile,W-x)
            blk=ap[y:y+h+2*pad, x:x+w+2*pad].transpose(2,0,1).copy()
            ex=net.create_extractor(); ex.input('data',ncnn.Mat(blk))
            _,o=ex.extract('output'); o=np.array(o).transpose(1,2,0)
            out[y*4:(y+h)*4, x*4:(x+w)*4]=o[pad*4:(pad+h)*4, pad*4:(pad+w)*4]
            n+=1
            if n%20==0: print(n,'/',tot, round(time.time()-t0),'s',flush=True)
    return Image.fromarray((np.clip(out,0,1)*255+0.5).astype(np.uint8))
if __name__=='__main__':
    up(Image.open(sys.argv[2]),sys.argv[1]).save(sys.argv[3])
