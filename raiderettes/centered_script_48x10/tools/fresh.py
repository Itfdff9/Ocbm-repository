"""Letters rebuilt from scratch in the uniform sequin style; shared centered DILEO script on top."""
import sys; sys.path[:0]=["/home/claude/ru","/home/claude/ru/tools","/home/claude/rd"]
import numpy as np, cv2, os
from PIL import Image
from scipy import ndimage as nd
import importlib.util
spec=importlib.util.spec_from_file_location("b","/home/claude/ru/build.py"); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
import engine
def _empty_script(txt,height_px,target_w,border=None,widen_max=2.4):
    h=max(int(height_px),8); w=h
    return np.zeros((h,w,3),np.float32), np.zeros((h,w),bool), 1.0, height_px/100.0
engine.script_layer=_empty_script
from engine import compose
FW,FH=14400,3000
def letters_only(name):
    img,A=compose(name,FW,FH,'straight',letter_h_frac=0.90,script_h_frac=0.62,script_w_frac=0.80,script_y_frac=0.58,
                  kmin=0.45,kmax=1.6,margin_frac=0.004,gap_frac=0.03,seed=(sum(map(ord,name))%89+5),widen_max=1.9,
                  script_text="x")
    return img,A
def main(name):
    img,A=letters_only(name)
    # stretch to full 48x10 like the others (letters box -> canvas)
    ys,xs=np.where(A); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    c=img[y0:y1,x0:x1]; ca=A[y0:y1,x0:x1]
    _,(iy,ix)=nd.distance_transform_edt(~ca,return_indices=True); c=c[iy,ix]
    r=cv2.resize(np.clip(c,0,255).astype(np.uint8),(FW,FH),interpolation=cv2.INTER_CUBIC)
    Af=cv2.resize(ca.astype(np.float32),(FW,FH),interpolation=cv2.INTER_LINEAR)>=0.5
    srgb,sal,shf=b.script_final()
    out=np.zeros((FH,FW,4),np.uint8)
    rf=r.astype(np.float32)*(1-shf)[...,None]
    out[...,:3]=np.clip(rf,0,255).astype(np.uint8); out[...,3]=np.where(Af,255,0)
    out[sal,:3]=srgb[sal]; out[sal,3]=255
    out[out[...,3]==0,:3]=0
    Image.fromarray(out,"RGBA").save(f"/home/claude/ru/final/{name}_Raiderettes_CENTERED_48x10in_300dpi.png",dpi=(300,300))
    b.previews(name,out)
    print(name,"done")
if __name__=="__main__":
    for n in sys.argv[1:]: main(n)
