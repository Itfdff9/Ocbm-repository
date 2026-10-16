"""Old-script segmentation: red/magenta density -> thick core -> border width from alpha edges -> silhouette."""
import numpy as np, cv2, sys, json
from PIL import Image
from scipy import ndimage as nd
NAMES=["SHEA","RENDA","VALENTI","CORSO","TRAVA","MCHUGH","VELEZ","DILEO","MATTEO","PALENCIA"]
def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def hsv(a):
    h=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2HSV_FULL).astype(np.float32)
    H=h[...,0]*360/256; H=np.where(H>300,H-360,H); return H,h[...,1],h[...,2]
def core_of(a, hmax=9, win=25, thr=0.45, ropen=8, minarea=3000):
    A=a[...,3]>128; H,S,V=hsv(a)
    red=A&(H<hmax)&(S>110)&(V>70)
    den=cv2.blur(red.astype(np.float32),(win,win))
    c=(den>thr)&A
    c=cv2.morphologyEx(c.astype(np.uint8),cv2.MORPH_OPEN,disk(ropen)).astype(bool)
    lab,k=nd.label(c); sz=np.bincount(lab.ravel()); sz[0]=0; c=(sz>minarea)[lab]
    c=cv2.morphologyEx(c.astype(np.uint8),cv2.MORPH_CLOSE,disk(6)).astype(bool)
    return c,den
def border_width(a,core):
    A=a[...,3]>128
    d=nd.distance_transform_edt(~core)
    edge=(~A)&nd.binary_dilation(A,iterations=1)      # transparent pixels touching opaque
    v=d[edge&(d<140)&(d>0)]
    hist=np.bincount(v.astype(int),minlength=141)
    return hist
if __name__=="__main__":
    P=json.load(open("segparams.json")) if len(sys.argv)<2 or sys.argv[1]!="-" else {}
    for n in [x for x in sys.argv[1:] if x!="-"] or NAMES:
        p=P.get(n,{})
        a=np.array(Image.open(f"src/{n}.png"))
        core,den=core_of(a,**p.get("core",{}))
        np.save(f"core_{n}.npy",core)
        hist=border_width(a,core)
        top=np.argsort(hist[5:])[::-1][:6]+5
        print(n,"core",int(core.sum()),"alpha-edge dist peaks",top.tolist(), "counts",hist[top].tolist())
        vis=a[...,:3].astype(np.float32)*0.5; vis[a[...,3]<128]=(0,90,0); vis[core]=(255,255,0)
        Image.fromarray(vis.astype(np.uint8)).resize((a.shape[1]//4,a.shape[0]//4)).save(f"/tmp/claude-0/rx/{n}_core.png")
