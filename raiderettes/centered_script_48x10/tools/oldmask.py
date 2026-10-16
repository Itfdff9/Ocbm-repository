"""Old-script hole = opening of the bridged warm mask by a disk wider than any letter border (script strokes are wide warm blobs)."""
import sys; sys.path.insert(0,"/home/claude/ru")
import numpy as np, cv2, json
from PIL import Image
from scipy import ndimage as nd
from warm2 import wide_warm
NAMES=["SHEA","RENDA","VALENTI","CORSO","TRAVA","MCHUGH","VELEZ","DILEO","MATTEO","PALENCIA"]
def oldmask(a, rc=16, r0=34, rec=16, grow=6, minarea=20000):
    A=a[...,3]>128
    wc=wide_warm(a,rc)
    d=nd.distance_transform_edt(wc)
    seed=d>=r0
    S0=nd.distance_transform_edt(~seed)<=r0          # opening
    S0&=wc
    lab,k=nd.label(S0); sz=np.bincount(lab.ravel()); sz[0]=0; S0=(sz>minarea)[lab]
    near=nd.distance_transform_edt(~S0)<=rec
    S1=S0|(wc&near)
    # fill small holes inside the script (white dots, keyline bits, tiny counters)
    holes=nd.binary_fill_holes(S1)&~S1
    lab,k=nd.label(holes); sz=np.bincount(lab.ravel()); sz[0]=0
    S1|=(sz<2500)[lab]&(sz>0)[lab]
    H=nd.distance_transform_edt(~S1)<=grow
    # soft alpha fringe of the script over transparent background
    soft=(a[...,3]>0)&(a[...,3]<250)
    H|=soft&(nd.distance_transform_edt(~H)<=10)
    return H,S1
if __name__=="__main__":
    out={}
    for n in sys.argv[1:] or NAMES:
        a=np.array(Image.open(f"src/{n}.png")); Hh,W=a.shape[:2]
        H,S1=oldmask(a)
        np.save(f"work/hole_{n}.npy",H); np.save(f"work/scr_{n}.npy",S1)
        vis=a[...,:3].copy(); vis[a[...,3]<128]=(0,150,0)
        e=H&~nd.binary_erosion(H,iterations=3); vis[e]=(0,255,255)
        ys,xs=np.where(H); y0=max(ys.min()-40,0); y1=min(ys.max()+40,Hh)
        Image.fromarray(vis[y0:y1,:W//2+60]).resize(((W//2+60)//2,(y1-y0)//2)).save(f"/tmp/claude-0/rx/om_{n}_L.png")
        Image.fromarray(vis[y0:y1,W//2-60:]).resize(((W-W//2+60)//2,(y1-y0)//2)).save(f"/tmp/claude-0/rx/om_{n}_R.png")
        print(n,"hole px",int(H.sum()))
