import numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
a=np.array(Image.open("src/DILEO.png")); A=a[...,3]>=128
raw=np.load("work/cut_DILEO.npy")          # cutout2 result (S0 + 3px)
for r in (26,30):
    op=cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_OPEN,disk(r)).astype(bool)
    print(r,"removed",int((raw&~op).sum()))
    np.save(f"work/cut5_{r}.npy",op)
    H,Wd=op.shape
    vis=np.zeros((H,Wd,3),np.uint8); vis[:]=(255,0,255); vis[op]=a[...,:3][op]
    ys,xs=np.where(op)
    Image.fromarray(vis[ys.min()-20:ys.max()+20]).resize((2048,int(2048*(ys.max()-ys.min()+40)/Wd))).save(f"/tmp/claude-0/rx/cut5_{r}.png")
    tiles=[vis[400:800,200:1200],vis[850:1250,1000:2000],vis[500:900,2400:3400],vis[700:1100,3000:4000]]
    Image.fromarray(np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0)).resize((1500,600)).save(f"/tmp/claude-0/rx/cut5z_{r}.png")
