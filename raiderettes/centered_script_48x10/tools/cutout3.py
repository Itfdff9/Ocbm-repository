import sys; sys.path.insert(0,"/home/claude/ru")
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
a=np.array(Image.open("src/DILEO.png")); A=a[...,3]>=128
raw=np.load("work/cut_DILEO.npy")
for r in (16,20,24):
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
    op=cv2.morphologyEx(raw.astype(np.uint8),cv2.MORPH_OPEN,k).astype(bool)
    print(r,"removed",int((raw&~op).sum()))
    H,Wd=op.shape
    vis=np.zeros((H,Wd,3),np.uint8); vis[:]=(255,0,255); vis[op]=a[...,:3][op]
    Image.fromarray(vis[400:800,200:1200]).save(f"/tmp/claude-0/rx/cutop{r}.png")
    np.save(f"work/cut_DILEO_op{r}.npy",op)
