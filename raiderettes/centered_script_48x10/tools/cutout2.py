import sys; sys.path.insert(0,"/home/claude/ru")
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
from warm2 import wide_warm
a=np.array(Image.open("src/DILEO.png")); A=a[...,3]>=128
wc=wide_warm(a,16)
d=nd.distance_transform_edt(wc)
S0=(nd.distance_transform_edt(~(d>=34))<=34)&wc
lab,k=nd.label(S0); sz=np.bincount(lab.ravel()); sz[0]=0; S0=(sz>20000)[lab]
cut=S0|(wc&(nd.distance_transform_edt(~S0)<=3))
holes=nd.binary_fill_holes(cut)&~cut
lab,k=nd.label(holes); sz=np.bincount(lab.ravel()); sz[0]=0
small=(sz<3000)[lab]&(sz>0)[lab]
cut|=small
cut&=A
# smooth outline slightly (remove 1-2px jaggies) via 4x supersampled contour blur
cut=cv2.GaussianBlur(cut.astype(np.float32),(0,0),1.2)>0.5
np.save("work/cut_DILEO.npy",cut)
print("cut px",int(cut.sum()),"big holes kept",int(((sz>=3000)).sum()))
H,Wd=cut.shape
for bgc,tag in (((57,255,20),"lime"),((255,0,255),"mag")):
    vis=np.zeros((H,Wd,3),np.uint8); vis[:]=bgc; vis[cut]=a[...,:3][cut]
    ys,xs=np.where(cut)
    Image.fromarray(vis[ys.min()-20:ys.max()+20]).resize((2048,int(2048*(ys.max()-ys.min()+40)/Wd))).save(f"/tmp/claude-0/rx/cut_{tag}.png")
    Image.fromarray(vis[850:1250,1000:2000]).save(f"/tmp/claude-0/rx/cutz_{tag}.png")
    Image.fromarray(vis[400:800,200:1200]).save(f"/tmp/claude-0/rx/cutz2_{tag}.png")
