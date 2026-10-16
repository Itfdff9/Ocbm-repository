"""DILEO's script cut-out: border = constant offset of the red core (matches how the design draws it), clipped to the measured warm outline."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
a=np.array(Image.open("src/DILEO.png"))
A=a[...,3]>=128
ref=np.load("refcore_DILEO.npy")
d=nd.distance_transform_edt(~ref)
S1=np.load("work/scr_DILEO.npy")
# measure border width where the script meets transparent background
edge=(~A)&nd.binary_dilation(A,iterations=1)&(nd.distance_transform_edt(~S1)<=3)
v=d[edge&(d<120)]
print("border dist at alpha edge: p10 %.1f p50 %.1f p90 %.1f"%tuple(np.percentile(v,[10,50,90])))
for W in (58,60,62,64):
    cut=(d<=W)&nd.binary_dilation(S1,iterations=3)&A
    print(W,int(cut.sum()))
W=62
cut=(d<=W)&nd.binary_dilation(S1,iterations=3)
# keep holes inside the script that are part of DILEO's letters out (only fill tiny pinholes)
holes=nd.binary_fill_holes(cut)&~cut
lab,k=nd.label(holes); sz=np.bincount(lab.ravel()); sz[0]=0
cut|=(sz<400)[lab]&(sz>0)[lab]
cut&=A
np.save("work/cut_DILEO.npy",cut)
H,Wd=cut.shape
for bgc,tag in (((57,255,20),"lime"),((255,0,255),"mag"),((235,235,235),"gray")):
    vis=np.zeros((H,Wd,3),np.uint8); vis[:]=bgc
    vis[cut]=a[...,:3][cut]
    ys,xs=np.where(cut)
    Image.fromarray(vis[ys.min()-20:ys.max()+20]).resize((2048,int(2048*(ys.max()-ys.min()+40)/Wd))).save(f"/tmp/claude-0/rx/cut_{tag}.png")
    Image.fromarray(vis[850:1250,1000:2000]).save(f"/tmp/claude-0/rx/cutz_{tag}.png")
