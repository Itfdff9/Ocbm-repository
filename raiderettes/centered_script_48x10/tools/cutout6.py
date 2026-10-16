import numpy as np, cv2, sys
from PIL import Image
from scipy import ndimage as nd
def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
a=np.array(Image.open("src/DILEO.png")); A=a[...,3]>=128
rgb=a[...,:3].astype(np.float32); R,G,B=rgb[...,0],rgb[...,1],rgb[...,2]
S0=np.load("work/cut_DILEO.npy")
for gr in (0.22,0.28):
    dr=A&(R>110)&(G<gr*R)&(B<0.3*R)
    inner=cv2.morphologyEx(dr.astype(np.uint8),cv2.MORPH_CLOSE,disk(7)).astype(bool)
    inner&=nd.binary_dilation(S0,iterations=2)
    lab,k=nd.label(inner); sz=np.bincount(lab.ravel()); sz[0]=0; inner=(sz>3000)[lab]
    inner=nd.binary_fill_holes(inner)|inner
    d=nd.distance_transform_edt(~inner)
    edge=(~A)&nd.binary_dilation(A,iterations=1)&(nd.distance_transform_edt(~S0)<=3)
    v=d[edge&(d<120)]
    print(gr,"inner px",int(inner.sum()),"outline dist p5 %.1f p25 %.1f p50 %.1f p75 %.1f p95 %.1f"%tuple(np.percentile(v,[5,25,50,75,95])))
    np.save(f"work/inner_{gr}.npy",inner)
