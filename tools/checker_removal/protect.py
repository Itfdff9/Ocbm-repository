from PIL import Image
import numpy as np, cv2
from scipy import ndimage as nd
a=np.asarray(Image.open("src.webp").convert("RGB")).astype(np.float32); I=a.mean(2); sat=a.max(2)-a.min(2)
H,W=I.shape
bg=np.load("bg10.npy"); bg0=np.load("bg.npy")      # bg.npy = edge flood fill (outer checker)
dark=nd.binary_dilation(a.max(2)<110,iterations=2)
lab,n=nd.label(~dark)
sz=np.bincount(lab.ravel())
border=np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]]))
fr=np.bincount(lab.ravel(),weights=bg.ravel())/np.maximum(sz,1)
prot=np.zeros(n+1,bool)
for k in range(1,n+1):
    if sz[k]>=2000 and k not in border and fr[k]<0.7 and fr[k]>0: prot[k]=True
P=prot[lab]
restore=bg&P&~bg0
# restored area also includes the 2px dark-dilation ring belonging to the component edge
restore|=bg&nd.binary_dilation(P,iterations=3)&~bg0&~P&(a.max(2)>=110)&nd.binary_dilation(restore,iterations=3)
print("protected comps",int(prot.sum()),"restored px",int(restore.sum()))
for k in np.where(prot)[0]:
    ys,xs=np.where(lab==k); print(k,int(sz[k]),round(fr[k],2),(ys.min(),ys.max(),xs.min(),xs.max()))
bg2=bg&~restore
np.save("bg11.npy",bg2)
u8=a.astype(np.uint8); blk=u8.copy(); blk[bg2]=0; mg=u8.copy(); mg[restore]=[0,255,0]
Image.fromarray(np.concatenate([blk,mg],1)).resize((W*4//5,H*2//5)).save("bg11_pair.png")
