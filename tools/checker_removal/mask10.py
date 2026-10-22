from PIL import Image
import numpy as np, cv2
from scipy import ndimage as nd
a=np.asarray(Image.open("src.webp").convert("RGB")).astype(np.float32); I=a.mean(2); sat=a.max(2)-a.min(2)
H,W=I.shape
E=np.load("E.npy"); bg=np.load("bg6.npy")
cells=np.load("altcells_tint.npy")|np.load("altcells4.npy")|np.load("core4.npy")
dcell=nd.distance_transform_edt(~cells)
light=(a.min(2)>=150)&(I>=185)
Lt=light&~bg
lab,n=nd.label(Lt); sz=np.bincount(lab.ravel())
fE=np.bincount(lab.ravel(),weights=(E>=14).ravel())/np.maximum(sz,1)
fN=np.bincount(lab.ravel(),weights=(dcell<=30).ravel())/np.maximum(sz,1)
# how much of the component's border touches removed background
bd=nd.binary_dilation(bg)&Lt
fB=np.bincount(lab.ravel(),weights=bd.ravel())
sel=(np.arange(n+1)>0)&(sz>=80)&(sz<=9000)&((fE>=0.3)|(fN>=0.9))&(fB>=10)
new=sel&~((sz>=80)&(sz<=9000)&(((fE>=0.33)&(fN>=0.3))|(fN>=0.9))&(fB>=10))
for k in np.where(new)[0]:
    ys,xs=np.where(lab==k); print("new",k,int(sz[k]),round(fE[k],2),round(fN[k],2),(ys.min(),ys.max(),xs.min(),xs.max()))
rm=sel[lab]
print("selected",int(sel.sum()),"px",int(rm.sum()))
bg2=bg|rm
for i in range(2):
    ring=nd.binary_dilation(bg2)&~bg2; bg2|=ring&(a.min(2)>175)&(sat<=35)
fg=~bg2; l3,n3=nd.label(fg); s3=np.bincount(l3.ravel()); bg2|=np.isin(l3,np.where(s3<10)[0])&fg
l4,n4=nd.label(bg2); s4=np.bincount(l4.ravel()); bg2&=~(np.isin(l4,np.where(s4<10)[0])&bg2)
np.save("bg10.npy",bg2)
print("bg share",round(bg2.mean(),4))
u8=a.astype(np.uint8); blk=u8.copy(); blk[bg2]=0; mg=u8.copy(); mg[rm]=[255,0,255]
Image.fromarray(np.concatenate([blk,mg],1)).resize((W*4//5,H*2//5)).save("bg10_pair.png")
for nm,(y0,y1,x0,x1) in {"tint":(1060,1220,300,1400),"b0":(640,800,300,1300),"b0r":(640,800,1300,2300),"b2":(1440,1600,300,2300),"top":(0,450,1000,1700)}.items():
    Image.fromarray(np.concatenate([u8[y0:y1,x0:x1],blk[y0:y1,x0:x1]],0)).save(f"x_{nm}.png")
