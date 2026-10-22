from PIL import Image
import numpy as np, cv2
from scipy import ndimage as nd
a=np.asarray(Image.open("src.webp").convert("RGB")).astype(np.float32); I=a.mean(2); sat=a.max(2)-a.min(2)
H,W=I.shape
N=(sat<=22)&(I>=188)
hz=(a.min(2)>=165)&(I>=185)&(a[...,2]>=a[...,0]-4)
T=np.load("altcells_tint.npy")&hz
T=nd.binary_dilation(T,iterations=3)&hz
det=np.load("core4.npy")|(np.load("altcells4.npy")&N)
det=nd.binary_dilation(det,iterations=2)&N&(I>=200)
dens=cv2.blur(det.astype(np.float32),(37,37))
near=nd.distance_transform_edt(~det)<=20
fill=(sat<=22)&(I>=200)&(I<=235)&near&(dens>=0.25)
bg=det|fill|T
for i in range(3): bg|=nd.binary_dilation(bg)&N&(I>=200)&(dens>=0.2)
# floating neutral islands (touch nothing but background)
rest=N&~bg
lab,n=nd.label(rest,structure=np.ones((3,3)))
art=~N&~bg
touch=nd.maximum(nd.binary_dilation(art,iterations=2).astype(np.int32),lab,range(1,n+1))
isl=np.isin(lab,1+np.where(touch==0)[0]); bg|=isl
for i in range(2):
    ring=nd.binary_dilation(bg)&~bg; bg|=ring&(a.min(2)>175)&(sat<=35)
fg=~bg
lab,n=nd.label(fg); sz=np.bincount(lab.ravel()); small=np.isin(lab,np.where(sz<10)[0])&fg; bg|=small
lab,n=nd.label(bg); sz=np.bincount(lab.ravel()); holes=np.isin(lab,np.where(sz<10)[0])&bg; bg&=~holes
np.save("bg6.npy",bg)
print("bg share",round(bg.mean(),4),"fill",int(fill.sum()),"islands",int(isl.sum()))
u8=a.astype(np.uint8); blk=u8.copy(); blk[bg]=0
Image.fromarray(blk).resize((W*2//5,H*2//5)).save("bg6_blk.png")
for nm,(y0,y1,x0,x1) in {"left":(550,1150,100,500),"bottom":(1550,1850,600,1300),"rows":(600,800,1300,1700),"top":(0,450,1000,1700),"heart":(0,350,1950,2450),"center":(1000,1250,1100,1600),"tint":(1060,1220,300,1400),"b0":(640,800,300,1300),"b0r":(640,800,1300,2300)}.items():
    Image.fromarray(np.concatenate([u8[y0:y1,x0:x1],blk[y0:y1,x0:x1]],1)).save(f"r_{nm}.png")
