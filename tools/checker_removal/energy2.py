from PIL import Image
import numpy as np, cv2
from scipy import ndimage as nd
a=np.asarray(Image.open("src.webp").convert("RGB")).astype(np.float32); I=a.mean(2)
B=cv2.blur(I,(5,5))
E=np.zeros(I.shape,np.float32)
for ax in (0,1):
    for s in range(15,22):
        d1=np.roll(B,-s,ax)-B; d0=B-np.roll(B,s,ax)
        ok=(np.roll(B,-s,ax)>=190)&(np.roll(B,s,ax)>=190)&(B>=190)&(np.abs(np.abs(d1)-np.abs(d0))<12)
        e=np.where(ok,np.minimum(np.sqrt(np.maximum(0,-d1*d0)),45),0)
        E=np.maximum(E,e)
np.save("E.npy",E)
bg=np.load("bg6.npy")
Lt=(a.min(2)>=150)&(I>=185)&~bg
lab,n=nd.label(Lt); sz=np.bincount(lab.ravel())
hiE=np.bincount(lab.ravel(),weights=(E>=14).ravel())/np.maximum(sz,1)
mE=np.bincount(lab.ravel(),weights=E.ravel())/np.maximum(sz,1)
for k in (868,615,662,123,645,412):
    print(k,int(sz[k]),"fracE>=14",round(hiE[k],2),"meanE",round(mE[k],1))
# reference: white letter tiles
for (y,x) in ((560,600),(900,1500),(1300,1800),(560,1150)):
    k=lab[y,x]; print("tile?",(y,x),k,int(sz[k]) if k else 0, round(hiE[k],2) if k else None, round(mE[k],1) if k else None)
