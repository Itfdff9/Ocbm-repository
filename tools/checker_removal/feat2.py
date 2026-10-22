from PIL import Image
import numpy as np, cv2
a=np.asarray(Image.open("src.webp").convert("RGB")).astype(np.float32); I=a.mean(2); sat=a.max(2)-a.min(2)
H,W=I.shape
M=((sat<=18)&(I>=185)).astype(np.float32)
SIG=7
G=lambda x: cv2.GaussianBlur(x,(0,0),SIG)
def corr(dx,dy):
    # pair (p, p+d); stats in a window centred on p
    Is=np.roll(I,(-dy,-dx),(0,1)); Ms=np.roll(M,(-dy,-dx),(0,1))
    w=M*Ms; Sw=G(w)+1e-6
    m1=G(w*I)/Sw; m2=G(w*Is)/Sw
    v1=np.maximum(G(w*I*I)/Sw-m1*m1,1e-3); v2=np.maximum(G(w*Is*Is)/Sw-m2*m2,1e-3)
    c=(G(w*I*Is)/Sw-m1*m2)/np.sqrt(v1*v2)
    c[Sw<0.25]=0
    # re-centre the window to the pair midpoint
    return np.roll(c,(dy//2,dx//2),(0,1)), np.roll(np.sqrt(v1),(dy//2,dx//2),(0,1))
L=range(14,23)
cx=np.min(np.stack([corr(s,0)[0] for s in L]),0)
cy=np.min(np.stack([corr(0,s)[0] for s in L]),0)
dp=np.max(np.stack([corr(s,s)[0] for s in L]),0)
dm=np.max(np.stack([corr(s,-s)[0] for s in L]),0)
_,sd=corr(0,0)
np.savez("feat2.npz",cx=cx,cy=cy,dp=dp,dm=dm,sd=sd)
zone=(cx<-0.4)&(cy<-0.4)&(dp>0.3)&(dm>0.3)&(sd>8)
np.save("zone.npy",zone)
rgb=a.astype(np.uint8).copy(); rgb[zone]=(rgb[zone]*0.3+np.array([255,0,255])*0.7).astype(np.uint8)
Image.fromarray(rgb).resize((W//2,H//2)).save("zone_half.png")
print("zone share",zone.mean())
