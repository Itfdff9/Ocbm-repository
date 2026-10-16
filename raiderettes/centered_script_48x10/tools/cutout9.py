import numpy as np, cv2, sys
from PIL import Image
from scipy import ndimage as nd
def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def local_open(a, S0, r=40, zone=40):
    A=a[...,3]>=128
    rgb=a[...,:3].astype(np.int32); R,G,B=rgb[...,0],rgb[...,1],rgb[...,2]
    warm=A&(R>110)&(R-B>50)
    Wout=warm&~S0
    bd=S0&~cv2.erode(S0.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    J=bd&(cv2.dilate(Wout.astype(np.uint8),np.ones((13,13),np.uint8))>0)
    lab,n=nd.label(J,structure=np.ones((3,3))); sz=np.bincount(lab.ravel()); sz[0]=0
    J=(sz>=8)[lab]
    Z=nd.distance_transform_edt(~J)<=zone
    op=cv2.morphologyEx(S0.astype(np.uint8),cv2.MORPH_OPEN,disk(r)).astype(bool)
    M=(S0&~Z)|(op&Z)
    # blend seam: small closing/opening to remove steps at zone borders
    M=cv2.morphologyEx(M.astype(np.uint8),cv2.MORPH_CLOSE,disk(3)).astype(bool)&S0
    M=cv2.morphologyEx(M.astype(np.uint8),cv2.MORPH_OPEN,disk(3)).astype(bool)
    return M&A, J, Z
if __name__=="__main__":
    r=int(sys.argv[1]); zone=int(sys.argv[2])
    a=np.array(Image.open("src/DILEO.png"))
    S0=np.load("work/cut_DILEO.npy")
    M,J,Z=local_open(a,S0,r,zone)
    np.save("work/cut9_DILEO.npy",M)
    print("removed",int((S0&~M).sum()),"junction px",int(J.sum()))
    H,Wd=M.shape
    vis=np.zeros((H,Wd,3),np.uint8); vis[:]=(255,0,255); vis[M]=a[...,:3][M]
    tiles=[vis[400:800,200:1200],vis[850:1250,1000:2000],vis[500:900,2400:3400],vis[700:1100,3000:4000]]
    Image.fromarray(np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0)).resize((1500,600)).save("/tmp/claude-0/rx/cut9z.png")
    vis2=vis.copy(); e=Z&~nd.binary_erosion(Z,iterations=2); vis2[e]=(0,255,255)
    ys,xs=np.where(M)
    Image.fromarray(vis2[ys.min()-20:ys.max()+20]).resize((2048,int(2048*(ys.max()-ys.min()+40)/Wd))).save("/tmp/claude-0/rx/cut9.png")
