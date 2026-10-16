import numpy as np, cv2, sys
from PIL import Image
from scipy import ndimage as nd
def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def peel_cut(a, S0, iters=30, ro=4):
    A=a[...,3]>=128
    med=cv2.medianBlur(a[...,:3],5).astype(np.float32)
    R,G,B=med[...,0]+1,med[...,1],med[...,2]; gr=G/R
    Y=A&(R>120)&(gr>=0.30)&(gr<0.75)
    Wt=A&(med.min(2)>185)
    stop=Y|Wt
    M=S0.copy().astype(np.uint8)
    k=np.ones((3,3),np.uint8)
    for i in range(iters):
        er=cv2.erode(M,k)
        bd=(M>0)&(er==0)
        rm=bd&~stop
        if not rm.any(): break
        M[rm]=0
    M=M.astype(bool)
    M=cv2.morphologyEx(M.astype(np.uint8),cv2.MORPH_OPEN,disk(ro)).astype(bool)
    lab,n=nd.label(M); sz=np.bincount(lab.ravel()); sz[0]=0; M=(sz>5000)[lab]
    holes=nd.binary_fill_holes(M)&~M
    lab,n=nd.label(holes); sz=np.bincount(lab.ravel()); sz[0]=0
    M|=(sz<300)[lab]&(sz>0)[lab]
    M=cv2.GaussianBlur(M.astype(np.float32),(0,0),1.0)>0.5
    return M&A
if __name__=="__main__":
    a=np.array(Image.open("src/DILEO.png"))
    S0=np.load("work/cut_DILEO.npy")
    M=peel_cut(a,S0)
    np.save("work/cut7_DILEO.npy",M)
    print("px",int(M.sum()),"removed from S0",int((S0&~M).sum()))
    H,Wd=M.shape
    vis=np.zeros((H,Wd,3),np.uint8); vis[:]=(255,0,255); vis[M]=a[...,:3][M]
    tiles=[vis[400:800,200:1200],vis[850:1250,1000:2000],vis[500:900,2400:3400],vis[700:1100,3000:4000]]
    Image.fromarray(np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0)).resize((1500,600)).save("/tmp/claude-0/rx/cut7z.png")
    ys,xs=np.where(M)
    Image.fromarray(vis[ys.min()-20:ys.max()+20]).resize((2048,int(2048*(ys.max()-ys.min()+40)/Wd))).save("/tmp/claude-0/rx/cut7.png")
