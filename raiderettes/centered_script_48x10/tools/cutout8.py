import numpy as np, cv2, sys
from PIL import Image
from scipy import ndimage as nd
def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def peel(a, S0, iters=30):
    A=a[...,3]>=128
    med=cv2.medianBlur(a[...,:3],5).astype(np.float32)
    R,G,B=med[...,0]+1,med[...,1],med[...,2]; gr=G/R
    stop=(A&(R>120)&(gr>=0.30)&(gr<0.75))|(A&(med.min(2)>185))
    M=S0.copy().astype(np.uint8); k=np.ones((3,3),np.uint8)
    for i in range(iters):
        er=cv2.erode(M,k); bd=(M>0)&(er==0); rm=bd&~stop
        if not rm.any(): break
        M[rm]=0
    return M.astype(bool)
def stub_free(a, S0, depth_min=8):
    P=peel(a,S0)
    removed=S0&~P
    dS=nd.distance_transform_edt(S0)          # depth inside S0
    lab,n=nd.label(removed,structure=np.ones((3,3)))
    if n==0: return S0
    mx=nd.maximum(dS,lab,np.arange(1,n+1))
    stub=np.zeros(n+1,bool); stub[1:]=mx>=depth_min
    M=S0&~stub[lab]
    # tidy: small opening, then fill tiny notches inside S0
    M=cv2.morphologyEx(M.astype(np.uint8),cv2.MORPH_OPEN,disk(3)).astype(bool)
    M=(cv2.morphologyEx(M.astype(np.uint8),cv2.MORPH_CLOSE,disk(5)).astype(bool)&S0)|M
    lab2,n2=nd.label(M); sz=np.bincount(lab2.ravel()); sz[0]=0; M=(sz>5000)[lab2]
    return M, stub[lab]
if __name__=="__main__":
    dm=int(sys.argv[1]) if len(sys.argv)>1 else 8
    a=np.array(Image.open("src/DILEO.png")); A=a[...,3]>=128
    S0=np.load("work/cut_DILEO.npy")
    M,st=stub_free(a,S0,dm)
    M&=A
    np.save("work/cut8_DILEO.npy",M)
    print("removed",int((S0&~M).sum()))
    H,Wd=M.shape
    vis=np.zeros((H,Wd,3),np.uint8); vis[:]=(255,0,255); vis[M]=a[...,:3][M]
    vis2=vis.copy(); vis2[st]=(0,255,255)
    tiles=[vis[400:800,200:1200],vis[850:1250,1000:2000],vis[500:900,2400:3400],vis[700:1100,3000:4000]]
    Image.fromarray(np.concatenate([np.concatenate(tiles[:2],1),np.concatenate(tiles[2:],1)],0)).resize((1500,600)).save("/tmp/claude-0/rx/cut8z.png")
    ys,xs=np.where(M)
    Image.fromarray(vis2[ys.min()-20:ys.max()+20]).resize((2048,int(2048*(ys.max()-ys.min()+40)/Wd))).save("/tmp/claude-0/rx/cut8.png")
