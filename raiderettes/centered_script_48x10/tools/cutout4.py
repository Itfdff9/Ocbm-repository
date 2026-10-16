import numpy as np, cv2, sys
from PIL import Image
from scipy import ndimage as nd
def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def script_cut(a, rc=5, ro=17, rec=2, minarea=20000):
    rgb=a[...,:3].astype(np.int32); A=a[...,3]>=128
    R,G,B=rgb[...,0],rgb[...,1],rgb[...,2]
    warm=A&(R>120)&(R-B>60)&(R-G>40)
    white=A&(R>190)&(G>150)&(B>140)
    darkk=A&(np.maximum(np.maximum(R,G),B)<70)
    w=(warm|(white&nd.binary_dilation(warm,iterations=3))).astype(np.uint8)
    wc=cv2.morphologyEx(w,cv2.MORPH_CLOSE,disk(rc)).astype(bool)&A
    op=cv2.morphologyEx(wc.astype(np.uint8),cv2.MORPH_OPEN,disk(ro)).astype(bool)
    lab,k=nd.label(op); sz=np.bincount(lab.ravel()); sz[0]=0; op=(sz>minarea)[lab]
    cut=op|(wc&(nd.distance_transform_edt(~op)<=rec))
    holes=nd.binary_fill_holes(cut)&~cut
    lab,k=nd.label(holes); sz=np.bincount(lab.ravel()); sz[0]=0
    cut|=(sz<1500)[lab]&(sz>0)[lab]
    cut&=A
    cut=cv2.GaussianBlur(cut.astype(np.float32),(0,0),1.0)>0.5
    return cut
if __name__=="__main__":
    n=sys.argv[1] if len(sys.argv)>1 else "DILEO"
    ro=int(sys.argv[2]) if len(sys.argv)>2 else 17
    a=np.array(Image.open(f"src/{n}.png"))
    cut=script_cut(a,ro=ro)
    np.save(f"work/cut4_{n}.npy",cut)
    H,Wd=cut.shape
    vis=np.zeros((H,Wd,3),np.uint8); vis[:]=(255,0,255); vis[cut]=a[...,:3][cut]
    ys,xs=np.where(cut)
    Image.fromarray(vis[ys.min()-20:ys.max()+20]).resize((2048,int(2048*(ys.max()-ys.min()+40)/Wd))).save(f"/tmp/claude-0/rx/cut4_{n}.png")
    Image.fromarray(vis[400:800,200:1200]).save(f"/tmp/claude-0/rx/cut4z_{n}.png")
    print(n,"cut px",int(cut.sum()), "bbox",xs.min(),xs.max(),ys.min(),ys.max())
