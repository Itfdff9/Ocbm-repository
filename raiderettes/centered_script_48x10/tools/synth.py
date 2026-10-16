"""Render the rebuild zone from the completed label map: sequin interior via shift-map restricted to deep
interior, border/inline/black bands from the source's depth-colour profile + stitch noise."""
import sys; sys.path[:0]=["/home/claude/ru","/home/claude/ru/tools"]
import numpy as np, cv2, json, os
from PIL import Image
from scipy import ndimage as nd
import importlib.util
spec=importlib.util.spec_from_file_location("b","/home/claude/ru/build.py"); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
from labels import labels
import structfill, shiftmap
def run(n, bw=35):
    a=b.load(n); H,W=a.shape[:2]
    L=labels(a); hole0=np.load(f"/home/claude/ru/work/hole_{n}.npy")
    hole0=b.bars_removed(n,hole0)
    B=b.letters_box(n); base=B[3]-1; Hl=B[3]-B[1]
    Lc,hole=structfill.complete(L,hole0,base,bw,int(0.21*Hl),int(0.027*Hl))
    S=Lc!=0
    d_in=nd.distance_transform_edt(S)
    known=~hole
    out=a.copy()
    DEEP=bw+40
    prof=np.zeros((DEEP+20,3)); cnt=np.zeros(DEEP+20)
    m=known&S
    dk=np.clip(d_in[m].astype(int),0,DEEP+19)
    np.add.at(prof,dk,a[...,:3][m].astype(np.float64)); np.add.at(cnt,dk,1)
    prof=prof/np.maximum(cnt,1)[:,None]
    deep=S&(d_in>DEEP)
    fill_deep=deep&hole
    known_deep=deep&known&(nd.distance_transform_edt(~hole)>30)
    g=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY).astype(np.float32)
    ys_k,xs_k=np.where(known_deep)
    cx=int(np.median(xs_k)); col=g[:,cx]; kn=known_deep[:,cx]
    seg=col[kn]-col[kn].mean()
    ac=np.correlate(seg,seg,mode="full")[len(seg)-1:]
    per=int(np.argmax(ac[20:120])+20)
    print("   sequin period",per)
    fy,fx=np.where(fill_deep)
    src_y=fy.copy(); ok=np.zeros(len(fy),bool)
    for k in range(1,60):
        cand=fy-k*per; inb=cand>=0
        hit=inb.copy(); hit[inb]=known_deep[cand[inb],fx[inb]]
        take=hit&~ok; src_y[take]=cand[take]; ok|=take
        if ok.all(): break
    fx2=fx.copy()
    if (~ok).any():
        _,(iy,ix)=nd.distance_transform_edt(~known_deep,return_indices=True)
        src_y[~ok]=iy[fy[~ok],fx[~ok]]; fx2[~ok]=ix[fy[~ok],fx[~ok]]
    out[fy,fx]=a[src_y,fx2]; out[fy,fx,3]=255
    band=S&hole&~deep
    src_band=S&known&~deep&(nd.distance_transform_edt(~hole)>30)
    dq=np.clip(np.round(d_in).astype(int),0,DEEP+19)
    by,bx=np.where(band)
    sy=by.copy(); ok=np.zeros(len(by),bool)
    # same column upward, same depth (+-2), period-snapped
    for k in range(1,80):
        cand=by-k*per; inb=cand>=0
        hit=inb.copy()
        hit[inb]=src_band[cand[inb],bx[inb]]&(np.abs(dq[cand[inb],bx[inb]]-dq[by[inb],bx[inb]])<=2)
        take=hit&~ok; sy[take]=cand[take]; ok|=take
        if ok.all(): break
    out[by[ok],bx[ok]]=a[sy[ok],bx[ok]]
    rest=~ok
    if rest.any():
        from sklearn.neighbors import KDTree
        ky,kx=np.where(src_band)
        feats=np.stack([ky,kx,dq[ky,kx]*25.0],1).astype(np.float32)
        tree=KDTree(feats)
        q=np.stack([by[rest],bx[rest],dq[by[rest],bx[rest]]*25.0],1).astype(np.float32)
        _,ind=tree.query(q,k=1)
        out[by[rest],bx[rest]]=a[ky[ind[:,0]],kx[ind[:,0]]]
    out[by,bx,3]=255
    out[hole&~S]=0
    out[base+4:]=0
    return out, hole
if __name__=="__main__":
    n=sys.argv[1]
    out,hole=run(n)
    Image.fromarray(out).save(f"/home/claude/ru/work/clean_{n}.png")
    np.save(f"/home/claude/ru/work/fill_{n}.npy",hole)
    out=b.fix_small(n,out)
    fin=b.final(n,out)
    Image.fromarray(fin,"RGBA").save(f"/home/claude/ru/final/{n}_Raiderettes_CENTERED_48x10in_300dpi.png",dpi=(300,300))
    b.previews(n,fin)
    print(n,"done")
