import numpy as np, cv2
from scipy import ndimage as nd
def axis_of(clean, box):
    x0,y0,x1,y1=box
    ys,xs=np.where(clean[y0:y1,x0:x1]); pts=np.stack([xs,ys],1).astype(float); pts-=pts.mean(0)
    w,v=np.linalg.eigh(np.cov(pts.T)); d=v[:,np.argmax(w)]
    return -d if d[1]>0 else d
def chunk_fill(rgb, mask, clean, d, chunk=150, ts=None):
    ts=ts or sorted(list(range(200,1600,20))+list(range(-200,-1400,-20)),key=abs)
    H,W=mask.shape; out=rgb.copy()
    yy,xx=np.where(mask); proj=xx*d[0]+yy*d[1]
    lo,hi=proj.min(),proj.max(); n=max(1,int(np.ceil((hi-lo)/chunk)))
    for k in range(n):
        a,b=lo+k*(hi-lo)/n, lo+(k+1)*(hi-lo)/n
        sel=(proj>=a-25)&(proj<=b+25)
        cy,cx=yy[sel],xx[sel]
        best=None
        for t in ts:
            dx,dy=int(round(d[0]*t)),int(round(d[1]*t))
            sy,sx=cy+dy,cx+dx
            ok=(sy>=0)&(sy<H)&(sx>=0)&(sx<W)
            if ok.mean()<1: continue
            frac=clean[sy,sx].mean()
            if frac>0.995: best=(dx,dy); break
        if best is None: print('chunk',k,'no clean source'); continue
        dx,dy=best
        cm=np.zeros((H,W),np.uint8); cm[cy,cx]=255; cm=cv2.dilate(cm,np.ones((7,7),np.uint8))
        y0,y1=np.where(cm.any(1))[0][[0,-1]]; x0,x1=np.where(cm.any(0))[0][[0,-1]]
        srcpatch=rgb[y0+dy:y1+dy+1, x0+dx:x1+dx+1].copy()
        m=cm[y0:y1+1,x0:x1+1]
        center=((x0+x1+1)//2,(y0+y1+1)//2)
        out=cv2.seamlessClone(srcpatch, out, m, center, cv2.NORMAL_CLONE)
    return out
