import numpy as np, cv2
from scipy import ndimage as nd
def shift_fill(rgb, mask, clean, box, ts=range(260,1400,40), side=+1):
    x0,y0,x1,y1=box
    ys,xs=np.where(clean[y0:y1,x0:x1]); pts=np.stack([xs,ys],1).astype(float); pts-=pts.mean(0)
    w,v=np.linalg.eigh(np.cov(pts.T)); d=v[:,np.argmax(w)]  # principal axis (x,y)
    if d[1]>0: d=-d   # point "up" the ribbon
    out=rgb.copy(); todo=mask.copy()
    H,W=mask.shape
    for t in ts:
        dx,dy=int(round(d[0]*t)),int(round(d[1]*t))
        yy,xx=np.where(todo)
        sy,sx=yy+dy,xx+dx
        ok=(sy>=0)&(sy<H)&(sx>=0)&(sx<W)
        ok[ok]&=clean[sy[ok],sx[ok]]
        out[yy[ok],xx[ok]]=rgb[sy[ok],sx[ok]]
        todo[yy[ok],xx[ok]]=False
        if not todo.any(): break
    print('axis',d.round(3),'unfilled',todo.sum())
    # seamless blend of the copied texture into place
    m8=(nd.binary_dilation(mask,iterations=3)).astype(np.uint8)*255
    ys,xs=np.where(m8>0); cx,cy=(xs.min()+xs.max())//2,(ys.min()+ys.max())//2
    blended=cv2.seamlessClone(out, rgb, m8, (int(cx),int(cy)), cv2.NORMAL_CLONE)
    return blended, todo
