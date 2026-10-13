import cv2, numpy as np, sys
from scipy import ndimage
from remove import oldmask
K=lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def build_mask(c,img):
    a=oldmask(c,img.shape)
    m=a>10
    m=ndimage.binary_fill_holes(cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_CLOSE,K(4))>0)
    return cv2.dilate(m.astype(np.uint8)*255,K(5))
def texture_fill(img,mask,bbox,P=24,seed=0):
    rng=np.random.default_rng(seed)
    g=img.astype(np.float32); hp=g-cv2.GaussianBlur(g,(0,0),2.0)
    lum=cv2.cvtColor(cv2.GaussianBlur(img,(0,0),2.0),cv2.COLOR_BGR2GRAY).astype(np.float32)
    gm=np.hypot(cv2.Sobel(lum,cv2.CV_32F,1,0),cv2.Sobel(lum,cv2.CV_32F,0,1))
    y0,y1,x0,x1=bbox; H,W=mask.shape
    bad=cv2.dilate(mask,K(4))>0
    fab=np.median(img[mask>0].reshape(-1,3),0) if False else None
    ring=(cv2.dilate(mask,K(40))>0)&~bad
    ref=np.median(img[ring].reshape(-1,3),0).astype(np.float32)
    cands=[]
    for yy in range(max(0,y0-60),min(H-P,y1+60),P//2):
        for xx in range(max(0,x0-60),min(W-P,x1+60),P//2):
            if bad[yy:yy+P,xx:xx+P].any(): continue
            pc=img[yy:yy+P,xx:xx+P].reshape(-1,3).astype(np.float32).mean(0)
            if np.abs(pc-ref).max()>40: continue
            cands.append((gm[yy:yy+P,xx:xx+P].mean(),yy,xx))
    cands.sort(); cands=cands[:max(6,len(cands)//4)]
    out=np.zeros((y1-y0,x1-x0,3),np.float32)
    for ty in range(0,y1-y0,P):
        for tx in range(0,x1-x0,P):
            _,yy,xx=cands[rng.integers(len(cands))]; p=hp[yy:yy+P,xx:xx+P]
            if rng.random()<.5: p=p[:,::-1]
            if rng.random()<.5: p=p[::-1]
            h=min(P,y1-y0-ty); w=min(P,x1-x0-tx); out[ty:ty+h,tx:tx+w]=p[:h,:w]
    return out,len(cands)
def texture_patch(img,mask,bbox):
    # high-pass grain from a clean torso area next to the logo (same rows, left/right not used: arms) -> rows just below
    y0,y1,x0,x1=bbox; h=y1-y0
    g=img.astype(np.float32); hp=g-cv2.GaussianBlur(g,(0,0),2.0)
    cands=[(y1+8,x0),(y0-h-8,x0)]
    for yy,xx in cands:
        if yy<0 or yy+h>img.shape[0]: continue
        if mask[yy:yy+h,x0:x1].any(): continue
        return hp[yy:yy+h,x0:x1]
    return None
def remove(c,img=None):
    img=cv2.imread(f'old/{c}.jpg') if img is None else img
    m=build_mask(c,img)
    for it in range(2):
        base=cv2.inpaint(img,m,7,cv2.INPAINT_TELEA)
        # grow mask where nearby pixels still look like logo (glow, halo)
        ring=cv2.dilate(m,K(10))>0
        sm=cv2.GaussianBlur(base,(0,0),3).astype(np.float32)
        dev=np.abs(img.astype(np.float32)-sm).max(2)
        add=(ring&(dev>28)&(m==0))
        if add.sum()<5: break
        m=np.maximum(m,cv2.dilate(add.astype(np.uint8)*255,K(2)))
    # smooth base: re-inpaint from a lightly blurred copy for even shading, then add grain
    blur=cv2.GaussianBlur(img,(0,0),1.5)
    base=cv2.inpaint(blur,m,9,cv2.INPAINT_TELEA).astype(np.float32)
    ys,xs=np.where(m>0); bbox=(ys.min(),ys.max()+1,xs.min(),xs.max()+1)
    tp,nc=texture_fill(img,m,bbox)
    out=img.astype(np.float32).copy()
    sel=m>0
    fill=base.copy()
    if tp is not None:
        y0,y1,x0,x1=bbox; fill[y0:y1,x0:x1]+=tp
    feather=cv2.GaussianBlur((m>0).astype(np.float32),(0,0),1.2)[...,None]
    out=out*(1-feather)+fill*feather
    return np.clip(out,0,255).astype(np.uint8),m
if __name__=='__main__':
    for c in sys.argv[1].split(','):
        out,m=remove(c); cv2.imwrite(f'rm_{c}.jpg',out,[cv2.IMWRITE_JPEG_QUALITY,95])
