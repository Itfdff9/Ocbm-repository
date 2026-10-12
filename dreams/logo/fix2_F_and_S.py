"""Round 2: open the FASTPITCH S (its two channels become real openings that reach the outside, like a true S)
and clean the F's upper gap (remove the grey blob in the inside corner). Works on the permanent file."""
from PIL import Image
import numpy as np, cv2
a=np.asarray(Image.open('DREAMS_Fastpitch_star_FIXED_300dpi.png')).copy()
L=a[...,:3].astype(np.float32).mean(2); A=a[...,3]>0
hole=np.zeros(L.shape,np.uint8)
def first_rise(seq, start, thr=50):
    for i in range(start,len(seq)):
        if seq[i]>=thr: return i
    return None
def channel(x0,x1,lip_win,peak_win,side,side_win):
    m=np.zeros(L.shape,np.uint8)
    for x in range(x0,x1):
        c=L[:,x]; ys=np.arange(*lip_win); ys=ys[A[ys,x]]
        if len(ys)==0: continue
        ymin=ys[np.argmin(c[ys])]; top=first_rise(c,ymin)
        if top is None: continue
        pk=[y for y in range(top+3,peak_win[1]) if A[y,x] and c[y]>200]
        if not pk: continue
        bot=pk[0]-8
        if bot-top>=3: m[top:bot,x]=1
    # side wall (left wall for the upper channel, right wall for the lower one), same "first rise after the dark lip" rule
    ys=np.where(m.any(1))[0]
    for y in ys:
        r=L[y]; xs=np.arange(*side_win); xs=xs[A[y,xs]]
        if len(xs)==0: continue
        xm=xs[np.argmin(r[xs])]
        if side=="left":
            e=first_rise(r,xm); m[y,:e]=0 if e else m[y,:xm]
        else:
            e=None
            for xx in range(xm,0,-1):
                if r[xx]>=50: e=xx; break
            if e: m[y,e+1:]=0
    return m
up=channel(1460,1632,(2035,2072),(0,2100),"left",(1455,1500))
lo=channel(1405,1600,(2098,2136),(0,2168),"right",(1552,1600))
S=up|lo
S=cv2.morphologyEx(S,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
S=(cv2.GaussianBlur(S.astype(np.float32),(0,0),1.6)>0.5).astype(np.uint8)   # smooth, no jaggies
hole|=S
# F: upper gap = same height as the open part to its right (y 2055-2075), starting just past the stem's dark lip (x 871),
# inside corners rounded like the F's lower inside corner
F=np.zeros(L.shape,np.uint8); cv2.rectangle(F,(871,2055),(1010,2075),1,-1)
r=6
for cx,cy in ((871+r,2055+r),(871+r,2075-r)):
    sq=np.zeros_like(F); sq[cy-r:cy+r+1,871:cx+1]=1
    cir=np.zeros_like(F); cv2.circle(cir,(cx,cy),r,1,-1)
    F[(sq>0)&(cir==0)&(((cy<2065)&(np.arange(F.shape[0])[:,None]<cy))|((cy>2065)&(np.arange(F.shape[0])[:,None]>cy)))]=0
hole|=F
a[hole>0,3]=0; a[a[...,3]==0,:3]=0
a[...,3]=np.where(a[...,3]>=128,255,0).astype(np.uint8)
Image.fromarray(a,"RGBA").save('DREAMS_Fastpitch_PERMANENT_300dpi.png',dpi=(300,300))
print("S cut px",int(S.sum()),"F cut px",int((F&A).sum()))

# ---- polish: remove the thin spur under the S's top-right terminal, and round the small step on the S's lower-left curve
a=np.asarray(Image.open('DREAMS_Fastpitch_PERMANENT_300dpi.png')).copy()
def polish(a, box, open_k=0, sigma=2.5):
    x0,y0,x1,y1=box; al=(a[y0:y1,x0:x1,3]>0).astype(np.uint8)
    if open_k: al=cv2.morphologyEx(al,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(open_k,open_k)))
    sm=(cv2.GaussianBlur(al.astype(np.float32),(0,0),sigma)>0.5).astype(np.uint8)
    # colour for newly-filled pixels: nearest opaque neighbour
    rgb=a[y0:y1,x0:x1,:3].copy(); src=(a[y0:y1,x0:x1,3]>0)
    if (sm&~src).any():
        _,idx=cv2.distanceTransformWithLabels((~src).astype(np.uint8),cv2.DIST_L2,5,labelType=cv2.DIST_LABEL_PIXEL)
        ys,xs=np.where(src); lut=np.zeros((idx.max()+1,3),np.uint8); lut[idx[ys,xs]]=rgb[ys,xs]
        fill=lut[idx]; rgb[(sm>0)&~src]=fill[(sm>0)&~src]
    a[y0:y1,x0:x1,:3]=rgb; a[y0:y1,x0:x1,3]=sm*255; a[y0:y1,x0:x1][sm==0,:3]=0
    return a
a=polish(a,(1592,2045,1630,2090),open_k=7,sigma=1.5)
a=polish(a,(1406,2082,1452,2132),sigma=3.0)
Image.fromarray(a,"RGBA").save('DREAMS_Fastpitch_PERMANENT_300dpi.png',dpi=(300,300)); print("polished")
