"""Structure completion for letters whose bottoms were covered: continue each stroke along its own
orientation down to the baseline, add the slab foot, re-derive the border; returns a label map."""
import numpy as np, cv2
from scipy import ndimage as nd
def orientation(S, sigma=25):
    g=cv2.GaussianBlur(S.astype(np.float32),(0,0),3)
    gx=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    Jxx=cv2.GaussianBlur(gx*gx,(0,0),sigma); Jyy=cv2.GaussianBlur(gy*gy,(0,0),sigma); Jxy=cv2.GaussianBlur(gx*gy,(0,0),sigma)
    # edge direction (perpendicular to gradient): angle of stroke
    th=0.5*np.arctan2(2*Jxy,Jxx-Jyy)          # gradient orientation
    stroke=th+np.pi/2
    coh=np.sqrt((Jxx-Jyy)**2+4*Jxy**2)/(Jxx+Jyy+1e-6)
    return stroke, coh
def complete(L, hole, base, bw, foot_h, foot_w, max_tan=1.2, verbose=True):
    H,W=L.shape
    S=(L!=0)
    # the letters were only drawn down to the old script: everything from the script's top edge to the
    # baseline that is not a known letter pixel is part of the rebuild region
    hole=nd.binary_dilation(hole,iterations=12)
    first=np.where(hole.any(0), np.argmax(hole,axis=0), H)
    yy=np.arange(H)[:,None]
    below=(yy>=first[None,:])&(yy<=base)
    hole|=below&~S
    known=~hole
    far=nd.distance_transform_edt(known)>45
    stroke,coh=orientation(S&known)
    # orientation near the hole boundary is contaminated by the old script's outline: take it from farther away
    _,(iy,ix)=nd.distance_transform_edt(~far,return_indices=True)
    stroke=stroke[iy,ix]; coh=coh[iy,ix]
    # dx per row step along the stroke: dx/dy = cos/sin of stroke angle (stroke angle measured from x axis)
    dxdy=np.cos(stroke)/np.where(np.abs(np.sin(stroke))<1e-3,1e-3,np.sin(stroke))
    dxdy=np.clip(dxdy,-max_tan,max_tan)
    dxdy[coh<0.25]=0.0
    Lc=L.copy().astype(np.int16); Lc[hole]=-1
    D=dxdy.copy(); D[hole]=np.nan
    rows=np.where(hole.any(1))[0]
    xs=np.arange(W)
    for y in range(rows.min(),min(rows.max(),base)+1):
        m=hole[y]
        if not m.any(): continue
        # source row y-1, shifted by -dx (follow the stroke downward). Use previous row's D; where prev D unknown use 0.
        dprev=D[y-1]; dprev=np.where(np.isnan(dprev),0.0,dprev)
        # for target x, source x_s = x - dprev[x_s]; approximate with dprev at x
        sx=np.clip(np.round(xs-dprev).astype(int),0,W-1)
        src=Lc[y-1,sx]; srcD=D[y-1,sx]
        # if source unknown (-1), fall back to straight above
        bad=src<0
        src=np.where(bad,Lc[y-1],src); srcD=np.where(bad,D[y-1],srcD)
        Lc[y,m]=src[m]; D[y,m]=np.where(np.isnan(srcD[m]),0.0,srcD[m])
    Lc[Lc<0]=0
    Lc[base+1:][hole[base+1:]]=0
    S2=(Lc!=0)
    # slab foot: widen strokes horizontally over the last foot_h rows (only in the hole)
    if foot_h>0 and foot_w>0:
        y0=max(base-foot_h,0)
        band=np.zeros_like(S2); band[y0:base+1]=True
        k=np.ones((1,2*foot_w+1),np.uint8)
        wide=cv2.dilate(S2.astype(np.uint8),k)>0
        S2=np.where(band&hole,wide,S2)
    # smooth the silhouette inside the hole a little, round bottom corners
    S2u=cv2.morphologyEx(S2.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    S2u=cv2.morphologyEx(S2u,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(15,15)))
    S3=np.where(hole,S2u>0,S)
    # inter-letter gaps: the AI drew a black slab under the old script, so straight-sided gaps between letters
    # must be re-cut: take each narrow transparent gap just above the rebuild region and carry it straight down.
    y_ref=int(first[first<H].min())-25
    def runs(row):
        xs=np.where(row)[0]; out=[]
        if len(xs)==0: return out
        s0=xs[0]; p=xs[0]
        for x in xs[1:]:
            if x!=p+1: out.append((s0,p)); s0=x
            p=x
        out.append((s0,p)); return out
    tr=~S
    cand=[r for r in runs(tr[y_ref]) if 8<=r[1]-r[0]<=700 and r[0]>0 and r[1]<W-1]
    gaps=[]
    GW=34
    for (g0,g1) in cand:
        up=[r for r in runs(tr[y_ref-120]) if r[0]<=g1 and r[1]>=g0]
        if not up: continue
        u=up[0]
        if abs(u[1]-g1)>25: continue      # right letter has a diagonal left side (an A): leave to the flow
        if g1-g0<=140 and abs(u[0]-g0)<=25: gaps.append((g0,g1))      # narrow straight gap: keep as is
        else: gaps.append((g1-GW,g1))                                 # wide gap: slit against the right letter
    if verbose: print("   gaps carried down:",gaps)
    for (g0,g1) in gaps:
        S3[y_ref:base+1, g0:g1+1]&=~hole[y_ref:base+1, g0:g1+1]
    # re-derive border inside the hole from the silhouette
    d=nd.distance_transform_edt(S3)
    Lout=L.copy()
    Lout[hole]=np.where(S3[hole],np.where(d[hole]<=bw,1,2),0)
    if verbose: print("   struct: filled",int(hole.sum()),"silhouette px in hole",int(S3[hole].sum()))
    return Lout.astype(np.uint8), hole
