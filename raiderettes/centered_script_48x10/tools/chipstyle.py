"""Build a name in CORSO's exact style: Graduate letters laid out edge-to-edge in CORSO's source space
(4096 x 2533, so the final 48x10 stretch gives the same fat letters and the same anisotropic borders),
orange/black/gray bands copied from CORSO's own straight edge (stitch texture, depth-matched),
sequin+chip interior quilted from CORSO's interiors with min-error seams. Then the shared centered script."""
import sys; sys.path[:0]=["/home/claude/ru","/home/claude/ru/tools"]
import numpy as np, cv2, json
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
import importlib.util
spec=importlib.util.spec_from_file_location("b","/home/claude/ru/build.py"); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
FONT="/home/claude/ocbm-repository/raiderettes/shoulder_40x6/fonts/Graduate-Regular.ttf"
FW,FH=14400,3000
SRC=np.array(Image.open("/home/claude/ru/src/CORSO.png")); SA=SRC[...,3]>=128
SH,SW=SA.shape
BOX=(5,4,4093,2530)           # CORSO letters box -> maps to the full 48x10
HOLE=nd.binary_dilation(np.load("/home/claude/ru/work/hole_CORSO.npy"),iterations=24)
SD=nd.distance_transform_edt(SA)
D0=88                          # orange + black + gray bands end here; sequin field beyond
# ---- canonical band strip from the C's straight left edge (rows 649..2218 at x=5)
rows=[y for y in range(649,2218) if not HOLE[y,5:5+D0+2].any()]
# longest contiguous run of clean rows
runs=[];s=rows[0];p=rows[0]
for y in rows[1:]:
    if y!=p+1: runs.append((s,p)); s=y
    p=y
runs.append((s,p)); s0,s1=max(runs,key=lambda r:r[1]-r[0])
STRIP=SRC[s0:s1+1, 5:5+D0+2, :3].astype(np.float32)      # [s, depth]
SL=STRIP.shape[0]
print("strip rows",s0,s1,"len",SL)
# ---- interior pool for quilting
_rgb=SRC[...,:3].astype(int); WARM=(_rgb[...,0]-_rgb[...,2]>50)&(_rgb[...,0]>100)
POOL=SA&~HOLE&(SD>D0+6)&~nd.binary_dilation(WARM&(SD>D0),iterations=12)
def glyph(ch,h):
    f=ImageFont.truetype(FONT,800); im=Image.new("L",(1400,1200),0); ImageDraw.Draw(im).text((200,150),ch,font=f,fill=255)
    m=np.array(im)>127; ys,xs=np.where(m); g=m[ys.min():ys.max()+1,xs.min():xs.max()+1]
    gh,gw=g.shape; w=int(round(gw*h/gh))
    return cv2.resize(g.astype(np.uint8)*255,(w,h),interpolation=cv2.INTER_AREA)>127
def layout(name,gap=0,bx=260,by=120):
    """Glyphs in FINAL space, bolded (heavy collegiate like CORSO), kerned so every adjacent pair touches
    (V-A, P-A tuck together), scaled to fill the full 48 in, then squeezed into CORSO's source space."""
    x0,y0,x1,y1=BOX
    Hg=FH-2*by
    gs=[glyph(c,Hg) for c in name]
    nat=sum(g.shape[1] for g in gs)
    ker=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*bx+1,2*by+1))
    hk=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*70+1,2*35+1))
    def render(k):
        outs=[]
        for g in gs:
            w=max(8,int(round(g.shape[1]*k)))
            gg=cv2.resize(g.astype(np.uint8)*255,(w,Hg),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_LINEAR)>127
            L=np.zeros((FH,w+2*bx),np.uint8); L[by:by+Hg,bx:bx+w]=gg
            holes=nd.binary_fill_holes(L>0)&~(L>0)
            lab,kk=nd.label(holes); sz=np.bincount(lab.ravel()); sz[0]=0; holes=(sz>20000)[lab]
            L=cv2.dilate(L,ker)
            L[cv2.erode(holes.astype(np.uint8),hk)>0]=0      # keep counters (P bowl, A) open, like CORSO's O
            outs.append(L>0)
        return outs
    def kern(Ls):
        pos=[0]; prevR=None
        for i,L in enumerate(Ls):
            rows=L.any(1)
            left=np.where(rows, np.argmax(L,axis=1), 10**9)
            right=np.where(rows, L.shape[1]-1-np.argmax(L[:,::-1],axis=1), -10**9)
            if prevR is not None:
                X=int(np.ceil(np.max(prevR-left)))+1+gap     # smallest absolute X so the pair just touches
                pos.append(X); prevR=np.maximum(prevR, X+right)
            else:
                prevR=right
        return pos
    k=(FW-2*bx*len(gs))/nat
    for it in range(6):
        Ls=render(k); pos=kern(Ls); T=pos[-1]+Ls[-1].shape[1]
        if abs(T-FW)<=2: break
        k*= (FW-2*bx*len(gs))/max(T-2*bx*len(gs),1)
    Ls=render(k); pos=kern(Ls); T=pos[-1]+Ls[-1].shape[1]
    shift=(FW-T)//2
    print("layout k %.3f total %d shift %d"%(k,T,shift))
    sils=[]
    for L,X in zip(Ls,pos):
        Lf=np.zeros((FH,FW),np.uint8); X=max(0,X+shift)
        w=min(L.shape[1],FW-X); Lf[:,X:X+w]=L[:,:w]
        Ls_=cv2.resize(Lf.astype(np.float32),(x1-x0,y1-y0),interpolation=cv2.INTER_AREA)>=0.5
        S=np.zeros((SH,SW),bool); S[y0:y1,x0:x1]=Ls_
        sils.append(S)
    return sils
def bands(S):
    d=nd.distance_transform_edt(S)
    ds=cv2.GaussianBlur(d.astype(np.float32),(0,0),2.0)
    gy,gx=np.gradient(ds)
    nrm=np.hypot(gx,gy)+1e-6; nx,ny=gx/nrm,gy/nrm
    tx,ty=-ny,nx
    yy,xx=np.mgrid[0:SH,0:SW]
    s=(xx*tx+yy*ty)
    band=S&(d<=D0)
    si=np.mod(np.round(s[band]).astype(np.int64),SL)
    di=np.clip(np.round(d[band]-0.5).astype(int),0,D0+1)
    out=np.zeros((SH,SW,3),np.float32)
    out[band]=STRIP[si,di]
    return out,band,d
def quilt(target_mask,B=192,O=40,K=24,seed=0):
    rng=np.random.default_rng(seed)
    valid=nd.minimum_filter(POOL.astype(np.uint8),size=B,mode="constant",cval=0).astype(bool)
    # valid[y,x] true if the BxB block centred... minimum_filter is centred: block at (y-B//2 .. y+B//2)
    vy,vx=np.where(valid); vy=vy-B//2; vx=vx-B//2
    keep=(vy>=0)&(vx>=0)&(vy+B<=SH)&(vx+B<=SW); vy,vx=vy[keep],vx[keep]
    print("source blocks",len(vy))
    src=SRC[...,:3].astype(np.float32)
    T=np.zeros((SH,SW,3),np.float32); F=np.zeros((SH,SW),bool)
    ys_,xs_=np.where(target_mask); ty0,ty1,tx0,tx1=ys_.min(),ys_.max()+1,xs_.min(),xs_.max()+1
    step=B-O
    for y in range(ty0,ty1,step):
        for x in range(tx0,tx1,step):
            yb,xb=min(y,SH-B),min(x,SW-B)
            if not target_mask[yb:yb+B,xb:xb+B].any(): continue
            cur=T[yb:yb+B,xb:xb+B]; filled=F[yb:yb+B,xb:xb+B]
            idx=rng.choice(len(vy),min(K,len(vy)),replace=False)
            best=None
            for i in idx:
                cand=src[vy[i]:vy[i]+B, vx[i]:vx[i]+B]
                e=((cand-cur)**2).sum(-1)*filled
                err=e.sum()
                if best is None or err<best[0]: best=(err,cand,e)
            _,cand,e=best
            # min-error boundary cut on overlaps
            m=np.ones((B,B),bool)        # True = take new
            if filled[:, :O].any():       # left overlap -> vertical cut
                E=e[:, :O]; cost=E.copy()
                for r in range(1,B):
                    prev=cost[r-1]; best_prev=np.minimum(np.minimum(np.roll(prev,1),prev),np.roll(prev,-1))
                    best_prev[0]=min(prev[0],prev[1]); best_prev[-1]=min(prev[-1],prev[-2])
                    cost[r]=E[r]+best_prev
                c=int(np.argmin(cost[-1])); path=np.zeros(B,int); path[-1]=c
                for r in range(B-2,-1,-1):
                    lo,hi=max(c-1,0),min(c+1,O-1); c=lo+int(np.argmin(cost[r,lo:hi+1])); path[r]=c
                for r in range(B): m[r,:path[r]]=False
            if filled[:O,:].any():        # top overlap -> horizontal cut
                E=e[:O,:]; cost=E.copy()
                for cc in range(1,B):
                    prev=cost[:,cc-1]; best_prev=np.minimum(np.minimum(np.roll(prev,1),prev),np.roll(prev,-1))
                    best_prev[0]=min(prev[0],prev[1]); best_prev[-1]=min(prev[-1],prev[-2])
                    cost[:,cc]=E[:,cc]+best_prev
                r=int(np.argmin(cost[:,-1])); path=np.zeros(B,int); path[-1]=r
                for cc in range(B-2,-1,-1):
                    lo,hi=max(r-1,0),min(r+1,O-1); r=lo+int(np.argmin(cost[lo:hi+1,cc])); path[cc]=r
                for cc in range(B): m[:path[cc],cc]=False
            m&=~(filled&~m)  # keep old where cut says old
            take=m|~filled
            cur[take]=cand[take]; F[yb:yb+B,xb:xb+B]|=True
    return T
def build(name):
    sils=layout(name)
    canvas=np.zeros((SH,SW,4),np.uint8)
    S_all=np.zeros((SH,SW),bool)
    for S in sils: S_all|=S
    interior=np.zeros((SH,SW),bool)
    bandimg=np.zeros((SH,SW,3),np.float32)
    for S in sils:
        bi,band,d=bands(S)
        bandimg[band]=bi[band]
        interior|=S&(d>D0)
    T=quilt(interior)
    rgb=np.zeros((SH,SW,3),np.float32)
    rgb[interior]=T[interior]
    rgb[S_all&~interior]=bandimg[S_all&~interior]
    canvas[...,:3]=np.clip(rgb,0,255).astype(np.uint8); canvas[...,3]=np.where(S_all,255,0)
    return canvas
def finish(name,canvas):
    x0,y0,x1,y1=BOX
    c=canvas[y0:y1,x0:x1]; m=c[...,3]>=128; rgb=c[...,:3].copy()
    _,(iy,ix)=nd.distance_transform_edt(~m,return_indices=True); rgb=rgb[iy,ix]
    r=np.array(Image.fromarray(rgb).resize((FW,FH),Image.LANCZOS))
    A=np.array(Image.fromarray((m*255).astype(np.uint8)).resize((FW,FH),Image.LANCZOS))>=128
    out=np.zeros((FH,FW,4),np.uint8)
    srgb,sal,shf=b.script_final()
    rf=r.astype(np.float32)*(1-shf)[...,None]
    out[...,:3]=np.clip(rf,0,255).astype(np.uint8); out[...,3]=np.where(A,255,0)
    out[sal,:3]=srgb[sal]; out[sal,3]=255; out[out[...,3]==0,:3]=0
    Image.fromarray(out,"RGBA").save(f"/home/claude/ru/final/{name}_Raiderettes_CENTERED_48x10in_300dpi.png",dpi=(300,300))
    b.previews(name,out); print(name,"done")
if __name__=="__main__":
    for n in sys.argv[1:]:
        c=build(n); Image.fromarray(c).save(f"/home/claude/ru/work/chip_{n}.png"); finish(n,c)
