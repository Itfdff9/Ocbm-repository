"""Raiderettes 48x10: remove each name's old script, rebuild uncovered letter parts (shift-map),
then lay DILEO's script on top, same size, dead center, on all 10."""
import sys; sys.path[:0]=["/home/claude/ru","/home/claude/ru/tools"]
import numpy as np, cv2, json, time, os
from PIL import Image
from scipy import ndimage as nd
import shiftmap
R0="/home/claude/ru"
FW,FH=14400,3000
def load(n): return np.array(Image.open(f"{R0}/src/{n}.png").convert("RGBA"))
def trim_box(a):
    ys,xs=np.where(a[...,3]>=128); return int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1
def compose(A,B):
    A3=np.vstack([A,[0,0,1]]); B3=np.vstack([B,[0,0,1]]); return (A3@B3)[:2]
aD=load("DILEO"); cutD=np.load(f"{R0}/work/cutF_12.npy")
bxD=trim_box(aD); sxD=FW/(bxD[2]-bxD[0]); syD=FH/(bxD[3]-bxD[1])
ys,xs=np.where(cutD)
cx=((xs.min()+xs.max()+1)/2-bxD[0])*sxD; cy=((ys.min()+ys.max()+1)/2-bxD[1])*syD
DY=float(os.environ.get("DY","0"))
M=np.array([[sxD,0,-bxD[0]*sxD-cx+FW/2],[0,syD,-bxD[1]*syD-cy+FH/2+DY]],np.float64)
BOXES=json.load(open(f"{R0}/work/boxes.json"))
def letters_box(n): return tuple(BOXES[n]["letters"])
def script_in_src(n,a):
    bx=letters_box(n); sx=FW/(bx[2]-bx[0]); sy=FH/(bx[3]-bx[1])
    F2S=np.array([[1/sx,0,bx[0]],[0,1/sy,bx[1]]])
    T=compose(F2S,M)
    H,W=a.shape[:2]
    return cv2.warpAffine(cutD.astype(np.uint8),T,(W,H),flags=cv2.INTER_NEAREST)>0, bx
def bars_removed(n,h):
    """Drop straight letter-border bars that the old-script mask picked up far from the script."""
    sil=np.load(f"{R0}/work/sil_{n}.npy")
    near=nd.distance_transform_edt(~sil)<=90
    far=h&~near
    if not far.any(): return h
    vo=cv2.morphologyEx(far.astype(np.uint8),cv2.MORPH_OPEN,np.ones((151,1),np.uint8)).astype(bool)
    ho=cv2.morphologyEx(far.astype(np.uint8),cv2.MORPH_OPEN,np.ones((1,151),np.uint8)).astype(bool)
    lab,k=nd.label(far,structure=np.ones((3,3)))
    if k==0: return h
    sz=nd.sum(np.ones_like(lab),lab,np.arange(1,k+1))
    bb=nd.sum(vo|ho,lab,np.arange(1,k+1))
    isbar=np.zeros(k+1,bool); isbar[1:]=(bb/np.maximum(sz,1))>0.4
    drop=isbar[lab]
    print(n,"bar px dropped",int(drop.sum()),"kept far px",int((far&~drop).sum()))
    return h&~drop
def holes(n,a):
    h=np.load(f"{R0}/work/hole_{n}.npy")
    h=bars_removed(n,h)
    # old script's soft shadow below it
    k=np.zeros((41,1),np.uint8); k[20:,0]=1          # dilate downward 20px
    hd=cv2.dilate(h.astype(np.uint8),k,anchor=(0,20))>0
    k2=np.zeros((1,13),np.uint8); k2[0,6:]=1
    hd=cv2.dilate(hd.astype(np.uint8),k2,anchor=(6,0))>0   # and a little to the right
    return (h|hd)&(a[...,3]>0) | h
def lama_guide(a,hole,f=4):
    import lama
    H,W=hole.shape
    KEY=np.array([0,255,0],np.float32)
    al=a[...,3:4].astype(np.float32)/255
    rgb=(a[...,:3]*al+KEY*(1-al)).astype(np.uint8)
    w,h=W//f,H//f
    sm=cv2.resize(rgb,(w,h),interpolation=cv2.INTER_AREA)
    hm=nd.binary_dilation(cv2.resize(hole.astype(np.uint8),(w,h),interpolation=cv2.INTER_AREA)>0,iterations=1)
    ph=max(512,h); pad=np.zeros((ph,w,3),np.uint8); pad[:h]=sm; pad[h:]=KEY.astype(np.uint8)
    pm=np.zeros((ph,w),bool); pm[:h]=hm
    r=lama.inpaint(pad,pm,tile=512,overlap=256)[:h]
    g=cv2.resize(r,(W,H),interpolation=cv2.INTER_CUBIC).astype(np.float32)/255.0
    return g
def clean(n, cycles=2, K=40):
    a=load(n); H,W=a.shape[:2]
    newS,bx=script_in_src(n,a)
    hole=holes(n,a)
    inner=nd.binary_erosion(newS,iterations=10)
    fill=hole&~inner
    known=~hole
    extra=[(d,0) for d in range(-260,261,4) if abs(d)>=4]
    t=time.time()
    if os.environ.get("GUIDE","1")=="struct":
        from labels import labels
        import structfill
        L=labels(a); B=letters_box(n); base=B[3]-1; Hl=B[3]-B[1]
        bw=int(os.environ.get("BW","35")); fh=int(0.21*Hl); fw=int(0.027*Hl)
        Lcf,hole=structfill.complete(L,hole,base,bw,fh,fw)
        fill=hole&~inner; known=~hole
        np.save(f"{R0}/work/labstruct_{n}.npy",Lcf)
        from labels import PAL
        Image.fromarray(PAL[Lcf]).resize((2048,int(2048*H/W)),Image.NEAREST).save(f"/tmp/claude-0/rx/struct_{n}.png")
        # fill = whole hole (letters) except under the new script interior
        out,info=shiftmap.inpaint_labels(a,fill,known,L,Lcf,scale=2,K=30,cycles=cycles,lam=float(os.environ.get("LAM","4")),extra_half=extra)
        out[base+4:]=0
        hole[base+4:]=True
    elif os.environ.get("GUIDE","1")=="label":
        from labels import labels
        import linefill
        L=labels(a)
        f=2; h2,w2=H//f,W//f
        Ls=cv2.resize(L,(w2,h2),interpolation=cv2.INTER_NEAREST)
        hs=nd.binary_dilation(cv2.resize(hole.astype(np.uint8),(w2,h2),interpolation=cv2.INTER_NEAREST)>0,iterations=1)
        fs=nd.binary_dilation(cv2.resize(fill.astype(np.uint8),(w2,h2),interpolation=cv2.INTER_NEAREST)>0,iterations=2)&hs
        Lc=linefill.complete(Ls,fs,~hs,ndir=24,verbose=True)
        for _ in range(2):   # remove thin streaks: mode filter inside the filled area
            oh=np.stack([(Lc==k).astype(np.float32) for k in range(3)],-1)
            md=np.argmax(cv2.blur(oh,(9,9)),-1).astype(np.uint8)
            Lc[fs]=md[fs]
        Lcf=cv2.resize(Lc,(W,H),interpolation=cv2.INTER_NEAREST)
        Lcf[~hole]=L[~hole]
        np.save(f"{R0}/work/labc_{n}.npy",Lc)
        out,info=shiftmap.inpaint_labels(a,fill,known,L,Lcf,scale=2,K=30,cycles=cycles,lam=float(os.environ.get("LAM","3")),extra_half=extra)
    elif os.environ.get("GUIDE","1")=="1":
        guide=lama_guide(a,hole)
        out,info=shiftmap.inpaint2g(a,fill,known,guide,scale=2,K=K,cycles=cycles,lam=float(os.environ.get("LAM","2")),extra_half=extra)
    else:
        out,info=shiftmap.inpaint2(a,fill,known,scale=2,K=K,cycles=cycles,extra_half=extra)
    print(n,"clean %.0fs"%(time.time()-t))
    # remove thin leftovers of the old script's outline/shadow around the rebuilt area
    zone=nd.distance_transform_edt(~hole)<=30
    Aop=cv2.morphologyEx((out[...,3]>=128).astype(np.uint8),cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))).astype(bool)
    rm=zone&(out[...,3]>0)&~Aop&~inner
    out[rm]=0
    print(n,"thin leftovers removed",int(rm.sum()))
    Image.fromarray(out).save(f"{R0}/work/clean_{n}.png")
    np.save(f"{R0}/work/fill_{n}.npy",fill)
    return out
def stretch(a, n):
    bx=letters_box(n); c=a[bx[1]:bx[3],bx[0]:bx[2]]
    m=c[...,3]>=128; rgb=c[...,:3].copy()
    _,(iy,ix)=nd.distance_transform_edt(~m,return_indices=True); rgb=rgb[iy,ix]
    r=np.array(Image.fromarray(rgb).resize((FW,FH),Image.LANCZOS))
    A=np.array(Image.fromarray((m*255).astype(np.uint8)).resize((FW,FH),Image.LANCZOS))>=128
    return r,A
_S=None
def script_final():
    global _S
    if _S is None:
        m=cutD; rgb=aD[...,:3].copy()
        _,(iy,ix)=nd.distance_transform_edt(~m,return_indices=True); rgb=rgb[iy,ix]
        srgb=cv2.warpAffine(rgb,M,(FW,FH),flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        sal=cv2.warpAffine(m.astype(np.float32),M,(FW,FH),flags=cv2.INTER_LINEAR)>=0.5
        # drop shadow (DILEO src px: 5 right, 12 down, soft)
        sh=np.zeros(m.shape,np.float32); sh[12:,5:]=m[:-12,:-5]
        sh=cv2.GaussianBlur(sh,(0,0),8)*0.55
        shf=cv2.warpAffine(sh,M,(FW,FH),flags=cv2.INTER_LINEAR)
        _S=(srgb,sal,shf)
    return _S
def final(n, cleaned):
    r,A=stretch(cleaned, n)
    srgb,sal,shf=script_final()
    out=np.zeros((FH,FW,4),np.uint8)
    rf=r.astype(np.float32)
    rf*= (1-shf)[...,None]
    out[...,:3]=np.clip(rf,0,255).astype(np.uint8)
    out[...,3]=np.where(A,255,0)
    out[sal,:3]=srgb[sal]; out[sal,3]=255
    al=out[...,3]>0
    lab,k=nd.label(al); sz=np.bincount(lab.ravel()); sz[0]=0
    out[(sz<5000)[lab]]=0
    out[~(out[...,3]>0),:3]=0
    return out
def previews(n,out,tag=""):
    im=Image.fromarray(out,"RGBA")
    bg=Image.new("RGBA",im.size,(57,255,20,255)); bg.alpha_composite(im)
    bg.convert("RGB").resize((2400,500),Image.LANCZOS).save(f"/tmp/claude-0/rx/F_{n}{tag}.jpg",quality=88)
    return bg
def fix_small(n, c, amax=8000):
    """Fill small see-through holes enclosed inside the letters (flaws in the uploads)."""
    from smallholes import small_holes
    a=load(n)
    newS,_=script_in_src(n,a)
    m,lst=small_holes(c,amax)
    m&=~nd.binary_erosion(newS,iterations=10)      # covered by the new script anyway
    if not m.any(): return c
    fill=nd.binary_dilation(m,iterations=4)
    known=(c[...,3]>=128)&~fill
    out,_=shiftmap.inpaint2(c,fill,known,scale=2,K=20,cycles=2,verbose=False)
    out[fill,3]=255
    print(n,"small holes filled",len(lst),int(m.sum()))
    return out
if __name__=="__main__":
    for n in sys.argv[1:]:
        if os.environ.get("STAGE")=="final":
            c=np.array(Image.open(f"{R0}/work/clean_{n}.png"))
        else:
            c=clean(n)
        c=fix_small(n,c)
        Image.fromarray(c).save(f"{R0}/work/clean2_{n}.png")
        out=final(n,c)
        Image.fromarray(out,"RGBA").save(f"{R0}/final/{n}_Raiderettes_CENTERED_48x10in_300dpi.png",dpi=(300,300))
        previews(n,out)
        print(n,"done")
