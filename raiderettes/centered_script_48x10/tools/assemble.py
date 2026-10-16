"""Assemble a name from donor letters cut out of the cleaned originals (real AI sequin texture), then the centered script."""
import sys; sys.path[:0]=["/home/claude/ru","/home/claude/ru/tools"]
import numpy as np, cv2, json
from PIL import Image
from scipy import ndimage as nd
import importlib.util
spec=importlib.util.spec_from_file_location("b","/home/claude/ru/build.py"); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
D=json.load(open("/home/claude/ru/work/donor_segs.json"))
FW,FH=14400,3000
DON={"V":("VELEZ",0),"A":("TRAVA",4),"L":("VELEZ",2),"E":("VELEZ",1),"N":("RENDA",2),"T":("MATTEO",3),"I":("DILEO",1),
     "C":("MCHUGH",1),"P":("RENDA",0),"O":("CORSO",1),"R":("CORSO",2),"S":("CORSO",3)}
_cache={}
def load(n):
    if n not in _cache:
        a=np.array(Image.open(f"/home/claude/ru/src/{n}.png"))
        hole=np.load(f"/home/claude/ru/work/hole_{n}.npy")
        hole=nd.binary_dilation(b.bars_removed(n,hole),iterations=(40 if n=='TRAVA' else 14))
        c=np.array(Image.open(D[n]["src"]))          # cleaned version gives the letter silhouette
        A=c[...,3]>=128
        # fill the hole band of each column by period-snapped copy from the same column outside the hole
        g=cv2.cvtColor(a[...,:3],cv2.COLOR_RGB2GRAY).astype(np.float32)
        known=A&~hole
        out=a.copy(); out[~A]=0
        # vertical period
        ys,xs=np.where(known); cx=int(np.median(xs)); col=g[:,cx]; kn=known[:,cx]
        seg=col[kn]-col[kn].mean(); ac=np.correlate(seg,seg,mode="full")[len(seg)-1:]; per=int(np.argmax(ac[20:120])+20)
        fy,fx=np.where(A&hole)
        sy=fy.copy(); ok=np.zeros(len(fy),bool)
        for k in list(range(1,60)):
            for sgn in (-1,1):
                cand=fy+sgn*k*per; inb=(cand>=0)&(cand<a.shape[0])
                hit=inb.copy(); hit[inb]=known[cand[inb],fx[inb]]
                take=hit&~ok; sy[take]=cand[take]; ok|=take
            if ok.all(): break
        if (~ok).any():
            _,(iy,ix)=nd.distance_transform_edt(~known,return_indices=True)
            out[fy[~ok],fx[~ok]]=a[iy[fy[~ok],fx[~ok]],ix[fy[~ok],fx[~ok]]]
        out[fy[ok],fx[ok]]=a[sy[ok],fx[ok]]
        out[A,3]=255
        _cache[n]=out
    return _cache[n]
def cut(ch):
    n,i=DON[ch]; x0,x1=D[n]["segs"][i]; a=load(n)
    sub=a[:,x0:x1].copy(); A=sub[...,3]>=128
    # keep only the largest connected opaque component touching that column band (drop neighbour slivers)
    lab,k=nd.label(A); sz=np.bincount(lab.ravel()); sz[0]=0
    big=sz.argmax(); A=lab==big
    # remove script leftovers hanging off the letter: anything outside a 30px opening of the silhouette
    k30=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(91,91))
    core=cv2.morphologyEx(A.astype(np.uint8),cv2.MORPH_OPEN,k30).astype(bool)
    A=A&nd.binary_dilation(core,iterations=6)
    # trim to the component's bbox
    ys,xs=np.where(A); sub=sub[ys.min():ys.max()+1, xs.min():xs.max()+1]; A=A[ys.min():ys.max()+1, xs.min():xs.max()+1]
    sub[~A]=0; sub[A,3]=255
    if ch=="P":
        # Build P from the R donor: keep R's stem + bowl exactly, cut the leg, and close the bowl bottom
        # with the bowl-top band flipped (border/inline/black), aligned to the stem.
        H,W=A.shape
        d=nd.distance_transform_edt(A)
        # bowl bottom row: lowest row where the right half is still fully opaque across the bowl (before the leg's inner notch)
        right=A[:, int(0.45*W):]
        occ=right.mean(1)
        yb=int(np.argmax((occ<0.55)&(np.arange(H)>H*0.4)))   # first row below 40% where coverage drops
        if yb<=0: yb=int(0.58*H)
        bh=int(0.10*H)
        band=sub[:bh, :].copy()                   # top band of the whole letter (stem+bowl top)
        flip=band[::-1]
        stem_w=int(np.argmax(~A[int(0.8*H), :]))  # first transparent column at 80% height = stem right edge
        stem_w=stem_w if stem_w>0 else int(0.42*W)
        sub[yb:, stem_w:]=0
        region=sub[yb-bh:yb, stem_w:]; fl=flip[:, stem_w:]
        sub[yb-bh:yb, stem_w:]=np.where(fl[...,3:4]>0, fl, region)
        # stem's right border below the bowl: mirror the stem's left border strip
        lb=int(0.11*W); strip=sub[yb:, :lb][:, ::-1]
        sub[yb:, stem_w-lb:stem_w]=np.where(strip[...,3:4]>0, strip, sub[yb:, stem_w-lb:stem_w])
    if ch=="P":
        Ai=sub[...,3]>=128
        filled=nd.binary_fill_holes(Ai)
        holes=filled&~Ai
        lab,k=nd.label(holes); sz=np.bincount(lab.ravel()); sz[0]=0
        small=(sz<4000)[lab]&(lab>0)
        _,(iy,ix)=nd.distance_transform_edt(~Ai,return_indices=True)
        sub[small]=sub[iy[small],ix[small]]
    return sub
def main(name):
    parts=[cut(c) for c in name]
    capH=int(FH*0.94)
    scaled=[]
    for p in parts:
        h,w=p.shape[:2]; s=capH/h
        rgb=p[...,:3]; A=p[...,3]>=128
        _,(iy,ix)=nd.distance_transform_edt(~A,return_indices=True); rgb=rgb[iy,ix]
        r=cv2.resize(rgb,(int(w*s),capH),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC)
        al=cv2.resize(A.astype(np.float32),(int(w*s),capH),interpolation=cv2.INTER_LINEAR)>=0.5
        q=np.dstack([r,(al*255).astype(np.uint8)]); q[~al]=0; scaled.append(q)
    total=sum(q.shape[1] for q in scaled)
    gap=max(0,(FW-total)/(len(scaled)-1)) if len(scaled)>1 else 0
    if total>FW:   # shrink all
        s=FW/total*0.995
        scaled=[cv2.resize(q,(int(q.shape[1]*s),int(q.shape[0]*s)),interpolation=cv2.INTER_AREA) for q in scaled]
        for q in scaled: q[q[...,3]<128]=0; q[q[...,3]>=128,3]=255
        total=sum(q.shape[1] for q in scaled); gap=(FW-total)/(len(scaled)-1)
    gap=min(gap,0.0)+max(0,(FW-total)/(len(scaled)-1))
    out=np.zeros((FH,FW,4),np.uint8)
    x=0.0
    for q in scaled:
        h,w=q.shape[:2]; X=int(round(x)); Y=(FH-h)//2
        m=q[...,3]>0
        out[Y:Y+h,X:X+w][m]=q[m]
        x+=w+gap
    srgb,sal,shf=b.script_final()
    rf=out[...,:3].astype(np.float32)*(1-shf)[...,None]
    out[...,:3]=np.clip(rf,0,255).astype(np.uint8)
    out[sal,:3]=srgb[sal]; out[sal,3]=255
    out[out[...,3]==0,:3]=0
    Image.fromarray(out,"RGBA").save(f"/home/claude/ru/final/{name}_Raiderettes_CENTERED_48x10in_300dpi.png",dpi=(300,300))
    b.previews(name,out); print(name,"done")
if __name__=="__main__":
    for n in sys.argv[1:]: main(n)
