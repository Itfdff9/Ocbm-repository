import numpy as np, cv2
from scipy import ndimage as nd
rng=np.random.default_rng(7)
def text_mask(rgb, box, min_h=60, inner_pad=60, white_min=150, angle=0):
    """white letters inside box (x0,y0,x1,y1); keeps tall components, plus small ones well inside the box"""
    x0,y0,x1,y1=box; sub=rgb[y0:y1,x0:x1].astype(int)
    mn=sub.min(2); mx=sub.max(2)
    w=(mn>white_min)&(mx-mn<70)
    lab,n=nd.label(w); m=np.zeros_like(w)
    for i,sl in enumerate(nd.find_objects(lab)):
        if sl is None: continue
        h=sl[0].stop-sl[0].start; wd=sl[1].stop-sl[1].start
        cy=(sl[0].start+sl[0].stop)/2; cx=(sl[1].start+sl[1].stop)/2
        tall=max(h,wd)>=min_h
        inner=(inner_pad<cx<(x1-x0)-inner_pad) and (inner_pad*0.5<cy<(y1-y0)-inner_pad*0.5) and h*wd>40
        if tall or inner: m|=(lab==i+1)
    full=np.zeros(rgb.shape[:2],bool); full[y0:y1,x0:x1]=m
    return full
def clean_glitter(rgb, region, avoid, erode=22):
    x0,y0,x1,y1=region; sub=rgb[y0:y1,x0:x1].astype(int)
    blue=(sub[...,2]>sub[...,0]+60)
    panel=nd.binary_fill_holes(nd.binary_closing(blue,iterations=6))   # glitter incl. its white sparkles
    ok=nd.binary_erosion(panel,iterations=erode)&~avoid[y0:y1,x0:x1]
    full=np.zeros(rgb.shape[:2],bool); full[y0:y1,x0:x1]=ok
    return full
def fill(rgb, mask, clean, P=48, seed=1):
    """Telea low-freq + quilted high-freq glitter from clean patches"""
    rgb=rgb.copy(); m8=mask.astype(np.uint8)*255
    base=cv2.inpaint(rgb, m8, 9, cv2.INPAINT_TELEA).astype(np.float32)
    # candidate patch centres (patch fully clean)
    cl=nd.binary_erosion(clean,iterations=P//2)
    cys,cxs=np.where(cl)
    if len(cys)==0: raise SystemExit('no clean patches')
    src=rgb.astype(np.float32); blur=cv2.GaussianBlur(src,(0,0),6)
    hf=src-blur
    ys,xs=np.where(mask); Y0,Y1,X0,X1=ys.min()-P,ys.max()+P,xs.min()-P,xs.max()+P
    acc=np.zeros((Y1-Y0,X1-X0,3),np.float32); wsum=np.zeros((Y1-Y0,X1-X0,1),np.float32)
    win=np.sqrt(np.outer(np.hanning(P),np.hanning(P))).astype(np.float32)[...,None]+1e-3
    step=P//2
    for y in range(0,Y1-Y0-P+1,step):
        for x in range(0,X1-X0-P+1,step):
            k=rng.integers(len(cys)); cy,cx=cys[k],cxs[k]
            p=hf[cy-P//2:cy-P//2+P, cx-P//2:cx-P//2+P]
            if rng.random()<0.5: p=p[:,::-1]
            acc[y:y+P,x:x+P]+=p*win; wsum[y:y+P,x:x+P]+=win
    hfill=acc/np.maximum(wsum,1e-3)
    # low-frequency colour from local clean surroundings (telea is fine) -> blur it so letter ghosts vanish
    cw=clean.astype(np.float32)[...,None]
    num=cv2.GaussianBlur(src*cw,(0,0),22); den=cv2.GaussianBlur(cw,(0,0),22)[...,None]
    lowf=np.where(den>0.02, num/np.maximum(den,1e-3), cv2.GaussianBlur(base,(0,0),6)).astype(np.float32)
    out=src.copy()
    region=(slice(Y0,Y1),slice(X0,X1))
    synth=lowf[region]+hfill
    # match the synthesized fill to the clean glitter's colour distribution (brightness + sparkle density)
    mreg=mask[region]; ref=src[clean]
    for c in range(3):
        v=synth[...,c][mreg]; order=np.argsort(v)
        q=np.sort(ref[:,c]); idx=(np.arange(len(v))*(len(q)-1)/max(len(v)-1,1)).astype(int)
        nv=np.empty_like(v); nv[order]=q[idx]; tmp=synth[...,c]; tmp[mreg]=nv; synth[...,c]=tmp
    feather=cv2.GaussianBlur(m8.astype(np.float32)/255,(0,0),2)[region][...,None]
    feather=np.clip(feather*1.6,0,1)
    out[region]=out[region]*(1-feather)+synth*feather
    return np.clip(out,0,255).astype(np.uint8)
