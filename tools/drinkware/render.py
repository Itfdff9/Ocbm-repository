"""Procedural product renders for UV DTF drinkware.
40oz handle tumbler (Hit 'Intrepid' style) and 21oz bottle with bamboo lid (Hit 'Liberty' style).
Front-on studio view, 2x supersampled. The print wraps around the cylinder (arc-length mapping)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage
FP='/tmp/claude-0/-home-claude/f4680fbe-84fa-5212-8106-a18cccebeb09/scratchpad/fonts/Graduate-Regular.ttf'
OUT=2000; SS=1; N=OUT*SS

def smooth(a,b,t):
    t=np.clip((t-a)/(b-a),0,1); return t*t*(3-2*t)

# ---------- profiles: r(y) in inches, y up from the base ----------
def prof_tumbler(y):
    r=1.50+0.45*smooth(3.2,4.1,y)+0.05*smooth(4.1,8.6,y)
    r=np.where(y<0.25,1.50-0.25+np.sqrt(np.clip(0.25**2-(0.25-y)**2,0,None)),r)
    return r
def prof_bottle(y):
    r=np.full_like(y,1.46)
    r=np.where(y<0.2,1.46-0.2+np.sqrt(np.clip(0.04-(0.2-y)**2,0,None)),r)
    r=r-0.24*smooth(7.35,7.95,y)
    return r

def shade_cyl(nx, ny=0.0):
    nz=np.sqrt(np.clip(1-nx**2,0,1))
    L=np.array([-0.45,0.30,0.84]); L=L/np.linalg.norm(L)
    diff=np.clip(nx*L[0]+ny*L[1]+nz*L[2],0,1)
    # broad soft specular (powder coat) + second rim light from right
    H=L+np.array([0,0,1.0]); H=H/np.linalg.norm(H)
    spec=np.clip(nx*H[0]+nz*H[2],0,1)**18
    rim=np.clip(nx,0,1)**6*0.35
    fres=1-0.45*(1-nz)**2
    return diff,spec,rim,fres

def wrap_print(design, rgba_mask_shape, X, Y, r, yc, h_in, side_ok):
    """design: RGBA float array; X,Y inch grids; arc-length mapping centered at x=0."""
    dh,dw=design.shape[:2]; w_in=h_in*dw/dh
    th=np.arcsin(np.clip(X/np.maximum(r,1e-3),-1,1)); s=r*th
    u=(s/w_in+0.5)*(dw-1); v=((yc+h_in/2-Y)/h_in)*(dh-1)
    inside=(u>=0)&(u<=dw-1)&(v>=0)&(v<=dh-1)&side_ok
    out=np.zeros(X.shape+(4,))
    if inside.any():
        uu,vv=u[inside],v[inside]
        for c in range(4):
            out[...,c][inside]=ndimage.map_coordinates(design[...,c],[vv,uu],order=1)
    return out

def background():
    yy,xx=np.mgrid[0:N,0:N]/N
    g=244-18*yy-6*np.abs(xx-0.5)
    bg=np.stack([g,g-1.5,g-4],-1)
    return bg

def contact_shadow(img, cx, by, rx, strength=0.42):
    yy,xx=np.mgrid[0:N,0:N].astype(float)
    d=((xx-cx)/(rx*1.25))**2+((yy-by)/(rx*0.16))**2
    sh=np.exp(-d*1.6)*strength
    d2=((xx-cx)/(rx*0.95))**2+((yy-by)/(rx*0.06))**2
    sh+=np.exp(-d2*3)*0.35
    return img*(1-np.clip(sh,0,0.75)[...,None])

def composite_body(img, cov, col, design_rgba, gloss_print=True, metal=False, grain=True):
    """cov: coverage 0..1; col: HxWx3 base colour (already incl. print); returns img."""
    a=cov[...,None]
    return img*(1-a)+col*a

def render_tumbler(base_rgb, design=None, design_h=3.4, design_yc=6.35, handle='right'):
    S=N/14.0  # px per inch
    cx=N/2; by=N*0.905
    img=background(); img=contact_shadow(img,cx,by,2.0*S)
    yy,xx=np.mgrid[0:N,0:N].astype(float)
    Y=(by-yy)/S; X=(xx-cx)/S
    base=np.array(base_rgb,float)
    lum=base.mean()/255
    # ----- handle (behind/beside body), drawn first -----
    sgn=1 if handle=='right' else -1
    hx=X*sgn
    def rrect(x0,x1,y0,y1,rad):
        qx=np.maximum(np.maximum(x0+rad-hx,hx-(x1-rad)),0); qy=np.maximum(np.maximum(y0+rad-Y,Y-(y1-rad)),0)
        return np.sqrt(qx**2+qy**2)-rad
    outer=rrect(1.55,3.30,4.55,8.35,0.62); inner=rrect(1.55,2.82,5.05,7.85,0.30)
    hsd=np.maximum(outer,-inner)  # signed distance of handle ring
    hcov=np.clip(0.5-hsd*S,0,1)
    # tube shading across ring thickness
    tube=np.clip(-hsd/0.24,0,1); hn=np.sqrt(tube)
    hdiff=0.35+0.55*hn*(0.6+0.4*np.clip(1-(hx-2.2)/1.2,0,1))
    hcol=base*hdiff[...,None]+255*0.10*hn[...,None]**8
    img=img*(1-hcov[...,None])+hcol*hcov[...,None]
    # ----- body -----
    r=prof_tumbler(np.clip(Y,0,8.62))
    bcov=np.clip(0.5-(np.abs(X)-r)*S,0,1)*np.clip(0.5+Y*S,0,1)*np.clip(0.5+(8.62-Y)*S,0,1)
    nx=np.clip(X/np.maximum(r,1e-3),-0.999,0.999)
    diff,spec,rim,fres=shade_cyl(nx)
    col=np.broadcast_to(base,X.shape+(3,)).copy()
    pmask=np.zeros(X.shape)
    if design is not None:
        side_ok=np.abs(nx)<0.995
        P=wrap_print(design,None,X,Y,r,design_yc,design_h,side_ok)
        pa=P[...,3:4]/255; col=col*(1-pa)+P[...,:3]*pa; pmask=pa[...,0]
    # powder coat grain
    rng=np.random.default_rng(7); grain=ndimage.gaussian_filter(rng.normal(0,1,X.shape),1.2)*4
    amb=0.30+0.22*lum
    shade=(amb+0.78*diff)*fres
    out=col*shade[...,None]+grain[...,None]*(1-pmask[...,None]*0.7)
    out+=255*(0.16+0.10*pmask)[...,None]*spec[...,None]+255*rim[...,None]*0.18
    # waist shadow line where the flare starts
    ws=np.exp(-((Y-3.55)/0.28)**2)*0.10; out*=(1-ws)[...,None]
    img=img*(1-bcov[...,None])+out*bcov[...,None]
    # ----- lid -----
    lr=np.where(Y<9.0,2.06,2.06-0.10*smooth(9.0,9.45,Y))
    lcov=np.clip(0.5-(np.abs(X)-lr)*S,0,1)*np.clip(0.5+(Y-8.60)*S,0,1)*np.clip(0.5+(9.45-Y)*S,0,1)
    lnx=np.clip(X/lr,-0.999,0.999); ld,ls,lrim,lf=shade_cyl(lnx)
    lcol=base*0.92*((amb+0.78*ld)*lf)[...,None]+255*0.18*ls[...,None]
    seam=np.exp(-((Y-8.66)/0.035)**2)*0.45+np.exp(-((Y-9.0)/0.03)**2)*0.25
    lcol*=(1-seam)[...,None]
    img=img*(1-lcov[...,None])+lcol*lcov[...,None]
    # lid top ellipse (slight elevation)
    ty=9.45; e=((X/1.96)**2+((Y-ty)/0.16)**2)
    tcov=np.clip((1-e)*S*0.6,0,1)*(Y>=ty-0.16)
    tcol=base*(0.55+0.35*lum)+255*0.06
    img=img*(1-tcov[...,None])+np.broadcast_to(tcol,X.shape+(3,))*tcov[...,None]
    # ----- straw -----
    sx0=-0.55*sgn
    scov=np.clip(0.5-(np.abs(X-sx0)-0.17)*S,0,1)*np.clip(0.5+(Y-9.42)*S,0,1)*np.clip(0.5+(12.35-Y)*S,0,1)
    snx=np.clip((X-sx0)/0.17,-0.99,0.99); sd,ss_,_,_=shade_cyl(snx)
    scol=base*(0.45+0.5*sd)[...,None]+255*0.35*ss_[...,None]
    img=img*(1-scov[...,None])+scol*scov[...,None]
    # straw grommet
    gcov=np.clip((1-(((X-sx0)/0.32)**2+((Y-9.47)/0.09)**2))*S*0.5,0,1)
    img=img*(1-gcov[...,None])+np.broadcast_to(base*0.45,X.shape+(3,))*gcov[...,None]
    return finish(img)

def render_bottle(base_rgb, design=None, design_h=2.7, design_yc=4.25):
    S=N/12.0; cx=N/2; by=N*0.90
    img=background(); img=contact_shadow(img,cx,by,1.46*S)
    yy,xx=np.mgrid[0:N,0:N].astype(float)
    Y=(by-yy)/S; X=(xx-cx)/S
    base=np.array(base_rgb,float); lum=base.mean()/255
    r=prof_bottle(np.clip(Y,0,7.95))
    bcov=np.clip(0.5-(np.abs(X)-r)*S,0,1)*np.clip(0.5+Y*S,0,1)*np.clip(0.5+(7.95-Y)*S,0,1)
    nx=np.clip(X/np.maximum(r,1e-3),-0.999,0.999)
    # shoulder normal tilts up
    ny=np.where((Y>7.3)&(Y<7.95),0.5*smooth(7.3,7.95,Y),0)
    diff,spec,rim,fres=shade_cyl(nx,ny)
    col=np.broadcast_to(base,X.shape+(3,)).copy(); pmask=np.zeros(X.shape)
    if design is not None:
        P=wrap_print(design,None,X,Y,r,design_yc,design_h,np.abs(nx)<0.995)
        pa=P[...,3:4]/255; col=col*(1-pa)+P[...,:3]*pa; pmask=pa[...,0]
    rng=np.random.default_rng(3); grain=ndimage.gaussian_filter(rng.normal(0,1,X.shape),1.2)*4
    amb=0.30+0.22*lum; shade=(amb+0.78*diff)*fres
    out=col*shade[...,None]+grain[...,None]*(1-pmask[...,None]*0.7)
    out+=255*(0.16+0.10*pmask)[...,None]*spec[...,None]+255*rim[...,None]*0.18
    img=img*(1-bcov[...,None])+out*bcov[...,None]
    # stainless neck ring
    ncov=np.clip(0.5-(np.abs(X)-1.24)*S,0,1)*np.clip(0.5+(Y-7.92)*S,0,1)*np.clip(0.5+(8.08-Y)*S,0,1)
    nnx=np.clip(X/1.24,-.99,.99); nd,ns,_,_=shade_cyl(nnx)
    ncol=np.array([175,178,182.])*(0.45+0.6*nd)[...,None]+255*0.5*ns[...,None]
    img=img*(1-ncov[...,None])+ncol*ncov[...,None]
    # bamboo lid
    lcov=np.clip(0.5-(np.abs(X)-1.28)*S,0,1)*np.clip(0.5+(Y-8.06)*S,0,1)*np.clip(0.5+(8.95-Y)*S,0,1)
    lnx=np.clip(X/1.28,-.99,.99); ld,ls,_,lf=shade_cyl(lnx)
    wood=np.array([196,158,108.])
    rng2=np.random.default_rng(11); g=ndimage.gaussian_filter(rng2.normal(0,1,(N,)),6)
    grainv=np.tile(g[None,:],(N,1))*0  # vertical-ish fine grain
    stripes=ndimage.gaussian_filter(rng2.normal(0,1,(N,N)),[18,1.0])*22
    lcol=(wood+stripes[...,None])*((0.38+0.72*ld)*lf)[...,None]+255*0.08*ls[...,None]
    img=img*(1-lcov[...,None])+lcol*lcov[...,None]
    # carry loop on top (bamboo arch)
    ay=8.95; ad=np.sqrt((X/0.62)**2+((Y-ay)/0.62)**2)
    acov=np.clip((0.5-np.abs(ad-1.0)*0.62*S/2.2+0.5*0)*1,0,1)
    ring=np.abs(np.sqrt(X**2+(Y-ay)**2)-0.55)
    acov=np.clip(0.5-(ring-0.13)*S,0,1)*(Y>ay)
    an=np.clip(1-ring/0.13,0,1)
    acol=(wood*0.95+stripes[...,None]*0.6)*(0.45+0.6*np.sqrt(an))[...,None]
    img=img*(1-acov[...,None])+acol*acov[...,None]
    return finish(img)

def finish(img):
    im=Image.fromarray(np.clip(img,0,255).astype(np.uint8))
    return im.resize((OUT,OUT),Image.LANCZOS)

# ---------- name art ----------
def word_mask(word,cap_px,tracking):
    probe=ImageFont.truetype(FP,400); hb=probe.getbbox('H'); fs=int(400*cap_px/(hb[3]-hb[1]))
    f=ImageFont.truetype(FP,fs); capoff=f.getbbox('H')[1]; capb=f.getbbox('H')[3]
    tr=tracking*cap_px; ws=[f.getbbox(c)[2]-f.getbbox(c)[0] for c in word]
    tot=int(sum(ws)+tr*(len(word)-1))+4; h=capb-capoff+4
    im=Image.new('L',(tot,h),0); d=ImageDraw.Draw(im); x=2
    for c,w in zip(word,ws):
        d.text((x-f.getbbox(c)[0],2-capoff),c,font=f,fill=255); x+=w+tr
    return np.asarray(im)>127
def name_art(name, num, colors, cap=120):
    """colors=(outer,key,fill). Returns RGBA float array. num=None for name only."""
    outer,key,fill=colors
    nm=word_mask(name,cap,0.16); ncap=cap
    if num:
        mm=word_mask(num,int(cap*2.4),0.08)
        lim=2.6*mm.shape[0]
        if nm.shape[1]>lim:  # long name: condense to 80% width first, then shrink letter height
            sx=max(0.8,lim/nm.shape[1]); w=nm.shape[1]*sx
            if w>lim: ncap=int(cap*lim/w)
            nm=word_mask(name,ncap,0.12)
            nm=np.asarray(Image.fromarray(nm.astype(np.uint8)*255).resize((max(1,round(nm.shape[1]*sx)),nm.shape[0]),Image.LANCZOS))>127
    parts=[(nm,(cap*0.105,cap*0.065,cap*0.01))]
    if num: parts.append((mm,(cap*0.2,cap*0.13,cap*0.02)))
    gap=int(cap*0.45); pad=int(cap*0.4)
    W=max(p[0].shape[1] for p in parts)+2*pad; H=sum(p[0].shape[0] for p in parts)+gap*(len(parts)-1)+2*pad
    canvas=np.zeros((H,W),np.uint8); y=pad
    for m,radii in parts:
        mh,mw=m.shape; x0=W//2-mw//2
        full=np.zeros((H,W),bool); full[y:y+mh,x0:x0+mw]=m
        dist=ndimage.distance_transform_edt(~full)
        for lvl,rr in zip((1,2,3),radii): canvas[dist<=rr]=lvl
        y+=mh+gap
    rgba=np.zeros((H,W,4))
    for lvl,col in zip((1,2,3),(outer,key,fill)): rgba[canvas==lvl]=list(col)+[255]
    ys,xs=np.where(canvas>0); return rgba[ys.min():ys.max()+1,xs.min():xs.max()+1]

def load_logo(path, maxpx=1400):
    im=Image.open(path).convert('RGBA'); bb=im.getbbox(); im=im.crop(bb)
    im.thumbnail((maxpx,maxpx),Image.LANCZOS)
    a=np.asarray(im).astype(float); return a

def fit_h(design, w_max, h_max):
    dh,dw=design.shape[:2]; return min(h_max, w_max*dh/dw)
