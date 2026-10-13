"""NSE cheer-bow NAME TEMPLATES (editable): put any names on the blank bows.
Usage:
  python3 bow_names.py bow3 "Annabella" "Madeline" "Lilyana"   # 3-name chevron bow (top -> bottom)
  python3 bow_names.py bow2 "Emily" "Nina"                      # 2-name ribbon bow (left, right)
Optional: --out DIR (default ./names)  --sizes 4200,8192
Uses the blank template PNGs next to this script (NSE_Bow3Names_BLANK_* / NSE_Bow2Names_BLANK_*), the Rokkitt font,
white slab letters with a soft navy drop shadow and light bevel, auto-shrunk so long names fit the panel/ribbon.
All positions are fractions of the image, so any template size works."""
import os, sys, argparse, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS=None
HERE=os.path.dirname(os.path.abspath(__file__))
FONT=os.path.join(HERE,'RobotoSlab.ttf')   # Roboto Slab, weight 900, condensed per bow
# slot = centre of the ascender band (cx,cy as fraction of the ORIGINAL canvas 'aspect'; the template is that canvas cropped to 'crop'), asc = ascender height (fraction of H),
# maxw = max text length (fraction of H for rotated, of W for straight), angle = degrees CCW
BOWS={
 'bow3':dict(blank='NSE_Bow3Names_BLANK', aspect=(1890,4095), crop=(30,674,1859,3319), shadow=(0.035,0.05), condense=0.62, bold=0.012, slots=[
    dict(cx=0.5011,cy=(1965+94+15)/4095,asc=0.82*189/4095,maxw=0.315,angle=0),
    dict(cx=0.5005,cy=(2377+104)/4095,asc=0.82*207/4095,maxw=0.34,angle=0),
    dict(cx=0.5040,cy=(2831+97)/4095,asc=0.82*194/4095,maxw=0.305,angle=0)]),
 'bow2':dict(blank='NSE_Bow2Names_BLANK', aspect=(2100,4550), crop=(143,1008,2001,3158), shadow=(0.03,0.045), condense=0.97, bold=0.008, slots=[
    dict(cx=597.6/2100,cy=2552.1/4550,asc=158/4550,maxw=0.125,angle=65.3,up=0.09),
    dict(cx=1488.7/2100,cy=2541.3/4550,asc=158/4550,maxw=0.115,angle=-68.1,up=0.0)]),
}
def text_layer(text, asc_px, max_len_px, shadow, condense=1.0, bold=0.0):
    f=ImageFont.truetype(FONT,100); f.set_variation_by_axes([900])
    a=f.getbbox('Hdlk'); asc100=a[3]-a[1]
    size=int(100*asc_px/asc100); f=ImageFont.truetype(FONT,size); f.set_variation_by_axes([900])
    L=f.getlength(text)*condense
    if L>max_len_px:
        size=int(size*max_len_px/L); f=ImageFont.truetype(FONT,size); f.set_variation_by_axes([900])
    asc=f.getbbox('Hdlk'); top_y=asc[1]; base_y=asc[3]
    l,t,r,b=f.getbbox(text); pad=int(size*0.4)
    W=r-l+2*pad; H=max(b,base_y)-min(t,top_y)+2*pad
    ox=pad-l; oy=pad-min(t,top_y)
    m=Image.new('L',(W,H),0); ImageDraw.Draw(m).text((ox,oy),text,font=f,fill=255)
    if condense!=1.0:
        nw=max(1,int(W*condense)); m=m.resize((nw,H),Image.LANCZOS)
        ox=ox*condense; l=l*condense; r=r*condense; W=nw
    m=np.array(m).astype(np.float32)/255
    hpx=base_y-top_y
    if bold>0:   # thicken strokes (the original lettering is heavier than the font)
        r_=bold*hpx; d=nd.distance_transform_edt(m<0.5); m=np.clip(np.maximum(m,(r_+0.5-d)),0,1).astype(np.float32)
    # fill: white with soft vertical gradient + light bevel near edges
    yy=np.arange(H,dtype=np.float32)[:,None]
    grad=255-23*np.clip((yy-(oy+top_y))/max(hpx,1),0,1)
    din=nd.distance_transform_edt(m>0.5)
    bevel=1-0.08*np.exp(-din/(0.02*hpx+1e-3))
    rng=np.random.default_rng(3); grain=rng.normal(0,3.0,(H,W)).astype(np.float32)
    v=np.clip(grad*bevel+grain,0,255)
    fill=np.dstack([v,v,np.clip(v+2,0,255)])
    # drop shadow
    dx,dy=int(shadow[0]*hpx),int(shadow[1]*hpx)
    sh=np.zeros_like(m); sh[dy:,dx:]=m[:H-dy,:W-dx] if dy>0 and dx>0 else m
    sh=cv2.GaussianBlur(sh,(0,0),0.045*hpx)*0.8
    rgba=np.zeros((H,W,4),np.float32)
    shc=np.array([4,12,55],np.float32)
    # composite: shadow under fill
    a_s=sh*(1-m); rgba[...,:3]=fill*m[...,None]+shc*a_s[...,None]
    rgba[...,3]=np.clip(m+a_s,0,1)
    rgba[...,:3]=rgba[...,:3]/np.maximum(rgba[...,3:4],1e-4)
    im=Image.fromarray(np.dstack([np.clip(rgba[...,:3],0,255),rgba[...,3]*255]).astype(np.uint8),'RGBA')
    anchor=(ox+(l+r)/2, oy+(top_y+base_y)/2)   # centre of ascender band
    return im, anchor, hpx
def place(canvas, layer, anchor, cx, cy, angle):
    W,H=layer.size
    if angle:
        big=Image.new('RGBA',(W*3,H*3+W*2),(0,0,0,0)); offx,offy=W,H+W
        big.alpha_composite(layer,(offx,offy)); ax,ay=anchor[0]+offx,anchor[1]+offy
        layer=big.rotate(angle,resample=Image.BICUBIC,center=(ax,ay)); anchor=(ax,ay)
        bb=layer.getbbox(); layer=layer.crop(bb); anchor=(anchor[0]-bb[0],anchor[1]-bb[1])
    canvas.alpha_composite(layer,(int(round(cx-anchor[0])),int(round(cy-anchor[1]))))
def render(bow, names, base):
    cfg=BOWS[bow]; W,H=base.size; out=base.copy()
    if len(names)!=len(cfg['slots']): sys.exit(f'{bow} needs {len(cfg["slots"])} names')
    alpha=np.array(base)[...,3]
    for s,nm in zip(cfg['slots'],names):
        if not nm.strip(): continue
        W0,H0=cfg['aspect']; x0,y0,x1,y1=cfg['crop']; k=W/(x1-x0)   # source px -> template px
        asc=s['asc']*H0*k; maxw=s['maxw']*(H0 if s['angle'] else W0)*k
        lay,anc,hpx=text_layer(nm,asc,maxw,cfg['shadow'],cfg.get('condense',1.0),cfg.get('bold',0.0))
        cx,cy=(s['cx']*W0-x0)*k,(s['cy']*H0-y0)*k
        if s.get('up'):
            th=np.radians(s['angle']); nx,ny=-np.sin(th),-np.cos(th)   # 'up' of rotated text
            cx+=nx*s['up']*asc*-1; cy+=ny*s['up']*asc*-1
        place(out,lay,anc,cx,cy,s['angle'])
    a=np.array(out); a[...,3]=alpha; return Image.fromarray(a,'RGBA')   # names never change the cut line
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('bow'); ap.add_argument('names',nargs='+')
    ap.add_argument('--out',default=os.path.join(HERE,'names')); ap.add_argument('--sizes',default='4200,8192')
    ap.add_argument('--blankdir',default=HERE)
    a=ap.parse_args(); os.makedirs(a.out,exist_ok=True); cfg=BOWS[a.bow]
    tag='_'.join(n.replace(' ','') for n in a.names if n.strip())
    for N in [int(x) for x in a.sizes.split(',')]:
        lbl='14in_300dpi' if N==4200 else '8K_14in'
        base=Image.open(os.path.join(a.blankdir,f"{cfg['blank']}_{lbl}.png")).convert('RGBA')
        im=render(a.bow,a.names,base); dpi=max(im.size)/14
        p=os.path.join(a.out,f"NSE_{'Bow3' if a.bow=='bow3' else 'Bow2'}_{tag}_{lbl}.png"); im.save(p,dpi=(dpi,dpi)); print(p)
