"""NSE name template, v2 style (Brian's 2026-10-14 reference): slanted rounded script (Courgette),
black glossy fill / blue band / white / black outline, straight, with white-and-blue 4-point stars beside the name.
Usage: python3 tools/nse_edit/make_name_v2.py "Laura" [more names]"""
import os, sys, math, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import name as nm, make_name as mn
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS=None
FONT=os.path.join(HERE,'fonts/Courgette-Regular.ttf')
BOLD=0.5          # extra stroke weight (U = font size/100)
def star4(R, inner=0.27):
    """4-point sparkle star (tall diamond with concave sides), radius R px"""
    C=R*1.2; pts=[]
    outer=[(0,-1.12),(0.82,0),(0,1.12),(-0.82,0)]
    for i in range(4):
        p=np.array(outer[i]); q=np.array(outer[(i+1)%4]); mid=(p+q)/2; ctrl=mid*inner/np.linalg.norm(mid)*0.75
        for t in np.linspace(0,1,40,endpoint=False):       # quadratic curve pulled toward the centre
            v=(1-t)**2*p+2*(1-t)*t*ctrl+t*t*q; pts.append((C+R*v[0],C+R*v[1]))
    m=np.zeros((int(C*2),int(C*2)),np.uint8); cv2.fillPoly(m,[np.array(pts,np.int32)],255,cv2.LINE_AA); return m
def star_rgba(R):
    m=star4(R); U=R/100
    blue=nm.dil(m,0); white_core=cv2.erode(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(int(18*U)*2+1,)*2))
    outer=nm.dil(np.pad(m,int(16*U)+4),12*U); m=np.pad(m,int(16*U)+4); white_core=np.pad(white_core,int(16*U)+4)
    H,W=m.shape; rgb=np.zeros((H,W,3),np.float32)
    lay=lambda mask,col: rgb.__setitem__(slice(None), rgb*(1-(mask.astype(np.float32)/255)[...,None])+np.array(col,np.float32)*(mask.astype(np.float32)/255)[...,None])
    lay(outer,nm.BLACK); lay(m,(60,110,255))
    yy,xx=np.mgrid[0:H,0:W]; r=np.hypot(xx-W/2,yy-H/2)/(R)
    glow=np.clip(1-r*1.3,0,1)[...,None]
    core=(white_core.astype(np.float32)/255)[...,None]
    rgb=rgb*(1-core)+np.array([255,255,255],np.float32)*core
    rgb=np.where((m[...,None]>127),rgb*(1-glow*0.6)+255*glow*0.6,rgb)
    return Image.fromarray(np.dstack([np.clip(rgb,0,255),outer]).astype(np.uint8),'RGBA')
def sharp(mask_bool):
    """1-px antialiased coverage from a hard mask"""
    d_in=nd.distance_transform_edt(mask_bool); d_out=nd.distance_transform_edt(~mask_bool)
    return np.clip(0.5+d_in-d_out,0,1)
def render_name(text,height):
    """crisp layers: glossy black script, blue band, white band, black outline"""
    size=int(height); U=size/100; pad=int(32*U)
    g=nm.glyph_mask(text,FONT,size,pad)>127
    g=nd.distance_transform_edt(~g)<=BOLD*U
    d_out=nd.distance_transform_edt(~g)
    cov=lambda r: np.clip(r+0.5-d_out,0,1)
    fill=sharp(g); blue=cov(4.5*U); white=cov(11*U); outer=cov(13.5*U)
    H,W=g.shape; rgb=np.zeros((H,W,3),np.float32)
    for m,col in ((outer,nm.BLACK),(white,nm.WHITE),(blue,nm.BLUE)):
        rgb=rgb*(1-m[...,None])+np.array(col,np.float32)*m[...,None]
    # glossy black: dark-blue rim light on the upper-left inside edge
    din=nd.distance_transform_edt(g)
    sh=np.roll(np.roll(g,int(1.6*U),0),int(1.1*U),1)
    rim=sharp(g&~sh)
    col=np.zeros((H,W,3),np.float32)+np.array([4,4,10],np.float32)+rim[...,None]*np.array([22,48,150],np.float32)
    rgb=rgb*(1-fill[...,None])+col*fill[...,None]
    return Image.fromarray(np.dstack([np.clip(rgb,0,255),outer*255]).astype(np.uint8),'RGBA')
def build(text,N,design):
    D=int(N*mn.SCALE); c=Image.new('RGBA',(N,N),(0,0,0,0))
    c.alpha_composite(design.resize((D,D),Image.LANCZOS),((N-D)//2,int(N*0.005)))
    t=render_name(text,int(N*0.26)); bb=t.getbbox(); t=t.crop(bb); W,H=t.size
    k=min(0.18*N/H, 0.52*N/W)                      # name ~17% of the height (like the reference), max 50% of the width
    t=t.resize((int(W*k),int(H*k)),Image.LANCZOS); W,H=t.size
    cx=N*0.47; y0=int(N*0.965-H); x0=int(cx-W/2)
    if x0<2: x0=2
    stars=[(x0-0.02*H, y0+0.70*H, 0.20*H),(x0+W+0.01*H, y0+0.72*H, 0.22*H),(x0+W+0.10*H, y0+0.32*H, 0.14*H)]
    for sx,sy,R in stars:
        s=star_rgba(R); c.alpha_composite(s,(int(sx-s.width/2),int(sy-s.height/2)))
    c.alpha_composite(t,(x0,y0)); return c
if __name__=='__main__':
    design=Image.open(mn.SRC).convert('RGBA')
    sizes=[(4200,'14x14in_300dpi',18),(8192,'8K_14in',70)]
    if '--preview' in sys.argv: sizes=[(1400,'preview',4)]
    for text in [a for a in sys.argv[1:] if not a.startswith('--')]:
        for N,tag,minA in sizes:
            im=mn.harden(build(text,N,design),minA); dpi=N/14
            p=os.path.join(mn.OUT,f'NSE_{text.replace(" ","_")}_v2_{tag}.png'); im.save(p,dpi=(dpi,dpi)); print(p)
