"""NSE name master template: put any child's name on the North Shore Elite (Sagamore) art.
Usage:  python3 tools/nse_edit/make_name.py "Ava" [more names...]
Output: north_shore_elite/print/names/NSE_<Name>_14x14in_300dpi.png (4200 px) and _8K_14in.png (8192 px, opens at 14 in)
Layout = master template (Khloe, 2026-10-13): art at 91% width, top-aligned; name in Pacifico, straight (no tilt),
centred at x 0.49 / y 0.885 of the square, size 0.145; black glossy fill, blue / white / black outline, 3 sparkles.
Transparent background, hard alpha (0/255), specks under ~0.4 mm removed."""
import os, sys, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import name as nm
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS=None
ROOT=os.path.dirname(os.path.dirname(HERE))
SRC=os.path.join(ROOT,'north_shore_elite/reference/NSE_Sagamore_master_art_5016px_source.png')
FONT=os.path.join(HERE,'fonts/Pacifico-Regular.ttf')
OUT=os.path.join(ROOT,'north_shore_elite/print/names'); os.makedirs(OUT,exist_ok=True)
SCALE,NAME_H,CX,CY=0.91,0.145,0.49,0.885
SPARKLES=[(0.18,0.3,1.0),(0.55,0.32,0.8),(0.83,0.6,0.7)]
def harden(im,min_area):
    a=np.array(im); m=a[...,3]>=128
    lab,_=nd.label(m); sz=np.bincount(lab.ravel()); keep=sz>=min_area; keep[0]=False; m=keep[lab]
    a[...,3]=np.where(m,255,0); a[~m,:3]=0; return Image.fromarray(a.astype(np.uint8),'RGBA')
def build(text,N,design):
    D=int(N*SCALE); c=Image.new('RGBA',(N,N),(0,0,0,0))
    c.alpha_composite(design.resize((D,D),Image.LANCZOS),((N-D)//2,int(N*0.005)))
    t=nm.render(text,FONT,int(N*NAME_H),angle=0,gaps={1:10} if len(text)>1 else None)
    W,H=t.size; t=nm.add_sparkles(t,[(x*W,y*H,s) for x,y,s in SPARKLES],int(N*0.03))
    c.alpha_composite(t,(int(N*CX-W/2),int(N*CY-H/2))); return c
if __name__=='__main__':
    design=Image.open(SRC).convert('RGBA')
    for text in sys.argv[1:]:
        for N,tag,minA in [(4200,'14x14in_300dpi',18),(8192,'8K_14in',70)]:
            im=harden(build(text,N,design),minA); dpi=N/14
            p=os.path.join(OUT,f'NSE_{text.replace(" ","_")}_{tag}.png'); im.save(p,dpi=(dpi,dpi)); print(p)
