# export.py KEY OUTNAME : up/KEY_x.png + up/KEY_a.png -> out/OUTNAME_14in_300dpi.png (long side 4200) + out/OUTNAME_8K_14in.png (long side 8192)
import sys, numpy as np
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS=None
key,name=sys.argv[1],sys.argv[2]
x=Image.open(f'/home/claude/six/up/{key}_x.png').convert('RGB'); a=Image.open(f'/home/claude/six/up/{key}_a.png')
# crop to the artwork (alpha bbox) + small margin so 14 in = the art, not the empty canvas
A=np.array(a); ys,xs=np.where(A>=128); mg=int(0.006*max(A.shape))
bx=(max(0,xs.min()-mg),max(0,ys.min()-mg),min(A.shape[1],xs.max()+1+mg),min(A.shape[0],ys.max()+1+mg))
S=x.size[0]/A.shape[1]
a=a.crop(bx); x=x.crop(tuple(int(round(v*S)) for v in bx))
print('crop (source px)',bx)
for L,lbl,minA in [(4200,'14in_300dpi',18),(8192,'8K_14in',70)]:
    W,H=x.size; s=L/max(W,H); size=(round(W*s),round(H*s))
    rgb=np.array(x.resize(size,Image.LANCZOS)); al=np.array(a.resize(size,Image.LANCZOS))
    m=al>=128; lab,_=nd.label(m); sz=np.bincount(lab.ravel()); keep=sz>=minA; keep[0]=False; m=keep[lab]
    rgb[~m]=0
    out=Image.fromarray(np.dstack([rgb,np.where(m,255,0).astype(np.uint8)]),'RGBA')
    dpi=L/14; p=f'/home/claude/six/out/{name}_{lbl}.png'; out.save(p,dpi=(dpi,dpi)); print(p,out.size,round(dpi,1))
