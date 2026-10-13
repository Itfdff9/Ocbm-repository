import sys, numpy as np, cv2
sys.path.insert(0,'/home/claude/nse_edit/tools')
from compose import compose
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS=None
INCH=14
def harden(rgba, min_area):
    a=np.array(rgba); al=a[...,3]
    m=al>=128
    lab,n=nd.label(m); sz=np.bincount(lab.ravel()); keep=sz>=min_area; keep[0]=False
    m=keep[lab]
    a[...,3]=np.where(m,255,0).astype(np.uint8)
    a[~m,:3]=0
    return Image.fromarray(a,'RGBA')
def save(im,path):
    dpi=im.size[0]/INCH
    im.save(path,dpi=(dpi,dpi),optimize=False,compress_level=6)
    print(path,im.size,round(dpi,2))
x4=Image.open('std_x4_rgb.png').convert('RGB')          # 5016
alpha=Image.open('std_alpha.png').resize(x4.size,Image.LANCZOS)
design=x4.copy(); design.putalpha(alpha)
out='out/'
sp=[(0.18,0.3,1.0),(0.55,0.32,0.8),(0.83,0.6,0.7)]
for N,tag,minA in [(8192,'8K_master_14in',70),(4200,'14x14in_300dpi',18)]:
    # standard, corrected (solid alpha)
    if 0:
      s=design.resize((N,N),Image.LANCZOS)
      save(harden(s,minA), out+f'NSE_Standard_{tag}.png')
    k=compose(design,N,scale=0.85,name_h=0.19,name_cy=0.84,name_cx=0.49,sparkle_pts=sp)
    save(harden(k,minA), out+f'NSE_Khloe_{tag}.png')
