import sys; sys.path.insert(0,'/home/claude/six/tools'); from erase import *
from PIL import Image
im=np.array(Image.open('src/4-48fac636-image.webp').convert('RGBA')); rgb=np.ascontiguousarray(im[...,:3])
boxes=[(400,2260,790,2850),(1320,2260,1640,2810)]
M=np.zeros(rgb.shape[:2],bool); panels=np.zeros_like(M)
for (x0,y0,x1,y1) in boxes:
    X0,Y0=x0-80,y0-80
    sub=rgb[Y0:y1+80,X0:x1+80].astype(int)
    blue=sub[...,2]>sub[...,0]+60
    panel=nd.binary_fill_holes(nd.binary_closing(blue,iterations=8))
    dist=nd.distance_transform_edt(panel)
    mn=sub.min(2); mx=sub.max(2); white=(mn>150)&(mx-mn<70)
    t=white&panel
    lab,n=nd.label(t); keep=np.zeros(n+1,bool)
    for i,sl in enumerate(nd.find_objects(lab)):
        comp=lab[sl]==i+1; area=comp.sum(); dmean=dist[sl][comp].mean()
        if (area>1500 and dmean>25) or (area>150 and dmean>45): keep[i+1]=True
    t=keep[lab]
    # only inside the box
    bx=np.zeros_like(t); bx[80:-80,80:-80]=True; t&=bx
    M[Y0:y1+80,X0:x1+80]|=t; panels[Y0:y1+80,X0:x1+80]|=panel
np.save('bow2_textmask.npy',M); np.save('bow2_panels.npy',panels)
Md=nd.binary_dilation(M,iterations=12)
vis=rgb.copy(); vis[Md]=[255,0,0]; Image.fromarray(vis[2150:2950,300:1750]).save('bow2_mask.jpg')
