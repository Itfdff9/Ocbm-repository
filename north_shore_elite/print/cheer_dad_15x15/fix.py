import numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
im=np.array(Image.open('src.png').convert('RGBA')).astype(np.int32)
al=im[...,3]; rgb=im[...,:3]
ys,xs=np.where(al>0); y0,y1,x0,x1=ys.min()-20,ys.max()+21,xs.min()-20,xs.max()+21
rgb=rgb[y0:y1,x0:x1]; al=al[y0:y1,x0:x1]
op=al>=128
dark=op&(rgb.max(2)<75)                                  # baked-in shirt black + dirty specks
darkish=op&(rgb.max(2)<120)&((rgb.max(2)-rgb.min(2))<40)
dark=dark|(nd.binary_dilation(dark,iterations=2)&darkish)
m=op&~dark
lab,n=nd.label(m); sz=np.bincount(lab.ravel()); keep=sz>=40; keep[0]=False; m=keep[lab]   # drop loose specks
holes=nd.binary_fill_holes(m)&~m
hl,hn=nd.label(holes); hs=np.bincount(hl.ravel()); small=hs<40; small[0]=False; m|=small[hl]  # pinholes
m=cv2.morphologyEx(m.astype(np.uint8),cv2.MORPH_OPEN,np.ones((3,3),np.uint8))>0
# decontaminate colors: everything takes the colour of the solid interior (eroded 3 px)
core=nd.binary_erosion(m,iterations=3)
_,(iy,ix)=nd.distance_transform_edt(~core,return_indices=True)
clean=rgb[iy,ix].astype(np.uint8)
Image.fromarray(clean).save('clean_rgb.png')
np.save('mask.npy',m); print(m.shape, 'removed black px',(op&~m).sum())
v=np.zeros(m.shape+(3,),np.uint8); v[:]=(0,200,0); v[m]=clean[m]; Image.fromarray(v).resize((m.shape[1]//2,m.shape[0]//2)).save('step1.png')
