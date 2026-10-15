import numpy as np, sys
from PIL import Image
from scipy import ndimage as nd
src=sys.argv[1]; out=sys.argv[2]
C=np.asarray(Image.open(src).convert('RGBA')).astype(np.float32)
H,W=C.shape[:2]
new=np.load('work/renda_new_mask.npy')>0.5
yy,xx=np.mgrid[0:H,0:W]
Z=((xx<813)&(yy>1508+0.4*(600-xx)))|((xx>=813)&(xx<3420)&(yy>1430))
keep=nd.binary_dilation(new,iterations=3)
kill=Z&~keep
C[kill]=0
Image.fromarray(np.clip(C+0.5,0,255).astype(np.uint8)).save(out)
print('cleared',int(kill.sum() and (C[...,3][kill]==0).sum()))
