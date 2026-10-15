import numpy as np, cv2, sys
from PIL import Image
out=np.load(sys.argv[1]); H,W=out.shape[:2]; OUT=sys.argv[2]
newm=np.load('work/renda_new_mask.npy')
pm=np.load('work/renda_new_pm.npy')
a=(pm[...,3:4]/255.0)*newm[...,None]
col=np.where(pm[...,3:4]>1e-3, pm[...,:3]*255.0/np.maximum(pm[...,3:4],1e-3),0)
M=np.float32([[1,0,5],[0,1,9]])
shm=cv2.warpAffine(newm.astype(np.float32),M,(W,H)); shm=cv2.GaussianBlur(shm,(0,0),9)
dark=(shm*0.5)[...,None]*(out[...,3:4]/255.0)
base=out.copy(); base[...,:3]=base[...,:3]*(1-dark)
ra=base[...,3:4]/255.0; oa=a+ra*(1-a)
comp=base.copy(); comp[...,:3]=(col*a+base[...,:3]*ra*(1-a))/np.maximum(oa,1e-4); comp[...,3:4]=oa*255
Image.fromarray(np.clip(comp+0.5,0,255).astype(np.uint8)).save(OUT)
