import numpy as np, cv2, sys, time
from PIL import Image
from scipy import ndimage as nd
n=sys.argv[1]
img=np.asarray(Image.open(f'../names48/src_{n}.png').convert('RGBA'))
H,W=img.shape[:2]
P=np.load(f'P_{n}.npy')
Pf=cv2.resize(P,(W,H),interpolation=cv2.INTER_LINEAR)
m=(P>0.5); m=nd.binary_opening(m,iterations=2)
lab,k=nd.label(m); sz=nd.sum(m,lab,range(1,k+1)); m=np.isin(lab,1+np.where(sz>3000)[0])
m=nd.binary_fill_holes(nd.binary_closing(m,iterations=3))
mf=cv2.resize(m.astype(np.uint8),(W,H),interpolation=cv2.INTER_NEAREST)>0
ys,xs=np.where(mf); x0,x1,y0,y1=max(xs.min()-40,0),min(xs.max()+40,W),max(ys.min()-40,0),min(ys.max()+40,H)
crop=img[y0:y1,x0:x1]; mc=mf[y0:y1,x0:x1]; Pc=Pf[y0:y1,x0:x1]; al=crop[...,3]
gc=np.full(mc.shape,cv2.GC_PR_BGD,np.uint8)
gc[(Pc>0.5)]=cv2.GC_PR_FGD
gc[nd.binary_erosion(mc,iterations=10)&(Pc>0.8)]=cv2.GC_FGD
gc[~nd.binary_dilation(mc,iterations=14)]=cv2.GC_BGD
gc[al<20]=cv2.GC_BGD
bgr=np.ascontiguousarray(crop[...,:3][...,::-1])
bgdM=np.zeros((1,65),np.float64); fgdM=np.zeros((1,65),np.float64)
t=time.time(); cv2.grabCut(bgr,gc,None,bgdM,fgdM,4,cv2.GC_INIT_WITH_MASK); print('grabcut',round(time.time()-t,1),'s')
fg=(gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD)
fg=nd.binary_opening(fg,iterations=2)
lab,k=nd.label(fg); sz=nd.sum(fg,lab,range(1,k+1)); fg=np.isin(lab,1+np.where(sz>5000)[0])
fg=nd.binary_fill_holes(fg)
full=np.zeros((H,W),bool); full[y0:y1,x0:x1]=fg
np.save(f'work/script_mask_{n}.npy',full)
print('mask px',full.sum(),'vs P-mask',mf.sum())
vis=img[...,:3].astype(np.float32).copy(); vis[img[...,3]<128]=(0,160,90)
edge=full^nd.binary_erosion(full,iterations=2)
vis[edge]=(0,255,255)
Image.fromarray(vis.astype(np.uint8)).save(f'work/script_mask_{n}_vis.png')
