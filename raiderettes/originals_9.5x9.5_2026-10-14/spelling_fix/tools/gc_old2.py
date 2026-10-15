import numpy as np, cv2, time
from PIL import Image
from scipy import ndimage as nd
img=np.asarray(Image.open('../names48/src_RENDA.png').convert('RGBA'))
H,W=img.shape[:2]
P=nd.gaussian_filter(np.load('P_RENDA.npy'),1.0)
Pf=cv2.resize(P,(W,H),interpolation=cv2.INTER_LINEAR)
m=(Pf>0.5)&(img[...,3]>128)
m=nd.binary_opening(m,iterations=3)
lab,k=nd.label(m); sz=nd.sum(m,lab,range(1,k+1)); m=np.isin(lab,1+np.where(sz>20000)[0])
ys,xs=np.where(m); x0,x1,y0,y1=max(xs.min()-60,0),min(xs.max()+60,W),max(ys.min()-60,0),min(ys.max()+60,H)
crop=img[y0:y1,x0:x1]; mc=m[y0:y1,x0:x1]; Pc=Pf[y0:y1,x0:x1]; al=crop[...,3]
gc=np.full(mc.shape,cv2.GC_PR_BGD,np.uint8)
gc[Pc>0.5]=cv2.GC_PR_FGD
gc[nd.binary_erosion(mc,iterations=14)&(Pc>0.85)]=cv2.GC_FGD
gc[~nd.binary_dilation(mc,iterations=10)]=cv2.GC_BGD
gc[(Pc<0.08)&~nd.binary_dilation(mc,iterations=2)]=cv2.GC_BGD
gc[al<20]=cv2.GC_BGD
bgr=np.ascontiguousarray(crop[...,:3][...,::-1])
bgdM=np.zeros((1,65),np.float64); fgdM=np.zeros((1,65),np.float64)
t=time.time(); cv2.grabCut(bgr,gc,None,bgdM,fgdM,5,cv2.GC_INIT_WITH_MASK); print('grabcut',round(time.time()-t,1))
fg=(gc==cv2.GC_FGD)|(gc==cv2.GC_PR_FGD)
fg=nd.binary_opening(fg,iterations=2)
lab,k=nd.label(fg); sz=nd.sum(fg,lab,range(1,k+1)); fg=np.isin(lab,1+np.where(sz>8000)[0])
holes=nd.binary_fill_holes(fg)&~fg
lh,nh=nd.label(holes); szh=nd.sum(holes,lh,range(1,nh+1)); fg|=np.isin(lh,1+np.where(szh<3000)[0])
k2=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(61,61))
fg=cv2.morphologyEx(fg.astype(np.uint8),cv2.MORPH_OPEN,k2)>0
full=np.zeros((H,W),bool); full[y0:y1,x0:x1]=fg
np.save('work/renda_old_gc2.npy',full)
print('gc mask',full.sum(),'P-mask',m.sum())
vis=img[...,:3].astype(np.float32).copy(); vis[img[...,3]<128]=(0,160,90)
edge=full^nd.binary_erosion(full,iterations=2); vis[edge]=(0,255,255)
Image.fromarray(vis.astype(np.uint8)).save('work/renda_old_gc2_vis.png')
