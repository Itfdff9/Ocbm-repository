import numpy as np, cv2, json
from PIL import Image
from scipy import ndimage as nd
from scipy.interpolate import CubicSpline
R=np.asarray(Image.open('../names48/src_RENDA.png').convert('RGBA')).astype(np.float32)
Hf,Wf=R.shape[:2]
D=np.asarray(Image.open('../names48/src_DILEO.png').convert('RGBA')).astype(np.float32)
Dm=np.load('work/script_mask_DILEO_r60.npy').astype(np.float32)
f2=json.load(open('work/fit2_DILEO.json')); f4=json.load(open('work/fit4_DILEO.json'))
M=np.array(f2['M']); Rc=np.array(f2['Rc']); k=f2['p'][5]
p=np.array(f4['p']); knots=np.array(f4['knots']); cyq=np.array(f4['cy']); NK=f4['NK']
Minv=cv2.invertAffineTransform(M)
Yf,Xf=np.mgrid[0:Hf,0:Wf].astype(np.float64)
Xq=(Xf-1.5)/4.0; Yq=(Yf-1.5)/4.0
xq_idx=np.arange(len(cyq))
u=CubicSpline(knots,p[:NK]); v=CubicSpline(knots,p[NK:2*NK]); s=CubicSpline(knots,p[2*NK:])
uq=u(Xq); vq=v(Xq); sq=1+s(Xq); cq=np.interp(Xq,xq_idx,cyq)
Xq2=Xq-uq; Yq2=cq+(Yq-cq-vq)/sq
Xh=2*Xq2+0.5; Yh=2*Yq2+0.5
Yw=Yh-k*1e-4*(Xh-Rc[0])**2; Xw=Xh
Xd=Minv[0,0]*Xw+Minv[0,1]*Yw+Minv[0,2]; Yd=Minv[1,0]*Xw+Minv[1,1]*Yw+Minv[1,2]
mapx=(2*Xd+0.5).astype(np.float32); mapy=(2*Yd+0.5).astype(np.float32)
np.save('work/renda_mapx.npy',mapx); np.save('work/renda_mapy.npy',mapy)
pm=D.copy(); pm[...,:3]*=pm[...,3:4]/255.0
warped=np.stack([cv2.remap(pm[...,c],mapx,mapy,cv2.INTER_LINEAR,borderValue=0) for c in range(4)],-1)
wm=cv2.remap(Dm,mapx,mapy,cv2.INTER_LINEAR,borderValue=0)
np.save('work/renda_new_pm.npy',warped.astype(np.float32)); np.save('work/renda_new_mask.npy',wm.astype(np.float32))
# scale sanity: local jacobian
J=np.hypot(np.gradient(mapx,axis=1),np.gradient(mapy,axis=1))
mm=wm>0.5
print('donor px per output px (x-dir) percentiles',np.percentile(J[mm],[5,50,95]).round(3))
a=(warped[...,3:4]/255.0)*wm[...,None]
col=np.where(warped[...,3:4]>1e-3, warped[...,:3]*255.0/np.maximum(warped[...,3:4],1e-3),0)
comp=R.copy()
ra=R[...,3:4]/255.0
oa=a+ra*(1-a)
comp[...,:3]=(col*a+R[...,:3]*ra*(1-a))/np.maximum(oa,1e-4); comp[...,3:4]=oa*255
Image.fromarray(np.clip(comp,0,255).astype(np.uint8)).save('work/RENDA_paste_raw.png')
print('done')
