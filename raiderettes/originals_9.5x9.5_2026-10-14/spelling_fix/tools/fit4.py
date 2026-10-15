import numpy as np, cv2, json, sys
from PIL import Image
from scipy import ndimage as nd
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
def clean(P, thr=0.5):
    m=(P>thr); m=nd.binary_opening(m,iterations=2)
    lab,k=nd.label(m); sz=nd.sum(m,lab,range(1,k+1))
    m=np.isin(lab,1+np.where(sz>3000)[0])
    return nd.binary_fill_holes(nd.binary_closing(m,iterations=3)).astype(np.float32)
n=sys.argv[1]; Q=2   # quarter res relative to half-res maps
f2=json.load(open(f'work/fit2_{n}.json'))
R=clean(np.load('P_RENDA.npy')); H,W=R.shape
D=clean(np.load(f'P_{n}.npy'))
art=np.asarray(Image.open('../names48/src_RENDA.png').resize((W,H),Image.LANCZOS))[...,3]>128
M=np.array(f2['M']); Rc=np.array(f2['Rc']); k=f2['p'][5]
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
Wd=cv2.warpAffine(D,M,(W,H)); A0=cv2.remap(Wd,xx,(yy-(k*1e-4*(xx-Rc[0])**2)).astype(np.float32),cv2.INTER_LINEAR)
# quarter-res working copies
def q(a): return cv2.resize(a.astype(np.float32),(W//Q,H//Q),interpolation=cv2.INTER_AREA)
Rq,Aq,artq=q(R),q(A0),q(art)
letq=artq*(1-Rq)
h,w=Rq.shape
ys,xs=np.where(Rq>0.5); x0,x1=xs.min(),xs.max()
# local centerline of the new script (per column) for vertical scaling
cy=np.array([ (np.where(Aq[:,x]>0.5)[0].mean() if (Aq[:,x]>0.5).any() else np.nan) for x in range(w)])
good=~np.isnan(cy); cy=np.interp(np.arange(w),np.where(good)[0],cy[good]); cy=nd.gaussian_filter1d(cy,15)
NK=9; knots=np.linspace(x0-10,x1+10,NK)
Y,X=np.mgrid[0:h,0:w].astype(np.float32)
def field(p):
    u=CubicSpline(knots,p[:NK])(np.arange(w)); v=CubicSpline(knots,p[NK:2*NK])(np.arange(w)); s=1+CubicSpline(knots,p[2*NK:])(np.arange(w))
    return u,v,s
def warpq(p, img):
    u,v,s=field(p)
    mx=(X-u[None,:]).astype(np.float32)
    my=(cy[None,:]+(Y-cy[None,:]-v[None,:])/s[None,:]).astype(np.float32)
    return cv2.remap(img,mx,my,cv2.INTER_LINEAR)
def cost(p):
    A=warpq(p,Aq)
    exposed=(Rq*(1-A)).sum(); exL=(A*(1-Rq)*letq).sum(); exT=(A*(1-artq)).sum()
    reg=0.3*(np.diff(p[:NK])**2).sum()+0.3*(np.diff(p[NK:2*NK])**2).sum()+400*(np.diff(p[2*NK:])**2).sum()+0.002*(p[:2*NK]**2).sum()+20*(p[2*NK:]**2).sum()
    return (exposed+0.25*exL+1.2*exT)/Rq.sum()+reg/Rq.sum()
p0=np.zeros(3*NK)
print('start',round(cost(p0),4))
bounds=[(-12,12)]*NK+[(-8,8)]*NK+[(-0.07,0.08)]*NK
r=minimize(cost,p0,method='Powell',bounds=bounds,options=dict(maxiter=6000,xtol=1e-3,ftol=1e-6))
print('powell',round(r.fun,4)); p=r.x
u,v,s=field(p); print('u',np.round(p[:NK],1)); print('v',np.round(p[NK:2*NK],1)); print('s',np.round(1+p[2*NK:],3))
A=warpq(p,Aq)>0.5; old=Rq>0.5
print('exposed',round((old&~A).sum()/old.sum(),3),'extra letters',round((A&~old&(letq>0.5)).sum()/old.sum(),3),'extra transparent',round((A&(artq<0.5)).sum()/old.sum(),3))
json.dump(dict(p=p.tolist(),knots=knots.tolist(),cy=cy.tolist(),Q=Q,NK=NK),open(f'work/fit4_{n}.json','w'))
src=np.asarray(Image.open('../names48/src_RENDA.png').convert('RGB').resize((w,h),Image.LANCZOS)).astype(np.float32)
vis=src*0.35; vis[old&~A]=(255,0,255); vis[A&~old]=(0,200,0); vis[A&old]=vis[A&old]*0.5+100
Image.fromarray(vis.astype(np.uint8)[190:440,125:880]).resize((1510,500),Image.NEAREST).save(f'work/fit4_{n}.png')
