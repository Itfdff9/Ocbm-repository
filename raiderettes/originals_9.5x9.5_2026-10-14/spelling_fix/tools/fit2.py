import numpy as np, cv2, json, sys
from PIL import Image
from scipy import ndimage as nd
from scipy.optimize import minimize
def clean(P, thr=0.5):
    m=(P>thr); m=nd.binary_opening(m,iterations=2)
    lab,k=nd.label(m); sz=nd.sum(m,lab,range(1,k+1))
    m=np.isin(lab,1+np.where(sz>3000)[0])
    return nd.binary_fill_holes(nd.binary_closing(m,iterations=3)).astype(np.float32)
n=sys.argv[1]
al=json.load(open(f'work/align_{n}.json'))[n]
R=clean(np.load('P_RENDA.npy')); H,W=R.shape
D=clean(np.load(f'P_{n}.npy'))
art=np.asarray(Image.open('../names48/src_RENDA.png').resize((W,H),Image.LANCZOS))[...,3]>128
letters=art&~(R>0.5)
ys,xs=np.where(R>0); Rc=np.array([(xs.min()+xs.max())/2,(ys.min()+ys.max())/2])
M0=np.array(al['M'])
# parametrize: extra transform around Rc: [sx, sy, rot(deg), tx, ty, k]
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
Rs=cv2.GaussianBlur(R,(0,0),2); Ls=letters.astype(np.float32); Ts=(~art).astype(np.float32)
def warp(p):
    sx,sy,rot,tx,ty,k=p
    c,s=np.cos(np.radians(rot)),np.sin(np.radians(rot))
    A=np.array([[sx*c,-sy*s],[sx*s,sy*c]])
    E=np.eye(3); E[:2,:2]=A; E[:2,2]=Rc-A@Rc+np.array([tx,ty])
    M=(E@np.vstack([M0,[0,0,1]]))[:2]
    Wd=cv2.warpAffine(D,M,(W,H))
    mapy=(yy-(k*1e-4*(xx-Rc[0])**2)).astype(np.float32)
    return cv2.remap(Wd,xx,mapy,cv2.INTER_LINEAR), M
def cost(p):
    A,_=warp(p)
    A=cv2.GaussianBlur(A,(0,0),1.5)
    exposed=(R*(1-A)).sum()
    extraL=(A*(1-R)*Ls).sum(); extraT=(A*(1-R)*Ts).sum()
    return (exposed+0.35*extraL+1.2*extraT)/R.sum()
p0=np.array([1,1,0,0,0,al['k']*1e4])
best=(cost(p0),p0)
print('start',round(best[0],4))
for it in range(3):
    r=minimize(cost,best[1],method='Nelder-Mead',options=dict(maxiter=400,xatol=1e-3,fatol=1e-5,initial_simplex=None if it else np.array([p0,p0+[0.04,0,0,0,0,0],p0+[0,0.04,0,0,0,0],p0+[0,0,1,0,0,0],p0+[0,0,0,10,0,0],p0+[0,0,0,0,10,0],p0+[0,0,0,0,0,0.5]])))
    if r.fun<best[0]: best=(r.fun,r.x)
    print('iter',it,round(r.fun,4),np.round(r.x,4))
p=best[1]; A,M=warp(p); A=A>0.5; old=R>0.5
print('exposed',round((old&~A).sum()/old.sum(),3),'extra over letters',round((A&~old&letters).sum()/old.sum(),3),'extra over transparent',round((A&~old&~art).sum()/old.sum(),3))
json.dump(dict(p=p.tolist(),M=M.tolist(),Rc=Rc.tolist()),open(f'work/fit2_{n}.json','w'))
src=np.asarray(Image.open('../names48/src_RENDA.png').convert('RGB').resize((W,H),Image.LANCZOS)).astype(np.float32)
vis=src*0.35; vis[old&~A]=(255,0,255); vis[A&~old]=(0,200,0); vis[A&old]=vis[A&old]*0.5+100
Image.fromarray(vis.astype(np.uint8)[380:880,250:1760]).save(f'work/fit2_{n}.png')
