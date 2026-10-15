import numpy as np, cv2, json, sys
from PIL import Image, ImageDraw
from scipy import ndimage as nd
R=np.asarray(Image.open('../names48/src_RENDA.png').convert('RGBA')).astype(np.float32)
H,W=R.shape[:2]
base=np.load(sys.argv[1]); OUT=sys.argv[2]
old=np.load('work/renda_old_gc2.npy')
newm=np.load('work/renda_new_mask.npy'); new=newm>0.5
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
# zone to rebuild: old curl area (generous), not the kept vertical band below 1262
Z=np.zeros((H,W),bool)
im=Image.new('L',(W,H),0)
ImageDraw.Draw(im).polygon([(600,1045),(735,1045),(800,1075),(870,1080),(870,1262),(789,1262),(789,1300),(600,1300)],fill=255)
Z=(np.asarray(im)>127)&~nd.binary_erosion(new,iterations=3)
# geometry
A=np.array([661.,960.]); C1=np.array([726.,1150.]); C2=np.array([826.,1150.])
t=(C1-A)/np.linalg.norm(C1-A)
xline=A[0]+(yy-A[1])*(t[0]/t[1])
inside=(xx<xline)|((yy>1150)&(xx<826))
d_out=nd.distance_transform_edt(~inside); d_in=nd.distance_transform_edt(inside); d=d_out-d_in
# nearest segment for band texture
def segdist(P0,P1):
    v=P1-P0; L2=(v**2).sum()
    tt=np.clip(((xx-P0[0])*v[0]+(yy-P0[1])*v[1])/L2,0,1)
    return np.hypot(xx-(P0[0]+tt*v[0]),yy-(P0[1]+tt*v[1]))
d1=segdist(A,C1); d2=segdist(C1,C2); d3=segdist(C2,np.array([826.,1500.]))
seg=np.argmin(np.stack([d1,d2,d3]),0)
out=base.copy()
ys,xs=np.where(Z)
B=80
# outside -> transparent
m_out=d[ys,xs]>0.5
out[ys[m_out],xs[m_out]]=0
# band zone
def bil(img,x,y):
    pm=img.copy(); 
    res=np.stack([cv2.remap(pm[...,c],x.astype(np.float32)[None],y.astype(np.float32)[None],cv2.INTER_LINEAR)[0] for c in range(4)],-1)
    return res
band=(~m_out)&(d[ys,xs]>-B)
by,bx=ys[band],xs[band]; bs=seg[by,bx]
src_x=np.zeros(len(by)); src_y=np.zeros(len(by))
# S1: along the diagonal, land near y=1015 (visible stretch), keep dot phase (period ~14.2 along the line)
m=bs==0
s=(by[m]-1015)/t[1]; s=np.round(s/14.2)*14.2
src_x[m]=bx[m]-s*t[0]; src_y[m]=by[m]-s*t[1]
# S2: horizontal top edge of the jut -> rotate the vertical band (outer edge x=826) texture
m=bs==1
e=by[m]-1150.0
src_x[m]=826-e; src_y[m]=1272+np.mod(bx[m]-726,34)
# S3: vertical band -> copy down by multiples of 34 into the visible part (y>=1262)
m=bs==2
k=np.ceil(np.maximum(1262-by[m],0)/34.0); src_x[m]=bx[m]; src_y[m]=by[m]+34*k
vals=bil(R,src_x,src_y)
out[by,bx]=vals
# interior: lattice copy from valid original (not old script)
valid=~nd.binary_dilation(old,iterations=16)&~Z&(d<=-B-4)
inter=(~m_out)&(d[ys,xs]<=-B)
iy,ix=ys[inter],xs[inter]
best_d=np.full(len(iy),np.inf); best=np.zeros((len(iy),4),np.float32)
for (vx,vy) in [(0,34),(28,1),(28,35)]:
    L=np.hypot(vx,vy)
    for sg in (1,-1):
        found=np.zeros(len(iy),bool); val=np.zeros((len(iy),4),np.float32); dist=np.full(len(iy),np.inf)
        for k in range(1,15):
            sy=iy-sg*k*vy; sx=ix-sg*k*vx
            inb=(sy>=0)&(sy<H)&(sx>=0)&(sx<W)
            v=np.zeros(len(iy),bool); v[inb]=valid[sy[inb],sx[inb]]
            take=v&~found; val[take]=R[sy[take],sx[take]]; dist[take]=k*L; found|=take
        better=dist<best_d; best_d[better]=dist[better]; best[better]=val[better]
out[iy,ix]=best
np.save(OUT,out)
print('R rebuilt: outside',m_out.sum(),'band',band.sum(),'interior',inter.sum(),'unfilled interior',(~np.isfinite(best_d)).sum())
