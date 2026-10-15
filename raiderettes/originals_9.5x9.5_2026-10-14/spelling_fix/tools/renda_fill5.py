import numpy as np, cv2, json, sys
from PIL import Image
from scipy import ndimage as nd
R=np.asarray(Image.open('../names48/src_RENDA.png').convert('RGBA')).astype(np.float32)
H,W=R.shape[:2]
P=json.load(open(sys.argv[1])) if len(sys.argv)>1 else {}
DIL_E=P.get('DIL_E',12); DIL_V=P.get('DIL_V',16)
old=np.load('work/renda_old_gc2.npy')
newm=np.load('work/renda_new_mask.npy'); new=newm>0.5
for (x0,y0,x1,y1) in P.get('FALSE',[(712,1262,830,1440)]): old[y0:y1,x0:x1]=False
E=nd.binary_dilation(old,iterations=DIL_E)&~nd.binary_erosion(new,iterations=3)&(R[...,3]>0)
E=nd.binary_opening(E,iterations=1)
from PIL import ImageDraw
def poly(pts):
    im=Image.new('L',(W,H),0); ImageDraw.Draw(im).polygon([tuple(p) for p in pts],fill=255); return np.asarray(im)>127
for pts in P.get('EXTRA',[]):
    E|=poly(pts)&~nd.binary_erosion(new,iterations=3)
valid=~nd.binary_dilation(old,iterations=DIL_V)
TRANSP=P.get('TRANSP',[])
RULES=P.get('RULES',[])
DEFAULT=P.get('DEFAULT',[(0,34),(28,1)])
ys,xs=np.where(E)
rule_id=np.full(len(ys),-1)
for j,(b,v) in enumerate(RULES):
    m=(xs>=b[0])&(xs<b[2])&(ys>=b[1])&(ys<b[3])&(rule_id==-1); rule_id[m]=j
tmask=np.zeros(len(ys),bool)
for b in TRANSP:
    tmask|=(xs>=b[0])&(xs<b[2])&(ys>=b[1])&(ys<b[3])
out=R.copy()
def fill(idx, vecs):
    yy=ys[idx]; xx=xs[idx]
    best_d=np.full(len(yy),np.inf); best=np.zeros((len(yy),4),np.float32)
    for (vx,vy) in vecs:
        L=np.hypot(vx,vy)
        for sgn in (1,-1):
            found=np.zeros(len(yy),bool); val=np.zeros((len(yy),4),np.float32); dist=np.full(len(yy),np.inf)
            for k in range(1,int(450/L)+2):
                sy=yy-sgn*k*vy; sx=xx-sgn*k*vx
                inb=(sy>=0)&(sy<H)&(sx>=0)&(sx<W)
                v=np.zeros(len(yy),bool); v[inb]=valid[sy[inb],sx[inb]]
                take=v&~found; val[take]=R[sy[take],sx[take]]; dist[take]=k*L; found|=take
                if found.all(): break
            better=dist<best_d; best_d[better]=dist[better]; best[better]=val[better]
    ok=np.isfinite(best_d)
    res=np.zeros((len(yy),4),np.float32); res[ok]=best[ok]
    out[yy,xx]=res
    return (~ok).sum()
for j,(b,v) in enumerate(RULES):
    idx=np.where((rule_id==j)&~tmask)[0]
    if len(idx): print('rule',j,b,len(idx),'unfilled',fill(idx,v))
idx=np.where((rule_id==-1)&~tmask)[0]; print('default',len(idx),'unfilled',fill(idx,DEFAULT))
out[ys[tmask],xs[tmask]]=0
for pts in P.get('CLEAR',[]):
    m=poly(pts)&~nd.binary_dilation(new,iterations=2); out[m]=0
np.save('work/renda_base4.npy',out)
pm=np.load('work/renda_new_pm.npy')
a=(pm[...,3:4]/255.0)*newm[...,None]
col=np.where(pm[...,3:4]>1e-3, pm[...,:3]*255.0/np.maximum(pm[...,3:4],1e-3),0)
SO=P.get('SHADOW',[5,9,9,0.5])
M=np.float32([[1,0,SO[0]],[0,1,SO[1]]])
shm=cv2.warpAffine(newm.astype(np.float32),M,(W,H)); shm=cv2.GaussianBlur(shm,(0,0),SO[2])
dark=(shm*SO[3])[...,None]*(out[...,3:4]/255.0)
base=out.copy(); base[...,:3]=base[...,:3]*(1-dark)
ra=base[...,3:4]/255.0; oa=a+ra*(1-a)
comp=base.copy(); comp[...,:3]=(col*a+base[...,:3]*ra*(1-a))/np.maximum(oa,1e-4); comp[...,3:4]=oa*255
OUT=P.get('OUT','work/RENDA_fix_v4.png')
Image.fromarray(np.clip(comp+0.5,0,255).astype(np.uint8)).save(OUT)
print('saved',OUT)
