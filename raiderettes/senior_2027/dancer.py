"""Extract the leaping-dancer silhouette from the photo and trace it to smooth vector curves (potrace)."""
from PIL import Image
import numpy as np, cv2, potrace, pickle
im=np.asarray(Image.open('reference_photo.jpg').convert('RGB')).astype(np.float32)
X0,Y0,X1,Y1=100,975,985,1600            # dancer region in source px
sub=im[Y0:Y1,X0:X1]
gray=sub.mean(2); sat=sub.max(2)-sub.min(2)
U=4                                       # trace at 4x for sub-pixel edges
g=cv2.resize(gray,None,fx=U,fy=U,interpolation=cv2.INTER_CUBIC)
s=cv2.resize(sat,None,fx=U,fy=U,interpolation=cv2.INTER_CUBIC)
g=cv2.GaussianBlur(g,(0,0),U*0.9)
m=((g>112)&(s<60)).astype(np.uint8)
n,lab,st,cen=cv2.connectedComponentsWithStats(m)
seed=(int((440-X0)*U),int((1250-Y0)*U))     # torso
k=lab[seed[1],seed[0]]; print("seed label",k, st[k])
m=(lab==k).astype(np.uint8)
# keep the S of SENIOR out: nothing of the dancer sits right of x=205 below y=1568
yy,xx=np.mgrid[0:m.shape[0],0:m.shape[1]]
m[((yy/U+Y0)>1567)&((xx/U+X0)>205)]=0
# fill pinholes, smooth
cnts,_=cv2.findContours(m,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
f=np.zeros_like(m); cv2.drawContours(f,[max(cnts,key=cv2.contourArea)],-1,1,-1)
f=cv2.morphologyEx(f,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
Image.fromarray(f*255).save('dancer_mask.png')
ys,xs=np.where(f); print("mask bbox src", xs.min()/U+X0, ys.min()/U+Y0, xs.max()/U+X0, ys.max()/U+Y0)
bm=potrace.Bitmap(f==0)   # potracer treats dark (0) as foreground
path=bm.trace(turdsize=40,alphamax=1.05,opticurve=True,opttolerance=0.45)
curves=[]
for c in path:
    segs=[]
    for sgm in c.segments:
        if sgm.is_corner: segs.append(("L",(sgm.c.x,sgm.c.y),(sgm.end_point.x,sgm.end_point.y)))
        else: segs.append(("C",(sgm.c1.x,sgm.c1.y),(sgm.c2.x,sgm.c2.y),(sgm.end_point.x,sgm.end_point.y)))
    curves.append(((c.start_point.x,c.start_point.y),segs))
# store in SOURCE pixel coordinates
def T(p): return (p[0]/U+X0, p[1]/U+Y0)
out=[(T(st_),[(s[0],)+tuple(T(q) for q in s[1:]) for s in segs]) for st_,segs in curves]
pickle.dump(out,open('dancer_path.pkl','wb')); print("curves",len(out),"segments",sum(len(s) for _,s in out))
