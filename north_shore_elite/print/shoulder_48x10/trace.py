import numpy as np, cv2, sys
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS=None
a=np.array(Image.open('/home/claude/nsy/ref.jpg').convert('RGB')).astype(np.float32)
fg=np.load('/home/claude/nsy/fg2.npy')
ys,xs=np.where(fg); y0,y1,x0,x1=ys.min()-8,ys.max()+9,xs.min()-8,xs.max()+9
c=a[y0:y1,x0:x1]; f=fg[y0:y1,x0:x1]
c=cv2.bilateralFilter(c.astype(np.uint8),5,30,5).astype(np.float32)
r,g,b=c[...,0],c[...,1],c[...,2]; mn=c.min(2)
sat=lambda v,lo,hi: np.clip((v-lo)/(hi-lo),0,1)
pW=sat(mn,140,190)
pB=sat(b-r,45,95)*sat(b,100,160)*(1-pW)
pL=sat(b-r,20,50)*sat(mn,80,115)*(1-pW)*(1-pB)
# script region: white strokes in the lower-middle that are the "Elites" script (not letter outlines)
H0,W0=f.shape
F=float(sys.argv[1]) if len(sys.argv)>1 else 5.24
Wt,Ht=int(W0*F),int(H0*F)
up=lambda m: cv2.resize(m.astype(np.float32),(Wt,Ht),interpolation=cv2.INTER_CUBIC)
sm=lambda m,s: cv2.GaussianBlur(m,(0,0),s)
W=sm(up(pW),0.75*F)>0.5
B=sm(up(pB),0.7*F)>0.5
L=sm(up(pL),0.5*F)>0.45
D=sm(up(f.astype(np.float32)),0.9*F)>0.5            # whole design footprint (incl. dark outline of Elites)
# remove dark shirt gaps that are not part of the script outline: keep dark only near the script
scriptW=np.zeros_like(W)
sy0,sy1,sx0,sx1=int(0.42*Ht),Ht,int(0.20*Wt),int(0.80*Wt)
scriptW[sy0:sy1,sx0:sx1]=W[sy0:sy1,sx0:sx1]
halo=nd.distance_transform_edt(~scriptW)<=4.2*F
swoosh=np.zeros_like(B); swoosh[int(0.80*Ht):,int(0.30*Wt):int(0.78*Wt)]=B[int(0.80*Ht):,int(0.30*Wt):int(0.78*Wt)]
halo|=nd.distance_transform_edt(~swoosh)<=2.0*F
halo=sm(halo.astype(np.float32),1.2*F)>0.5
L&=(nd.distance_transform_edt(~W)>2.2*F)&nd.binary_dilation(B,iterations=max(1,int(F)))
A=W|B|L|(halo&D)
A=nd.binary_opening(A,iterations=2)
lab,n=nd.label(A); sz=np.bincount(lab.ravel()); keep=sz>=int((1.2*F)**2*3); keep[0]=False; A=keep[lab]
holes=nd.binary_fill_holes(A)&~A
hl,hn=nd.label(holes); hs=np.bincount(hl.ravel()); small=hs<0.0035*Ht*Wt; small[0]=False
HOLE=small[hl]; A|=HOLE
col=np.zeros((Ht,Wt,3),np.uint8); col[:]=(14,20,48)        # navy-black outline of the script
col[HOLE]=(10,10,14); col[L]=(140,168,232); col[B&~L]=(16,64,214); col[W]=(250,251,255)
col[~A]=0
out=np.dstack([col,np.where(A,255,0).astype(np.uint8)])
np.save('/home/claude/nsy/trace_shape.npy',np.array([Ht,Wt]))
Image.fromarray(out,'RGBA').save('/home/claude/nsy/traced_design.png')
print(Wt,Ht)
