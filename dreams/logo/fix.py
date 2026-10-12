"""DREAMS logo fixes Brian asked for: S openings -> transparent, FASTPITCH A counter -> transparent, hard alpha."""
from PIL import Image
import numpy as np, cv2
im=Image.open('DREAMS_Fastpitch_star_ORIGINAL.png').convert('RGBA'); a=np.asarray(im).copy()
rgb=a[...,:3].astype(np.float32); L=rgb.mean(2); sat=rgb.max(2)-rgb.min(2)
holes=np.zeros(L.shape,np.uint8)

# 1) the two light pill-shaped slots inside the S (enclosed by the navy band)
X0,Y0,X1,Y1=3600,1150,3900,1800
light=((L>150)&(sat<70)).astype(np.uint8); light[:Y0]=0; light[Y1:]=0; light[:,:X0]=0; light[:,X1:]=0
n,lab,st,cen=cv2.connectedComponentsWithStats(light)
pills=[i for i in range(1,n) if 800<st[i][4]<12000 and st[i][3]>1.4*st[i][2]]
print("S slots:",[tuple(int(v) for v in st[i]) for i in pills])
for i in pills:
    m=(lab==i).astype(np.uint8)
    m=cv2.dilate(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))      # take the soft light rim with it
    hull=cv2.convexHull(cv2.findNonZero(m)); cv2.fillPoly(holes,[hull],1)

# 2) FASTPITCH "A": the recessed floor of the triangle counter (inside its dark bevel walls)
Lb=cv2.GaussianBlur(L,(0,0),1.2)
floor=((Lb>100)).astype(np.uint8); box=np.zeros_like(floor); box[2040:2114,1160:1242]=1; floor&=box
n,lab,st,cen=cv2.connectedComponentsWithStats(floor)
k=lab[2100,1200]; print("A floor:",tuple(int(v) for v in st[k]))
hull=cv2.convexHull(cv2.findNonZero((lab==k).astype(np.uint8))); cv2.fillPoly(holes,[hull],1)

a[holes>0,3]=0
# 3) hard alpha everywhere (no semi-transparent pixels for the DTF underbase/powder)
a[...,3]=np.where(a[...,3]>=128,255,0).astype(np.uint8); a[a[...,3]==0,:3]=0
out=Image.fromarray(a,"RGBA"); out.save('DREAMS_Fastpitch_star_FIXED_300dpi.png',dpi=(300,300))

print("cut px",int(holes.sum()), "alpha values",np.unique(a[...,3]).tolist(), out.size)
