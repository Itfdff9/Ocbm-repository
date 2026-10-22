from PIL import Image
import numpy as np, cv2
from scipy import ndimage as nd
from scipy.spatial import cKDTree
from collections import deque
a=np.asarray(Image.open("src.webp").convert("RGB")).astype(np.float32); I=a.mean(2); sat=a.max(2)-a.min(2)
H,W=I.shape
# ---------- corners (loose + strict) ----------
r=6; h=(r+1)//2
box=cv2.blur(I,(r,r)); box2=cv2.blur(I*I,(r,r)); bsd=np.sqrt(np.maximum(box2-box*box,0))
bs=cv2.blur(sat,(r,r)); bmin=nd.minimum_filter(I,size=r)
def sh(x,dx,dy): return np.roll(x,(-dy*(h+1),-dx*(h+1)),(0,1))
Q=[(1,1),(-1,1),(-1,-1),(1,-1)]
q=[sh(box,*d) for d in Q]; s=[sh(bs,*d) for d in Q]; m=[sh(bmin,*d) for d in Q]; sd=[sh(bsd,*d) for d in Q]
X=((q[0]+q[2])-(q[1]+q[3]))/2
cons=np.maximum(abs(q[0]-q[2]),abs(q[1]-q[3]))
loose_ok=(np.maximum.reduce(s)<=16)&(np.minimum.reduce(m)>=195)&(cons<12)
score=np.where(loose_ok,abs(X),0)
pk=(score>=12)&(score==nd.maximum_filter(score,size=9))
ys,xs=np.where(pk); C=np.stack([ys,xs],1); sg=np.sign(X[ys,xs])
strict=(np.maximum.reduce(sd)<=9)[ys,xs]&(abs(X[ys,xs])>=14)
t=cKDTree(C); pairs=np.array(list(t.query_pairs(36)))
d=np.abs(C[pairs[:,0]]-C[pairs[:,1]]); same=sg[pairs[:,0]]==sg[pairs[:,1]]
axis=(((d[:,0]>=13)&(d[:,0]<=24)&(d[:,1]<=5))|((d[:,1]>=13)&(d[:,1]<=24)&(d[:,0]<=5)))&~same
diag=(d[:,0]>=13)&(d[:,0]<=24)&(d[:,1]>=13)&(d[:,1]<=24)&same
E=pairs[axis|diag]
# strict seeds: strict corners with >=2 strict alternating axis neighbours (as before)
Pa=pairs[axis]; sm=strict[Pa[:,0]]&strict[Pa[:,1]]
cnt=np.bincount(Pa[sm].ravel(),minlength=len(C)); seed=strict&(cnt>=2)
for it in range(3):
    keep=seed[Pa[:,0]]&seed[Pa[:,1]]&sm; cnt=np.bincount(Pa[keep].ravel(),minlength=len(C)); seed=strict&(cnt>=2)
# grow over lattice edges; a loose corner joins if it has >=2 lattice edges to accepted corners
adj=[[] for _ in range(len(C))]
for p,q2 in E: adj[p].append(q2); adj[q2].append(p)
V=seed.copy(); changed=True
while changed:
    changed=False
    for i in np.where(~V)[0]:
        if sum(V[j] for j in adj[i])>=2: V[i]=True; changed=True
print("corners",len(C),"seeds",seed.sum(),"accepted",V.sum())
CV=C[V]; SV=sg[V]
# per-corner cell size (median axis-neighbour spacing)
tv=cKDTree(CV); pv=np.array(list(tv.query_pairs(26))); dv=np.abs(CV[pv[:,0]]-CV[pv[:,1]])
cx_=np.full(len(CV),18.5); cy_=np.full(len(CV),18.5)
hx=(dv[:,1]>=13)&(dv[:,1]<=24)&(dv[:,0]<=5); hy=(dv[:,0]>=13)&(dv[:,0]<=24)&(dv[:,1]<=5)
for msk,arr,col in ((hx,cx_,1),(hy,cy_,0)):
    acc=np.zeros(len(CV)); n=np.zeros(len(CV))
    for k in (0,1): np.add.at(acc,pv[msk,k],dv[msk,col]); np.add.at(n,pv[msk,k],1)
    arr[n>0]=acc[n>0]/n[n>0]
# ---------- lattice prediction ----------
cmap=np.ones((H,W),bool); cmap[CV[:,0],CV[:,1]]=False
_,(iy,ix)=nd.distance_transform_edt(cmap,return_indices=True)
idx=-np.ones((H,W),np.int64); idx[CV[:,0],CV[:,1]]=np.arange(len(CV))
nid=idx[iy,ix]
yy,xx=np.mgrid[0:H,0:W]
dx=xx-ix; dy=yy-iy; cxs=cx_[nid]; cys=cy_[nid]
kx=np.floor(dx/cxs); ky=np.floor(dy/cys)
even=((kx+ky)%2)==0
pred_light=np.where(SV[nid]>0,even,~even)
rng=np.maximum(np.abs(dx)/cxs,np.abs(dy)/cys)           # distance to the nearest accepted corner, in cells
fx=np.mod(dx,cxs); fy=np.mod(dy,cys)
lined=np.minimum.reduce([fx,cxs-fx,fy,cys-fy])             # distance to the predicted grid lines
# ---------- dark cells ----------
neu=sat<=16
D=neu&(I>=198)&(I<=233)
De=nd.binary_erosion(D)
lab,n=nd.label(De); objs=nd.find_objects(lab); area=np.bincount(lab.ravel())
cent=np.array(nd.center_of_mass(De,lab,range(1,n+1)))
acc=np.zeros(n+1,bool)
for i,sl in enumerate(objs,1):
    if sl is None: continue
    hh=sl[0].stop-sl[0].start; ww=sl[1].stop-sl[1].start
    if not(25<=area[i]<=520 and hh<=27 and ww<=27): continue
    y,x=cent[i-1]; yi,xi=int(round(y)),int(round(x))
    if rng[yi,xi]>2.2 or pred_light[yi,xi]: continue
    # centroid near the predicted cell centre
    if abs(fx[yi,xi]-cxs[yi,xi]/2)>6.5 or abs(fy[yi,xi]-cys[yi,xi]/2)>6.5: continue
    acc[i]=True
dark=acc[lab]
dark=nd.binary_dilation(dark,iterations=2)&D
print("dark blobs",n,"accepted",acc.sum())
# light pixels sandwiched between accepted dark cells (predicted light)
def nearest_dir(mask,axis,rev):
    # distance to the nearest mask pixel scanning along axis in one direction
    m=mask if not rev else np.flip(mask,axis)
    L=m.shape[axis]; ar=np.arange(L,dtype=np.float32)
    shp=[1,1]; shp[axis]=L; ar=ar.reshape(shp)
    last=np.where(m,ar,-1e9); last=np.maximum.accumulate(last,axis=axis); dist=ar-last
    return dist if not rev else np.flip(dist,axis)
L_=neu&(I>=226)
dl=nearest_dir(dark,1,False); dr=nearest_dir(dark,1,True); du=nearest_dir(dark,0,False); dd=nearest_dir(dark,0,True)
sand=((dl<=26)&(dr<=26))|((du<=26)&(dd<=26))
light=L_&pred_light&sand&(rng<=2.2)
N=(sat<=20)&(I>=188)
cm=np.zeros((H,W),np.uint8); cm[CV[:,0],CV[:,1]]=1
seedbox=np.zeros((H,W),np.uint8); cs=C[seed]; seedbox[cs[:,0],cs[:,1]]=1
boxm=(cv2.dilate(seedbox,np.ones((39,39),np.uint8))>0)&N
zone=np.load("zone.npy")&(cv2.dilate(cm,np.ones((61,61),np.uint8))>0)&N
bg=dark|light|boxm|zone
np.save('core4.npy',bg); np.save('lined.npy',(lined<=3.5)&(rng<=2.2)); raise SystemExit
# transitions around found cells (grid-line pixels, neutral, checker tones)
trans=N&(I>=200)&(lined<=3.5)&(rng<=2.2)
for i in range(3): bg|=nd.binary_dilation(bg)&trans
for i in range(2): bg|=nd.binary_dilation(bg)&N&(I>=200)
# floating neutral islands
rest=N&~bg
lab,n=nd.label(rest,structure=np.ones((3,3)))
art=~N&~bg
touch=nd.maximum(nd.binary_dilation(art,iterations=2).astype(np.int32),lab,range(1,n+1))
isl=np.isin(lab,1+np.where(touch==0)[0]); bg|=isl
for i in range(2):
    ring=nd.binary_dilation(bg)&~bg; bg|=ring&(a.min(2)>175)&(sat<=35)
fg=~bg
lab,n=nd.label(fg); sz=np.bincount(lab.ravel()); small=np.isin(lab,np.where(sz<8)[0])&fg; bg|=small
lab,n=nd.label(bg); sz=np.bincount(lab.ravel()); holes=np.isin(lab,np.where(sz<8)[0])&bg; bg&=~holes
np.save("bg4.npy",bg)
print("bg share",round(bg.mean(),4),"dark px",int(dark.sum()),"light px",int(light.sum()),"islands",int(isl.sum()))
u8=a.astype(np.uint8); blk=u8.copy(); blk[bg]=0
Image.fromarray(blk).resize((W*2//5,H*2//5)).save("bg4_blk.png")
for nm,(y0,y1,x0,x1) in {"left":(550,1150,100,500),"bottom":(1550,1850,600,1300),"rows":(600,800,1300,1700),"top":(0,450,1000,1700),"heart":(0,350,1950,2450),"center":(1000,1250,1100,1600)}.items():
    Image.fromarray(np.concatenate([u8[y0:y1,x0:x1],blk[y0:y1,x0:x1]],1)).save(f"p_{nm}.png")
# ---- debug overlays
for nm,(y0,y1,x0,x1) in {"rows":(600,800,1300,1700),"center":(1000,1250,1100,1600),"bottom":(1550,1850,600,1300)}.items():
    v=u8[y0:y1,x0:x1].copy()
    dm=dark[y0:y1,x0:x1]; v[dm]=(v[dm]*0.4+np.array([0,200,0])*0.6).astype(np.uint8)
    lm=light[y0:y1,x0:x1]; v[lm]=(v[lm]*0.4+np.array([255,200,0])*0.6).astype(np.uint8)
    for (y,x),sv,vv in zip(C,seed,V):
        if y0<=y<y1 and x0<=x<x1:
            v[max(0,y-y0-2):y-y0+3,max(0,x-x0-2):x-x0+3]=[255,0,255] if sv else ([0,255,255] if vv else [255,0,0])
    Image.fromarray(v).resize(((x1-x0)*2,(y1-y0)*2),Image.NEAREST).save(f"dbg_{nm}.png")
