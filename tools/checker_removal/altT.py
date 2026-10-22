from PIL import Image
import numpy as np, cv2
from scipy import ndimage as nd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
a=np.asarray(Image.open("src.webp").convert("RGB")).astype(np.float32); I=a.mean(2); sat=a.max(2)-a.min(2)
H,W=I.shape
B=cv2.blur(I,(5,5)); BS=cv2.blur(sat,(5,5)); BM=nd.minimum_filter(I,5)
okp=(BS<=70)&(BM>=175)&(cv2.blur(a[...,2]-a[...,0],(5,5))>=-6)
def shift(x,d,axis):  # value at p+d along axis
    return np.roll(x,-d,axis)
best_a={}; core={}
for axis in (1,0):
    bestabs=np.zeros((H,W),np.float32); besta=np.zeros((H,W),np.float32)
    for s in range(13,23):
        bm,bp=shift(B,-s,axis),shift(B,s,axis); bm2,bp2=shift(B,-2*s,axis),shift(B,2*s,axis)
        av=B-(bm+bp)/2
        c2=np.where(shift(okp,-2*s,axis),abs(bm2-B),999); c3=np.where(shift(okp,2*s,axis),abs(bp2-B),999)
        cons=np.maximum(abs(bm-bp),np.minimum(c2,c3))
        ok=okp&shift(okp,-s,axis)&shift(okp,s,axis)&(cons<12)
        v=np.where(ok,abs(av),0)
        m=v>bestabs; bestabs[m]=v[m]; besta[m]=av[m]
    core[axis]=(bestabs>=13); best_a[axis]=besta
pos=(core[1]&(best_a[1]>0))|(core[0]&(best_a[0]>0)); neg=(core[1]&(best_a[1]<0))|(core[0]&(best_a[0]<0))
# reject pixels where the two axes disagree in sign
both=core[1]&core[0]&(np.sign(best_a[1])!=np.sign(best_a[0])); pos&=~both; neg&=~both
labs=[]; cents=[]; sgn=[]; offs=0; L=np.zeros((H,W),np.int32)
for m,sv in ((pos,1),(neg,-1)):
    lab,n=nd.label(m); sz=np.bincount(lab.ravel())
    objs=nd.find_objects(lab)
    keep=[]
    for k in range(1,len(sz)):
        sl=objs[k-1]
        if sl is None or sz[k]<40: continue
        hh=sl[0].stop-sl[0].start; ww=sl[1].stop-sl[1].start
        if hh>24 or ww>24 or sz[k]/(hh*ww)<0.5: continue
        mi=I[sl][lab[sl]==k].mean()
        if (sv>0 and mi<228) or (sv<0 and (mi>236 or mi<180)): continue
        keep.append(k)
    keep=np.array(keep)
    c=np.array(nd.center_of_mass(m,lab,keep))
    for k,(cy,cx) in zip(keep,c):
        offs+=1; L[lab==k]=offs; cents.append((cy,cx)); sgn.append(sv)
cents=np.array(cents); sgn=np.array(sgn)
t=cKDTree(cents); pr=np.array(list(t.query_pairs(28)))
d=np.abs(cents[pr[:,0]]-cents[pr[:,1]])
ok=(sgn[pr[:,0]]!=sgn[pr[:,1]])&(((d[:,1]>=11)&(d[:,1]<=27)&(d[:,0]<=7))|((d[:,0]>=11)&(d[:,0]<=27)&(d[:,1]<=7)))
pr=pr[ok]; n=len(cents)
g=coo_matrix((np.ones(len(pr)),(pr[:,0],pr[:,1])),shape=(n,n))
nc,cl=connected_components(g,directed=False); csz=np.bincount(cl)
good=csz[cl]>=3
keepmask=np.zeros(n+1,bool); keepmask[1:]=good
cells=keepmask[L]
np.save("altcells_tint.npy",cells)
print("cell blobs",n,"kept",good.sum(),"px",int(cells.sum()))
u8=a.astype(np.uint8)
for nm,(y0,y1,x0,x1) in {"rows":(600,800,1300,1700),"center":(1000,1250,1100,1600),"bottom":(1550,1850,600,1300),"top":(0,450,1000,1700)}.items():
    v=u8[y0:y1,x0:x1].copy(); cm=cells[y0:y1,x0:x1]
    v[cm]=(v[cm]*0.3+np.array([255,0,255])*0.7).astype(np.uint8)
    Image.fromarray(np.concatenate([u8[y0:y1,x0:x1],v],1)).save(f"altT_{nm}.png")
