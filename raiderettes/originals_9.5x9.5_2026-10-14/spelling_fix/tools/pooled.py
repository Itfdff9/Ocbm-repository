import numpy as np, time
from PIL import Image
from scipy import ndimage as nd
from sklearn.ensemble import RandomForestClassifier
def feats(name):
    im=Image.open(f'../names48/src_{name}.png').convert('RGBA'); W,H=im.size
    sm=im.resize((W//2,H//2),Image.LANCZOS); a=np.asarray(sm).astype(np.float32)
    rgb=a[...,:3]/255; al=a[...,3]/255
    r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
    lum=rgb.mean(-1); mx=rgb.max(-1); mn=rgb.min(-1)
    F=[r,g,b,al,mx-mn,r-g,g-b,lum]
    cl=[((r>0.65)&(g>0.2*r)&(g<0.65*r)),((r>0.35)&(g<0.16*r)),(mx<0.18),(mn>0.78),((mx-mn<0.12)&(mx>=0.18)&(mn<=0.78))]
    cl=[c.astype(np.float32) for c in cl]
    for s in (2,5,10,20):
        for ch in (r,g,b,lum,al): F.append(nd.gaussian_filter(ch,s))
        m=nd.gaussian_filter(lum,s); F.append(np.sqrt(np.maximum(nd.gaussian_filter(lum**2,s)-m**2,0)))
        for c in cl: F.append(nd.gaussian_filter(c,s))
    gm=np.hypot(nd.sobel(lum,0),nd.sobel(lum,1)); F+=[nd.gaussian_filter(gm,1),nd.gaussian_filter(gm,4)]
    return np.stack(F,-1).astype(np.float32), al>0.5, sm
X=[];Y=[]
rng=np.random.default_rng(0)
for n in ['TRAVA','SHEA','RENDA','VELEZ']:
    F,op,_=feats(n); P=np.load(f'P_{n}.npy')
    P=P[:op.shape[0],:op.shape[1]]
    i0=np.flatnonzero((op&(P<0.08)).ravel()); i1=np.flatnonzero((op&(P>0.92)).ravel())
    i0=rng.choice(i0,min(40000,len(i0)),replace=False); i1=rng.choice(i1,min(40000,len(i1)),replace=False)
    Ff=F.reshape(-1,F.shape[-1]); X+= [Ff[i0],Ff[i1]]; Y+=[np.zeros(len(i0)),np.ones(len(i1))]
t=time.time(); clf=RandomForestClassifier(n_estimators=80,min_samples_leaf=6,n_jobs=-1,random_state=0).fit(np.concatenate(X),np.concatenate(Y)); print('fit',round(time.time()-t,1))
for n in ['DILEO','CORSO','MATTEO','MCHUGH','PALENCIA','VALENTI']:
    F,op,sm=feats(n); Ff=F.reshape(-1,F.shape[-1]); m=op.ravel()
    P=np.zeros(m.size,np.float32); P[m]=clf.predict_proba(Ff[m])[:,1]; P=P.reshape(op.shape)
    np.save(f'P_{n}.npy',P)
    vis=np.asarray(sm.convert('RGB')).astype(np.float32)*0.5; vis[...,0]+=P*120; vis[...,2]+=(1-P)*op*120; vis[~op]=(0,80,50)
    Image.fromarray(np.clip(vis,0,255).astype(np.uint8)).resize((800,round(op.shape[0]*800/op.shape[1]))).save(f'work/Pv_{n}.jpg',quality=80)
    print(n,'script frac',round((P>0.5).sum()/op.sum(),3))
