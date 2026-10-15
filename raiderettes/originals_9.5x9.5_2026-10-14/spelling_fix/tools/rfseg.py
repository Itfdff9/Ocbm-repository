import numpy as np, sys, time
from PIL import Image
from scipy import ndimage as nd
from sklearn.ensemble import RandomForestClassifier
name=sys.argv[1]; ytop_letters=int(sys.argv[2]); ybot_script=int(sys.argv[3]); xr=tuple(map(int,sys.argv[4].split(',')))
im=Image.open(f'../names48/src_{name}.png').convert('RGBA')
W,H=im.size; S=2
sm=im.resize((W//S,H//S),Image.LANCZOS)
a=np.asarray(sm).astype(np.float32)
rgb=a[...,:3]/255; al=a[...,3]/255
r,g,b=rgb[...,0],rgb[...,1],rgb[...,2]
lum=rgb.mean(-1); mx=rgb.max(-1); mn=rgb.min(-1)
feats=[r,g,b,al,mx-mn,r-g,g-b,lum]
orange=((r>0.65)&(g>0.2*r)&(g<0.65*r)).astype(np.float32)
deep=((r>0.35)&(g<0.16*r)).astype(np.float32)
black=(mx<0.18).astype(np.float32)
white=(mn>0.78).astype(np.float32)
gray=((mx-mn<0.12)&(mx>=0.18)&(mn<=0.78)).astype(np.float32)
for s in (2,5,10,20):
    for ch in (r,g,b,lum,al):
        feats.append(nd.gaussian_filter(ch,s))
    m=nd.gaussian_filter(lum,s); feats.append(np.sqrt(np.maximum(nd.gaussian_filter(lum**2,s)-m**2,0)))
    for cl in (orange,deep,black,white,gray):
        feats.append(nd.gaussian_filter(cl,s))
gm=np.hypot(nd.sobel(lum,0),nd.sobel(lum,1)); feats+= [nd.gaussian_filter(gm,1),nd.gaussian_filter(gm,4)]
F=np.stack(feats,-1).reshape(-1,len(feats))
yy,xx=np.mgrid[0:H//S,0:W//S]
op=al>0.5
lab=np.full(op.shape,-1,np.int8)
lab[op&(yy*S<ytop_letters)]=0
lab[op&(yy*S>ybot_script)&(xx*S>xr[0])&(xx*S<xr[1])]=1
idx0=np.flatnonzero(lab.ravel()==0); idx1=np.flatnonzero(lab.ravel()==1)
rng=np.random.default_rng(0)
idx0=rng.choice(idx0,min(60000,len(idx0)),replace=False); idx1=rng.choice(idx1,min(60000,len(idx1)),replace=False)
X=np.concatenate([F[idx0],F[idx1]]); Y=np.concatenate([np.zeros(len(idx0)),np.ones(len(idx1))])
t=time.time(); clf=RandomForestClassifier(n_estimators=80,min_samples_leaf=5,n_jobs=-1,random_state=0).fit(X,Y); print('fit',round(time.time()-t,1),'s', len(idx0),len(idx1))
P=np.zeros(op.size,np.float32); m=op.ravel()
P[m]=clf.predict_proba(F[m])[:,1]
P=P.reshape(op.shape)
np.save(f'P_{name}.npy',P)
vis=(np.asarray(sm.convert('RGB')).astype(np.float32)*0.5)
vis[...,0]+=P*120; vis[...,1]+= (1-P)*op*0; vis[...,2]+=(1-P)*op*120
vis[~op]=(0,80,50)
Image.fromarray(np.clip(vis,0,255).astype(np.uint8)).save(f'P_{name}.png')
print('done')
