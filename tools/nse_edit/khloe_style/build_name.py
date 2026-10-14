import sys, numpy as np, cv2
sys.path.insert(0,'/home/claude/ocbm-repository/tools/nse_edit')
import make_name_v2 as v, make_name as mn
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS=None
NAME=sys.argv[1] if len(sys.argv)>1 else 'Laura'
K=4
x4=np.array(Image.open('/home/claude/sc/ref_x4.png').convert('RGB'))
H,W=x4.shape[:2]
a1=np.array(Image.open('/home/claude/sc/ref_alpha.png'))
alpha=cv2.resize(a1,(W,H),interpolation=cv2.INTER_LINEAR)>=128
blob1=np.load('/home/claude/sc/khloe_blob.npy')
a=np.array(Image.open('/root/.claude/uploads/2b454143-e1ad-5b30-a586-acd58e182021/1b5f39af-image.png').convert('RGB')).astype(int)
dark=(a.max(2)<90)
extra=nd.binary_dilation(blob1,iterations=4)&dark&~np.load('/home/claude/sc/ref_bg.npy')
blob1=blob1|extra
grow_=nd.binary_dilation(blob1,iterations=5)&~np.load('/home/claude/sc/ref_bg.npy')
keep=nd.binary_dilation(np.load('/home/claude/sc/banner_raw.npy'),iterations=8)
keep[1110:1240,290:352]=True; keep[1075:1250,838:930]=True      # the three sparkle stars
blob1=blob1|(grow_&~keep)
blob=cv2.resize(blob1.astype(np.uint8)*255,(W,H),interpolation=cv2.INTER_LINEAR)>=128
ys,xs=np.where(blob); bx0,bx1,by0,by1=xs.min(),xs.max(),ys.min(),ys.max()
# name: same footprint height as Khloe's block, centred on it
t=v.render_name(NAME,int(H*0.26)); t=t.crop(t.getbbox())
k=min((by1-by0)*1.0/t.height, 505*K*0.98/t.width)
t=t.resize((int(t.width*k),int(t.height*k)),Image.LANCZOS)
tx=int((bx0+bx1)/2-t.width/2); ty=int(by1-t.height)
ta=np.zeros((H,W),bool); ta[ty:ty+t.height,tx:tx+t.width]=np.array(t)[...,3]>=128
NONAME=NAME=='-'
if NONAME: ta[:]=False
# remove Khloe; fill uncovered stripe area above the drip line by extending stripes downward
resid=blob&~ta
if NONAME:
    import os as _o
    _fp='/home/claude/sc/name_footprint_%s.npy'%_o.environ.get('OLDNAME','laura').lower()
    resid=resid|nd.binary_dilation(np.load(_fp),iterations=10)
    resid[:by0,:]=False
np.save('/home/claude/sc/resid4.npy',np.packbits(resid)); np.save('/home/claude/sc/ta4.npy',np.packbits(ta))
img=x4.copy(); al=alpha&~blob
# fill uncovered Khloe area with the stripes + drips from under the banner of the master art (same palette/texture)
M=np.array(Image.open('/home/claude/ocbm-repository/north_shore_elite/reference/NSE_Sagamore_master_art_5016px_source.png').convert('RGBA'))
yy,xx=np.where(resid)
sy=yy+115*K; sx=380*K+((xx-450*K)%(240*K))
if NONAME:   # no name: stretch the drips down so the middle reaches the same depth as the sides
    END=int(__import__('os').environ.get('DRIPEND','4820'))
    k_=(4740-(by0+115*K))/(END-by0)
    sy=(by0+115*K+(yy-by0)*k_).astype(int)
sy=np.clip(sy,0,M.shape[0]-1)
img[yy,xx]=M[sy,sx,:3]; al[yy,xx]=M[sy,sx,3]>=128
fill=resid
out=Image.fromarray(np.dstack([img,np.where(al,255,0).astype(np.uint8)]),'RGBA')
bgname=out.copy()
out.alpha_composite(t,(tx,ty))
# put the name on the CURRENT master (megaphone/banner edits kept): swap only the old name area
import os
cur=np.array(Image.open(sys.argv[2]).convert('RGBA'))
OLD=os.environ.get('OLDNAME','laura').lower(); fp='/home/claude/sc/name_footprint_%s.npy'%OLD
oldname=np.load(fp) if os.path.exists(fp) else None
bgA=np.array(bgname)
zone=nd.binary_dilation(oldname,iterations=6) if oldname is not None else np.zeros((H,W),bool)
if NONAME: zone=zone|resid
cur[zone]=bgA[zone]
res=Image.fromarray(cur,'RGBA')
if not NONAME: res.alpha_composite(t,(tx,ty))
r=np.array(res); s_=r[...,3]>=128; r[...,3]=np.where(s_,255,0); r[~s_,:3]=0
Image.fromarray(r,'RGBA').save(sys.argv[3],dpi=(300,300))
if not NONAME: np.save('/home/claude/sc/name_footprint_%s.npy'%NAME.lower(),ta)
print('blob bbox',bx0,bx1,by0,by1,'name',t.size, 'resid px',resid.sum(), 'filled',fill.sum())
