import sys, numpy as np, cv2
sys.path.insert(0,'/home/claude/six/tools'); from erase import fill
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS=None
SRC='/home/claude/six/out/NSE_Bow3Names_BLANK_8K_14in.png'
CX,CY,RIN=2840.6,2230.3,458
H_FRAC=float(sys.argv[1]) if len(sys.argv)>1 else 0.80
im=np.array(Image.open(SRC)); rgb=np.ascontiguousarray(im[...,:3])
x0,y0=int(CX-560),int(CY-560); sub=rgb[y0:y0+1120,x0:x0+1120].copy()
yy,xx=np.mgrid[:1120,:1120]; r=np.hypot(xx-(CX-x0),yy-(CY-y0))
disc=r<RIN
dark=(sub.astype(int).mean(2)<200)&(r<RIN+4)
lab,n=nd.label(dark); sz=np.bincount(lab.ravel()); big=sz>300; big[0]=False
dancer=big[lab]
bx=np.zeros_like(dancer); bx[1008:1060,484:548]=True
dancer|=bx&(sub.astype(int).mean(2)<200)
m=nd.binary_dilation(dancer,iterations=10)&((r<RIN+6)|bx)
clean=disc&~nd.binary_dilation(dancer,iterations=22)&(r<RIN-14)
sub2=fill(sub,m,clean,P=40)
# silhouette
sil0=np.array(Image.open('cheer_mask.png'))
BOLD=int(sys.argv[2]) if len(sys.argv)>2 else 0
sil0=cv2.dilate(sil0,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*BOLD+1,2*BOLD+1))) if BOLD else sil0
sil0=np.pad(sil0,BOLD+2)
sil=sil0.astype(np.float32)/255
th=int(2*RIN*H_FRAC); tw=int(sil.shape[1]*th/sil.shape[0])
s=cv2.resize(sil,(tw,th),interpolation=cv2.INTER_AREA)
oy=int((CY-y0)-th/2+RIN*0.02); ox=int((CX-x0)-tw/2)
a=np.zeros((1120,1120),np.float32); a[oy:oy+th,ox:ox+tw]=s
a*=disc
ink=np.array([10,10,14],np.float32)
out=sub2.astype(np.float32)*(1-a[...,None])+ink*a[...,None]
rgb[y0:y0+1120,x0:x0+1120]=np.clip(out,0,255).astype(np.uint8)
im[...,:3]=rgb
Image.fromarray(im).save('Bow3_BLANK_cheer_8K.png',dpi=(585.142857,585.142857))
Image.fromarray(rgb[y0:y0+1120,x0:x0+1120]).save('circle_after.png')
Image.fromarray(sub).save('circle_before.png')
