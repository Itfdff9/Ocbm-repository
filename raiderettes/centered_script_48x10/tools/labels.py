"""Letter structure labels: 0=transparent, 1=orange border, 2=dark (gray band + sequin fill)."""
import sys; sys.path[:0]=["/home/claude/ru"]
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
def labels(a):
    A=a[...,3]>=128
    med=cv2.medianBlur(a[...,:3],5).astype(np.int32)
    R,G,B=med[...,0],med[...,1],med[...,2]
    warm=A&(R>110)&(R-B>55)&(R-G>35)
    L=np.full(A.shape,2,np.uint8); L[~A]=0; L[warm]=1
    # majority smoothing
    oh=np.stack([(L==k).astype(np.float32) for k in range(3)],-1)
    oh=cv2.blur(oh,(9,9))
    L=np.argmax(oh,-1).astype(np.uint8)
    return L
PAL=np.array([[255,255,255],[255,110,0],[20,20,20]],np.uint8)
if __name__=="__main__":
    n=sys.argv[1]
    a=np.array(Image.open(f"src/{n}.png"))
    L=labels(a); np.save(f"work/lab_{n}.npy",L)
    Image.fromarray(PAL[L]).resize((2048,int(2048*L.shape[0]/L.shape[1])),Image.NEAREST).save(f"/tmp/claude-0/rx/lab_{n}.png")
