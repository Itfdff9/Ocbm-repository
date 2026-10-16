import numpy as np, cv2, sys
from PIL import Image
from scipy import ndimage as nd
NAMES=["SHEA","RENDA","VALENTI","CORSO","TRAVA","MCHUGH","VELEZ","DILEO","MATTEO","PALENCIA"]
def disk(r): return cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
def wide_warm(a, rc=11):
    rgb=a[...,:3].astype(np.int32); A=a[...,3]>128
    R,G,B=rgb[...,0],rgb[...,1],rgb[...,2]
    warm=A&(R>120)&(R-B>60)&(R-G>40)
    white=A&(R>190)&(G>150)&(B>140)
    w=(warm|(white&nd.binary_dilation(warm,iterations=4))).astype(np.uint8)
    wc=cv2.morphologyEx(w,cv2.MORPH_CLOSE,disk(rc)).astype(bool)&A
    return wc
if __name__=="__main__":
    rc=int(sys.argv[1]); names=sys.argv[2:] or NAMES
    for n in names:
        a=np.array(Image.open(f"src/{n}.png"))
        wc=wide_warm(a,rc)
        d=nd.distance_transform_edt(wc)
        mx=nd.maximum_filter(d,size=9); ridge=(d==mx)&(d>4)
        h=np.bincount(d[ridge].astype(int),minlength=150)[:150]
        hs=np.convolve(h,np.ones(3),mode="same")
        print(n,"peaks",(np.argsort(hs[5:])[::-1][:8]+5).tolist(), " count r>=45:",int(h[45:].sum()),"r 15-30:",int(h[15:31].sum()))
        np.save(f"work/wc_{n}.npy",wc)
