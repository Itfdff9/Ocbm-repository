"""North Shore Elite camo hoodie mockups: NSE logo 3.5in left chest (drawstrings drawn over it), Blue Black White on the back."""
import json, numpy as np, cv2
from PIL import Image
D="/home/claude/nse/"
P=json.load(open(D+"place.json")); B=json.load(open(D+"backs.json"))
# small nudges so the logo clears a cord that only grazes it (Pink, Desert); cords that cross it are drawn over it instead
NUDGE={"Pink_Front":6,"Desert_Front":8}
CORDS={  # wearer's-left drawstring paths that cross the logo (x,y on the 2000 px photo), with left/right half-widths
 "Black_Front":([(1088,553),(1088,600),(1089,650),(1090,700),(1091,745)],12,7),
 "Navy_Front": ([(1096,575),(1093,600),(1089,650),(1084,700),(1080,750)],10,6),
 "Blue_Front": ([(1083,512),(1077,540),(1071,580),(1065,620),(1060,660),(1056,700)],7,7),
}
BACK_W_IN=12.0
def harden(rgba): a=rgba.copy(); a[...,3]=np.where(a[...,3]>=128,255,0); return a
LOGO=harden(np.asarray(Image.open(D+"nse_logo_src.webp").convert("RGBA")))
BBW=np.asarray(Image.open(D+"bbw_cutout.png").convert("RGBA"))
def fit(art,w):
    h=int(round(art.shape[0]*w/art.shape[1])); im=Image.fromarray(art,"RGBA").resize((int(round(w)),h),Image.LANCZOS)
    a=np.asarray(im).copy(); a[...,3]=np.where(a[...,3]>=128,255,0); return a
def composite(photo, art, cx, cy):
    out=photo.astype(np.float32).copy(); h,w=art.shape[:2]; x0=int(round(cx-w/2)); y0=int(round(cy-h/2))
    reg=out[y0:y0+h,x0:x0+w]; a=art[...,3:4].astype(np.float32)/255*0.97
    L=cv2.cvtColor(photo,cv2.COLOR_RGB2GRAY).astype(np.float32)
    grain=(L-cv2.GaussianBlur(L,(0,0),1.2))[y0:y0+h,x0:x0+w,None]*0.30      # knit texture through the ink
    ink=np.clip(art[...,:3].astype(np.float32)+grain,0,255)
    out[y0:y0+h,x0:x0+w]=ink*a+reg*(1-a)
    return out.clip(0,255).astype(np.uint8)
def cord_mask(shape, pts, lh, rh):
    m=np.zeros(shape[:2],np.uint8)
    for (x1,y1),(x2,y2) in zip(pts[:-1],pts[1:]):
        for t in np.linspace(0,1,max(2,int(abs(y2-y1)))):
            x=x1+(x2-x1)*t; y=int(round(y1+(y2-y1)*t)); m[y,int(round(x-lh)):int(round(x+rh))+1]=1
    return cv2.GaussianBlur(m.astype(np.float32),(0,0),0.8)
def front(k):
    p=P[k]; ph=np.asarray(Image.open(D+f"crops/{k}.png").convert("RGB"))
    art=fit(LOGO,p["logo_px"]); out=composite(ph,art,p["cx"]+NUDGE.get(k,0),p["cy"])
    if k in CORDS:                                                            # drawstring stays on top of the print
        pts,lh,rh=CORDS[k]; m=cord_mask(ph.shape,pts,lh,rh)[...,None]
        out=(ph*m+out*(1-m)).astype(np.uint8)
    return out
def back(k):
    b=B[k]; ph=np.asarray(Image.open(D+f"crops/{k}.png").convert("RGB")); ppi=b["body_w"]/23.0
    art=fit(BBW,BACK_W_IN*ppi); top=b["hood_y"]+1.0*ppi                      # 1 in clear of the hood point
    return composite(ph,art,b["cx"],top+art.shape[0]/2), dict(width_px=art.shape[1],top=round(top),bottom=round(top+art.shape[0]))
if __name__=="__main__":
    import os; os.makedirs(D+"out",exist_ok=True)
    for c in ["Woodland","Desert","Winter","Pink","Red","Blue","Black","Navy"]:
        Image.fromarray(front(f"{c}_Front")).save(D+f"out/{c}-Camo_Front_NorthShoreElite.jpg",quality=92)
        im,meta=back(f"{c}_Back"); Image.fromarray(im).save(D+f"out/{c}-Camo_Back_BlueBlackWhite.jpg",quality=92); print(c,meta)
