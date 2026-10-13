import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage as nd
BLUE=(18,52,245); WHITE=(255,255,255); BLACK=(0,0,0)
def glyph_mask(text,font,size,pad,gaps=None):
    # gaps: {index: extra px before char index} to open up touching letters
    f=ImageFont.truetype(font,size); gaps=gaps or {}
    l,t,r,b=f.getbbox(text); extra=sum(gaps.values())
    W=r-l+2*pad+extra; H=b-t+2*pad
    m=Image.new('L',(W,H),0); d=ImageDraw.Draw(m)
    cuts=sorted(gaps)+[len(text)]; start=0; x=pad-l
    for c in cuts:
        seg=text[start:c]
        if seg:
            d.text((x,pad-t),seg,font=f,fill=255); x+=f.getlength(seg)
        if c in gaps: x+=gaps[c]
        start=c
    return np.array(m)
def dil(m,r):
    if r<=0: return m
    d=nd.distance_transform_edt(m<128)
    return np.clip((r+0.5-d)*255,0,255).astype(np.uint8)  # antialiased offset
def sparkle(size, core=1.0):
    s=size; y,x=np.mgrid[-s:s+1,-s:s+1].astype(np.float32)/s
    r=np.sqrt(x*x+y*y)+1e-6
    glow=np.exp(-(r/0.10)**2)*1.0+np.exp(-(r/0.35)**2)*0.35
    rays=(np.exp(-(np.abs(y)/0.012)**2)*np.clip(1-np.abs(x),0,1)**2.2 + np.exp(-(np.abs(x)/0.012)**2)*np.clip(1-np.abs(y),0,1)**2.2)
    diag=(np.exp(-(np.abs(x-y)/0.02)**2)+np.exp(-(np.abs(x+y)/0.02)**2))*np.clip(1-r*2.2,0,1)**2*0.5
    v=np.clip(glow+rays+diag,0,1)*core
    return v
def render(text,font,height,angle=-6,ss=1,gaps=None):
    # height = cap height-ish target in px for final glyph mask
    size=int(height*1.0)
    U=size/100.0
    pad=int(30*U)
    g=glyph_mask(text,font,size,pad,{k:int(v*U) for k,v in (gaps or {}).items()})
    g=cv2.GaussianBlur(g,(0,0),0.6*U/1.5)  # tiny smoothing
    blue=dil(g,4.5*U); white=dil(g,11*U); outer=dil(g,13.5*U)
    H,W=g.shape
    rgb=np.zeros((H,W,3),np.float32); a=outer.astype(np.float32)/255
    def lay(mask,col):
        m=(mask.astype(np.float32)/255)[...,None]; rgb[:]=rgb*(1-m)+np.array(col,np.float32)*m
    lay(outer,BLACK); lay(white,WHITE)
    lay(blue,BLUE)
    # glossy black fill: black with dark-blue rim light on upper-left inner edge and soft specular streak
    gf=g.astype(np.float32)/255
    din=nd.distance_transform_edt(g>127)
    sh=np.roll(np.roll(g>127,int(3.2*U),0),int(2.2*U),1)
    sh=np.roll(np.roll(g>127,int(1.6*U),0),int(1.1*U),1)
    rim=((g>127)&~sh).astype(np.float32)
    rim=cv2.GaussianBlur(rim,(0,0),0.2*U)
    fill=np.zeros((H,W,3),np.float32)+np.array([4,4,10],np.float32)
    fill+= rim[...,None]*np.array([22,40,115],np.float32)
    # inner highlight line along the stroke centre-left (glossy tube look)
    core=np.exp(-((din-2.6*U)/(0.25*U))**2)*(din>1.5*U)
    shift=np.roll(np.roll(core,-int(1.5*U),0),-int(1.5*U),1)*(g>127)
    fill+= cv2.GaussianBlur(shift.astype(np.float32),(0,0),0.2*U)[...,None]*np.array([45,55,95],np.float32)
    fill=np.clip(fill,0,255)
    m=gf[...,None]; rgb=rgb*(1-m)+fill*m
    out=np.dstack([np.clip(rgb,0,255),a*255]).astype(np.uint8)
    im=Image.fromarray(out,'RGBA')
    if angle: im=im.rotate(angle,resample=Image.BICUBIC,expand=True)
    return im
def add_sparkles(im,pts,size):
    arr=np.array(im).astype(np.float32)
    for (x,y,sc) in pts:
        s=int(size*sc); sp=sparkle(s)
        y0,x0=int(y)-s,int(x)-s
        sl=arr[max(0,y0):y0+2*s+1, max(0,x0):x0+2*s+1]
        spc=sp[max(0,-y0):max(0,-y0)+sl.shape[0], max(0,-x0):max(0,-x0)+sl.shape[1]]
        tint=np.array([230,238,255],np.float32)
        a=sl[...,3:4]/255
        k=spc[...,None]*(a>0.5)  # only on the artwork
        sl[...,:3]=sl[...,:3]*(1-k)+tint*k
    return Image.fromarray(np.clip(arr,0,255).astype(np.uint8),'RGBA')
