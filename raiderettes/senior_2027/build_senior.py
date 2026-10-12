"""Pat-Med Varsity Raiderettes SENIOR 2027 — rebuilt as clean artwork (fonts + traced dancer), transparent, hard alpha.
usage: python3 build_senior.py OUT.png CANVAS_PX DPI [YEAR] [--stars]   (stars off by default — Brian wants them removed)"""
import sys, math, pickle, numpy as np, cv2, cairocffi as cairo
from PIL import Image, ImageDraw, ImageFont
OSW="/home/claude/ocbm-repository/raiderettes/senior_2027/oswald-latin-700-normal.ttf"
GRAD="/home/claude/ocbm-repository/raiderettes/shoulder_40x6/fonts/Graduate-Regular.ttf"
WHITE=(255,255,255); RED=(214,22,30)
RIM,WL,BOLD,KMIN=0.024,0.03,0.038,0.7
SRC_BOX=(73,720,1020,1870)                       # design bounds in the photo (px)
# ---- measurements from the photo (bbox x,y,w,h) ----
ARC=[("P",161,775,37,66),("A",216,758,42,69),("T",264,744,35,68),("-",322,771,22,12),("M",359,727,54,68),("E",428,722,32,67),("D",477,720,34,65),
     ("V",580,719,36,67),("A",630,721,34,68),("R",684,726,37,68),("S",737,736,37,66),("I",792,746,26,66),("T",839,754,38,69),("Y",887,770,43,70)]
RAID=[('R', 72, 861, 83, 109), ('A', 164, 861, 90, 108), ('I', 264, 861, 42, 109), ('D', 323, 861, 78, 109), ('E', 420, 861, 72, 109), ('R', 509, 861, 80, 108), ('E', 604, 862, 74, 108), ('T', 689, 862, 78, 108), ('T', 772, 861, 78, 108), ('E', 861, 861, 73, 108), ('S', 951, 860, 70, 110)]   # full extents incl. outer rim, minus JPEG blur
SEN=[('S', 211, 1569, 91, 140), ('E', 328, 1568, 92, 142), ('N', 448, 1569, 121, 141), ('I', 595, 1569, 52, 141), ('O', 674, 1569, 95, 140), ('R', 797, 1569, 105, 140)]
YEAR_BOX=[(428,1762,55,108),(495,1763,54,108),(563,1763,55,107),(627,1764,54,106)]
STARS=[(323,1788,57,52),(732,1789,56,51)]

def glyph_mask(ch, font_path, size):
    f=ImageFont.truetype(font_path,size); l,t,r,b=f.getbbox(ch); pad=size//6
    m=Image.new("L",(r-l+2*pad,b-t+2*pad),0); ImageDraw.Draw(m).text((pad-l,pad-t),ch,font=f,fill=255)
    a=np.asarray(m); ys,xs=np.where(a>127); return a[ys.min():ys.max()+1, xs.min():xs.max()+1]

def grow(a,px,square=False):
    if px<=0: return a
    k=np.ones((2*px+1,2*px+1),np.uint8) if square else cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*px+1,2*px+1))
    return cv2.dilate(a,k)

def paste(canvas, rgba, cx, cy):
    h,w=rgba.shape[:2]; x0=int(round(cx-w/2)); y0=int(round(cy-h/2))
    H,W=canvas.shape[:2]; xs0,ys0=max(0,x0),max(0,y0); xs1,ys1=min(W,x0+w),min(H,y0+h)
    src=rgba[ys0-y0:ys1-y0, xs0-x0:xs1-x0]; dst=canvas[ys0:ys1, xs0:xs1]
    # proper "over" compositing (straight alpha) so edges keep their own colour instead of blending toward black
    sa=src[...,3:4].astype(np.float32)/255; da=dst[...,3:4].astype(np.float32)/255
    oa=sa+da*(1-sa); safe=np.where(oa>0,oa,1)
    oc=(src[...,:3]*sa+dst[...,:3]*da*(1-sa))/safe
    dst[...,:3]=np.clip(oc+0.5,0,255).astype(np.uint8); dst[...,3]=np.clip(oa[...,0]*255+0.5,0,255).astype(np.uint8)

def flat(mask,color):
    out=np.zeros(mask.shape+(4,),np.uint8); out[...,:3]=color; out[...,3]=mask; return out

def varsity_letter(ch, H, S, fit_w=None):
    """red core (Graduate, slightly bolded) + white line + thin red rim; total height H (out px)"""
    rim=max(1,round(RIM*H)); wl=max(1,round(WL*H)); bold=max(1,round(BOLD*H))
    core_h=H-2*(rim+wl+bold)
    cap=glyph_mask("H",GRAD,1000).shape[0]/1000
    m=glyph_mask(ch,GRAD,int(core_h/cap))
    pad=rim+wl+bold+4; m=np.pad(m,pad)
    core=grow((m>127).astype(np.uint8)*255,bold,square=True)
    white=grow(core,wl); outer=grow(white,rim)
    rgba=np.zeros(core.shape+(4,),np.uint8)
    rgba[outer>127]=(*RED,255); rgba[white>127]=(*WHITE,255); rgba[core>127]=(*RED,255)
    ys,xs=np.where(outer>127); rgba=rgba[ys.min():ys.max()+1, xs.min():xs.max()+1]
    rgba[rgba[...,3]==0,:3]=RED                  # transparent pixels carry the rim colour so resizing can't pull in black
    if fit_w:                                    # match the photo's letter width (within 12%)
        k=max(KMIN,min(1.25,fit_w/rgba.shape[1]))
        rgba=cv2.resize(rgba,(max(1,int(rgba.shape[1]*k)),rgba.shape[0]),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_LINEAR)   # no cubic overshoot
    return rgba

def star(R):
    pts=[]
    for i in range(10):
        r=R if i%2==0 else R*0.382; a=math.radians(-90+36*i); pts.append((r*math.cos(a),r*math.sin(a)))
    return pts

def render(canvas_px, margin_px, year="2027", stars=False):
    x0,y0,x1,y1=SRC_BOX; S=(canvas_px-2*margin_px)/(y1-y0)
    ox=canvas_px/2-S*(x0+x1)/2; oy=canvas_px/2-S*(y0+y1)/2
    T=lambda x,y:(ox+S*x, oy+S*y)
    cv=np.zeros((canvas_px,canvas_px,4),np.uint8)
    # 1) arched PAT-MED VARSITY: fit a circle to the letter centres, rotate each letter to the tangent
    P=np.array([(x+w/2,y+h/2) for _,x,y,w,h in ARC],float)
    A=np.c_[2*P[:,0],2*P[:,1],np.ones(len(P))]; b=(P**2).sum(1); xc,yc,c=np.linalg.lstsq(A,b,rcond=None)[0]; R=math.sqrt(c+xc*xc+yc*yc)
    capf=glyph_mask("H",OSW,1000).shape[0]/1000; cap_px=65*S
    for ch,x,y,w,h in ARC:
        cx,cy=x+w/2,y+h/2
        ang=math.degrees(math.atan2(cx-xc, yc-cy))
        m=glyph_mask(ch,OSW,int(cap_px/capf)) if ch!="-" else None
        if ch=="-":
            hw,hh=int(22*S),int(10*S); m=np.full((hh,hw),255,np.uint8)
        m=np.pad(m,20); im=Image.fromarray(m).rotate(-ang,resample=Image.BICUBIC,expand=True)
        paste(cv,flat(np.asarray(im),WHITE),*T(cx,cy))
    # 2) RAIDERETTES and SENIOR in varsity letters
    for row,Hs in ((RAID,109),(SEN,141)):
        for ch,x,y,w,h in row:
            paste(cv,varsity_letter(ch,Hs*S,S,fit_w=w*S),*T(x+w/2,y+h/2))
    # 3) dancer (traced curves) via cairo
    curves=pickle.load(open("/home/claude/ocbm-repository/raiderettes/senior_2027/dancer_path.pkl","rb"))
    surf=cairo.ImageSurface(cairo.FORMAT_A8,canvas_px,canvas_px); ctx=cairo.Context(surf); ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    for st,segs in curves:
        ctx.move_to(*T(*st))
        for sg in segs:
            if sg[0]=="L": ctx.line_to(*T(*sg[1])); ctx.line_to(*T(*sg[2]))
            else: ctx.curve_to(*T(*sg[1]),*T(*sg[2]),*T(*sg[3]))
        ctx.close_path()
    ctx.fill(); surf.flush()
    a=np.frombuffer(surf.get_data(),np.uint8).reshape(canvas_px,surf.get_stride())[:, :canvas_px]
    paste(cv,flat(a.copy(),WHITE),canvas_px/2,canvas_px/2)
    # 4) year + stars
    capd=glyph_mask("2",OSW,1000).shape[0]/1000
    for ch,(x,y,w,h) in zip(year,YEAR_BOX):
        paste(cv,flat(glyph_mask(ch,OSW,int(107*S/capd)),WHITE),*T(x+w/2,y+h/2))
    for x,y,w,h in (STARS if stars else []):
        Rr=w*S/(2*math.sin(math.radians(72))); cx,cy=T(x+w/2,y+h/2); cy+=0.0955*Rr
        surf=cairo.ImageSurface(cairo.FORMAT_A8,canvas_px,canvas_px); ctx=cairo.Context(surf)
        pts=star(Rr); ctx.move_to(cx+pts[0][0],cy+pts[0][1])
        for px_,py_ in pts[1:]: ctx.line_to(cx+px_,cy+py_)
        ctx.close_path(); ctx.fill(); surf.flush()
        a=np.frombuffer(surf.get_data(),np.uint8).reshape(canvas_px,surf.get_stride())[:, :canvas_px]
        paste(cv,flat(a.copy(),RED),canvas_px/2,canvas_px/2)
    # hard alpha
    cv[...,3]=np.where(cv[...,3]>=128,255,0).astype(np.uint8); cv[cv[...,3]==0,:3]=0
    return Image.fromarray(cv,"RGBA")

if __name__=="__main__":
    args=[a for a in sys.argv[1:] if not a.startswith("--")]
    out=args[0]; px=int(args[1]); dpi=int(args[2]); yr=args[3] if len(args)>3 else "2027"
    im=render(px, int(0.25*dpi), yr, stars="--stars" in sys.argv); im.save(out,dpi=(dpi,dpi)); print(out, im.size)
