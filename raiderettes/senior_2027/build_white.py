"""Pat-Med Varsity Raiderettes SENIOR / COACH — WHITE-SHIRT colorway (from Brian's 2026-10-14 references).
PAT-MED VARSITY, dancer and year: black fill + red keyline. RAIDERETTES and SENIOR/COACH: red core + white line + black rim.
usage: python3 build_white.py OUT.png CANVAS_PX DPI [SENIOR|COACH] [YEAR]"""
import sys, math, pickle, numpy as np, cv2, cairocffi as cairo
from PIL import Image
sys.path.insert(0, "/home/claude/ocbm-repository/raiderettes/senior_2027")
import build_senior as B
from build_senior import glyph_mask, grow, paste, flat, OSW, GRAD, ARC, RAID, SEN, YEAR_BOX, SRC_BOX
BLACK=(10,10,10); RED=(214,22,30); WHITE=(255,255,255)
KEY_IN=0.03          # red keyline width in inches (printable on DTF)

def keyed(mask, key_px):
    """black fill + red keyline around a uint8 mask"""
    pad=key_px+4; m=np.pad(mask,pad)
    outer=grow(m,key_px)
    rgba=np.zeros(m.shape+(4,),np.uint8)
    rgba[outer>127]=(*RED,255); rgba[m>127]=(*BLACK,255)
    rgba[rgba[...,3]==0,:3]=RED
    return rgba

def varsity(ch,H,fit_w):
    """red core + white line + black rim (same geometry as the black-shirt build)"""
    rim=max(1,round(0.03*H)); wl=max(1,round(B.WL*H)); bold=max(1,round(B.BOLD*H))
    core_h=H-2*(rim+wl+bold); cap=glyph_mask("H",GRAD,1000).shape[0]/1000
    m=np.pad(glyph_mask(ch,GRAD,int(core_h/cap)),rim+wl+bold+4)
    core=grow((m>127).astype(np.uint8)*255,bold,square=True); white=grow(core,wl); outer=grow(white,rim)
    rgba=np.zeros(core.shape+(4,),np.uint8)
    rgba[outer>127]=(*BLACK,255); rgba[white>127]=(*WHITE,255); rgba[core>127]=(*RED,255)
    ys,xs=np.where(outer>127); rgba=rgba[ys.min():ys.max()+1, xs.min():xs.max()+1]; rgba[rgba[...,3]==0,:3]=BLACK
    if fit_w:
        k=max(B.KMIN,min(1.25,fit_w/rgba.shape[1]))
        rgba=cv2.resize(rgba,(max(1,int(rgba.shape[1]*k)),rgba.shape[0]),interpolation=cv2.INTER_AREA if k<1 else cv2.INTER_LINEAR)
    return rgba

def coach_row():
    """COACH laid out with SENIOR's letter height, gaps and centre line"""
    xs=[x for _,x,_,_,_ in SEN]; ws=[w for _,_,_,w,_ in SEN]
    gap=np.mean([xs[i+1]-(xs[i]+ws[i]) for i in range(len(xs)-1)])
    sen_wh={c:w for c,_,_,w,_ in SEN}
    est={"C":90,"O":95,"A":119,"H":121}
    word="COACH"; wsum=sum(est[c] for c in word)+gap*(len(word)-1)
    cx=(xs[0]+xs[-1]+ws[-1])/2; x=cx-wsum/2; y=SEN[0][2]; out=[]
    for c in word: out.append((c,x,y,est[c],141)); x+=est[c]+gap
    return out

def render(canvas_px, dpi, word="SENIOR", year="2027"):
    margin=int(0.25*dpi); x0,y0,x1,y1=SRC_BOX
    if word!="SENIOR": y1=SEN[0][2]+141+12          # no year line: design ends under COACH
    S=(canvas_px-2*margin)/(y1-y0)
    if (x1-x0)*S>canvas_px-2*margin: S=(canvas_px-2*margin)/(x1-x0)
    ox=canvas_px/2-S*(x0+x1)/2; oy=canvas_px/2-S*(y0+y1)/2
    T=lambda x,y:(ox+S*x, oy+S*y)
    key=max(2,int(round(KEY_IN*dpi*canvas_px/(canvas_px))))
    key=max(2,int(round(KEY_IN*canvas_px/16)))      # canvas is 16 in wide
    cv=np.zeros((canvas_px,canvas_px,4),np.uint8)
    # arched PAT-MED VARSITY
    P=np.array([(x+w/2,y+h/2) for _,x,y,w,h in ARC],float)
    A=np.c_[2*P[:,0],2*P[:,1],np.ones(len(P))]; b=(P**2).sum(1); xc,yc,c=np.linalg.lstsq(A,b,rcond=None)[0]
    capf=glyph_mask("H",OSW,1000).shape[0]/1000; cap_px=65*S
    for ch,x,y,w,h in ARC:
        cx,cy=x+w/2,y+h/2; ang=math.degrees(math.atan2(cx-xc, yc-cy))
        m=np.full((int(10*S),int(22*S)),255,np.uint8) if ch=="-" else glyph_mask(ch,OSW,int(cap_px/capf))
        r=keyed(m,key); im=Image.fromarray(r,"RGBA").rotate(-ang,resample=Image.BICUBIC,expand=True)
        paste(cv,np.asarray(im).copy(),*T(cx,cy))
    rows=[(RAID,109),((SEN if word=="SENIOR" else coach_row()),141)]
    for row,Hs in rows:
        for ch,x,y,w,h in row:
            paste(cv,varsity(ch,Hs*S,w*S if word=="SENIOR" or row is RAID else None),*T(x+w/2,y+h/2))
    # dancer: black fill + red keyline
    curves=pickle.load(open("/home/claude/ocbm-repository/raiderettes/senior_2027/dancer_path.pkl","rb"))
    surf=cairo.ImageSurface(cairo.FORMAT_A8,canvas_px,canvas_px); ctx=cairo.Context(surf); ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    for st,segs in curves:
        ctx.move_to(*T(*st))
        for sg in segs:
            if sg[0]=="L": ctx.line_to(*T(*sg[1])); ctx.line_to(*T(*sg[2]))
            else: ctx.curve_to(*T(*sg[1]),*T(*sg[2]),*T(*sg[3]))
        ctx.close_path()
    ctx.fill(); surf.flush()
    a=np.frombuffer(surf.get_data(),np.uint8).reshape(canvas_px,surf.get_stride())[:, :canvas_px].copy()
    outer=grow(a,key); d=np.zeros((canvas_px,canvas_px,4),np.uint8)
    d[outer>127]=(*RED,255); d[a>127]=(*BLACK,255)
    paste(cv,d,canvas_px/2,canvas_px/2)
    if word=="SENIOR" and year:
        capd=glyph_mask("2",OSW,1000).shape[0]/1000
        for ch,(x,y,w,h) in zip(year,YEAR_BOX):
            paste(cv,keyed(glyph_mask(ch,OSW,int(107*S/capd)),key),*T(x+w/2,y+h/2))
    cv[...,3]=np.where(cv[...,3]>=128,255,0).astype(np.uint8); cv[cv[...,3]==0,:3]=0
    return Image.fromarray(cv,"RGBA")

if __name__=="__main__":
    out,px,dpi=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
    word=sys.argv[4] if len(sys.argv)>4 else "SENIOR"; yr=sys.argv[5] if len(sys.argv)>5 else "2027"
    im=render(px,dpi,word,yr); im.save(out,dpi=(dpi,dpi)); print(out,im.size)
