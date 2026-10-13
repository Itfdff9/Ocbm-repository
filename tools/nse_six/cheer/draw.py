# Original cheer-stunt silhouette (two-base extension, flyer in a high V with pom-poms). Drawn from scratch with shapes.
import numpy as np, cv2, math
from PIL import Image, ImageDraw
S=3; W,H=1000,1250; OY=170
img=Image.new('L',(W*S,H*S),0); d=ImageDraw.Draw(img)
def P(x,y): return (x*S,(y+OY)*S)
def circ(x,y,r): X,Y=P(x,y); d.ellipse([X-r*S,Y-r*S,X+r*S,Y+r*S],fill=255)
def ell(x,y,rx,ry,ang=0):
    t=np.linspace(0,2*np.pi,80); a=math.radians(ang)
    pts=[P(x+rx*math.cos(u)*math.cos(a)-ry*math.sin(u)*math.sin(a), y+rx*math.cos(u)*math.sin(a)+ry*math.sin(u)*math.cos(a)) for u in t]
    d.polygon(pts,fill=255)
def limb(a,b,r1,r2):
    (x1,y1),(x2,y2)=a,b; dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy); nx,ny=-dy/L,dx/L
    d.polygon([P(x1+nx*r1,y1+ny*r1),P(x2+nx*r2,y2+ny*r2),P(x2-nx*r2,y2-ny*r2),P(x1-nx*r1,y1-ny*r1)],fill=255)
    circ(x1,y1,r1); circ(x2,y2,r2)
def chain(pts,rs):
    for i in range(len(pts)-1): limb(pts[i],pts[i+1],rs[i],rs[i+1])
def poly(pts): d.polygon([P(*p) for p in pts],fill=255)
rng=np.random.default_rng(11)
def pompom(x,y,r,n=34):
    circ(x,y,r*0.62)
    for i in range(n):
        a=2*math.pi*i/n+rng.uniform(-0.08,0.08); L=r*rng.uniform(0.85,1.12); w=r*0.16
        tip=(x+L*math.cos(a),y+L*math.sin(a))
        limb((x+r*0.3*math.cos(a),y+r*0.3*math.sin(a)),tip,w,w*0.25)
def base(cx,flip,hold):
    s=-1 if flip else 1
    X=lambda v: cx+s*v
    # wide athletic stance, knees bent outward
    chain([(X(-24),820),(X(-78),920),(X(-92),1040)],[26,20,13])
    chain([(X(24),820),(X(78),920),(X(92),1040)],[26,20,13])
    ell(X(-104),1052,32,14); ell(X(104),1052,32,14)
    hem=[(X(-84+168*i/8), 868+(7 if i%2 else 0)) for i in range(9)]
    poly([(X(-40),775),(X(40),775)]+hem[::-1])
    # torso with waist
    poly([(X(-50),660),(X(50),660),(X(38),735),(X(36),760),(X(44),795),(X(-44),795),(X(-36),760),(X(-38),735)])
    circ(X(-46),672,20); circ(X(46),672,20)
    limb((cx,650),(cx,622),13,13); ell(cx,590,32,39)
    # high ponytail swinging outward
    chain([(X(-6),560),(X(-36),550),(X(-60),566),(X(-74),598),(X(-80),622)],[17,16,14,9,3])
    hy=hold[1]
    chain([(X(-46),672),(X(-64),578),(cx-16,hy+16)],[17,13,12])
    chain([(X(46),672),(X(64),578),(cx+16,hy+16)],[17,13,12])
    ell(cx,hy+12,28,17)
def flyer(cx):
    # feet standing in the bases' hands
    lf,rf=(cx-140,455),(cx+140,455)
    chain([lf,(cx-86,355),(cx-26,262)],[13,18,25]); chain([rf,(cx+86,355),(cx+26,262)],[13,18,25])
    ell(lf[0]-6,lf[1]+4,26,12,-8); ell(rf[0]+6,rf[1]+4,26,12,8)
    # skirt
    hem=[(cx-70+140*i/8, 290+(6 if i%2 else 0)) for i in range(9)]
    poly([(cx-34,205),(cx+34,205)]+hem[::-1])
    # torso
    poly([(cx-46,115),(cx+46,115),(cx+34,205),(cx+38,225),(cx-38,225),(cx-34,205)])
    circ(cx-44,124,16); circ(cx+44,124,16)
    limb((cx,104),(cx,82),12,12); ell(cx,52,30,36)
    # side ponytail with bow
    chain([(cx+4,24),(cx+18,0),(cx+36,-14),(cx+56,-12)],[19,15,10,3])
    # arms in a high V, pom-poms up
    chain([(cx-44,124),(cx-86,52),(cx-122,-14)],[15,12,10])
    chain([(cx+44,124),(cx+86,52),(cx+122,-14)],[15,12,10])
    pompom(cx-138,-50,60); pompom(cx+138,-50,60)
base(360,False,(360,470)); base(640,True,(640,470)); flyer(500)
m=np.array(img)
m=cv2.GaussianBlur(m,(0,0),6); m=np.where(m>127,255,0).astype(np.uint8)
ys,xs=np.where(m>0); m=m[ys.min():ys.max()+1,xs.min():xs.max()+1]
Image.fromarray(m).save('cheer_mask.png'); print(m.shape)
Image.fromarray(255-m).resize((m.shape[1]//3,m.shape[0]//3),Image.LANCZOS).save('cheer_prev.png')
