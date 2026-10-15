import numpy as np, sys, json
from surgery import *
from scipy import ndimage as nd
SRC='../names48/src_TRAVA.png'
img=load(SRC); H,W=img.shape[:2]
P=json.load(open(sys.argv[1])) if len(sys.argv)>1 else {}
OUT=P.get('OUT','work/TRAVA_fix_v12.png')
S_ZONE=P.get('S_ZONE',[(2300,1228),(2321,1212),(2337,1193),(2344,1170),(2360,1143),(2390,1121),(2418,1107),(2447,1101),(2480,1103),(2497,1112),
        (2500,1150),(2502,1200),(2503,1260),(2503,1300),(2470,1346),(2468,1386),(2490,1410),(2520,1440),(2540,1480),(2240,1480),(2242,1350),(2300,1345)])
R_KEEP=P.get('R_KEEP',[(2100,1040),(2310,1040),(2310,1346),(2245,1346),(2215,1312),(2190,1330),(2150,1480),(2100,1480)])
mZ=poly_mask(img.shape,S_ZONE); mR=poly_mask(img.shape,R_KEEP)
REM=mZ&~mR
yy,xx=np.mgrid[0:H,0:W]
A_BOTTOM=P.get('A_BOTTOM',1370)
clean=np.zeros_like(img)
bz=(xx>=2400)&(xx<2493)&(yy<A_BOTTOM)
k=np.ceil(np.maximum(yy-1111,0)/14.0); ysb=np.clip(yy-14*k,0,H-1).astype(int)
clean[bz]=img[ysb[bz],xx[bz]]
sz=(xx<2400)&(yy<A_BOTTOM)
SEQ=img[900:970,2250:2320]
AX,AY=P.get('SEQ_ANCHOR',[42,59])
clean[sz]=SEQ[((yy[sz]-AY)%70),((xx[sz]-AX)%70)]
lum=img[...,:3].mean(-1)
cx,cy=2333,1112; R=75
dy,dx=np.mgrid[-R:R+1,-R:R+1]
ty,tx=cy+dy,cx+dx; sy,sx=cy-dy,cx-dx
ok=(np.hypot(dx,dy)<=R)&(dy>0)
sv=lum[sy,sx]; tv=REM[ty,tx]
add=ok&tv&(sv>120)
w=np.clip((sv-120)/80,0,1)[...,None]
patch=clean[ty,tx]
patch=np.where(add[...,None],patch*(1-w)+img[sy,sx]*w,patch)
clean[ty[ok],tx[ok]]=patch[ok]
path0=None
BR=P.get('BRIDGE',[(2285,1288),(2312,1296),(2338,1322),(2360,1356),(2385,1385),(2415,1400),(2448,1396),(2474,1378),(2492,1350),(2502,1318),(2508,1290)])
HW0,HWm=P.get('HW',[55.0,51.0])
path=catmull_rom(BR)
# clean plate: clear below the connector centre line, and left of the foot's inner edge
from scipy.interpolate import interp1d
fcy=interp1d(path[:,0],path[:,1],bounds_error=False,fill_value=(path[0,1],path[-1,1]))
below=(yy>fcy(xx))&(xx>=2262)&(xx<=2530)
clean[below]=0
clean[(xx<2262)]=0
base=img.copy(); base[REM]=clean[REM]
bb=(2200,1150,2620,1480)
S,N,L=stroke_coords(path,bb)
u=np.clip(S/L,0,1); hw=HW0+(HWm-HW0)*np.sin(np.pi*u)**2
SX0,SX1=2698.0,2746.0
sxx=SX0+tri(S,SX1-SX0)
cyb=1075.5+(sxx-SX0)*(1073.0-1075.5)/(SX1-SX0)
tex=sample(img,sxx,cyb+N*(54.5/hw))
inside=np.clip((hw+0.5-np.abs(N)),0,1)
F0,F1=P.get('FADE',[35,40])
ramp_in=smoothstep(0,F0,S); ramp_out=smoothstep(0,F1,L-S)
x0,y0,x1,y1=bb
B=base[y0:y1,x0:x1]; O=img[y0:y1,x0:x1]; inR=mR[y0:y1,x0:x1]
a_conn=inside*np.where(inR,ramp_in,1.0)*ramp_out*(tex[...,3]/255.0)
def over(top,ta,bot):
    ba=bot[...,3:4]/255.0; ta=ta[...,None]
    oa=ta+ba*(1-ta)
    oc=(top[...,:3]*ta+bot[...,:3]*ba*(1-ta))/np.maximum(oa,1e-4)
    return np.concatenate([oc,oa*255],-1)
res=over(tex,a_conn,B)
wR=inR*(1-inside*ramp_in)
res=res*(1-wR[...,None])+O*wR[...,None]
out=base.copy(); out[y0:y1,x0:x1]=res
save(out,OUT); print('saved',OUT)
