import numpy as np, sys, json
from surgery import *
from scipy import ndimage as nd
SRC='../names48/src_VELEZ.png'
img=load(SRC); H,W=img.shape[:2]
P=json.load(open(sys.argv[1]))
OUT=P.get('OUT','work/VELEZ_fix.png')
mZ=poly_mask(img.shape,P['S_ZONE']); mR=poly_mask(img.shape,P['R_KEEP'])
REM=mZ&~mR
yy,xx=np.mgrid[0:H,0:W]
clean=np.zeros_like(img)
def vcopy(xmask, y_src0, period, xsrc=None):
    """fill pixels (xmask & below y_src0+period) by copying the same (or mapped) column from rows [y_src0, y_src0+period)"""
    ys=yy; k=np.floor((ys-y_src0)/period); ysrc=(ys-period*k).astype(int)
    xs=xx if xsrc is None else xsrc
    m=xmask
    clean[m]=img[np.clip(ysrc[m],0,H-1),np.clip(xs[m],0,W-1)]
# --- L side columns
LP0,LP1=P.get('L_PANEL_ROWS',[1050,33])
LB0,LB1=P.get('L_BORDER_ROWS',[1050,29])
Lpanel=(xx<2345)
Lborder=(xx>=2345)&(xx<=2392)
vcopy(Lpanel,LP0,LP1)
vcopy(Lborder,LB0,LB1)
# gap
gap=(xx>2392)&(xx<2428)
clean[gap]=0
# --- E side = mirror of L side about the gap
xm=(4820-xx)
Eborder=(xx>=2428)&(xx<=2475)
Epanel=(xx>2475)
vcopy(Eborder,LB0,LB1,xsrc=xm)
vcopy(Epanel,LP0,LP1,xsrc=xm)
# --- L bottom structure: rows 1287..1386 copied horizontally from columns [2185,2214)
LBOT0, LBOT1 = 1287, 1387
cx0,cper=2185,29
def bottom_sample(x, y):
    k=np.floor((x-cx0)/cper); xs=(x-cper*k).astype(int)
    return img[np.clip(y,0,H-1).astype(int), np.clip(xs,0,W-1)]
# depth fractions for the mitre
Rdepth0, Rdepth1 = 2305, 2392      # right structure: panel edge -> outer edge
Bdepth0, Bdepth1 = 1287, 1386      # bottom structure: panel edge -> outer edge
reg=(xx>=2185)&(xx<=2392)&(yy>=LBOT0)
fr=(xx-Rdepth0)/(Rdepth1-Rdepth0); fb=(yy-Bdepth0)/(Bdepth1-Bdepth0)
use_bottom=reg&((xx<Rdepth0)|(fb>=fr))
vals=bottom_sample(xx[use_bottom],yy[use_bottom]); clean[use_bottom]=vals
clean[(xx<=2392)&(yy>LBOT1-1)]=0
# --- E bottom structure: outer edge ye(x) = E_Y0 + E_SL*(x-2440); mapped from the L bottom rows
E_Y0,E_SL=P.get('E_BOTTOM',[1379,0.21])
ye=E_Y0+E_SL*(xx-2440)
dy=ye-1386
ysrc=yy-dy
Ed0,Ed1=2515,2428                  # E left structure: panel edge (inner) -> outer edge
fr_e=(Ed0-xx)/(Ed0-Ed1)
fb_e=(ysrc-Bdepth0)/(Bdepth1-Bdepth0)
regE=(xx>=2428)&(xx<=2600)&(ysrc>=LBOT0)
use_bE=regE&((xx>Ed0)|(fb_e>=fr_e))
clean[use_bE]=bottom_sample(xx[use_bE],np.round(ysrc[use_bE]))
clean[(xx>=2428)&(yy>ye)]=0
# star rays hidden under the s? (L panel star at ~2250,985 is far above) - none
FE=P.get('FEATHER',0)
if FE>0:
    dist=nd.distance_transform_edt(REM)
    wrem=np.clip(dist/FE,0,1)[...,None]
    base=img*(1-wrem)+clean*wrem
    base[~REM]=img[~REM]
else:
    base=img.copy(); base[REM]=clean[REM]
# --- strokes (drawn in order; later ones on top)
SX0,SX1=P.get('TEX_X',[2690.0,2726.0]); CY0,CY1=P.get('TEX_CY',[1059.5,1059.0]); THW=P.get('TEX_HW',53.5)
def over(top,ta,bot):
    ba=bot[...,3:4]/255.0; ta=ta[...,None]
    oa=ta+ba*(1-ta)
    oc=(top[...,:3]*ta+bot[...,:3]*ba*(1-ta))/np.maximum(oa,1e-4)
    return np.concatenate([oc,oa*255],-1)
out=base.copy()
strokes=P['STROKES'] if 'STROKES' in P else [{'path':P['BRIDGE'],'hw':P.get('HW',[54.0,50.0]),'fade':P.get('FADE',[35,40])}]
for st in strokes:
    path=catmull_rom(st['path'])
    pad=80
    bb=(int(path[:,0].min())-pad,int(path[:,1].min())-pad,int(path[:,0].max())+pad,int(path[:,1].max())+pad)
    S,N,L=stroke_coords(path,bb)
    HW0,HWm=st['hw'][0],st['hw'][1]; HW1=st['hw'][2] if len(st['hw'])>2 else HW0
    u=np.clip(S/L,0,1)
    hw=np.where(u<0.5, HW0+(HWm-HW0)*np.sin(np.pi*u)**2, HW1+(HWm-HW1)*np.sin(np.pi*u)**2)
    off=st.get('tex_off',0.0)
    sxx=SX0+tri(S+off,SX1-SX0)
    cyb=CY0+(sxx-SX0)*(CY1-CY0)/(SX1-SX0)
    rowN=N*(THW/hw)
    if 'hl_end' in st:
        # progressively move the highlight toward the left rim near the end (match the target stroke's lighting)
        d_end, wlen = st['hl_end']            # target distance of highlight from the left rim (px), warp length
        w=smoothstep(L-wlen, L-15, S)
        hl_tex=-THW+24.5                      # texture highlight row (24.5 px inside the rim)
        Nh=-hw+d_end                          # where the highlight should land
        warped=np.where(N<Nh, -THW+(N+hw)*(24.5/np.maximum(d_end,1e-3)), hl_tex+(N-Nh)*((0-hl_tex)/np.maximum(0-Nh,1e-3)))
        warped=np.where(N>=0, rowN, warped)
        rowN=rowN*(1-w)+warped*w
    tex=sample(img,sxx,cyb+rowN)
    inside=np.clip((hw+0.5-np.abs(N)),0,1)
    F0,F1=st['fade']
    ramp_in=smoothstep(0,F0,S); ramp_out=smoothstep(0,F1,L-S)
    x0,y0,x1,y1=bb
    inR=mR[y0:y1,x0:x1]; notS=~mZ[y0:y1,x0:x1]
    fin=np.where(inR,ramp_in,1.0) if st.get('fade_in_only_outside',True) else ramp_in
    fout=np.where(notS&~inR,ramp_out,1.0) if st.get('fade_out_only_outside',True) else ramp_out
    cap=smoothstep(-1.0,1.0,S)*smoothstep(-1.0,1.0,L-S)
    a_conn=inside*cap*fin*fout*(tex[...,3]/255.0)
    out[y0:y1,x0:x1]=over(tex,a_conn,out[y0:y1,x0:x1])
save(out,OUT); print('saved',OUT)
