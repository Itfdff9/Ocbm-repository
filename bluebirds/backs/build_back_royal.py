"""Jersey back name + number — BLUEBIRDS ROYAL-FILL colorway (BURUCA 12 black-hoodie reference, 2026-10-13):
royal fill / white outline / thin vegas-gold outer edge, for black and white shirts.
(was: BLUEBIRDS colorway (from Brian's COOK 6 reference, 2026-10-13):
white fill / navy outline / thin vegas-gold outer edge, letters spaced apart (outlines do not touch).
Graduate varsity (name condensed + weighted to match the reference), hard alpha.
usage: python3 build_back.py NAME NUMBER OUTDIR [name_width_in=12] [number_height_in=8] [dpi=300] [suffix]"""
import sys, os, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
GRAD="/home/claude/ocbm-repository/raiderettes/shoulder_40x6/fonts/Graduate-Regular.ttf"
WHITE=(18,64,160); BLACK=(255,255,255); ROYAL=(192,170,118)   # fill = royal, inner outline = white, outer edge = vegas gold
# proportions measured from Brian's reference (fill = white part)
NAME_ASPECT=None          # natural Graduate widths + LETTER_GAP spacing (COOK reference: 3.71 for 4 letters)
NUM_GAP=0.26              # digit spacing (12 ref: 141 px between digit fills at 533 px)
LETTER_GAP=0.24           # BURUCA ref: fill aspect 5.69 for 6 letters
NAME_OUT=11/168           # outline per side / fill height (name, measured on BURUCA ref)
NUM_OUT=21/533            # outline per side / fill height (number, measured on the 12 ref)
BLACK_SHARE=0.55          # white part of the outline; the rest is the gold edge
NAME_BOLD=0.015; NUM_BOLD=0.022
NUM_ASPECT=328/533        # white-fill width / height of the number in the reference   # weight-up as a fraction of fill height (kept light so counters stay open)

def glyph(ch, px):
    f=ImageFont.truetype(GRAD,px); l,t,r,b=f.getbbox(ch); pad=px//5
    m=Image.new("L",(r-l+2*pad,b-t+2*pad),0); ImageDraw.Draw(m).text((pad-l,pad-t),ch,font=f,fill=255)
    a=np.asarray(m); ys,xs=np.where(a>127); return a[ys.min():ys.max()+1, xs.min():xs.max()+1], ys.min()-pad+t  # mask, top offset vs font top

def word_fill(text, fill_h, aspect=None, gap=0, bold=0):
    """white-fill mask for the whole word at fill height fill_h px; squeezed so width/height == aspect"""
    cap_ratio=glyph("H",1000)[0].shape[0]/1000
    px=int(round(fill_h/cap_ratio)); glyphs=[glyph(c,px)[0] for c in text]
    capH=glyph("H",px)[0].shape[0]
    natural=sum(g.shape[1] for g in glyphs)
    k=1.0 if aspect is None else (aspect*fill_h-gap*(len(text)-1))/natural
    out_w=int(sum(int(g.shape[1]*k) for g in glyphs)+gap*(len(text)-1))+2*bold+4
    H=max(g.shape[0] for g in glyphs)+2*bold+4; canvas=np.zeros((H,out_w),np.uint8); x=bold+2
    for g in glyphs:
        g2=cv2.resize(g,(max(1,int(g.shape[1]*k)),g.shape[0]),interpolation=cv2.INTER_AREA)
        y=H-bold-2-g2.shape[0]                                    # baseline-align (all caps / digits)
        canvas[y:y+g2.shape[0],x:x+g2.shape[1]]=np.maximum(canvas[y:y+g2.shape[0],x:x+g2.shape[1]],g2); x+=g2.shape[1]+gap
    if bold>0: canvas=cv2.dilate(canvas,np.ones((2*bold+1,2*bold+1),np.uint8))   # square kernel keeps the slab corners sharp
    return canvas, k

def stack(fill, blk, roy):
    pad=blk+roy+8; m=np.pad(fill,pad)
    el=lambda r:cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*r+1,2*r+1))
    b=cv2.dilate(m,el(blk)); r=cv2.dilate(b,el(roy))
    rgba=np.zeros(m.shape+(4,),np.uint8)
    rgba[r>127]=(*ROYAL,255); rgba[b>127]=(*BLACK,255); rgba[m>127]=(*WHITE,255)
    ys,xs=np.where(rgba[...,3]>0); rgba=rgba[ys.min():ys.max()+1, xs.min():xs.max()+1]
    return rgba

def build_name(text, width_in, dpi, _fixed=False):
    """name sized so the whole name (incl. outlines) is width_in wide; natural letter shapes, spaced like the reference"""
    probe,_=word_fill(text, 1000, None, int(LETTER_GAP*1000), 0)
    aspect=(probe.shape[1]-4)/1000.0
    total_w=width_in*dpi
    fill_h=total_w/(aspect+2*NAME_OUT); out=NAME_OUT*fill_h
    blk=int(round(out*BLACK_SHARE)); roy=int(round(out-blk)); bold=int(round(NAME_BOLD*fill_h))
    fill,k=word_fill(text, fill_h-2*bold, None, int(round(LETTER_GAP*fill_h)), bold)
    rgba=stack(fill,blk,roy)
    # the weight-up eats into the width estimate (names came out ~1.5-2% narrow); correct once so the
    # finished name lands on width_in
    want=width_in*dpi
    if not _fixed and abs(rgba.shape[1]-want)/want > 0.002:
        return build_name(text, width_in*want/rgba.shape[1], dpi, True)
    return rgba, k

def build_number(num, height_in, dpi):
    total_h=height_in*dpi; fill_h=total_h/(1+2*NUM_OUT)
    out=NUM_OUT*fill_h; blk=int(round(out*BLACK_SHARE)); roy=int(round(out-blk)); bold=int(round(NUM_BOLD*fill_h))
    fill,_=word_fill(num, fill_h-2*bold, NUM_ASPECT*fill_h/(fill_h-2*bold) if len(num)==1 else None, int(NUM_GAP*fill_h), bold)
    return stack(fill,blk,roy)

def save(rgba, path, dpi):
    rgba[...,3]=np.where(rgba[...,3]>=128,255,0); rgba[rgba[...,3]==0,:3]=0
    Image.fromarray(rgba,"RGBA").save(path,dpi=(dpi,dpi))
    h,w=rgba.shape[:2]; print(path, (w,h), "%.2f x %.2f in @ %d dpi"%(w/dpi,h/dpi,dpi))

if __name__=="__main__":
    name,num,od=sys.argv[1].upper(),sys.argv[2],sys.argv[3]
    nw=float(sys.argv[4]) if len(sys.argv)>4 else 12; nh=float(sys.argv[5]) if len(sys.argv)>5 else 8
    dpi=int(sys.argv[6]) if len(sys.argv)>6 else 300; sfx=sys.argv[7] if len(sys.argv)>7 else "%ddpi"%dpi
    os.makedirs(od,exist_ok=True)
    n,k=build_name(name,nw,dpi); print("name squeeze %.2f"%k)
    save(n, f"{od}/{name}_name_{sfx}.png", dpi)
    save(build_number(num,nh,dpi), f"{od}/{num}_number_{sfx}.png", dpi)
