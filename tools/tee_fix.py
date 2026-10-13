import sys, os, glob, subprocess, numpy as np
from PIL import Image
R='/home/claude/ocbm-repository'
TEAMS={ # team: (logo, visible width px, outdir, tag)
 'DREAMS':(R+'/dreams/logo/DREAMS_PARENT_cleaned_300dpi.png',540,R+'/dreams/g5000','DREAMS'),
 'BUFC':(R+'/bufc/logo/BUFC_crest_transparent.png',430,R+'/bufc/g5000','BUFC'),
 'BLUEBIRDS':(R+'/bluebirds/logo/BLUEBIRDS_crest_transparent.png',525,R+'/bluebirds/g5000','BLUEBIRDS'),
 'NSE':(R+'/north_shore_elite/logo/NSE_logo_transparent.png',430,R+'/north_shore_elite/g5000','NSE'),
}
CX, TOP = 1018, 470   # collar centre; top of print 3 in below the collar seam (~46 px/in on these 2048 photos)
def tight(src,dst):
    im=Image.open(src).convert('RGBA'); a=np.asarray(im)[...,3]; ys,xs=np.where(a>8)
    im.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1)).save(dst); return im.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1)).size
def render(team, color, outdir):
    logo,w,_,tag=TEAMS[team]; tl=f'/home/claude/teefix/logos/{team}.png'
    if not os.path.exists(tl): tight(logo,tl)
    lw,lh=Image.open(tl).size; h=lh*w/lw; cy=TOP+h/2
    out=f'{outdir}/G5000_{color}_Front_{tag}.jpg'
    subprocess.run(['python3','/home/claude/royal/front_logo.py',f'/home/claude/g5000/blank/{color}.jpg',tl,out,str(CX),str(cy),str(w)],check=True,capture_output=True)
    return out
if __name__=='__main__':
    teams=sys.argv[1].split(','); colors=sys.argv[2].split(',') if len(sys.argv)>2 and sys.argv[2]!='all' else [os.path.basename(f)[:-4] for f in sorted(glob.glob('/home/claude/g5000/blank/*.jpg'))]
    outroot=sys.argv[3] if len(sys.argv)>3 else None
    for t in teams:
        od=outroot+'/'+t if outroot else TEAMS[t][2]; os.makedirs(od,exist_ok=True)
        for c in colors: render(t,c,od)
        print(t,len(colors))
