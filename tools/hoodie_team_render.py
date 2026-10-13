import json, subprocess, cv2, numpy as np, os
from PIL import Image
R='/home/claude/ocbm-repository'
lm=json.load(open('landmarks_final.json')); pl=json.load(open('final/placements.json'))
T={'BUFC':(R+'/bufc/logo/BUFC_crest_transparent.png',235,'bufc'),
   'BLUEBIRDS':(R+'/bluebirds/logo/BLUEBIRDS_crest_transparent.png',295,'bluebirds'),
   'NSE':(R+'/north_shore_elite/logo/NSE_logo_transparent.png',245,'north_shore_elite')}
def tight(p,o):
    im=Image.open(p).convert('RGBA'); a=np.asarray(im)[...,3]; ys,xs=np.where(a>8); t=im.crop((xs.min(),ys.min(),xs.max()+1,ys.max()+1)); t.save(o); return t.size
out={}
for tag,(logo,W0,folder) in T.items():
    tl=f'teams/{tag}_tight.png'; lw,lh=tight(logo,tl); asp=lh/lw
    for c,L in sorted(lm.items()):
        if c in ('Royal-Seated','Black','Ash'): continue
        space=L['pocket']-L['tip']; W=min(W0,(space-50)/asp)
        cy=(L['tip']+L['pocket'])/2
        tmp=f'teams/_tmp.jpg'
        subprocess.run(['python3','/home/claude/royal/front_logo.py',f'clean/{c}.png',tl,tmp,str(L['cx']),str(cy),str(W),'--quality','97'],check=True,capture_output=True)
        im=cv2.imread(tmp); g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY).astype(float)
        top=0
        while g[top].std()<8 and g[top].mean()>185: top+=1
        bot=im.shape[0]-1
        while g[bot].std()<8 and g[bot].mean()>185: bot-=1
        o=f'{R}/{folder}/g18500/G18500_{c}_Front_{tag}.jpg'
        cv2.imwrite(o,im[top+2:bot-1],[cv2.IMWRITE_JPEG_QUALITY,93])
        out.setdefault(tag,{})[c]=dict(cx=L['cx'],cy=cy,width=round(W,1))
json.dump(out,open(R+'/tools/place_g18500_teams.json','w'),indent=0)
print({k:len(v) for k,v in out.items()}, {k:min(x['width'] for x in v.values()) for k,v in out.items()})
