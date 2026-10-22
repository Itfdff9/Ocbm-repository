import sys, json, os, glob; sys.path.insert(0,'/home/claude/ocbm-repository/tools')
import numpy as np, design_render as dr
from render import clean_name
from PIL import Image
R='/home/claude/ocbm-repository'; SS='/tmp/claude-0/-home-claude/a27aa439-f433-53f6-993d-86ad7f18ed24/scratchpad/ssblanks'
D2='/tmp/claude-0/-home-claude/a27aa439-f433-53f6-993d-86ad7f18ed24/scratchpad/drive2048'
HERO='/home/claude/work/hero'
LOGO=dr.tight(Image.open('/tmp/claude-0/-home-claude/a27aa439-f433-53f6-993d-86ad7f18ed24/scratchpad/bluecrush/bluecrush_trimmed_hard.png').convert('RGBA'))
LOGO.thumbnail((1600,1600),Image.LANCZOS)
ASP=LOGO.height/LOGO.width
fix={'Bella-Canvas-Millitary':'Military-Green','Blue-Jeans':'Blue-Jean','Chilli':'Chili','Sand-Stone':'Sandstone'}
def P(j):
    d={}
    for k,v in json.load(open(f'{R}/tools/{j}')).items():
        n=clean_name(k); d[fix.get(n,n)]=v
    return d
LIFE={'seated':(P('place_bc3001_a.json'),f'{R}/customizer/blanks/bc3001/seated',426),
      'couch':(P('place_bc3001_b.json'),f'{R}/customizer/blanks/bc3001/couch',443),
      'flatlay':(P('place_cc1717_flatlay.json'),f'{R}/customizer/blanks/cc1717/flatlay',590)}
def life(set_, color):
    p,d,w=LIFE[set_]; im=Image.open(f'{d}/{color}.jpg').convert('RGB'); k=2048/im.width
    im=im.resize((2048,2048),Image.LANCZOS)
    return dr.blend(np.asarray(im),LOGO,p[color]['cx']*k,p[color]['top']*k,w*k)
def tee_front(color):   # Gildan 5000 lifestyle, 2048
    return dr.blend(np.asarray(Image.open(f'{D2}/g5000/{color}.jpg').convert('RGB')),LOGO,1018,470,440)
def crew_front(color):  # Gildan 18000 lifestyle, 2048
    return dr.blend(np.asarray(Image.open(f'{D2}/g18000/{color}.jpg').convert('RGB')),LOGO,1022,551,426)
HP={'Black_Front':(388,305,210),'Ash_Front':(378,295,210),'Black_Back':(406.4,258.6,250),'Ash_Back':(400,270,250)}
def hoodie_model(color, view):
    cx,c2,w=HP[f'{color}_{view}']; s=2048/800
    im=Image.open(f'{HERO}/{color}_-_Studio_{view}_-_Gildan_18500_Sweatshirt.jpg').convert('RGB').resize((2048,2048),Image.LANCZOS)
    if view=='Front':   # print top just below the tied drawstring tips (Black ~537, Ash ~519 px on 2048)
        return dr.blend(np.asarray(im),LOGO,cx*s,{'Black':552,'Ash':535}[color],w*s)
    return dr.blend(np.asarray(im),LOGO,cx*s,c2*s,w*s)
_MED={}
def flat(set_, color, view):
    key=(set_,view)
    if key not in _MED: _MED[key]=dr.median_box(sorted(glob.glob(f'{SS}/{set_}/*_{view}.jpg')))
    res,ok=dr.render_flat(f"{SS}/{set_}/{color.replace(' ','-')}_{view}.jpg",LOGO,dr.FLAT_SPECS[set_][view],_MED[key],
                          cords=(view=='Front' and set_ in ('hoodie_18500','youth_hoodie_18500b')))
    return res
def save(arr, path): os.makedirs(os.path.dirname(path),exist_ok=True); Image.fromarray(arr).save(path,quality=90)
def _load2048(path):
    im = Image.open(path).convert('RGB')
    if im.size != (2048, 2048): im = im.resize((2048, 2048), Image.LANCZOS)
    return np.asarray(im)
def tee_front_file(path): return dr.blend(_load2048(path), LOGO, 1018, 470, 440)
def crew_front_file(path): return dr.blend(_load2048(path), LOGO, 1022, 551, 426)
