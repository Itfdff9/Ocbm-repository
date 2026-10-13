import sys, os; sys.path.insert(0,'.')
from render import *
R='../ocbm/'
NAVY=(14,28,70); WHITE=(248,248,246); BLACK=(12,12,14); ROYAL_G=(22,72,158); ROYAL=(42,93,179); GOLD=(201,164,92)
TUMBLER=[('Black',(22,22,24)),('White',(236,236,234)),('Blue',(32,82,170)),('Gray',(122,125,129)),('Pink',(238,140,180)),
 ('Navy',(26,36,72)),('Light Blue',(138,185,222)),('Purple',(95,62,145)),('Red',(185,30,40)),('Lime Green',(150,200,60)),('Orange',(232,110,35))]
BOTTLE=[('Black',(22,22,24)),('White',(236,236,234)),('Graphite',(62,64,68)),('Navy Blue',(26,36,72))]
LIGHT={'White','Light Blue','Lime Green','Pink'}
TEAMS={
 'dreams':dict(logo='dreams/logo/DREAMS_Fastpitch_PERMANENT_300dpi.png',name=('MILDENBERGER','8'),cols=lambda c:(ROYAL_G,BLACK,WHITE)),
 'bufc':dict(logo='bufc/logo/BUFC_crest_transparent.png',name=('BURUCA','11'),cols=lambda c:(NAVY,WHITE,NAVY)),
 'bluebirds':dict(logo='bluebirds/logo/BLUEBIRDS_crest_transparent.png',name=('COOK','6'),cols=lambda c:(GOLD,WHITE,ROYAL) if c in LIGHT else (GOLD,ROYAL,WHITE)),
 'north_shore_elite':dict(logo='customizer/designs/nse_emblem.png',name=('SMITH','12'),cols=lambda c:(ROYAL_G,BLACK,WHITE)),
}
team=sys.argv[1]; T=TEAMS[team]
od=f'out/{team}/drinkware'; os.makedirs(od,exist_ok=True)
logo=load_logo(R+T['logo'])
slug=lambda s:s.replace(' ','-')
cache={}
def nm(c):
    k=T['cols'](c)
    if k not in cache: cache[k]=name_art(*T['name'],k)
    return cache[k]
for c,rgb in TUMBLER:
    f=f'{od}/Tumbler40_{slug(c)}_logo.jpg'
    if not os.path.exists(f): render_tumbler(rgb,logo,fit_h(logo,3.5,3.4),6.3,'right').save(f,quality=92)
    f=f'{od}/Tumbler40_{slug(c)}_name.jpg'
    if not os.path.exists(f): n=nm(c); render_tumbler(rgb,n,fit_h(n,3.2,3.4),6.3,'left').save(f,quality=92)
for c,rgb in BOTTLE:
    f=f'{od}/Bottle21_{slug(c)}_logo.jpg'
    if not os.path.exists(f): render_bottle(rgb,logo,fit_h(logo,2.6,3.0),4.3).save(f,quality=92)
    f=f'{od}/Bottle21_{slug(c)}_name.jpg'
    if not os.path.exists(f): n=nm(c); render_bottle(rgb,n,fit_h(n,2.45,3.0),4.3).save(f,quality=92)
print(team,'done')
