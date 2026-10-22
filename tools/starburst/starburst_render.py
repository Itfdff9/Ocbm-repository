"""Starburst (NSE) mockups: front + back for 8 blanks. Writes designs/nse-starburst/<blank>/<Color>_<Front|Back>.jpg + plan.json.
Placement/blend follow tools/team_design_render.py (same print boxes as the other NSE single-design listings)."""
import sys, os, json, glob
sys.path.insert(0, '/home/claude/w/repo/tools')
import numpy as np
from PIL import Image
import design_render as dr
from render import clean_name

R = '/home/claude/w/repo'; SS = '/home/claude/w/ss'; HERO = '/home/claude/w/hero'
OUT = '/home/claude/w/out/designs/nse-starburst'
PRINT = '/tmp/claude-0/-home-claude/f4680fbe-84fa-5212-8106-a18cccebeb09/scratchpad/nse_new/var_C.png'
HR = {'tee': 1.2, 'crew': 1.1, 'hoodie': 0.96, 'life': 1.15, 'flatlay': 1.1,
      'tee_5000': 1.2, 'bc_3001': 1.2, 'cc_1717': 1.2, 'crew_18000': 1.05, 'hoodie_18500': 0.96,
      'youth_tee_5000b': 1.2, 'toddler_tee_3301t': 1.15, 'onesie_4400': 1.1}
fix = {'Bella-Canvas-Millitary': 'Military-Green', 'Blue-Jeans': 'Blue-Jean', 'Chilli': 'Chili', 'Sand-Stone': 'Sandstone'}
def _P(j):
    d = {}
    for k, v in json.load(open(f'{R}/tools/{j}')).items():
        n = clean_name(k); d[fix.get(n, n)] = v
    return d
LIFE = {'seated': (_P('place_bc3001_a.json'), f'{R}/customizer/blanks/bc3001/seated', 426),
        'couch': (_P('place_bc3001_b.json'), f'{R}/customizer/blanks/bc3001/couch', 443),
        'flatlay': (_P('place_cc1717_flatlay.json'), f'{R}/customizer/blanks/cc1717/flatlay', 590)}
HP = {'Black_Front': (388, 305, 210), 'Ash_Front': (378, 295, 210), 'Black_Back': (406.4, 258.6, 250), 'Ash_Back': (400, 270, 250)}
HOOD_TOP = {'Black': 552, 'Ash': 535}

def fit(logo, W, hr): return min(W, W * hr * logo.width / logo.height)
def load2048(path):
    im = Image.open(path).convert('RGB')
    if im.size != (2048, 2048): im = im.resize((2048, 2048), Image.LANCZOS)
    return np.asarray(im)
_LOGO = None
def logo():
    global _LOGO
    if _LOGO is None:
        lg = dr.tight(Image.open(PRINT).convert('RGBA')); lg.thumbnail((1600, 1600), Image.LANCZOS); _LOGO = lg
    return _LOGO
_MED = {}
def med(set_, view):
    if (set_, view) not in _MED: _MED[(set_, view)] = dr.median_box(sorted(glob.glob(f'{SS}/{set_}/*_{view}.jpg')))
    return _MED[(set_, view)]

def render(kind, args):
    lg = logo()
    if kind == 'tee':
        return dr.blend(load2048(f'{R}/customizer/blanks/tee/{args[0]}.jpg'), lg, 1018, 470, fit(lg, 440, HR['tee']))
    if kind == 'crew':
        return dr.blend(load2048(f'{R}/customizer/blanks/crew/{args[0]}.jpg'), lg, 1022, 551, fit(lg, 426, HR['crew']))
    if kind == 'hoodie_model':
        c, view = args; cx, cy, w = HP[f'{c}_{view}']; s = 2048 / 800
        im = load2048(f'{HERO}/{c}_-_Studio_{view}_-_Gildan_18500_Sweatshirt.jpg')
        if view == 'Front': return dr.blend(im, lg, cx * s, HOOD_TOP[c], fit(lg, w, HR['hoodie']) * s)
        return dr.blend(im, lg, cx * s, cy * s, w * s)
    if kind == 'life':
        set_, color = args; p, d, W = LIFE[set_]
        k = 2048 / Image.open(f'{d}/{color}.jpg').width
        hr = HR['flatlay'] if set_ == 'flatlay' else HR['life']
        return dr.blend(load2048(f'{d}/{color}.jpg'), lg, p[color]['cx'] * k, p[color]['top'] * k, fit(lg, W, hr) * k)
    if kind == 'flat':
        set_, color, view = args
        f = f"{SS}/{set_}/{color.replace(' ', '-')}_{view}.jpg"
        img = np.asarray(Image.open(f).convert('RGB'))
        cx, top, W, ok = dr.flat_place(img, dr.FLAT_SPECS[set_][view], med(set_, view))
        sq, k, xo = dr.to_square(img)
        w = fit(lg, W, HR[set_])
        res = dr.blend(sq, lg, cx * k + xo, top * k, w * k)
        if view == 'Front' and set_ == 'hoodie_18500':
            res = dr.cords_over(res, img, sq, k, xo, cx, top, w, w * lg.height / lg.width)
        return res
    raise ValueError(kind)

CORE = ['Black', 'White', 'Royal', 'Ash', 'Charcoal', 'Light Pink', 'Heliconia', 'Navy', 'Sport Grey', 'Dark Heather']
TEEFILE = {'Ash': 'Ash-Grey'}
PLAN = {
 'g5000': [(c, ('tee', (TEEFILE.get(c, c.replace(' ', '-')),)), ('flat', ('tee_5000', c, 'Back'))) for c in CORE],
 'g18000': [(c, ('crew', (c.replace(' ', '-'),)), ('flat', ('crew_18000', c, 'Back'))) for c in CORE],
 'g18500': [(c, ('hoodie_model', (c, 'Front')) if c in ('Black', 'Ash') else ('flat', ('hoodie_18500', c, 'Front')),
             ('hoodie_model', (c, 'Back')) if c in ('Black', 'Ash') else ('flat', ('hoodie_18500', c, 'Back'))) for c in CORE],
 'bc3001': [(c, front, ('flat', ('bc_3001', c, 'Back'))) for c, front in [
     ('Black', ('life', ('seated', 'Black'))), ('White', ('life', ('couch', 'White'))), ('True Royal', ('flat', ('bc_3001', 'True Royal', 'Front'))),
     ('Ash', ('life', ('seated', 'Ash'))), ('Asphalt', ('life', ('seated', 'Asphalt'))), ('Soft Pink', ('life', ('seated', 'Soft-Pink'))),
     ('Fuchsia', ('life', ('seated', 'Fuchsia'))), ('Navy', ('life', ('seated', 'Navy'))),
     ('Athletic Heather', ('life', ('couch', 'Athletic-Heather'))), ('Dark Grey Heather', ('life', ('couch', 'Dark-Grey-Heather')))]],
 'cc1717': [(c, ('life', ('flatlay', c.replace(' ', '-'))), None if c == 'Heliconia' else ('flat', ('cc_1717', c, 'Back')))
            for c in ['Black', 'White', 'Grey', 'Pepper', 'Blossom', 'Heliconia', 'True Navy', 'Midnight']],
 'youth_5000b': [(c, ('flat', ('youth_tee_5000b', c, 'Front')), ('flat', ('youth_tee_5000b', c, 'Back'))) for c in CORE],
 'toddler_3301t': [(c, ('flat', ('toddler_tee_3301t', c, 'Front')), ('flat', ('toddler_tee_3301t', c, 'Back')))
                   for c in ['Black', 'White', 'Royal', 'Ash', 'Charcoal', 'Pink', 'Hot Pink', 'Navy', 'Heather']],
 'onesie_4400': [(c, ('flat', ('onesie_4400', c, 'Front')), ('flat', ('onesie_4400', c, 'Back')))
                 for c in ['Black', 'White', 'Royal', 'Heather', 'Charcoal', 'Pink', 'Hot Pink', 'Navy', 'Granite Heather']],
}
def jobs():
    J, P = [], {}
    for blank, rows in PLAN.items():
        for color, front, back in rows:
            for view, spec in (('Front', front), ('Back', back)):
                if spec is None: continue
                fn = f"{OUT}/{blank}/{color.replace(' ', '-')}_{view}.jpg"
                J.append((spec[0], spec[1], fn)); P.setdefault(blank, {}).setdefault(color, {})[view] = fn
    return J, P

def run(job):
    kind, args, fn = job
    if os.path.exists(fn): return fn, 'cached'
    try:
        a = render(kind, args); os.makedirs(os.path.dirname(fn), exist_ok=True)
        Image.fromarray(a).save(fn, quality=90); return fn, 'ok'
    except Exception as e:
        import traceback; return fn, 'ERR ' + traceback.format_exc()[-600:]

if __name__ == '__main__':
    J, P = jobs()
    json.dump(P, open(f'{OUT}/plan.json', 'w'), indent=0)
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    if only: J = [j for j in J if any(o in j[2] for o in only)]
    print(len(J), 'images', flush=True)
    from multiprocessing import Pool
    with Pool(2) as pool:
        for fn, st in pool.imap_unordered(run, J):
            if st != 'ok' and st != 'cached': print(fn, st, flush=True)
    print('done')
