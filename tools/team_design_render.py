"""Front mockups for any team design on the OCBM blank library (Core-7 style color plans).
Each garment has a print box: width W and max height W*hr. The design is fitted inside it
(wide art fills the width, tall art is limited by height), centred on cx with its top at `top`.
Usage: python3 team_design_render.py plan.json   (see make_nse_plan in nse_plan.py)"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
import design_render as dr
from render import clean_name

R = '/home/claude/ocbm-repository'
SS = '/tmp/claude-0/-home-claude/a27aa439-f433-53f6-993d-86ad7f18ed24/scratchpad/ssblanks'
D2 = '/tmp/claude-0/-home-claude/a27aa439-f433-53f6-993d-86ad7f18ed24/scratchpad/drive2048'
HERO = '/home/claude/work/hero'
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
HP = {'Black': (388, 305, 210), 'Ash': (378, 295, 210)}   # hoodie studio model fronts, 800 grid (cx, cy, w)
HOOD_TOP = {'Black': 552, 'Ash': 535}   # print top on the 2048 px photo: just below the tied drawstring tips (Black ~537, Ash ~519)

def fit(logo, W, hr):
    asp = logo.width / logo.height
    return min(W, W * hr * asp)

def load2048(path):
    im = Image.open(path).convert('RGB')
    if im.size != (2048, 2048): im = im.resize((2048, 2048), Image.LANCZOS)
    return np.asarray(im)

_MED = {}
def render(logo, kind, args):
    if kind == 'tee':      # Gildan 5000 lifestyle (male model), 2048
        return dr.blend(load2048(args[0]), logo, 1018, 470, fit(logo, 440, HR['tee']))
    if kind == 'crew':     # Gildan 18000 lifestyle, 2048
        return dr.blend(load2048(args[0]), logo, 1022, 551, fit(logo, 426, HR['crew']))
    if kind == 'hoodie_model':
        c = args[0]; cx, cy, w = HP[c]; s = 2048 / 800
        im = Image.open(f'{HERO}/{c}_-_Studio_Front_-_Gildan_18500_Sweatshirt.jpg').convert('RGB').resize((2048, 2048), Image.LANCZOS)
        return dr.blend(np.asarray(im), logo, cx * s, HOOD_TOP[c], fit(logo, w, HR['hoodie']) * s)
    if kind == 'life':
        set_, color = args; p, d, W = LIFE[set_]
        im = Image.open(f'{d}/{color}.jpg').convert('RGB'); k = 2048 / im.width
        hr = HR['flatlay'] if set_ == 'flatlay' else HR['life']
        return dr.blend(load2048(f'{d}/{color}.jpg'), logo, p[color]['cx'] * k, p[color]['top'] * k, fit(logo, W, hr) * k)
    if kind == 'flat':
        set_, color = args
        key = (set_, 'Front')
        if key not in _MED: _MED[key] = dr.median_box(sorted(glob.glob(f'{SS}/{set_}/*_Front.jpg')))
        f = f"{SS}/{set_}/{color.replace(' ', '-')}_Front.jpg"
        img = np.asarray(Image.open(f).convert('RGB'))
        cx, top, W, ok = dr.flat_place(img, dr.FLAT_SPECS[set_]['Front'], _MED[key])
        sq, k, xo = dr.to_square(img)
        w = fit(logo, W, HR[set_])
        res = dr.blend(sq, logo, cx * k + xo, top * k, w * k)
        if set_ in ('hoodie_18500', 'youth_hoodie_18500b'):
            res = dr.cords_over(res, img, sq, k, xo, cx, top, w, w * logo.height / logo.width)
        return res
    raise ValueError(kind)

def run_job(job):
    design_png, kind, args, out = job
    if os.path.exists(out): return out, 'cached'
    try:
        logo = _logo(design_png)
        a = render(logo, kind, args)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray(a).save(out, quality=90)
        return out, 'ok'
    except Exception as e:
        return out, 'ERR ' + repr(e)

_LOGOS = {}
def _logo(p):
    if p not in _LOGOS:
        lg = dr.tight(Image.open(p).convert('RGBA')); lg.thumbnail((1600, 1600), Image.LANCZOS); _LOGOS[p] = lg
    return _LOGOS[p]

if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [tuple(j) for j in json.load(open(sys.argv[1]))['jobs']]
    with Pool(2) as pool:
        for out, st in pool.imap_unordered(run_job, jobs):
            if st.startswith('ERR'): print(out, st, flush=True)
    print('done', len(jobs))
