"""2026-10-22 mockup sizing (adult fronts fit 12x14 in, backs 14x14 in, hoodie fronts above the pocket,
drawstrings over the print). One renderer for Blue Crush + NSE, fronts and backs.
Lifestyle boxes are in px of each photo set's own canvas, measured from the torso width (size L)."""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from PIL import Image
import design_render as dr
from team_design_render import LIFE, load2048, _logo, SS, HERO

SPEC = {k: v for k, v in json.load(open(os.path.join(os.path.dirname(__file__), 'flat_specs_v2.json'))).items() if not k.startswith('_')}
# lifestyle print boxes (cx, top fixed per photo; W = 12 in, H = 14 in in that photo's px)
BOX = {'tee': (1018, 470, 520, 607),        # Gildan 5000 male model, 2048 px (oversized tee, ~43 px/in)
       'crew': (1022, 551, 520, 607),       # Gildan 18000 model, 2048 px (oversized crew, ~43 px/in)
       'seated': (None, None, 600, 700),    # B+C seated, 2000 px canvas, ~50 px/in
       'couch': (None, None, 645, 750),     # B+C couch, 2000 px canvas, ~54 px/in
       'flatlay': (None, None, 790, 920)}   # CC flat lay, 2048 px, ~66 px/in
HOOD = {  # studio model (2048 px, ~45 px/in front, ~48 px/in back): (cx, top, W, H)
 ('Black', 'Front'): (993, 552, 544, 1064 - 20 - 552), ('Ash', 'Front'): (968, 535, 544, 1105 - 20 - 535),
 ('Black', 'Back'): (1040, 662, 672, 672), ('Ash', 'Back'): (1024, 691, 672, 672)}

def fit(logo, W, H):
    asp = logo.width / logo.height
    return min(W, H * asp)

_MED = {}
def flat(logo, set_, color, view):
    key = (set_, view)
    if key not in _MED: _MED[key] = dr.median_box(sorted(glob.glob(f'{SS}/{set_}/*_{view}.jpg')))
    img = np.asarray(Image.open(f"{SS}/{set_}/{color.replace(' ', '-')}_{view}.jpg").convert('RGB'))
    sp = SPEC[set_][view]
    b = dr.flat_bbox(img); med = _MED[key]
    if b is None or abs(b['y0'] - med['y0']) > 30 or abs(b['y1'] - med['y1']) > 30 or abs(b['cx'] - med['cx']) > 25: b = med
    hgt = b['y1'] - b['y0']; base = b['body_w'] if sp.get('w_ref') == 'body' else (b['x1'] - b['x0'])
    cx, top, W = b['cx'], b['y0'] + sp['ft'] * hgt, sp['fw'] * base
    H = W * sp['hr']
    if 'pocket' in sp: H = min(H, (sp['pocket'] - sp['ft']) * hgt - 10)
    w = fit(logo, W, H)
    sq, k, xo = dr.to_square(img)
    res = dr.blend(sq, logo, cx * k + xo, top * k, w * k)
    if view == 'Front' and set_.startswith('hoodie'):
        res = dr.cords_over(res, img, sq, k, xo, cx, top, w, w * logo.height / logo.width)
    return res

def render(logo, kind, args):
    if kind in ('tee', 'crew'):
        cx, top, W, H = BOX[kind]
        return dr.blend(load2048(args[0]), logo, cx, top, fit(logo, W, H))
    if kind == 'life':
        set_, color = args; p, d, _ = LIFE[set_]; _, _, W, H = BOX[set_]
        im = Image.open(f'{d}/{color}.jpg'); k = 2048 / im.width
        return dr.blend(load2048(f'{d}/{color}.jpg'), logo, p[color]['cx'] * k, p[color]['top'] * k, fit(logo, W, H) * k)
    if kind == 'hoodie_model':
        c, view = args[0], (args[1] if len(args) > 1 else 'Front')
        cx, top, W, H = HOOD[(c, view)]
        im = Image.open(f'{HERO}/{c}_-_Studio_{view}_-_Gildan_18500_Sweatshirt.jpg').convert('RGB').resize((2048, 2048), Image.LANCZOS)
        return dr.blend(np.asarray(im), logo, cx, top, fit(logo, W, H))
    if kind == 'flat':
        set_, color = args[0], args[1]; view = args[2] if len(args) > 2 else 'Front'
        return flat(logo, set_, color, view)
    raise ValueError(kind)

def run_job(job):
    design_png, kind, args, out = job
    if os.path.exists(out): return out, 'cached'
    try:
        a = render(_logo(design_png), kind, args)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        Image.fromarray(a).save(out, quality=90)
        return out, 'ok'
    except Exception as e:
        return out, 'ERR ' + repr(e)

if __name__ == '__main__':
    from multiprocessing import Pool
    jobs = [tuple(j) for j in json.load(open(sys.argv[1]))['jobs']]
    with Pool(2) as pool:
        for out, st in pool.imap_unordered(run_job, jobs):
            if st.startswith('ERR'): print(out, st, flush=True)
    print('done', len(jobs))
