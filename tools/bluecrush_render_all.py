"""Render every color, front + back, for the Bethpage Blue Crush listings.
Writes designs/bethpage-blue-crush/<blank>/<Color>_<Front|Back>.jpg and plan.json (color -> images)."""
import sys, os, json, re, glob
sys.path.insert(0, '/home/claude/work'); sys.path.insert(0, '/home/claude/ocbm-repository/tools')
from multiprocessing import Pool
import numpy as np
from PIL import Image
import bluecrush_lib as L

R = '/home/claude/ocbm-repository'; OUT = f'{R}/designs/bethpage-blue-crush'
def slug(c): return c.replace('/', '-').replace(' ', '-')
def norm(s): return re.sub(r'[^a-z0-9]', '', s.lower())
def find(d, name, alias=None):
    alias = alias or {}
    n = norm(alias.get(name, name))
    for f in glob.glob(d + '/*.jpg'):
        if norm(os.path.basename(f)[:-4]) == n: return f
    return None
def ssfile(set_, name, view, alias=None):
    alias = alias or {}
    f = f'{L.SS}/{set_}/{slug(alias.get(name, name))}_{view}.jpg'
    return f if os.path.exists(f) else None

DREAMS = json.load(open('/home/claude/work/dreams_colors.json'))
KIDS = json.load(open('/home/claude/work/kids_colors.json'))

def plan():
    jobs, P = [], {}
    def add(prod, color, kind, args, view):
        fn = f'{OUT}/{prod}/{slug(color)}_{view}.jpg'
        jobs.append((kind, args, fn)); P.setdefault(prod, {}).setdefault(color, []).append(fn)
    # Gildan 5000 tee: lifestyle front, S&S back
    al = {'Sapphire': 'Saphire', 'Heather Sapphire': 'Heather Saphire'}
    for c in DREAMS['tee']:
        f = find(f'{L.D2}/g5000', c, al)
        if f: add('g5000', c, 'tee', (f,), 'Front')
        elif ssfile('tee_5000', c, 'Front'): add('g5000', c, 'flat', ('tee_5000', c, 'Front'), 'Front')
        if ssfile('tee_5000', c, 'Back'): add('g5000', c, 'flat', ('tee_5000', c, 'Back'), 'Back')
    # Gildan 18000 crew: Drive 2048 front, else 1000 px repo blank, else S&S; S&S back
    al = {'Ash': 'Ash Grey', 'Dark Heather': 'Dark Heather Grey', 'Heather Sport Dark Navy': 'Heather Dark Navy', 'Navy': 'Navy Blue', 'Royal': 'Royal Blue'}
    for c in DREAMS['crew']:
        f = find(f'{L.D2}/g18000', c, al) or find(f'{R}/customizer/blanks/crew', c)
        if f: add('g18000', c, 'crew', (f,), 'Front')
        elif ssfile('crew_18000', c, 'Front'): add('g18000', c, 'flat', ('crew_18000', c, 'Front'), 'Front')
        if ssfile('crew_18000', c, 'Back'): add('g18000', c, 'flat', ('crew_18000', c, 'Back'), 'Back')
    # Gildan 18500 hoodie: model shots for Black/Ash, S&S flats otherwise
    al = {'Heather Sport Dark Navy': 'Heather Dark Navy', 'Heather Sport Royal': 'Heather Deep Royal', 'Forest Green': 'Forest'}
    for c in DREAMS['hoodie']:
        if c in ('Black', 'Ash'):
            add('g18500', c, 'hoodie_model', (c, 'Front'), 'Front'); add('g18500', c, 'hoodie_model', (c, 'Back'), 'Back')
            continue
        for v in ('Front', 'Back'):
            if ssfile('hoodie_18500', c, v, al): add('g18500', c, 'flat', ('hoodie_18500', al.get(c, c), v), v)
    # Bella+Canvas 3001: seated, else couch, else S&S front; S&S back
    for c in DREAMS['bc3001']:
        f = find(f'{R}/customizer/blanks/bc3001/seated', c)
        g = find(f'{R}/customizer/blanks/bc3001/couch', c)
        if f: add('bc3001', c, 'life', ('seated', os.path.basename(f)[:-4]), 'Front')
        elif g: add('bc3001', c, 'life', ('couch', os.path.basename(g)[:-4]), 'Front')
        elif ssfile('bc_3001', c, 'Front'): add('bc3001', c, 'flat', ('bc_3001', c, 'Front'), 'Front')
        if ssfile('bc_3001', c, 'Back'): add('bc3001', c, 'flat', ('bc_3001', c, 'Back'), 'Back')
    # Comfort Colors 1717: flatlay front, S&S back
    for c in DREAMS['cc1717']:
        f = find(f'{R}/customizer/blanks/cc1717/flatlay', c)
        if f: add('cc1717', c, 'life', ('flatlay', os.path.basename(f)[:-4]), 'Front')
        elif ssfile('cc_1717', c, 'Front'): add('cc1717', c, 'flat', ('cc_1717', c, 'Front'), 'Front')
        if ssfile('cc_1717', c, 'Back', {'Lagoon Blue': 'Lagoon Blue'}): add('cc1717', c, 'flat', ('cc_1717', c, 'Back'), 'Back')
    # Kids: S&S flats front + back
    for prod, set_ in (('youth_5000b', 'youth_tee_5000b'), ('toddler_3301t', 'toddler_tee_3301t'), ('onesie_4400', 'onesie_4400')):
        for c in KIDS[prod]:
            for v in ('Front', 'Back'):
                if ssfile(set_, c, v): add(prod, c, 'flat', (set_, c, v), v)
    return jobs, P

def run(job):
    kind, args, fn = job
    if os.path.exists(fn): return fn, 'cached'
    try:
        if kind == 'tee': a = L.tee_front_file(args[0])
        elif kind == 'crew': a = L.crew_front_file(args[0])
        elif kind == 'hoodie_model': a = L.hoodie_model(*args)
        elif kind == 'life': a = L.life(*args)
        else: a = L.flat(*args)
        L.save(a, fn); return fn, 'ok'
    except Exception as e:
        return fn, 'ERR ' + repr(e)

if __name__ == '__main__':
    jobs, P = plan()
    json.dump(P, open(f'{OUT}/plan.json', 'w'), indent=0)
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else None
    if only: jobs = [j for j in jobs if j[2].split('/')[-2] in only]
    print(len(jobs), 'images', {k: len(v) for k, v in P.items()}, flush=True)
    with Pool(2) as pool:
        for i, (fn, st) in enumerate(pool.imap_unordered(run, jobs)):
            if st.startswith('ERR') or i % 50 == 0: print(i, fn.split('designs/')[-1], st, flush=True)
    print('done')
