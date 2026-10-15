"""Pat-Med Varsity Raiderettes SENIOR 2027 (black-dancer version) -> clean hard-edged print file.
usage: python3 build4.py OUT.png CANVAS_PX DESIGN_HEIGHT_PX DPI
Every varsity letter (RAIDERETTES, SENIOR, 2027) gets the COACH construction:
black outline -> white inline -> deep red fill. PAT-MED VARSITY and the dancer keep black + bright red outline.
Classes: 0 clear, 1 black, 2 white, 3 deep red (letters), 4 bright red (PAT-MED + dancer outlines)."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as nd

out, CAN, DH, DPI = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
OLW, INL, RING = 10.0, 8.0, 5.0
PAD = 40
a = np.asarray(Image.open('src.png').convert('RGBA')).astype(np.float32)
H0, W0 = a.shape[:2]
a = np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)))
al = a[..., 3] / 255; rgb = a[..., :3]; r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
lum = rgb.mean(-1); sat = rgb.max(-1) - rgb.min(-1)
COLS = np.array([(0, 0, 0, 0), (0, 0, 0, 255), (255, 255, 255, 255), (200, 2, 16, 255), (250, 8, 8, 255)], np.uint8)
EYE = np.eye(5, dtype=np.float32)
red = (al > 0.5) & (r > g + 80) & (r > b + 60)
black = (al > 0.5) & (lum < 60)
white = (al > 0.5) & (lum > 200) & (sat < 60)

# starting class map: nearest of clear / black / white / bright red
cls = np.zeros(al.shape, np.uint8)
cls[al > 0.5] = 1
cls[white] = 2
cls[red] = 4
Y = lambda y0, y1: (slice(y0 + PAD, y1 + PAD), slice(None))

# ---- PAT-MED VARSITY (rows 0-480): solid black letters, even see-through gap, bright red outline ----
S = Y(0, 480)
bk = black[S]
holes = nd.binary_fill_holes(bk) & ~bk
lab, n = nd.label(holes); sz = nd.sum(holes, lab, range(1, n + 1))
letters = bk | np.isin(lab, 1 + np.where(sz < 1500)[0])          # shine smudges inside letters -> black
dl = nd.distance_transform_edt(~letters)
c = cls[S]
c[letters] = 1
c[(dl > 0) & (dl <= 9.5) & ~red[S]] = 0                          # gap between letter and red outline -> clear
c[(c == 2)] = 0                                                  # no white anywhere in PAT-MED VARSITY
c[(c == 1) & ~letters] = 0                                       # dark smudges outside the letters
# the V's red outline touches the top edge: carry it 3 px upward
toprow = c[PAD] == 4
for k in range(1, 4):
    c[PAD - k][toprow] = 4
cls[S] = c

# ---- Dancer: black silhouette + bright red outline, closed all the way around, no white ----
S = Y(960, 3560)
bk = black[S]
lab, n = nd.label(nd.binary_opening(bk, iterations=3)); sz = nd.sum(bk, lab, range(1, n + 1))
core = lab == 1 + np.argmax(sz)
sil = nd.binary_dilation(core, iterations=3) & (bk | (al[S] > 0.5) & ~red[S] & ~white[S])
sil = nd.binary_fill_holes(sil | core)
dsil = nd.distance_transform_edt(~sil)
near = dsil <= 30
c = cls[S]
c[near & sil] = 1
ring = (dsil > 0) & (dsil <= 13)
c[near & ~sil & ~ring] = np.where(c[near & ~sil & ~ring] == 4, 4, 0)   # keep only true outline red outside the ring
c[near & ~sil & ~ring & (dsil > 16)] = 0
c[ring] = 4
cls[S] = c
DANCER = np.zeros(al.shape, bool); DANCER[S] = nd.binary_dilation(sil, iterations=16)

# ---- Varsity letters: rebuild each from its black outline ----
HOLES = []
def rebuild(y0, y1, olw, inl, need_red=True, extend_right=False):
    S = Y(y0, y1)
    bk = (al[S] > 0.5) & (lum[S] < 60) & ~DANCER[S]
    if extend_right:                                   # outline cut by the image's right edge: carry it 3 px further
        ext = np.zeros_like(bk); ext[:, PAD + W0:PAD + W0 + 3] = bk[:, [PAD + W0 - 1] * 3]
        bk |= ext
    bk = nd.binary_closing(bk, iterations=1)
    lab, n = nd.label(~bk)
    edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    ids = np.arange(1, n + 1)
    sizes = nd.sum(np.ones_like(bk), lab, ids)
    redfrac = nd.sum(red[S], lab, ids) / np.maximum(sizes, 1)
    clearfrac = nd.sum(al[S] < 0.5, lab, ids) / np.maximum(sizes, 1)
    inner = [i for i in ids if i not in edge_ids and not DANCER[S][lab == i].any()]
    # black islands that never touch the outside = rings around letter openings (counters)
    lb, nb = nd.label(bk)
    outside = nd.binary_dilation(np.isin(lab, list(edge_ids)), iterations=2)
    touch = nd.maximum(outside, lb, range(1, nb + 1))
    islands = np.isin(lb, [j + 1 for j in range(nb) if not touch[j]])
    counter_area = nd.binary_fill_holes(islands)
    if need_red:
        letter = np.isin(lab, [i for i in inner if redfrac[i - 1] > 0.3 and sizes[i - 1] >= 3000])
    else:                                              # hollow outline digits: every big enclosed body that is not a counter
        letter = np.isin(lab, [i for i in inner if sizes[i - 1] >= 3000]) & ~counter_area
    # small enclosed spots that were not see-through = glare holes in the outline -> black
    pin = np.isin(lab, [i for i in inner if sizes[i - 1] < 3000 and
                        (redfrac[i - 1] > 0.3 or not (clearfrac[i - 1] > 0.4 and sizes[i - 1] >= 150))])
    bk = bk | pin
    keep = nd.binary_dilation(letter, iterations=int(olw) + 6)
    bkl = bk & keep
    # the real opening = everything inside a counter ring that is not the ring itself (white glare or see-through)
    hole = nd.binary_opening(counter_area & ~bk, iterations=2)
    # outer silhouette (letter + its outline), counters keep their original ring shape (smoothed)
    lc, nc = nd.label(counter_area)                         # whole counters that belong to a letter
    hit = nd.maximum(keep, lc, range(1, nc + 1))
    ca = np.isin(lc, [j + 1 for j in range(nc) if hit[j]])
    ca = nd.binary_closing(nd.binary_opening(ca, iterations=2), iterations=2)
    hole = nd.binary_closing(nd.binary_opening(hole & ca, iterations=2), iterations=2)
    ring = ca & ~hole
    shape = nd.binary_opening(letter | bkl | ca, iterations=2)
    fh = nd.binary_fill_holes(shape) & ~shape              # only fill tiny pinholes, never pockets or bays
    lf, nf = nd.label(fh); szf = nd.sum(fh, lf, range(1, nf + 1))
    shape |= np.isin(lf, 1 + np.where(szf < 400)[0])
    d_out = nd.distance_transform_edt(shape)                 # distance to the outside
    d_cnt = nd.distance_transform_edt(~ca)                   # distance to a counter ring
    new = np.zeros(bk.shape, np.uint8)
    body = shape & ~ca
    new[body] = 3
    new[body & ((d_out <= olw + inl) | (d_cnt <= inl))] = 2
    new[shape & (d_out <= olw)] = 1
    new[ring] = 1
    new[hole] = 0
    lb, nb = nd.label(new == 1); szb = nd.sum(new == 1, lb, range(1, nb + 1))
    new[np.isin(lb, 1 + np.where(szb < 300)[0])] = 0
    region = ~DANCER[S]                                # the whole band is redrawn (clears leftover white in E/R pockets)
    HOLES.append((S, hole))
    c = cls[S]; c[region] = new[region]; cls[S] = c
    return int(letter.sum() > 0), int(nd.label(letter)[1])

print('RAIDERETTES', rebuild(505, 920, 17.0, 13.5, extend_right=True))
print('SENIOR', rebuild(3010, 3550, 17.5, 16.0))
print('2027', rebuild(3575, H0, 17.5, 16.0, need_red=False))

# ---- tiny see-through pinholes enclosed by the art, and tiny crumbs of any color -> surrounding color ----
for k, lim in ((0, 600), (1, 200), (2, 200), (3, 200), (4, 120)):
    mk = cls == k
    lab, n = nd.label(mk); sz = nd.sum(mk, lab, range(1, n + 1))
    small = np.isin(lab, 1 + np.where(sz < lim)[0])
    if k == 0:                                         # only holes that are fully enclosed (not the open background)
        edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        small &= ~np.isin(lab, list(edge_ids))
        for Sh, hl in HOLES:                           # never fill a real letter opening
            m = np.zeros(cls.shape, bool); m[Sh] = hl; small &= ~m
    if small.any():
        _, (iy, ix) = nd.distance_transform_edt(small, return_indices=True)
        cls[small] = cls[iy, ix][small]

# ---- PAT-MED VARSITY restyled to match RAIDERETTES (red, white inline, black outline) ----
sys.path.insert(0, '..'); from patmed_style import patmed_arc_even; import os
pm, info = patmed_arc_even(al, lum, 480, PAD, OLW, INL, RING, '../fonts/Graduate-Regular.ttf', band=(505, 920), bold=8, widen_ref=1.25, spacing=os.environ.get('SPACING', 'optical'))
print('PAT-MED letters', info['n'], 'openings', info['counters'])
Hp = pm.shape[0]
clash = (pm > 0) & (cls[:Hp] > 0); clash[:480 + PAD] = False
print('overlap with the art below:', int(clash.sum()))
cls[0:480 + PAD] = 0
m = pm > 0
sub = cls[:Hp]; sub[m] = pm[m]; cls[:Hp] = sub
import pickle; pickle.dump(info, open(out + '.arc.pkl', 'wb'))

# ---- crop, scale (smoothed one-hot maps), center on the canvas ----
ys, xs = np.where(cls > 0)
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
import os as _os; print('CROPBOX', y0, y1, x0, x1)
if _os.environ.get('CROP'): y0, y1, x0, x1 = map(int, _os.environ['CROP'].split(','))
cls = cls[y0:y1, x0:x1]
SH, SW = cls.shape
s = DH / SH; AW, AH = int(round(SW * s)), DH
if AW > CAN: s = CAN / SW; AW, AH = CAN, int(round(SH * s))
best = None
for k in range(5):
    ch = nd.gaussian_filter((cls == k).astype(np.float32), 0.6)
    up = np.asarray(Image.fromarray(ch, 'F').resize((AW, AH), Image.BICUBIC))
    if best is None:
        best = up; arg = np.zeros(up.shape, np.uint8)
    else:
        mm = up > best; best = np.where(mm, up, best); arg[mm] = k
    del up
canvas = np.zeros((CAN, CAN, 4), np.uint8)
oy, ox = (CAN - AH) // 2, (CAN - AW) // 2
canvas[oy:oy + AH, ox:ox + AW] = COLS[arg]
Image.fromarray(canvas).save(out, dpi=(DPI, DPI), optimize=True)
print(out, canvas.shape, 'art', AW, AH, 'in', round(AW / DPI, 2), 'x', round(AH / DPI, 2))
