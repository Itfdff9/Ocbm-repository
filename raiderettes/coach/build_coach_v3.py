"""Pat-Med Varsity Raiderettes COACH -> clean hard-edged print file.
usage: python3 build3.py OUT.png CANVAS_PX DESIGN_HEIGHT_PX DPI
Classes: clear, black, white, red (206,3,15)."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as nd

out, CAN, DH, DPI = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
OLW, INL, RING = 11.0, 9.0, 6.0
PAD = 40
a = np.asarray(Image.open('src.png').convert('RGBA')).astype(np.float32)
H0, W0 = a.shape[:2]
a = np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)))
al = a[..., 3] / 255; rgb = a[..., :3]; r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
lum = rgb.mean(-1); sat = rgb.max(-1) - rgb.min(-1)
PAL = np.array([[0, 0, 0], [255, 255, 255], [225, 5, 12]], np.float32)
COLS = np.array([(0, 0, 0, 0), (0, 0, 0, 255), (255, 255, 255, 255), (200, 2, 16, 255), (250, 8, 8, 255)], np.uint8)
d2 = ((rgb[..., None, :] - PAL[None, None]) ** 2).sum(-1)
w = np.exp(-(d2 - d2.min(-1, keepdims=True)) / (2 * 40.0 ** 2)); w /= w.sum(-1, keepdims=True)
cov = np.concatenate([(1 - al)[..., None], w * al[..., None], np.zeros(al.shape + (1,))], -1).astype(np.float32)
# bright red lives in the PAT-MED outline (rows <545) and the dancer outline (1100-3400); deep red elsewhere
for y0, y1 in [(0, 545)]:
    cov[y0 + 40:y1 + 40, :, 4] = cov[y0 + 40:y1 + 40, :, 3]; cov[y0 + 40:y1 + 40, :, 3] = 0

def setcls(mask, k):
    cov[mask] = 0; cov[mask, k] = 1

Y = lambda y0, y1: (slice(y0 + PAD, y1 + PAD), slice(None))
red = (al > 0.5) & (r > g + 80) & (r > b + 60)
black = (al > 0.5) & (lum < 60)
white = (al > 0.5) & (lum > 200) & (sat < 60)

# ---- PAT-MED VARSITY (rows 0-545): solid black letters, see-through gap, red outline ----
S = Y(0, 545)
bk = black[S].copy()
holes = nd.binary_fill_holes(bk) & ~bk
lab, n = nd.label(holes); sz = nd.sum(holes, lab, range(1, n + 1))
letters = bk | np.isin(lab, 1 + np.where(sz < 1500)[0])          # shine smudges inside letters -> black
dl = nd.distance_transform_edt(~letters)
sub = cov[S]
sub[letters] = [0, 1, 0, 0, 0]
gap = (dl > 0) & (dl <= 9.5) & ~red[S]
sub[gap] = [1, 0, 0, 0, 0]                                           # white shine / half-clear ring -> see-through
stray_white = white[S] & ~letters & ~gap
sub[stray_white] = [1, 0, 0, 0, 0]
sub[..., 0] += sub[..., 2]; sub[..., 2] = 0                       # no white anywhere in PAT-MED VARSITY
cov[S] = sub

# ---- RAIDERETTES (590-1045) and COACH (3415-4096): white left in counters/notches -> clear ----
dred = nd.distance_transform_edt(~red)
for y0, y1 in [(590, 1045), (3415, 4096)]:
    S = Y(y0, y1)
    wf = white[S] & (dred[S] > 22)
    lab, n = nd.label(wf); sz = nd.sum(wf, lab, range(1, n + 1))
    trapped = np.isin(lab, 1 + np.where(sz > 250)[0])
    # grow over the soft edge / pink holes inside those openings, but never into black or red
    trapped = nd.binary_dilation(trapped, iterations=4) & ~black[S] & ~red[S] & (dred[S] > 14)
    trapped = nd.binary_fill_holes(trapped) & ~black[S] & ~red[S]
    sub = cov[S]; sub[trapped] = [1, 0, 0, 0, 0]
    # semi-transparent, non-dark pixels (pink flecks in the see-through gap) -> clear
    soft = (al[S] < 0.8) & (lum[S] > 70)
    sub[soft] = [1, 0, 0, 0, 0]
    cov[S] = sub

# COACH: see-through streak under the A's crossbar -> nearest red/white
box = (slice(3870 + PAD, 3915 + PAD), slice(1860 + PAD, 1995 + PAD))
legit = red | white
_, (iy, ix) = nd.distance_transform_edt(~legit, return_indices=True)
m = np.zeros(al.shape, bool); m[box] = (al[box] < 0.9) | ~(red[box] | white[box] | black[box])
setcls(m & red[iy, ix], 3); setcls(m & ~red[iy, ix], 2)

# S at the right edge: extend its black outline 3 px past the old canvas edge
rows = np.where(a[:, PAD + W0 - 1, 3] > 60)[0]
for yy in rows:
    if lum[yy, PAD + W0 - 1] < 60:
        cov[yy, PAD + W0:PAD + W0 + 3] = [0, 1, 0, 0, 0]

# ---- Dancer: no white anywhere; bright-red outline completed all the way around (mask-limited) ----
S = Y(1100, 3530)
bk = black[S]
lab, n = nd.label(nd.binary_opening(bk, iterations=2)); sz = nd.sum(bk, lab, range(1, n + 1))
core = lab == 1 + np.argmax(sz)
sil = nd.binary_dilation(core, iterations=2) & (bk | (al[S] > 0.5) & ~red[S])
sil = nd.binary_fill_holes(sil | core)
near = nd.binary_dilation(sil, iterations=30)
dsil = nd.distance_transform_edt(~sil)
sub = cov[S]
sub[near, 4] += sub[near, 3]; sub[near, 3] = 0                     # dancer outline = bright red
wd = near & (white[S] | ((al[S] > 0.3) & (lum[S] > 120) & ~red[S]))
_, (iy, ix) = nd.distance_transform_edt(wd | (al[S] < 0.3), return_indices=True)
src_red = red[S][iy, ix]
sub[wd & src_red] = [0, 0, 0, 0, 1]; sub[wd & ~src_red] = [0, 1, 0, 0, 0]
sub[near, 2] = 0                                                   # the dancer has no white at all
ring = (dsil > 0) & (dsil <= 13) & (sub.argmax(-1) == 0)
sub[ring] = [0, 0, 0, 0, 1]
cov[S] = sub

# ---- tiny see-through pinholes enclosed by the art -> surrounding color ----
cl = cov.argmax(-1)
pin = cl == 0
lab, n = nd.label(pin); sz = nd.sum(pin, lab, range(1, n + 1))
small = np.isin(lab, 1 + np.where(sz < 600)[0])
_, (iy, ix) = nd.distance_transform_edt(small, return_indices=True)
fk = cl[iy, ix]
for k in (1, 2, 3, 4):
    setcls(small & (fk == k), k)
# tiny red/white crumbs left in the RAIDERETTES / COACH see-through gaps -> clear
cl = cov.argmax(-1)
for y0, y1 in [(590, 1045), (3415, 4096)]:
    S = (slice(y0 + PAD, y1 + PAD), slice(None))
    for k in (2, 3):
        mk = cl[S] == k
        lab, n = nd.label(mk); sz = nd.sum(mk, lab, range(1, n + 1))
        crumbs = np.isin(lab, 1 + np.where(sz < 400)[0])
        sub = cov[S]; sub[crumbs] = [1, 0, 0, 0, 0]; cov[S] = sub

# ---- final pass on the letter bands: hairline white glare + tiny crumbs -> surrounding color ----
from scipy.ndimage import maximum_filter
cl = cov.argmax(-1)
for y0, y1 in [(590, 1045), (3415, 4096)]:
    S = (slice(y0 + PAD, y1 + PAD), slice(None))
    c = cl[S].copy()
    wm = c == 2
    dt = nd.distance_transform_edt(wm)
    thin = wm & (maximum_filter(dt, size=7) < 3.5)               # white lines under ~6 px wide = glare, not inline
    c[thin] = 255
    for k, lim in ((0, 150), (1, 400), (2, 1500), (3, 400)):    # small islands of any color -> surrounding color
        mk = c == k
        lab, n = nd.label(mk); sz = nd.sum(mk, lab, range(1, n + 1))
        c[np.isin(lab, 1 + np.where(sz < lim)[0])] = 255
    bad = c == 255
    _, (iy, ix) = nd.distance_transform_edt(bad, return_indices=True)
    c[bad] = c[iy, ix][bad]
    sub = cov[S]
    for k in range(5):
        sub[bad & (c == k)] = np.eye(5, dtype=np.float32)[k]
    cov[S] = sub

# ---- RAIDERETTES: complete the white inline (R, A, I had see-through gaps where the others are white) ----
cl = cov.argmax(-1)
S = (slice(590 + PAD, 1045 + PAD), slice(None))
c = cl[S]
clr = c == 0
lab, n = nd.label(clr)
edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
big = 1 + int(np.argmax(nd.sum(clr, lab, range(1, n + 1))))
background = np.isin(lab, list(edge_ids | {big}))
dred_c = nd.distance_transform_edt(c != 3)
inline = clr & ~background & (dred_c <= 14)
sub = cov[S]; sub[inline] = [0, 0, 1, 0, 0]; cov[S] = sub

# ---- RAIDERETTES + COACH: rebuild every letter from its black outline (varsity construction) ----
# inside each closed black outline: an even white inline, then solid red; openings with no red -> see-through
for (y0, y1, inl, olw) in [(590, 1045, 15.5, 17.0), (3415, 4096, 20.5, 18.0)]:
    S = (slice(y0 + PAD, y1 + PAD), slice(None))
    bk = (al[S] > 0.5) & (lum[S] < 60)
    if y0 == 590:                                    # the S at the old right edge: carry its outline 3 px further
        ext = np.zeros_like(bk); ext[:, PAD + W0:PAD + W0 + 3] = bk[:, PAD + W0 - 1:PAD + W0][:, [0, 0, 0]]
        bk |= ext
    bk = nd.binary_closing(bk, iterations=1)
    rd = red[S]
    lab, n = nd.label(~bk)
    edge_ids = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    sizes = nd.sum(np.ones_like(bk), lab, range(1, n + 1))
    redfrac = nd.sum(rd, lab, range(1, n + 1)) / np.maximum(sizes, 1)
    clearfrac = nd.sum(al[S] < 0.5, lab, range(1, n + 1)) / np.maximum(sizes, 1)
    letter = np.isin(lab, [i + 1 for i in range(n) if (i + 1) not in edge_ids and redfrac[i] > 0.3 and sizes[i] >= 3000])
    # small enclosed spots that were NOT see-through in the original = glare holes in the outline -> black
    # (real openings such as the A's triangle were see-through and stay open)
    pinholes = np.isin(lab, [i + 1 for i in range(n) if (i + 1) not in edge_ids and sizes[i] < 3000
                             and redfrac[i] <= 0.3 and not (clearfrac[i] > 0.4 and sizes[i] >= 150)])
    pinholes |= np.isin(lab, [i + 1 for i in range(n) if (i + 1) not in edge_ids and sizes[i] < 3000 and redfrac[i] > 0.3])
    bk = bk | pinholes
    # uniform construction: outer shape -> black outline (olw px) -> white inline (inl px) -> red
    keep = nd.binary_dilation(letter, iterations=int(olw) + 4)   # outline pieces that belong to a letter
    bkl = bk & keep
    background = np.isin(lab, list(edge_ids)) | ~np.isin(lab, [i + 1 for i in range(n)]) & ~bk
    # black islands that never touch the outside are the rings around letter openings (A, R, D, O counters)
    lb, nb = nd.label(bkl)
    touch = nd.maximum(nd.binary_dilation(np.isin(lab, list(edge_ids)), iterations=2), lb, range(1, nb + 1))
    islands = np.isin(lb, [j + 1 for j in range(nb) if not touch[j]])
    counter_area = nd.binary_fill_holes(islands)
    hole = nd.binary_erosion(counter_area, iterations=int(olw)) | (counter_area & (al[S] < 0.5) & ~bk)
    shape = (letter | bkl | counter_area) & ~hole
    shape = nd.binary_opening(shape, iterations=2)
    dout = nd.distance_transform_edt(shape)
    new_cl = np.zeros(bk.shape, np.uint8)            # clear
    new_cl[shape & (dout <= olw)] = 1
    new_cl[shape & (dout > olw) & (dout <= olw + inl)] = 2
    new_cl[shape & (dout > olw + inl)] = 3
    # drop specks of black outside the outlines (smaller than 300 px)
    lb, nb = nd.label(new_cl == 1); szb = nd.sum(new_cl == 1, lb, range(1, nb + 1))
    new_cl[np.isin(lb, 1 + np.where(szb < 300)[0])] = 0
    region = nd.binary_dilation(letter | bkl | counter_area | shape, iterations=4)
    sub = cov[S]; sub[region] = np.eye(5, dtype=np.float32)[new_cl[region]]; cov[S] = sub

# ---- PAT-MED VARSITY restyled to match RAIDERETTES (red, white inline, black outline) ----
sys.path.insert(0, '..'); from patmed_style import patmed_arc_even; import os
pm, info = patmed_arc_even(al, lum, 545, PAD, OLW, INL, RING, '../fonts/Graduate-Regular.ttf', band=(590, 1045), bold=9, widen_ref=1.25, spacing=os.environ.get('SPACING', 'optical'))
print('PAT-MED letters', info['n'], 'openings', info['counters'])
Hp = pm.shape[0]
cur = cov[:Hp].argmax(-1)
clash = (pm > 0) & (cur > 0); clash[:545 + PAD] = False
print('overlap with the art below:', int(clash.sum()))
cov[0:545 + PAD] = np.eye(5, dtype=np.float32)[0]
m = pm > 0
sub = cov[:Hp]; sub[m] = np.eye(5, dtype=np.float32)[pm[m]]; cov[:Hp] = sub
import pickle; pickle.dump(info, open(out + '.arc.pkl', 'wb'))

# ---- crop, scale (argmax of smoothed class maps), center on the canvas ----
cl = cov.argmax(-1)
ys, xs = np.where(cl > 0)
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
import os as _os; print('CROPBOX', y0, y1, x0, x1)
if _os.environ.get('CROP'): y0, y1, x0, x1 = map(int, _os.environ['CROP'].split(','))
cov = cov[y0:y1, x0:x1]
SH, SW = cov.shape[:2]
s = DH / SH; AW, AH = int(round(SW * s)), DH
if AW > CAN: s = CAN / SW; AW, AH = CAN, int(round(SH * s))
best = None
for k in range(5):
    ch = nd.gaussian_filter(cov[..., k], 0.6)
    up = np.asarray(Image.fromarray(ch.astype(np.float32), 'F').resize((AW, AH), Image.BICUBIC))
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
