"""North Shore Elite (blue arch + white Elite) -> clean, hard-edged print file.
usage: python3 build2.py OUT.png WIDTH_PX HEIGHT_PX [stretch|fit]
Classes: clear, black, white, letter blue (0,65,222), Elite blue (0,48,198)."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as nd

out, TW, TH = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
mode = sys.argv[4] if len(sys.argv) > 4 else 'stretch'
PAD = 40
a = np.asarray(Image.open('src.png').convert('RGBA')).astype(np.float32)
a = np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)))
al = a[..., 3] / 255; rgb = a[..., :3]; r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
lum = rgb.mean(-1)
PAL = np.array([[0, 0, 0], [255, 255, 255], [0, 65, 222], [0, 48, 198]], np.float32)
COLS = [(0, 0, 0, 0), (0, 0, 0, 255), (255, 255, 255, 255), (0, 65, 222, 255), (0, 48, 198, 255)]
d2 = ((rgb[..., None, :] - PAL[None, None]) ** 2).sum(-1)
w = np.exp(-(d2 - d2.min(-1, keepdims=True)) / (2 * 14.0 ** 2)); w /= w.sum(-1, keepdims=True)
cov = np.concatenate([(1 - al)[..., None], w * al[..., None]], -1).astype(np.float32)

def setcls(mask, k):
    cov[mask] = 0; cov[mask, k] = 1

blue = (al > 0.5) & (b > r + 60)
blueL = blue & (g >= 57) & (b >= 211)
clear = al < 0.24
dblue = nd.distance_transform_edt(~blue)

# 1) background trapped in narrow gaps/notches between letters -> clear
light = (al > 0.24) & (lum > 105) & ~blue
cand = light & (dblue > 34)
lab, n = nd.label(cand); sz = nd.sum(cand, lab, range(1, n + 1))
touch = nd.maximum(nd.binary_dilation(clear, iterations=3), lab, range(1, n + 1))
keep = [i + 1 for i in range(n) if touch[i] and 15 < sz[i] < 20000]
trapped = nd.binary_dilation(np.isin(lab, keep), iterations=2) & ~blue & ~((al > 0.5) & (lum < 60))
setcls(trapped, 0)

# 2) warm/tan off-palette debris -> nearest legit black or clear
legit = ((al > 0.9) & (lum < 50)) | clear
_, (iy, ix) = nd.distance_transform_edt(~legit, return_indices=True)
src_black = al[iy, ix] > 0.5
tan = (al > 0.24) & (r > b + 12) & (lum > 110) & (lum < 248)
tan = nd.binary_dilation(tan, iterations=2) & ~blue & ~((al > 0.9) & (lum > 245)) & ~((al > 0.9) & (lum < 50))
setcls(tan & src_black, 1); setcls(tan & ~src_black, 0)
# the tan sliver above Elite's 'l' is the tail of a see-through gap -> clear (keep real black/blue)
gx = np.s_[1180 + PAD:1250 + PAD, 1700 + PAD:1880 + PAD]
sub_al, sub_lum, sub_blue = al[gx], lum[gx], blue[gx]
gap = (sub_al > 0.24) & (sub_lum > 90) & ~sub_blue
gap = nd.binary_dilation(gap, iterations=2) & ~sub_blue & ~((sub_al > 0.9) & (sub_lum < 50))
sub = cov[gx]; sub[gap] = 0; sub[gap, 0] = 1; cov[gx] = sub

# 3) distress specks inside the blue letters -> kept, as crisp opaque white
holes = nd.binary_fill_holes(blueL) & ~blueL
lab, n = nd.label(holes); sz = nd.sum(holes, lab, range(1, n + 1))
inbody = np.isin(lab, 1 + np.where(sz < 3000)[0])
# keep only the bright core of each speck (same footprint as the visible speck), rest of the fringe -> letter blue
specks = inbody & (lum > 150)
setcls(inbody & ~specks, 3)
setcls(specks, 2)

# 4) top of the H was cut by the canvas edge: rebuild the black outline above the white inline
H0 = a.shape[0]
cols = np.where(a[PAD, :, 3] > 60)[0]
white = (al > 0.5) & (lum > 200) & ~blue
for x in cols:
    yw = np.where(white[PAD:PAD + 80, x])[0]
    if len(yw):
        top = PAD + yw[0]
        cov[max(0, top - 21):top, x] = 0; cov[max(0, top - 21):top, x, 1] = 1

# tiny see-through pinholes enclosed by the art -> filled with the surrounding color
cl = cov.argmax(-1)
pin = cl == 0
lab, n = nd.label(pin); sz = nd.sum(pin, lab, range(1, n + 1))
small = np.isin(lab, 1 + np.where(sz < 150)[0])
_, (iy, ix) = nd.distance_transform_edt(small, return_indices=True)
fillk = cl[iy, ix]
for k in range(1, 5):
    setcls(small & (fillk == k), k)

# crop to content
cl = cov.argmax(-1)
ys, xs = np.where(cl > 0)
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
cov = cov[y0:y1, x0:x1]
SH, SW = cov.shape[:2]
if mode == 'stretch':
    AW, AH = TW, TH
else:
    s = min(TW / SW, TH / SH); AW, AH = int(round(SW * s)), int(round(SH * s))

best = None; arg = None
for k in range(5):
    ch = nd.gaussian_filter(cov[..., k], 0.5)
    up = np.asarray(Image.fromarray(ch.astype(np.float32), 'F').resize((AW, AH), Image.BICUBIC))
    if best is None:
        best = up; arg = np.zeros(up.shape, np.uint8)
    else:
        m = up > best; best = np.where(m, up, best); arg[m] = k
    del up
palette = np.array(COLS, np.uint8)
img = palette[arg]
canvas = np.zeros((TH, TW, 4), np.uint8)
oy, ox = (TH - AH) // 2, (TW - AW) // 2
canvas[oy:oy + AH, ox:ox + AW] = img
Image.fromarray(canvas).save(out, dpi=(300, 300), optimize=True)
print(out, canvas.shape, 'art', AW, AH, 'trapped px', int(trapped.sum()), 'tan px', int(tan.sum()), 'specks px', int(specks.sum()))
