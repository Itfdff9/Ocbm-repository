"""Rebuild the North Shore Elite arch at any size with crisp hard edges.
usage: python3 build.py OUT.png WIDTH_PX HEIGHT_PX MODE [preview]
MODE: stretch (art fills the canvas) | fit (art keeps its shape, centered)
Layers (bottom->top): black arch letters, blue outer edge (only over clear background), white outline (20 src px), blue fill."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as nd

out, TW, TH, mode = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
PAD = 40
cov = np.load('cov.npy')                      # clear, black, white, blue (padded source, cleaned)
a = np.asarray(Image.open('src.png').convert('RGBA')).astype(np.float32)
a = np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)))
al = a[..., 3] / 255; rgb = a[..., :3]; r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
bluecls = (al > 0.5) & (b > r + 40) & (b > g + 30)
fill = nd.binary_opening(bluecls, structure=np.ones((3, 3)), iterations=9)
black = cov[..., 1] > 0.5
# drop tiny blue fill fragments (stray blob under the 'e' outline)
lab_f, nf = nd.label(fill); szf = nd.sum(fill, lab_f, range(1, nf + 1))
fill = np.isin(lab_f, 1 + np.where(szf >= 800)[0])
# restore the NORTH 'H' crossbar (matched to SHORE's H: 39%-56% of letter height, flat at the arch apex)
black[396 + PAD:564 + PAD, 1800 + PAD:1892 + PAD] = True
d = nd.distance_transform_edt(~fill)
elite_zone = d <= 20.5                         # fill + white outline footprint at source
black = black & ~nd.binary_erosion(elite_zone, iterations=1) | (black & elite_zone)
# Force the two debris pockets (i-t, t-e) to solid black under the outline
for (x0, y0, x1, y1) in [(2165, 615, 2345, 765), (2375, 585, 2610, 765)]:
    sl = (slice(y0 + PAD, y1 + PAD), slice(x0 + PAD, x1 + PAD))
    black[sl] |= ~fill[sl] & (d[sl] > 0)
# small specks: remove tiny black islands / fill tiny holes in letters
black = nd.binary_opening(black, iterations=1)
lab, n = nd.label(black); sz = nd.sum(black, lab, range(1, n + 1))
black = np.isin(lab, 1 + np.where(sz >= 400)[0])
holes = nd.binary_fill_holes(black) & ~black
lab, n = nd.label(holes); sz = nd.sum(holes, lab, range(1, n + 1))
black |= np.isin(lab, 1 + np.where(sz < 400)[0])

# crop to content
ys, xs = np.where(black | elite_zone)
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
black, fillm = black[y0:y1, x0:x1], fill[y0:y1, x0:x1]
SH, SW = black.shape

if mode == 'stretch':
    AW, AH = TW, TH
else:
    s = min(TW / SW, TH / SH); AW, AH = int(round(SW * s)), int(round(SH * s))
sx, sy = AW / SW, AH / SH
sgeo = (sx * sy) ** 0.5                         # outline widths scale with the geometric mean

def up(mask):
    f = nd.gaussian_filter(mask.astype(np.float32), 1.2)
    im = Image.fromarray((f * 255).astype(np.uint8)).resize((AW, AH), Image.BICUBIC)
    return np.asarray(im) >= 128

B = up(black); F = up(fillm)
D = nd.distance_transform_edt(~F)
WHITE = (D > 0) & (D <= 20 * sgeo)
EDGE = (D > 20 * sgeo) & (D <= 31 * sgeo) & ~B
img = np.zeros((AH, AW, 4), np.uint8)
img[B] = (0, 0, 0, 255)
img[EDGE] = (0, 40, 160, 255)
img[WHITE] = (255, 255, 255, 255)
img[F] = (0, 40, 160, 255)
canvas = np.zeros((TH, TW, 4), np.uint8)
oy, ox = (TH - AH) // 2, (TW - AW) // 2
canvas[oy:oy + AH, ox:ox + AW] = img
Image.fromarray(canvas).save(out, dpi=(300, 300), optimize=True)
print(out, canvas.shape, 'art', AW, AH, 'alpha values', np.unique(canvas[..., 3]))
