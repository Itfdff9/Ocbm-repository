"""North Shore Elite — blue arch + white script Elite with blue outline (Brian's vertical upload 2026-10-16,
already cut out) -> 48 x 10 in sleeve-to-sleeve print, hard alpha, flat colours (white + the upload's blue).

usage: python3 build_white_elite.py SOURCE.webp OUT.png TW TH [stretch|fit]
 1. turn the vertical upload 90 deg clockwise so NORTH SHORE reads left to right
 2. split every pixel into clear / white / blue coverage (alpha x colour unmix); leftover black from the
    original background (pockets in the N, H, A notches etc.) -> clear, like the rest of each pocket
 3. specks / pinholes smaller than ~3 px -> surrounding colour
 4. each colour's coverage smoothed and enlarged on its own; every output pixel takes the strongest:
    crisp smooth edges, exactly 3 values (clear, white, blue 8,49,175), alpha 0/255
"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd

Image.MAX_IMAGE_PIXELS = None
BLUE = np.array([8, 49, 175], np.float32)          # measured: solid blue of the upload (uniform, p5-p95 +-4)
WHITE = np.array([253, 253, 254], np.float32)
CLEAR, WHT, BLU = 0, 1, 2
OUT = np.array([[0, 0, 0, 0], [255, 255, 255, 255], [8, 49, 175, 255]], np.uint8)

src, out, TW, TH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
mode = sys.argv[5] if len(sys.argv) > 5 else 'stretch'
im = Image.open(src).convert('RGBA').transpose(Image.Transpose.ROTATE_270)
a = np.asarray(im).astype(np.float32)
al = a[..., 3] / 255.0; rgb = a[..., :3]

# barycentric unmix over black / white / blue
M = np.stack([WHITE, BLUE], 1)
ab = np.clip(rgb.reshape(-1, 3) @ np.linalg.pinv(M).T, 0, None)
s = ab.sum(1, keepdims=True); ab = np.where(s > 1, ab / np.maximum(s, 1e-6), ab)
wW = ab[:, 0].reshape(al.shape); wB = ab[:, 1].reshape(al.shape); wK = np.clip(1 - wW - wB, 0, 1)
cov = np.stack([(1 - al) + al * wK, al * wW, al * wB], -1).astype(np.float32)

# specks / pinholes: islands of any class under 3 x 3 px -> surrounding class (one-hot there)
cls = cov.argmax(-1).astype(np.uint8)
fixed = np.zeros(cls.shape, bool)
for k in (CLEAR, WHT, BLU):
    lb, n = nd.label(cls == k)
    if n == 0: continue
    cnt = np.bincount(lb.ravel()); small = cnt < 9; small[0] = False
    sm = small[lb]
    if sm.any():
        _, (iy, ix) = nd.distance_transform_edt(sm, return_indices=True)
        cls[sm] = cls[iy, ix][sm]; fixed |= sm
cov[fixed] = 0; cov[fixed, cls[fixed]] = 1

# crop to the art, render
ys, xs = np.where(cls != CLEAR)
y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
pad = 6
Y0, X0 = max(0, y0 - pad), max(0, x0 - pad)
cov = cov[Y0:y1 + pad, X0:x1 + pad]
cy0, cx0, ch_, cw_ = y0 - Y0, x0 - X0, y1 - y0, x1 - x0
if mode == 'stretch':
    AW, AH = TW, TH
else:
    k = min(TW / cw_, TH / ch_); AW, AH = int(round(cw_ * k)), int(round(ch_ * k))
sx, sy = AW / cw_, AH / ch_
PW, PH = int(round(cov.shape[1] * sx)), int(round(cov.shape[0] * sy))
best = None
for c in range(3):
    ch = cv2.GaussianBlur(cov[..., c], (0, 0), 1.2)
    up = cv2.resize(ch, (PW, PH), interpolation=cv2.INTER_CUBIC)
    if best is None: best = up; arg = np.zeros(up.shape, np.uint8)
    else:
        m = up > best; best = np.where(m, up, best); arg[m] = c
    del up
oy, ox = int(round(cy0 * sy)), int(round(cx0 * sx))
arg = arg[oy:oy + AH, ox:ox + AW]
if arg.shape != (AH, AW): arg = np.pad(arg, ((0, AH - arg.shape[0]), (0, AW - arg.shape[1])))
canvas = np.zeros((TH, TW), np.uint8)
ty, tx = (TH - AH) // 2, (TW - AW) // 2
canvas[ty:ty + AH, tx:tx + AW] = arg
Image.fromarray(OUT[canvas], 'RGBA').save(out, dpi=(300, 300), optimize=True)
u, c = np.unique(canvas, return_counts=True)
print(out, (TW, TH), mode, 'art', (AW, AH), {['clear', 'white', 'blue'][i]: round(n / canvas.size * 100, 2) for i, n in zip(u, c)},
      'specks fixed', int(fixed.sum()))
