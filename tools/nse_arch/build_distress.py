"""North Shore Elite — distressed blue NORTH SHORE (white keyline) + black script Elite with white trim
(Brian's upload 2026-10-18) -> 48 x 10 in at 300 DPI, transparent, hard alpha, flat colours.
Nothing is redrawn: the upload is enlarged and its edges made crisp; the distress marks stay white.

usage: python3 build_distress.py SRC.png RG_X4.png OUT.png TW TH [stretch|fit]
  RG_X4.png = Real-ESRGAN x4plus-anime of SRC flattened over grey (128)
 1. alpha: upload alpha enlarged 4x (cubic); colour: AI-enlarged 4x
 2. every pixel split into clear / black / white / blue coverage (alpha x barycentric colour unmix, so the
    black-to-white edge of Elite never reads as blue)
 3. dust floating in the clear area and pinholes under ~1 src px -> surrounding colour (distress marks are
    far larger and are kept)
 4. each colour smoothed + enlarged on its own; every output pixel takes the strongest colour
    -> exactly 4 values, alpha 0/255 (the upload was 99% opaque everywhere, i.e. faintly see-through)
"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS = None

BLUE = np.array([0, 57, 199], np.float32); WHITE = np.array([252, 252, 252], np.float32)
OUT = np.array([[0, 0, 0, 0], [0, 0, 0, 255], [255, 255, 255, 255], [0, 57, 199, 255]], np.uint8)
CLEAR, BLK, WHT, BLU = range(4)
S = 4

src, rgp, out, TW, TH = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
mode = sys.argv[6] if len(sys.argv) > 6 else 'stretch'
A = np.asarray(Image.open(src).convert('RGBA'))[..., 3].astype(np.float32) / 255.0
A = np.clip(A / 0.99, 0, 1)                                   # 253/255 inside the art = solid
rgb = np.asarray(Image.open(rgp).convert('RGB')).astype(np.float32)
H, W = rgb.shape[:2]
al = np.clip(cv2.resize(A, (W, H), interpolation=cv2.INTER_CUBIC), 0, 1)

# barycentric unmix over black / white / blue
M = np.stack([WHITE, BLUE], 1)
ab = np.clip(rgb.reshape(-1, 3) @ np.linalg.pinv(M).T, 0, None)
s = ab.sum(1, keepdims=True); ab = np.where(s > 1, ab / np.maximum(s, 1e-6), ab)
wW = ab[:, 0].reshape(H, W); wB = ab[:, 1].reshape(H, W); wK = np.clip(1 - wW - wB, 0, 1)
cov = np.stack([1 - al, al * wK, al * wW, al * wB], -1).astype(np.float32)
del ab, wW, wB, wK

# dust / pinholes (under ~1.2 src px^2 = 20 px at 4x)
cls = cov.argmax(-1).astype(np.uint8)
fixed = np.zeros(cls.shape, bool)
for k in range(4):
    lb, n = nd.label(cls == k)
    if n == 0: continue
    cnt = np.bincount(lb.ravel()); small = cnt < 20; small[0] = False
    sm = small[lb]
    if sm.any():
        _, (iy, ix) = nd.distance_transform_edt(sm, return_indices=True)
        cls[sm] = cls[iy, ix][sm]; fixed |= sm
cov[fixed] = 0; cov[fixed, cls[fixed]] = 1

ys, xs = np.where(cls != CLEAR); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
pad = 8; Y0, X0 = max(0, y0 - pad), max(0, x0 - pad)
cov = cov[Y0:y1 + pad, X0:x1 + pad]; cy0, cx0, ch_, cw_ = y0 - Y0, x0 - X0, y1 - y0, x1 - x0
if mode == 'stretch': AW, AH = TW, TH
else:
    k = min(TW / cw_, TH / ch_); AW, AH = int(round(cw_ * k)), int(round(ch_ * k))
sx, sy = AW / cw_, AH / ch_
PW, PH = int(round(cov.shape[1] * sx)), int(round(cov.shape[0] * sy))
best = None
for c in range(4):
    ch = cv2.GaussianBlur(cov[..., c], (0, 0), 1.0)
    up = cv2.resize(ch, (PW, PH), interpolation=cv2.INTER_CUBIC)
    if best is None: best = up; arg = np.zeros(up.shape, np.uint8)
    else:
        m = up > best; best = np.where(m, up, best); arg[m] = c
    del up
oy, ox = int(round(cy0 * sy)), int(round(cx0 * sx))
arg = arg[oy:oy + AH, ox:ox + AW]
if arg.shape != (AH, AW): arg = np.pad(arg, ((0, AH - arg.shape[0]), (0, AW - arg.shape[1])))
canvas = np.zeros((TH, TW), np.uint8); ty, tx = (TH - AH) // 2, (TW - AW) // 2
canvas[ty:ty + AH, tx:tx + AW] = arg
Image.fromarray(OUT[canvas], 'RGBA').save(out, dpi=(300, 300), optimize=True)
u, c = np.unique(canvas, return_counts=True)
print(out, (TW, TH), mode, 'art', (AW, AH), {['clear', 'black', 'white', 'blue'][i]: round(n / canvas.size * 100, 2) for i, n in zip(u, c)},
      'dust/pinholes fixed', int(fixed.sum()))
