"""NORTH SHORE + black script 'Elite' shoulder print - cleanup/rebuild of Brian's upload (2026-10-14).
Letters: blue fill from the upload (alpha>=128, distress specks kept), clean white keyline rebuilt at the
upload's own spacing (24-34 px from the fill, source scale), transparent gap kept.
Fixes: H crossbar restored; dark bar/spike in H, stray dark lines in R removed; black wedges in Elite opened;
t crossbar trimmed so the i has one dot; white keyline added around Elite; specks removed; hard alpha.
usage: python3 fix_shoulder.py SRC OUT.png prop|stretch"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
src, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
a = np.array(Image.open(src).convert("RGBA")).astype(np.float32); A = a[..., 3]; rgb = a[..., :3]
H, W = A.shape; mx = rgb.max(2); mn = rgb.min(2)
# ---- letter fill
F = (A >= 128) & (mx >= 80)
lab, _ = nd.label(F); sz = np.bincount(lab.ravel()); sz[0] = 0; F = sz[lab] >= 300
holes = nd.binary_fill_holes(F) & ~F; hl, _ = nd.label(holes); hs = np.bincount(hl.ravel()); hs[0] = 0
F = F | (holes & (hs[hl] < 1500))
# H crossbar
cb = np.zeros((H, W), np.uint8)
# distress specks (white inside the fill)
specks = (mn > 125) & (A > 0) & nd.binary_erosion(F, iterations=8)
sl, _ = nd.label(specks); ss = np.bincount(sl.ravel()); ss[0] = 0; specks &= ss[sl] < 400
# ---- Elite
K = (mx < 80) & (A >= 128)
lab, _ = nd.label(K); sz = np.bincount(lab.ravel()); sz[0] = 0
main = np.argmax(sz); dot = lab[550, 2168]
K = (lab == main) | (lab == dot)
# trim t crossbar's left end (it read as a second i dot) and round it off
yy, xx = np.mgrid[0:H, 0:W]
cut = (xx >= 2175) & (xx < 2268) & (yy >= 480) & (yy <= 565) & (lab != dot)
col = np.where(K[480:560, 2272])[0] + 480; yc = (col.min() + col.max()) / 2; r = (col.max() - col.min()) / 2
K &= ~cut
K |= ((xx - 2270) ** 2 + (yy - yc) ** 2 <= r * r) & (xx < 2272) & (xx > 2230)
# ---- smooth + scale
ys, xs = np.where(nd.binary_dilation(F, iterations=36) | nd.binary_dilation(K, iterations=14))
bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
CW, CH, M = 14400, 3000, 30
if mode == "prop":
    s = min((CW - 2 * M) / (bx1 - bx0), (CH - 2 * M) / (by1 - by0)); sx = sy = s
else:
    sx = (CW - 2 * M) / (bx1 - bx0); sy = (CH - 2 * M) / (by1 - by0)
tw, th = int(round((bx1 - bx0) * sx)), int(round((by1 - by0) * sy))
ox, oy = (CW - tw) // 2, (CH - th) // 2
def up(m, sig=1.4):
    f = cv2.GaussianBlur(m[by0:by1, bx0:bx1].astype(np.float32), (0, 0), sig)
    return cv2.resize(f, (tw, th), interpolation=cv2.INTER_CUBIC) >= 0.5
Fc = nd.binary_closing(F, structure=np.ones((3, 3)), iterations=8) | F
Ft = up(F); Fct = up(Fc, 3.0); Kt = up(K, 1.2); St = up(specks, 0.8)
S = sy
dK = cv2.distanceTransform((~Kt).astype(np.uint8), cv2.DIST_L2, 5)
halo = dK <= 22 * S
Ft &= ~halo; St &= ~halo
Fct &= ~halo
dF = cv2.distanceTransform((~(Fct | Ft)).astype(np.uint8), cv2.DIST_L2, 5)
line = (dF >= 24 * S) & (dF <= 34 * S) & (dK > 45 * S)
ring = (dK > 0) & (dK <= 11 * S)
# blue texture: bleed blue pixels, upscale
bl = F & ~specks & (A >= 128) & ((rgb[..., 2] - rgb[..., 0]) > 60) & (cb == 0)
_, (iy, ix) = nd.distance_transform_edt(~bl, return_indices=True)
tex = rgb[iy, ix][by0:by1, bx0:bx1]
cbr = (cb > 0)[by0:by1, bx0:bx1]
tb = cv2.GaussianBlur(tex, (0, 0), 12); tex[cbr] = tb[cbr]
tex = cv2.resize(cv2.GaussianBlur(tex, (0, 0), 0.6), (tw, th), interpolation=cv2.INTER_CUBIC)
o = np.zeros((th, tw, 4), np.uint8)
o[line] = (255, 255, 255, 255)
o[Ft, :3] = np.clip(tex[Ft], 0, 255).astype(np.uint8); o[Ft, 3] = 255
o[St & Ft] = (242, 244, 250, 255)
o[ring] = (255, 255, 255, 255)
o[Kt] = (8, 8, 10, 255)
# specks: drop tiny islands
al = o[..., 3] > 0; lab, _ = nd.label(al); sz = np.bincount(lab.ravel()); small = sz[lab] < 60 * S * S; small &= al
o[small] = 0
canvas = np.zeros((CH, CW, 4), np.uint8); canvas[oy:oy + th, ox:ox + tw] = o
Image.fromarray(canvas, "RGBA").save(out, dpi=(300, 300))
print(out, "design %dx%d px = %.2f x %.2f in" % (tw, th, tw / 300, th / 300))
