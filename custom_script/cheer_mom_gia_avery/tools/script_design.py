"""Cursive name/phrase stack with hand-drawn hearts, black ink + white outline, transparent, hard alpha.
usage: python3 script_design.py giaavery|cheermom FONT OUT.png LONGSIDE_PX [preview]"""
import sys, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
kind, FONT, OUT, LS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
U = 1000                                     # working unit: font size in px at the work scale
WORK = 2.0                                   # supersample
def word_mask(txt, size, extra):
    f = ImageFont.truetype(FONT, int(size))
    l, t, r, b = f.getbbox(txt, stroke_width=int(extra))
    pad = int(size * 0.1) + int(extra)
    im = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t), txt, font=f, fill=255, stroke_width=int(extra), stroke_fill=255)
    m = np.asarray(im) > 127
    ys, xs = np.where(m); return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
def cr(pts, n=80):
    p = np.array(pts, float); p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]]); o = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            o.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    o.append(p[-2]); return np.array(o)
# hand-drawn heart (unit box ~1 wide, 1.1 tall), with the little top tick like the original
HEART = [[[0.52, 0.26], [0.44, 0.10], [0.28, 0.03], [0.11, 0.09], [0.04, 0.28], [0.10, 0.52], [0.27, 0.76], [0.45, 0.95], [0.50, 1.05], [0.49, 1.12]],
         [[0.52, 0.26], [0.61, 0.10], [0.77, 0.03], [0.93, 0.10], [0.97, 0.30], [0.87, 0.55], [0.69, 0.79], [0.50, 1.00]]]
def heart_mask(size, pen):
    W = int(size * 1.1 + 2 * pen + 4); H = int(size * 1.25 + 2 * pen + 4)
    c = np.zeros((H, W), np.uint8)
    for path in HEART:
        p = cr(path) * size + pen + 2
        cv2.polylines(c, [np.round(p * 16).astype(np.int32)], False, 1, 1, cv2.LINE_8, shift=4)
    d = cv2.distanceTransform((c == 0).astype(np.uint8), cv2.DIST_L2, 5)
    m = d <= pen / 2
    ys, xs = np.where(m); return m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
def place(canvas, m, x, y, ang=0):
    if ang:
        im = Image.fromarray(m.astype(np.uint8) * 255).rotate(ang, Image.BICUBIC, expand=True); m = np.asarray(im) > 127
    h, w = m.shape; x, y = int(x - w / 2), int(y - h / 2)
    canvas[y:y + h, x:x + w] |= m
S = U * WORK
CW, CH = int(S * 4), int(S * 5)
can = np.zeros((CH, CW), bool)
fs = S
# pen weight: the font's own stroke + a little extra so it holds up in DTF
ex = float(sys.argv[6]) if len(sys.argv) > 6 else 0.012
pen_px = fs * ex
hp = fs * 0.028 + 2 * pen_px
def stack(items):
    # items: (mask, x_center_frac_of_widest, gap_before_frac_of_fs)
    y = int(S * 0.3)
    for m, xc, gap, ang in items:
        if ang:
            im = Image.fromarray(m.astype(np.uint8) * 255).rotate(ang, Image.BICUBIC, expand=True); m = np.asarray(im) > 127
        y += int(gap * fs)
        h, w = m.shape; x = int(CW * 0.5 + xc * fs - w / 2)
        can[y:y + h, x:x + w] |= m
        y += h
if kind == "giaavery":
    g = word_mask("Gia", fs, pen_px); a = word_mask("Avery", fs, pen_px)
    stack([(g, 0.35, 0, 5), (heart_mask(fs * 0.30, hp), 0.22, 0.02, 5), (a, -0.05, -0.02, 5), (heart_mask(fs * 0.30, hp), -0.18, -0.16, 5)])
else:
    c1 = word_mask("cheer", fs, pen_px); m1 = word_mask("mom", fs, pen_px)
    stack([(c1, -0.10, 0, 4), (m1, 0.22, -0.03, 4), (heart_mask(fs * 0.34, hp), 0.20, 0.06, 4)])
ys, xs = np.where(can)
OL = fs * float(sys.argv[7] if len(sys.argv) > 7 else 0.03)
pad = int(OL + 8)
can = can[ys.min() - pad:ys.max() + pad, xs.min() - pad:xs.max() + pad]
H, W = can.shape; k = LS / max(H, W)
inkf = cv2.resize(can.astype(np.float32), (int(round(W * k)), int(round(H * k))), interpolation=cv2.INTER_AREA)
inkb = inkf >= 0.5
d = cv2.distanceTransform((~inkb).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
solid = d <= OL * k
g = (255 - np.clip(inkf, 0, 1) * 245).astype(np.uint8)
o = np.zeros(inkb.shape + (4,), np.uint8)
o[..., 0] = g; o[..., 1] = g; o[..., 2] = g; o[..., 3] = np.where(solid, 255, 0); o[~solid, :3] = 0
dpi = float(sys.argv[5]) if len(sys.argv) > 5 else 300
Image.fromarray(o, "RGBA").save(OUT, dpi=(dpi, dpi))
print(OUT, o.shape[1], o.shape[0], "%.2f x %.2f in @ %.0f" % (o.shape[1] / dpi, o.shape[0] / dpi, dpi))
