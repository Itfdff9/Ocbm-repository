"""Hand-traced script lettering -> print PNG: black monoline pen strokes + white outline, transparent, hard alpha.
usage: python3 build_outlined.py paths.json OUT.png WIDTH_IN DPI_TAG LONGSIDE_PX [outline_srcpx]"""
import sys, json, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
J = json.load(open(sys.argv[1])); out = sys.argv[2]; win = float(sys.argv[3]); dpi = float(sys.argv[4]); LS = int(sys.argv[5])
OL = float(sys.argv[6]) if len(sys.argv) > 6 else 2.2
hw = J["hw"]
P = np.concatenate([np.array(p, float) for p in J["paths"]])
pad = hw + OL + 3
x0, y0 = P.min(0) - pad; x1, y1 = P.max(0) + pad
S = LS / max(x1 - x0, y1 - y0)
W, H = int(round((x1 - x0) * S)), int(round((y1 - y0) * S))
def cr(pts, n=60):
    p = np.array(pts, float)
    if len(p) == 2: return np.linspace(p[0], p[1], n)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]]); o = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            o.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    o.append(p[-2]); return np.array(o)
center = np.zeros((H, W), np.uint8)
for path in J["paths"]:
    c = (cr(path) - [x0, y0]) * S
    cv2.polylines(center, [np.round(c * 16).astype(np.int32)], False, 1, 1, cv2.LINE_8, shift=4)
d = cv2.distanceTransform((center == 0).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
r_in = hw * S; r_out = (hw + OL) * S
ink = np.clip(r_in + 0.5 - d, 0, 1).astype(np.float32)
solid = d <= r_out
del d
g = (255 - ink * 245).astype(np.uint8); del ink          # white outline -> black pen (10,10,12)
o = np.zeros((H, W, 4), np.uint8)
o[..., 0] = g; o[..., 1] = g; o[..., 2] = np.maximum(g, 12) if False else g
o[..., 3] = np.where(solid, 255, 0); o[~solid, :3] = 0
d_ = W / win if win > 0 else dpi
Image.fromarray(o, "RGBA").save(out, dpi=(d_, d_))
print(out, W, H, "-> %.2f x %.2f in at %.1f dpi" % (W / d_, H / d_, d_))
