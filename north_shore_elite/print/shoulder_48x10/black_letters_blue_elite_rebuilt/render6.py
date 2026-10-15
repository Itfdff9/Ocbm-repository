"""Black NORTH SHORE (font rebuild) + blue script Elite (traced, smoothed), thin white keylines with a gap,
transparent background, hard alpha, 48x10 in @ 300 DPI.  usage: python3 render6.py OUT.png stretch|photo"""
import sys, os, pickle, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
out, mode = sys.argv[1], sys.argv[2]
polys = pickle.load(open("polys.pkl", "rb"))
K = np.load("K0.npy")
Ku = cv2.morphologyEx(K.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
lab, _ = nd.label(Ku); sz = np.bincount(lab.ravel()); sz[0] = 0; Ku = (sz[lab] >= 1500).astype(np.uint8)
kc, kh = cv2.findContours(Ku, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
kpolys = []
for i, c in enumerate(kc):
    if len(c) < 10: continue
    p = c[:, 0, :].astype(np.float64); sg = 6.0 if len(p) > 200 else 3.0
    p = np.stack([nd.gaussian_filter1d(p[:, 0], sg, mode="wrap"), nd.gaussian_filter1d(p[:, 1], sg, mode="wrap")], 1)
    kpolys.append((p, kh[0][i][3] >= 0))
allp = np.concatenate([p for p, _ in polys] + [p for p, _ in kpolys])
PAD = 40
bx0, by0 = allp.min(0) - PAD; bx1, by1 = allp.max(0) + PAD
CW, CH, M = 14400, 3000, 20
if mode == "photo":
    sy = (CH - 2 * M) / (by1 - by0); sx = sy
else:
    sx = (CW - 2 * M) / (bx1 - bx0); sy = (CH - 2 * M) / (by1 - by0)
tw, th = int(round((bx1 - bx0) * sx)), int(round((by1 - by0) * sy))
def draw(pl):
    m = np.zeros((th, tw), np.uint8)
    o = [np.round(((p - [bx0, by0]) * [sx, sy]) * 16).astype(np.int32) for p, h in pl if not h]
    i = [np.round(((p - [bx0, by0]) * [sx, sy]) * 16).astype(np.int32) for p, h in pl if h]
    cv2.fillPoly(m, o, 255, cv2.LINE_AA, shift=4); cv2.fillPoly(m, i, 0, cv2.LINE_AA, shift=4)
    return m >= 128
Ft = draw(polys); Kt = draw(kpolys); S = sy
G1, G2 = float(os.environ.get("G1", "9")), float(os.environ.get("G2", "15"))     # letter keyline (src px)
E1, E2 = float(os.environ.get("E1", "15")), float(os.environ.get("E2", "22"))    # Elite keyline
dK = cv2.distanceTransform((~Kt).astype(np.uint8), cv2.DIST_L2, 5)
clear = dK <= (E2 + G2 - G1) * S
Ft &= ~clear
dF = cv2.distanceTransform((~Ft).astype(np.uint8), cv2.DIST_L2, 5)
lineF = (dF > G1 * S) & (dF <= G2 * S) & ~clear
lineK = (dK > E1 * S) & (dK <= E2 * S)
o = np.zeros((th, tw, 4), np.uint8)
o[lineF] = (255, 255, 255, 255)
o[Ft] = (10, 10, 12, 255)
o[lineK] = (255, 255, 255, 255)
BL = tuple(int(v) for v in os.environ.get("BLUE", "8,42,165").split(","))
o[Kt] = (*BL, 255)
al = o[..., 3] > 0; lab, _ = nd.label(al); sz = np.bincount(lab.ravel()); small = (sz[lab] < 60 * S * S) & al; o[small] = 0
canvas = np.zeros((CH, CW, 4), np.uint8); ox, oy = (CW - tw) // 2, (CH - th) // 2
canvas[oy:oy + th, ox:ox + tw] = o
Image.fromarray(canvas, "RGBA").save(out, dpi=(300, 300)); print(out, tw, th)
