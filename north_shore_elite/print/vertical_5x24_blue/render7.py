"""Blue NORTH SHORE (Graduate rebuild, white + black trim) + white script Elite (traced, smoothed; black band + blue edge).
Built landscape, saved rotated back to the upload's vertical orientation. usage: python3 render7.py OUT.png LONGSIDE DPI"""
import sys, pickle, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
out, LS, DPI = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
polys = pickle.load(open("polys.pkl", "rb"))
E = np.load("Ew.npy").astype(np.uint8)
kc, kh = cv2.findContours(E, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
kpolys = []
for i, c in enumerate(kc):
    if len(c) < 30 or abs(cv2.contourArea(c)) < 400: continue
    p = c[:, 0, :].astype(np.float64); sg = 5.0
    p = np.stack([nd.gaussian_filter1d(p[:, 0], sg, mode="wrap"), nd.gaussian_filter1d(p[:, 1], sg, mode="wrap")], 1)
    kpolys.append((p, kh[0][i][3] >= 0))
H0, W0 = 1500, 7200
s = LS / W0; tw, th = LS, int(round(H0 * s))
def draw(pl):
    m = np.zeros((th, tw), np.uint8)
    o = [np.round(p * s * 16).astype(np.int32) for p, h in pl if not h]
    i = [np.round(p * s * 16).astype(np.int32) for p, h in pl if h]
    cv2.fillPoly(m, o, 255, cv2.LINE_AA, shift=4); cv2.fillPoly(m, i, 0, cv2.LINE_AA, shift=4)
    return m >= 128
Ft = draw(polys); Et = draw(kpolys)
for M_ in ("Ft", "Et"):
    m_ = locals()[M_]; hh = nd.binary_fill_holes(m_) & ~m_; lab_, _ = nd.label(hh); sz_ = np.bincount(lab_.ravel()); sz_[0] = 0
    m_ |= (sz_[lab_] > 0) & (sz_[lab_] < 3000 * s * s)
dist = lambda m: cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
dF, dE = dist(Ft), dist(Et)
LW, LK = 30 * s, 46 * s          # letter: white band, black edge
EK, EB = 34 * s, 44 * s          # Elite: black band, blue edge
BLUE = (16, 70, 215); EBLUE = (20, 60, 200); BLACK = (8, 8, 10)
o = np.zeros((th, tw, 4), np.uint8)
U = (dF <= LK).astype(np.uint8)
U = cv2.morphologyEx(U, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(40 * s) | 1,) * 2)) > 0
o[U] = (*BLACK, 255)
o[dF <= LW] = (255, 255, 255, 255)
o[Ft] = (*BLUE, 255)
o[dE <= EB] = (*EBLUE, 255)
o[dE <= EK] = (*BLACK, 255)
o[Et] = (255, 255, 255, 255)
o = np.ascontiguousarray(np.rot90(o, -1))
Image.fromarray(o, "RGBA").save(out, dpi=(DPI, DPI)); print(out, o.shape[1], o.shape[0], "%.2f x %.2f in" % (o.shape[1] / DPI, o.shape[0] / DPI))
