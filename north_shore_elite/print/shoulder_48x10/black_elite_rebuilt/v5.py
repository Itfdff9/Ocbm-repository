"""NORTH SHORE Elite shoulder - clean high-quality rebuild (v3).
Same letter shapes/positions as Brian's upload (traced from it), but drawn as crisp vector shapes:
varsity letters = straight-edged polygons fitted to the traced letters, solid royal blue, crisp white distress
specks, clean white keyline at the upload's spacing; 'Elite' = smoothed traced script, black with white keyline.
usage: python3 clean_v3.py SRC OUT.png prop|stretch"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
src, out, mode = sys.argv[1], sys.argv[2], sys.argv[3]
s_ = open('/home/claude/nse3/fix_base.py').read()
code = s_[s_.index("a = np.array(Image.open(src)"):s_.index("# ---- smooth + scale")]
code = code.replace("K &= ~cut", "pass").replace("K |= ((xx - 2270)", "K |= False & ((xx - 2270)")
exec(code)
BLUE = np.median(rgb[F & ((rgb[..., 2] - rgb[..., 0]) > 80) & (A >= 250)], 0)
BLUE = np.array([8., 84., 200.]); print("blue", BLUE)
# ---- clean letter mask: remove edge bites/spikes, then polygon fit
disk = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
Fu = F.astype(np.uint8)
Fu = cv2.morphologyEx(Fu, cv2.MORPH_CLOSE, disk(int(__import__('os').environ.get('CL','11'))))
Fu = cv2.morphologyEx(Fu, cv2.MORPH_OPEN, disk(int(__import__('os').environ.get('OP','8'))))
lab, _ = nd.label(Fu); sz = np.bincount(lab.ravel()); sz[0] = 0; Fu = (sz[lab] >= 2000).astype(np.uint8)
cs, hier = cv2.findContours(Fu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
polys = []
for i, c in enumerate(cs):
    if cv2.contourArea(c) < 150: continue
    p = cv2.approxPolyDP(c, 4.0, True)[:, 0, :].astype(np.float64)
    polys.append((p, hier[0][i][3] >= 0))
import pickle
if __import__('os').environ.get('FONTL','1')=='1': polys = pickle.load(open('/home/claude/nse3/letters_polys.pkl','rb'))
# ---- Elite: smoothed contours
Ku = K.astype(np.uint8)
kc, kh = cv2.findContours(Ku, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
kpolys = []
for i, c in enumerate(kc):
    if len(c) < 10: continue
    p = c[:, 0, :].astype(np.float64)
    sg = 6.0 if len(p) > 200 else 3.0
    p = np.stack([nd.gaussian_filter1d(p[:, 0], sg, mode="wrap"), nd.gaussian_filter1d(p[:, 1], sg, mode="wrap")], 1)
    kpolys.append((p, kh[0][i][3] >= 0))
# ---- specks (crisp)
sp = specks & (F)
lab, _ = nd.label(sp); sz = np.bincount(lab.ravel()); sz[0] = 0; sp = sz[lab] >= 6
# ---- target geometry
ys, xs = np.where(nd.binary_dilation(F, iterations=36) | nd.binary_dilation(K, iterations=14))
bx0, bx1, by0, by1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
CW, CH, M = 14400, 3000, 30
if mode == "prop":
    sx = sy = min((CW - 2 * M) / (bx1 - bx0), (CH - 2 * M) / (by1 - by0))
elif mode == "photo":
    sy = (CH - 2 * M) / (by1 - by0); sx = sy * 4.08 * (by1 - by0) / (bx1 - bx0)
else:
    sx = (CW - 2 * M) / (bx1 - bx0); sy = (CH - 2 * M) / (by1 - by0)
tw, th = int(round((bx1 - bx0) * sx)), int(round((by1 - by0) * sy))
def draw(pl):
    m = np.zeros((th, tw), np.uint8)
    outer = [np.round(((p - [bx0, by0]) * [sx, sy]) * 16).astype(np.int32) for p, h in pl if not h]
    inner = [np.round(((p - [bx0, by0]) * [sx, sy]) * 16).astype(np.int32) for p, h in pl if h]
    cv2.fillPoly(m, outer, 255, cv2.LINE_AA, shift=4); cv2.fillPoly(m, inner, 0, cv2.LINE_AA, shift=4)
    return m >= 128
Ft = draw(polys); Kt = draw(kpolys)
f = cv2.GaussianBlur(sp[by0:by1, bx0:bx1].astype(np.float32), (0, 0), 0.7)
St = cv2.resize(f, (tw, th), interpolation=cv2.INTER_CUBIC) >= 0.45
S = sy
dK = cv2.distanceTransform((~Kt).astype(np.uint8), cv2.DIST_L2, 5)
BAND = float(__import__('os').environ.get('BAND', '16')); KR = float(__import__('os').environ.get('KR', '9'))
Ft &= ~(dK <= KR * S)
lab, _ = nd.label(Ft); sz = np.bincount(lab.ravel()); sz[0] = 0; Ft = sz[lab] >= 4000 * S * S
St &= Ft
labS, nS = nd.label(St); objs = nd.find_objects(labS)
for i, sl in enumerate(objs, 1):
    hh_, ww_ = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
    if max(hh_, ww_) > 5 * max(1, min(hh_, ww_)) and max(hh_, ww_) > 40 * S: St[sl][labS[sl] == i] = False
dF = cv2.distanceTransform((~Ft).astype(np.uint8), cv2.DIST_L2, 5)
line = (dF > 0) & (dF <= BAND * S)
ring = (dK > 0) & (dK <= KR * S)
o = np.zeros((th, tw, 4), np.uint8)
o[line] = (255, 255, 255, 255)
o[Ft] = (*np.clip(BLUE, 0, 255).astype(np.uint8), 255)
o[St] = (255, 255, 255, 255)
o[ring] = (255, 255, 255, 255)
o[Kt] = (8, 8, 10, 255)
al = o[..., 3] > 0; lab, _ = nd.label(al); sz = np.bincount(lab.ravel()); small = (sz[lab] < 80 * S * S) & al; o[small] = 0
canvas = np.zeros((CH, CW, 4), np.uint8); ox, oy = (CW - tw) // 2, (CH - th) // 2
canvas[oy:oy + th, ox:ox + tw] = o
Image.fromarray(canvas, "RGBA").save(out, dpi=(300, 300))
print(out, tw, th)
