"""Add an Elite-style trim (white band + thin outer edge) around NORTH SHORE; Elite kept exactly as uploaded.
usage: python3 trim.py OUT.png native|photo|stretch"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
out, mode = sys.argv[1], sys.argv[2]
a = np.array(Image.open("src.png").convert("RGBA")).astype(np.float32); A = a[..., 3]; rgb = a[..., :3]
H, W = A.shape
op = A >= 128
blue = ((rgb[..., 2] - rgb[..., 0]) > 60) & (rgb[..., 2] > 110) & op
dark = (rgb.max(2) < 70) & op
# Elite: original blue fill; trim rebuilt cleanly at the upload's own widths (white 21 px, blue edge to 32 px)
fill = cv2.morphologyEx(blue.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
lab, _ = nd.label(fill); sz = np.bincount(lab.ravel()); sz[0] = 0; fill = (sz[lab] >= 2000) | (fill & blue & (sz[lab] >= 2000))
fill = nd.binary_closing(fill, iterations=2)
dfl = nd.distance_transform_edt(~fill)
EDGEC = np.median(rgb[blue & (dfl > 22) & (dfl < 31)], 0)
E = dfl <= 32
# letters
Lm = dark & ~nd.binary_dilation(E, iterations=2)
lab, _ = nd.label(Lm); sz = np.bincount(lab.ravel()); sz[0] = 0; Lm = sz[lab] >= 800
# --- output geometry
ys, xs = np.where(Lm | E); PAD = 50
x0, x1, y0, y1 = max(0, xs.min() - PAD), xs.max() + PAD, max(0, ys.min() - PAD), ys.max() + PAD
def crop(m): 
    z = np.zeros((y1 - y0, x1 - x0) + m.shape[2:], m.dtype); yy = slice(y0, min(y1, H)); xx = slice(x0, min(x1, W))
    z[:yy.stop - y0, :xx.stop - x0] = m[yy, xx]; return z
if mode == "native":
    CW, CH = x1 - x0, y1 - y0; sx = sy = 1.0
else:
    CW, CH, M = 14400, 3000, 20
    sy = (CH - 2 * M) / (y1 - y0); sx = sy if mode == "photo" else (CW - 2 * M) / (x1 - x0)
tw, th = int(round((x1 - x0) * sx)), int(round((y1 - y0) * sy)); S = sy
def up(m, sig=1.0):
    f = cv2.GaussianBlur(crop(m).astype(np.float32), (0, 0), sig)
    return cv2.resize(f, (tw, th), interpolation=cv2.INTER_CUBIC) >= 0.5
# complete the letters hidden under Elite (so no trim is drawn along the cut)
Lc = Lm | (nd.binary_closing(Lm | E, structure=np.ones((3, 3)), iterations=0) if False else Lm)
k = cv2.getStructuringElement(cv2.MORPH_RECT, (121, 121))
Lclose = cv2.morphologyEx(Lm.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
Ed = nd.binary_dilation(E, iterations=6)
Lc = Lm | (Lclose & Ed)
hh = nd.binary_fill_holes(Lc | E) & ~Lc & ~E
lab, _ = nd.label(nd.binary_fill_holes(Lc) & ~Lc); sz = np.bincount(lab.ravel()); sz[0] = 0
Lc |= (sz[lab] > 0) & (sz[lab] < 8000)
lab, _ = nd.label(hh); sz = np.bincount(lab.ravel()); sz[0] = 0
Lc |= (sz[lab] > 0) & (sz[lab] < 8000) & nd.binary_dilation(Lc, iterations=3)
Lc |= (sz[lab] > 0) & (sz[lab] < 8000)
Lt = up(Lc, 1.2); Ft_ = up(fill, 1.0)
WB, EB = 21.0, 11.0      # white band / outer edge (source px) = Elite's trim
dL = cv2.distanceTransform((~Lt).astype(np.uint8), cv2.DIST_L2, 5)
white = (dL > 0) & (dL <= WB * S); edge = (dL > WB * S) & (dL <= (WB + EB) * S)
_, (iy, ix) = nd.distance_transform_edt(~(fill & blue), return_indices=True)
erg = crop(rgb[iy, ix]); erg = cv2.resize(erg, (tw, th), interpolation=cv2.INTER_LANCZOS4)
dE = cv2.distanceTransform((~Ft_).astype(np.uint8), cv2.DIST_L2, 5)
Ew = (dE > 0) & (dE <= WB * S); Ee = (dE > WB * S) & (dE <= (WB + EB) * S); Et = Ft_ | Ew | Ee
o = np.zeros((th, tw, 4), np.uint8)
o[edge] = (10, 10, 12, 255)
o[white] = (255, 255, 255, 255)
o[Lt] = (10, 10, 12, 255)
o[Ee] = (*np.clip(EDGEC, 0, 255).astype(np.uint8), 255)
o[Ew] = (255, 255, 255, 255)
o[Ft_, :3] = np.clip(erg[Ft_], 0, 255).astype(np.uint8); o[Ft_, 3] = 255
al = o[..., 3] > 0; lab, _ = nd.label(al); sz = np.bincount(lab.ravel()); small = (sz[lab] < 50 * S * S) & al; o[small] = 0
canvas = np.zeros((CH, CW, 4), np.uint8); ox, oy = (CW - tw) // 2, (CH - th) // 2
canvas[oy:oy + th, ox:ox + tw] = o
dpi = 300 if mode != "native" else 300
Image.fromarray(canvas, "RGBA").save(out, dpi=(dpi, dpi)); print(out, canvas.shape[1], canvas.shape[0], "design", tw, th)
