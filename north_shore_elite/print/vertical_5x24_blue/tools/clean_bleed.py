"""Keep Brian's NORTH SHORE exactly; only clean the blue bleeding around/through Elite.
Elite: white fill traced + smoothed, clean black band + thin blue edge. Bleed below the letter band -> transparent."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
a = np.array(Image.open("rot.png").convert("RGBA")); H, W = a.shape[:2]
rgb = a[..., :3].astype(int); A = a[..., 3]
E = np.load("Ew.npy").astype(np.uint8)
# smooth Elite white fill
kc, kh = cv2.findContours(E, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
SS = 4; m = np.zeros((H * SS, W * SS), np.uint8)
for i, c in enumerate(kc):
    if len(c) < 30 or abs(cv2.contourArea(c)) < 400: continue
    p = c[:, 0, :].astype(np.float64)
    p = np.stack([nd.gaussian_filter1d(p[:, 0], 5, mode="wrap"), nd.gaussian_filter1d(p[:, 1], 5, mode="wrap")], 1)
    cv2.fillPoly(m, [np.round((p + 0.5) * SS * 16).astype(np.int32)], 0 if kh[0][i][3] >= 0 else 255, cv2.LINE_AA, shift=4)
Ef = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA) >= 128
dE = nd.distance_transform_edt(~Ef)
K1, K2 = 36, 45                                   # black band, blue edge (px)
# letter band bottom (blue fill bottom curve + letter trim)
pb = np.array([3.25772772e-05, -2.42580710e-01, 1.23945472e+03])
xx = np.arange(W); yb = np.polyval(pb, xx) + 50
yy = np.arange(H)[:, None]
below = yy > yb[None, :]
Z = dE <= 160                                     # Elite neighbourhood
out = a.copy()
clear = (dE > K2) & below
out[clear] = 0
# inside the letter band: remove blue/black smear that sits right outside the new Elite edge
band_smear = Z & (dE > K2) & (dE <= 75) & ~below
blueish = (rgb[..., 2] - rgb[..., 0] > 40)
darkish = rgb.max(2) < 70
# keep letter pixels: letter blue that continues away from Elite (connected to blue farther than 75 px)
lblue = blueish & (A >= 128)
lab, _ = nd.label(lblue); far = np.unique(lab[lblue & (dE > 90)]); far = far[far > 0]
legit = np.isin(lab, far)
smear = band_smear & (A > 0) & ~legit & (blueish | (darkish & (dE > K2 + 6)))
out[smear] = 0
# doubled old Elite outline (black + blue line) just outside the new edge, inside the letter band:
# replace with the letter pixels next to it (nearest original pixel farther from Elite)
oldline = ~below & (dE > K2) & (dE <= 85) & (out[..., 3] >= 128) & ((rgb.max(2) < 70) | (blueish & ~legit))
src_ok = ~below & (dE > 85) & (a[..., 3] >= 128)
lab3, _ = nd.label(oldline)
_, (iy, ix) = nd.distance_transform_edt(~src_ok, return_indices=True)
dsrc = nd.distance_transform_edt(~src_ok)
rep = oldline & (dsrc < 60)
out[rep] = a[iy[rep], ix[rep]]
print("oldline replaced", int(rep.sum()))
# small transparent gaps left between Elite and the letters -> Elite black outline
tr = (out[..., 3] < 128) & ~below
lab2, _ = nd.label(tr); sz2 = np.bincount(lab2.ravel()); sz2[0] = 0
gap = tr & (sz2[lab2] < 8000) & (dE <= 90)
gapc = np.unique(lab2[gap]); gapc = gapc[gapc > 0]
# only components that lie wholly near Elite
okc = [c for c in gapc if dE[lab2 == c].max() <= 110]
fillg = np.isin(lab2, okc)
out[fillg] = (8, 8, 10, 255)
print("gap fill", int(fillg.sum()))
# new Elite on top
out[(dE <= K2)] = (20, 60, 200, 255)
out[(dE <= K1)] = (8, 8, 10, 255)
out[Ef] = (255, 255, 255, 255)
# hard alpha + specks
al = out[..., 3] >= 128; out[..., 3] = np.where(al, 255, 0); out[~al, :3] = 0
lab, _ = nd.label(al); sz = np.bincount(lab.ravel()); small = (sz[lab] < 150) & al; out[small] = 0
np.save("clean_land.npy", out)
Image.fromarray(out, "RGBA").save("clean_land.png")
print("cleared", int(clear.sum()), "smear", int(smear.sum()))
