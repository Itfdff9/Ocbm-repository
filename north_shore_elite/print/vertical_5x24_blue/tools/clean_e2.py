"""Pass 2 around Elite: one clean outline (black envelope that fills narrow gaps between strokes + thin blue edge),
then remove leftover old-Elite debris (blue/black/white fragments that live only next to Elite)."""
import numpy as np, cv2, os
from PIL import Image
from scipy import ndimage as nd
base = np.load("clean_land.npy"); H, W = base.shape[:2]
a0 = np.array(Image.open("rot.png").convert("RGBA"))
rgb = base[..., :3].astype(int); A = base[..., 3] >= 128
Ew = (base[..., :3].min(2) == 255) & A
# recover the smoothed Elite white: pure (255,255,255) pixels written by pass 1 that sit in the Elite region
Ef = np.load("Ef.npy") if os.path.exists("Ef.npy") else None
K1 = int(os.environ.get("K1", 36)); BL = int(os.environ.get("BL", 9)); CR = int(os.environ.get("CR", 28))
R = int(os.environ.get("R", 95))
dE = nd.distance_transform_edt(~Ef)
env = dE <= K1
# close narrow gaps between strokes (morphological closing via distance transforms)
dil = nd.distance_transform_edt(~env) <= CR
env = nd.distance_transform_edt(dil) > CR
env |= dE <= K1
# fill small enclosed holes in the envelope
holes = nd.binary_fill_holes(env) & ~env
lab, n = nd.label(holes); sz = np.bincount(lab.ravel()); sz[0] = 0
env |= (sz[lab] > 0) & (sz[lab] < 40000)
denv = nd.distance_transform_edt(~env)
edge = (denv > 0) & (denv <= BL)
outer = denv > BL
out = base.copy()
# classify outside pixels
blue = A & (rgb[..., 2] - rgb[..., 0] > 40)
dark = A & (rgb.max(2) < 75)
white = A & (rgb.min(2) > 175)
other = A & ~blue & ~dark & ~white
resid = np.zeros((H, W), bool)
zone = outer & (denv <= R)
for M in (blue, dark, white, other):
    m = M & outer
    l, k = nd.label(m)
    if k == 0: continue
    mx = nd.maximum(denv, l, np.arange(1, k + 1))
    bad = np.zeros(k + 1, bool); bad[1:] = mx <= R
    resid |= bad[l]
print("residue px", int(resid.sum()))
# replace each residue component by the colour that surrounds it (flat letter blue / black / white / transparent)
LB = np.median(base[blue & outer & (denv > 200)][:, :3], 0).astype(np.uint8)
print("letter blue", LB)
cls = np.zeros((H, W), np.int8)  # 0 transparent,1 blue,2 dark,3 white,4 other
cls[blue] = 1; cls[dark] = 2; cls[white] = 3; cls[other] = 4
pal = {0: (0, 0, 0, 0), 1: (*LB, 255), 2: (8, 8, 10, 255), 3: (255, 255, 255, 255), 4: (*LB, 255)}
l, k = nd.label(resid)
sl = nd.find_objects(l)
for i, s_ in enumerate(sl, 1):
    y0, y1 = max(s_[0].start - 8, 0), min(s_[0].stop + 8, H); x0, x1 = max(s_[1].start - 8, 0), min(s_[1].stop + 8, W)
    comp = l[y0:y1, x0:x1] == i
    ring = nd.binary_dilation(comp, iterations=6) & ~comp & outer[y0:y1, x0:x1] & ~resid[y0:y1, x0:x1]
    if ring.sum() == 0: c = 0
    else: c = int(np.bincount(cls[y0:y1, x0:x1][ring], minlength=5).argmax())
    out[y0:y1, x0:x1][comp] = pal[c]
# thin dark scratch lines left inside letter blue near Elite -> letter blue
near = outer & (denv < 220)
dk = (out[..., 3] == 255) & (out[..., :3].astype(int).max(2) < 90) & near
thick = cv2.morphologyEx(dk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((9, 9), np.uint8)).astype(bool)
thin = dk & ~thick
bl = (out[..., 3] == 255) & (out[..., 2].astype(int) - out[..., 0].astype(int) > 40)
frac = cv2.blur(bl.astype(np.float32), (21, 21))
fixl = thin & (frac > 0.45)
out[fixl] = (*LB, 255); print("thin lines", int(fixl.sum()))
# dark-navy smudges (old bleed) inside letters near Elite -> letter blue
o = out[..., :3].astype(int)
navy = (out[..., 3] == 255) & near & (o[..., 2] - o[..., 0] > 25) & (o[..., 2] < 170) & (o.max(2) >= 75)
out[navy] = (*LB, 255); print("navy", int(navy.sum()))
# small see-through holes enclosed by letter art near Elite -> surrounding colour
tr = (out[..., 3] < 128) & outer
l2, k2 = nd.label(tr); s2 = np.bincount(l2.ravel())
for i, s_ in enumerate(nd.find_objects(l2), 1):
    if s_ is None or s2[i] > 9000: continue
    y0, y1 = max(s_[0].start - 8, 0), min(s_[0].stop + 8, H); x0, x1 = max(s_[1].start - 8, 0), min(s_[1].stop + 8, W)
    comp = l2[y0:y1, x0:x1] == i
    if denv[y0:y1, x0:x1][comp].min() > 200: continue
    ring = nd.binary_dilation(comp, iterations=5) & ~comp
    sub = out[y0:y1, x0:x1]; rc = sub[ring]
    if (rc[:, 3] < 128).any(): continue          # touches open background -> real gap, keep
    oc = rc[:, :3].astype(int)
    votes = [((oc[:, 2] - oc[:, 0]) > 40).sum(), (oc.max(1) < 90).sum(), (oc.min(1) > 175).sum()]
    c = [(*LB, 255), (8, 8, 10, 255), (255, 255, 255, 255)][int(np.argmax(votes))]
    sub[comp] = c
# wipe what was between old outline and new one
out[edge] = (20, 60, 200, 255)
out[env] = (8, 8, 10, 255)
out[Ef] = (255, 255, 255, 255)
# anti-alias the new outline edges at 4x? keep hard alpha for DTF; just smooth by supersampled masks
al = out[..., 3] >= 128; out[..., 3] = np.where(al, 255, 0); out[~al, :3] = 0
l, _ = nd.label(al); s = np.bincount(l.ravel()); sm = (s[l] < 150) & al; out[sm] = 0
# tiny transparent pinholes inside art -> nearest colour
tr = ~al; l, _ = nd.label(tr); s = np.bincount(l.ravel()); pin = tr & (s[l] < 60)
_, (iy, ix) = nd.distance_transform_edt(pin, return_indices=True); out[pin] = out[iy[pin], ix[pin]]
np.save("elite_layer.npy", env | edge); np.save("clean2.npy", out); Image.fromarray(out, "RGBA").save("clean2.png")
