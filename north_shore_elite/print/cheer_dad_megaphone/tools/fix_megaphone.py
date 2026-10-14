"""CHEER DAD megaphone - cleanup of Brian's upload (no redraw).
usage: python3 fix_megaphone.py SRC OUT.png [--fill-spikes]"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd


def smooth_fill(img, mask, known, sigmas=(3, 6, 12, 25)):
    num = np.zeros_like(img); den = np.zeros(mask.shape, np.float32)
    k = known.astype(np.float32)
    for sg in sigmas:
        n_ = cv2.GaussianBlur(img * k[..., None], (0, 0), sg); d_ = cv2.GaussianBlur(k, (0, 0), sg)
        take = (den < 0.05) & (d_ > 0.05)
        num[take] = n_[take]; den[take] = d_[take]
    est = num / np.maximum(den, 1e-6)[..., None]
    img[mask] = est[mask]
    return img

src, out = sys.argv[1], sys.argv[2]
FILL = "--fill-spikes" in sys.argv
a = np.array(Image.open(src).convert("RGBA")).astype(np.float32)
rgb, A = a[..., :3].copy(), a[..., 3].copy()
H, W = A.shape
mx, mn = rgb.max(2), rgb.min(2)

# --- 1/2. leftover flat grey between CHEER letters, C opening, CHEER/DAD strip
neutral = (mx - mn <= 14) & (mx >= 160) & (mx <= 222) & (A > 0)
zone = np.zeros_like(neutral); zone[:1000, :] = True
core = cv2.morphologyEx((neutral & zone).astype(np.uint8), cv2.MORPH_OPEN,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))) > 0
loose = (mx - mn <= 24) & (mx >= 140) & (mx <= 230) & (A > 0) & zone
grey_bg = nd.binary_dilation(core, iterations=4) & loose
lab_, n_ = nd.label(grey_bg); sz_ = nd.sum(grey_bg, lab_, range(1, n_ + 1))
grey_bg = np.isin(lab_, np.where(sz_ >= 2500)[0] + 1)
A[grey_bg] = 0

# --- 3. spike interiors (white centres eaten by the earlier removal)
if FILL:
    inv = A < 128
    lab, n = nd.label(inv)
    ctrs = [(424, 2663), (2576, 2637), (295, 3341), (2719, 3411), (2786, 3287)]
    hole = np.zeros_like(inv)
    for x, y in ctrs:
        l = lab[y, x]
        if l and (lab == l).sum() < 60000: hole |= lab == l
    near = nd.binary_dilation(hole, iterations=70)
    cand = hole | (near & (mx >= 80) & (((mx - mn <= 40) & (A >= 128)) | (A < 255)))
    labc, _ = nd.label(cand)
    keep = np.unique(labc[hole]); keep = keep[keep > 0]
    R = np.isin(labc, keep)
    rim = nd.binary_dilation(R, iterations=10) & ~R & (A >= 250) & (mx - mn <= 40) & (mx >= 90)
    num = np.zeros_like(rgb); den = np.zeros((H, W), np.float32)
    for sg in (30, 60, 120):
        k = rim.astype(np.float32)
        n_ = cv2.GaussianBlur(rgb * k[..., None], (0, 0), sg); d_ = cv2.GaussianBlur(k, (0, 0), sg)
        take = (den < 1e-3) & (d_ > 1e-4)
        num[take] = n_[take]; den[take] = d_[take]
    est = num / np.maximum(den, 1e-6)[..., None]
    est = cv2.GaussianBlur(est, (0, 0), 14)
    dR = nd.distance_transform_edt(R)
    w = np.clip(dR / 6.0, 0, 1)[..., None]
    rgb = np.where(R[..., None], est * w + rgb * (1 - w) * (A[..., None] >= 128) + est * (1 - w) * (A[..., None] < 128), rgb)
    A[R] = 255

# --- 5. pseudo-text scribble on the blue streak right of the left star
# --- 6. loop on the megaphone grip
inp_mask = np.zeros((H, W), np.uint8)
box5 = (500, 2180, 595, 2230)
sub = rgb[box5[1]:box5[3], box5[0]:box5[2]]
inp_mask[box5[1]:box5[3], box5[0]:box5[2]] = ((sub.min(2) > 95) | ((sub.max(2) - sub.min(2)) < 60) & (sub.max(2) > 90)).astype(np.uint8) * 255
box6 = (905, 2430, 975, 2492)
sub = rgb[box6[1]:box6[3], box6[0]:box6[2]]
m6 = (sub.max(2) > 45).astype(np.uint8) * 255
inp_mask[box6[1]:box6[3], box6[0]:box6[2]] |= m6
inp_mask = cv2.dilate(inp_mask, np.ones((7, 7), np.uint8))
rgb = cv2.inpaint(np.clip(rgb, 0, 255).astype(np.uint8), inp_mask, 6, cv2.INPAINT_TELEA).astype(np.float32)

# --- 7. ghost face fragment inside the cheerleader silhouette
x0, y0, x1, y1 = 1250, 2046, 1335, 2128
def first_dark(y):
    row = rgb[y, x0:x1].max(1) < 110
    r = np.where(np.convolve(row.astype(int), np.ones(4, int), 'valid') == 4)[0]
    return x0 + r[0] if len(r) else x1
bt, bb = first_dark(y0 - 2), first_dark(y1 + 2)
sil = np.array([3., 3., 4.])
fm = np.zeros((H, W), bool); right = np.zeros((H, W), bool)
for y in range(y0, y1):
    bx = int(round(bt + (bb - bt) * (y - y0) / (y1 - y0)))
    fm[y, x0 - 6:bx] = True; right[y, bx:x1] = True
rgb[right & (rgb.max(2) > 10)] = sil
bad = fm & (rgb.max(2) < 228)
bad = nd.binary_dilation(bad, iterations=2) & fm
roi = np.zeros((H, W), bool); roi[y0 - 40:y1 + 40, x0 - 60:x1] = True
known = roi & ~bad & ~right & (rgb.max(2) >= 228)
rgb = smooth_fill(rgb, bad, known)
for y in range(y0, y1):
    bx = int(round(bt + (bb - bt) * (y - y0) / (y1 - y0)))
    rgb[y, bx - 1] = 0.5 * rgb[y, bx - 2] + 0.5 * sil

# loop on the grip: fill the curl area from the surrounding dark only (white squares kept)
gm = np.zeros((H, W), np.uint8); cv2.ellipse(gm, (940, 2458), (30, 30), 0, 0, 360, 255, -1)
sq = nd.binary_dilation(rgb.min(2) > 150, iterations=3)
gmb = (gm > 0) & ~sq
roi = np.zeros((H, W), bool); roi[2400:2520, 880:1000] = True
known = roi & ~(gm > 0) & ~sq & (rgb.max(2) < 90)
rgb = smooth_fill(rgb, gmb, known, sigmas=(6, 12, 25))

# --- 4. hard alpha, specks, pinholes
solid = A >= 128
lab, n = nd.label(solid); sz = nd.sum(solid, lab, range(1, n + 1))
solid &= ~np.isin(lab, np.where(sz < 40)[0] + 1)
inv = ~solid; lab, n = nd.label(inv); sz = nd.sum(inv, lab, range(1, n + 1))
pin = np.isin(lab, np.where(sz < 40)[0] + 1)
if pin.any():
    _, (iy, ix) = nd.distance_transform_edt(pin | ~solid & ~pin, return_indices=True) if False else nd.distance_transform_edt(pin, return_indices=True)
    rgb[pin] = rgb[iy[pin], ix[pin]]
solid |= pin
# decontaminate edge pixels that were semi-transparent: borrow colour from the nearest fully-opaque pixel within 3px
was_semi = solid & (a[..., 3] < 230)
full = solid & ~was_semi
d, (iy, ix) = nd.distance_transform_edt(~full, return_indices=True)
fix = was_semi & (d <= 3)
rgb[fix] = rgb[iy[fix], ix[fix]] * 0.6 + rgb[fix] * 0.4

o = np.zeros((H, W, 4), np.uint8)
o[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8); o[..., 3] = np.where(solid, 255, 0)
o[~solid, :3] = 0
Image.fromarray(o, "RGBA").save(out, dpi=(300, 300))
print(out, W, H, "grey removed", int(grey_bg.sum()))
