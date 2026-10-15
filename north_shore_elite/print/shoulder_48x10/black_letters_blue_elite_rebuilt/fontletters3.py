"""NORTH SHORE in Graduate (collegiate), bolded to the design's stem weight, letters kept upright at the design's
x positions, then envelope-warped so tops/bottoms follow the design's arch. Output polygons in source coords."""
import numpy as np, cv2, pickle, os
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
FONT = "/home/claude/nse3/fonts/Graduate-Regular.ttf"
import json
L = json.loads(os.environ["LBOX"]) if "LBOX" in os.environ else [("N", 40, 468), ("O", 516, 811), ("R", 881, 1204), ("T", 1256, 1578), ("H", 1651, 2002), ("S", 2277, 2600), ("H", 2664, 3009), ("O", 3075, 3376), ("R", 3436, 3809), ("E", 3798, 4062)]
XSKIP = json.loads(os.environ.get("XSKIP", "[1250, 3050]"))
STEM = float(os.environ.get("STEM", "97"))
F = np.load(os.environ.get("FMASK", "/home/claude/nse2/F.npy")); H0, W0 = F.shape
# --- envelope curves from the traced letters
cols, tp, bt = [], [], []
for x in range(0, W0, 4):
    ys = np.where(F[:, x])[0]
    if len(ys) < 50: continue
    cols.append(x); tp.append(ys.min()); bt.append(ys.max())
cols, tp, bt = map(np.array, (cols, tp, bt))
def rfit(x, y, keep):
    x, y = x[keep], y[keep]
    for _ in range(6):
        p = np.polyfit(x, y, 2); r = y - np.polyval(p, x)
        k = np.abs(r) < max(12, np.percentile(np.abs(r), 70)); x, y = x[k], y[k]
    return p
pt = rfit(cols, tp, np.ones(len(cols), bool))
pb = rfit(cols, bt, (cols < XSKIP[0]) | (cols > XSKIP[1]))
print("top", pt, "bot", pb)
SS = 3; HF = 1000                                  # flat letter height (src px) before warp
def glyph(ch, w):
    f = ImageFont.truetype(FONT, 1400)
    l, t, r, b = f.getbbox(ch); im = Image.new("L", (r - l + 200, b - t + 200), 0)
    ImageDraw.Draw(im).text((100 - l, 100 - t), ch, font=f, fill=255)
    g = np.asarray(im) > 127; ys, xs = np.where(g); return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
# bolding radius from O stem
def stem_of(m):
    r = m[m.shape[0] // 2].astype(int); d = np.diff(np.r_[0, r, 0]); st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
    return (en - st).min()
flat = np.zeros((HF * SS, W0 * SS), np.uint8)
gO = glyph("O", 1)
def place(ch, x0, x1, R):
    w = x1 - x0; g = glyph(ch, w)
    tw, th_ = int((w - 2 * R) * SS), int((HF - 2 * R) * SS)
    m = cv2.resize(g.astype(np.uint8) * 255, (tw, th_), interpolation=cv2.INTER_AREA)
    pad = int(R * SS) + 2; m = np.pad(m, pad)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * int(R * SS) + 1, 2 * int(R * SS) + 1))
    m = cv2.dilate(m, k) > 127
    if ch == "H":   # keep the notch between the top serifs (dark spike in the shirt photo)
        yr = int(m.shape[0] * 0.30); r = m[yr]; xs_ = np.where(~r[pad:-pad])[0] + pad
        mid = m.shape[1] // 2; gap = xs_[(xs_ > m.shape[1] * 0.25) & (xs_ < m.shape[1] * 0.75)]
        yb = int(m.shape[0] * 0.6); kk = np.ones((1, 30 * SS), np.uint8)
        m[yb:] = cv2.morphologyEx(m[yb:].astype(np.uint8), cv2.MORPH_CLOSE, kk) > 0
        if len(gap):
            gx0, gx1 = gap.min(), gap.max(); gc = (gx0 + gx1) / 2; hw = (gx1 - gx0) / 2
            for y in range(0, yr):
                w_ = hw * (0.18 + 0.82 * y / yr)
                m[y, int(gc - w_):int(gc + w_) + 1] = False
    X = int(x0 * SS) - pad + int(R * SS); Y = 0 - pad + int(R * SS)
    ys, xs = np.where(m)
    yy, xx = ys + Y, xs + X; ok = (yy >= 0) & (yy < flat.shape[0]) & (xx >= 0) & (xx < flat.shape[1])
    flat[yy[ok], xx[ok]] = 1
# solve R so O stems (after its own resize) reach STEM
w = L[1][2] - L[1][1]; R = 20.0
for _ in range(8):
    m = cv2.resize(gO.astype(np.uint8) * 255, (int(w - 2 * R), int(HF - 2 * R)), interpolation=cv2.INTER_AREA) > 127
    s = stem_of(m) * (HF / (HF)) ; R = max(0, (STEM * HF / 870 - s) / 2)
print("R", R)
for ch, a, b in L: place(ch, a, b, R)
# --- envelope warp to output src coords
OH = int((np.polyval(pb, W0 / 2) + 300) * SS); OH = max(OH, int((H0 + 100) * SS))
Xo = (np.arange(W0 * SS) + 0.5) / SS
top = np.polyval(pt, Xo); bot = np.polyval(pb, Xo)
Yo = (np.arange(OH) + 0.5) / SS
v = (Yo[:, None] - top[None, :]) / (bot - top)[None, :] * HF
mapx = np.broadcast_to(np.arange(W0 * SS, dtype=np.float32)[None, :], (OH, W0 * SS)).copy()
mapy = (v * SS - 0.5).astype(np.float32)
flat = cv2.morphologyEx(flat, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
out = cv2.remap(flat * 255, mapx, mapy, cv2.INTER_LINEAR, borderValue=0) > 127
cs, hier = cv2.findContours(out.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
polys = []
for i, c in enumerate(cs):
    if cv2.contourArea(c) < 100: continue
    p = cv2.approxPolyDP(c, 1.5, True)[:, 0, :].astype(np.float64) / SS
    polys.append((p, hier[0][i][3] >= 0))
pickle.dump(polys, open(os.environ.get("OUTP", "letters_polys.pkl"), "wb"))
Image.fromarray((out * 255).astype(np.uint8)).resize((2048, int(OH / SS / 2))).save("letters_preview.png")
np.save(os.environ.get("OUTM","font_mask_src.npy"), cv2.resize(out.astype(np.uint8), (W0, int(OH / SS)), interpolation=cv2.INTER_AREA) > 0)
