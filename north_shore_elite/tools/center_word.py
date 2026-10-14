"""NSE banner: script word 'Elites' -> 'Elite' with NO tail, centred in the banner (between the two hearts).
Works from the ORIGINAL 5016 px banner art. The word layer (letters + their blue rim + white halo) is cut out,
the s and the long swash are dropped, the e gets a round end, the empty banner is rebuilt (band-follow clone for the
border, banner texture for the inside), and the word is put back centred, following the banner's arc.
usage: python3 center_word.py IN.png OUT.png laura|sag [preview.png]"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
src, out, kind = sys.argv[1], sys.argv[2], sys.argv[3]
CFG = {
 "sag":   dict(roi=(600, 2900, 4950, 4560), topA=(2600, 3226), botA=(2600, 4107),
               Q=(3342.9, 3824.3), qdir=(0.72, -0.69), epoint=(3150, 3700), Ecomp=(1499, 3650),
               idot=(2445, 3440), hearts=(1043, 4163)),
 "laura": dict(roi=(600, 2650, 4950, 4150), topA=(2600, 2966), botA=(2600, 3881),
               Q=(3342.4, 3601.8), qdir=(0.70, -0.71), epoint=(3150, 3500), Ecomp=(1550, 3500),
               idot=(2525, 3180), hearts=(1168, 4178)),
}[kind]
im = Image.open(src).convert("RGBA"); FULL = np.array(im)
X0, Y0, X1, Y1 = CFG["roi"]
A = FULL[Y0:Y1, X0:X1].astype(np.float32); rgb = A[..., :3].copy(); H, W = rgb.shape[:2]
L = lambda p: (p[0] - X0, p[1] - Y0)
mx, mn = rgb.max(2), rgb.min(2); R_, B_ = rgb[..., 0], rgb[..., 2]
dark = mx < 75; blue = (B_ > 140) & (B_ - R_ > 80); light = mn > 165
core = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

# ---------- word cores
cut_d = np.array(CFG["qdir"], float); cut_d /= np.linalg.norm(cut_d); cut_n = np.array([-cut_d[1], cut_d[0]])
Q = np.array(L(CFG["Q"]))
alQ = (xx - Q[0]) * cut_d[0] + (yy - Q[1]) * cut_d[1]; acQ = (xx - Q[0]) * cut_n[0] + (yy - Q[1]) * cut_n[1]
band = (np.abs(alQ) <= 3) & (np.abs(acQ) <= 90)
lab, n = nd.label(core & ~band)
def near_id(p, r=80):
    x, y = L(p); w = lab[y - r:y + r, x - r:x + r]; v = w[w > 0]
    if not len(v): return 0
    yy_, xx_ = np.where(w > 0); k = np.argmin((yy_ - r) ** 2 + (xx_ - r) ** 2); return w[yy_[k], xx_[k]]
idE = near_id(CFG["Ecomp"]); idM = near_id(CFG["epoint"])
ix, iy = L(CFG["idot"]); win = lab[iy - 40:iy + 40, ix - 40:ix + 40]; ids = [v for v in np.unique(win) if v and v not in (idE, idM)]
idI = max(ids, key=lambda v: (win == v).sum()) if ids else 0
print("ids E/main/idot", idE, idM, idI)
dt = nd.distance_transform_edt(core); rr = float(dt[band & core].max())
cap = ((xx - Q[0]) ** 2 + (yy - Q[1]) ** 2 <= rr ** 2) & core & (alQ >= -2)
keep = (lab == idE) | (lab == idM) | (lab == idI) | (band & core & (alQ <= 0)) | cap
keep = nd.binary_fill_holes(keep) & core | cap
# everything that was the word before (incl. s + swash): core components touching the s side of the cut
s_ids = [v for v in np.unique(lab[nd.binary_dilation(band, iterations=4) & core]) if v and v not in (idE, idM, idI)]
oldword = keep | np.isin(lab, s_ids) | (band & core)
# include anything connected to the swash beyond (whole s component already)
d_keep = nd.distance_transform_edt(~keep); d_old = nd.distance_transform_edt(~oldword); d_rem = nd.distance_transform_edt(~(oldword & ~keep))
RIM = float(np.percentile(d_keep[blue & (d_keep < 60) & (d_keep > 0)], 90)); HAL = 13.0; D = RIM + HAL
print("rr %.1f RIM %.1f D %.1f" % (rr, RIM, D))
# ---------- banner interior = region enclosed by the surrounding black (largest dark component)
labk, nk = nd.label(core); szk = np.bincount(labk.ravel()); szk[0] = 0
wordids = np.unique(labk[oldword & core]); wordids = wordids[wordids > 0]
bigids = np.where(szk > 40000)[0]; bigids = np.setdiff1d(bigids, wordids)
bg = np.isin(labk, bigids)
labi, _ = nd.label(~bg); interior_full = nd.binary_fill_holes(labi == labi[L(CFG["topA"])[1] + 60, L(CFG["topA"])[0]])
# blue line pixels of the banner edge (blue touching the background black)
edgeblue = blue & (nd.distance_transform_edt(~bg) < float(__import__("os").environ.get("EB", "34")))
# ---------- word layer (from original pixels) with synthetic rim where the connector was
layer_m = d_keep <= D + 1
synth = layer_m & (d_rem < d_keep + 2) & (np.hypot(xx - Q[0], yy - Q[1]) < 160)
rim_col = np.median(rgb[blue & (d_keep <= RIM) & (d_keep > 3) & ~synth], 0)
hal_sel = (d_keep > RIM + 2) & (d_keep <= D) & light & ~synth
hal_col = np.median(rgb[hal_sel], 0) if hal_sel.any() else np.array([245, 245, 248.])
blk_col = np.median(rgb[keep & dark], 0)
lay = rgb.copy()
s3 = synth[..., None]
wr = np.clip(RIM + 0.5 - d_keep, 0, 1)[..., None]
lay = np.where(s3, hal_col * (1 - wr) + rim_col * wr, lay)
lay[synth & keep] = blk_col
lay[cap] = np.where(dark[cap][:, None], rgb[cap], blk_col)
alpha = np.clip(D + 1.5 - d_keep, 0, 1)
# ---------- empty banner rebuild over the old word footprint
Rz = d_old <= D + 4
Rz = nd.binary_dilation(Rz, iterations=2)
d_bg = nd.distance_transform_edt(~bg)
EDGE = float(__import__("os").environ.get("EDGEW", "48"))
interior = interior_full & (d_bg > EDGE)
out = rgb.copy()
print("word footprint outside banner interior px", int((Rz & ~interior_full).sum()))
# (a) edge band under the old word: fit the banner's inner blue line (top & bottom) from the clean columns,
#     then copy each missing pixel from the nearest clean column at the same offset from that line.
labB, _ = nd.label(blue); szB = np.bincount(labB.ravel()); bigblue = blue & (szB[labB] > 3000)
cand = bigblue & (d_bg < 110) & ~nd.binary_dilation(Rz, iterations=12)
colr = np.arange(W)
tops, bots = {}, {}
topc = np.full(W, np.nan); botc = np.full(W, np.nan)
for c in range(0, W, 4):
    r_ = np.where(interior_full[:, c])[0]
    if len(r_) < 300: continue
    m_ = (r_.min() + r_.max()) / 2; topc[c] = r_.min(); botc[c] = r_.max()
    bt = np.where(cand[:, c])[0]
    up = bt[bt < m_]; dn = bt[bt > m_]
    if len(up):
        y_ = up.max() + 1
        if light[y_ + 2:y_ + 30, c].mean() > 0.55 and interior_full[y_ + 10, c]: tops[c] = y_
    if len(dn):
        y_ = dn.min() - 1
        if light[max(0, y_ - 29):y_ - 1, c].mean() > 0.55 and interior_full[y_ - 10, c]: bots[c] = y_
from scipy.interpolate import PchipInterpolator
def rfit(dct):
    xs_ = np.array(sorted(dct), float); ys2 = np.array([dct[int(k)] for k in xs_], float)
    # robust: compare with running median, drop jumps
    med = nd.median_filter(ys2, size=9, mode="nearest"); k = np.abs(ys2 - med) < 6; xs_, ys2 = xs_[k], ys2[k]
    bx, by = [], []
    for b0 in np.arange(xs_.min(), xs_.max() + 1, 32):
        m_ = (xs_ >= b0) & (xs_ < b0 + 32)
        if m_.sum() >= 3: bx.append(xs_[m_].mean()); by.append(np.median(ys2[m_]))
    bx, by = np.array(bx), np.array(by)
    # drop bins that break smoothness (2nd difference outliers)
    for _ in range(3):
        if len(bx) < 5: break
        nb = np.r_[by[1], (by[:-2] + by[2:]) / 2, by[-2]]
        d2 = np.abs(by - nb)
        k = d2 < max(8, np.percentile(d2, 90) * 1.5); bx, by = bx[k], by[k]
    f = PchipInterpolator(bx, by, extrapolate=True)
    print("edge bins", len(bx), "x %d-%d" % (bx.min() + X0, bx.max() + X0))
    return lambda x: f(np.clip(x, bx.min(), bx.max())) + (np.clip(x, bx.min(), bx.max()) != x) * 0
yT = rfit(tops); yB = rfit(bots); 
# the word's sticker (white halo + black outline) where it sticks out over the top border
above = yy < yT(xx) + 6
halo_out = above & light & (d_old <= 110) & (d_old > RIM - 2)
labh, _ = nd.label(halo_out | (above & (d_old <= RIM + 2)))
hid = np.unique(labh[above & (d_old <= RIM + 2) & (d_old > 0)]); hid = hid[hid > 0]
sticker_old = nd.binary_fill_holes(np.isin(labh, hid) | oldword)
sticker_old = nd.binary_dilation(sticker_old, iterations=int(__import__('os').environ.get('SDIL', '45'))) & (above | (d_old <= D + 4))
Rz = Rz | sticker_old
sticker_new = nd.binary_fill_holes(np.isin(labh, hid) | keep) & (d_keep <= 110) & (d_keep < d_rem + 4)
sticker_new = nd.binary_dilation(sticker_new, iterations=10) & above & (d_keep <= 110)
lay_sel = sticker_new & ~synth
alpha = np.maximum(alpha, np.where(lay_sel, 1.0, 0.0))
print("sticker px old %d new %d" % (int(sticker_old.sum()), int(sticker_new.sum())))
interior = interior_full & (yy > yT(xx) + 4) & (yy < yB(xx) - 4)
bord = Rz & ~interior
ys_, xs_ = np.where(bord)
midc = np.interp(np.arange(W), np.where(~np.isnan(topc))[0], ((topc + botc) / 2)[~np.isnan(topc)])
use_top = ys_ < np.minimum(midc[xs_], (yT(xs_) + yB(xs_)) / 2 + 400)
off = np.where(use_top, ys_ - yT(xs_), ys_ - yB(xs_))
filled = np.zeros(len(ys_), bool); Rzd = nd.binary_dilation(Rz, iterations=3)
for k in range(2, 3000, 2):
    for sgn in (1, -1):
        rem = ~filled
        pass
        if not rem.any(): break
        sx = xs_[rem] + sgn * k
        ok = (sx >= 0) & (sx < W)
        sxc = np.clip(sx, 0, W - 1)
        sy = np.where(use_top[rem], yT(sxc) + off[rem], yB(sxc) + off[rem]).round().astype(int)
        ok &= (sy >= 0) & (sy < H); syc = np.clip(sy, 0, H - 1)
        ok &= ~Rzd[syc, sxc]
        idx = np.where(rem)[0][ok]
        out[ys_[idx], xs_[idx]] = rgb[syc[ok], sxc[ok]]
        filled[idx] = True
    if filled.all(): break
print("edge px rebuilt", len(ys_), "unfilled", int((~filled).sum()))
# top edge: plain blue line (same thickness as next to it) + solid border black above it
LW = float(__import__('os').environ.get('LW', '40'))
blk_bg = np.median(rgb[bg & ~nd.binary_dilation(Rz, iterations=5) & (np.abs(yy - yT(xx)) < 200)], 0)
tsel = use_top & (off < -LW)
out[ys_[tsel], xs_[tsel]] = blk_bg

# ghost lines left in the black just outside the old sticker
o_mx = out.max(2); o_mn = out.min(2)
darkfrac = cv2.blur((o_mx < 60).astype(np.float32), (25, 25))
ghost = (yy < yT(xx)) & (d_old <= 200) & (o_mx >= 12) & (o_mx - o_mn < 60) & (darkfrac > 0.75)
ghost = nd.binary_dilation(ghost, iterations=2) & (yy < yT(xx)) & (d_old <= 200) & (darkfrac > 0.6)
out[ghost] = blk_bg if 'blk_bg' in dir() else np.array([8, 8, 10.])
print("ghost px", int(ghost.sum()))
# (b) interior: texture mosaic from clean banner interior
inside = Rz & interior
labd, _ = nd.label(dark); szd = np.bincount(labd.ravel()); labb, _ = nd.label(blue); szb = np.bincount(labb.ravel())
small = (dark & (szd[labd] <= 400)) | (blue & (szb[labb] <= 400))
okp = interior & ~nd.binary_dilation(Rz, iterations=6) & (light | nd.binary_dilation(small, iterations=2))
okp &= ~nd.binary_dilation((dark & (szd[labd] > 400)) | (blue & (szb[labb] > 400)), iterations=8)
PS = 36; ii = cv2.integral(okp.astype(np.uint8))
cands = np.array([(y, x) for y in range(0, H - PS, 5) for x in range(0, W - PS, 5)
                  if ii[y + PS, x + PS] - ii[y, x + PS] - ii[y + PS, x] + ii[y, x] == PS * PS])
print("texture windows", len(cands))
rng = np.random.default_rng(5); best = np.zeros((H, W), np.float32); mos = rgb.copy()
wv0 = np.outer(np.hanning(PS + 2)[1:-1], np.hanning(PS + 2)[1:-1]).astype(np.float32) + 1e-3
zy, zx = np.where(inside)
for tyy in range(max(0, zy.min() - PS), zy.max() + 1, 14):
    for txx in range(max(0, zx.min() - PS), zx.max() + 1, 14):
        if not inside[tyy:tyy + PS, txx:txx + PS].any(): continue
        dd = np.hypot(cands[:, 0] - tyy, cands[:, 1] - txx); pool = cands[dd <= np.percentile(dd, 20)]
        sy, sx = pool[rng.integers(len(pool))]; h = min(PS, H - tyy); w = min(PS, W - txx)
        wv = wv0[:h, :w] * (0.9 + 0.2 * rng.random()); upd = wv > best[tyy:tyy + h, txx:txx + w]
        mos[tyy:tyy + h, txx:txx + w][upd] = rgb[sy:sy + h, sx:sx + w][upd]
        best[tyy:tyy + h, txx:txx + w] = np.maximum(best[tyy:tyy + h, txx:txx + w], wv)
out[inside] = mos[inside]
# keep the hearts / stars etc. that sit inside Rz but are not word: anything big not belonging to the word stays
other = ((dark & (szd[labd] > 400)) | (blue & (szb[labb] > 400))) & (d_old > D + 1)
other &= ~bg & interior_full
other = nd.binary_dilation(other, iterations=2) & (d_old > D - 4) & interior_full
out[other & Rz] = rgb[other & Rz]
print('Rz outside interior px', int((Rz & ~interior).sum()))
# ---------- place the word: centre between the hearts, follow the arc
ys_k, xs_k = np.where(keep)
ocx = (xs_k.min() + xs_k.max()) / 2; ocy = (ys_k.min() + ys_k.max()) / 2
h0, h1 = CFG["hearts"]; ncx = (h0 + h1) / 2 - X0
cols = np.arange(W); topc = np.full(W, np.nan); botc = np.full(W, np.nan)
for c in range(0, W, 10):
    r_ = np.where(interior_full[:, c])[0]
    if len(r_) > 200: topc[c] = r_.min(); botc[c] = r_.max()
okc = ~np.isnan(topc); okc &= (cols > (CFG["hearts"][0] - X0)) & (cols < (CFG["hearts"][1] - X0))
pm = np.polyfit(cols[okc], (topc[okc] + botc[okc]) / 2, 3)
mid = lambda x: np.polyval(pm, x)
ang = lambda x: np.degrees(np.arctan((mid(x + 50) - mid(x - 50)) / 100))
ncy = ocy + mid(ncx) - mid(ocx); rot = ang(ncx) - ang(ocx)
print("move dx %.0f dy %.0f rot %.2f" % (ncx - ocx, ncy - ocy, rot))
M = cv2.getRotationMatrix2D((ocx, ocy), -rot, 1.0); M[0, 2] += ncx - ocx; M[1, 2] += ncy - ocy
premul = np.dstack([lay * alpha[..., None], alpha])
wl = cv2.warpAffine(premul, M, (W, H), flags=cv2.INTER_CUBIC, borderValue=0)
wa = np.clip(wl[..., 3], 0, 1)
res = wl[..., :3] + out * (1 - wa[..., None])
FULL2 = FULL.copy(); FULL2[Y0:Y1, X0:X1, :3] = np.clip(res, 0, 255).astype(np.uint8)
Image.fromarray(FULL2, "RGBA").save(out_path := out if False else sys.argv[2], dpi=im.info.get("dpi", (300, 300)))
if len(sys.argv) > 4:
    pv = np.clip(res, 0, 255).astype(np.uint8).copy()
    for x in range(0, W, 4):
        for f_ in (mid, yT, yB):
            y = int(f_(x))
            if 0 <= y < H: pv[max(0, y - 2):y + 2, x] = (255, 0, 200)
    Image.fromarray(pv).resize((W // 3, H // 3)).save(sys.argv[4])
print("saved", sys.argv[2])
