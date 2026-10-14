"""NSE banner script 'Elites' -> 'Elite' (5016 px masters), keeping the original letters and the original swash.
The 's' body and the e->s connector are removed; the e's bottom stroke now flows straight into the
original swash tail (a smooth bridge stroke drawn with the same black core width and blue rim), so the word
still fills the banner the same way. The banner is refilled with its own speckle texture.
usage: python3 elite_fix.py IN.png OUT.png laura|sag"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
src, out, kind = sys.argv[1], sys.argv[2], sys.argv[3]
CFG = {"laura": dict(Q=(3300, 3560), qdir=(0.70, -0.71), S0=3900, send=(4000, 3322), epoint=(3150, 3500),
                     box=(2700, 2900, 4400, 3850), tex=(1000, 2900, 4400, 3850)),
       "sag":   dict(Q=(3310, 3790), qdir=(0.72, -0.69), S0=3880, send=(3980, 3571), epoint=(3150, 3700),
                     box=(2700, 3150, 4400, 4100), tex=(900, 3150, 4400, 4100))}[kind]
im = Image.open(src).convert("RGBA"); FULL = np.array(im)
X0, Y0, X1, Y1 = CFG["tex"]
A = FULL[Y0:Y1, X0:X1].astype(np.float32); rgb = A[..., :3].copy()
H, W = rgb.shape[:2]
def L(p): return (p[0] - X0, p[1] - Y0)
mx, mn = rgb.max(2), rgb.min(2); R_, G_, B_ = rgb[..., 0], rgb[..., 1], rgb[..., 2]
dark = mx < 75; blue = (B_ > 140) & (B_ - R_ > 80)
core = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
def snap(p, d):
    """centre + width of the core stroke through p, measured across direction d"""
    d = np.array(d, float); d /= np.linalg.norm(d); n = np.array([-d[1], d[0]])
    ts = np.arange(-80, 81); pts = np.array(p)[None] + ts[:, None] * n[None]
    v = core[np.clip(pts[:, 1].round().astype(int), 0, H - 1), np.clip(pts[:, 0].round().astype(int), 0, W - 1)]
    i0 = 80
    if not v[i0]:
        idx = np.where(v)[0]; i0 = idx[np.argmin(np.abs(idx - 80))]
    a = i0; b = i0
    while a > 0 and v[a - 1]: a -= 1
    while b < len(v) - 1 and v[b + 1]: b += 1
    c = (ts[a] + ts[b]) / 2
    return np.array(p) + c * n, (b - a + 1)
Q, wq = snap(L(CFG["Q"]), CFG["qdir"])
# swash start: vertical column at S0
sx = CFG["S0"] - X0; col = np.where(core[:, sx])[0]
sy_guess = L(CFG["send"])[1] + 0.35 * (CFG["send"][0] - CFG["S0"])
runs = np.split(col, np.where(np.diff(col) > 1)[0] + 1); run = min(runs, key=lambda r: abs(r.mean() - sy_guess))
S0 = np.array([sx, run.mean()])
nx_ = CFG["send"][0] - X0
col2 = np.where(core[:, nx_])[0]; runs2 = np.split(col2, np.where(np.diff(col2) > 1)[0] + 1)
run2 = min(runs2, key=lambda r: abs(r.mean() - L(CFG["send"])[1]))
S1 = np.array([nx_, run2.mean()])
t1 = (S1 - S0) / np.linalg.norm(S1 - S0)
ws = len(run) * abs(t1[0])
t0 = np.array(CFG["qdir"], float); t0 /= np.linalg.norm(t0)
print("Q", Q + [X0, Y0], "wq", wq, "S0", S0 + [X0, Y0], "ws %.1f" % ws, "t1", t1)
# ---- split the core with two cut bands
def band(p, d, half=90, thick=3, ret_ac=False):
    d = np.array(d, float); d /= np.linalg.norm(d); n = np.array([-d[1], d[0]])
    al = (xx - p[0]) * d[0] + (yy - p[1]) * d[1]; ac = (xx - p[0]) * n[0] + (yy - p[1]) * n[1]
    if ret_ac: return (np.abs(al) <= thick) & (np.abs(ac) <= half), al, ac
    return (np.abs(al) <= thick) & (np.abs(ac) <= half), al
bQ, alQ = band(Q, t0); bS, alS, acS = band(S0, t1, half=ws / 2 + 6, ret_ac=True)
lab, n = nd.label(core & ~bQ & ~bS)
eid = lab[int(L(CFG["epoint"])[1]), int(L(CFG["epoint"])[0])]; wid = lab[int(S1[1]), int(S1[0])]
assert eid > 0 and wid > 0 and eid != wid, (eid, wid)
# the removed part = everything connected to the band pieces between the cuts that is neither e-side nor swash end
mid = np.unique(lab[(bQ | bS) & core]); 
cand = [l for l in np.unique(lab[nd.binary_dilation(bQ | bS, iterations=4) & core]) if l not in (0, eid, wid)]
removed = np.isin(lab, cand) | ((bQ & core) & (alQ > 0)) | ((bS & core) & (alS < 0))
removed = nd.binary_fill_holes(removed | (bQ & core) | (bS & core)) & ~(lab == eid) & ~(lab == wid)
sw = (lab == wid) & ((np.abs(acS) <= ws / 2 + 8) | (alS > 70))
keep_core = (lab == eid) | sw | (bQ & core & (alQ <= 0)) | (bS & core & (alS >= 0))
removed |= ((lab == wid) & ~sw)
# ---- bridge stroke (cubic Bezier Q->S0) as a swept disk, width interpolated
Ld = np.linalg.norm(S0 - Q)
P0, P1, P2, P3 = Q, Q + t0 * Ld * 0.30, S0 - t1 * Ld * 0.40, S0
SS = 4
bridge = np.zeros((H * 1, W * 1), np.float32)
pts = []
for t in np.linspace(0, 1, 1200):
    p = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * P1 + 3 * (1 - t) * t ** 2 * P2 + t ** 3 * P3
    w = max(wq, ws) * (1 - 0.0 * t); pts.append((p, w / 2))
cen = np.zeros((H, W), np.float32); rad = np.full((H, W), 0, np.float32)
bm = np.zeros((H, W), bool)
for p, r in pts:
    x, y = p; ix, iy = int(x), int(y); R = int(r) + 3
    sl = (slice(max(0, iy - R), iy + R + 1), slice(max(0, ix - R), ix + R + 1))
    d = np.hypot(xx[sl] - x, yy[sl] - y)
    bridge[sl] = np.maximum(bridge[sl], np.clip(r + 0.5 - d, 0, 1))
bridge_m = bridge >= 0.5
new_core = keep_core | bridge_m
# ---- zones
dist_core0 = nd.distance_transform_edt(~core)
RIM = float(np.percentile(dist_core0[blue & (dist_core0 < 60) & (dist_core0 > 0)], 92))
D = RIM + 22
d_rem = nd.distance_transform_edt(~removed)
d_keep0 = nd.distance_transform_edt(~keep_core)
Z = (d_rem <= D) | ((d_rem <= D + 25) & (d_rem < d_keep0 - 4) & (blue | (mx < 200)))
# protect banner borders, hearts, stars: big dark/blue structures that are not this word
labd, _ = nd.label(dark)
word_ids = np.unique(labd[dark & (removed | keep_core)]); word_ids = word_ids[word_ids > 0]
labb, nb = nd.label(blue)
szb = nd.sum(blue, labb, range(1, nb + 1))
d_wordcore = nd.distance_transform_edt(~(removed | keep_core))
rim_ids = np.unique(labb[blue & (d_wordcore <= RIM + 2)]); rim_ids = rim_ids[rim_ids > 0]
szd = nd.sum(dark, labd, range(1, labd.max() + 1))
big_other = (dark & np.isin(labd, np.where(szd > 400)[0] + 1) & ~np.isin(labd, word_ids)) | \
            (blue & np.isin(labb, np.where(szb > 400)[0] + 1) & ~np.isin(labb, rim_ids))
protect = nd.binary_dilation(big_other, iterations=5)
Z &= ~protect
d_new = nd.distance_transform_edt(~new_core)
Z &= ~(keep_core)
print("RIM %.1f D %.1f Z %d" % (RIM, D, Z.sum()))
# ---- texture windows from the whole banner
light = (mn > 170) & ((mx - mn) < 28)
bigstruct = nd.binary_dilation(big_other | core | nd.binary_dilation(blue & np.isin(labb, rim_ids), iterations=2), iterations=10)
small_d = dark & np.isin(labd, np.where(szd <= 400)[0] + 1)
small_b = blue & np.isin(labb, np.where(szb <= 400)[0] + 1)
allowed = light | nd.binary_dilation(small_d | small_b, iterations=2)
ok = ~bigstruct & ~Z & allowed
PS = 36
ii = cv2.integral(ok.astype(np.uint8)); li = cv2.integral(light.astype(np.uint8))
cands = []
for y in range(0, H - PS, 5):
    for x in range(0, W - PS, 5):
        if ii[y + PS, x + PS] - ii[y, x + PS] - ii[y + PS, x] + ii[y, x] == PS * PS:
            if li[y + PS, x + PS] - li[y, x + PS] - li[y + PS, x] + li[y, x] > 0.62 * PS * PS: cands.append((y, x))
cands = np.array(cands); print("texture windows", len(cands))
rng = np.random.default_rng(11)
acc = np.zeros((H, W, 3), np.float32); wsum = np.zeros((H, W), np.float32)
win = np.outer(np.hanning(PS + 2)[1:-1], np.hanning(PS + 2)[1:-1]).astype(np.float32) + 1e-3
zy, zx = np.where(Z); ST = 14
for ty in range(max(0, zy.min() - PS), zy.max() + 1, ST):
    for tx in range(max(0, zx.min() - PS), zx.max() + 1, ST):
        if not Z[ty:ty + PS, tx:tx + PS].any(): continue
        dd = np.hypot(cands[:, 0] - ty, cands[:, 1] - tx); pool = cands[dd <= np.percentile(dd, 30)]
        sy, sx2 = pool[rng.integers(len(pool))]
        h = min(PS, H - ty); w = min(PS, W - tx)
        acc[ty:ty + h, tx:tx + w] += rgb[sy:sy + h, sx2:sx2 + w] * win[:h, :w, None]
        wsum[ty:ty + h, tx:tx + w] += win[:h, :w]
tex = acc / np.maximum(wsum, 1e-6)[..., None]
# overlap-averaging blurs speckles -> pick the single patch with max weight instead (sharper): redo as max-weight mosaic
best = np.zeros((H, W), np.float32); mos = rgb.copy()
for ty in range(max(0, zy.min() - PS), zy.max() + 1, ST):
    for tx in range(max(0, zx.min() - PS), zx.max() + 1, ST):
        if not Z[ty:ty + PS, tx:tx + PS].any(): continue
        dd = np.hypot(cands[:, 0] - ty, cands[:, 1] - tx); pool = cands[dd <= np.percentile(dd, 30)]
        sy, sx2 = pool[rng.integers(len(pool))]
        h = min(PS, H - ty); w = min(PS, W - tx)
        wv = win[:h, :w] * (0.9 + 0.2 * rng.random())
        upd = wv > best[ty:ty + h, tx:tx + w]
        mos[ty:ty + h, tx:tx + w][upd] = rgb[sy:sy + h, sx2:sx2 + w][upd]
        best[ty:ty + h, tx:tx + w] = np.maximum(best[ty:ty + h, tx:tx + w], wv)
new = rgb.copy(); new[Z] = mos[Z]
# ---- redraw rim + core where the stroke changed (inside Z and on the bridge)
blue_col = np.median(rgb[blue & np.isin(labb, rim_ids) & ~Z], 0)
black_col = np.median(rgb[dark & keep_core], 0)
wb = np.clip(RIM + 0.5 - d_new, 0, 1)
paint = Z | nd.binary_dilation(bridge_m, iterations=int(RIM) + 12)
paint &= ~protect
new = np.where((paint & (wb > 0))[..., None], new * (1 - wb[..., None]) + blue_col * wb[..., None], new)
wk = np.where(bridge_m | (new_core & paint), 1.0, 0.0)
wk = np.maximum(wk * 0, bridge)  # antialiased bridge core
hal = (d_new > RIM + 1) & (d_new <= RIM + 16)
ring = hal & (mn > 120) & ~Z & nd.binary_dilation(keep_core, iterations=int(RIM) + 20)
halo_col = np.median(rgb[ring & (d_new <= RIM + 8)], 0) if ring.any() else np.array([248, 248, 250.])
wh = np.clip(RIM + 9.5 - d_new, 0, 1) * (d_new > RIM)
new = np.where((paint & (wh > 0))[..., None], new * (1 - wh[..., None]) + halo_col * wh[..., None], new)
core_paint = paint & new_core
new[core_paint] = black_col
edge = paint & ~new_core & (bridge > 0)
new[edge] = new[edge] * (1 - bridge[edge, None]) + black_col * bridge[edge, None]
zf = cv2.GaussianBlur((Z | paint).astype(np.float32), (0, 0), 1.0)[..., None]
res = rgb * (1 - zf) + new * zf
res[core_paint] = black_col
FULL2 = FULL.copy(); FULL2[Y0:Y1, X0:X1, :3] = np.clip(res, 0, 255).astype(np.uint8)
Image.fromarray(FULL2, "RGBA").save(out, dpi=im.info.get("dpi", (300, 300)))
print("saved", out)
