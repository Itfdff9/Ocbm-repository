"""NSE Laura/Khloe-style: straighten the megaphone into a clean cone and centre NORTH SHORE inside it.
- top band (outline + checkers + blue lines) and bottom band continue in straight lines to a round mouthpiece
  (continued by perspective-scaled copies of the original band art, no kink)
- NORTH SHORE (original lettering pixels) re-fitted inside the middle black band with even margins
- bell, banner, pom-pom, star and background kept from the original; star stays in front
Geometry measured on the 5016 px master; works on any square size of the same art.
usage: python3 straighten_megaphone.py IN.png OUT.png"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd

To  = (998.0, -0.1002); Bb = (1286.5, -0.198); L2t = (2275.0, -0.410); Bo = (2656.6, -0.570)
def ly(L, x): return L[0] + L[1] * (x - 2500.0)
VT = (5449.9, 702.4); KT, XST = 1.0964, 3017.0      # top band: vanishing pt, period ratio, seam
VB = (4885.0, 1297.15); KB, XSB = 1.1380, 3055.0    # bottom band
XE = 4470.0                                          # mouthpiece centre x
AX = np.array([0.084, 0.996]); NX = np.array([0.996, -0.084])   # bell/mouthpiece major axis, axis normal
BLACK = np.array([7, 7, 9], np.float32); WHITE = np.array([248, 248, 248], np.float32)

def run(img):
    full = np.array(img.convert("RGBA"))
    H0, W0 = full.shape[:2]; s = W0 / 5016.0
    x0, x1, y0, y1 = int(900 * s), W0, int(650 * s), int(2960 * s)
    a = full[y0:y1, x0:x1].astype(np.float32); orig = a.copy()
    H, W = a.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    X = (xx + x0 + 0.5) / s; Y = (yy + y0 + 0.5) / s              # master units
    del xx, yy
    def P(xm, ym): return int(ym * s) - y0, int(xm * s) - x0      # master -> (row, col) in the ROI
    tT, tB = ly(To, X), ly(Bb, X); tL, tO = ly(L2t, X), ly(Bo, X)
    yE0, yE1 = ly(To, XE), ly(Bo, XE); C = np.array([XE, (yE0 + yE1) / 2])
    along = (X - C[0]) * NX[0] + (Y - C[1]) * NX[1]               # >0 = beyond mouthpiece centre line
    body_x = along <= 0

    # ---- bell (kept): interior + white rim + outer black
    dark = (orig[..., :3].max(2) < 100) & (orig[..., 3] > 128)
    dk = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(25 * s) | 1,) * 2))
    lab, _ = nd.label(dk); interior = lab == lab[P(1180, 2200)]
    ring = nd.binary_dilation(interior, iterations=int(48 * s)) & ~interior
    wh = (orig[..., :3].min(2) > 160) & ring
    bell = nd.binary_fill_holes(interior | wh)
    bell = nd.binary_dilation(bell, iterations=int(22 * s))
    bellR = bell

    # ---- text layer from the original (NORTH, SHORE)
    rgb0 = orig[..., :3]; mx0 = rgb0.max(2)
    blue0 = (rgb0[..., 2] > 140) & (rgb0[..., 2] - rgb0[..., 0] > 70)
    nond = (mx0 >= 80) & (orig[..., 3] > 128)
    roi = np.zeros((H, W), bool); r0, c0 = P(1150, 950); r1, c1 = P(4400, 2650); roi[r0:r1, c0:c1] = True
    lab, n = nd.label(nond & roi); idx = np.arange(1, n + 1)
    sz = nd.sum(nond & roi, lab, idx); bl = nd.sum(blue0, lab, idx)
    com = nd.center_of_mass(nond & roi, lab, idx)
    words = {}
    for i, z, b_, c in zip(idx, sz, bl, com):
        if z > 300000 * s * s and 0.3 < b_ / z < 0.6:
            cx = (c[1] + x0) / s
            if 1800 < cx < 2700: words["NORTH"] = i
            elif 3300 < cx < 4000: words["SHORE"] = i
    assert len(words) == 2, words
    env = {}; tmask = np.zeros((H, W), bool)
    for w, i in words.items():
        m = nd.binary_fill_holes(lab == i); m = nd.binary_dilation(m, iterations=max(1, int(2 * s)))
        tmask |= m
        ys, xs = np.where(m); ys = ys + y0; xs = xs + x0; xs_m = (xs + 0.5) / s; ys_m = (ys + 0.5) / s
        cols = np.unique(xs)
        tops = np.array([ys[xs == c].min() for c in cols[::max(1, int(4 * s))]]); bots = np.array([ys[xs == c].max() for c in cols[::max(1, int(4 * s))]])
        cx_ = (cols[::max(1, int(4 * s))] + 0.5) / s
        pt = np.polyfit(cx_, (tops + 0.5) / s, 1); pb = np.polyfit(cx_, (bots + 0.5) / s, 1)
        pt[1] += ((tops + 0.5) / s - np.polyval(pt, cx_)).min()      # envelope above every letter
        pb[1] += ((bots + 0.5) / s - np.polyval(pb, cx_)).max()      # envelope below every letter
        env[w] = (pt, pb, xs_m.min(), xs_m.max())
    sx0 = env["NORTH"][2]; sx1 = env["SHORE"][3]; split = (env["NORTH"][3] + env["SHORE"][2]) / 2
    text_p = orig.copy(); text_p[..., 3] = np.where(tmask, orig[..., 3], 0); text_p[..., :3] *= text_p[..., 3:4] / 255.0

    out = a
    # ---- middle band: black
    mid = (Y > tB) & (Y < tL) & body_x & ~bellR & (X > 1250)
    out[mid, :3] = BLACK; out[mid, 3] = 255

    # ---- band continuation by perspective copy (one sampling from the original)
    def band_cont(V, XS, rows, TL, Ltop, Lbot, sel):
        """continue a band past XS. Stripes outside the checker rows come from the band profile at XS
        (kept at the same place across the band, so they shrink with it); each checker row repeats its
        own two-square tile from the original, squares shrinking with perspective, row edges aligned."""
        tb = (Y - ly(Ltop, X)) / (ly(Lbot, X) - ly(Ltop, X))
        hX = ly(Lbot, XS) - ly(Ltop, XS)
        tT, tM, tB = [(ly(L, XS) - ly(Ltop, XS)) / hX for L in TL]
        xs_ = np.full(X.shape, XS - 1.0, np.float32); ys_ = ly(Ltop, xs_) + tb * hX
        for (lo, hi, Ltop_r, Lbot_r, BREF, P, XSr) in ((tT, tM, TL[0], TL[1]) + rows[0], (tM, tB, TL[1], TL[2]) + rows[1]):
            m = (tb >= lo) & (tb < hi)
            phi = (V[0] - XSr) / P * np.log((V[0] - XSr) / np.maximum(V[0] - X, 1.0))
            xt = BREF + np.mod(phi, 2.0) * P
            c = (tb - lo) / (hi - lo)
            yt = ly(Ltop_r, xt) + c * (ly(Lbot_r, xt) - ly(Ltop_r, xt))
            xs_ = np.where(m, xt, xs_); ys_ = np.where(m, yt, ys_)
        mx_ = (xs_ * s - 0.5 - x0).astype(np.float32); my_ = (ys_ * s - 0.5 - y0).astype(np.float32)
        smp = cv2.remap(orig, mx_, my_, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        out[sel] = smp[sel]; out[sel, 3] = 255
    def L29(y29, m): return (y29 - m * 400.0, m)     # line given at x=2900 -> at 2500
    TOPL = (L29(1061.63, -0.1186), L29(1132.07, -0.1310), L29(1195.82, -0.2060))
    BOTL = (L29(2145.54, -0.4031), L29(2246.15, -0.4542), L29(2312.11, -0.5484))
    XST2, XSB2 = 3014.0, 3048.0
    selT = (X >= XST2) & (Y >= tT) & (Y <= tB) & body_x
    selB = (X >= XSB2) & (Y >= tL) & (Y <= tO) & body_x
    band_cont(VT, XST2, ((2781.0, 118.0, 3017.0), (2787.0, 116.5, 3020.0)), TOPL, To, Bb, selT)
    band_cont(VB, XSB2, ((2798.0, 126.5, 3051.0), (2807.0, 126.0, 3059.0)), BOTL, L2t, Bo, selB)

    # ---- clear the old pointed tip / old cap outside the new body
    poly = np.array([(3880, 840), (4480, 880), (4740, 960), (4740, 1290), (4600, 1330), (4300, 1500), (4140, 1680), (3900, 1790)], np.float32)
    pm = np.zeros(out.shape[:2], np.uint8); cv2.fillPoly(pm, [((poly * s) - [x0, y0]).astype(np.int32)], 1)
    newbody = (Y >= tT) & (Y <= tO) & body_x & (X > 1250)

    # ---- mouthpiece: ellipse at XE
    a_out = (yE1 - yE0) / 2 * 1.002; b_out = 0.24 * a_out
    u = (X - C[0]) * NX[0] + (Y - C[1]) * NX[1]; v = (X - C[0]) * AX[0] + (Y - C[1]) * AX[1]
    e = (u / b_out) ** 2 + (v / a_out) ** 2
    def ein(d): return (u / max(b_out - d, 1)) ** 2 + (v / max(a_out - d, 1)) ** 2
    cap = e <= 1
    clear = (pm > 0) & ~newbody & ~cap  # star is pasted back afterwards
    out[clear] = 0
    capc = np.zeros(out.shape[:2] + (3,), np.float32); capc[:] = BLACK
    whr = cap & (ein(22) <= 1) & (ein(22 + 30) > 1)
    capc[whr] = WHITE
    out[cap, :3] = capc[cap]; out[cap, 3] = 255

    # ---- NORTH SHORE re-fitted in the middle band
    xd0, xd1 = 1545.0, 4255.0
    xsrc = sx0 + (X - xd0) * (sx1 - sx0) / (xd1 - xd0)
    hb = tL - tB; dt = tB + 0.075 * hb; db = tL - 0.075 * hb
    t = (Y - dt) / (db - dt)
    isN = xsrc < split
    ptN, pbN = env["NORTH"][0], env["NORTH"][1]; ptS, pbS = env["SHORE"][0], env["SHORE"][1]
    st = np.where(isN, np.polyval(ptN, xsrc), np.polyval(ptS, xsrc)); sb = np.where(isN, np.polyval(pbN, xsrc), np.polyval(pbS, xsrc))
    ysrc = st + t * (sb - st)
    ok = (X >= xd0 - 20) & (X <= xd1 + 20) & (t > -0.05) & (t < 1.05)
    mxp = np.where(ok, xsrc * s - 0.5 - x0, -10).astype(np.float32); myp = np.where(ok, ysrc * s - 0.5 - y0, -10).astype(np.float32)
    tl = cv2.remap(text_p, mxp, myp, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ta = np.clip(tl[..., 3:4] / 255.0, 0, 1)
    out[..., :3] = np.clip(tl[..., :3], 0, 255) + out[..., :3] * (1 - ta)
    out[..., 3] = np.maximum(out[..., 3], tl[..., 3])

    # ---- star in front (original star art + its black outline)
    sroi = np.zeros((H, W), bool); r0, c0 = P(4380, 1150); r1, c1 = P(5100, 1950); sroi[r0:r1, c0:c1] = True
    wht = (rgb0.min(2) > 160) & (orig[..., 3] > 128) & sroi
    lab, n = nd.label(wht); szs = nd.sum(wht, lab, range(1, n + 1))
    star_in = nd.binary_fill_holes(lab == (np.argmax(szs) + 1))          # white star ring + everything inside
    star_out = nd.binary_dilation(star_in, iterations=int(26 * s))
    so, si = star_out, star_in
    out[so & ~si, :3] = BLACK; out[so & ~si, 3] = 255
    out[si] = orig[si]

    solid = a[..., 3] >= 128
    # drop loose specks left around the new mouthpiece
    reg = np.zeros((H, W), bool); r0, c0 = P(4380, 750); r1, c1 = P(4800, 1450); reg[r0:r1, c0:c1] = True
    lab, n = nd.label(solid); szs = nd.sum(solid, lab, range(1, n + 1))
    small = np.isin(lab, np.where(szs < 2500 * s * s)[0] + 1) & reg
    solid &= ~small
    o = np.zeros((H, W, 4), np.uint8); o[..., :3] = np.clip(a[..., :3], 0, 255).astype(np.uint8)
    o[..., 3] = np.where(solid, 255, 0); o[~solid, :3] = 0
    full[y0:y1, x0:x1] = o
    fa = full[..., 3] >= 128; full[..., 3] = np.where(fa, 255, 0); full[~fa, :3] = 0
    return Image.fromarray(full, "RGBA")

if __name__ == "__main__":
    im = Image.open(sys.argv[1]); dpi = im.info.get("dpi", (300, 300))
    run(im).save(sys.argv[2], dpi=dpi); print(sys.argv[2], im.size)
