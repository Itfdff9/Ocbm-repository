"""Raiderettes uniform style engine.
Block last name: black sequin fill, gray inner line, black satin band, orange satin stitched border with running stitch,
sparkles + white distress.  Script "Raiderettes": red glitter core with white highlight, black keyline,
orange beveled border with rhinestone dots, soft shadow on the letters only.  Hard alpha output."""
import math, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS = None

FONTS = "/home/claude/ocbm-repository/raiderettes/shoulder_40x6/fonts/"
BLOCK = FONTS + "Graduate-Regular.ttf"
SCRIPT = FONTS + "MsMadi-Regular.ttf"

ORANGE = np.array([228, 52, 8], np.float32)       # median of the reference borders
ORANGE_HI = np.array([252, 128, 62], np.float32)
ORANGE_LO = np.array([140, 14, 0], np.float32)
RED_CORE = np.array([196, 18, 10], np.float32)
LV = np.array([0.55, 0.83], np.float32)          # light travels toward bottom-right (light at top-left)

def text_mask(txt, font, pad):
    l, t, r, b = font.getbbox(txt)
    m = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(m).text((pad - l, pad - t), txt, font=font, fill=255)
    return np.asarray(m) > 127

def crop_mask(m, pad):
    ys, xs = np.where(m)
    m = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return np.pad(m, pad)

def mixdist(outside):
    """distance from the shape for points outside it: mix of euclid and chessboard -> semi-mitred corners"""
    u = outside.astype(np.uint8)
    d2 = cv2.distanceTransform(u, cv2.DIST_L2, 5)
    dc = cv2.distanceTransform(u, cv2.DIST_C, 3)
    return 0.45 * d2 + 0.55 * dc

def normals(d, sigma):
    ds = cv2.GaussianBlur(d, (0, 0), sigma)
    gx = cv2.Sobel(ds, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(ds, cv2.CV_32F, 0, 1, ksize=3)
    n = np.sqrt(gx * gx + gy * gy) + 1e-6
    return gx / n, gy / n

def contour_dots(mask, step, radius, img, color, jitter=0, rng=None, alpha=1.0, inside=None):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    lay = np.zeros(mask.shape, np.uint8)
    for c in cs:
        c = c[:, 0, :]
        if len(c) < step: continue
        seg = np.sqrt(((np.diff(c, axis=0)) ** 2).sum(1)); s = np.concatenate([[0], np.cumsum(seg)])
        for t in np.arange(step / 2, s[-1], step):
            i = np.searchsorted(s, t); x, y = c[min(i, len(c) - 1)]
            cv2.circle(lay, (int(x), int(y)), max(1, int(radius)), 255, -1, lineType=cv2.LINE_AA)
    a = lay.astype(np.float32) / 255 * alpha
    if inside is not None: a *= inside
    img[:] = img * (1 - a[..., None]) + np.asarray(color, np.float32) * a[..., None]

def satin(shape, period, seed, angle=45):
    h, w = shape; y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    th = math.radians(angle)
    s = np.sin(2 * math.pi * (x * math.cos(th) + y * math.sin(th)) / period)
    rng = np.random.default_rng(seed)
    nz = cv2.GaussianBlur(rng.standard_normal(shape).astype(np.float32), (0, 0), period * 0.3)
    return 0.6 * s + 0.4 * nz / (nz.std() + 1e-6)

def sequins(shape, cell, seed, ox=0, oy=0):
    """black sequin grid with top-left corner highlights, like the reference fill"""
    h, w = shape; rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    gx = (x + ox) // cell; gy = (y + oy) // cell
    fx = ((x + ox) % cell) / cell; fy = ((y + oy) % cell) / cell
    nx, ny = gx.max() + 2, gy.max() + 2
    base = rng.uniform(8, 30, (ny, nx)).astype(np.float32)
    hi = rng.uniform(60, 210, (ny, nx)).astype(np.float32) * (rng.random((ny, nx)) < 0.82)
    b = base[gy, gx]; hv = hi[gy, gx]
    gap = (fx > 0.9) | (fy > 0.9)
    lw = 0.13
    corner = ((fy < lw) & (fx < 0.55) & (fx > 0.08)) | ((fx < lw) & (fy < 0.55) & (fy > 0.08))
    v = np.where(corner, hv, b)
    v = np.where(gap, 4, v)
    return np.repeat(v[..., None], 3, axis=2).astype(np.float32)

def distress(shape, scale, seed, density):
    """organic white chips / splatter, clustered, rough-edged (scale = 10U)"""
    rng = np.random.default_rng(seed); h, w = shape
    U = scale / 10.0
    sig = max(0.6, 0.32 * U)
    imp = np.zeros(shape, np.float32)
    n = max(1, int(w * h / (scale * scale) * density * 0.5))
    for _ in range(n):
        cx, cy = rng.integers(0, w), rng.integers(0, h)
        k = int(rng.integers(1, 14)); spread = rng.gamma(1.4, 0.45 * U)
        ang = rng.uniform(0, math.pi); el = rng.uniform(1, 3.5)
        for _j in range(k):
            a, b = rng.normal(0, spread * el), rng.normal(0, spread)
            x = int(cx + a * math.cos(ang) - b * math.sin(ang)); y = int(cy + a * math.sin(ang) + b * math.cos(ang))
            if 0 <= x < w and 0 <= y < h: imp[y, x] += 2 * math.pi * sig * sig * rng.uniform(0.6, 1.4)
    S = cv2.GaussianBlur(imp, (0, 0), sig)
    N = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), max(0.5, 0.15 * U))
    N = (N - N.mean()) / (N.std() + 1e-6)
    m = (S + 0.12 * N) > 0.5
    for _ in range(int(n * 0.12)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h)); L = rng.gamma(1.5, 1.2 * U); a = rng.uniform(0, math.pi)
        cv2.line(m.view(np.uint8), (x, y), (int(x + L * math.cos(a)), int(y + L * math.sin(a))), 1, max(1, int(0.15 * U)))
    return m

def sparkle(img, x, y, r):
    h, w = img.shape[:2]; R = int(r)
    x0, x1, y0, y1 = max(0, x - R), min(w, x + R + 1), max(0, y - R), min(h, y + R + 1)
    if x1 <= x0 or y1 <= y0: return
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32); dx = (xx - x) / r; dy = (yy - y) / r
    rr = np.sqrt(dx * dx + dy * dy) + 1e-6
    core = np.exp(-(rr / 0.16) ** 2) + 0.45 * np.exp(-(rr / 0.35) ** 2)
    ang = np.arctan2(dy, dx)
    rays = (np.abs(np.cos(4 * ang)) ** 40) * np.clip(1 - rr, 0, 1) ** 1.5
    rays2 = (np.abs(np.cos(4 * ang + math.pi / 4)) ** 60) * np.clip(1 - rr * 1.6, 0, 1) ** 2 * 0.7
    a = np.clip(core + rays + rays2, 0, 1)[..., None]
    img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + 255 * a

def band_shade(t, nx, ny, k):
    return 1 + k * np.cos(np.pi * np.clip(t, 0, 1)) * (nx * LV[0] + ny * LV[1])

_GW = {}
def glyph_norm(ch, cap_px):
    """glyph ink mask scaled uniformly so the ink height of 'H' equals cap_px"""
    if "H" not in _GW:
        mh = crop_mask(text_mask("H", ImageFont.truetype(BLOCK, 1000), 20), 0); _GW["H"] = mh.shape[0] / 1000
    size = int(cap_px / _GW["H"])
    m = crop_mask(text_mask(ch, ImageFont.truetype(BLOCK, size), 20), 0)
    return m

def glyph_w(ch, cap_px):
    return glyph_norm(ch, cap_px).shape[1]

def block_letter(ch, cap_px, xscale, seed, border=None):
    """returns RGBA float image (premult not used) + alpha bool for one block letter.
    cap_px = letter fill height in px. xscale = horizontal scale of the glyph."""
    border = border or {}
    U = cap_px / 100.0
    m = glyph_norm(ch, cap_px)
    h, w = m.shape
    m = cv2.resize(m.astype(np.uint8) * 255, (max(2, int(w * xscale)), h), interpolation=cv2.INTER_AREA) > 127
    # weight-up so strokes match the heavier reference letters
    bold = max(1, int(2.8 * U))
    m = cv2.dilate(m.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_RECT, (2 * bold + 1, 2 * bold + 1))) > 0
    g, bk, dl, o, e = 0.45 * U, 1.75 * U, 0.35 * U, 2.25 * U, 0.35 * U
    tot = g + bk + dl + o + e
    pad = int(tot + 6)
    m = np.pad(m, pad); h, w = m.shape
    d = mixdist(~m)
    nx, ny = normals(d, 1.2 * U)
    img = np.zeros((h, w, 3), np.float32)
    # fill: sequins + distress + sparkles
    rng = np.random.default_rng(seed)
    seq = sequins((h, w), max(4, int(2.6 * U)), seed, int(rng.integers(0, 100)), int(rng.integers(0, 100)))
    img[m] = seq[m]
    inner = cv2.erode(m.astype(np.uint8), np.ones((int(2 * U) + 1,) * 2, np.uint8)) > 0
    ds = distress((h, w), 10 * U, seed + 1, 0.9) & inner
    img[ds] = 250
    # inner bevel shading on fill edges (slight)
    din = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    edge = np.clip(1 - din / (1.2 * U), 0, 1)
    img[m] *= (1 - 0.25 * edge[m])[..., None]
    # bands
    r1, r2, r3, r4, r5 = g, g + bk, g + bk + dl, g + bk + dl + o, tot
    gray = (d > 0) & (d <= r1)
    blk = (d > r1) & (d <= r2)
    dkl = (d > r2) & (d <= r3)
    org = (d > r3) & (d <= r4)
    edg = (d > r4) & (d <= r5)
    sat = satin((h, w), max(3, 0.55 * U), seed + 2)
    img[gray] = 118
    tb = (d - r1) / bk
    sh = band_shade(tb, nx, ny, 0.9)
    v = (34 + 10 * sat) * sh
    v = v + 120 * np.exp(-((tb - 0.25) / 0.12) ** 2) * np.clip(-(nx * LV[0] + ny * LV[1]) * 0.6 + 0.5, 0, 1)   # gray satin highlight
    img[blk] = np.repeat(v[blk][:, None], 3, 1)
    img[dkl] = (border.get("dark", (60, 8, 0)))
    to = (d - r3) / o
    sho = band_shade(to, nx, ny, 0.55)
    base = np.asarray(border.get("base", ORANGE), np.float32)
    hi_c = np.asarray(border.get("hi", ORANGE_HI), np.float32)
    lo_c = np.asarray(border.get("lo", ORANGE_LO), np.float32)
    s3 = (sho - 1)[..., None]
    col = base[None, None, :] + np.where(s3 > 0, s3 * (hi_c - base) * 1.6, -s3 * (lo_c - base) * 1.6)
    col = col * (1 + 0.10 * sat[..., None])
    img[org] = col[org]
    img[edg] = border.get("edge", (110, 10, 0))
    # running stitch along the middle of the orange band
    mid = d <= (r3 + 0.5 * o)
    contour_dots(mid, max(3, 1.0 * U), max(1, 0.2 * U), img, border.get("stitch", (255, 178, 130)), alpha=0.8, inside=org.astype(np.float32))
    alpha = d <= r5
    img = np.clip(img, 0, 255)
    # sparkles on the fill
    ys, xs = np.where(inner)
    for _ in range(int(4 + rng.integers(0, 3))):
        i = int(rng.integers(0, len(ys))); sparkle(img, int(xs[i]), int(ys[i]), float(rng.uniform(5.0, 9.0) * U))
    img[~alpha] = 0
    return img, alpha

def script_layer(txt, height_px, target_w, border=None, widen_max=2.4):
    """'Raiderettes' script, total height ~ height_px, widened to target_w (capped)."""
    border = border or {}
    f0 = ImageFont.truetype(SCRIPT, 400); m0 = crop_mask(text_mask(txt, f0, 80), 0)
    h0, w0 = m0.shape
    U = height_px / 100.0
    k_, b_, e_ = 0.9 * U, 3.3 * U, 0.45 * U
    thick = 1.55 * U
    extra = 2 * (thick + k_ + b_ + e_)
    fs = int(400 * (height_px - extra) / h0)
    f = ImageFont.truetype(SCRIPT, fs)
    m = crop_mask(text_mask(txt, f, int(extra) + 10), int(extra) + 10)
    h, w = m.shape
    kx = 1.0
    natural = w
    if target_w and natural < target_w: kx = min(widen_max, target_w / natural)
    if kx != 1.0:
        m = cv2.resize(m.astype(np.uint8) * 255, (int(w * kx), h), interpolation=cv2.INTER_LINEAR) > 127
    d0 = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5)
    core = d0 <= thick
    d = cv2.distanceTransform((~core).astype(np.uint8), cv2.DIST_L2, 5)
    h, w = core.shape
    nx, ny = normals(d, 0.8 * U)
    img = np.zeros((h, w, 3), np.float32)
    rng = np.random.default_rng(99)
    # red glitter core
    gl = rng.random((h, w)).astype(np.float32)
    glc = cv2.GaussianBlur(gl, (0, 0), 0.35 * U)
    glc = (glc - glc.mean()) / (glc.std() + 1e-6)
    core_c = np.asarray(border.get("core", RED_CORE), np.float32)
    cc = core_c[None, None, :] * (1 + 0.22 * glc[..., None])
    spark = gl > 0.985
    cc[spark] = (255, 150, 130)
    din = cv2.distanceTransform(core.astype(np.uint8), cv2.DIST_L2, 5)
    # inner white highlight line on the side facing the light
    ndx, ndy = normals(-din + 0, 0.6 * U)
    facing = -(ndx * LV[0] + ndy * LV[1])
    hl = (np.abs(din - 0.55 * thick) < 0.16 * thick + 0.6) & (facing < -0.25)
    cc[hl] = cc[hl] * 0.2 + 255 * 0.8
    img[core] = cc[core]
    kl = (d > 0) & (d <= k_)
    img[kl] = (12, 4, 4)
    ob = (d > k_) & (d <= k_ + b_)
    to = (d - k_) / b_
    sho = band_shade(to, nx, ny, 0.6)
    base = np.asarray(border.get("base", ORANGE), np.float32); hi_c = np.asarray(border.get("hi", ORANGE_HI), np.float32); lo_c = np.asarray(border.get("lo", ORANGE_LO), np.float32)
    s3 = (sho - 1)[..., None]
    col = base[None, None, :] + np.where(s3 > 0, s3 * (hi_c - base) * 1.6, -s3 * (lo_c - base) * 1.6)
    img[ob] = col[ob]
    contour_dots(d <= (k_ + 0.5 * b_), max(3, 1.15 * U), max(1, 0.32 * U), img, border.get("dots", (255, 150, 90)), alpha=0.9, inside=ob.astype(np.float32))
    ed = (d > k_ + b_) & (d <= k_ + b_ + e_)
    img[ed] = border.get("edge", (110, 10, 0))
    alpha = d <= k_ + b_ + e_
    img[~alpha] = 0
    return np.clip(img, 0, 255), alpha, kx, U

def compose(name, W, H, mode, letter_h_frac, script_h_frac, script_w_frac, script_y_frac, kmin, kmax, margin_frac,
            gap_frac=0.02, rise_frac=0.0, border=None, seed=7, script_text="Raiderettes", widen_max=2.4, script_px=None, script_fit=None):
    """flat layout on a W x H canvas; mode 'straight' or 'curved' (curve applied after)."""
    name = name.upper()
    canvas = np.zeros((H, W, 3), np.float32); A = np.zeros((H, W), bool)
    cap = int(H * letter_h_frac)
    # measure natural letter widths (xscale 1) cheaply via font
    nat = [glyph_w(c, cap) for c in name]
    U = cap / 100
    bord = (0.45 + 1.75 + 0.35 + 2.25 + 0.35) * U + 2.8 * U
    span = W * (1 - 2 * margin_frac)
    gap = gap_frac * cap
    k = (span - gap * (len(name) - 1) - 2 * bord * len(name)) / sum(nat)
    k = max(kmin, min(kmax, k))
    lw = [n * k + 2 * bord for n in nat]
    total = sum(lw) + gap * (len(name) - 1)
    if total > span:      # shrink everything
        s = span / total * 0.99; cap = int(cap * s); return compose(name, W, H, mode, letter_h_frac * s, script_h_frac, script_w_frac, script_y_frac,
                                                          kmin, kmax, margin_frac, gap_frac, rise_frac, border, seed, script_text, widen_max, script_px, script_fit)
    gap = (span - sum(lw)) / max(1, len(name) - 1) if mode == "straight" else gap
    total = sum(lw) + gap * (len(name) - 1)
    x = (W - total) / 2
    y_top = (H - cap) / 2
    for i, c in enumerate(name):
        img, al = block_letter(c, cap, k, seed * 31 + i, border)
        hh, ww = al.shape
        cx = x + lw[i] / 2; X = int(round(cx - ww / 2)); Y = int(round(y_top + cap / 2 - hh / 2))
        x0, y0 = max(0, X), max(0, Y); x1, y1 = min(W, X + ww), min(H, Y + hh)
        sub = al[y0 - Y:y1 - Y, x0 - X:x1 - X]
        canvas[y0:y1, x0:x1][sub] = img[y0 - Y:y1 - Y, x0 - X:x1 - X][sub]
        A[y0:y1, x0:x1] |= sub
        x += lw[i] + gap
    # script
    sh = script_px or int(cap * script_h_frac)
    if script_fit:   # (width fraction of the name, max height fraction of the letters): uniform script for every name
        wf, hf = script_fit
        _, sal0, _, _ = script_layer(script_text, 400, 0, border, 1.0)
        r = sal0.shape[1] / sal0.shape[0]
        sh = int(min(hf * cap, wf * total / r))
        simg, sal, kx, SU = script_layer(script_text, sh, 0, border, 1.0)
    else:
        simg, sal, kx, SU = script_layer(script_text, sh, int(total * script_w_frac), border, widen_max)
    hh, ww = sal.shape
    lim = min(W * (1 - 2 * margin_frac), total * 0.97)
    if ww > lim:
        s = lim / ww
        simg = cv2.resize(simg, (int(ww * s), int(hh * s)), interpolation=cv2.INTER_AREA)
        sal = cv2.resize(sal.astype(np.uint8) * 255, (int(ww * s), int(hh * s)), interpolation=cv2.INTER_AREA) > 127
        hh, ww = sal.shape
    X = int(W / 2 - ww / 2); Y = int(y_top + cap * script_y_frac - hh / 2)
    Y = max(0, min(H - hh, Y))
    # shadow of the script onto the letters
    shm = np.zeros((H, W), np.float32)
    off = int(1.6 * SU)
    ys0, xs0 = Y + off, X + int(off * 0.6)
    sy1, sx1 = min(H, ys0 + hh), min(W, xs0 + ww)
    shm[ys0:sy1, xs0:sx1] = sal[:sy1 - ys0, :sx1 - xs0]
    shm = cv2.GaussianBlur(shm, (0, 0), 1.6 * SU) * 0.75
    canvas *= (1 - shm * A)[..., None]
    sub = sal
    canvas[Y:Y + hh, X:X + ww][sub] = simg[sub]
    A[Y:Y + hh, X:X + ww] |= sub
    return canvas, A

def arch(img, A, rise):
    """vertical-shift arch: centre up, ends down by `rise` px (circular)"""
    H, W = A.shape
    half = W / 2
    R = (half ** 2 + rise ** 2) / (2 * rise)
    xs = np.arange(W, dtype=np.float32) - half
    dy = (R - np.sqrt(np.maximum(0, R * R - xs * xs))).astype(np.float32)   # 0 at centre, rise at ends
    outH = H + int(rise) + 2
    map_x = np.tile(np.arange(W, dtype=np.float32), (outH, 1))
    map_y = np.arange(outH, dtype=np.float32)[:, None] - dy[None, :]
    o = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    a = cv2.remap(A.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= 0.5
    return o, a

def to_rgba(img, A):
    out = np.zeros(A.shape + (4,), np.uint8)
    out[..., :3] = np.clip(img, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(A, 255, 0)
    out[~A, :3] = 0
    return Image.fromarray(out, "RGBA")

def trim(img, A, pad=0):
    ys, xs = np.where(A)
    y0, y1, x0, x1 = max(0, ys.min() - pad), ys.max() + 1 + pad, max(0, xs.min() - pad), xs.max() + 1 + pad
    return img[y0:y1, x0:x1], A[y0:y1, x0:x1]
