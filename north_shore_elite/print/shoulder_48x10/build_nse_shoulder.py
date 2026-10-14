"""North Shore Elites sleeve-to-sleeve shoulder print (from Brian's black-crewneck photo, 2026-10-14).
48 x 10 in at 300 DPI (14400 x 3000), transparent, hard alpha. Resize down for S/M/L/XL per the Gildan 18000 chart.
NORTH SHORE: distressed royal-blue varsity letters (Graduate, condensed) with a thin white outline, arched.
Elites: white brush script, black outline, thin royal edge, with a royal swoosh underline.
usage: python3 build_nse_shoulder.py OUT.png [scale]"""
import sys, math, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as nd
GRAD = "/home/claude/ocbm-repository/raiderettes/shoulder_40x6/fonts/Graduate-Regular.ttf"
SCRIPT = "/home/claude/nse_edit/fonts/KaushanScript-Regular.ttf"
ROYAL = np.array([22, 74, 222], np.float32)
WHITE = np.array([255, 255, 255], np.float32)
BLACK = np.array([8, 8, 12], np.float32)
NAVY = np.array([16, 34, 110], np.float32)

def text_mask(txt, font, pad):
    l, t, r, b = font.getbbox(txt)
    m = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(m).text((pad - l, pad - t), txt, font=font, fill=255)
    a = np.asarray(m) > 127
    ys, xs = np.where(a); return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

def grow(m, r):
    if r <= 0: return m.copy()
    return nd.distance_transform_edt(~m) <= r

def distress(shape, U, seed):
    """worn-print texture: vertical scratches + specks (True = worn)"""
    rng = np.random.default_rng(seed); h, w = shape
    m = np.zeros(shape, np.uint8)
    for _ in range(int(w * h / (60 * U) ** 2 * 14)):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        L = rng.gamma(2.0, 5 * U); th = max(1, int(rng.gamma(1.6, 0.45 * U)))
        a = math.radians(rng.normal(90, 12))
        cv2.line(m, (x, y), (int(x + L * math.cos(a)), int(y + L * math.sin(a))), 255, th, cv2.LINE_AA)
    n = cv2.GaussianBlur(rng.random(shape).astype(np.float32), (0, 0), 0.7 * U)
    n = (n - n.mean()) / (n.std() + 1e-6)
    m = (m > 100) | (n > 2.0)
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    return m

def letter(ch, cap, k, seed):
    U = cap / 100
    f = ImageFont.truetype(GRAD, int(cap / 0.72))
    g = text_mask(ch, f, 40)
    g = cv2.resize(g.astype(np.uint8) * 255, (max(2, int(g.shape[1] * k)), int(g.shape[0] * cap / g.shape[0])), interpolation=cv2.INTER_AREA) > 127
    g = nd.binary_dilation(g, structure=np.ones((3, 3)), iterations=max(1, int(1.2 * U)))   # weight-up
    out_w = 3.0 * U
    pad = int(out_w + 6); g = np.pad(g, pad)
    outer = grow(g, out_w)
    rgb = np.zeros(g.shape + (3,), np.float32)
    rgb[outer] = WHITE
    # royal fill with slight tonal variation + worn white scratches
    rng = np.random.default_rng(seed)
    tone = cv2.GaussianBlur(rng.standard_normal(g.shape).astype(np.float32), (0, 0), 6 * U)
    tone = tone / (np.abs(tone).max() + 1e-6)
    fill = ROYAL[None, None, :] * (1 + 0.08 * tone[..., None])
    rgb[g] = fill[g]
    worn = distress(g.shape, U, seed) & g
    rgb[worn] = rgb[worn] * 0.25 + np.array([225, 232, 248], np.float32) * 0.75
    alpha = outer.copy()
    # a little wear on the white outline too
    return np.clip(rgb, 0, 255), alpha

def compose(W0, H):
    W = int(W0 * 1.15)
    canvas = np.zeros((H, W, 3), np.float32); A = np.zeros((H, W), bool)
    txt = "NORTH SHORE"
    margin = int(0.075 * W0); rise = int(0.24 * H)
    cap = int((H - rise) * 0.86)
    U = cap / 100
    f = ImageFont.truetype(GRAD, 1000)
    nat = [text_mask(c, f, 10).shape[1] * cap / text_mask("H", f, 10).shape[0] if c != " " else 0 for c in txt]
    gap = 0.035 * cap; space = 0.20 * cap
    n_l = sum(1 for c in txt if c != " ")
    span = W - 2 * margin
    bord = 2 * (3.0 * U + 1.2 * U)
    k = (span - gap * (n_l - 2) - space - bord * n_l) / sum(nat)
    k = min(k, 1.0)
    ws = [(n * k + bord) if c != " " else space for n, c in zip(nat, txt)]
    total = sum(ws) + gap * (len(txt) - 1 - 2)
    x = (W - total) / 2; cx = W / 2
    R = (span / 2) ** 2 / (2 * rise) + rise / 2
    y0 = int(0.035 * H)
    for i, c in enumerate(txt):
        if c == " ": x += ws[i]; continue
        img, al = letter(c, cap, k, 31 + i)
        mx = x + ws[i] / 2; dx = mx - cx
        dy = R - math.sqrt(max(0.0, R * R - dx * dx))
        ang = -math.degrees(math.asin(dx / R))
        rgba = np.dstack([img, al.astype(np.float32) * 255]).astype(np.uint8)
        im = Image.fromarray(rgba, "RGBA").rotate(ang, resample=Image.BICUBIC, expand=True)
        a2 = np.array(im); hh, ww = a2.shape[:2]
        X = int(mx - ww / 2); Y = int(y0 + cap / 2 + dy - hh / 2 + 3 * U)
        x0_, y0_ = max(0, X), max(0, Y); x1_, y1_ = min(W, X + ww), min(H, Y + hh)
        sub = a2[y0_ - Y:y1_ - Y, x0_ - X:x1_ - X]; m = sub[..., 3] >= 128
        canvas[y0_:y1_, x0_:x1_][m] = sub[..., :3][m]; A[y0_:y1_, x0_:x1_] |= m
        x += ws[i] + gap
    # fit the arched letters exactly into the final width (no clipping at the sleeves)
    xs_ = np.where(A.any(0))[0]; m0 = int(0.006 * W0)
    c0, c1 = xs_.min(), xs_.max() + 1
    tw = W0 - 2 * m0
    canvas = cv2.resize(canvas[:, c0:c1], (tw, H), interpolation=cv2.INTER_AREA)
    A = cv2.resize(A[:, c0:c1].astype(np.float32), (tw, H), interpolation=cv2.INTER_AREA) >= 0.5
    canvas = np.pad(canvas, ((0, 0), (m0, W0 - tw - m0), (0, 0))); A = np.pad(A, ((0, 0), (m0, W0 - tw - m0)))
    W = W0
    # --- script "Elites" ---
    sh = int(H * 0.80)
    sf = ImageFont.truetype(SCRIPT, 1000); sm = text_mask("Elites", sf, 60)
    s = sh / sm.shape[0]
    sm = cv2.resize(sm.astype(np.uint8) * 255, (int(sm.shape[1] * s * 1.32), sh), interpolation=cv2.INTER_AREA) > 127
    SU = sh / 100
    shx = 0.18; hh_, ww_ = sm.shape; Msh = np.float32([[1, -shx, shx * hh_], [0, 1, 0]])
    sm = cv2.warpAffine(sm.astype(np.uint8) * 255, Msh, (int(ww_ + shx * hh_), hh_)) > 127
    core = grow(sm, 0.6 * SU)                      # bold up the brush strokes
    pad = int(6 * SU); core = np.pad(core, pad)
    blk = grow(core, 1.3 * SU); edge = grow(blk, 0.7 * SU)
    h2, w2 = core.shape
    # royal swoosh under the word
    sw = np.zeros((h2, w2), np.uint8)
    pts = []
    for t in np.linspace(0, 1, 3000):
        xx = w2 * (0.06 + 0.92 * t); yy = h2 * (0.965 - 0.20 * t - 0.02 * math.sin(math.pi * t))
        pts.append((xx, yy, (0.3 + 3.2 * math.sin(math.pi * t) ** 0.7) * SU))
    for xx, yy, r in pts: cv2.circle(sw, (int(xx), int(yy)), max(1, int(r)), 255, -1, cv2.LINE_AA)
    sw = (sw > 127) & ~blk
    swo = grow(sw, 0.6 * SU) & ~blk
    srgb = np.zeros((h2, w2, 3), np.float32)
    srgb[swo] = NAVY; srgb[sw] = ROYAL
    srgb[edge] = NAVY; srgb[blk] = BLACK; srgb[core] = WHITE
    # soft grey shading on the lower-right of the white strokes (brush depth)
    din = nd.distance_transform_edt(core)
    shade = np.roll(np.roll(core, -int(0.6 * SU), 0), -int(0.5 * SU), 1)
    rim = core & ~shade
    srgb[rim] = srgb[rim] * 0.55 + np.array([150, 165, 200], np.float32) * 0.45
    sal = edge | swo
    X = int(W * 0.50 - w2 / 2); Y = int(H * 0.995 - h2)
    Y = min(Y, H - h2)
    sub = sal
    canvas[Y:Y + h2, X:X + w2][sub] = srgb[sub]; A[Y:Y + h2, X:X + w2] |= sub
    return canvas, A

if __name__ == "__main__":
    out = sys.argv[1]; sc = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    W, H = int(14400 * sc), int(3000 * sc)
    img, A = compose(W, H)
    rgba = np.zeros((H, W, 4), np.uint8); rgba[..., :3] = np.clip(img, 0, 255); rgba[..., 3] = np.where(A, 255, 0); rgba[~A, :3] = 0
    Image.fromarray(rgba, "RGBA").save(out, dpi=(300 * sc, 300 * sc)); print(out, W, H)
