"""SHEA / Raiderettes sleeve-to-sleeve back print: 40 x 6 in at 300 DPI (12000 x 1800 px), hard alpha.
usage: python3 build.py NAME [arch|straight] [out.png]"""
import sys, math, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

DPI = 300; W, H = 40 * DPI, 6 * DPI
BLOCK = "/home/claude/ocbm-repository/raiderettes/shoulder_40x6/fonts/Graduate-Regular.ttf"
SCRIPT = "/home/claude/ocbm-repository/raiderettes/shoulder_40x6/fonts/MsMadi-Regular.ttf"
BLACK = (12, 12, 12); WHITE = (255, 255, 255); RED = (214, 22, 30)

def text_mask(txt, font, pad=200):
    l, t, r, b = font.getbbox(txt)
    m = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0)
    ImageDraw.Draw(m).text((pad - l, pad - t), txt, font=font, fill=255)
    return np.asarray(m) > 127

def grow(mask, px):
    if px <= 0: return mask.copy()
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0

def distress(shape, seed, density=1.0):
    """white specks / scratches like the source artwork"""
    rng = np.random.default_rng(seed); h, w = shape
    m = np.zeros(shape, np.uint8)
    n = int(w * h / 9000 * density)
    for _ in range(n):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        L = float(rng.gamma(1.6, 9)); T = max(1.5, float(rng.gamma(1.5, 2.2)))
        ang = float(rng.normal(-20, 35))
        cv2.ellipse(m, (x, y), (int(L), int(T)), ang, 0, 360, 255, -1)
    # roughen
    noise = rng.random(shape) < 0.002
    m[noise] = 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return m > 0

def letter_rgba(ch, font, wht, red, seed):
    fill = text_mask(ch, font)
    white = grow(fill, wht); outer = grow(fill, wht + red)
    h, w = fill.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[outer] = (*RED, 255)
    rgba[white] = (*WHITE, 255)
    rgba[fill] = (*BLACK, 255)
    specks = distress(fill.shape, seed) & cv2.erode(fill.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    rgba[specks] = (*WHITE, 255)
    ys, xs = np.where(outer)
    return Image.fromarray(rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1])

def script_rgba(txt, font, thick, red, key=0):
    fill = grow(text_mask(txt, font, pad=300), thick)       # thicken the monoline script
    outer = grow(fill, red); kl = grow(outer, key)
    h, w = fill.shape
    rgba = np.zeros((h, w, 4), np.uint8)
    if key: rgba[kl] = (*BLACK, 255)
    rgba[outer] = (*RED, 255); rgba[fill] = (*BLACK, 255)
    ys, xs = np.where(kl)
    return Image.fromarray(rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1])

def full_script(txt, target_w, target_h, thick, kmax=1.6):
    """script sized to target height, then widened (max 1.6x) to reach target width"""
    red, key = int(0.08 * DPI), int(0.035 * DPI)
    extra = 2 * (thick + red + key)
    f0 = ImageFont.truetype(SCRIPT, 400); m0 = text_mask(txt, f0, pad=50)
    ys, xs = np.where(m0); h0, w0 = np.ptp(ys) + 1, np.ptp(xs) + 1
    fs = int(400 * (target_h - extra) / h0)
    s = script_rgba(txt, ImageFont.truetype(SCRIPT, fs), thick, red, key)
    k = min(kmax, target_w / s.width)
    if k < 1: s = s.resize((int(s.width * k), int(s.height * k)), Image.LANCZOS)
    else: s = s.resize((int(s.width * k), s.height), Image.LANCZOS)
    print("script natural ratio %.2f, widened x%.2f" % (w0 / h0, max(k, 1)))
    return s

def build(block_text, script_text, arch=True, script_frac=0.5, script_h=2.4, script_drop=0.35, script_wt=0.05, script_full=False):
    """block_text in varsity letters across the full 40 in; script_text overlapping the middle."""
    block_text = block_text.upper()
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    margin = int(1.0 * DPI)
    rise = int(0.8 * DPI * H / 1800) if arch else 0      # arch scales with print height (0.8 in at 6 in)
    pad = int(0.22 * DPI)
    wht, red = int(0.12 * DPI), int(0.10 * DPI)
    lh_max = H - 2 * pad - rise
    cap = int((lh_max - 2 * (wht + red)) * 0.97)
    font = ImageFont.truetype(BLOCK, int(cap / 0.72))
    letters = [letter_rgba(c, font, wht, red, 7 + i) for i, c in enumerate(block_text)]
    # fit horizontally: condense (>=0.8) or extend (<=1.6), then distribute any leftover as tracking
    span = W - 2 * margin; n = len(letters)
    g0 = int(0.05 * cap)
    natural = sum(l.width for l in letters) + g0 * (n - 1)
    k = max(0.8, min(1.6, (span - g0 * (n - 1)) / sum(l.width for l in letters)))
    letters = [l.resize((max(1, int(l.width * k)), l.height), Image.LANCZOS) for l in letters]
    if sum(l.width for l in letters) + g0 * (n - 1) > span:          # still too long: shrink uniformly
        f = (span - g0 * (n - 1)) / sum(l.width for l in letters)
        letters = [l.resize((int(l.width * f), int(l.height * f)), Image.LANCZOS) for l in letters]
    ws = [l.width for l in letters]
    gap = (span - sum(ws)) / max(1, n - 1)
    xs = []; x = margin
    for w_ in ws: xs.append(x + w_ / 2); x += w_ + gap
    cx = W / 2; lh = max(l.height for l in letters)
    R = (span / 2) ** 2 / (2 * rise) + rise / 2 if arch else None
    for l, x in zip(letters, xs):
        if arch:
            dx = x - cx; dy = R - math.sqrt(max(0.0, R * R - dx * dx))
            ang = -math.degrees(math.asin(dx / R))
            l2 = l.rotate(ang, resample=Image.BICUBIC, expand=True)
            y = pad + lh / 2 + dy
        else:
            l2 = l; y = H / 2
        canvas.alpha_composite(l2, (int(x - l2.width / 2), int(y - l2.height / 2)))
    if not script_text.strip():
        a = np.asarray(canvas).copy(); a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        return Image.fromarray(a)
    if script_full:
        s = full_script(script_text, int(span * 0.985), H - 2 * pad - rise, int(script_wt * DPI))
        flat_h = s.height
        if arch:   # bend the script along the same arch as the letters
            sh, sw = s.height, s.width
            xs_ = np.arange(sw, dtype=np.float32) + (cx - sw / 2) - cx
            dys = (R - np.sqrt(np.maximum(0, R * R - xs_ ** 2))).astype(np.float32)
            outh = sh + rise
            map_x = np.tile(np.arange(sw, dtype=np.float32), (outh, 1))
            map_y = np.arange(outh, dtype=np.float32)[:, None] - dys[None, :]
            arr = cv2.remap(np.asarray(s), map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
            s = Image.fromarray(arr)
        yoff = int(max(0, (H - 2 * pad - rise) - flat_h) * 0.62)   # sit a little below the middle of the letters
        canvas.alpha_composite(s, (int(cx - s.width / 2), int(pad + yoff)))
        a = np.asarray(canvas).copy(); a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
        return Image.fromarray(a)
    sfont = ImageFont.truetype(SCRIPT, int(script_h * DPI / 0.62))
    s = script_rgba(script_text, sfont, int(script_wt * DPI), int(0.075 * DPI), int(0.035 * DPI))
    tw = int(span * script_frac)
    if s.width > tw: s = s.resize((tw, int(s.height * tw / s.width)), Image.LANCZOS)
    if s.height > H - 2 * pad: s = s.resize((int(s.width * (H - 2 * pad) / s.height), H - 2 * pad), Image.LANCZOS)
    sy = pad + lh / 2 + int(script_drop * DPI)
    sy = min(sy, H - pad - s.height / 2)
    canvas.alpha_composite(s, (int(cx - s.width / 2), int(sy - s.height / 2)))
    a = np.asarray(canvas).copy(); a[..., 3] = np.where(a[..., 3] >= 128, 255, 0)
    return Image.fromarray(a)

if __name__ == "__main__":
    # usage: build.py BLOCK_TEXT SCRIPT_TEXT [arch|straight] out.png
    bt, st = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "arch"
    out = sys.argv[4] if len(sys.argv) > 4 else f"{bt.upper()}_{st}_40x6_{mode}.png"
    kw = {}
    if len(bt) <= 6: kw = dict(script_frac=0.6, script_h=2.6, script_drop=0.5, script_wt=0.045)
    else: kw = dict(script_frac=0.38, script_h=4.6, script_drop=0.55, script_wt=0.06)
    if len(sys.argv) > 5 and sys.argv[5] == "full":
        kw = dict(script_full=True, script_wt=0.06)
        _fs = full_script
        globals()["full_script"] = lambda txt, tw, th, thick: _fs(txt, tw, int(3.4 * DPI), thick, kmax=2.3)
    im = build(bt, st, mode == "arch", **kw); im.save(out, dpi=(DPI, DPI)); print(out, im.size)
