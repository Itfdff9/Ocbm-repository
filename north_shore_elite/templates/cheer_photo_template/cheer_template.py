"""OCBM "CHEER" senior-style photo template (North Shore Elite) — original artwork, editable name + photos.
usage: python3 cheer_template.py --name Emily --photo main.png [--ghost1 face.jpg] [--ghost2 action.jpg]
                                 [--bottom "North Shore Elite"] [--year 2026] [--out OUT.png] [--scale 1.0] [--layers DIR]
--photo   : the main cheerleader picture (any photo; background is removed automatically unless it is already a transparent PNG)
--ghost1  : faded picture upper-middle (defaults to the main photo, face crop)
--ghost2  : faded picture on the left (defaults to the main photo, mirrored)
Output: 12 x 14.67 in at 300 DPI (3600 x 4400), transparent outside a distressed edge, hard alpha."""
import argparse, math, os, numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
F_SCRIPT = os.path.join(HERE, "fonts/Hurricane-Regular.ttf")
F_BLOCK = os.path.join(HERE, "fonts/Anton-Regular.ttf")
BLUE = np.array([30, 90, 235], np.float32)
DEEP = np.array([6, 20, 70], np.float32)
ICE = np.array([190, 220, 255], np.float32)

def noise(h, w, scale, seed, octaves=5):
    rng = np.random.default_rng(seed); out = np.zeros((h, w), np.float32); amp = 1.0; tot = 0
    for o in range(octaves):
        s = max(2, int(scale / (2 ** o)))
        small = rng.standard_normal((h // s + 2, w // s + 2)).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC); tot += amp; amp *= 0.55
    out /= tot; return (out - out.min()) / (out.max() - out.min() + 1e-6)

def smoke(h, w, seed, focus=None, power=2.2):
    n = noise(h, w, w * 0.18, seed)
    n2 = noise(h, w, w * 0.05, seed + 1, 4)
    s = np.clip((n * 0.75 + n2 * 0.25 - 0.38) * 2.4, 0, 1) ** power
    if focus is not None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        f = np.zeros((h, w), np.float32)
        for (cx, cy, r, k) in focus: f = np.maximum(f, k * np.exp(-(((xx - cx * w) ** 2 + (yy - cy * h) ** 2) / (r * w) ** 2)))
        s *= f
    return s

def lightning(img, x0, y0, x1, y1, width, seed, depth=7, glow=1.0):
    rng = np.random.default_rng(seed)
    def subdiv(p, q, d, disp):
        if d == 0: return [p, q]
        m = ((p[0] + q[0]) / 2 + rng.normal(0, disp), (p[1] + q[1]) / 2 + rng.normal(0, disp))
        return subdiv(p, m, d - 1, disp / 2)[:-1] + subdiv(m, q, d - 1, disp / 2)
    L = math.hypot(x1 - x0, y1 - y0)
    pts = subdiv((x0, y0), (x1, y1), depth, L * 0.12)
    lay = np.zeros(img.shape[:2], np.float32)
    def draw(pts, w):
        cv2.polylines(lay, [np.array(pts, np.int32)], False, 1.0, max(1, int(w)), cv2.LINE_AA)
    draw(pts, width)
    for _ in range(4):   # branches
        i = int(rng.integers(len(pts) // 4, len(pts) - 2)); p = pts[i]
        ang = math.atan2(y1 - y0, x1 - x0) + rng.normal(0, 0.7); l = L * rng.uniform(0.12, 0.3)
        draw(subdiv(p, (p[0] + l * math.cos(ang), p[1] + l * math.sin(ang)), depth - 2, l * 0.15), width * 0.55)
    g = cv2.GaussianBlur(lay, (0, 0), width * 4) * 2.2 * glow + cv2.GaussianBlur(lay, (0, 0), width * 1.2) * 1.2
    img[:] = img + g[..., None] * BLUE * 0.9
    img[:] = img * (1 - lay[..., None]) + lay[..., None] * ICE * 1.25

def text_mask(txt, font, size, pad=60):
    f = ImageFont.truetype(font, size); l, t, r, b = f.getbbox(txt)
    m = Image.new("L", (r - l + 2 * pad, b - t + 2 * pad), 0); ImageDraw.Draw(m).text((pad - l, pad - t), txt, font=f, fill=255)
    a = np.asarray(m) > 127; ys, xs = np.where(a)
    return a[max(0, ys.min() - pad // 2):ys.max() + pad // 2, max(0, xs.min() - pad // 2):xs.max() + pad // 2]

def grow(m, r): return nd.distance_transform_edt(~m) <= r if r > 0 else m

def paste(dst, dsta, src, srca, x, y, mode="over", op=1.0):
    h, w = srca.shape; H, W = dsta.shape
    x0, y0 = max(0, x), max(0, y); x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0: return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]; a = srca[y0 - y:y1 - y, x0 - x:x1 - x][..., None] * op
    d = dst[y0:y1, x0:x1]
    if mode == "screen": d[:] = d * (1 - a) + (255 - (255 - d) * (255 - s) / 255) * a
    else: d[:] = d * (1 - a) + s * a
    dsta[y0:y1, x0:x1] = np.maximum(dsta[y0:y1, x0:x1], srca[y0 - y:y1 - y, x0 - x:x1 - x] * op)

def cutout(path, cache=True):
    im = Image.open(path)
    if im.mode == "RGBA" and np.array(im)[..., 3].min() < 10: return im
    cp = os.path.splitext(path)[0] + "_cutout.png"
    if cache and os.path.exists(cp): return Image.open(cp)
    from rembg import remove, new_session
    out = remove(im.convert("RGB"), session=new_session("isnet-general-use"))
    out.save(cp); return out

def duotone(rgb, lo=DEEP, hi=ICE):
    g = (rgb.astype(np.float32) @ np.array([0.3, 0.59, 0.11], np.float32)) / 255
    g = np.clip((g - 0.08) / 0.85, 0, 1) ** 0.9
    mid = BLUE
    out = np.where(g[..., None] < 0.55, lo + (mid - lo) * (g[..., None] / 0.55), mid + (hi - mid) * ((g[..., None] - 0.55) / 0.45))
    return out

def script_layer(txt, height, thick, slant=0.22, shadow=True, swash=None):
    m = text_mask(txt, F_SCRIPT, int(height * 1.6), pad=int(height * 0.5))
    k = height / m.shape[0] * 1.25; m = cv2.resize(m.astype(np.uint8) * 255, (int(m.shape[1] * k), int(m.shape[0] * k))) > 127
    m = grow(m, thick)
    h, w = m.shape; pad = int(h * 0.25); m = np.pad(m, pad); h, w = m.shape
    if swash:   # long brush underline: from under the first letters out to the right
        sw = np.zeros((h, w), np.uint8); x0, x1 = int(w * swash[0]), int(w * swash[1]); yb = int(h * swash[2])
        for i in range(400):
            t = i / 399; x = x0 + (x1 - x0) * t; y = yb - (h * 0.10) * t
            r = thick * (0.5 + 1.1 * math.sin(math.pi * t) ** 0.6); cv2.circle(sw, (int(x), int(y)), max(1, int(r)), 255, -1, cv2.LINE_AA)
        m |= sw > 127
    M = np.float32([[1, -slant, slant * h], [0, 1, 0]])
    m = cv2.warpAffine(m.astype(np.uint8) * 255, M, (int(w + slant * h), h)) > 127
    h, w = m.shape
    rgb = np.zeros((h, w, 3), np.float32); a = np.zeros((h, w), np.float32)
    if shadow:
        off = max(2, int(thick * 1.6)); sh = np.roll(np.roll(m, off, 0), off, 1)
        glow = cv2.GaussianBlur(grow(m, thick * 2).astype(np.float32), (0, 0), thick * 3)
        rgb += glow[..., None] * BLUE * 0.0; a = np.maximum(a, glow * 0.0)
        rgb[sh] = BLUE * 0.85; a[sh] = 1
    din = nd.distance_transform_edt(m)
    edge = (din < max(1.5, thick * 0.45)) & m
    rgb[m] = 252; rgb[edge] = ICE; a[m] = 1
    return rgb, a

def cheer_layer(Wb, Hb, tex, seed=3):
    """vertical CHEER: Anton letters rotated 90deg CW, smoke-filled, white outline + inline"""
    m = text_mask("CHEER", F_BLOCK, 1200, pad=80)
    m = cv2.resize(m.astype(np.uint8) * 255, (Hb, Wb), interpolation=cv2.INTER_AREA) > 127   # text runs along the long side
    m = np.rot90(m, -1)                                                                      # rotate CW: reads top to bottom
    U = Wb / 100
    out_w = 2.6 * U; gap = 1.5 * U; inl = 0.9 * U
    m = np.pad(m, int(out_w + 8)); h, w = m.shape
    inner = nd.binary_erosion(m, iterations=int(gap + inl)) if False else (nd.distance_transform_edt(m) > (gap + inl))
    inline = (nd.distance_transform_edt(m) > gap) & ~inner
    outer = grow(m, out_w) & ~m
    t = cv2.resize(tex, (w, h))
    rgb = np.zeros((h, w, 3), np.float32); a = np.zeros((h, w), np.float32)
    rgb[m] = t[m] * 1.15 + 8; a[m] = 1
    rgb[outer] = 250; a[outer] = 1
    rgb[inline] = 245
    gap_m = m & ~inline & ~inner; rgb[gap_m] = rgb[gap_m] * 0.35
    return np.clip(rgb, 0, 255), a

def heart(size, thick):
    t = np.linspace(0, 2 * math.pi, 400)
    x = 16 * np.sin(t) ** 3; y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    x = (x - x.min()) / (x.max() - x.min()) * size + thick * 2; y = (y - y.min()) / (y.max() - y.min()) * size * 0.95 + thick * 2
    lay = np.zeros((int(size + thick * 4), int(size + thick * 4)), np.uint8)
    cv2.polylines(lay, [np.stack([x, y], 1).astype(np.int32)], True, 255, max(1, int(thick)), cv2.LINE_AA)
    return lay > 127

def build(name, photo, ghost1=None, ghost2=None, bottom="North Shore Elite", year="2026", scale=1.0, seed=11, layers_dir=None):
    W, H = int(3600 * scale), int(4400 * scale)
    img = np.zeros((H, W, 3), np.float32) + np.array([4, 7, 18], np.float32)
    A = np.ones((H, W), np.float32)
    # --- background smoke + glitter ---
    s1 = smoke(H, W, seed, focus=[(0.55, 0.42, 0.55, 1.0), (0.85, 0.55, 0.4, 1.0), (0.2, 0.9, 0.45, 0.9), (0.5, 0.95, 0.5, 1.0), (0.3, 0.12, 0.35, 0.8)])
    img += s1[..., None] * (BLUE * 1.25)
    s2 = smoke(H, W, seed + 7, focus=[(0.5, 0.5, 0.6, 1.0)], power=3.5)
    img += s2[..., None] * ICE * 0.75
    rng = np.random.default_rng(seed)
    for _ in range(int(2600 * scale * scale)):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H)); r = max(1, int(rng.gamma(1.2, 1.4) * scale))
        cv2.circle(img, (x, y), r, tuple(float(v) for v in (ICE * rng.uniform(0.5, 1.0))), -1, cv2.LINE_AA)
    for (x0, y0, x1, y1, wd, sd) in [(0.66, 0.10, 0.74, 0.42, 5, 1), (0.30, 0.20, 0.10, 0.40, 4, 3), (0.02, 0.80, 0.12, 0.95, 3, 4)]:
        lightning(img, x0 * W, y0 * H, x1 * W, y1 * H, wd * scale, seed * 10 + sd)
    # stadium lights (left)
    for r_ in range(3):
        for c_ in range(6):
            cx, cy = int((0.03 + c_ * 0.024) * W), int((0.42 + r_ * 0.018) * H)
            g = np.zeros((int(80 * scale), int(80 * scale)), np.float32); cv2.circle(g, (g.shape[1] // 2, g.shape[0] // 2), max(2, int(16 * scale)), 1, -1, cv2.LINE_AA)
            g = cv2.GaussianBlur(g, (0, 0), 6 * scale) * 1.6 + g
            paste(img, np.zeros((H, W), np.float32), np.ones(g.shape + (3,), np.float32) * 255, np.clip(g, 0, 1), cx - g.shape[1] // 2, cy - g.shape[0] // 2, "screen")
    bg_only = img.copy()
    # --- ghost photos (blue duotone, screen) ---
    main = cutout(photo)
    mrgb = np.array(main.convert("RGBA")).astype(np.float32)
    def ghost(src, box, crop=None, mirror=False, op=0.62):
        if src is None:
            g = mrgb.copy()
        else:
            gi = Image.open(src).convert("RGBA"); g = np.array(gi).astype(np.float32)
        if crop:
            h0, w0 = g.shape[:2]; g = g[int(crop[1] * h0):int(crop[3] * h0), int(crop[0] * w0):int(crop[2] * w0)]
        if mirror: g = g[:, ::-1]
        bx, by, bw, bh = [int(v) for v in (box[0] * W, box[1] * H, box[2] * W, box[3] * H)]
        k = min(bw / g.shape[1], bh / g.shape[0]); g = cv2.resize(g, (max(1, int(g.shape[1] * k)), max(1, int(g.shape[0] * k))), interpolation=cv2.INTER_AREA)
        rgb = duotone(g[..., :3]); a = g[..., 3] / 255
        yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]].astype(np.float32)
        fade = np.clip(1.0 - np.hypot((xx / g.shape[1] - 0.5) * 2, (yy / g.shape[0] - 0.45) * 1.9), 0, 1) ** 0.9
        paste(img, np.zeros((H, W), np.float32), rgb, a * fade, bx + (bw - g.shape[1]) // 2, by + (bh - g.shape[0]) // 2, "screen", op)
    ghost(ghost1, (0.43, 0.10, 0.25, 0.25), crop=None if ghost1 else (0.05, 0.0, 0.95, 0.42), op=0.70)
    ghost(ghost2, (0.00, 0.20, 0.32, 0.50), mirror=ghost2 is None, op=0.55)
    # --- main photo with blue rim glow ---
    bx, by, bw, bh = int(0.03 * W), int(0.215 * H), int(0.66 * W), int(0.69 * H)
    k = min(bw / mrgb.shape[1], bh / mrgb.shape[0]); mr = cv2.resize(mrgb, (int(mrgb.shape[1] * k), int(mrgb.shape[0] * k)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC)
    ma = mr[..., 3] / 255; px, py = bx + (bw - mr.shape[1]) // 2, by + bh - mr.shape[0]
    glow = cv2.GaussianBlur(ma, (0, 0), 30 * scale)
    paste(img, np.zeros((H, W), np.float32), np.ones(mr.shape[:2] + (3,), np.float32) * BLUE * 1.4, np.clip(glow * 1.3, 0, 1), px, py, "screen", 0.9)
    paste(img, np.zeros((H, W), np.float32), mr[..., :3], ma, px, py)
    # foreground smoke at the bottom (over the photo)
    s3 = smoke(H, W, seed + 21, focus=[(0.25, 1.0, 0.35, 1.0), (0.75, 1.02, 0.35, 1.0), (0.5, 1.05, 0.3, 0.8)], power=1.6)
    img = img * (1 - 0.55 * s3[..., None]) + s3[..., None] * (BLUE * 0.75 + ICE * 0.25) * 0.55 + s3[..., None] * 25
    # --- CHEER (right) ---
    ts = smoke(H, W, seed + 33, power=1.3); ts2 = smoke(H, W, seed + 34, power=3.0)
    tex = np.clip(np.array([3, 10, 40], np.float32) + ts[..., None] * BLUE * 1.6 + ts2[..., None] * ICE * 0.9, 0, 255)
    for (x0, y0, x1, y1, wd, sd) in [(0.78, 0.15, 0.92, 0.45, 3, 41), (0.95, 0.40, 0.75, 0.70, 3, 42), (0.80, 0.62, 0.93, 0.85, 3, 43)]:
        lightning(tex, x0 * W, y0 * H, x1 * W, y1 * H, wd * scale, sd, glow=0.6)
    cw, ch = int(0.27 * W), int(0.71 * H)
    crgb, ca = cheer_layer(cw, ch, tex[int(0.13 * H):int(0.13 * H) + ch, int(0.70 * W):int(0.70 * W) + cw] if True else None)
    cxp, cyp = int(0.705 * W), int(0.125 * H)
    paste(img, np.zeros((H, W), np.float32), crgb, ca, cxp, cyp)
    layers = {"CHEER": (crgb, ca, cxp, cyp)}
    # --- name (top-left) ---
    nrgb, na = script_layer(name, int(0.16 * H), max(2, int(4.5 * scale * 3)), swash=(0.30, 1.0, 0.86))
    k = min(0.55 * W / na.shape[1], 0.215 * H / na.shape[0]); nrgb = cv2.resize(nrgb, (int(na.shape[1] * k), int(na.shape[0] * k))); na = cv2.resize(na, (nrgb.shape[1], nrgb.shape[0]))
    nim = Image.fromarray(np.dstack([np.clip(nrgb, 0, 255), na * 255]).astype(np.uint8), "RGBA").rotate(9, resample=Image.BICUBIC, expand=True)
    nn = np.array(nim).astype(np.float32); nx, ny = int(0.0 * W), int(-0.01 * H)
    paste(img, np.zeros((H, W), np.float32), nn[..., :3], nn[..., 3] / 255, nx, ny)
    layers["NAME"] = (nn[..., :3], nn[..., 3] / 255, nx, ny)
    # --- bottom line: "North Shore Elite 2026" + heart ---
    brgb, ba = script_layer(f"{bottom}  {year}", int(0.10 * H), max(1, int(2.6 * scale * 3)), slant=0.05)
    k = min(0.84 * W / ba.shape[1], 0.15 * H / ba.shape[0]); brgb = cv2.resize(brgb, (int(ba.shape[1] * k), int(ba.shape[0] * k))); ba = cv2.resize(ba, (brgb.shape[1], brgb.shape[0]))
    bim = Image.fromarray(np.dstack([np.clip(brgb, 0, 255), ba * 255]).astype(np.uint8), "RGBA").rotate(4, resample=Image.BICUBIC, expand=True)
    bb = np.array(bim).astype(np.float32); bxp, byp = int(0.02 * W), int(0.99 * H) - bb.shape[0]
    paste(img, np.zeros((H, W), np.float32), bb[..., :3], bb[..., 3] / 255, bxp, byp)
    hs = int(0.045 * W); hm = heart(hs, max(2, int(4 * scale * 2)))
    hx, hy = min(W - hm.shape[1] - int(0.03 * W), bxp + bb.shape[1] - int(0.01 * W)), byp + int(bb.shape[0] * 0.45)
    paste(img, np.zeros((H, W), np.float32), np.ones(hm.shape + (3,), np.float32) * 245, hm.astype(np.float32), hx, hy)
    layers["BOTTOM"] = (bb[..., :3], bb[..., 3] / 255, bxp, byp)
    # --- distressed edge -> transparent outside (hard alpha) ---
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.minimum.reduce([xx, W - 1 - xx, yy, H - 1 - yy]) / (0.035 * W)
    ed = noise(H, W, W * 0.02, seed + 99, 4)
    A = ((d + (ed - 0.5) * 1.6) > 0.55).astype(np.float32)
    A *= (noise(H, W, W * 0.004, seed + 98, 2) > 0.12) | (d > 1.2)
    img = np.clip(img, 0, 255)
    out = np.dstack([img, A * 255]).astype(np.uint8); out[A == 0, :3] = 0
    if layers_dir:
        os.makedirs(layers_dir, exist_ok=True)
        bgl = np.dstack([np.clip(bg_only, 0, 255), A * 255]).astype(np.uint8); Image.fromarray(bgl, "RGBA").save(os.path.join(layers_dir, "01_background.png"), dpi=(300 * scale,) * 2)
        for nm_, (r_, a_, x_, y_) in layers.items():
            L = np.zeros((H, W, 4), np.uint8); sub = np.zeros((H, W), np.float32); tmp = np.zeros((H, W, 3), np.float32)
            paste(tmp, sub, r_, a_, x_, y_); L[..., :3] = np.clip(tmp, 0, 255); L[..., 3] = (sub >= 0.5) * 255
            Image.fromarray(L, "RGBA").save(os.path.join(layers_dir, f"layer_{nm_}.png"), dpi=(300 * scale,) * 2)
    return Image.fromarray(out, "RGBA")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--photo", required=True)
    ap.add_argument("--ghost1"); ap.add_argument("--ghost2"); ap.add_argument("--bottom", default="North Shore Elite"); ap.add_argument("--year", default="2026")
    ap.add_argument("--out"); ap.add_argument("--scale", type=float, default=1.0); ap.add_argument("--layers")
    a = ap.parse_args()
    im = build(a.name, a.photo, a.ghost1, a.ghost2, a.bottom, a.year, a.scale, layers_dir=a.layers)
    out = a.out or f"NSE_CHEER_{a.name.replace(' ', '_')}_12x14.67in_300dpi.png"
    im.save(out, dpi=(300 * a.scale,) * 2); print(out, im.size)
