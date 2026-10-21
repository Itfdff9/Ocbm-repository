"""Design -> garment mockups (front lifestyle photos + S&S white-background flats, front and back).

blend(): warp the print by the garment luminance gradient, shade by local luminance,
add knit grain (x0.35) and ink opacity 0.97 (OCBM fabric-blend spec, sigmas scale with image width).
"""
import os, json, sys, glob
import numpy as np, cv2
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from flatdetect import bbox as flat_bbox

OUT_PX = 2048

def tight(logo):
    a = np.asarray(logo)[..., 3]; ys, xs = np.where(a > 8)
    return logo.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))

def blend(base, logo, cx, top, width, occ=None, opacity=0.97, grain=0.35):
    """base: RGB uint8 array. logo: RGBA PIL (tight). Places logo width px wide, centred cx, top edge at top."""
    H, W = base.shape[:2]; s = W / 1092.0
    lg = logo.resize((int(round(width)), int(round(width * logo.height / logo.width))), Image.LANCZOS)
    lw, lh = lg.size; x0, y0 = int(round(cx - lw / 2)), int(round(top))
    # clip to canvas
    X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(W, x0 + lw), min(H, y0 + lh)
    la = np.asarray(lg, np.float32)[Y0 - y0:Y1 - y0, X0 - x0:X1 - x0] / 255.0
    f = base.astype(np.float32) / 255.0
    lum = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    # warp: displace the print along the gradient of blurred garment luminance
    lb = cv2.GaussianBlur(lum, (0, 0), 5 * s)
    gx = cv2.Sobel(lb, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(lb, cv2.CV_32F, 0, 1, ksize=3)
    K, lim = 1.4 * s * 8, 2.0 * s          # gentle: folds shift the ink ~1 px, hard edges (drawstrings) capped
    h, w = Y1 - Y0, X1 - X0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = np.clip(gx[Y0:Y1, X0:X1] * K, -lim, lim); dy = np.clip(gy[Y0:Y1, X0:X1] * K, -lim, lim)
    mx = (xx + dx).astype(np.float32); my = (yy + dy).astype(np.float32)
    la = cv2.remap(la, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # shade by local luminance relative to the patch
    ls = cv2.GaussianBlur(lum, (0, 0), 3 * s)[Y0:Y1, X0:X1]
    ref = np.median(ls[la[..., 3] > 0.5]) if (la[..., 3] > 0.5).any() else ls.mean()
    shade = np.clip(ls / max(ref, 0.04), 0.72, 1.12)[..., None]
    # knit grain
    gr = (lum - cv2.GaussianBlur(lum, (0, 0), 1.2 * s))[Y0:Y1, X0:X1, None] * grain
    rgb = np.clip(la[..., :3] * shade + gr, 0, 1)
    a = la[..., 3:4] * opacity
    if occ is not None: a = a * occ[Y0:Y1, X0:X1, None]
    out = f.copy(); patch = out[Y0:Y1, X0:X1]
    out[Y0:Y1, X0:X1] = patch * (1 - a) + rgb * a
    return (np.clip(out, 0, 1) * 255).round().astype(np.uint8)

# ---------- S&S flats ----------
# placement as fractions of the garment bbox: cx = body centre, top = y0 + ft*(y1-y0),
# width = fw * body_w (tees/onesies: sleeve-free body width) or fw * bbox width (hoodies/crews: arms included)
FLAT_SPECS = json.load(open(os.path.join(os.path.dirname(__file__), "flat_specs.json")))

def to_square(img):
    """S&S flat 1000x1250 -> 2048 square on white, garment scaled to full height. Returns (canvas, scale, xoff)."""
    h, w = img.shape[:2]; k = OUT_PX / h
    r = cv2.resize(img, (int(round(w * k)), OUT_PX), interpolation=cv2.INTER_LANCZOS4)
    c = np.full((OUT_PX, OUT_PX, 3), 255, np.uint8); xo = (OUT_PX - r.shape[1]) // 2
    c[:, xo:xo + r.shape[1]] = r
    return c, k, xo

def flat_place(img, spec, med=None):
    b = flat_bbox(img)
    ok = b is not None
    if ok and med:
        if abs(b["y0"] - med["y0"]) > 30 or abs(b["y1"] - med["y1"]) > 30 or abs(b["cx"] - med["cx"]) > 25: ok = False
    if not ok:
        if not med: raise RuntimeError("no bbox")
        b = med
    hgt = b["y1"] - b["y0"]
    basew = b["body_w"] if spec.get("w_ref") == "body" else (b["x1"] - b["x0"])
    return b["cx"], b["y0"] + spec["ft"] * hgt, spec["fw"] * basew, ok

def render_flat(blank, logo, spec, med=None, out=None):
    img = np.asarray(Image.open(blank).convert("RGB"))
    cx, top, w, ok = flat_place(img, spec, med)
    sq, k, xo = to_square(img)
    res = blend(sq, logo, cx * k + xo, top * k, w * k)
    if out: Image.fromarray(res).save(out, quality=90)
    return res, ok

def median_box(files):
    bs = [flat_bbox(np.asarray(Image.open(f).convert("RGB"))) for f in files]
    bs = [b for b in bs if b]
    return {k: float(np.median([b[k] for b in bs])) for k in ("x0", "x1", "y0", "y1", "cx", "body_w")}
