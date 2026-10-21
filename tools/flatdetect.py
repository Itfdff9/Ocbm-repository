"""Garment bbox on S&S-style white-background flats (works for white garments too)."""
import numpy as np, cv2

def garment_mask(img, thr=246):
    """img RGB uint8 -> bool mask of the garment (everything not connected to the white border)."""
    g = img.min(axis=2)
    bg_like = (g >= thr).astype(np.uint8)
    h, w = bg_like.shape
    ff = bg_like.copy(); pad = np.zeros((h + 2, w + 2), np.uint8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
        if ff[seed[1], seed[0]] == 1:
            cv2.floodFill(ff, pad, seed, 2)
    m = (ff != 2).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n <= 1: return None
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    return lab == k

def bbox(img):
    m = garment_mask(img)
    if m is None or m.sum() < 20000: return None
    ys, xs = np.where(m)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    # centre column from the lower body rows (sleeve-free on tees, symmetric on hoodies)
    r0, r1 = int(y0 + 0.55 * (y1 - y0)), int(y0 + 0.75 * (y1 - y0))
    cs = [np.where(m[r])[0] for r in range(r0, r1, 4)]
    cs = [c for c in cs if len(c)]
    cx = float(np.median([(c[0] + c[-1]) / 2 for c in cs])) if cs else (x0 + x1) / 2
    bw = float(np.median([c[-1] - c[0] for c in cs])) if cs else (x1 - x0)
    return {"x0": int(x0), "x1": int(x1), "y0": int(y0), "y1": int(y1), "cx": cx, "body_w": bw, "mask": m}
