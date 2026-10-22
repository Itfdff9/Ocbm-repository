"""Hoodie drawstring mask, so cords that cross a front print are drawn OVER it (OCBM standing rule).
cord_mask(img, x0, x1, y0, y1) -> float mask 0..1 (1 = cord) for the whole image, searched only inside the box.
Cords are twisted rope: much higher fine-texture energy than the smooth fleece around them, in thin vertical runs."""
import numpy as np, cv2

def cord_mask(img, x0, x1, y0, y1, k=2.6, min_h=None, max_w=None, cord_half=6.5):
    H, W = img.shape[:2]; s = W / 1000.0
    x0, x1, y0, y1 = [int(round(v)) for v in (max(0, x0), min(W, x1), max(0, y0), min(H, y1))]
    L = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    hp = L - cv2.GaussianBlur(L, (0, 0), 5 * s)
    en = np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 1.5 * s))
    reg = en[y0:y1, x0:x1]
    ref = np.percentile(reg, 40) + 0.6          # smooth-fleece texture level (+ floor for black fabric)
    thr = max(k * ref, 0.28 * np.percentile(reg, 99))
    m = (reg > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(7 * s) | 1, int(5 * s) | 1), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((int(3 * s) | 1, int(3 * s) | 1), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros_like(m)
    mh = min_h or 45 * s; mw = max_w or 70 * s
    half = cord_half * s                       # a Gildan 18500 cord is ~13 px wide on a 1000 px flat
    keep = keep.astype(np.float32)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (h >= mh and w <= mw and h >= 2.2 * w): continue
        comp = lab == i
        for r in range(y, y + h):
            xs = np.where(comp[r])[0]
            if len(xs) == 0: continue
            wts = reg[r, xs]; c = float((xs * wts).sum() / max(wts.sum(), 1e-6))
            lo, hi = int(np.floor(c - half)), int(np.ceil(c + half)) + 1
            keep[r, max(0, lo):min(keep.shape[1], hi)] = 1
    full = np.zeros((H, W), np.float32); full[y0:y1, x0:x1] = keep
    return cv2.GaussianBlur(full, (0, 0), 1.2 * s)

def lay_over(result, blank, mask):
    """put the blank's cord pixels back on top of the printed result"""
    m = mask[..., None]
    return (result.astype(np.float32) * (1 - m) + blank.astype(np.float32) * m).round().astype(np.uint8)
