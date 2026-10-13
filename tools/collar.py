"""Collar finder for on-model and flatlay tee photos.
Returns the centre of the neck opening (cx), the collar dip (bottom of the opening at the centre, y)
and the opening width, which sets the print scale."""
import numpy as np, cv2
from detect import shirt_mask


def _runs(idx):
    if len(idx) == 0:
        return []
    return np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)


def top_contour(m, x0, x1):
    H = m.shape[0]
    t = np.full(x1 - x0, H, np.int32)
    for i, x in enumerate(range(x0, x1)):
        col = np.where(m[:, x])[0]
        if len(col):
            t[i] = col[0]
    return t


def skin(img):
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.int16)
    Y, Cr, Cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    return (Cr >= 136) & (Cr <= 178) & (Cb >= 80) & (Cb <= 130) & (Y > 50)


def find(img, seed, flat=False, span=360):
    H, W = img.shape[:2]
    m = shirt_mask(img, seed)
    sx = int(seed[0]); x0, x1 = max(0, sx - span), min(W, sx + span)
    t = top_contour(m, x0, x1).astype(np.float32)
    t = cv2.medianBlur(t.reshape(1, -1).astype(np.float32), 5).ravel() if len(t) > 5 else t
    xs = np.arange(x0, x1)
    if flat:
        # flatlay: the neckline is the deepest point of the U in the top edge, near the seed column
        win = (np.abs(xs - sx) < span * 0.6)
        i = np.argmax(np.where(win, t, -1))
        dip = t[i]
        # opening = where the top edge is within 45% of the U depth from the dip
        shoulder = np.percentile(t[win], 10)
        depth = max(20.0, dip - shoulder)
        inside = np.where(win & (t > dip - 0.55 * depth))[0]
        run = next((r for r in _runs(inside) if r[0] <= i <= r[-1]), inside)
        cx = float((xs[run[0]] + xs[run[-1]]) / 2); ow = float(xs[run[-1]] - xs[run[0]])
        return {"cx": cx, "dip": float(dip), "open_w": ow, "mask": m}
    # on a model: the neck is the skin just above the shirt's top edge
    sk = skin(img)
    above = np.zeros(len(xs), bool)
    for i, x in enumerate(xs):
        y = int(t[i])
        if y >= H or y < 30:
            continue
        seg = sk[max(0, y - 40):y - 4, x]
        above[i] = seg.mean() > 0.6 if len(seg) else False
    runs = [r for r in _runs(np.where(above)[0]) if len(r) > 40]
    if not runs:
        return None
    # robust parabola through the neckline points (skin above the shirt edge): its vertex is the collar centre
    idx = np.where(above)[0]
    px, py = xs[idx].astype(np.float64), t[idx].astype(np.float64)
    best = None
    rng = np.random.default_rng(0)
    if len(idx) > 60:
        for _ in range(300):
            k = rng.choice(len(idx), 3, replace=False)
            try:
                A = np.polyfit(px[k], py[k], 2)
            except Exception:
                continue
            if A[0] >= -1e-5:            # image y grows downward: the U opens upward -> a < 0? (y max at centre)
                continue
            res = np.abs(np.polyval(A, px) - py); inl = res < 8
            if best is None or inl.sum() > best[1].sum():
                best = (A, inl)
    if best is not None and best[1].sum() > 80:
        A = np.polyfit(px[best[1]], py[best[1]], 2)
        vx = -A[1] / (2 * A[0])
        if px[best[1]].min() - 40 <= vx <= px[best[1]].max() + 40 and A[0] < 0:
            vy = float(np.polyval(A, vx))
            span = px[best[1]]
            return {"cx": float(vx), "dip": vy, "open_w": float(span.max() - span.min()), "mask": m, "fit": True}
    # the neck: the run whose centre is nearest the seed column, preferring wide runs
    run = min(runs, key=lambda r: abs((xs[r[0]] + xs[r[-1]]) / 2 - sx) - 0.5 * len(r))
    cx = float((xs[run[0]] + xs[run[-1]]) / 2); ow = float(xs[run[-1]] - xs[run[0]])
    mid = run[len(run) // 3: 2 * len(run) // 3 + 1]
    dip = float(np.median(t[mid])) if len(mid) else float(t[run].max())
    return {"cx": cx, "dip": dip, "open_w": ow, "mask": m}
