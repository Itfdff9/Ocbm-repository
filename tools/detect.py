"""Per-image shirt detection: find the shirt mask, its neckline, chest centre and tilt."""
import numpy as np, cv2
from PIL import Image

def shirt_mask(img, seed):
    """img: RGB uint8. seed: (x,y) point known to be on the shirt. Returns bool mask of the shirt body."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.int16)
    sx, sy = seed
    patch = lab[sy-40:sy+40, sx-40:sx+40].reshape(-1, 3)
    ref = np.median(patch, axis=0)
    spread = np.percentile(np.abs(patch - ref), 90, axis=0)
    tol = np.maximum(spread * 2.2 + 6, [14, 7, 7])       # tolerant on L (shading), tight on a/b (colour)
    m = (np.abs(lab - ref) <= tol).all(axis=2).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(m)
    k = lab_[sy, sx]
    if k == 0:  # seed fell in a hole: take the biggest component
        k = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return lab_ == k

def analyse(img, seed, flat=False):
    """Returns dict: neck (x,y) = bottom of the neckline at centre, cx, width (chest width), angle (deg, for flatlay)."""
    H, W = img.shape[:2]
    m = shirt_mask(img, seed)
    ys, xs = np.where(m)
    if len(xs) < 5000: return None
    angle = 0.0
    if flat:
        # principal axis of the shirt body -> tilt
        pts = np.stack([xs, ys], 1).astype(np.float32)
        mean = pts.mean(0); cov = np.cov((pts - mean).T)
        ev, evec = np.linalg.eigh(cov); v = evec[:, np.argmax(ev)]   # long axis (vertical for a tee)
        angle = np.degrees(np.arctan2(v[0], v[1]))                   # deviation from vertical
        if angle > 90: angle -= 180
        if angle < -90: angle += 180
        angle = -angle if abs(angle) < 25 else 0.0
    # body column range at the seed row (excludes sleeves when the row passes under the armpits)
    row = m[seed[1]]
    xs_row = np.where(row)[0]
    # take the contiguous run containing the seed x
    runs = np.split(xs_row, np.where(np.diff(xs_row) > 1)[0] + 1)
    run = next((r for r in runs if r[0] <= seed[0] <= r[-1]), max(runs, key=len))
    cx = int((run[0] + run[-1]) / 2); body_w = int(run[-1] - run[0])
    # neckline: topmost shirt pixel near the centre columns, then walk down the collar dip at cx
    band = m[:, max(0, cx - W // 40): cx + W // 40]
    tops = [np.argmax(band[:, i]) for i in range(band.shape[1]) if band[:, i].any()]
    neck_y = int(np.percentile(tops, 50)) if tops else seed[1] - 200
    return {"cx": cx, "neck_y": neck_y, "body_w": body_w, "angle": float(angle), "mask": m}
