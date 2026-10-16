"""Straight-line structure completion of a label map: each unknown pixel takes the label that is
consistent on both sides along the direction with the shortest span (borders continue straight)."""
import numpy as np, cv2, time
from scipy import ndimage as nd

def _scan_rows(known, lab):
    """For each pixel: distance to nearest known pixel to the left/right in its row, and that pixel's label."""
    H, W = known.shape
    idx = np.arange(W)[None, :].repeat(H, 0)
    li = np.where(known, idx, -1); li = np.maximum.accumulate(li, axis=1)
    ri = np.where(known, idx, W); ri = np.minimum.accumulate(ri[:, ::-1], axis=1)[:, ::-1]
    rows = np.arange(H)[:, None].repeat(W, 1)
    dl = np.where(li >= 0, idx - li, 10**6).astype(np.float32)
    dr = np.where(ri < W, ri - idx, 10**6).astype(np.float32)
    ll = np.where(li >= 0, lab[rows, np.clip(li, 0, W - 1)], 255).astype(np.uint8)
    lr = np.where(ri < W, lab[rows, np.clip(ri, 0, W - 1)], 255).astype(np.uint8)
    return dl, ll, dr, lr

def complete(L, todo, known, ndir=24, maxspan=500, verbose=False):
    t0 = time.time()
    H, W = L.shape
    best_span = np.full((H, W), np.inf, np.float32)
    best_lab = np.full((H, W), 255, np.uint8)
    near_d = np.full((H, W), np.inf, np.float32)
    near_l = np.full((H, W), 255, np.uint8)
    diag = int(np.ceil(np.hypot(H, W))) + 4
    cx, cy = W / 2, H / 2
    for k in range(ndir):
        th = 180.0 * k / ndir
        M = cv2.getRotationMatrix2D((cx, cy), th, 1.0)
        M[0, 2] += diag / 2 - cx; M[1, 2] += diag / 2 - cy
        kn_r = cv2.warpAffine(known.astype(np.uint8), M, (diag, diag), flags=cv2.INTER_NEAREST, borderValue=0) > 0
        lab_r = cv2.warpAffine(L, M, (diag, diag), flags=cv2.INTER_NEAREST, borderValue=255)
        # pixels outside the original image are "known transparent"? -> treat as unknown (no info)
        dl, ll, dr, lr = _scan_rows(kn_r, lab_r)
        Mi = cv2.invertAffineTransform(M)
        def back(x, interp=cv2.INTER_NEAREST, bv=0):
            return cv2.warpAffine(x, Mi, (W, H), flags=interp, borderValue=bv)
        dlb = back(dl, bv=10**6); drb = back(dr, bv=10**6)
        llb = back(ll, bv=255); lrb = back(lr, bv=255)
        span = dlb + drb
        cons = (llb == lrb) & (llb != 255) & (span < maxspan)
        upd = todo & cons & (span < best_span)
        best_span[upd] = span[upd]; best_lab[upd] = llb[upd]
        # nearest known along any direction (fallback)
        for d_, l_ in ((dlb, llb), (drb, lrb)):
            u2 = todo & (d_ < near_d) & (l_ != 255)
            near_d[u2] = d_[u2]; near_l[u2] = l_[u2]
    out = L.copy()
    ok = todo & (best_lab != 255)
    out[ok] = best_lab[ok]
    fb = todo & ~ok
    out[fb] = near_l[fb]
    if verbose: print(f"   linefill {ndir} dirs {time.time()-t0:.0f}s, fallback px {int(fb.sum())}")
    return out
