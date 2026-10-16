"""Criminisi-style exemplar completion of a flat label map (structure first, edges propagate straight)."""
import numpy as np, cv2, time
from scipy import ndimage as nd

def complete(L, todo, srcok, ncls, p=13, search=150, verbose=False):
    """L: HxW uint8 labels. todo: bool, pixels to fill. srcok: bool, pixels usable as source (known).
    Returns completed copy of L (only todo pixels changed)."""
    H, W = L.shape
    r = p // 2
    L = L.copy()
    known = srcok.copy()                       # pixels with trusted values (grows as we fill)
    unknown = todo.copy()
    oh = np.zeros((H, W, ncls), np.float32)
    for k in range(ncls): oh[..., k] = (L == k)
    oh[~known] = 0
    conf = known.astype(np.float32)
    # source patch centres: patch fully inside srcok (original known)
    src_centre = cv2.erode(srcok.astype(np.uint8), np.ones((p, p), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    kern = np.ones((p, p), np.float32)
    it = 0; t0 = time.time()
    while unknown.any():
        it += 1
        kn8 = known.astype(np.uint8)
        front = unknown & (cv2.dilate(kn8, np.ones((3, 3), np.uint8)) > 0)
        if not front.any():
            break                                   # remaining unknown not reachable
        fy, fx = np.nonzero(front)
        # confidence term
        C = cv2.filter2D(conf, -1, kern, borderType=cv2.BORDER_CONSTANT)[fy, fx] / (p * p)
        # data term: label edges among known pixels inside the patch
        lab = L.copy(); lab[~known] = 255
        mx = cv2.dilate(lab, np.ones((3, 3), np.uint8)); mn = cv2.erode(np.where(known, L, 0).astype(np.uint8), np.ones((3, 3), np.uint8))
        ed = (known & (cv2.erode(kn8, np.ones((3, 3), np.uint8)) > 0) & (mx != mn) & (mx != 255)).astype(np.float32)
        D = cv2.filter2D(ed, -1, kern, borderType=cv2.BORDER_CONSTANT)[fy, fx] / p
        P = C * (D + 0.05)
        # process several far-apart front points per iteration for speed
        order = np.argsort(-P)
        taken = np.zeros((H, W), bool)
        nsel = 0
        for j in order[:400]:
            y, x = fy[j], fx[j]
            if taken[max(y - p, 0):y + p + 1, max(x - p, 0):x + p + 1].any(): continue
            taken[y, x] = True
            y0, y1, x0, x1 = y - r, y + r + 1, x - r, x + r + 1
            if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
                # clip target patch at image border: fill by nearest known later
                continue
            tpatch = oh[y0:y1, x0:x1]
            tmask = known[y0:y1, x0:x1].astype(np.float32)
            if tmask.sum() < 3: continue
            sy0, sy1 = max(y0 - search, 0), min(y1 + search, H)
            sx0, sx1 = max(x0 - search, 0), min(x1 + search, W)
            win = oh[sy0:sy1, sx0:sx1]
            m3 = np.repeat(tmask[..., None], ncls, -1)
            res = cv2.matchTemplate(win, tpatch, cv2.TM_SQDIFF, mask=m3)
            ok = src_centre[sy0 + r:sy1 - r, sx0 + r:sx1 - r]
            ok = ok[:res.shape[0], :res.shape[1]]
            if not ok.any(): continue
            res = np.where(ok, res, np.inf)
            # tie-break toward nearer sources
            yy, xx = np.mgrid[0:res.shape[0], 0:res.shape[1]]
            dist = np.hypot(yy + sy0 + r - y, xx + sx0 + r - x)
            res = res + 1e-4 * dist
            by, bx = np.unravel_index(np.argmin(res), res.shape)
            cy, cx = by + sy0 + r, bx + sx0 + r
            sp = L[cy - r:cy + r + 1, cx - r:cx + r + 1]
            need = unknown[y0:y1, x0:x1]
            L[y0:y1, x0:x1][need] = sp[need]
            for k in range(ncls): oh[y0:y1, x0:x1, k][need] = (sp[need] == k)
            cval = C[j]
            conf[y0:y1, x0:x1][need] = cval
            known[y0:y1, x0:x1] |= need
            unknown[y0:y1, x0:x1] &= ~need
            nsel += 1
            if nsel >= 12: break
        if nsel == 0:
            # fallback: fill the front by nearest known label
            _, (iy, ix) = nd.distance_transform_edt(~known, return_indices=True)
            L[front] = L[iy[front], ix[front]]
            for k in range(ncls): oh[..., k][front] = (L[front] == k)
            known |= front; unknown &= ~front
    # anything left unreachable -> nearest known
    if unknown.any():
        _, (iy, ix) = nd.distance_transform_edt(~known, return_indices=True)
        L[unknown] = L[iy[unknown], ix[unknown]]
    if verbose: print(f"   crim iters {it} {time.time()-t0:.0f}s")
    return L
