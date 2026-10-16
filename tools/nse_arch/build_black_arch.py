"""North Shore Elite — black arch letters + blue script Elite (Brian's vertical upload 2026-10-16)
-> 48 x 10 in sleeve-to-sleeve print, hard alpha, flat colours. Fixes ONLY the five listed problems;
letters, script, keylines and spacing are Brian's art as uploaded.

usage: python3 build_black_arch.py SOURCE.png OUT.png TW TH [stretch|fit] [--angle DEG]
 1. turn the vertical upload 90 deg clockwise (reads left to right)
 2. per pixel coverage of clear / black / white / Elite fill blue / Elite edge blue (alpha x colour unmix)
 3. faint dust (alpha < 50%) -> clear; specks under 3 x 3 px -> surrounding colour
 4. missing black outer outline: every clear pixel within 10 px (the outline's measured width) of a letter's
    white keyline becomes black — restores it where it was cut (tops of T-H / S-H, left of the N)
    and leaves it alone where it is already there
 5. Elite: streaky fill -> one flat blue; outer band -> its own flat blue; soft white edge -> crisp
 6. level the arch: rotate so the N and E feet / tops sit level (angle measured, ~1.5 deg)
 7. each colour smoothed and enlarged on its own; every output pixel takes the strongest -> crisp edges
"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd

Image.MAX_IMAGE_PIXELS = None
PAL = {  # measured from the upload
    'black': (10, 10, 12), 'white': (255, 255, 255), 'fill': (0, 41, 161), 'edge': (0, 28, 158)}
CLEAR, BLK, WHT, FIL, EDG = range(5)
OUT = np.array([[0, 0, 0, 0], [*PAL['black'], 255], [*PAL['white'], 255], [*PAL['fill'], 255], [*PAL['edge'], 255]], np.uint8)
OUTLINE = 10.0          # outer black outline width, src px (measured median 10.0, p10 9.1)

def classes(src):
    im = Image.open(src).convert('RGBA').transpose(Image.Transpose.ROTATE_270)
    a = np.asarray(im).astype(np.float32); al = a[..., 3] / 255.0; rgb = a[..., :3]
    # unmix over black / white / blue (straight alpha)
    K = np.array(PAL['black'], np.float32); Wc = np.array([254.9] * 3, np.float32); Bc = np.array([0, 36, 160], np.float32)
    M = np.stack([Wc - K, Bc - K], 1)
    ab = np.clip((rgb.reshape(-1, 3) - K) @ np.linalg.pinv(M).T, 0, None)
    s = ab.sum(1, keepdims=True); ab = np.where(s > 1, ab / np.maximum(s, 1e-6), ab)
    wW = ab[:, 0].reshape(al.shape); wB = ab[:, 1].reshape(al.shape); wK = np.clip(1 - wW - wB, 0, 1)
    cov = np.stack([1 - al, al * wK, al * wW, al * wB], -1)
    k4 = cov.argmax(-1)
    k4[al < 0.5] = CLEAR                                        # faint dust and soft fringe -> clear
    # specks
    for k in range(4):
        lb, n = nd.label(k4 == k)
        if n == 0: continue
        cnt = np.bincount(lb.ravel()); small = cnt < 9; small[0] = False
        sm = small[lb]
        if sm.any():
            _, (iy, ix) = nd.distance_transform_edt(sm, return_indices=True); k4[sm] = k4[iy, ix][sm]
    blue = k4 == 3
    # Elite fill vs outer band: blue enclosed by white = fill, blue touching black/clear = band
    lab, n = nd.label(blue); sz = np.bincount(lab.ravel())
    dark = (k4 == BLK) | (k4 == CLEAR)
    touch = nd.maximum(nd.binary_dilation(dark, iterations=2).astype(np.int32), lab, range(1, n + 1))
    fill = np.isin(lab, 1 + np.where(touch == 0)[0])
    cls = np.zeros(k4.shape, np.uint8)
    cls[k4 == BLK] = BLK; cls[k4 == 2] = WHT; cls[blue & fill] = FIL; cls[blue & ~fill] = EDG
    # missing outer outline: clear pixels within OUTLINE of a LETTER keyline (not Elite's white outline)
    # Elite's own white outline (~19 px) sits between its fill and its blue band, so it is always within
    # ~20 px of Elite blue; the letters' keylines touch it where the script crosses them, so no flood fill
    elite_white = (cls == WHT) & (nd.distance_transform_edt(~((cls == FIL) | (cls == EDG))) <= 21)
    letter_white = (cls == WHT) & ~elite_white
    d = nd.distance_transform_edt(~letter_white)
    add = (cls == CLEAR) & (d <= OUTLINE + 0.5)
    cls[add] = BLK
    return cls, add, im.size

def level_angle(cls):
    """angle (deg, + = counter-clockwise) that puts the outer ends of the arch level (feet and tops)"""
    ink = (cls != CLEAR).astype(np.uint8)
    sm = cv2.resize(ink, (ink.shape[1] // 4, ink.shape[0] // 4), interpolation=cv2.INTER_AREA)
    best = None
    for ang in np.arange(-3.0, 3.001, 0.05):
        h, w = sm.shape; M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
        c, s_ = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s_ + w * c), int(h * c + w * s_)
        M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
        r = cv2.warpAffine(sm, M, (nw, nh), flags=cv2.INTER_NEAREST) > 0
        xs = np.where(r.any(0))[0]; x0, x1 = xs.min(), xs.max(); band = max(8, (x1 - x0) // 40)
        L = r[:, x0:x0 + band]; Rr = r[:, x1 - band:x1 + 1]
        lt, lb_ = np.where(L.any(1))[0][[0, -1]]; rt, rb = np.where(Rr.any(1))[0][[0, -1]]
        err = abs(lt - rt) + abs(lb_ - rb)
        if best is None or err < best[0]: best = (err, ang)
    return best[1]

def render(cls, angle, TW, TH, mode):
    H, W = cls.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
    c, s_ = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(H * s_ + W * c) + 20, int(H * c + W * s_) + 20
    M[0, 2] += nw / 2 - W / 2; M[1, 2] += nh / 2 - H / 2
    covs = []
    for k in range(5):
        ch = (cls == k).astype(np.float32)
        r = cv2.warpAffine(ch, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=1.0 if k == CLEAR else 0.0)
        covs.append(cv2.GaussianBlur(r, (0, 0), 0.9))
    arg0 = np.argmax(np.stack(covs, -1), -1)
    ys, xs = np.where(arg0 != CLEAR); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 6; Y0, X0 = max(0, y0 - pad), max(0, x0 - pad)
    cy0, cx0, ch_, cw_ = y0 - Y0, x0 - X0, y1 - y0, x1 - x0
    if mode == 'stretch': AW, AH = TW, TH
    else:
        k = min(TW / cw_, TH / ch_); AW, AH = int(round(cw_ * k)), int(round(ch_ * k))
    sx, sy = AW / cw_, AH / ch_
    sub = [cv[Y0:y1 + pad, X0:x1 + pad] for cv in covs]
    PW, PH = int(round(sub[0].shape[1] * sx)), int(round(sub[0].shape[0] * sy))
    best = None
    for k in range(5):
        up = cv2.resize(sub[k], (PW, PH), interpolation=cv2.INTER_CUBIC)
        if best is None: best = up; arg = np.zeros(up.shape, np.uint8)
        else:
            m = up > best; best = np.where(m, up, best); arg[m] = k
        del up
    oy, ox = int(round(cy0 * sy)), int(round(cx0 * sx))
    arg = arg[oy:oy + AH, ox:ox + AW]
    if arg.shape != (AH, AW): arg = np.pad(arg, ((0, AH - arg.shape[0]), (0, AW - arg.shape[1])))
    canvas = np.zeros((TH, TW), np.uint8); ty, tx = (TH - AH) // 2, (TW - AW) // 2
    canvas[ty:ty + AH, tx:tx + AW] = arg
    return canvas, (AW, AH)

if __name__ == '__main__':
    src, out, TW, TH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    mode = sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith('--') else 'stretch'
    cls, add, size = classes(src)
    ang = float(sys.argv[sys.argv.index('--angle') + 1]) if '--angle' in sys.argv else level_angle(cls)
    canvas, art = render(cls, ang, TW, TH, mode)
    Image.fromarray(OUT[canvas], 'RGBA').save(out, dpi=(300, 300), optimize=True)
    np.save(out.rsplit('.', 1)[0] + '_cls_src.npy', cls)
    u, c = np.unique(canvas, return_counts=True)
    print(out, (TW, TH), mode, 'art', art, 'level angle %.2f deg' % ang, 'outline px added (src)', int(add.sum()),
          {['clear', 'black', 'white', 'fill', 'edge'][i]: round(n / canvas.size * 100, 2) for i, n in zip(u, c)})
