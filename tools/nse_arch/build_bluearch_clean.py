"""North Shore Elite — clean blue arch + white script logo (Brian's upload 2026-10-14, on black)
-> sleeve-to-sleeve print file, hard alpha, flat palette.

Source: north_shore_elite/print/shoulder_48x10/blue_arch_clean/NSE_NorthShoreElite_BlueArch-Clean_SOURCE_as_uploaded.jpg
(1079 x 696 phone JPEG; art is only 390 px tall -> 7.7x enlargement for 10 in at 300 DPI)

Rebuild (models: realesrgan-x4plus-anime from the Real-ESRGAN v0.2.5.0 ncnn release zip):
  python3 -c "from PIL import Image; Image.open('SOURCE.jpg').convert('RGB').crop((0,118,1079,556)).save('src_crop.png')"
  python3 tools/nse_arch/upscale_rg.py MODELDIR realesrgan-x4plus-anime 4 src_crop.png rg_anime_x4.png
  python3 tools/nse_arch/build_bluearch_clean.py rg_anime_x4.png OUT_48x10.png 14400 3000 stretch
  python3 tools/nse_arch/build_bluearch_clean.py rg_anime_x4.png OUT_true-proportion.png 14400 3000 fit

usage: python3 build_bluearch_clean.py RG_X4.png OUT.png TW TH [stretch|fit]

Steps
 1. unmix every pixel into black / white / royal blue coverage (barycentric, so white->black
    anti-aliasing is not mistaken for blue)
 2. script = the 3 thick white components (E-l-i-t, t-e-s + flourish, underline)
 3. "Elites" -> "Elite": erase the s (top hook, spine, bottom loop) and the e's old exit; a new
    brush stroke carries the e straight into the original long flourish. Black outline (8.5 px)
    and blue band (6.5 px) are regrown around the edit, measured from the original.
 4. background = black outside the script silhouette -> transparent (letter counters and the
    gaps between arch letters included); black inside the script silhouette stays black ink
 5. specks (JPEG / AI debris) -> surrounding colour
 6. each colour's coverage is smoothed and enlarged on its own, then every output pixel takes the
    strongest colour: crisp curved edges, exactly 4 values (clear, black, white, blue), alpha 0/255
"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd

Image.MAX_IMAGE_PIXELS = None
S, OY = 4, 118                                   # working scale vs upload, crop row offset
W1, W2 = 8.5, 6.5                                # script black outline / blue band (src px, measured)
SRC_W = np.array([246, 248, 251], np.float32)   # upload white (JPEG)
SRC_B = np.array([1, 58, 210], np.float32)      # upload royal blue (arch + script band)
CLEAR, BLACK, WHITE, BLUE = 0, 1, 2, 3
OUT_RGBA = np.array([[0, 0, 0, 0], [0, 0, 0, 255], [255, 255, 255, 255], [0, 58, 210, 255]], np.uint8)


def unmix(rgb):
    M = np.stack([SRC_W, SRC_B], 1)
    ab = np.clip(rgb.reshape(-1, 3) @ np.linalg.pinv(M).T, 0, None)
    s = ab.sum(1, keepdims=True)
    ab = np.where(s > 1, ab / np.maximum(s, 1e-6), ab)
    return np.concatenate([1 - ab.sum(1, keepdims=True), ab], 1).reshape(rgb.shape[:2] + (3,)).astype(np.float32)


def src_grid(shape):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    return xx / S, yy / S + OY                    # upload coordinates


def bezier(p0, p1, p2, p3, n=600):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * t * t * p2 + (t ** 3) * p3


def stroke_mask(shape, pts, hw):
    m = np.zeros(shape, np.uint8)
    for (x, y), r in zip(pts, hw):
        cv2.circle(m, (int(round(x * S * 16)), int(round((y - OY) * S * 16))), int(round(r * S * 16)), 255, -1, cv2.LINE_AA, 4)
    return m > 127


def build_classes(rg_path):
    rgb = np.asarray(Image.open(rg_path).convert('RGB')).astype(np.float32)
    w = unmix(rgb)
    k3 = w.argmax(-1)                              # 0 black, 1 white, 2 blue
    H, Wd = k3.shape
    X, Y = src_grid((H, Wd))

    # --- script white components
    white = k3 == 1
    lab, n = nd.label(white)
    objs = nd.find_objects(lab); sz = nd.sum(white, lab, range(1, n + 1))
    ids = [i + 1 for i, sl in enumerate(objs) if sz[i] / S / S >= 1000 and sl[0].start / S + OY >= 285]
    SW0 = np.isin(lab, ids)
    main = max(ids, key=lambda i: (objs[i - 1][1].stop))          # the t-e-s(+flourish) component

    # --- Elite edit: erase the s and the e's old exit
    m = lab == main
    y_sep = 425 - (X - 670) * 0.47                                   # between e-loop and e-exit
    flour = (X >= 782) & (Y >= 395)                                   # original long flourish, kept
    erase = m & ~flour & (Y >= 377) & (
        (X > 705) | ((X > 670) & (Y > y_sep)) | ((Y > 460) & (X > 625)))
    # new e-exit -> flourish brush stroke (tangent to both ends)
    P0 = np.array([668.0, 440.2]); d0 = np.array([1, -0.51]); d0 /= np.linalg.norm(d0)
    P3 = np.array([782.0, 417.5]); d3 = np.array([1, -0.36]); d3 /= np.linalg.norm(d3)
    pts = bezier(P0, P0 + 15 * d0, P3 - 35 * d3, P3)
    hw = np.linspace(4.5, 5.2, len(pts))
    conn = stroke_mask((H, Wd), pts, hw)
    SW1 = (SW0 & ~erase) | conn

    # --- regrow outline + band around the edit
    d_old = nd.distance_transform_edt(~SW0) / S
    d_new = nd.distance_transform_edt(~SW1) / S
    d_edit = nd.distance_transform_edt(~(erase | conn)) / S
    env_old = nd.binary_fill_holes(d_old <= W1 + W2 + 3)
    env_new = nd.binary_fill_holes(d_new <= W1 + W2 + 3)
    aff = ((d_edit <= W1 + W2 + 6) & (env_old | env_new)) | (env_old & ~env_new)
    # the t-crossbar's outline used to curl into the s hook at its tip; regrow it round there too
    aff |= ((X - 792) ** 2 + (Y - 372) ** 2 <= 15 ** 2) & (env_old | env_new)

    # --- which black is ink and which is background
    #  * background = black reachable from the canvas edge without crossing colour or the script's
    #    own black outline zone; plus trapped pockets (letter counters, gaps between arch letters)
    #  * ink = the script's black outline and anything it encloses (gaps inside the script, loops)
    blk = k3 == 0
    openb = blk & (d_old > W1)
    lb, nb = nd.label(openb)
    edge_ids = np.unique(np.concatenate([lb[0], lb[-1], lb[:, 0], lb[:, -1]]))
    ext = np.isin(lb, edge_ids[edge_ids > 0])
    trapped = blk & ~ext
    lt, nt = nd.label(trapped, structure=np.ones((3, 3)))
    mind = nd.minimum(d_old, lt, range(1, nt + 1))
    ink = np.isin(lt, 1 + np.where(mind <= W1 + 1)[0])

    cls = np.full((H, Wd), CLEAR, np.uint8)
    cls[k3 == 1] = WHITE; cls[k3 == 2] = BLUE; cls[ink] = BLACK
    # what lies under the removed s: above the new stroke it sat on the arch letters (royal blue,
    # like the band/arch merge everywhere else between script strokes); below it, open background
    xs_c = np.concatenate([[600.0], pts[:, 0], [900.0]])
    ys_c = np.concatenate([[440.2 + 0.51 * 68], pts[:, 1], [417.5 - 0.36 * 118]])
    yc = np.interp(X, xs_c, ys_c)
    above = (Y < yc) & (X >= 700)
    core = d_new <= W1 + W2
    hole = nd.binary_fill_holes(core) & ~core                          # enclosed by the script -> ink
    under = np.where(above, BLUE, np.where(hole, BLACK, CLEAR)).astype(np.uint8)
    new = np.where(SW1, WHITE, np.where(d_new <= W1, BLACK, np.where(core, BLUE, under)))
    cls[aff] = new[aff]
    # blend the seam where regrown outline meets the original (widths differ by ~1 px there)
    bd = aff ^ nd.binary_erosion(aff)
    seam = (nd.distance_transform_edt(~bd) <= 4 * S) & ~SW1 & ~(SW0 & ~aff)
    g = np.stack([cv2.GaussianBlur((cls == k).astype(np.float32), (0, 0), 2 * S) for k in range(4)], -1)
    sm = g.argmax(-1).astype(np.uint8)
    cls[seam] = sm[seam]
    aff = aff | seam

    # --- JPEG ringing: a dark hairline between the blue letters and their white keyline reads as
    #     background; it is not a design line -> give it to the nearer of blue / white
    fixed = np.zeros((H, Wd), bool)
    dB = nd.distance_transform_edt(cls != BLUE) / S
    dWt = nd.distance_transform_edt(cls != WHITE) / S
    fringe = (cls == CLEAR) & (dB <= 1.5) & (dWt <= 1.5)
    cls[fringe] = np.where(dB[fringe] <= dWt[fringe], BLUE, WHITE); fixed |= fringe
    # small background pockets: keep the real ones (letter notches/counters, walled by the white
    # keyline); fill compression shadows (walled by blue), hairlines, and leftovers in the edit
    lb, nn = nd.label(cls == CLEAR)
    cnt = np.bincount(lb.ravel())
    objs = nd.find_objects(lb)
    for i in np.where((cnt < 40 * S * S) & (np.arange(len(cnt)) > 0))[0]:
        sl = tuple(slice(max(0, s.start - 4), s.stop + 4) for s in objs[i - 1])
        comp = lb[sl] == i
        ring = nd.binary_dilation(comp, iterations=2) & ~comp
        rc = cls[sl][ring]
        thin = nd.distance_transform_edt(comp).max() / S <= 0.75
        fb, fw = (rc == BLUE).mean(), (rc == WHITE).mean()
        if fb > 0.5 or thin or aff[sl][comp].any() or (cnt[i] < 12 * S * S and fw < 0.5):
            sub = cls[sl].copy()
            _, (iy, ix) = nd.distance_transform_edt(comp, return_indices=True)
            sub[comp] = sub[iy, ix][comp]
            cls[sl] = sub
            f = fixed[sl]; f |= comp

    # --- specks: any island of one colour under SPECK src px^2 -> surrounding colour
    for k, SPECK in ((BLACK, 4.0), (WHITE, 4.0), (BLUE, 4.0)):
        mk = cls == k
        lb, nn = nd.label(mk)
        if nn == 0: continue
        cnt = np.bincount(lb.ravel()); small = cnt < SPECK * S * S
        small[0] = False
        sm = small[lb]
        if sm.any():
            _, (iy, ix) = nd.distance_transform_edt(sm, return_indices=True)
            cls[sm] = cls[iy, ix][sm]; fixed |= sm
    return cls, w, k3, fixed | aff, dict(erase=erase, conn=conn, aff=aff, SW0=SW0, SW1=SW1)


def coverage(cls, w, k3, hard):
    """4-channel coverage: soft unmix weights where the art is untouched, one-hot where edited"""
    cov = np.zeros(cls.shape + (4,), np.float32)
    bgsoft = cv2.GaussianBlur((cls == CLEAR).astype(np.float32), (0, 0), 1.0)
    cov[..., CLEAR] = w[..., 0] * bgsoft
    cov[..., BLACK] = w[..., 0] * (1 - bgsoft)
    cov[..., WHITE] = w[..., 1]
    cov[..., BLUE] = w[..., 2]
    # where the class was decided by rules (edit, speck fix), or the soft weights disagree, use one-hot
    disagree = (np.where(cls == CLEAR, BLACK, cls) - 1) != k3
    oh = hard | disagree
    cov[oh] = 0
    cov[oh, cls[oh]] = 1
    return cov


def render(cov, TW, TH, mode='stretch', sigma=0.9):
    k = cov.argmax(-1)
    ys, xs = np.where(k != CLEAR)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad = 6
    cov = cov[max(0, y0 - pad):y1 + pad, max(0, x0 - pad):x1 + pad]
    # content box inside the padded crop (so the art lands edge-to-edge after resize)
    cy0, cx0 = y0 - max(0, y0 - pad), x0 - max(0, x0 - pad)
    ch_, cw_ = y1 - y0, x1 - x0
    if mode == 'stretch':
        AW, AH = TW, TH
    else:
        s = min(TW / cw_, TH / ch_); AW, AH = int(round(cw_ * s)), int(round(ch_ * s))
    sx, sy = AW / cw_, AH / ch_
    PW, PH = int(round(cov.shape[1] * sx)), int(round(cov.shape[0] * sy))
    best = None; arg = None
    for c in range(4):
        ch = cv2.GaussianBlur(cov[..., c], (0, 0), sigma)
        up = cv2.resize(ch, (PW, PH), interpolation=cv2.INTER_CUBIC)
        if best is None:
            best = up; arg = np.zeros(up.shape, np.uint8)
        else:
            msk = up > best; best = np.where(msk, up, best); arg[msk] = c
        del up
    oy, ox = int(round(cy0 * sy)), int(round(cx0 * sx))
    arg = arg[oy:oy + AH, ox:ox + AW]
    if arg.shape != (AH, AW):
        arg = np.pad(arg, ((0, AH - arg.shape[0]), (0, AW - arg.shape[1])))
    canvas = np.zeros((TH, TW), np.uint8)
    ty, tx = (TH - AH) // 2, (TW - AW) // 2
    canvas[ty:ty + AH, tx:tx + AW] = arg
    return canvas


if __name__ == '__main__':
    rg, out, TW, TH = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    mode = sys.argv[5] if len(sys.argv) > 5 and not sys.argv[5].startswith('--') else 'stretch'
    cls, w, k3, hard, dbg = build_classes(rg)
    np.save(out.rsplit('.', 1)[0] + '_cls4x.npy', cls)
    cov = coverage(cls, w, k3, hard)
    lab = render(cov, TW, TH, mode)
    Image.fromarray(OUT_RGBA[lab], 'RGBA').save(out, dpi=(300, 300), optimize=True)
    u, c = np.unique(lab, return_counts=True)
    print(out, (TW, TH), mode, {['clear', 'black', 'white', 'blue'][i]: round(n / lab.size * 100, 2) for i, n in zip(u, c)})
