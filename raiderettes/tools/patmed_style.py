"""Restyle PAT-MED VARSITY to the RAIDERETTES varsity look: red letter, white inline, black outline.
Works on the padded source arrays. Returns a class map for rows [0, y1+PAD): 0 clear, 1 black, 2 white, 3 red."""
import numpy as np
from scipy import ndimage as nd

def patmed(al, lum, y1, PAD, olw, inl, ring):
    S = slice(0, y1 + PAD)
    bk = (al[S] > 0.5) & (lum[S] < 90)
    bk = nd.binary_opening(bk, iterations=1)
    lab, n = nd.label(bk); sz = nd.sum(bk, lab, range(1, n + 1))
    glyph = np.isin(lab, 1 + np.where(sz > 2000)[0])
    holes = nd.binary_fill_holes(glyph) & ~glyph
    hl, hn = nd.label(holes); hs = nd.sum(holes, hl, range(1, hn + 1))
    glyph |= np.isin(hl, 1 + np.where(hs < 500)[0])          # shine smudges inside a letter -> letter
    glyph = nd.gaussian_filter(glyph.astype(np.float32), 1.5) > 0.5
    holes = nd.binary_fill_holes(glyph) & ~glyph
    hl, hn = nd.label(holes); hs = nd.sum(holes, hl, range(1, hn + 1))
    counters = np.isin(hl, 1 + np.where(hs >= 500)[0])       # real openings: P, A, D, R
    d_out = nd.distance_transform_edt(~glyph)
    d_in = nd.distance_transform_edt(glyph)
    new = np.zeros(glyph.shape, np.uint8)
    new[glyph] = 3
    new[glyph & (d_in <= inl)] = 2
    new[~glyph & ~counters & (d_out <= olw)] = 1
    new[counters & (d_out <= ring)] = 1
    return new, int(nd.label(glyph)[1]), int(nd.label(counters)[1])

def patmed_varsity(al, lum, y1, PAD, olw, inl, ring, font, text='PAT-MEDVARSITY', widen=1.0, bold=0):
    """Option B: same positions/angles/heights, letters redrawn in the RAIDERETTES varsity font (Graduate)."""
    from PIL import Image, ImageDraw, ImageFont
    S = slice(0, y1 + PAD)
    bk = (al[S] > 0.5) & (lum[S] < 90)
    bk = nd.binary_opening(bk, iterations=1)
    lab, n = nd.label(bk); sz = nd.sum(bk, lab, range(1, n + 1))
    ids = 1 + np.where(sz > 2000)[0]
    objs = nd.find_objects(lab)
    ids = sorted(ids, key=lambda i: objs[i - 1][1].start)
    cents = np.array([nd.center_of_mass(lab == i) for i in ids])
    # circle fit through letter centres -> tangent angle per letter
    y, x = cents[:, 0], cents[:, 1]
    A = np.c_[x, y, np.ones_like(x)]; bb = -(x ** 2 + y ** 2)
    D, E, F = np.linalg.lstsq(A, bb, rcond=None)[0]; cx0, cy0 = -D / 2, -E / 2
    glyph = np.zeros(bk.shape, bool)
    probe = ImageFont.truetype(font, 400); hb = probe.getbbox('H')
    for k, i in enumerate(ids):
        ch = text[k]
        m = lab == i
        ang = np.degrees(np.arctan2(x[k] - cx0, -(y[k] - cy0)))   # rotation of the letter (clockwise +)
        # upright size of the original letter
        sl = objs[i - 1]; sub = m[sl]
        up = np.asarray(Image.fromarray(sub.astype(np.uint8) * 255).rotate(ang, expand=True, resample=Image.BILINEAR)) > 127
        yy, xx = np.where(up); h = yy.max() - yy.min() + 1; w = xx.max() - xx.min() + 1
        if ch == '-':
            capH = h; tw = w
        fs = int(400 * h / (hb[3] - hb[1])) if ch != '-' else int(400 * 250 / (hb[3] - hb[1]))
        f = ImageFont.truetype(font, fs)
        bx = f.getbbox(ch)
        img = Image.new('L', (bx[2] - bx[0] + 4, bx[3] - bx[1] + 4), 0)
        ImageDraw.Draw(img).text((2 - bx[0], 2 - bx[1]), ch, font=f, fill=255)
        # match the original letter's height exactly, width = original width * widen (keeps the spacing)
        img = img.crop(img.getbbox()).resize((max(1, int(w * widen)), h), Image.LANCZOS)
        img = img.rotate(-ang, expand=True, resample=Image.BICUBIC)
        g = np.asarray(img) > 127
        cy, cx = int(round(y[k])), int(round(x[k]))
        gy, gx = g.shape; y0, x0 = cy - gy // 2, cx - gx // 2
        glyph[y0:y0 + gy, x0:x0 + gx] |= g
    if bold:
        glyph = nd.distance_transform_edt(~glyph) <= bold
    glyph = nd.gaussian_filter(glyph.astype(np.float32), 1.0) > 0.5
    holes = nd.binary_fill_holes(glyph) & ~glyph
    hl, hn = nd.label(holes); hs = nd.sum(holes, hl, range(1, hn + 1))
    counters = np.isin(hl, 1 + np.where(hs >= 150)[0])
    glyph |= holes & ~counters
    d_out = nd.distance_transform_edt(~glyph); d_in = nd.distance_transform_edt(glyph)
    new = np.zeros(glyph.shape, np.uint8)
    new[glyph] = 3
    new[glyph & (d_in <= inl)] = 2
    new[~glyph & ~counters & (d_out <= olw)] = 1
    new[counters & (d_out <= ring)] = 1
    return new, len(ids), int(nd.label(counters)[1])


# ---------------------------------------------------------------------------------------------
# v3: PAT-MED VARSITY set as a true uniform arch (same height, same curve, even spacing)
# ---------------------------------------------------------------------------------------------
def _fit_circle(x, y, cx=None):
    if cx is None:
        A = np.c_[x, y, np.ones_like(x)]; b = -(x ** 2 + y ** 2)
        D, E, F = np.linalg.lstsq(A, b, rcond=None)[0]
        cx, cy = -D / 2, -E / 2; R = np.sqrt(cx ** 2 + cy ** 2 - F)
    else:                                   # centre constrained to the design's vertical centre line
        A = np.c_[2 * y, np.ones_like(y)]; b = (x - cx) ** 2 + y ** 2
        cy, C = np.linalg.lstsq(A, b, rcond=None)[0]; R = np.sqrt(C + cy ** 2)
    return cx, cy, R


def _geom(mask_lab, ids, cx, cy):
    g = []
    for i in ids:
        yy, xx = np.nonzero(mask_lab == i)
        r = np.hypot(xx - cx, yy - cy); ph = np.arctan2(xx - cx, -(yy - cy))
        g.append((r.min(), r.max(), ph.min(), ph.max()))
    return np.array(g)


def _ink_metrics(font, chars):
    """true ink extents per glyph at font size 1000 (cap height 750 for Graduate)"""
    from PIL import Image, ImageDraw, ImageFont
    f = ImageFont.truetype(font, 1000)
    m = {}
    for c in set(chars):
        img = Image.new('L', (1700, 1400), 0)
        ImageDraw.Draw(img).text((300, 1100), c, font=f, fill=255, anchor='ls')
        a = np.asarray(img) > 127; ys, xs = np.nonzero(a)
        m[c] = dict(adv=f.getlength(c), l=xs.min() - 300, r=xs.max() - 300 + 1, top=1100 - ys.min(), bot=1100 - ys.max() - 1)
    capP = m['H']['top'] if 'H' in m else None
    return f, m


def patmed_arc(al, lum, y1, PAD, olw, inl, ring, font, band, text='PAT-MED VARSITY',
               bold=9, widen_ref=1.25, SS=4, extra=40, verbose=True):
    """Rebuild PAT-MED VARSITY on one smooth arch.
    The arch (circle centre / radius), the letter height and the overall span are measured from the
    original letters; the circle centre is put on the design's centre line (centre of the row below).
    Every letter: same cap height, centre on the same circle, rotated to the curve, natural Graduate
    proportions x one shared width factor, font spacing + kerning + one shared tracking value.
    Returns class map rows [0, y1+PAD+extra): 0 clear, 1 black, 2 white, 3 red."""
    from PIL import Image, ImageDraw, ImageFont
    S = slice(0, y1 + PAD)
    bk = (al[S] > 0.5) & (lum[S] < 90)
    bk = nd.binary_opening(bk, iterations=1)
    lab, n = nd.label(bk); sz = nd.sum(bk, lab, range(1, n + 1))
    ids = 1 + np.where(sz > 2000)[0]
    objs = nd.find_objects(lab)
    ids = sorted(ids, key=lambda i: objs[i - 1][1].start)
    words = text.split(' ')
    chars = ''.join(words)
    assert len(ids) == len(chars), (len(ids), chars)
    isl = np.array([c != '-' for c in chars])
    B = slice(band[0] + PAD, band[1] + PAD)
    cols = np.where((al[B] > 0.5).any(0))[0]; xc = (cols.min() + cols.max()) / 2

    # ---- the arch as it is: circle through the letters' mid-points, centred on the design ----
    cm = np.array([nd.center_of_mass(lab == i) for i in ids])
    cx, cy, R = _fit_circle(cm[isl, 1], cm[isl, 0], cx=xc)
    for _ in range(3):
        g = _geom(lab, ids, cx, cy)
        rm = (g[:, 0] + g[:, 1]) / 2; pmid = (g[:, 2] + g[:, 3]) / 2
        cx, cy, R = _fit_circle((cx + rm * np.sin(pmid))[isl], (cy - rm * np.cos(pmid))[isl], cx=xc)
    g = _geom(lab, ids, cx, cy)
    capH = float(np.median(g[isl, 1] - g[isl, 0]))
    # original upright widths (for the shared width factor, as in v2: 1.25 x original)
    wo = []
    for k, i in enumerate(ids):
        sl = objs[i - 1]; sub = (lab[sl] == i)
        ang = np.degrees((g[k, 2] + g[k, 3]) / 2)
        up = np.asarray(Image.fromarray(sub.astype(np.uint8) * 255).rotate(ang, expand=True, resample=Image.BILINEAR)) > 127
        xs = np.nonzero(up.any(0))[0]; wo.append(xs.max() - xs.min() + 1)
    wo = np.array(wo, float)

    f1000, M = _ink_metrics(font, chars + 'H')
    k1 = capH / M['H']['top']                       # 1000-size units -> source px at the uniform cap height
    nat = np.array([(M[c]['r'] - M[c]['l']) * k1 for c in chars])
    sx = float(np.median((widen_ref * wo / nat)[isl]))

    # word gap and span as in v2 (letters widened 25% about their original centres)
    jD = len(words[0]) - 1; jV = jD + 1
    gapDV0 = (g[jV, 2] - g[jD, 3]) * R
    word_gap = gapDV0 - (widen_ref - 1) / 2 * (wo[jD] + wo[jV])
    span0 = (g[-1, 3] - g[0, 2]) + (widen_ref - 1) / 2 * (wo[0] + wo[-1]) / R      # target angular span

    # kerning (Graduate pair kerning via raqm), in 1000-size units
    def kern(a, b):
        return f1000.getlength(a + b) - f1000.getlength(a) - f1000.getlength(b)

    ks = k1 * sx                                    # horizontal scale: 1000-units -> source px
    fsS = int(round(1000 * k1 * SS))                # font size at supersample scale
    fS = ImageFont.truetype(font, fsS)
    capS = 750 * fsS / 1000.0

    def layout(track):
        """advance-centre arc positions (px at radius R), plus ink left/right of the whole line"""
        pos = []; u = 0.0
        for w_i, wd in enumerate(words):
            if w_i == 1:                            # word gap measured ink-to-ink
                prev = words[0][-1]; nxt = wd[0]
                # place so that ink gap D->V == word_gap
                u_prev_inkr = pos[-1][1] - M[prev]['adv'] * ks / 2 + M[prev]['r'] * ks
                u = u_prev_inkr + word_gap - M[nxt]['l'] * ks
            for c_i, c in enumerate(wd):
                if c_i > 0:
                    u += kern(wd[c_i - 1], c) * ks + track
                pos.append((c, u + M[c]['adv'] * ks / 2))
                u += M[c]['adv'] * ks
        left = pos[0][1] - M[pos[0][0]]['adv'] * ks / 2 + M[pos[0][0]]['l'] * ks
        right = pos[-1][1] - M[pos[-1][0]]['adv'] * ks / 2 + M[pos[-1][0]]['r'] * ks
        return pos, left, right

    H1 = y1 + PAD + extra; W1 = al.shape[1]

    def render(pos, shift):
        acc = np.zeros((H1 * SS, W1 * SS), np.uint8)
        for c, uc in pos:
            th = (uc - shift) / R
            gw = int(M[c]['adv'] * fsS / 1000 * 1.0) + 4 * SS + 40
            gh = int(capS) + 4 * SS + 40
            ox, oy = 20, int(capS) + 20
            img = Image.new('L', (gw, gh), 0)
            ImageDraw.Draw(img).text((ox, oy), c, font=fS, fill=255, anchor='ls')
            nw = max(1, int(round(gw * sx)))
            img = img.resize((nw, gh), Image.LANCZOS)
            rcx = (ox + M[c]['adv'] * fsS / 1000 / 2) * (nw / gw); rcy = oy - capS / 2
            Rm = int(np.ceil(np.hypot(max(rcx, img.width - rcx), max(rcy, img.height - rcy)))) + 4
            big = Image.new('L', (2 * Rm, 2 * Rm), 0)
            big.paste(img, (int(round(Rm - rcx)), int(round(Rm - rcy))))
            rot = np.asarray(big.rotate(-np.degrees(th), resample=Image.BICUBIC))
            X = (cx + R * np.sin(th)) * SS; Y = (cy - R * np.cos(th)) * SS
            x0 = int(round(X - Rm)); y0 = int(round(Y - Rm))
            ya, yb = max(0, y0), min(acc.shape[0], y0 + 2 * Rm); xa, xb = max(0, x0), min(acc.shape[1], x0 + 2 * Rm)
            np.maximum(acc[ya:yb, xa:xb], rot[ya - y0:yb - y0, xa - x0:xb - x0], out=acc[ya:yb, xa:xb])
        cov = acc.reshape(H1, SS, W1, SS).mean((1, 3)) / 255.0
        return cov

    def span_of(cov):
        yy, xx = np.nonzero(cov > 0.5)
        ph = np.arctan2(xx - cx, -(yy - cy))
        return ph.min(), ph.max()

    # solve the shared tracking so the arch spans what it spans now, then centre it on the design
    track = 0.0
    for it in range(4):
        pos, left, right = layout(track)
        shift = (left + right) / 2
        cov = render(pos, shift)
        a0, a1 = span_of(cov)
        err = span0 - (a1 - a0)
        nletters_gaps = len(chars) - len(words)
        track += err * R / nletters_gaps
        if verbose: print(f'   iter {it}: span {np.degrees(a1-a0):.3f} deg (target {np.degrees(span0):.3f}), track -> {track:.2f}px')
        if abs(err * R) < 0.5: break
    pos, left, right = layout(track)
    cov = render(pos, (left + right) / 2)
    a0, a1 = span_of(cov)
    cov = render(pos, (left + right) / 2 + (a0 + a1) / 2 * R)      # balance the angular span left/right
    glyph = cov > 0.5

    if verbose:
        print(f'   arc centre ({cx-PAD:.0f},{cy-PAD:.0f}) R {R:.0f}; cap {capH:.1f}px; width factor {sx:.3f}; '
              f'tracking {track:.1f}px; word gap {word_gap:.0f}px')

    # ---- same finish as v2: bold, smooth, openings, white inline, black outline ----
    if bold:
        glyph = nd.distance_transform_edt(~glyph) <= bold
    glyph = nd.gaussian_filter(glyph.astype(np.float32), 1.0) > 0.5
    holes = nd.binary_fill_holes(glyph) & ~glyph
    hl, hn = nd.label(holes); hs = nd.sum(holes, hl, range(1, hn + 1))
    counters = np.isin(hl, 1 + np.where(hs >= 150)[0])
    glyph |= holes & ~counters
    d_out = nd.distance_transform_edt(~glyph); d_in = nd.distance_transform_edt(glyph)
    new = np.zeros(glyph.shape, np.uint8)
    new[glyph] = 3
    new[glyph & (d_in <= inl)] = 2
    new[~glyph & ~counters & (d_out <= olw)] = 1
    new[counters & (d_out <= ring)] = 1
    info = dict(cx=cx, cy=cy, R=R, capH=capH, sx=sx, track=track, word_gap=word_gap, pos=pos,
                shift=(left + right) / 2, n=len(ids), counters=int(nd.label(counters)[1]))
    return new, info


# ---------------------------------------------------------------------------------------------
# v3b: same arch, with spacing equalised between every pair of neighbouring letters
#   spacing='font'    : font advance + kerning + one tracking value (patmed_arc)
#   spacing='closest' : nearest edge-to-edge distance equal for every pair
#   spacing='optical' : open space between neighbours equal (mean gap over the letter height,
#                       open counters counted only up to a limited depth)
# ---------------------------------------------------------------------------------------------
def patmed_arc_even(al, lum, y1, PAD, olw, inl, ring, font, band, text='PAT-MED VARSITY',
                    bold=9, widen_ref=1.25, SS=4, extra=40, spacing='optical', depth=0.22, verbose=True):
    from PIL import Image, ImageDraw, ImageFont
    S = slice(0, y1 + PAD)
    bk = (al[S] > 0.5) & (lum[S] < 90)
    bk = nd.binary_opening(bk, iterations=1)
    lab, n = nd.label(bk); sz = nd.sum(bk, lab, range(1, n + 1))
    ids = 1 + np.where(sz > 2000)[0]
    objs = nd.find_objects(lab)
    ids = sorted(ids, key=lambda i: objs[i - 1][1].start)
    words = text.split(' '); chars = ''.join(words)
    assert len(ids) == len(chars), (len(ids), chars)
    isl = np.array([c != '-' for c in chars])
    B = slice(band[0] + PAD, band[1] + PAD)
    cols = np.where((al[B] > 0.5).any(0))[0]; xc = (cols.min() + cols.max()) / 2
    cm = np.array([nd.center_of_mass(lab == i) for i in ids])
    cx, cy, R = _fit_circle(cm[isl, 1], cm[isl, 0], cx=xc)
    for _ in range(3):
        g = _geom(lab, ids, cx, cy)
        rm = (g[:, 0] + g[:, 1]) / 2; pmid = (g[:, 2] + g[:, 3]) / 2
        cx, cy, R = _fit_circle((cx + rm * np.sin(pmid))[isl], (cy - rm * np.cos(pmid))[isl], cx=xc)
    g = _geom(lab, ids, cx, cy)
    capH = float(np.median(g[isl, 1] - g[isl, 0]))
    wo = []
    for k, i in enumerate(ids):
        sl = objs[i - 1]; sub = (lab[sl] == i)
        ang = np.degrees((g[k, 2] + g[k, 3]) / 2)
        up = np.asarray(Image.fromarray(sub.astype(np.uint8) * 255).rotate(ang, expand=True, resample=Image.BILINEAR)) > 127
        xs = np.nonzero(up.any(0))[0]; wo.append(xs.max() - xs.min() + 1)
    wo = np.array(wo, float)
    f1000, M = _ink_metrics(font, chars + 'H')
    k1 = capH / M['H']['top']
    nat = np.array([(M[c]['r'] - M[c]['l']) * k1 for c in chars])
    sx = float(np.median((widen_ref * wo / nat)[isl]))
    jD = len(words[0]) - 1; jV = jD + 1
    span0 = (g[-1, 3] - g[0, 2]) + (widen_ref - 1) / 2 * (wo[0] + wo[-1]) / R
    ks = k1 * sx
    fsS = int(round(1000 * k1 * SS)); fS = ImageFont.truetype(font, fsS); capS = 750 * fsS / 1000.0
    Rb, Rt = R - capH / 2, R + capH / 2
    H1 = y1 + PAD + extra; W1 = al.shape[1]

    # upright glyph images (rendered once), rotation centre = advance centre at mid-cap
    G = {}
    for c in set(chars):
        advS = M[c]['adv'] * fsS / 1000
        gw = int(advS) + 40; gh = int(capS) + 4 * SS + 40
        ox, oy = 20, int(capS) + 20
        img = Image.new('L', (gw, gh), 0)
        ImageDraw.Draw(img).text((ox, oy), c, font=fS, fill=255, anchor='ls')
        nw = max(1, int(round(gw * sx))); img = img.resize((nw, gh), Image.LANCZOS)
        rcx = (ox + advS / 2) * (nw / gw); rcy = oy - capS / 2
        Rm = int(np.ceil(np.hypot(max(rcx, img.width - rcx), max(rcy, img.height - rcy)))) + 4
        big = Image.new('L', (2 * Rm, 2 * Rm), 0)
        big.paste(img, (int(round(Rm - rcx)), int(round(Rm - rcy))))
        G[c] = (big, Rm)

    def window(c, th):
        big, Rm = G[c]
        rot = np.asarray(big.rotate(-np.degrees(th), resample=Image.BICUBIC))
        X = (cx + R * np.sin(th)) * SS; Y = (cy - R * np.cos(th)) * SS
        x0 = int(round(X - Rm)); y0 = int(round(Y - Rm))
        xa = (x0 // SS) * SS; ya = (y0 // SS) * SS
        wW = ((x0 + 2 * Rm - xa) // SS + 1) * SS; wH = ((y0 + 2 * Rm - ya) // SS + 1) * SS
        win = np.zeros((wH, wW), np.uint8); win[y0 - ya:y0 - ya + 2 * Rm, x0 - xa:x0 - xa + 2 * Rm] = rot
        cov = win.reshape(wH // SS, SS, wW // SS, SS).mean((1, 3)) / 255.0
        return cov > 0.5, xa // SS, ya // SS

    def pair_gap(wa, wb):
        ma, xa, ya = wa; mb, xb, yb = wb
        X0, Y0 = min(xa, xb), min(ya, yb)
        X1 = max(xa + ma.shape[1], xb + mb.shape[1]); Y1 = max(ya + ma.shape[0], yb + mb.shape[0])
        A = np.zeros((Y1 - Y0, X1 - X0), bool); Bm = np.zeros_like(A)
        A[ya - Y0:ya - Y0 + ma.shape[0], xa - X0:xa - X0 + ma.shape[1]] = ma
        Bm[yb - Y0:yb - Y0 + mb.shape[0], xb - X0:xb - X0 + mb.shape[1]] = mb
        if spacing == 'closest':
            return float(nd.distance_transform_edt(~A)[Bm].min())
        # optical: arc gap per radius level between A's right edge and B's left edge
        yyA, xxA = np.nonzero(A); yyB, xxB = np.nonzero(Bm)
        def prof(yy, xx):
            r = np.hypot(xx + X0 - cx, yy + Y0 - cy); ph = np.arctan2(xx + X0 - cx, -(yy + Y0 - cy))
            return np.round(r).astype(int), ph
        rA, pA = prof(yyA, xxA); rB, pB = prof(yyB, xxB)
        lo = int(round(Rb)); hi = int(round(Rt))
        mxA = np.full(hi - lo + 1, -np.inf); mnB = np.full(hi - lo + 1, np.inf)
        sA = (rA >= lo) & (rA <= hi); sB = (rB >= lo) & (rB <= hi)
        np.maximum.at(mxA, rA[sA] - lo, pA[sA]); np.minimum.at(mnB, rB[sB] - lo, pB[sB])
        both = np.isfinite(mxA) & np.isfinite(mnB)
        rr = np.arange(lo, hi + 1)
        gap = (mnB - mxA)[both] * rr[both]
        if gap.size == 0:
            return float(nd.distance_transform_edt(~A)[Bm].min())
        gmin = gap.min()
        return float(np.minimum(gap, gmin + depth * capH).mean())

    # start from the font layout (advance widths), then equalise
    u = []; pos = 0.0
    for w_i, wd in enumerate(words):
        if w_i:
            pos += 250
        for c_i, c in enumerate(wd):
            if c_i:
                pos += 40
            u.append(pos + M[c]['adv'] * ks / 2); pos += M[c]['adv'] * ks
    u = np.array(u)
    word_pair = jD                                   # pair index D->V

    def measure(u, shift):
        W_ = [window(c, (u[k] - shift) / R) for k, c in enumerate(chars)]
        gaps = np.array([pair_gap(W_[k], W_[k + 1]) for k in range(len(chars) - 1)])
        yy, xx = np.nonzero(W_[0][0]); ph0 = np.arctan2(xx + W_[0][1] - cx, -(yy + W_[0][2] - cy)).min()
        yy, xx = np.nonzero(W_[-1][0]); ph1 = np.arctan2(xx + W_[-1][1] - cx, -(yy + W_[-1][2] - cy)).max()
        return W_, gaps, ph0, ph1

    # word-gap target: the original D-V gap (same yardstick), minus the 25% widening used in v2
    def orig_win(i):
        sl = objs[i - 1]; return (lab[sl] == i, sl[1].start, sl[0].start)
    gW_target = pair_gap(orig_win(ids[jD]), orig_win(ids[jV])) - (widen_ref - 1) / 2 * (wo[jD] + wo[jV])

    shift = (u[0] + u[-1]) / 2
    for it in range(12):
        W_, gaps, ph0, ph1 = measure(u, shift)
        inner = np.delete(gaps, word_pair)
        span_err = (span0 - (ph1 - ph0)) * R
        g_star = inner.mean() + (span_err - (gW_target - gaps[word_pair])) / len(inner)
        tgt = np.full(len(gaps), g_star); tgt[word_pair] = gW_target
        u = u + np.r_[0, np.cumsum(tgt - gaps)]
        shift = shift + (ph0 + ph1) / 2 * R
        if verbose:
            print(f'   iter {it}: letter gaps {inner.min():.1f}-{inner.max():.1f} (target {g_star:.1f}), '
                  f'word {gaps[word_pair]:.1f} (target {gW_target:.1f}), span err {span_err:.1f}px')
        if np.abs(tgt - gaps).max() < 0.6 and abs(span_err) < 1.0:
            break
    W_, gaps, ph0, ph1 = measure(u, shift)
    cov = np.zeros((H1, W1), bool)
    for m_, x0, y0 in W_:
        cov[y0:y0 + m_.shape[0], x0:x0 + m_.shape[1]] |= m_
    glyph = cov
    if verbose:
        inner = np.delete(gaps, word_pair)
        print(f'   [{spacing}] arc centre ({cx-PAD:.0f},{cy-PAD:.0f}) R {R:.0f}; cap {capH:.1f}px; width x{sx:.3f}; '
              f'letter gaps {inner.min():.1f}-{inner.max():.1f}px; word gap {gaps[word_pair]:.1f}px')
    if bold:
        glyph = nd.distance_transform_edt(~glyph) <= bold
    glyph = nd.gaussian_filter(glyph.astype(np.float32), 1.0) > 0.5
    holes = nd.binary_fill_holes(glyph) & ~glyph
    hl, hn = nd.label(holes); hs = nd.sum(holes, hl, range(1, hn + 1))
    counters = np.isin(hl, 1 + np.where(hs >= 150)[0])
    glyph |= holes & ~counters
    d_out = nd.distance_transform_edt(~glyph); d_in = nd.distance_transform_edt(glyph)
    new = np.zeros(glyph.shape, np.uint8)
    new[glyph] = 3
    new[glyph & (d_in <= inl)] = 2
    new[~glyph & ~counters & (d_out <= olw)] = 1
    new[counters & (d_out <= ring)] = 1
    info = dict(cx=cx, cy=cy, R=R, capH=capH, sx=sx, gaps=gaps, n=len(ids), counters=int(nd.label(counters)[1]), spacing=spacing)
    return new, info
