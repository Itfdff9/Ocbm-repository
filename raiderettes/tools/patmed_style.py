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
