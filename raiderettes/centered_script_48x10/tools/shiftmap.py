"""Shift-map inpainting (He & Sun style): dominant patch offsets + alpha-expansion graph cut (PyMaxflow).
Fills a hole by copying whole pixels from shifted copies of the known image, seams placed where shifts agree."""
import numpy as np, cv2, time, maxflow
from scipy import ndimage as nd
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

KEY = np.array([0, 255, 0], np.float32)

def keyed(rgba):
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    return rgba[..., :3].astype(np.float32) * a + KEY * (1 - a)

def minpool(m, f):
    H, W = m.shape
    Hh, Wh = H // f, W // f
    return m[:Hh * f, :Wh * f].reshape(Hh, f, Wh, f).all(axis=(1, 3))

def maxpool(m, f):
    H, W = m.shape
    Hh, Wh = -(-H // f), -(-W // f)
    mm = np.zeros((Hh * f, Wh * f), bool); mm[:H, :W] = m
    return mm.reshape(Hh, f, Wh, f).any(axis=(1, 3))

def offset_stats(img, known, hole, K=60, tau=10, band=120, psz=5, nsample=40000, seed=0, extra=()):
    """img: float RGB (small scale), known/hole bool. Returns list of (dy,dx) offsets at this scale."""
    H, W = known.shape
    r = psz // 2
    pad = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_REFLECT)
    # patch descriptors for all pixels
    feats = np.concatenate([pad[dy:dy + H, dx:dx + W] for dy in range(psz) for dx in range(psz)], axis=2).reshape(H * W, -1)
    kpatch = minpool_known = nd.binary_erosion(known, structure=np.ones((psz, psz)), border_value=0)
    dist_h = nd.distance_transform_edt(~hole)
    qmask = kpatch & (dist_h <= band)
    db = np.flatnonzero(kpatch.ravel())
    qi = np.flatnonzero(qmask.ravel())
    rng = np.random.default_rng(seed)
    if len(qi) > nsample: qi = rng.choice(qi, nsample, replace=False)
    if len(db) > 400000: db = rng.choice(db, 400000, replace=False)
    pca = PCA(16, random_state=seed).fit(feats[rng.choice(db, min(len(db), 50000), replace=False)])
    Fdb = pca.transform(feats[db]).astype(np.float32); Fq = pca.transform(feats[qi]).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=12).fit(Fdb)
    _, ind = nn.kneighbors(Fq)
    qy, qx = np.divmod(qi, W)
    offs = []
    for j in range(ind.shape[1]):
        ty, tx = np.divmod(db[ind[:, j]], W)
        offs.append(np.stack([ty - qy, tx - qx], 1))
    offs = np.stack(offs, 1)  # n x 12 x 2
    mag = np.hypot(offs[..., 0], offs[..., 1])
    ok = mag >= tau
    first = np.argmax(ok, axis=1); has = ok.any(1)
    o = offs[np.arange(len(qi)), first][has]
    # 2D histogram
    hist = np.zeros((2 * H + 1, 2 * W + 1), np.float32)
    np.add.at(hist, (o[:, 0] + H, o[:, 1] + W), 1)
    hs = cv2.GaussianBlur(hist, (0, 0), 1.2)
    mx = nd.maximum_filter(hs, size=5)
    peaks = np.argwhere((hs == mx) & (hs > 0))
    vals = hs[peaks[:, 0], peaks[:, 1]]
    order = np.argsort(vals)[::-1]
    res = []
    for k in order:
        dy, dx = peaks[k, 0] - H, peaks[k, 1] - W
        if abs(dy) + abs(dx) < tau: continue
        res.append((int(dy), int(dx)))
        if len(res) >= K: break
    for e in extra:
        if tuple(e) not in res: res.append(tuple(e))
    return res

def refine_offset(img, known, off, rad=3, nsamp=60000, seed=1):
    H, W = known.shape
    rng = np.random.default_rng(seed)
    ys, xs = np.nonzero(known)
    if len(ys) > nsamp:
        s = rng.choice(len(ys), nsamp, replace=False); ys, xs = ys[s], xs[s]
    best = (1e18, off)
    for ddy in range(-rad, rad + 1):
        for ddx in range(-rad, rad + 1):
            dy, dx = off[0] + ddy, off[1] + ddx
            ty, tx = ys + dy, xs + dx
            v = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
            v[v] &= known[ty[v], tx[v]]
            if v.sum() < 500: continue
            d = ((img[ys[v], xs[v]] - img[ty[v], tx[v]]) ** 2).sum(1)
            c = np.mean(np.minimum(d, 0.25))
            if c < best[0]: best = (c, (dy, dx))
    return best[1]

def expand(img, known, hole, offsets, cycles=2, unknown_cost=3.0, bnd_w=2.0, init=None, verbose=True):
    """Alpha-expansion over hole pixels. img float RGB [0,1]. Returns label per hole pixel (index into offsets)."""
    H, W = hole.shape
    ys, xs = np.nonzero(hole); N = len(ys)
    idx = -np.ones((H, W), np.int64); idx[ys, xs] = np.arange(N)
    L = len(offsets)
    O = np.array(offsets, np.int64)
    def src(lbl_off, py, px):
        ty, tx = py + lbl_off[..., 0], px + lbl_off[..., 1]
        inb = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
        tyc, txc = np.clip(ty, 0, H - 1), np.clip(tx, 0, W - 1)
        ok = inb & known[tyc, txc]
        col = img[tyc, txc]
        return col, ok
    # validity per label
    valid = np.zeros((L, N), bool)
    for l in range(L):
        _, ok = src(np.broadcast_to(O[l], (N, 2)), ys, xs); valid[l] = ok
    nvalid = valid.sum(0)
    if (nvalid == 0).any():
        # give those pixels the nearest-known-pixel offset as an extra label
        bad = nvalid == 0
        _, (iy, ix) = nd.distance_transform_edt(~known, return_indices=True)
        oy = iy[ys[bad], xs[bad]] - ys[bad]; ox = ix[ys[bad], xs[bad]] - xs[bad]
        # cluster into a few offsets
        from collections import Counter
        cnt = Counter(zip((oy // 4) * 4, (ox // 4) * 4))
        add = [tuple(int(v) for v in k) for k, _ in cnt.most_common(12)]
        offsets = list(offsets) + add
        return expand(img, known, hole, offsets, cycles, unknown_cost, bnd_w, init, verbose)
    if init is None:
        mag = np.hypot(O[:, 0], O[:, 1]).astype(np.float64)
        cost = np.where(valid, mag[:, None], np.inf)
        f = np.argmin(cost, 0)
    else:
        f = init.copy()
    # edges
    E = []
    for (dy, dx) in ((0, 1), (1, 0)):
        qy, qx = ys + dy, xs + dx
        inb = (qy < H) & (qx < W)
        p = np.flatnonzero(inb); qyy, qxx = qy[inb], qx[inb]
        qi = idx[qyy, qxx]
        hh = qi >= 0
        E.append(("hh", p[hh], qi[hh]))
        kk = (~hh) & known[qyy, qxx]
        E.append(("hk", p[kk], qyy[kk], qxx[kk]))
        # reverse direction: known pixel before hole pixel
        ry, rx = ys - dy, xs - dx
        inb2 = (ry >= 0) & (rx >= 0)
        p2 = np.flatnonzero(inb2); ryy, rxx = ry[inb2], rx[inb2]
        k2 = (idx[ryy, rxx] < 0) & known[ryy, rxx]
        E.append(("hk", p2[k2], ryy[k2], rxx[k2]))
    hhP = np.concatenate([e[1] for e in E if e[0] == "hh"]); hhQ = np.concatenate([e[2] for e in E if e[0] == "hh"])
    hkP = np.concatenate([e[1] for e in E if e[0] == "hk"]); hkY = np.concatenate([e[2] for e in E if e[0] == "hk"]); hkX = np.concatenate([e[3] for e in E if e[0] == "hk"])
    def V(pp, la, lb):
        """seam cost at hole pixel set pp between labels la, lb (arrays)."""
        ca, oka = src(O[la], ys[pp], xs[pp]); cb, okb = src(O[lb], ys[pp], xs[pp])
        d = ((ca - cb) ** 2).sum(1)
        d = np.where(oka & okb, d, unknown_cost)
        return np.where(la == lb, 0.0, d)
    def Vpair(la_p, lb_p, la_q, lb_q):
        # cost between neighbor p (label a) and q (label b): diff at p plus diff at q
        return V(hhP, la_p, lb_p) + V(hhQ, la_q, lb_q)
    def bcost(lab_p):
        # boundary: hole pixel p with label l next to known pixel k (identity): compare I(k+o_l) with I(k)
        o = O[lab_p]
        ty, tx = hkY + o[:, 0], hkX + o[:, 1]
        inb = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
        tyc, txc = np.clip(ty, 0, H - 1), np.clip(tx, 0, W - 1)
        ok = inb & known[tyc, txc]
        d = ((img[tyc, txc] - img[hkY, hkX]) ** 2).sum(1)
        return bnd_w * np.where(ok, d, unknown_cost)
    def energy(f):
        e = Vpair(f[hhP], f[hhQ], f[hhP], f[hhQ]).sum() if False else (V(hhP, f[hhP], f[hhQ]) + V(hhQ, f[hhP], f[hhQ])).sum()
        e += bcost(f[hkP]).sum()
        return e
    t0 = time.time()
    if verbose: print(f"   N={N} L={L} E_hh={len(hhP)} E_hk={len(hkP)}  E0={energy(f):.1f}")
    for cyc in range(cycles):
        changed = 0
        for a in np.random.default_rng(cyc).permutation(L):
            cand = valid[a] & (f != a)
            if not cand.any(): continue
            g = maxflow.Graph[float]()
            nodes = np.asarray(g.add_nodes(N))
            if nodes.ndim == 0: nodes = int(nodes) + np.arange(N)
            # unary: x=1 -> switch to a. forbid switch where invalid
            U0 = np.zeros(N); U1 = np.zeros(N)
            U1[~valid[a]] = 1e9
            # boundary terms
            b0 = bcost(f[hkP]); b1 = bcost(np.full(len(hkP), a))
            np.add.at(U0, hkP, b0); np.add.at(U1, hkP, b1)
            # pairwise hole-hole
            fa = np.full(len(hhP), a)
            A_ = V(hhP, f[hhP], f[hhQ]) + V(hhQ, f[hhP], f[hhQ])
            B_ = V(hhP, f[hhP], fa) + V(hhQ, f[hhP], fa)      # p keeps, q -> a
            C_ = V(hhP, fa, f[hhQ]) + V(hhQ, fa, f[hhQ])      # p -> a, q keeps
            D_ = np.zeros(len(hhP))
            # submodularity fix
            viol = B_ + C_ < A_ + D_
            A_ = np.where(viol, B_ + C_, A_)
            # E = A + (C-A) x_p + (D-C) x_q + (B+C-A-D)(1-x_p) x_q
            cp = C_ - A_; cq = D_ - C_; w = B_ + C_ - A_ - D_
            np.add.at(U1, hhP, np.maximum(cp, 0)); np.add.at(U0, hhP, np.maximum(-cp, 0))
            np.add.at(U1, hhQ, np.maximum(cq, 0)); np.add.at(U0, hhQ, np.maximum(-cq, 0))
            m = np.minimum(U0, U1); U0 -= m; U1 -= m
            # x=1 (sink side) pays source cap; x=0 pays sink cap
            g.add_grid_tedges(nodes, U1, U0)
            g.add_edges(nodes[hhP], nodes[hhQ], w, np.zeros_like(w))
            g.maxflow()
            x = g.get_grid_segments(nodes)
            x &= valid[a]
            nf = np.where(x, a, f)
            if (nf != f).any():
                changed += int((nf != f).sum()); f = nf
        if verbose: print(f"   cycle {cyc}: changed {changed}, E={energy(f):.1f}, {time.time()-t0:.0f}s")
        if changed == 0: break
    return ys, xs, f

def inpaint(rgba, hole, scale=2, K=60, cycles=2, tau=None, band=None, extra=(), verbose=True):
    """rgba uint8 full res; hole bool full res. Returns rgba with hole filled by shift-map at 1/scale labeling."""
    H, W = hole.shape
    known_full = ~hole
    img_full = keyed(rgba) / 255.0
    # half-res labeling grid
    hs = maxpool(hole, scale); ks = ~hs
    Hs, Ws = hs.shape
    img_s = cv2.resize(img_full, (Ws, Hs), interpolation=cv2.INTER_AREA)
    if img_s.shape[:2] != hs.shape: img_s = img_s[:Hs, :Ws]
    # offset statistics at quarter of labeling scale
    q = 2
    img_q = cv2.resize(img_s, (Ws // q, Hs // q), interpolation=cv2.INTER_AREA)
    hq = maxpool(hs, q)[:img_q.shape[0], :img_q.shape[1]]; kq = ~hq
    t0 = time.time()
    offs_q = offset_stats(img_q, kq, hq, K=K, tau=(tau or 10), band=(band or 60))
    offs = [(dy * q, dx * q) for dy, dx in offs_q]
    offs = [refine_offset(img_s, ks, o, rad=2) for o in offs]
    for e in extra: offs.append(tuple(e))
    offs = list(dict.fromkeys(offs))
    if verbose: print(f"   offsets {len(offs)} ({time.time()-t0:.0f}s): {offs[:12]} ...")
    ys, xs, f = expand(img_s, ks, hs, offs, cycles=cycles, verbose=verbose)
    lab = -np.ones((Hs, Ws), np.int64); lab[ys, xs] = f
    # compose at full res: refine offsets to full res (x scale), small local refinement
    O = np.array(offs, np.int64) * scale
    out = rgba.copy()
    fy, fx = np.nonzero(hole)
    l = lab[np.minimum(fy // scale, Hs - 1), np.minimum(fx // scale, Ws - 1)]
    miss = l < 0
    if miss.any():
        # nearest labelled
        _, (iy, ix) = nd.distance_transform_edt(lab < 0, return_indices=True)
        l[miss] = lab[iy[fy[miss] // scale, fx[miss] // scale], ix[fy[miss] // scale, fx[miss] // scale]]
    ty = np.clip(fy + O[l, 0], 0, H - 1); tx = np.clip(fx + O[l, 1], 0, W - 1)
    okf = known_full[ty, tx]
    out[fy, fx] = rgba[ty, tx]
    # any full-res pixel whose source is unknown -> nearest known pixel (should be rare)
    if (~okf).any():
        bad = np.zeros((H, W), bool); bad[fy[~okf], fx[~okf]] = True
        good = ~hole
        _, (iy, ix) = nd.distance_transform_edt(~good, return_indices=True)
        out[bad] = rgba[iy[bad], ix[bad]]
        if verbose: print("   fallback px", int(bad.sum()))
    return out, dict(offsets=offs, labels=lab)

# ---------------- guided variant ----------------
def tile_offsets(img_b, known, hole, guide_b, tile=40, topk=3, search=None, scale_q=2, verbose=False):
    """For each hole tile, find offsets whose (blurred) known content best matches the guide (SSD), at 1/scale_q."""
    H, W = hole.shape
    q = scale_q
    Ib = cv2.resize(img_b, (W // q, H // q), interpolation=cv2.INTER_AREA)
    Gb = cv2.resize(guide_b, (W // q, H // q), interpolation=cv2.INTER_AREA)
    Kq = minpool(known, q)[:Ib.shape[0], :Ib.shape[1]]
    Hq = maxpool(hole, q)[:Ib.shape[0], :Ib.shape[1]]
    t = tile // q
    offs = []
    hq, wq = Hq.shape
    # penalise unknown source pixels by setting them to an impossible colour
    Ibk = Ib.copy(); Ibk[~Kq] = 5.0
    for y in range(0, hq, t):
        for x in range(0, wq, t):
            hm = Hq[y:y + t, x:x + t]
            if not hm.any(): continue
            y0, x0 = max(y - t // 2, 0), max(x - t // 2, 0)
            y1, x1 = min(y + t + t // 2, hq), min(x + t + t // 2, wq)
            tmpl = Gb[y0:y1, x0:x1]
            if tmpl.shape[0] >= hq or tmpl.shape[1] >= wq: continue
            res = cv2.matchTemplate(Ibk, tmpl.astype(np.float32), cv2.TM_SQDIFF)
            # forbid the trivial zero offset region (the hole itself)
            r2 = res.copy()
            for _ in range(topk):
                _, _, mnl, _ = cv2.minMaxLoc(r2)
                dx, dy = mnl[0] - x0, mnl[1] - y0
                offs.append((int(dy * q), int(dx * q)))
                cv2.rectangle(r2, (mnl[0] - 3, mnl[1] - 3), (mnl[0] + 3, mnl[1] + 3), float(r2.max()), -1)
    return offs

def inpaint_guided(rgba, hole, guide, scale=2, K=40, cycles=2, lam=4.0, gsig=2.0, tile=48, topk=3, maxtile=90, extra=(), verbose=True):
    """Shift-map with a low-frequency guide image (float RGB 0..1, full-res sized)."""
    H, W = hole.shape
    img_full = keyed(rgba) / 255.0
    hs = maxpool(hole, scale); ks = ~hs
    Hs, Ws = hs.shape
    img_s = cv2.resize(img_full, (Ws, Hs), interpolation=cv2.INTER_AREA)[:Hs, :Ws]
    g_s = cv2.resize(guide.astype(np.float32), (Ws, Hs), interpolation=cv2.INTER_AREA)[:Hs, :Ws]
    img_b = cv2.GaussianBlur(img_s, (0, 0), gsig); g_b = cv2.GaussianBlur(g_s, (0, 0), gsig)
    t0 = time.time()
    q = 2
    img_q = cv2.resize(img_s, (Ws // q, Hs // q), interpolation=cv2.INTER_AREA)
    hq = maxpool(hs, q)[:img_q.shape[0], :img_q.shape[1]]; kq = ~hq
    offs = [(dy * q, dx * q) for dy, dx in offset_stats(img_q, kq, hq, K=K, tau=10, band=60)]
    to = tile_offsets(img_b, ks, hs, g_b, tile=tile, topk=topk, scale_q=2)
    # rank tile offsets by frequency (clustered with tolerance), keep the most used
    from collections import Counter
    cl = []
    for o in to:
        for c in cl:
            if abs(o[0] - c[0][0]) + abs(o[1] - c[0][1]) <= 4: c[1] += 1; break
        else: cl.append([o, 1])
    cl.sort(key=lambda c: -c[1])
    to = [c[0] for c in cl[:maxtile]]
    offs = [o for o in offs + to if o != (0, 0)]
    uniq = []
    for o in offs:
        if all(abs(o[0] - u[0]) + abs(o[1] - u[1]) > 2 for u in uniq): uniq.append(o)
    offs = uniq + [tuple(e) for e in extra]
    if verbose: print(f"   guided offsets {len(offs)} ({time.time()-t0:.0f}s)")
    ys, xs, f = expand_guided(img_s, ks, hs, offs, img_b, g_b, lam=lam, cycles=cycles, verbose=verbose)
    lab = -np.ones((Hs, Ws), np.int64); lab[ys, xs] = f
    O = np.array(offs, np.int64) * scale
    out = rgba.copy()
    fy, fx = np.nonzero(hole)
    l = lab[np.minimum(fy // scale, Hs - 1), np.minimum(fx // scale, Ws - 1)]
    ty = np.clip(fy + O[l, 0], 0, H - 1); tx = np.clip(fx + O[l, 1], 0, W - 1)
    out[fy, fx] = rgba[ty, tx]
    okf = (~hole)[ty, tx]
    if (~okf).any():
        bad = np.zeros((H, W), bool); bad[fy[~okf], fx[~okf]] = True
        _, (iy, ix) = nd.distance_transform_edt(hole, return_indices=True)
        out[bad] = rgba[iy[bad], ix[bad]]
        if verbose: print("   fallback px", int(bad.sum()))
    return out, dict(offsets=offs, labels=lab)

def expand_guided(img, known, hole, offsets, img_b, g_b, lam=4.0, cycles=2, unknown_cost=3.0, bnd_w=2.0, verbose=True):
    H, W = hole.shape
    ys, xs = np.nonzero(hole); N = len(ys)
    idx = -np.ones((H, W), np.int64); idx[ys, xs] = np.arange(N)
    L = len(offsets); O = np.array(offsets, np.int64)
    def src(o, py, px):
        ty, tx = py + o[..., 0], px + o[..., 1]
        inb = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
        tyc, txc = np.clip(ty, 0, H - 1), np.clip(tx, 0, W - 1)
        return img[tyc, txc], inb & known[tyc, txc], tyc, txc
    valid = np.zeros((L, N), bool); data = np.zeros((L, N), np.float32)
    gp = g_b[ys, xs]
    for l in range(L):
        _, ok, tyc, txc = src(np.broadcast_to(O[l], (N, 2)), ys, xs)
        valid[l] = ok
        data[l] = lam * ((img_b[tyc, txc] - gp) ** 2).sum(1)
    if (valid.sum(0) == 0).any(): raise RuntimeError("pixels without valid offset")
    f = np.argmin(np.where(valid, data, np.inf), 0)
    E = []
    for (dy, dx) in ((0, 1), (1, 0)):
        qy, qx = ys + dy, xs + dx; inb = (qy < H) & (qx < W)
        p = np.flatnonzero(inb); qyy, qxx = qy[inb], qx[inb]; qi = idx[qyy, qxx]; hh = qi >= 0
        E.append(("hh", p[hh], qi[hh]))
        kk = (~hh) & known[qyy, qxx]; E.append(("hk", p[kk], qyy[kk], qxx[kk]))
        ry, rx = ys - dy, xs - dx; inb2 = (ry >= 0) & (rx >= 0)
        p2 = np.flatnonzero(inb2); ryy, rxx = ry[inb2], rx[inb2]
        k2 = (idx[ryy, rxx] < 0) & known[ryy, rxx]; E.append(("hk", p2[k2], ryy[k2], rxx[k2]))
    hhP = np.concatenate([e[1] for e in E if e[0] == "hh"]); hhQ = np.concatenate([e[2] for e in E if e[0] == "hh"])
    hkP = np.concatenate([e[1] for e in E if e[0] == "hk"]); hkY = np.concatenate([e[2] for e in E if e[0] == "hk"]); hkX = np.concatenate([e[3] for e in E if e[0] == "hk"])
    def V(pp, la, lb):
        ca, oka, _, _ = src(O[la], ys[pp], xs[pp]); cb, okb, _, _ = src(O[lb], ys[pp], xs[pp])
        d = ((ca - cb) ** 2).sum(1); d = np.where(oka & okb, d, unknown_cost)
        return np.where(la == lb, 0.0, d)
    def bcost(lab_p):
        o = O[lab_p]; ty, tx = hkY + o[:, 0], hkX + o[:, 1]
        inb = (ty >= 0) & (ty < H) & (tx >= 0) & (tx < W)
        tyc, txc = np.clip(ty, 0, H - 1), np.clip(tx, 0, W - 1); ok = inb & known[tyc, txc]
        d = ((img[tyc, txc] - img[hkY, hkX]) ** 2).sum(1)
        return bnd_w * np.where(ok, d, unknown_cost)
    def energy(f):
        return (V(hhP, f[hhP], f[hhQ]) + V(hhQ, f[hhP], f[hhQ])).sum() + bcost(f[hkP]).sum() + data[f, np.arange(N)].sum()
    t0 = time.time()
    if verbose: print(f"   N={N} L={L} E0={energy(f):.1f}")
    for cyc in range(cycles):
        changed = 0
        for a in np.random.default_rng(cyc).permutation(L):
            cand = valid[a] & (f != a)
            if not cand.any(): continue
            g = maxflow.Graph[float](); nodes = np.asarray(g.add_nodes(N))
            U0 = data[f, np.arange(N)].astype(np.float64).copy(); U1 = np.where(valid[a], data[a], 1e9).astype(np.float64)
            np.add.at(U0, hkP, bcost(f[hkP])); np.add.at(U1, hkP, bcost(np.full(len(hkP), a)))
            fa = np.full(len(hhP), a)
            A_ = V(hhP, f[hhP], f[hhQ]) + V(hhQ, f[hhP], f[hhQ])
            B_ = V(hhP, f[hhP], fa) + V(hhQ, f[hhP], fa)
            C_ = V(hhP, fa, f[hhQ]) + V(hhQ, fa, f[hhQ])
            A_ = np.where(B_ + C_ < A_, B_ + C_, A_)
            cp = C_ - A_; cq = -C_; w = B_ + C_ - A_
            np.add.at(U1, hhP, np.maximum(cp, 0)); np.add.at(U0, hhP, np.maximum(-cp, 0))
            np.add.at(U1, hhQ, np.maximum(cq, 0)); np.add.at(U0, hhQ, np.maximum(-cq, 0))
            m = np.minimum(U0, U1); U0 -= m; U1 -= m
            g.add_grid_tedges(nodes, U1, U0)
            g.add_edges(nodes[hhP], nodes[hhQ], w, np.zeros_like(w))
            g.maxflow()
            x = g.get_grid_segments(nodes) & valid[a]
            nf = np.where(x, a, f)
            if (nf != f).any(): changed += int((nf != f).sum()); f = nf
        if verbose: print(f"   cycle {cyc}: changed {changed}, E={energy(f):.1f}, {time.time()-t0:.0f}s")
        if changed == 0: break
    return ys, xs, f

# ---------------- fill with separate known/fill masks ----------------
def inpaint2(rgba, fill, known, scale=2, K=40, cycles=2, extra_half=(), verbose=True):
    """Fill `fill` from `known` sources. Pixels in neither are don't-care (ignored)."""
    H, W = fill.shape
    img_full = keyed(rgba) / 255.0
    fs = maxpool(fill, scale)
    ks = minpool(known, scale)
    Hs, Ws = fs.shape
    ks = np.pad(ks, ((0, Hs - ks.shape[0]), (0, Ws - ks.shape[1])))
    ks &= ~fs
    img_s = cv2.resize(img_full, (Ws, Hs), interpolation=cv2.INTER_AREA)[:Hs, :Ws]
    t0 = time.time()
    q = 2
    img_q = cv2.resize(img_s, (Ws // q, Hs // q), interpolation=cv2.INTER_AREA)
    hq = maxpool(fs, q)[:img_q.shape[0], :img_q.shape[1]]
    kq = minpool(ks, q)[:img_q.shape[0], :img_q.shape[1]] & ~hq
    offs = [(dy * q, dx * q) for dy, dx in offset_stats(img_q, kq, hq, K=K, tau=8, band=50)]
    offs = [refine_offset(img_s, ks, o, rad=2) for o in offs]
    offs += [tuple(e) for e in extra_half]
    offs = [o for o in offs if o != (0, 0)]
    offs = list(dict.fromkeys(offs))
    # keep only offsets that are valid for at least some fill pixels
    ys, xs = np.nonzero(fs)
    good = []
    for o in offs:
        ty, tx = ys + o[0], xs + o[1]
        v = (ty >= 0) & (ty < Hs) & (tx >= 0) & (tx < Ws)
        v[v] &= ks[ty[v], tx[v]]
        if v.mean() > 0.02: good.append(o)
    offs = good
    if verbose: print(f"   offsets {len(offs)} ({time.time()-t0:.0f}s)")
    # pixels with no valid offset -> add nearest-known fallback offsets
    ys, xs, f = expand(img_s, ks, fs, offs, cycles=cycles, verbose=verbose)
    lab = -np.ones((Hs, Ws), np.int64); lab[ys, xs] = f
    O = np.array(offs, np.int64) * scale
    out = rgba.copy()
    fy, fx = np.nonzero(fill)
    l = lab[np.minimum(fy // scale, Hs - 1), np.minimum(fx // scale, Ws - 1)]
    ty = np.clip(fy + O[l, 0], 0, H - 1); tx = np.clip(fx + O[l, 1], 0, W - 1)
    out[fy, fx] = rgba[ty, tx]
    okf = known[ty, tx]
    if (~okf).any():
        bad = np.zeros((H, W), bool); bad[fy[~okf], fx[~okf]] = True
        _, (iy, ix) = nd.distance_transform_edt(~known, return_indices=True)
        out[bad] = rgba[iy[bad], ix[bad]]
        if verbose: print("   fallback px", int(bad.sum()))
    return out, dict(offsets=offs, labels=lab)

def inpaint2g(rgba, fill, known, guide, scale=2, K=40, cycles=2, lam=2.0, gsig=2.0, tile=64, topk=2, maxtile=60, extra_half=(), verbose=True):
    """inpaint2 + data term from a low-frequency guide (float RGB 0..1 at full res, keyed bg)."""
    H, W = fill.shape
    img_full = keyed(rgba) / 255.0
    fs = maxpool(fill, scale); ks = minpool(known, scale)
    Hs, Ws = fs.shape
    ks = np.pad(ks, ((0, Hs - ks.shape[0]), (0, Ws - ks.shape[1]))); ks &= ~fs
    img_s = cv2.resize(img_full, (Ws, Hs), interpolation=cv2.INTER_AREA)[:Hs, :Ws]
    g_s = cv2.resize(guide.astype(np.float32), (Ws, Hs), interpolation=cv2.INTER_AREA)[:Hs, :Ws]
    img_b = cv2.GaussianBlur(img_s, (0, 0), gsig); g_b = cv2.GaussianBlur(g_s, (0, 0), gsig)
    t0 = time.time()
    q = 2
    img_q = cv2.resize(img_s, (Ws // q, Hs // q), interpolation=cv2.INTER_AREA)
    hq = maxpool(fs, q)[:img_q.shape[0], :img_q.shape[1]]
    kq = minpool(ks, q)[:img_q.shape[0], :img_q.shape[1]] & ~hq
    offs = [(dy * q, dx * q) for dy, dx in offset_stats(img_q, kq, hq, K=K, tau=8, band=50)]
    offs = [refine_offset(img_s, ks, o, rad=2) for o in offs]
    to = tile_offsets(img_b, ks, fs, g_b, tile=tile, topk=topk, scale_q=2)
    cl = []
    for o in to:
        for c in cl:
            if abs(o[0] - c[0][0]) + abs(o[1] - c[0][1]) <= 4: c[1] += 1; break
        else: cl.append([o, 1])
    cl.sort(key=lambda c: -c[1])
    offs += [c[0] for c in cl[:maxtile]]
    offs += [tuple(e) for e in extra_half]
    offs = [o for o in offs if o != (0, 0)]
    uniq = []
    for o in offs:
        if all(abs(o[0] - u[0]) + abs(o[1] - u[1]) > 1 for u in uniq): uniq.append(o)
    ys, xs = np.nonzero(fs)
    good = []
    for o in uniq:
        ty, tx = ys + o[0], xs + o[1]
        v = (ty >= 0) & (ty < Hs) & (tx >= 0) & (tx < Ws)
        v[v] &= ks[ty[v], tx[v]]
        if v.mean() > 0.01: good.append(o)
    offs = good
    if verbose: print(f"   guided offsets {len(offs)} ({time.time()-t0:.0f}s)")
    ys, xs, f = expand_guided(img_s, ks, fs, offs, img_b, g_b, lam=lam, cycles=cycles, verbose=verbose)
    lab = -np.ones((Hs, Ws), np.int64); lab[ys, xs] = f
    O = np.array(offs, np.int64) * scale
    out = rgba.copy()
    fy, fx = np.nonzero(fill)
    l = lab[np.minimum(fy // scale, Hs - 1), np.minimum(fx // scale, Ws - 1)]
    ty = np.clip(fy + O[l, 0], 0, H - 1); tx = np.clip(fx + O[l, 1], 0, W - 1)
    out[fy, fx] = rgba[ty, tx]
    okf = known[ty, tx]
    if (~okf).any():
        bad = np.zeros((H, W), bool); bad[fy[~okf], fx[~okf]] = True
        _, (iy, ix) = nd.distance_transform_edt(~known, return_indices=True)
        out[bad] = rgba[iy[bad], ix[bad]]
        if verbose: print("   fallback px", int(bad.sum()))
    return out, dict(offsets=offs, labels=lab)

# ---------------- label-guided fill ----------------
def onehot(L, ncls=3):
    return np.stack([(L == k) for k in range(ncls)], -1).astype(np.float32)

def inpaint_labels(rgba, fill, known, Lsrc, Lc, scale=2, K=30, cycles=2, lam=3.0, tile=64, topk=2, maxtile=60,
                   extra_half=(), verbose=True):
    """Shift-map whose data term forces the copied pixels' structure label (transparent / orange border / dark)
    to match the completed label map Lc."""
    H, W = fill.shape
    img_full = keyed(rgba) / 255.0
    fs = maxpool(fill, scale); ks = minpool(known, scale)
    Hs, Ws = fs.shape
    ks = np.pad(ks, ((0, Hs - ks.shape[0]), (0, Ws - ks.shape[1]))); ks &= ~fs
    img_s = cv2.resize(img_full, (Ws, Hs), interpolation=cv2.INTER_AREA)[:Hs, :Ws]
    Ls = cv2.resize(Lsrc, (Ws, Hs), interpolation=cv2.INTER_NEAREST)[:Hs, :Ws]
    Lcs = cv2.resize(Lc, (Ws, Hs), interpolation=cv2.INTER_NEAREST)[:Hs, :Ws]
    ob = cv2.GaussianBlur(onehot(Ls), (0, 0), 1.0); gb = cv2.GaussianBlur(onehot(Lcs), (0, 0), 1.0)
    t0 = time.time()
    q = 2
    img_q = cv2.resize(img_s, (Ws // q, Hs // q), interpolation=cv2.INTER_AREA)
    hq = maxpool(fs, q)[:img_q.shape[0], :img_q.shape[1]]
    kq = minpool(ks, q)[:img_q.shape[0], :img_q.shape[1]] & ~hq
    offs = [(dy * q, dx * q) for dy, dx in offset_stats(img_q, kq, hq, K=K, tau=8, band=50)]
    offs = [refine_offset(img_s, ks, o, rad=2) for o in offs]
    to = tile_offsets(ob, ks, fs, gb, tile=tile, topk=topk, scale_q=2)
    cl = []
    for o in to:
        for c in cl:
            if abs(o[0] - c[0][0]) + abs(o[1] - c[0][1]) <= 4: c[1] += 1; break
        else: cl.append([o, 1])
    cl.sort(key=lambda c: -c[1])
    offs += [c[0] for c in cl[:maxtile]]
    offs += [tuple(e) for e in extra_half]
    offs = [o for o in offs if o != (0, 0)]
    uniq = []
    for o in offs:
        if all(abs(o[0] - u[0]) + abs(o[1] - u[1]) > 1 for u in uniq): uniq.append(o)
    ys, xs = np.nonzero(fs)
    good = []
    for o in uniq:
        ty, tx = ys + o[0], xs + o[1]
        v = (ty >= 0) & (ty < Hs) & (tx >= 0) & (tx < Ws)
        v[v] &= ks[ty[v], tx[v]]
        if v.mean() > 0.01: good.append(o)
    offs = good
    if verbose: print(f"   label-guided offsets {len(offs)} ({time.time()-t0:.0f}s)")
    ys, xs, f = expand_guided(img_s, ks, fs, offs, ob, gb, lam=lam, cycles=cycles, verbose=verbose)
    lab = -np.ones((Hs, Ws), np.int64); lab[ys, xs] = f
    O = np.array(offs, np.int64) * scale
    out = rgba.copy()
    fy, fx = np.nonzero(fill)
    l = lab[np.minimum(fy // scale, Hs - 1), np.minimum(fx // scale, Ws - 1)]
    ty = np.clip(fy + O[l, 0], 0, H - 1); tx = np.clip(fx + O[l, 1], 0, W - 1)
    out[fy, fx] = rgba[ty, tx]
    okf = known[ty, tx]
    if (~okf).any():
        bad = np.zeros((H, W), bool); bad[fy[~okf], fx[~okf]] = True
        _, (iy, ix) = nd.distance_transform_edt(~known, return_indices=True)
        out[bad] = rgba[iy[bad], ix[bad]]
        if verbose: print("   fallback px", int(bad.sum()))
    return out, dict(offsets=offs, labels=lab)
