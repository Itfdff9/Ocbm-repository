"""NSE Laura/Khloe-style: replace the pointed megaphone tip (right end) with a blunt mouthpiece end.
Geometry measured on the 5016 px master; scales to any square size of the same art.
usage: python3 blunt_tip.py IN.png OUT.png [flat|cap] [t_px_at_5016]"""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as nd

def edit(img, mode="flat", t=360):
    a = np.array(img.convert("RGBA")).astype(np.float32)
    H, W = a.shape[:2]; k = W / 5016.0
    # edge lines on the master: top y = 0.1718x + 157.8 ; bottom x = -1.033y + 5883.9
    T = np.array([4858.6, 992.5]) * k
    axis = np.array([0.9555, -0.2951]); nrm = np.array([0.2951, 0.9555])
    C = T - t * k * axis
    half = t * k * np.tan(np.radians(26.9))
    # window
    x0, x1 = int(4250 * k), W; y0, y1 = int(800 * k), int(1420 * k)
    SS = 4
    yy, xx = np.mgrid[y0 * SS:y1 * SS, x0 * SS:x1 * SS].astype(np.float32) / SS + 0.5 / SS
    s = (xx - C[0]) * axis[0] + (yy - C[1]) * axis[1]        # along axis (toward old tip = +)
    u = (xx - C[0]) * nrm[0] + (yy - C[1]) * nrm[1]          # along the cut (down = +)
    # signed distance inside the cone (positive = inside), normals of the two edge lines
    top_n = np.array([-0.1693, 0.9855])   # inward normal of top edge
    bot_n = np.array([-0.6955, -0.7184])  # inward normal of bottom edge
    top_p = np.array([4600.0, 0.1718 * 4600 + 157.8]) * k
    bot_p = np.array([-1.033 * 1150 + 5883.9, 1150.0]) * k
    d_top = (xx - top_p[0]) * top_n[0] + (yy - top_p[1]) * top_n[1]
    d_bot = (xx - bot_p[0]) * bot_n[0] + (yy - bot_p[1]) * bot_n[1]
    wedge_loose = (d_top > -12 * k) & (d_bot > -12 * k)
    frac = np.clip((u + half) / (2 * half), 0, 1)              # 0 at top corner, 1 at bottom corner
    ob = (25 + 8 * frac) * k; wh = (35 + 9 * frac) * k; ib = (19 + 26 * frac) * k
    BLACK = np.array([6, 6, 8], np.float32); WHITE = np.array([250, 250, 250], np.float32)
    rgb_ss = np.zeros(xx.shape + (3,), np.float32); a_ss = np.zeros(xx.shape, np.float32)
    keep_orig = np.ones(xx.shape, bool)
    if mode == "flat":
        d_cut = -s
        inside = (d_top > 0) & (d_bot > 0) & (d_cut > 0)
        band = inside & (d_cut < d_top) & (d_cut < d_bot) & (d_cut < ob + wh + ib)
        rgb_ss[band] = BLACK
        w_ = band & (d_cut >= ob) & (d_cut < ob + wh); rgb_ss[w_] = WHITE
        a_ss[band] = 1
        clear = wedge_loose & (s >= 0)
        keep_orig = ~(clear | band)
    else:
        b = float(__import__("os").environ.get("CAPB", "0.36")) * half
        e = np.sqrt((u / (half + 6 * k)) ** 2 + (s / b) ** 2)  # 1 at outer ring edge
        r = (half + 6 * k)
        rad_in = lambda th: 1 - th / r
        cap = e <= 1
        rgb_ss[cap] = BLACK
        white = cap & (e <= rad_in(ob)) & (e > rad_in(ob + wh))
        rgb_ss[white] = WHITE
        a_ss[cap] = 1
        clear = wedge_loose & (s >= 0)
        keep_orig = ~(clear | cap)
    # downsample
    def ds(z): return z.reshape(z.shape[0] // SS, SS, z.shape[1] // SS, SS, *z.shape[2:]).mean((1, 3))
    new_a = ds(a_ss); new_rgb = ds(rgb_ss * a_ss[..., None]) / np.maximum(new_a, 1e-6)[..., None]
    keep = ds(keep_orig.astype(np.float32))
    win = a[y0:y1, x0:x1]
    oa = win[..., 3] / 255 * keep
    out_a = new_a + oa * (1 - new_a)
    out_rgb = (new_rgb * new_a[..., None] + win[..., :3] * (oa * (1 - new_a))[..., None]) / np.maximum(out_a, 1e-6)[..., None]
    win[..., :3] = out_rgb; win[..., 3] = out_a * 255
    # drop specks left floating where the old tip was
    A = a[..., 3] >= 128
    lab, n = nd.label(A); sz = nd.sum(A, lab, range(1, n + 1))
    reg = np.zeros((H, W), bool); reg[y0:y1, x0:x1] = ds(clear.astype(np.float32)) > 0
    reg = nd.binary_dilation(reg, iterations=int(40 * k))
    for i in np.where(sz < 900 * k * k)[0] + 1:
        m = lab == i
        if (m & reg).sum() == m.sum(): a[..., 3][m] = 0
    # hard alpha
    solid = a[..., 3] >= 128
    o = np.zeros((H, W, 4), np.uint8); o[..., :3] = np.clip(a[..., :3], 0, 255).astype(np.uint8)
    o[..., 3] = np.where(solid, 255, 0); o[~solid, :3] = 0
    return Image.fromarray(o, "RGBA")

if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else "flat"
    t = float(sys.argv[4]) if len(sys.argv) > 4 else 360
    im = Image.open(src); dpi = im.info.get("dpi", (300, 300))
    edit(im, mode, t).save(out, dpi=dpi); print(out, im.size, dpi)
