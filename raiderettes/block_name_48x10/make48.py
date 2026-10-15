"""Raiderettes block-name uploads -> 48 x 10 in @300 DPI (14400 x 3000).
FILL: design stretched to fill the whole 48 x 10.  TRUE: design kept in its own proportions, 10 in tall, centred."""
import numpy as np, sys
from PIL import Image
from scipy import ndimage as nd
U = "/root/.claude/uploads/2b454143-e1ad-5b30-a586-acd58e182021/"
names = {"a78f3ffe": "SHEA", "690eee18": "RENDA", "e89f58e0": "VALENTI", "3adab9dd": "CORSO", "cc3b6d6e": "TRAVA",
         "6e979b9d": "MCHUGH", "4a764574": "VELEZ", "12288da3": "DILEO", "c3df10ce": "MATTEO", "b33a3678": "PALENCIA"}
W, H = 14400, 3000
def put(a, nw, nh):
    m = a[..., 3] >= 128; rgb = a[..., :3].copy()
    _, (iy, ix) = nd.distance_transform_edt(~m, return_indices=True); rgb = rgb[iy, ix]
    r = np.array(Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS))
    A = np.array(Image.fromarray(a[..., 3]).resize((nw, nh), Image.LANCZOS)) >= 128
    t = np.dstack([r, A.astype(np.uint8) * 255]); t[~A] = 0
    l, _ = nd.label(A); s = np.bincount(l.ravel()); t[(s[l] < 30) & A] = 0
    o = np.zeros((H, W, 4), np.uint8); y0, x0 = (H - nh) // 2, (W - nw) // 2
    o[y0:y0 + nh, x0:x0 + nw] = t; return o
for k, n in names.items():
    if n not in sys.argv[1:]: continue
    a = np.array(Image.open(U + k + "-image.webp").convert("RGBA"))
    ys, xs = np.where(a[..., 3] >= 128); a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape[:2]
    Image.fromarray(put(a, W, H), "RGBA").save(f"out48/{n}_Raiderettes_FILL_48x10in_300dpi.png", dpi=(300, 300))
    tw = round(w * H / h)
    Image.fromarray(put(a, tw, H), "RGBA").save(f"out48/{n}_Raiderettes_TRUE-PROPORTION_48x10in_300dpi.png", dpi=(300, 300))
    print(n, "true-proportion width", round(tw / 300, 2), "in", flush=True)
