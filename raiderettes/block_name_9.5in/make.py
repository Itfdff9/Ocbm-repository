"""Raiderettes name designs -> 9.5 x 9.5 in square @300 DPI (2850 px) + 8K master (8192 px).
Design trimmed to its edges, scaled (no stretching) so the long side fills 9.5 in, centred, hard alpha."""
import numpy as np, sys
from PIL import Image
from scipy import ndimage as nd
U = "/root/.claude/uploads/2b454143-e1ad-5b30-a586-acd58e182021/"
names = dict(a78f3ffe="SHEA", **{"690eee18": "RENDA"}, e89f58e0="VALENTI", **{"3adab9dd": "CORSO"}, cc3b6d6e="TRAVA",
             **{"6e979b9d": "MCHUGH", "4a764574": "VELEZ", "12288da3": "DILEO", "c3df10ce": "MATTEO", "b33a3678": "PALENCIA"})
def render(a, S):
    al = a[..., 3]
    ys, xs = np.where(al >= 128); a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = a.shape[:2]; sc = S / max(h, w); nw, nh = round(w * sc), round(h * sc)
    m = a[..., 3] >= 128; rgb = a[..., :3].copy()
    _, (iy, ix) = nd.distance_transform_edt(~m, return_indices=True); rgb = rgb[iy, ix]   # bleed colour, no dark halos
    r = np.array(Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS))
    A = np.array(Image.fromarray(a[..., 3]).resize((nw, nh), Image.LANCZOS)) >= 128
    o = np.zeros((S, S, 4), np.uint8); y0, x0 = (S - nh) // 2, (S - nw) // 2
    t = np.dstack([r, A.astype(np.uint8) * 255]); t[~A] = 0
    l, _ = nd.label(A); s = np.bincount(l.ravel()); t[(s[l] < 12) & A] = 0     # loose specks
    o[y0:y0 + nh, x0:x0 + nw] = t
    return o, (nw / 300 if S == 2850 else None, nh / 300 if S == 2850 else None)
for k, n in [(k, n) for k, n in names.items() if n in sys.argv[1:]]:
    a = np.array(Image.open(U + k + "-image.webp").convert("RGBA"))
    o, (wi, hi) = render(a, 2850)
    Image.fromarray(o, "RGBA").save(f"out/{n}_Raiderettes_9.5x9.5in_300dpi.png", dpi=(300, 300))
    o8, _ = render(a, 8192)
    Image.fromarray(o8, "RGBA").save(f"out8k/{n}_Raiderettes_8K_master.png", dpi=(8192 / 9.5, 8192 / 9.5))
    print(n, f"design {wi:.2f} x {hi:.2f} in")
