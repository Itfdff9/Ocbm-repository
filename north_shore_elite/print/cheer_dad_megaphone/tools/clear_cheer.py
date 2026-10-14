"""Make the leftover white inside/between the CHEER letters transparent (gaps, C opening, letter counters).
A white region counts as leftover background when it is bordered by the blue outer line, never by the black letter fill.
usage: python3 clear_cheer.py SRC OUT.png"""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as nd
src, out = sys.argv[1], sys.argv[2]
a = np.array(Image.open(src).convert("RGBA")).astype(np.float32)
rgb, A = a[..., :3], a[..., 3].copy()
H, W = A.shape
mx, mn = rgb.max(2), rgb.min(2)
white = (mn > 175) & (mx - mn < 40) & (A > 0); white[900:] = False
dark = (mx < 70) & (A > 128); blue = (rgb[..., 2] > 150) & (rgb[..., 2] - rgb[..., 0] > 60)
lab, n = nd.label(white, structure=np.ones((3, 3))); sz = nd.sum(white, lab, range(1, n + 1))
bg = np.zeros((H, W), bool)
for i in range(1, n + 1):
    if sz[i - 1] < 300: continue
    m = lab == i
    ring = nd.binary_dilation(m, iterations=5) & ~m
    if dark[ring].mean() < 0.02 and blue[ring].mean() > 0.6: bg |= m
# take the soft white/blue blend pixels on the hole edge too
edge = nd.binary_dilation(bg, iterations=3) & ~bg & (mn > 140) & ~dark
bg |= edge
A[bg] = 0
# hard alpha for DTF
solid = A >= 128
o = np.zeros((H, W, 4), np.uint8)
o[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8); o[..., 3] = np.where(solid, 255, 0); o[~solid, :3] = 0
dpi = Image.open(src).info.get("dpi", (300, 300))
Image.fromarray(o, "RGBA").save(out, dpi=(300, 300))
print(out, W, H, "cleared px", int(bg.sum()))
