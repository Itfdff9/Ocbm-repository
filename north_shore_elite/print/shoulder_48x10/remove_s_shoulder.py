"""Shoulder print: remove the script 's' (Elites -> Elite). The s is its own white component in the traced art;
remove it with its navy outline and its inner dark fills, keep everything closer to the other letters.
usage: python3 remove_s_shoulder.py IN.png OUT.png"""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS = None
im = Image.open(sys.argv[1]); a = np.array(im); A = a[..., 3] >= 128; rgb = a[..., :3].astype(int)
white = A & (rgb.min(2) > 200)
lab, n = nd.label(white); sz = nd.sum(white, lab, range(1, n + 1))
# the s = white component whose bbox sits at y ~1874-2852 and right of the e (largest comp in that band right of centre)
cands = []
for l in range(1, n + 1):
    if sz[l - 1] < 100000: continue
    ys, xs = np.where(lab == l)
    if ys.min() > 1800 and ys.max() > 2700 and ys.min() < 1950: cands.append((xs.mean(), l))
sid = max(cands)[1]
s = lab == sid
others = white & ~s
d_s = nd.distance_transform_edt(~s); d_o = nd.distance_transform_edt(~others)
OW = 34
R = nd.binary_fill_holes(d_s <= OW) & (d_s <= d_o)
blue = (rgb[..., 2] > 150) & (rgb[..., 2] - rgb[..., 0] > 80)
R &= ~blue
a[R, 3] = 0; a[R, :3] = 0
# drop specks left behind
A2 = a[..., 3] >= 128; lab2, n2 = nd.label(A2); sz2 = np.bincount(lab2.ravel()); small = (sz2 < 400); small[0] = False
a[small[lab2], 3] = 0; a[a[..., 3] == 0, :3] = 0
Image.fromarray(a, "RGBA").save(sys.argv[2], dpi=im.info.get("dpi", (300, 300)))
print(sys.argv[2], "removed", int(R.sum()))
