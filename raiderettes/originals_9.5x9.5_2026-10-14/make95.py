"""Raiderettes curved names -> 9.5 x 9.5 in @ 300 DPI (2850 x 2850), design full 9.5 in wide, proportional, centred.
Exact match of the upload: only resampling + hard alpha (0/255). usage: python3 make95.py NAME SRC OUTDIR"""
import sys, numpy as np
from PIL import Image
from scipy import ndimage as nd
name, src, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
CAN = 2850
im = Image.open(src).convert('RGBA')
al = np.asarray(im)[..., 3]
ys, xs = np.where(al >= 128)
M = 8
box = (max(0, xs.min() - M), max(0, ys.min() - M), min(im.width, xs.max() + 1 + M), min(im.height, ys.max() + 1 + M))
crop = im.crop(box).convert('RGBa')                       # premultiplied -> no dark/colour halo from resampling
w0 = xs.max() + 1 - xs.min()
s = CAN / w0
for it in range(6):
    W, H = round(crop.width * s), round(crop.height * s)
    r = np.asarray(crop.resize((W, H), Image.LANCZOS).convert('RGBA')).copy()
    op = r[..., 3] >= 128
    yy, xx = np.where(op)
    bw = xx.max() + 1 - xx.min()
    if bw <= CAN: break
    s *= (CAN - 0.5) / bw
# tiny detached specks created by the threshold (not part of the art)
lab, n = nd.label(op); sz = nd.sum(op, lab, range(1, n + 1))
specks = np.isin(lab, 1 + np.where(sz < 30)[0]); op &= ~specks
r[..., 3] = np.where(op, 255, 0); r[~op, :3] = 0
yy, xx = np.where(op)
art = r[yy.min():yy.max() + 1, xx.min():xx.max() + 1]
ah, aw = art.shape[:2]
canvas = np.zeros((CAN, CAN, 4), np.uint8)
oy, ox = (CAN - ah) // 2, (CAN - aw) // 2
canvas[oy:oy + ah, ox:ox + aw] = art
out = f'{outdir}/{name}_Raiderettes_9.5x9.5in_300dpi.png'
Image.fromarray(canvas).save(out, dpi=(300, 300), optimize=True)
print(f'{name:9s} src {im.width}x{im.height} -> art {aw}x{ah}px = {aw/300:.2f} x {ah/300:.2f} in | specks removed {int(specks.sum())} | alpha {np.unique(canvas[...,3]).tolist()} | islands {n - int((sz<30).sum())}')
