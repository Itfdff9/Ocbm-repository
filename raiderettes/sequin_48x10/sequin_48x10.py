"""Raiderettes sequin names (black sequin fill, red + grey stitched outline): vertical upload -> turned to read
left to right -> stretched to fill 48 x 10 in @300 DPI (14400 x 3000), Lanczos (premultiplied) + light sharpen,
hard alpha. Picked over plain resize and AI upscale on TRAVA (2026-10-20): AI changed the sequin texture.
usage: python3 sequin_48x10.py SRC OUT.png [PREVIEW.jpg]"""
import sys, numpy as np
from PIL import Image, ImageFilter
Image.MAX_IMAGE_PIXELS = None
TW, TH = 14400, 3000
src, out = sys.argv[1], sys.argv[2]
im = Image.open(src).convert('RGBA')
if im.height > im.width: im = im.transpose(Image.Transpose.ROTATE_270)
a = np.asarray(im).astype(np.float32) / 255.0
ys, xs = np.where(a[..., 3] > 0.5); a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]   # art edge to edge
al = a[..., 3:4]; pre = np.concatenate([a[..., :3] * al, al], -1)
ch = [np.asarray(Image.fromarray(pre[..., k].astype(np.float32), 'F').resize((TW, TH), Image.LANCZOS)) for k in range(4)]
A = np.clip(ch[3], 0, 1); rgb = np.stack(ch[:3], -1)
rgb = np.clip(np.where(A[..., None] > 1e-4, rgb / np.maximum(A[..., None], 1e-4), 0), 0, 1)
img = Image.fromarray((rgb * 255 + 0.5).astype(np.uint8), 'RGB').filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=2))
alpha = np.where(A >= 0.5, 255, 0).astype(np.uint8)
o = np.dstack([np.asarray(img), alpha]); o[alpha == 0, :3] = 0
Image.fromarray(o, 'RGBA').save(out, dpi=(300, 300), optimize=True)
print(out, (TW, TH), 'from art', a.shape[1::-1], 'stretch x%.2f wide, x%.2f tall' % (TW / a.shape[1], TH / a.shape[0]))
if len(sys.argv) > 3:
    r = Image.open(out); sm = r.resize((1800, 375), Image.LANCZOS)
    s = Image.new('RGB', (1800, 375 * 3 + 20), (255, 255, 255))
    for i, bg in enumerate(((0, 0, 0), (128, 128, 128), (255, 255, 255))):
        b = Image.new('RGBA', sm.size, bg + (255,)); b.alpha_composite(sm); s.paste(b.convert('RGB'), (0, i * 385))
    s.save(sys.argv[3], quality=90)
