"""Add a white trim around the black script Elite (Brian's blue-glitter NORTH SHORE arch, 2026-10-17).
Everything else stays exactly as uploaded. usage: python3 elite_trim.py SRC.png OUT.png WIDTH_PX
 - Elite = the black script strokes (one connected shape)
 - trim  = every pixel within WIDTH px of Elite, solid opaque white, hard outer edge (3x supersampled
           distance, so the edge is smooth), painted over the blue / the existing knockout gap
 - Elite's own soft edge is laid over the white, so black meets white cleanly; Elite stays black
"""
import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd

src, out, W = sys.argv[1], sys.argv[2], float(sys.argv[3])
im = Image.open(src); dpi = im.info.get('dpi')
a = np.asarray(im.convert('RGBA')).astype(np.float32)
al = a[..., 3]; rgb = a[..., :3]; lum = rgb.mean(-1)
dark = (al > 127) & (rgb.max(-1) < 70)
lab, n = nd.label(dark); sz = np.bincount(lab.ravel()); sz[0] = 0
core = lab == sz.argmax()                                    # the Elite script
ys, xs = np.where(core); P = int(W) + 20
y0, y1, x0, x1 = max(0, ys.min() - P), ys.max() + P + 1, max(0, xs.min() - P), xs.max() + P + 1
sub = core[y0:y1, x0:x1].astype(np.float32)
S = 3                                                         # supersample for a smooth trim edge
big = cv2.resize(sub, (sub.shape[1] * S, sub.shape[0] * S), interpolation=cv2.INTER_LINEAR) > 0.5
d = nd.distance_transform_edt(~big) / S
trim_big = (d <= W).astype(np.float32)
trim = cv2.resize(trim_big, (sub.shape[1], sub.shape[0]), interpolation=cv2.INTER_AREA) >= 0.5
T = np.zeros(core.shape, bool); T[y0:y1, x0:x1] = trim
# Elite coverage on its soft edge (dark, near the core) -> laid over the white
near = nd.binary_dilation(core, iterations=4)
e = np.clip((al / 255.0) * np.clip((150 - lum) / (150 - 7), 0, 1), 0, 1) * near
e[core] = np.maximum(e[core], al[core] / 255.0)
o = a.copy()
zone = T | core
white = np.array([255, 255, 255], np.float32)
o[zone, :3] = e[zone, None] * rgb[zone] + (1 - e[zone, None]) * white
o[zone, 3] = 255
Image.fromarray(np.clip(o + 0.5, 0, 255).astype(np.uint8), 'RGBA').save(out, dpi=dpi) if dpi else \
    Image.fromarray(np.clip(o + 0.5, 0, 255).astype(np.uint8), 'RGBA').save(out)
print(out, 'trim width', W, 'px | trim px', int((T & ~core).sum()))
