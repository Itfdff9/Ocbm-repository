"""Clean the North Shore Elite arch design into 4 flat classes (clear, black, white, blue) as soft coverage fields.
Saves cov.npy (H,W,4 float32: clear, black, white, blue) at source resolution, padded with a margin."""
import numpy as np
from PIL import Image
from scipy import ndimage as nd

PAD = 40
a = np.asarray(Image.open('src.png').convert('RGBA')).astype(np.float32)
a = np.pad(a, ((PAD, PAD), (PAD, PAD), (0, 0)))
al = a[..., 3] / 255.0
rgb = a[..., :3]
r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
lum = rgb.mean(-1)
sat = rgb.max(-1) - rgb.min(-1)

PAL = np.array([[0, 0, 0], [255, 255, 255], [0, 40, 160]], np.float32)  # black, white, blue
d2 = ((rgb[..., None, :] - PAL[None, None]) ** 2).sum(-1)
w = np.exp(-(d2 - d2.min(-1, keepdims=True)) / (2 * 45.0 ** 2))
w /= w.sum(-1, keepdims=True)

bluecls = (al > 0.5) & (b > r + 40) & (b > g + 30)
blue_core = nd.binary_opening(bluecls, iterations=2)
dist_blue = nd.distance_transform_edt(~blue_core)

# debris: warm/grey off-palette pixels, and "white" that is not part of Elite's outline (too far from blue)
offpal = (al > 0.3) & (lum > 85) & (lum < 238) & ~bluecls & (sat > 12)
greyish = (al > 0.3) & (lum > 85) & (lum < 215) & ~bluecls
white_far = (al > 0.3) & (lum >= 215) & (dist_blue > 26)
debris = offpal | white_far | (greyish & (dist_blue > 26))

# Elite = blue + white outline (white within 26 px of blue, not debris)
elite = bluecls | ((al > 0.5) & (lum >= 200) & (dist_blue <= 26) & ~offpal)
elite = nd.binary_closing(elite, iterations=1)
# pockets = areas fully enclosed by Elite
filled = nd.binary_fill_holes(elite)
pockets = filled & ~elite
lab, n = nd.label(pockets)

H, W = al.shape
cov = np.zeros((H, W, 4), np.float32)
cov[..., 0] = 1 - al
cov[..., 1:] = w * al[..., None]

def box_hit(sl, box):
    x0, y0, x1, y1 = [v + PAD for v in box]
    return not (sl[1].stop < x0 or sl[1].start > x1 or sl[0].stop < y0 or sl[0].start > y1)

FORCE_BLACK = [(2165, 615, 2345, 765), (2375, 585, 2610, 765)]  # i-t and t-e pockets (S body behind)
objs = nd.find_objects(lab)
forced = np.zeros_like(pockets)
for i, sl in enumerate(objs):
    if sl is None: continue
    if any(box_hit(sl, bx) for bx in FORCE_BLACK):
        m = lab == i + 1
        # only small pockets (ignore anything huge)
        if m.sum() < 60000:
            forced |= m
forced = nd.binary_dilation(forced, iterations=2) & ~elite
cov[forced] = [0, 1, 0, 0]

# remaining debris (outside forced pockets): replace by nearest legit black/clear pixel
legit = ((al > 0.9) & (lum < 60)) | (al < 0.1)
fix = (debris & ~forced) | (debris & ~elite)
fix = nd.binary_dilation(fix, iterations=2) & ~elite & ~forced
_, (iy, ix) = nd.distance_transform_edt(~legit, return_indices=True)
src_black = (al[iy, ix] > 0.5)
cov[fix] = 0
cov[fix & src_black, 1] = 1
cov[fix & ~src_black, 0] = 1

np.save('cov.npy', cov)
print('pockets', n, 'forced px', int(forced.sum()), 'debris fixed px', int(fix.sum()), 'shape', cov.shape)

# quick preview at source res: argmax with smoothing, on magenta
c = nd.gaussian_filter(cov, sigma=(0.8, 0.8, 0))
labm = c.argmax(-1)
COL = np.array([[255, 0, 255], [0, 0, 0], [255, 255, 255], [0, 40, 160]], np.uint8)
Image.fromarray(COL[labm]).save('clean_preview_magenta.png')
