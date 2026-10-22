"""Starburst: remove the black backdrop, keep the NS letter fills. Writes RGBA variants at 4x scale."""
import numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
a = np.asarray(Image.open('crop_x4.png').convert('RGB'))
mx = a.max(-1).astype(np.int16)
T, r = 45, 9
dark = mx < T; ink = ~dark
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
free = dark & ~(cv2.dilate(ink.astype(np.uint8), k) > 0)
lab, n = nd.label(free)
LETTER_FILLS = [262, 684, 433, 498, 290, 520, 528, 697]
fills = np.isin(lab, LETTER_FILLS)
k2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (r + 3) + 1, 2 * (r + 3) + 1))
L = cv2.dilate(fills.astype(np.uint8), k2) > 0
protect = dark & L
# tiny dark holes inside ink (JPEG noise) are filled; larger pockets are backdrop
holes = nd.binary_fill_holes(ink) & dark & ~protect
hl, hn = nd.label(holes); hs = nd.sum(holes, hl, range(1, hn + 1))
tiny = np.isin(hl, 1 + np.where(hs < 150)[0])
keep = ink | protect | tiny
# drop isolated ink specks that are too small to print (< 40 px at 4x ~ 2.6 px at 300 dpi)
kl, kn = nd.label(keep); ks = nd.sum(keep, kl, range(1, kn + 1))
keep &= ~np.isin(kl, 1 + np.where(ks < 40)[0])
np.save('keep_B.npy', keep)
def save(mask, rgb, name):
    out = np.dstack([rgb, np.where(mask, 255, 0).astype(np.uint8)])
    out[~mask, :3] = 0
    Image.fromarray(out).save(name)
save(keep, a, 'var_B.png')
# B+: thin black keyline = the original black within d px of the kept art
d = 10
dist = nd.distance_transform_edt(~keep)
kl_mask = keep | (dark & (dist <= d))
save(kl_mask, a, 'var_Bkey.png')
# C: black backing inside a smoothed silhouette of the art
R = 40
kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * R + 1, 2 * R + 1))
sil = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, kc) > 0
sil = nd.binary_fill_holes(sil)
save(sil | keep, a, 'var_C.png')
print('kept px', keep.sum(), 'protected', protect.sum(), 'tiny holes', tiny.sum(), 'keyline add', (kl_mask & ~keep).sum(), 'backing add', (sil & ~keep).sum())
