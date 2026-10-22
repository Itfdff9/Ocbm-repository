"""Literally just a cheerleader — DTF print prep.
usage: python3 build_final.py SRC MASK.npy UPSCALED.png OUT.png LONG_PX [lanczos]
 - background = the fake checkerboard (bg mask: edge flood fill + checker-cell detection + checker pockets,
   letter tiles protected) -> fully transparent; everything else fully opaque (hard alpha 0/255)
 - colour: AI-upscaled art (or Lanczos from the source if 'lanczos'), edge pixels re-filled from just inside
   so no light checker fringe survives on the cut line
"""
import sys, numpy as np, cv2
from PIL import Image, ImageFilter
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS = None
src, mpath, upath, out, LONG = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
mode = sys.argv[6] if len(sys.argv) > 6 else 'ai'
s = Image.open(src).convert('RGB'); W, H = s.size
fg = ~np.load(mpath)
k = LONG / max(W, H); TW, TH = round(W * k), round(H * k)
# hard alpha: soft edge only for resampling, then threshold
m = cv2.GaussianBlur(fg.astype(np.float32), (0, 0), 0.7)
A = cv2.resize(m, (TW, TH), interpolation=cv2.INTER_CUBIC) >= 0.5
# colour: bleed the art outward over the background before resampling (so the cut edge is art colour)
if mode == 'lanczos':
    rgb = np.asarray(s).copy()
    iy, ix = nd.distance_transform_edt(~nd.binary_erosion(fg, iterations=1), return_distances=False, return_indices=True)
    rgb = rgb[iy, ix]
    im = Image.fromarray(rgb).resize((TW, TH), Image.LANCZOS).filter(ImageFilter.UnsharpMask(round(1.2 * k / 1.64, 1), 60, 2))
    C = np.asarray(im)
else:
    U = Image.open(upath).convert('RGB')
    C = np.asarray(U.resize((TW, TH), Image.LANCZOS))
    # re-fill a thin band at the cut line from pixels just inside (AI saw the checker there)
    band = max(2, round(1.0 * TW / W))
    core = nd.binary_erosion(A, iterations=band)
    iy, ix = nd.distance_transform_edt(~core, return_distances=False, return_indices=True)
    fix = A & ~core
    C = C.copy(); C[fix] = C[iy[fix], ix[fix]]
    del iy, ix
O = np.dstack([C, (A * 255).astype(np.uint8)])
O[~A, :3] = 0
INCH = 14.0                                   # long side in inches (14 x ~12.1 in)
d = round(LONG / INCH)                        # 300 for the print file; ~585 for the 8K master (same inches)
Image.fromarray(O, 'RGBA').save(out, dpi=(d, d), compress_level=6)
print(out, (TW, TH), 'alpha values', np.unique(O[..., 3]).tolist(), 'opaque %.1f%%' % (A.mean() * 100))
