import sys, numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS = None
R = "/home/claude/ocbm-repository/north_shore_elite/"
def down(master, N, dpi, out):
    a = np.array(master); A = a[..., 3] >= 128
    _, (iy, ix) = nd.distance_transform_edt(~A, return_indices=True)
    rgb = a[..., :3][iy, ix]
    r = cv2.resize(rgb, (N, N), interpolation=cv2.INTER_AREA if N < a.shape[0] else cv2.INTER_LANCZOS4)
    al = cv2.resize(a[..., 3], (N, N), interpolation=cv2.INTER_AREA if N < a.shape[0] else cv2.INTER_LANCZOS4)
    o = np.dstack([r, al]); s = o[..., 3] >= 128; o[..., 3] = np.where(s, 255, 0); o[~s, :3] = 0
    Image.fromarray(o, "RGBA").save(out, dpi=(dpi, dpi)); print(out, N)
def patch_small(orig_master, new_master, target, out=None):
    o = np.array(orig_master).astype(int); n = np.array(new_master)
    ch = np.abs(o - n.astype(int)).max(2) > 0
    t = Image.open(target); mode = t.mode; T = np.array(t.convert("RGBA")); N = T.shape[0]
    k = N / o.shape[0]
    ys, xs = np.where(ch); y0, y1, x0, x1 = ys.min() - 8, ys.max() + 9, xs.min() - 8, xs.max() + 9
    Y0, Y1, X0, X1 = int(y0 * k), int(np.ceil(y1 * k)), int(x0 * k), int(np.ceil(x1 * k))
    crop = cv2.resize(n[int(Y0 / k):int(Y1 / k), int(X0 / k):int(X1 / k), :3], (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_AREA)
    m = cv2.resize(ch[int(Y0 / k):int(Y1 / k), int(X0 / k):int(X1 / k)].astype(np.float32), (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur((cv2.dilate((m > 0.01).astype(np.uint8), np.ones((3, 3))) > 0).astype(np.float32), (0, 0), 0.8)[..., None]
    T[Y0:Y1, X0:X1, :3] = (T[Y0:Y1, X0:X1, :3] * (1 - m) + crop * m).astype(np.uint8)
    Image.fromarray(T, "RGBA").convert(mode).save(out or target, dpi=t.info.get("dpi", (300, 300))); print("patched", target)
