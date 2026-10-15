"""Helpers for editing the sequin/script designs: polygon masks, spline paths, stroke texture mapping."""
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy import ndimage as nd

def load(path):
    return np.asarray(Image.open(path).convert('RGBA')).astype(np.float32)

def save(arr, path):
    Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8)).save(path)

def poly_mask(shape, pts):
    im = Image.new('L', (shape[1], shape[0]), 0)
    ImageDraw.Draw(im).polygon([tuple(map(float, p)) for p in pts], fill=255)
    return np.asarray(im) > 127

def catmull_rom(pts, step=0.5):
    P = np.asarray(pts, float)
    P = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        L = np.linalg.norm(p2 - p1); n = max(2, int(L / step))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-2][None])
    return np.vstack(out)

def path_frame(path):
    d = np.gradient(path, axis=0)
    d = nd.gaussian_filter1d(d, 4, axis=0)
    t = d / np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-t[:, 1], t[:, 0]], 1)          # image coords: walking right -> normal points down
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
    return s, t, nrm

def stroke_coords(path, bbox):
    """For every pixel in bbox=(x0,y0,x1,y1): arc length s, signed normal offset n, and the along-path overshoot."""
    x0, y0, x1, y1 = bbox
    s, t, nrm = path_frame(path)
    tree = cKDTree(path)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    P = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    _, idx = tree.query(P)
    v = P - path[idx]
    along = (v * t[idx]).sum(1)
    n = (v * nrm[idx]).sum(1)
    S = s[idx] + along
    return S.reshape(yy.shape), n.reshape(yy.shape), s[-1]

def sample(img, xs, ys):
    """bilinear sample of a float RGBA image at float coords (premultiplied to avoid fringe)."""
    pm = img.copy(); pm[..., :3] *= pm[..., 3:4] / 255.0
    out = np.stack([cv2.remap(pm[..., c], xs.astype(np.float32), ys.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) for c in range(4)], -1)
    a = out[..., 3:4]
    out[..., :3] = np.where(a > 1e-3, out[..., :3] * 255.0 / np.maximum(a, 1e-3), 0)
    return out

def tri(s, L):
    """mirror tiling of arc length into [0, L]."""
    m = np.mod(s, 2 * L)
    return np.where(m <= L, m, 2 * L - m)

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)
