import sys, time, os, numpy as np, cv2; sys.path.insert(0,'/home/claude/rd'); from engine import *
from scipy import ndimage as nd
OUT='/home/claude/rd/out'
def bleed(img, A):
    _, (iy, ix) = nd.distance_transform_edt(~A, return_indices=True)
    return img[iy, ix]
def save(img, A, path, dpi):
    to_rgba(img, A).save(path, dpi=(dpi, dpi)); print(path, A.shape[::-1], dpi, flush=True)
def curved(name, border=None, tag='', seed=None):
    W = 8192
    img, A = compose(name, W, int(W * 0.62), 'curved', letter_h_frac=0.80, script_h_frac=0.45, script_w_frac=0.90, script_y_frac=0.80,
                     widen_max=1.15, script_fit=(0.86, 0.60), kmin=0.48, kmax=0.66, margin_frac=0.01, gap_frac=-0.05, seed=seed or (sum(map(ord,name)) % 97 + 3), border=border)
    img, A = trim(img, A, 4); img, A = arch(img, A, rise=int(W * 0.13)); img, A = trim(img, A, 0)
    N = 8192; s = min(N * 0.985 / A.shape[1], N * 0.66 / A.shape[0]); tw = int(A.shape[1] * s)
    im2 = cv2.resize(bleed(img, A), (tw, int(A.shape[0] * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    a2 = cv2.resize(A.astype(np.float32), (tw, int(A.shape[0] * s)), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR) >= 0.5
    c = np.zeros((N, N, 3), np.float32); ca = np.zeros((N, N), bool)
    y0 = (N - a2.shape[0]) // 2; x0 = (N - tw) // 2
    c[y0:y0 + a2.shape[0], x0:x0 + tw] = im2; ca[y0:y0 + a2.shape[0], x0:x0 + tw] = a2
    save(c, ca, f'{OUT}/{name}_Raiderettes_CURVED{tag}_12x12in_8K.png', N / 12)
    cb = bleed(c, ca)
    s3 = cv2.resize(cb, (3600, 3600), interpolation=cv2.INTER_AREA); a3 = cv2.resize(ca.astype(np.float32), (3600, 3600), interpolation=cv2.INTER_AREA) >= 0.5
    save(s3, a3, f'{OUT}/{name}_Raiderettes_CURVED{tag}_12x12in_300dpi.png', 300)
def straight(name, border=None, tag='', seed=None):
    W, H = 14400, 3000
    img, A = compose(name, W, H, 'straight', letter_h_frac=0.90, script_h_frac=0.62, script_w_frac=0.80, script_y_frac=0.58,
                     kmin=0.45, kmax=1.6, margin_frac=0.004, gap_frac=0.03, seed=seed or (sum(map(ord,name)) % 89 + 5), widen_max=1.9, border=border)
    save(img, A, f'{OUT}/{name}_Raiderettes_STRAIGHT{tag}_48x10in_300dpi.png', 300)
if __name__ == '__main__':
    kind = sys.argv[1]
    for n in sys.argv[2:]:
        t = time.time(); (curved if kind == 'curved' else straight)(n); print(n, kind, round(time.time() - t), 's', flush=True)
