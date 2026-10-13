"""Collar centre by mirror symmetry of the neck/collar region."""
import numpy as np, cv2
def sym_cx(img, y0=300, y1=650, xc=1000, search=200, half=230):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), 2)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    band_gy = np.abs(gy[y0:y1]); band_gx = gx[y0:y1]
    best = (-1e18, xc)
    for c in range(xc - search, xc + search + 1, 2):
        L = band_gy[:, c - half:c]; Rr = band_gy[:, c + 1:c + half + 1][:, ::-1]
        Lx = band_gx[:, c - half:c]; Rx = -band_gx[:, c + 1:c + half + 1][:, ::-1]   # x-gradient flips sign in a mirror
        if L.shape != Rr.shape: continue
        s = (L * Rr).sum() + (Lx * Rx).sum()
        n = np.sqrt(((L ** 2).sum() + (Lx ** 2).sum()) * ((Rr ** 2).sum() + (Rx ** 2).sum())) + 1e-6
        sc = s / n
        if sc > best[0]: best = (sc, c)
    return best[1], best[0]
