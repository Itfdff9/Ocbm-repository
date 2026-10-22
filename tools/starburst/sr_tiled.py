import cv2, numpy as np, sys
src, dst = sys.argv[1], sys.argv[2]
T, O, S = 256, 16, 4
sr = cv2.dnn_superres.DnnSuperResImpl_create(); sr.readModel('/tmp/claude-0/-home-claude/f4680fbe-84fa-5212-8106-a18cccebeb09/scratchpad/models/LapSRN_x4.pb'); sr.setModel('lapsrn', S)
img = cv2.imread(src); H, W = img.shape[:2]
out = np.zeros((H * S, W * S, 3), np.float32); wsum = np.zeros((H * S, W * S, 1), np.float32)
ramp = lambda n: np.minimum(np.minimum(np.arange(n) + 1, n - np.arange(n)) / (O * S), 1.0).astype(np.float32)
for y in range(0, H, T):
    for x in range(0, W, T):
        y0, x0 = max(0, y - O), max(0, x - O); y1, x1 = min(H, y + T + O), min(W, x + T + O)
        t = sr.upsample(img[y0:y1, x0:x1]).astype(np.float32)
        wy = ramp(t.shape[0]); wx = ramp(t.shape[1])
        if y0 == 0: wy[:O * S] = 1
        if x0 == 0: wx[:O * S] = 1
        if y1 == H: wy[-O * S:] = 1
        if x1 == W: wx[-O * S:] = 1
        w = (wy[:, None] * wx[None, :])[..., None]
        out[y0 * S:y1 * S, x0 * S:x1 * S] += t * w; wsum[y0 * S:y1 * S, x0 * S:x1 * S] += w
cv2.imwrite(dst, np.clip(out / np.maximum(wsum, 1e-6), 0, 255).round().astype(np.uint8))
print(out.shape)
