import numpy as np, cv2
from PIL import Image
from scipy import ndimage as nd
out = np.load("clean3.npy"); el = np.load("elite_layer.npy")
denv = nd.distance_transform_edt(~el)
near = (denv > 0) & (denv < 260) & ~el
o = out[..., :3].astype(int); op = out[..., 3] == 255
LB = np.array([0, 65, 221])
# faint dark scratch lines inside letter blue
blu = op & (o[..., 2] - o[..., 0] > 40)
frac = cv2.blur(blu.astype(np.float32), (25, 25))
line = op & near & (frac > 0.6) & ~((o.min(2) > 175)) & ((o[..., 2] < 200) | (o.max(2) < 90)) & (o[..., 2] - o[..., 0] > 15)
out[line] = (*LB, 255); print("lines", int(line.sum()))
al = out[..., 3] >= 128; out[..., 3] = np.where(al, 255, 0); out[~al, :3] = 0
np.save("clean4.npy", out); Image.fromarray(out, "RGBA").save("clean4.png")
