"""H (NORTH) counter bottom beside Elite: keep Brian's narrow drip (like the T gap) as one clean straight slit
between solid black edges down to the Elite outline; remove the ragged notches/flecks."""
import numpy as np
from PIL import Image
out = np.load("clean4.npy"); el = np.load("elite_layer.npy")
H, W = el.shape; yy, xx = np.mgrid[0:H, 0:W]
o = out[..., :3].astype(int)
white = (o.min(2) > 175) & (out[..., 3] == 255)
blue = (o[..., 2] - o[..., 0] > 40) & (o.min(2) <= 175) & (out[..., 3] == 255)
box = (xx >= 2806) & (xx < 2990) & (yy >= 640) & (yy < 770) & ~el
m = box & ~white & ~blue
out[m] = (8, 8, 10, 255)
# stray grey/white flecks inside that black
fl = box & white & (xx > 2850) & (xx < 2975) & (yy < 700)
out[fl] = (8, 8, 10, 255)
# left of the drip: letter base blue (as under the T), straight black stub edges
lb = (xx >= 2806) & (xx < 2866) & (yy >= 660) & (yy < 800) & ~el & ~white
out[lb] = (0, 65, 221, 255)
stub = (xx >= 2866) & (xx < 2976) & (yy >= 648) & (yy < 800) & ~el
out[stub] = (8, 8, 10, 255)
slit = (xx >= 2904) & (xx < 2936) & (yy >= 648) & (yy < 800) & ~el
out[slit] = 0
np.save("clean5.npy", out); Image.fromarray(out, "RGBA").save("clean5.png")
