"""S (SHORE) bottom terminal beside Elite: redraw Brian's trim geometry (black inside white bracket,
transparent channel, inner white bar) where the old blue bleed had smeared it. Elite outline stays on top."""
import numpy as np
from PIL import Image
out = np.load("clean2.npy"); el = np.load("elite_layer.npy")
H, W = el.shape; yy, xx = np.mgrid[0:H, 0:W]
was_tr = np.load("clean2.npy")[..., 3] < 128
reg = (xx >= 4100) & (xx < 4437) & (yy >= 532) & (yy < 650) & ~el
reg &= ~(was_tr & (yy < 547))            # leave the open background left of the S diagonal
out[reg] = (8, 8, 10, 255)
ch = ((xx < 4397) & (yy >= 547) & (yy < 570)) | ((xx >= 4350) & (xx < 4397) & (yy >= 570) & (yy < 637))
out[reg & ch] = 0
bar = (xx >= 4100) & (xx < 4310) & (yy >= 582) & (yy < 620)
out[reg & bar] = (255, 255, 255, 255)
# continue the bracket's bottom white trim left until it meets the Elite outline
bot = (xx >= 4150) & (xx < 4330) & (yy >= 650) & (yy < 673) & ~el
out[bot] = (255, 255, 255, 255)
np.save("clean3.npy", out); Image.fromarray(out, "RGBA").save("clean3.png")
