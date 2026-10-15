# Tiled Real-ESRGAN via ncnn (CPU). upscale_rg.py MODELDIR MODEL SCALE in.png out.png
import sys, numpy as np, ncnn, time
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
def up(img, mdir, model, S, tile=200, pad=12):
    net = ncnn.Net(); net.opt.use_vulkan_compute = False; net.opt.num_threads = 2
    net.load_param(f"{mdir}/{model}.param"); net.load_model(f"{mdir}/{model}.bin")
    a = np.asarray(img.convert('RGB')).astype(np.float32) / 255.
    H, W, _ = a.shape; out = np.zeros((H * S, W * S, 3), np.uint8)
    ap = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    t0 = time.time(); n = 0; tot = ((H + tile - 1) // tile) * ((W + tile - 1) // tile)
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            h = min(tile, H - y); w = min(tile, W - x)
            blk = ap[y:y + h + 2 * pad, x:x + w + 2 * pad].transpose(2, 0, 1).copy()
            ex = net.create_extractor(); ex.input('data', ncnn.Mat(blk))
            _, o = ex.extract('output'); o = np.array(o).transpose(1, 2, 0)
            out[y * S:(y + h) * S, x * S:(x + w) * S] = (np.clip(o[pad * S:(pad + h) * S, pad * S:(pad + w) * S], 0, 1) * 255 + 0.5).astype(np.uint8)
            n += 1; print(n, '/', tot, round(time.time() - t0), 's', flush=True)
    return Image.fromarray(out)
if __name__ == '__main__':
    up(Image.open(sys.argv[4]), sys.argv[1], sys.argv[2], int(sys.argv[3])).save(sys.argv[5]); print('DONE', flush=True)
