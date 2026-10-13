# bleed.py in_rgba out_rgb.png out_alpha.png : fill transparent area with nearest solid colour (no halos on upscale)
import sys, numpy as np
from PIL import Image
from scipy import ndimage as nd
Image.MAX_IMAGE_PIXELS=None
im=np.array(Image.open(sys.argv[1]).convert('RGBA')); a=im[...,3]
solid=a>200
_,(iy,ix)=nd.distance_transform_edt(~solid,return_indices=True)
Image.fromarray(im[...,:3][iy,ix]).save(sys.argv[2]); Image.fromarray(a).save(sys.argv[3])
