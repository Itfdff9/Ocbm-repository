import numpy as np
from PIL import Image
from scipy import ndimage as nd
im=np.array(Image.open('std.png').convert('RGBA'))
a=im[...,3]; rgb=im[...,:3]
solid=a>200
# bleed: transparent/edge pixels take nearest solid color so upscaling doesn't create halos
_,(iy,ix)=nd.distance_transform_edt(~solid,return_indices=True)
bled=rgb[iy,ix]
Image.fromarray(bled).save('std_bled_rgb.png')
Image.fromarray(a).save('std_alpha.png')
