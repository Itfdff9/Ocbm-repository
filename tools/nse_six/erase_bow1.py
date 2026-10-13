import sys; sys.path.insert(0,'/home/claude/six/tools'); from erase import *
from PIL import Image
im=np.array(Image.open('clean/1_full.png')); rgb=np.ascontiguousarray(im[...,:3])
boxes=[(640,1965,1255,2165),(615,2370,1280,2605),(655,2820,1255,3080)]
M=np.zeros(rgb.shape[:2],bool)
for b in boxes: M|=text_mask(rgb,b)
Md=nd.binary_dilation(M,iterations=15)
out=rgb.copy()
for b in boxes:
  x0,y0,x1,y1=b; reg=(x0-60,y0-150,x1+60,y1+150)
  mm=np.zeros_like(Md); mm[y0-20:y1+20,x0-20:x1+20]=Md[y0-20:y1+20,x0-20:x1+20]
  cl=clean_glitter(out,reg,nd.binary_dilation(M,iterations=28))
  out=fill(out,mm,cl)
im[...,:3]=out; Image.fromarray(im).save('clean/1_blank.png')
Image.fromarray(out[1900:3150,500:1400]).save('bow1_blank_crop.jpg')
