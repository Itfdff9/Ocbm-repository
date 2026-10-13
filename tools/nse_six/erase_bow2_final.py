import sys; sys.path.insert(0,'/home/claude/six/tools'); from fill_chunks import *
from PIL import Image
im=np.array(Image.open('src/4-48fac636-image.webp').convert('RGBA')); rgb=np.ascontiguousarray(im[...,:3])
M=np.load('bow2_textmask.npy')
s=rgb.astype(int); blue=(s[...,2]>s[...,0]+60)&(im[...,3]>128)
P=nd.binary_fill_holes(nd.binary_closing(blue,iterations=8))
Md=nd.binary_dilation(M,iterations=12)&P
clean=nd.binary_erosion(P,iterations=8)&~nd.binary_dilation(M,iterations=22)
out=rgb.copy()
for box in [(400,2260,790,2850),(1320,2260,1640,2810)]:
  x0,y0,x1,y1=box
  mm=np.zeros_like(Md); mm[y0-120:y1+60,x0-60:x1+60]=Md[y0-120:y1+60,x0-60:x1+60]
  d=axis_of(clean,(x0-150,y0-200,x1+150,y1+100))
  out=chunk_fill(out,mm,clean,d,chunk=110)
im[...,:3]=out; Image.fromarray(im).save('clean/4_blank.png')
Image.fromarray(out[2150:2950,300:1750]).save('bow2_blank_crop.jpg')
