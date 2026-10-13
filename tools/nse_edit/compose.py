import sys, numpy as np
sys.path.insert(0,'/home/claude/nse_edit/tools'); import name
from PIL import Image
def compose(design, N, scale=0.88, name_text='Khloe', name_h=0.105, name_cx=0.47, name_cy=0.865, angle=-5, sparkle_pts=None):
    """design: RGBA square art; N: output size. returns RGBA NxN"""
    D=int(N*scale)
    d=design.resize((D,D),Image.LANCZOS)
    canvas=Image.new('RGBA',(N,N),(0,0,0,0))
    canvas.alpha_composite(d,((N-D)//2,int(N*0.005)))
    nm=name.render(name_text,'/home/claude/nse_edit/fonts/Pacifico-Regular.ttf',int(N*name_h),angle=angle,gaps={1:10})
    nm=add_sp(nm,sparkle_pts or [],N)
    x=int(N*name_cx-nm.size[0]/2); y=int(N*name_cy-nm.size[1]/2)
    canvas.alpha_composite(nm,(x,y))
    return canvas
def add_sp(nm,pts,N):
    W,H=nm.size
    return name.add_sparkles(nm,[(px*W,py*H,sc) for px,py,sc in pts],int(N*0.03))
