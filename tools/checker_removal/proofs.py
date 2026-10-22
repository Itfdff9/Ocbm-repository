import sys, numpy as np
from PIL import Image, ImageDraw, ImageFont
Image.MAX_IMAGE_PIXELS=None
p=sys.argv[1]; out=sys.argv[2]
im=Image.open(p); W,H=im.size
sm=im.resize((W*900//W, H*900//W), Image.LANCZOS)
def on(bgc):
    c=Image.new("RGBA",sm.size,bgc+(255,)); c.alpha_composite(sm); return c.convert("RGB")
pad=30
canvas=Image.new("RGB",(sm.width*3+pad*4, sm.height+pad*2+40),(235,235,235))
d=ImageDraw.Draw(canvas)
try: f=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",26)
except: f=ImageFont.load_default()
# checker shows real transparency
ck=Image.new("RGB",sm.size,(255,255,255)); dd=ImageDraw.Draw(ck)
for y in range(0,sm.height,20):
    for x in range(0,sm.width,20):
        if (x//20+y//20)%2: dd.rectangle([x,y,x+19,y+19],fill=(205,205,205))
ckc=ck.convert("RGBA"); ckc.alpha_composite(sm)
for i,(lab,img) in enumerate((("On black shirt",on((0,0,0))),("On white shirt",on((255,255,255))),("Clear areas (checker = see-through)",ckc.convert("RGB")))):
    x=pad+i*(sm.width+pad); canvas.paste(img,(x,pad+40)); d.text((x,pad),lab,fill=(30,30,30),font=f)
canvas.save(out,quality=90)
