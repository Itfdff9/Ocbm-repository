import qrcode, glob
from PIL import Image, ImageDraw, ImageFont
W,H=2550,3300  # 8.5x11 @300dpi
F=lambda n,s: ImageFont.truetype(n,s)
POP_B='/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf'
POP=glob.glob('/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf')[0]
POP_SB='/usr/share/fonts/truetype/google-fonts/Poppins-Medium.ttf'
GRAD='/home/claude/dreams/fonts/ttf/graduate-latin-400-normal.ttf'
def flyer(team, logo, url, bg, accent, text, mock1, mock2, out, logo_w=1100, show_name=True):
    im=Image.new('RGB',(W,H),bg); d=ImageDraw.Draw(im)
    # top band
    d.rectangle([0,0,W,160],fill=accent)
    d.text((W//2,80),"OFFICIAL TEAM STORE",font=F(POP_B,64),fill=bg,anchor="mm")
    # logo
    lg=Image.open(logo).convert('RGBA'); r=logo_w/lg.width; lg=lg.resize((logo_w,int(lg.height*r)),Image.LANCZOS)
    im.paste(lg,((W-lg.width)//2,230),lg); y=230+lg.height+60
    if show_name: d.text((W//2,y),team.upper(),font=F(GRAD,130),fill=text,anchor="mm"); y+=130
    d.text((W//2,y),"HOODIES · CREWNECKS · TEES",font=F(POP_SB,70),fill=accent,anchor="mm"); y+=120
    # mockups
    def sq(p):
        m=Image.open(p).convert('RGB'); w,h=m.size; s_=min(w,h); return m.crop(((w-s_)//2,(h-s_)//2,(w-s_)//2+s_,(h-s_)//2+s_)).resize((820,820),Image.LANCZOS)
    im.paste(sq(mock1),(W//2-850,y)); im.paste(sq(mock2),(W//2+30,y)); y+=870
    # bullet lines
    lines=["Player's last name + number on the back — FREE","One flat price, every size S–5XL","Hoodie $40  ·  Crewneck $35  ·  Tee $25","Printed in-house. Pick-up or ship."]
    for l in lines:
        d.text((W//2,y),l,font=F(POP,58),fill=text,anchor="mm"); y+=85
    y+=40
    # QR
    q=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_H,border=2); q.add_data(url); q.make(fit=True)
    qi=q.make_image(fill_color=text,back_color=bg).convert('RGB').resize((560,560),Image.NEAREST)
    im.paste(qi,(W//2-280,y))
    d.text((W//2,y+600),"SCAN TO SHOP",font=F(POP_B,70),fill=accent,anchor="mm")
    d.text((W//2,y+670),url.replace('https://',''),font=F(POP,48),fill=text,anchor="mm")
    # footer
    d.rectangle([0,H-150,W,H],fill=accent)
    d.text((W//2,H-75),"One Crafty Boy Mama  ·  @onecraftyboymama  ·  Marissa (516) 265-4399  ·  Brian (631) 830-9674",font=F(POP,44),fill=bg,anchor="mm")
    im.save(out,quality=95,dpi=(300,300)); print(out)
R='/home/claude/ocbm-repository/'
flyer("Bethpage United FC", R+'bufc/logo/BUFC_crest_transparent.png', "https://onecraftyboymama.com/collections/bethpage-united-fc", (255,255,255),(14,28,70),(14,28,70), R+'bufc/g18500/G18500_Black_Front_BUFC.jpg', R+'bufc/g18500/G18500_Ash_Back_SAVVA_18_opt1.jpg', 'BUFC_Team_Store_Flyer.jpg', 640)
flyer("Bethpage Bluebirds FC", R+'bluebirds/logo/BLUEBIRDS_crest_transparent.png', "https://onecraftyboymama.com/collections/bethpage-bluebirds-fc", (255,255,255),(42,93,179),(20,40,110), R+'bluebirds/g18500/G18500_Ash_Front_BLUEBIRDS.jpg', R+'bluebirds/g18500/G18500_Black_Back_COOK_6.jpg', 'Bluebirds_Team_Store_Flyer.jpg', 820)
flyer("DREAMS Fastpitch", R+'dreams/logo/DREAMS_PARENT_cleaned_300dpi.png', "https://onecraftyboymama.com/collections/dreams-fastpitch", (255,255,255),(22,72,158),(12,12,12), R+'dreams/g18000/G18000_Royal_Front_DREAMS.jpg', R+'dreams/g18500/G18500_Royal_Back_MILDENBERGER_8.jpg', 'DREAMS_Team_Store_Flyer.jpg', 1300, False)
