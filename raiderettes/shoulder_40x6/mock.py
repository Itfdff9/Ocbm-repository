"""Approximate adult-L crewneck back, laid flat with arms straight out, 30 px per inch. Places a 40 x 6 in print."""
import sys
from PIL import Image, ImageDraw, ImageFilter
PPI = 30
def crew(color):
    W, H = int(70 * PPI), int(34 * PPI)
    im = Image.new("RGB", (W, H), (236, 234, 229)); d = ImageDraw.Draw(im)
    cx = W // 2; top = int(2 * PPI)
    bw, bl = 24 * PPI, 28 * PPI               # body width / length
    x0, x1 = cx - bw // 2, cx + bw // 2
    shade = tuple(max(0, int(c * 0.86)) for c in color)
    # sleeves: from shoulder (x0, top+1in) out 22in, slight droop
    for side in (-1, 1):
        sx = x0 if side < 0 else x1
        ex = sx + side * 22 * PPI
        poly = [(sx, top + 1 * PPI), (ex, top + int(2.2 * PPI)), (ex, top + int(7.0 * PPI)), (sx, top + int(10.5 * PPI))]
        d.polygon(poly, fill=color)
        cuff = [(ex - side * int(2.5 * PPI), top + int(2.05 * PPI)), (ex, top + int(2.2 * PPI)), (ex, top + int(7.0 * PPI)), (ex - side * int(2.5 * PPI), top + int(7.2 * PPI))]
        d.polygon(cuff, fill=shade)
    d.rounded_rectangle([x0, top, x1, top + bl], radius=int(1.2 * PPI), fill=color)
    d.rectangle([x0, top + bl - int(2.5 * PPI), x1, top + bl], fill=shade)                 # waistband
    d.chord([cx - int(4 * PPI), top - int(1.4 * PPI), cx + int(4 * PPI), top + int(1.1 * PPI)], 0, 180, fill=shade)  # back collar rib
    return im, (cx, top)
def place(print_png, color, out):
    im, (cx, top) = crew(color)
    p = Image.open(print_png).convert("RGBA")
    pw, ph = 40 * PPI, 6 * PPI
    p = p.resize((pw, ph), Image.LANCZOS)
    im = im.convert("RGBA"); im.alpha_composite(p, (cx - pw // 2, top + int(1.6 * PPI)))
    im.convert("RGB").save(out, quality=88)
if __name__ == "__main__":
    place(sys.argv[1], tuple(int(x) for x in sys.argv[2].split(",")), sys.argv[3])
