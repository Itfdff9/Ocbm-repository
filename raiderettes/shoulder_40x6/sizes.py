"""One print file per shirt size (Gildan 18000 sleeve-to-sleeve chart): S 40x5, M 42x5.5, L 44x6.5, XL+ 46x7.
usage: python3 sizes.py NAME [outdir]"""
import sys, os
sys.path.insert(0, "/home/claude/ocbm-repository/raiderettes/shoulder_40x6")
import build as B
SIZES = [("S", 40, 5.0), ("M", 42, 5.5), ("L", 44, 6.5), ("XL-and-up", 46, 7.0)]
name = sys.argv[1].upper(); outdir = sys.argv[2] if len(sys.argv) > 2 else "out/sizes"
os.makedirs(outdir, exist_ok=True)
orig = B.full_script
for lab, w, h in SIZES:
    B.W, B.H = int(w * B.DPI), int(h * B.DPI)
    B.full_script = lambda txt, tw, th, thick: orig(txt, tw, int(0.567 * B.H), thick, kmax=2.8)
    im = B.build(name, "Raiderettes", True, script_full=True, script_wt=0.06)
    fn = f"{outdir}/{name}_Raiderettes_{lab}_{w}x{str(h).rstrip('0').rstrip('.')}in.png"
    im.save(fn, dpi=(B.DPI, B.DPI)); print(fn, im.size)
