import os, sys, glob, re, json, hashlib
import numpy as np
from PIL import Image, ImageFilter
from detect import analyse

LOGOS = {
 "DREAMS": "/home/claude/ocbm-repository/dreams/logo/DREAMS_PARENT_cleaned_300dpi.png",
 "BUFC": "/home/claude/ocbm-repository/bufc/logo/BUFC_crest_transparent.png",
 "BLUEBIRDS": "/home/claude/ocbm-repository/bluebirds/logo/BLUEBIRDS_crest_transparent.png",
}
# set -> (src dir, garment code, pose, (cx, cy, width) in px of the source canvas)
SETS = {
 "g5000_selfie":   ("/home/claude/mockups/g5000_selfie",      "G5000",  "Selfie",   (1010, 790, 640)),
 "bc3001_a":       ("/home/claude/mockups/bc3001/set_a",      "BC3001", "Seated",   (1000, 690, 520)),
 "bc3001_b":       ("/home/claude/mockups/bc3001/set_b",      "BC3001", "Couch",    (1000, 640, 540)),
 "cc1717_halloween":("/home/claude/mockups/cc1717_halloween", "CC1717", "Fall",     (1000, 640, 580)),
 "cc1717_flatlay": ("/home/claude/mockups/cc1717_flatlay",    "CC1717", "Flatlay",  (1024, 780, 720)),
}
MAXW = {"DREAMS": 1.0, "BUFC": 0.78, "BLUEBIRDS": 0.86}  # crests are taller -> slightly narrower than text logo

def clean_name(fn):
    n = os.path.splitext(os.path.basename(fn))[0]
    n = re.sub(r"\s*\(\d+\)\s*$", "", n)            # drop " (1)"
    n = re.sub(r"^Bella Canvas\s+", "", n, flags=re.I); n = re.sub(r"\s+T-?Shirt$", "", n, flags=re.I)
    n = re.sub(r"[\s-]*3001$", "", n)
    n = n.replace("Guildan 5000 black", "Black")
    fixes = {"Blue Strom":"Blue Storm","Millitary":"Military","Militery Green":"Military Green","Antique Saphire":"Antique Sapphire",
             "Sports Grey":"Sport Grey","Brown Savana":"Brown Savana","Peprika":"Paprika","Charity Pink":"Charity Pink"}
    n = fixes.get(n.strip(), n.strip())
    return "-".join(w.capitalize() if w.islower() else w for w in n.split())

def dedupe(files):
    seen = {}; out = []
    for f in sorted(files):
        h = hashlib.md5(open(f, "rb").read()).hexdigest()
        key = clean_name(f)
        if key in seen or h in seen.values(): continue
        seen[key] = h; out.append(f)
    return out

def occlusion(mask):
    """shirt mask -> soft alpha that is 1 on the shirt (holes filled) and 0 on hair/hands/background."""
    import cv2
    m = mask.astype(np.uint8)
    ff = m.copy(); h, w = m.shape; pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)                  # flood the outside from the corner
    filled = (m | (ff == 0)).astype(np.uint8)          # mask + enclosed holes (fabric shading)
    filled = cv2.dilate(filled, np.ones((7, 7), np.uint8))
    return cv2.GaussianBlur(filled.astype(np.float32), (0, 0), 2.5)

def composite(blank, logo, cx, cy, width, angle=0.0, top=None, occ=None):
    im = Image.open(blank).convert("RGB"); W, H = im.size
    lg = logo.copy(); lg.thumbnail((width, width * 1.1), Image.LANCZOS)
    if abs(angle) > 0.3: lg = lg.rotate(angle, resample=Image.BICUBIC, expand=True)
    if top is not None: cy = top + lg.height / 2
    x0, y0 = int(cx - lg.width / 2), int(cy - lg.height / 2)
    # fabric shading: scale logo colour by local shirt luminance relative to the patch mean
    base = np.asarray(im, np.float32) / 255.0
    patch = base[y0:y0 + lg.height, x0:x0 + lg.width]
    lum = patch.mean(axis=2, keepdims=True)
    lum_s = np.asarray(Image.fromarray((lum[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6)), np.float32)[..., None] / 255.0
    shade = np.clip(lum_s / max(lum_s.mean(), 0.05), 0.7, 1.12)
    la = np.asarray(lg, np.float32) / 255.0
    rgb, a = la[..., :3] * shade, la[..., 3:4] * 0.97
    if occ is not None: a = a * occ[y0:y0 + lg.height, x0:x0 + lg.width, None]
    out = base.copy(); out[y0:y0 + lg.height, x0:x0 + lg.width] = patch * (1 - a) + rgb * a
    return Image.fromarray((np.clip(out, 0, 1) * 255).round().astype(np.uint8))

def run(set_keys, teams, outroot="/home/claude/mockups/out", sample=False):
    logos = {t: Image.open(LOGOS[t]).convert("RGBA") for t in teams}
    manifest = {}
    for sk in set_keys:
        src, code, pose, (cx, cy, w) = SETS[sk]
        files = dedupe(glob.glob(src + "/*.jpg"))
        flat = "flatlay" in sk
        det = {}
        for f in files:
            r = analyse(np.asarray(Image.open(f).convert("RGB")), (cx, cy), flat=flat)
            det[f] = r
        ok = [r for r in det.values() if r]
        med = {k: float(np.median([r[k] for r in ok])) for k in ("cx", "neck_y", "body_w")}
        def pick(r):
            if not r: return med["cx"], med["neck_y"], 0.0
            dx, dy = r["cx"] - med["cx"], r["neck_y"] - med["neck_y"]
            x = r["cx"] if abs(dx) < 140 else med["cx"]
            y = r["neck_y"] if abs(dy) < 140 else med["neck_y"]
            a = max(-8.0, min(8.0, r["angle"])) if flat else 0.0
            return x, y, a
        gap = 0.16 * med["body_w"]          # collar -> top of print (~2.5-3 in on an adult tee)
        if sample: files = files[:2]
        for f in files:
            color = clean_name(f)
            px, ny, ang = pick(det.get(f))
            for t in teams:
                od = f"{outroot}/{t}/{code.lower()}_{pose.lower()}"; os.makedirs(od, exist_ok=True)
                of = f"{od}/{code}_{color}_{pose}_{t}.jpg"
                if os.path.exists(of) and not sample: continue
                r = det.get(f); oc = occlusion(r["mask"]) if (r and not flat) else None
                composite(f, logos[t], px, cy, int(w * MAXW[t]), angle=ang, top=ny + gap, occ=oc).save(of, quality=90)
                manifest.setdefault(sk, []).append(of)
        print(sk, len(files), "colors")
    return manifest

if __name__ == "__main__":
    sets = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else list(SETS)
    teams = sys.argv[2].split(",") if len(sys.argv) > 2 else list(LOGOS)
    run(sets, teams, sample="--sample" in sys.argv)
