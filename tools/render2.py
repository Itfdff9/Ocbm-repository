"""Re-render the extra tee sets with per-photo collar placement.
On-model sets: collar.find gives the neck centre (cx) and collar dip for every photo.
Flatlays: photos come from 7 base layouts; each layout's collar was measured by hand."""
import os, sys, glob, json, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
from render import SETS, dedupe, clean_name, composite, occlusion
from collar import find

R = "/home/claude/ocbm-repository"
TEAMS = {  # tag: (logo, repo folder, width factor)
    "DREAMS": (R + "/dreams/logo/DREAMS_PARENT_cleaned_300dpi.png", "dreams", 1.0),
    "BUFC": (R + "/bufc/logo/BUFC_crest_transparent.png", "bufc", 0.78),
    "BLUEBIRDS": (R + "/bluebirds/logo/BLUEBIRDS_crest_transparent.png", "bluebirds", 0.86),
    "NSE": (R + "/north_shore_elite/logo/NSE_logo_transparent.png", "north_shore_elite", 0.80),
}
DIRS = {"g5000_selfie": "g5000_selfie", "bc3001_a": "bc3001_seated", "bc3001_b": "bc3001_couch",
        "cc1717_halloween": "cc1717_fall", "cc1717_flatlay": "cc1717_flatlay"}
# flatlay base layouts: representative -> (collar centre x, bottom of the collar rib y)
FLAT = {"Banana.jpg": (1060, 514), "Berry.jpg": (1034, 514), "Black.jpg": (1057, 518), "Butter.jpg": (1072, 514),
        "Grey.jpg": (1062, 510), "Pepper.jpg": (1050, 518), "Washed Denim.jpg": (1050, 634)}
OVERRIDE = {}   # file name -> {"cx":..., "top":...} after visual review
# sets whose photos are all the same shot recoloured (checked: zero pixel offset between colours).
# One measured placement for the whole set; per-photo detection drifted on 6 of the 32 selfies.
FIXED = {"g5000_selfie": (1044, 632)}   # collar rib bottom ~425 at x 1044 -> print top ~207 px below
FLAT_GAP = 150          # ~2.5 in below the rib on these flatlays (~60 px/in)
MODEL_GAP = 0.55        # x median neck-opening width: ~2.75-3 in below the collar on a model


def tight(path):
    im = Image.open(path).convert("RGBA"); a = np.asarray(im)[..., 3]; ys, xs = np.where(a > 8)
    return im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def plan(set_key):
    src, code, pose, (cx0, cy0, w) = SETS[set_key]
    files = dedupe(glob.glob(src + "/*.jpg"))
    out = {}
    if set_key == "cc1717_flatlay":
        sys.path.insert(0, "/home/claude/audit"); from cluster import clusters
        g = clusters(files)
        for members in g.values():
            rep = os.path.basename(members[0]); cx, seam = FLAT[rep]
            for f in members:
                out[f] = {"cx": cx, "top": seam + FLAT_GAP, "mask": None}
        return out, code, pose, w
    from symm import sym_cx
    det = {}
    for f in files:
        a = np.asarray(Image.open(f).convert("RGB"))
        d = find(a, (cx0, cy0)) or {}
        sc, score = sym_cx(a, xc=int(cx0 if set_key != "cc1717_halloween" else 950))
        d["sym"], d["score"] = sc, score
        det[f] = d
    ok = [d for d in det.values() if "dip" in d]
    ow = float(np.median([d["open_w"] for d in ok]))
    cxm = float(np.median([d["sym"] for d in det.values() if d["score"] >= 0.3]))
    for f, d in det.items():
        if d["score"] >= 0.3: cx = d["sym"]
        elif d.get("fit") and abs(d["cx"] - cxm) < 150: cx = d["cx"]
        else: cx = cxm
        d["cx_final"] = cx
        m = d.get("mask")
        if m is not None:
            cols = [np.argmax(m[:, x]) for x in range(int(cx) - 20, int(cx) + 21) if m[:, x].any()]
            d["dip_c"] = float(np.median(cols)) if cols else None
        else:
            d["dip_c"] = None
    dips = [d["dip_c"] for d in det.values() if d["dip_c"] is not None]
    dipm = float(np.median(dips))
    for f, d in det.items():
        dip = d["dip_c"] if d["dip_c"] is not None and abs(d["dip_c"] - dipm) < 110 else dipm
        o = OVERRIDE.get(os.path.basename(f), {})
        if set_key in FIXED:
            o = {"cx": FIXED[set_key][0], "top": FIXED[set_key][1], **o}
        out[f] = {"cx": o.get("cx", d["cx_final"]), "top": o.get("top", dip + MODEL_GAP * ow), "mask": d.get("mask"),
                  "score": d["score"]}
    return out, code, pose, w


def run(set_keys, teams, files_only=None, outroot=None):
    logos = {t: tight(TEAMS[t][0]) for t in teams}
    for sk in set_keys:
        pl, code, pose, w = plan(sk)
        json.dump({os.path.basename(f): {"cx": v["cx"], "top": v["top"]} for f, v in pl.items()},
                  open(f"{R}/tools/place_{sk}.json", "w"), indent=0)
        n = 0
        for f, v in pl.items():
            color = clean_name(f)
            if files_only and color not in files_only: continue
            occ = None
            if v["mask"] is not None:
                mk = v["mask"]; x0, x1 = int(v["cx"] - w * 0.4), int(v["cx"] + w * 0.4); y0, y1 = int(v["top"]), int(v["top"] + w * 0.8)
                box = mk[max(0, y0):y1, max(0, x0):x1]
                hole_free = box.mean() if box.size else 0
                if hole_free > 0.93: occ = occlusion(mk)   # only trust the shirt mask when it is clean under the print
            for t in teams:
                od = f"{outroot or R}/{TEAMS[t][1]}/{DIRS[sk]}"; os.makedirs(od, exist_ok=True)
                of = f"{od}/{code}_{color}_{pose}_{t}.jpg"
                composite(f, logos[t], v["cx"], 0, int(w * TEAMS[t][2]), top=v["top"], occ=occ).save(of, quality=90)
            n += 1
        print(sk, n, "photos x", len(teams), "teams", flush=True)


if __name__ == "__main__":
    sets = sys.argv[1].split(",")
    teams = sys.argv[2].split(",")
    only = sys.argv[3].split(",") if len(sys.argv) > 3 and sys.argv[3] != "-" else None
    outroot = sys.argv[4] if len(sys.argv) > 4 else None
    run(sets, teams, only, outroot)
