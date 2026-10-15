"""NORTH SHORE: rebuild the outer black outline at one constant width with smooth edges (like Elite),
and make the black between letters solid and clean. Letter fills/trims and counters are untouched;
Elite is painted back on top."""
import numpy as np, cv2, os, sys
from PIL import Image
from scipy import ndimage as nd
WID = float(os.environ.get("WID", 24)); OUT = os.environ.get("OUT", "letters_out")
o = np.load("clean5.npy"); el = np.load("elite_layer.npy"); Ef = np.load("Ef.npy")
H, W = el.shape
rgb = o[..., :3].astype(int); op = o[..., 3] == 255
navy = (rgb[..., 2] - rgb[..., 0] > 25) & (rgb[..., 2] < 170)
body = op & (rgb.max(2) >= 110) & ~navy & ~el & ~Ef
# drop tiny loose body specks sitting in the black/outside (old bleed flecks)
l, _ = nd.label(body); s = np.bincount(l.ravel()); body &= s[l] >= 400
# light jaggy cleanup of the body edge (1px open/close)
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
body = cv2.morphologyEx(cv2.morphologyEx(body.astype(np.uint8), cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k).astype(bool)
# counters = holes fully enclosed by the letter body -> left exactly as they are
filled = nd.binary_fill_holes(body | el | Ef)
counters = filled & ~body & ~el & ~Ef
lc, _ = nd.label(counters); sc = np.bincount(lc.ravel()); counters &= sc[lc] < 200000
# outline: constant distance from the body, outer edge smoothed (4x supersampled contour)
d = nd.distance_transform_edt(~body)
ring = (d <= WID).astype(np.uint8)
cs, hi = cv2.findContours(ring, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
SS = 4; m = np.zeros((H * SS, W * SS), np.uint8)
for i, c in enumerate(cs):
    if len(c) < 20: continue
    p = c[:, 0, :].astype(np.float64)
    p = np.stack([nd.gaussian_filter1d(p[:, 0], 3, mode="wrap"), nd.gaussian_filter1d(p[:, 1], 3, mode="wrap")], 1)
    cv2.fillPoly(m, [np.round((p + 0.5) * SS * 16).astype(np.int32)], 0 if hi[0][i][3] >= 0 else 255, cv2.LINE_AA, shift=4)
ringS = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA) >= 128
ringS |= body
out = np.zeros_like(o)
out[ringS] = (8, 8, 10, 255)                 # black outline + black between letters
out[body] = o[body]                           # Brian's letter fill + white trim untouched
# counters: solid-black ones stay exactly as drawn; see-through ones (the open spaces between letters
# that are closed off by touching trims) get the same constant-width black edge as the outside
lc, nc = nd.label(counters)
has_tr = np.zeros(nc + 1, bool); has_tr[np.unique(lc[counters & ~op])] = True; has_tr[0] = False
keep = counters & ~has_tr[lc]
redo = counters & has_tr[lc]
out[keep] = o[keep]
kn = keep & ~((rgb.min(2) > 175) | ((rgb[..., 2] >= 170) & (rgb[..., 2] - rgb[..., 0] > 40))); out[kn] = (8, 8, 10, 255)   # old blue bleed smudges inside solid counters -> black
out[redo & ringS] = (8, 8, 10, 255)
print("counters kept", int(keep.sum()), "rebuilt", int(redo.sum()))
# the area under/around Elite stays exactly as the cleaned file had it
near_el = nd.distance_transform_edt(~(el | Ef)) <= 2
out[el | Ef | near_el] = o[el | Ef | near_el]
# see-through slots/holes between letters (enclosed by the outline, not a letter counter) -> solid black
tr = out[..., 3] < 128
lt, nt = nd.label(tr); st = np.bincount(lt.ravel())
bigbg = st.argmax() if nt else 0
edge_ids = np.unique(np.concatenate([lt[0], lt[-1], lt[:, 0], lt[:, -1]]))
hole = tr & (lt != bigbg) & ~np.isin(lt, edge_ids) & ~counters
# keep the open pockets inside Elite's swash (they belong to Elite)
nearE = nd.distance_transform_edt(~(el | Ef)) <= 6
hl = np.unique(lt[hole & nearE & ~ringS]); hl = hl[hl > 0]
elite_pockets = np.isin(lt, hl) & (nd.distance_transform_edt(~body) > WID + 4)
hole &= ~elite_pockets
out[hole] = (8, 8, 10, 255); print("slots filled", int(hole.sum()))
al = out[..., 3] >= 128; out[..., 3] = np.where(al, 255, 0); out[~al, :3] = 0
l, _ = nd.label(al); s = np.bincount(l.ravel()); out[(s[l] < 150) & al] = 0
np.save(OUT + ".npy", out); Image.fromarray(out, "RGBA").save(OUT + ".png")
