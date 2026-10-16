"""Find clean letter donors across the 8 originals: letters whose columns are untouched by the old script band
except where it crosses (we need letters where the old script covered only the MIDDLE, i.e., uploads with centred scripts? none) -
so instead: take each donor letter from the CLEANED versions (work/clean2_NAME.png) where the old script region was rebuilt
and new script not yet applied."""
import numpy as np, json
from PIL import Image
from scipy import ndimage as nd
import importlib.util
spec=importlib.util.spec_from_file_location("b","/home/claude/ru/build.py"); b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
NAMES=["SHEA","RENDA","CORSO","TRAVA","MCHUGH","VELEZ","DILEO","MATTEO"]
out={}
for n in NAMES:
    import os
    p=f"/home/claude/ru/work/clean2_{n}.png"
    if not os.path.exists(p): p=f"/home/claude/ru/work/clean_{n}.png"
    c=np.array(Image.open(p))
    A=c[...,3]>=128
    B=b.letters_box(n)
    # column occupancy -> letter segments
    occ=A.any(0); segs=[]; ins=False
    for x in range(A.shape[1]):
        if occ[x] and not ins: x0=x; ins=True
        elif not occ[x] and ins: segs.append((x0,x)); ins=False
    if ins: segs.append((x0,A.shape[1]))
    segs=[s for s in segs if s[1]-s[0]>60]
    print(n, len(n), "segs", len(segs), segs)
    out[n]=dict(segs=segs,box=B)
json.dump(out,open("/home/claude/ru/work/donor_segs.json","w"))
