# step 0: fake-checker background connected to the image edge (flood fill, never a global colour match)
from PIL import Image
import numpy as np
from scipy import ndimage as nd
a=np.asarray(Image.open("src.webp").convert("RGB")).astype(np.float32)
lum=a.mean(2); sat=a.max(2)-a.min(2)
lab,n=nd.label((sat<=16)&(lum>=195))
edge=np.unique(np.concatenate([lab[0],lab[-1],lab[:,0],lab[:,-1]])); edge=edge[edge>0]
bg=np.isin(lab,edge)
for i in range(4):
    ring=nd.binary_dilation(bg)&~bg; bg|=ring&(a.min(2)>165)&(sat<=40)
np.save("bg.npy",bg); print("edge background",round(bg.mean(),3))
