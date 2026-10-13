"""variant bulk-update payload: python3 varupd.py <payload.json> <firstVariantId> <secondVariantId> <firstMediaId> <nExtras>
first variant is the original default one; the rest are sequential from secondVariantId (step 32768)."""
import sys, json
d = json.load(open(sys.argv[1])); v0, v1, m0, nx = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
S = 32768
out = []
for i, c in enumerate(d["colors"]):
    vid = v0 if i == 0 else v1 + S * (i - 1)
    mid = m0 if i == 0 else m0 + S * (nx + i)
    out.append({"id": f"gid://shopify/ProductVariant/{vid}", "price": str(d["price"]), "mediaId": f"gid://shopify/MediaImage/{mid}"})
print(json.dumps(out, separators=(",", ":")))
