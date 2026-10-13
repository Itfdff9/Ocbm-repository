"""after productOptionsCreate(Size) -> new variants are sequential: colour-major, sizes M..3XL.
python3 sizeupd.py <payload.json> <firstNewVariantId> <firstMediaId> <nExtras> [part] [nparts]"""
import sys, json
d = json.load(open(sys.argv[1])); b, m0, nx = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
part = int(sys.argv[5]) if len(sys.argv) > 5 else 0; nparts = int(sys.argv[6]) if len(sys.argv) > 6 else 1
S = 32768; P = d["price"]; up = {"2XL": 2, "3XL": 4}
out = []
for i, c in enumerate(d["colors"]):
    mid = m0 if i == 0 else m0 + S * (nx + i)
    for j, sz in enumerate(["M", "L", "XL", "2XL", "3XL"]):
        e = {"id": f"gid://shopify/ProductVariant/{b + S * (5 * i + j)}", "mediaId": f"gid://shopify/MediaImage/{mid}"}
        if sz in up: e["price"] = str(P + up[sz])
        out.append(e)
k = (len(out) + nparts - 1) // nparts
print(json.dumps(out[part * k:(part + 1) * k], separators=(",", ":")))
