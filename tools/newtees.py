"""Build the Shopify payloads for the Bella+Canvas 3001 and Comfort Colors 1717 team tees.
python3 newtees.py <sha>  -> writes /tmp/.../newtees/<TEAM>_<GARMENT>.json (create input, option input, extras)"""
import os, sys, json
R = "/home/claude/ocbm-repository"
SHA = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/newtees"
os.makedirs(OUT, exist_ok=True)
RAW = f"https://raw.githubusercontent.com/Itfdff9/ocbm-repository/{SHA}"

TEAMS = {
    "DREAMS": dict(folder="dreams", name="DREAMS Fastpitch", tag="DREAMS", short="DREAMS", mark="DREAMS star logo", front="DREAMS Fastpitch logo printed center chest (about 10 in. wide)",
                   where="on and off the field", made="Made for game days, practices and everything in between.",
                   tags=["DREAMS Fastpitch", "Fastpitch", "Softball", "Softball Mom"], coll="gid://shopify/Collection/492324126976", nse=False),
    "BUFC": dict(folder="bufc", name="Bethpage United FC", tag="BUFC", short="BUFC", mark="BUFC crest", front="Bethpage United FC crest printed center chest (about 8 in. wide)",
                 where="on and off the pitch", made="Made for cold mornings at the field, travel tournaments and every day in between.",
                 tags=["Bethpage United FC", "BUFC", "Soccer"], coll="gid://shopify/Collection/492328747264", nse=False),
    "BLUEBIRDS": dict(folder="bluebirds", name="Bethpage Bluebirds FC", tag="BLUEBIRDS", short="Bluebirds", mark="Bluebirds crest", front="Bethpage Bluebirds FC crest printed center chest (about 10 in. wide)",
                      where="on and off the pitch", made="Made for cold mornings at the field, travel tournaments and every day in between.",
                      tags=["Bethpage Bluebirds FC", "Bluebirds", "Soccer"], coll="gid://shopify/Collection/492332744960", nse=False),
    "NSE": dict(folder="north_shore_elite", name="North Shore Elite", tag="NSE", short="NSE", mark="North Shore Elite emblem", front="North Shore Elite emblem printed center chest (about 10 in. wide)",
                where="on and off the mat", made="Made for early call times, long practice nights, competition weekends and every day in between.",
                tags=["North Shore Elite", "NSE", "Cheer", "Cheerleading", "designer"], coll="gid://shopify/Collection/492522275072", nse=True),
}
SIZES = ["S", "M", "L", "XL", "2XL", "3XL"]

# colour lists in Brian's default order: black, white, royal, light grey, dark grey, baby pink, hot pink, then the rest by family
BELLA = ["Black", "White", "Athletic Heather", "Dark Grey Heather", "Pink", "Charity Pink",
         "Ash", "Silver", "Solid Athletic Grey", "Dust", "Pebble", "Asphalt", "Dark Grey", "Storm", "Vintage Black", "Oxblood Black",
         "Vintage White", "Natural", "Soft Cream", "Tan",
         "Team Navy", "Navy", "Heather Navy", "Vintage Navy", "Steel Blue", "Slate", "Vintage Denim", "Blue Storm", "Cool Blue",
         "Carolina Blue", "Light Blue", "Dusty Blue", "Ocean Blue", "Atlantic", "Marine", "Deep Teal",
         "Team Purple", "Royal Purple", "Heather Team Purple", "Purple Storm", "Dark Lavender", "Lavender Blue", "Lavender Dust",
         "Light Violet", "Lilac", "Orchid", "Mauve",
         "Soft Pink", "Fuchsia", "Heather Peach", "Coral", "Red", "Poppy", "Cardinal", "Vintage Red", "Maroon",
         "Forest Green", "Heather Forest Green", "Evergreen", "Pine", "Dark Olive", "Army", "Military Green", "Synthetic Green", "Mint",
         "Yellow", "Gold", "Mustard", "Sunset", "Orange", "Burnt Orange", "Autumn", "Rust", "Clay",
         "Toast", "Chestnut", "Heather Brown", "Pebble Brown", "Vintage Brown"]
CC = ["Black", "White", "Grey", "Pepper", "Blossom", "Heliconia",
      "Ivory", "Butter", "Banana", "Sandstone", "Khaki", "Hemp", "Bay", "Sage",
      "Navy", "True Navy", "Midnight", "Denim", "Blue Jean", "Washed Denim", "Chambray", "Ice Blue", "Flo Blue", "Lagoon Blue",
      "Blue Spruce", "Seafoam", "Chalky Mint",
      "Grape", "Violet", "Neon Violet", "Orchid", "Berry",
      "Watermelon", "Peachy", "Bright Salmon", "Red", "Crimson", "Chili", "Paprika", "Brick", "Terracotta",
      "Moss", "Light Green", "Mustard", "Yam", "Burnt Orange", "Espresso"]
CC_FILE = {"Chili": "Chilli", "Blue Jean": "Blue-Jeans", "Sandstone": "Sand-Stone"}

GARMENTS = {
    "BC3001": dict(label="Bella+Canvas Tee", sub="Soft Premium Team Tee", price=28, colors=BELLA, tag="Bella+Canvas 3001",
                   intro="This soft, lightweight Bella+Canvas tee",
                   bullets=["Bella+Canvas 3001 Unisex Jersey tee, soft retail fit",
                            "Airlume combed and ring-spun cotton, 4.2 oz (heathers are cotton/poly blends)",
                            "Side-seamed, with shoulder taping"],
                   chart=[("S", 18, 28), ("M", 20, 29), ("L", 22, 30), ("XL", 24, 31), ("2XL", 26, 32), ("3XL", 28, 33)],
                   fit="Unisex retail fit (a little trimmer than a classic tee). For a roomier fit, size up.",
                   care="Machine wash cold, inside out. Tumble dry low or hang dry. Do not iron directly on the print.",
                   seo="Bella+Canvas 3001 soft tee"),
    "CC1717": dict(label="Comfort Colors Tee", sub="Garment-Dyed Team Tee", price=32, colors=CC, tag="Comfort Colors 1717",
                   intro="This heavyweight, garment-dyed Comfort Colors tee",
                   bullets=["Comfort Colors 1717 garment-dyed heavyweight tee, relaxed fit",
                            "100% ring-spun cotton, 6.1 oz",
                            "Garment-dyed for a soft, broken-in feel; every shirt's color varies a little",
                            "Twill-taped neck and shoulders"],
                   chart=[("S", 20, 28), ("M", 22, 29), ("L", 24, 30), ("XL", 26, 31), ("2XL", 28, 32), ("3XL", 30, 33)],
                   fit="Unisex relaxed fit. Runs roomy; size down for a closer fit.",
                   care="Machine wash cold, inside out, with like colors (garment-dyed colors can bleed in the first few washes). Tumble dry low or hang dry. Do not iron directly on the print.",
                   seo="Comfort Colors 1717 garment-dyed tee"),
}


def exists(p): return os.path.exists(f"{R}/{p}")


def images(team, g):
    t = TEAMS[team]; f = t["folder"]; tag = t["tag"]
    main, extra = {}, []
    if g == "BC3001":
        for c in BELLA:
            d = c.replace(" ", "-")
            for sub, pose in (("bc3001_seated", "Seated"), ("bc3001_couch", "Couch")):
                p = f"{f}/{sub}/BC3001_{d}_{pose}_{tag}.jpg"
                if exists(p): main[c] = p; break
            else:
                raise SystemExit(f"missing Bella image {c}")
        for c in ["Black", "Asphalt", "Charity Pink", "Soft Pink"]:
            extra.append((c, f"{f}/bc3001_couch/BC3001_{c.replace(' ', '-')}_Couch_{tag}.jpg"))
    else:
        for c in CC:
            d = CC_FILE.get(c, c.replace(" ", "-"))
            p = f"{f}/cc1717_flatlay/CC1717_{d}_Flatlay_{tag}.jpg"
            if not exists(p): raise SystemExit(f"missing CC image {c}")
            main[c] = p
        for c in ["Black", "White", "Grey", "Pepper"]:
            extra.append((c, f"{f}/cc1717_fall/CC1717_{c}_Fall_{tag}.jpg"))
    for _, p in extra:
        if not exists(p): raise SystemExit("missing extra " + p)
    return main, extra


def desc(team, g):
    t, G = TEAMS[team], GARMENTS[g]
    if t["nse"]:
        p1 = (f"<p>Rep <strong>{t['name']}</strong> {t['where']}. {G['intro']} has the North Shore Elite emblem on the chest. "
              f"You pick the back: the Blue Black White design, your athlete's last name and number in varsity jersey lettering, or no back print, all at no extra charge. {t['made']}</p>")
        back = ("<li>Back, your choice, included free: the Blue Black White design (about 12 in. wide), OR your athlete's last name over number in varsity jersey lettering, "
                "OR no back print. Choose your back and enter the name and number when you order.</li>")
    else:
        p1 = (f"<p>Rep <strong>{t['name']}</strong> {t['where']}. {G['intro']} has the {t['mark']} on the chest and your player's last name and number on the back, "
              f"jersey style, at no extra charge. {t['made']}</p>")
        back = "<li>Back: player last name over number in varsity jersey lettering, included free. Enter the name and number when you order.</li>"
    li = "".join(f"<li>{b}</li>" for b in G["bullets"])
    rows = "".join(f"<tr><td>{s}</td><td>{w}</td><td>{l}</td></tr>" for s, w, l in G["chart"])
    return (p1 + "\n<h3>Details</h3>\n<ul>" + f"<li>Front: {t['front']}</li>" + back + li +
            "<li>Printed in-house with DTF (Direct to Film) for a bright, soft, long-lasting print</li>"
            "<li>Imported blank garment; printed in-house on Long Island, NY</li></ul>\n"
            "<h3>Size chart (inches, approximate)</h3><table><tr><th>Size</th><th>Width</th><th>Length</th></tr>" + rows + "</table>\n"
            f"<p>{G['fit']}</p>\n<p><em>(Sizes 2XL and up are a few dollars more to cover the larger garment: 2XL +$2, 3XL +$4.)</em></p>\n"
            f"<h3>Care</h3>\n<p>{G['care']}</p>\n"
            "<p>Questions? Email onecraftyboymama@gmail.com or message us on Instagram or Facebook @onecraftyboymama.</p>")


def build(team, g):
    t, G = TEAMS[team], GARMENTS[g]
    main, extra = images(team, g)
    word = "tee"
    media = []
    order = list(G["colors"])
    # featured first, then the lifestyle extras, then every other colour
    media.append({"originalSource": f"{RAW}/{main[order[0]]}", "alt": f"{t['name']} {G['label']} in {order[0]}", "mediaContentType": "IMAGE"})
    for c, p in extra:
        media.append({"originalSource": f"{RAW}/{p}", "alt": f"{t['name']} {G['label']}, {c}, worn", "mediaContentType": "IMAGE"})
    for c in order[1:]:
        media.append({"originalSource": f"{RAW}/{main[c]}", "alt": f"{t['name']} {G['label']} in {c}", "mediaContentType": "IMAGE"})
    n = len(order)
    if t["nse"]:
        seo_t = f"{t['name']} {G['label']} – Custom Back | One Crafty Boy Mama"
        seo_d = (f"{t['name']} cheer tee with the NSE emblem on the chest. Pick your back: Blue Black White, your athlete's name and number, or blank, free. "
                 f"{G['seo']}, {n} colors, S–3XL, 2XL and up a few dollars more.")
    else:
        seo_t = f"{t['name']} {G['label']} – Custom Name & Number | One Crafty Boy Mama"
        seo_d = (f"{t['name']} tee with the {t['short']} logo on the chest and your player's name and number on the back, free. "
                 f"{G['seo']}, {n} colors, S–3XL, 2XL and up a few dollars more.")
    product = {"title": f"{t['name']} {G['label']} | {G['sub']}", "descriptionHtml": desc(team, g), "vendor": "One Crafty Boy Mama",
               "productType": "T-Shirt", "status": "ACTIVE", "collectionsToJoin": [t["coll"]],
               "tags": sorted(set(t["tags"] + [G["tag"], "DTF", "T-Shirt", "Tee", "Team Gear", "Team Spirit"])),
               "seo": {"title": seo_t, "description": seo_d}}
    return {"product": product, "media": media, "colors": order, "price": G["price"],
            "colorMedia": {c: f"{RAW}/{main[c]}" for c in order}}


if __name__ == "__main__":
    for team in TEAMS:
        for g in GARMENTS:
            d = build(team, g)
            json.dump(d, open(f"{OUT}/{team}_{g}.json", "w"), ensure_ascii=False)
            print(team, g, len(d["colors"]), "colors", len(d["media"]), "images", len(json.dumps({"product": d["product"], "media": d["media"]})), "bytes")
