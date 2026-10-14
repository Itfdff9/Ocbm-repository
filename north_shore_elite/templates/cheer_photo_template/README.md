# NSE "CHEER" photo template (made 2026-10-14)
Original artwork in the style Brian asked for (smoky blue background, lightning, stadium lights, vertical CHEER, script name).
12 x 14.67 in at 300 DPI (3600 x 4400), transparent outside a distressed edge, hard alpha.

Make one:  python3 cheer_template.py --name "Emily" --photo main.jpg [--ghost1 face.jpg] [--ghost2 action.jpg] [--bottom "North Shore Elite"] [--year 2026] --out OUT.png [--layers DIR]
- main photo: background removed automatically (rembg, isnet-general-use); best with a full-length photo holding poms.
- ghost1 / ghost2: optional faded blue photos (upper middle / left). Default = the main photo.
- --layers writes separate transparent PNGs (background, CHEER, name, bottom line) for Canva.
Customer photos are NOT stored in this public repo.
