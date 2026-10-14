# NSE "CHEER" photo template — LOCKED MASTER (locked by Brian 2026-10-14)
Do not change the layout, fonts, sizes or colors. Only the name and photos change per order.

- Size: 12 x 14.67 in at 300 DPI (3600 x 4400), transparent outside a distressed edge, hard alpha.
- Background: smoky royal-blue clouds, glitter specks, lightning bolts, stadium-light grid (left), two faded blue duotone photos (upper middle, left).
- Main photo: background removed automatically (rembg isnet-general-use), blue rim glow, sits center-left.
- NAME (editable): Hurricane-Regular brush script, white with ice-blue edge and royal drop shadow, rotated 7 deg, top-left corner, sized to 74% width / 21.5% height of the canvas, underline swash from 22% to 92% of the word. Clear of the face and stadium lights.
- CHEER (fixed): Anton, rotated to read top-to-bottom, right side, smoky-blue fill with lightning, white outline + white inline.
- Bottom line (fixed): "North Shore Elite  2026" in Hurricane, plus outline heart to the right.
- Fonts in fonts/: Hurricane-Regular.ttf (script), Anton-Regular.ttf (CHEER). (MrsSaintDelafield = old, not used.)

Make one:
  python3 cheer_template.py --name "Emily" --photo main.jpg [--ghost1 face.jpg] [--ghost2 action.jpg] --out OUT.png [--layers DIR]
- --layers writes background / CHEER / name / bottom-line PNGs for Canva.
- Customer photos are NOT stored in this public repo.
Files: LOCKED_MASTER_preview_Emily_*.png (approved mockup, stock sample photo), LOCKED_vs_original.jpg, layers_sample/.
