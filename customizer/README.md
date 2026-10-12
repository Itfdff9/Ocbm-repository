# OCBM live designer ("Design it" on product pages)

- `ocbm-designer.js` — copy of the theme asset `assets/ocbm-designer.js` (theme "Trade – team name/number field").
- `config.json` — loaded live by the script from raw.githubusercontent.com. Edit here to add teams, designs, colors or print spots; no theme edit needed.
- `blanks/` — 1000 px garment photos used as the preview base (crew, tee, hoodie, camo). Tee/crew backs are drawn as a tinted silhouette.
- `designs/` — preview PNGs of each team's preset designs (not print files).
- `fonts/Graduate-Regular.ttf` — varsity font for the name/number preview.

Shown on any product tagged `designer`; the team comes from the product's team tag (e.g. `NSE`), the garment from its blank tag (Gildan 18000/5000/18500, Port & Company PC78HC).
Order fields (line-item properties): Front Design, Front Placement, Front Upload (file), Back Design, Back Placement, Back Upload (file), Athlete Last Name, Athlete Number.
Print spots (inches): left chest 3.5×3.5 · center chest 10×10 · full front 12×14 · upper back 10×4 · full back 12×14. Uploads are scaled to fit the chosen spot; the page warns under 150 dpi.
