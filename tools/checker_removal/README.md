# Fake-checkerboard removal (designs that come with the grey/white "transparency" squares baked in)

Built for *Literally just a cheerleader* (2026-10-21). Run in order from a folder holding `src.webp`:

| step | script | makes | what it finds |
|---|---|---|---|
| 0 | `step0_edgefill.py` | `bg.npy` | checker connected to the image edge (flood fill + 4 defringe passes) |
| 1 | `feat2.py` | `zone.npy` | areas with checker correlation (axis + diagonal) |
| 2 | `mask4core.py` | `core4.npy` | checker X-corners grown along the lattice, dark/light cells on that lattice |
| 3 | `alt4.py` | `altcells4.npy` | neutral cells that alternate light/dark every ~18 px (catches thin strips between tile rows) |
| 4 | `altT.py` | `altcells_tint.npy` | same, for checker seen through the blue haze |
| 5 | `mask6.py` | `bg6.npy` | combines 0-4, fills leftover cells, floating islands, defringe |
| 6 | `energy2.py` | `E.npy` | checker "energy" per pixel (light samples only) |
| 7 | `mask10.py` | `bg10.npy` | removes small enclosed light pockets that are mostly checker |
| 8 | `protect.py` | `bg11.npy` | **letter-tile protection**: anything removed inside a closed tile outline is put back |
| 9 | `build_final.py SRC bg11.npy UPSCALED OUT LONGPX [lanczos]` | print PNG | hard alpha 0/255, edge colour re-filled from inside, 300 DPI |

Checker square size in that upload: ~18 px (period 35-40). Change the 13-23 px search ranges for other sizes.
Always check the cut-out on black at 100% before printing — the semi-see-through white splatter keeps a faint grey texture where the checker showed through it.
