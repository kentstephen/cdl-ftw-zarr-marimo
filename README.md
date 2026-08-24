# cdl-ftw-zarr-marimo

One marimo notebook, `cdl-ftw.py`: the USDA Cropland Data Layer (icechunk Zarr v3
on source.coop, 30 m 2008-2025 and 10 m 2024-2025, with majority pyramids) under
Fields of the World (Robinson et al. 2026: field polygons and a P(field)
probability Zarr from Sentinel-2, 10 m, 2024-2025), as a map you can fly:
xarray + numpy + lonboard, nothing else on the map path.

- **The map is xarray + numpy.** The CDL window and the P(field) grid are xarray
  slices of the Zarr (grids, no row expansion); the view is drawn as tiles: deck's
  TileLayer asks for z/x/y, the kernel answers 256 px PNGs, one read per batch of
  tiles, the FTW clip / disagreement decided per output pixel, field outlines from
  FTW's per-state PMTiles. Tiles cached in memory, FTW mask chunks on disk.
- No DuckDB on the map path (since 2026-08-21): xarray + numpy + lonboard only.
- Two checkboxes: **fields** (pixels clipped to P(field) >= 0.5, outlines from
  z12) and **disagreement** (CDL crop / non-crop x FTW field / no field;
  2024-2025 only); a year strip, crops-only, a legend that isolates a class on
  click, "analyze what's in view", a place search (Photon), and a refresh
  button that rebuilds the tile layer.

A second notebook, `aef-similarity.py` (branch work, 2026-08-24): click a field
and the view repaints as cosine similarity to it in the **AlphaEarth
Foundations** embeddings (64-dim annual vectors at 10 m, 2017-2025, the
`tge-labs/aef-mosaic` Zarr on the same bucket). The click flood-fills the FTW
P(field) grid into the field's pixels and averages their AEF vectors into one
unit reference; similarity renders as viridis tiles through the same serve
machinery, FTW outlines on top. The mosaic has no pyramid, so similarity lives
from camera ~z12. The click is not lonboard's `on_click` (never worked under
marimo): the HUD's JS catches canvas clicks and the kernel unprojects them.
First step toward the agreement map.

The third notebook, `aef-agreement.py`: the click-a-field question asked of
every field at once. Per view, the FTW fields (connected components of
P(field) at 10 m) each get a mean AEF vector and a CDL majority crop; each
crop field is scored by how many of its 10 nearest embedding look-alikes in
view carry its CDL label. Dark = the datasets agree; bright yellow = they
disagree (a young orchard, a CDL mislabel, an odd field): a surprise
detector. The panel lists the most surprising fields; clicking a field gives
its story (CDL class, acres, what its look-alikes are). First run found an
88 ac Delta field CDL calls Cotton whose look-alikes are Tomatoes 6/10.

Data:

- CDL: <https://source.coop/chill/usda-cropland-data-layer>
- Fields of the World: <https://source.coop/ftw/global-data>
- AlphaEarth Foundations mosaic: <https://source.coop/tge-labs/aef-mosaic>

Run:

```bash
uv sync
uv run marimo edit cdl-ftw.py            # or: uv run marimo edit cdl-ftw.py --sandbox
```

`tools/patch_lonboard_raster_unlit.py` makes three edits to lonboard 0.16's
shipped JS: turns off deck's default lighting on the raster tile mesh (every
tile otherwise draws ~0.69x darker; no Python prop reaches it), raises the 10 s
tile request timeout to 120 s (a slow batch otherwise leaves tiles deck never
re-asks for), and gives each raster layer its own deck id plus a reload trigger
(under marimo the model id is undefined, so a rebuilt layer read to deck as an
update of the old one and kept its tiles). The notebook's first cell applies
the tool in whatever environment runs the notebook (so `--sandbox` works too),
before the Map is created.

History: grew out of [x-sql-marimo](https://github.com/kentstephen/x-sql-marimo)
(the xarray-sql / DataFusion / DuckDB fold notebooks) on 2026-08-20 and moved here because its map no longer uses that
engine. Record in `docs/ftw-cdl-notes.md`.
