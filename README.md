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
- No DuckDB in the map notebook (since 2026-08-21). The per-field joins (the
  fiboa GeoParquet over httpfs, `ST_Contains` of pixel centres into field
  polygons, per-field majority crop and purity, the CDL x FTW 2x2) were moved
  to a second notebook that is kept out of the repo for now.
- Two checkboxes: **fields** (pixels clipped to P(field) >= 0.5, outlines from
  z12) and **disagreement** (CDL crop / non-crop x FTW field / no field;
  2024-2025 only); a year strip, crops-only, a legend that isolates a class on
  click, "analyze what's in view", a place search (Photon), and a refresh
  button that rebuilds the tile layer.

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
