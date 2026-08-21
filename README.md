# cdl-ftw-zarr-marimo

One marimo notebook, `cdl-ftw.py`: the USDA Cropland Data Layer (icechunk Zarr v3
on source.coop, 30 m 2008-2025 and 10 m 2024-2025, with majority pyramids) under
Fields of the World (Robinson et al. 2026: field polygons and a P(field)
probability Zarr from Sentinel-2, 10 m, 2024-2025), as a map you can fly plus
SQL you can read.

- **The map is xarray + numpy.** The CDL window and the P(field) grid are xarray
  slices of the Zarr (grids, no row expansion); the view is drawn as tiles: deck's
  TileLayer asks for z/x/y, the kernel answers 256 px PNGs, one read per batch of
  tiles, the FTW clip / disagreement decided per output pixel, field outlines from
  FTW's per-state PMTiles. Tiles cached in memory, FTW mask chunks on disk.
- **DuckDB is for the vector side**: the fiboa GeoParquet (one file per state) over
  httpfs with `cache_httpfs`, `ST_Contains` of pixel centres into field polygons,
  per-field majority crop and purity, the CDL x FTW 2x2. Those are the SQL cells
  under the map.
- Two checkboxes: **fields** (pixels clipped to P(field) >= 0.5 with outlines) and
  **disagreement** (CDL crop / non-crop x FTW field / no field; 2024-2025 only).

Run:

```bash
uv sync
uv run python tools/patch_lonboard_raster_unlit.py   # once per install, see below
uv run marimo edit cdl-ftw.py
```

`tools/patch_lonboard_raster_unlit.py` raises lonboard's 10 s tile request
timeout to 120 s (a slow batch otherwise leaves blank tiles deck never re-asks
for) and turns off deck's default lighting on
lonboard's raster tile mesh (every tile otherwise draws ~0.69x darker; no Python
prop reaches it). Re-run after any install, then restart the kernel (the JS is read into the
Map widget when it is created; a browser reload changes nothing).

History: grew out of [x-sql-marimo](https://github.com/kentstephen/x-sql-marimo)
(the xarray-sql / DataFusion / DuckDB fold notebooks) on 2026-08-20 and moved here because its map no longer uses that
engine. Record in `docs/ftw-cdl-notes.md`.
