# CLAUDE.md

Guidance for Claude Code in this repository. Inherits the global rules in
`~/CLAUDE.md` (tone, no em dashes, memory in `.claude/memory/` here,
colorblind-safe encodings: Stephen has trouble seeing red).

## What this is

`cdl-ftw.py`, one marimo notebook: USDA Cropland Data Layer (icechunk Zarr v3,
`s3://us-west-2.opendata.source.coop/chill/usda-cropland-data-layer/v0.1.0.icechunk`,
30 m 2008-2025 + 10 m 2024-2025, majority pyramids) x Fields of the World
(`tge-labs/ftw-global-data` on the same bucket: P(field) Zarr at 10 m + pyramid,
per-state fiboa GeoParquet, per-state PMTiles). Born in
`~/dev/projects/x-sql-marimo` (as `xsql-cdl-fields.py`, one day, 2026-08-20) and
moved here because its map stopped being an xarray-sql / DuckDB-fold notebook.
That repo's `CLAUDE.md`, `docs/ftw-cdl-notes.md` and `docs/cdl-crops-notes.md`
hold the full history; a copy of the FTW notes is in `docs/`.

## The division of tools (Stephen's call, 2026-08-20 night)

- **The map is xarray + numpy.** CDL window = `ds.crop_type.sel(year).sel(x, y)`
  on the level's Zarr group; P(field) = the same on the probability Zarr,
  `>= 0.5`; render = numpy closed-form EPSG:5070 forward (`albers_xy`, verified
  to the mm) sampling the grid per tile lattice, PIL for the outline polylines
  and the PNG. No DuckDB, no xarray-sql on this path. It used to go through
  DuckDB rows and back; that was a detour (a per-query ~0.2 s fixed overhead on
  the xql table and an array -> rows -> array round trip).
- **DuckDB is for the vector joins**: the parquet through httpfs + the
  `cache_httpfs` community extension (byte ranges kept on disk under the OS tmp
  dir), `ST_Contains`, per-field crop / purity, the 2x2. Those are the SQL cells
  under the map. `xarray-sql` (`xql.register`) is still used there to expose the
  CDL 10 m levels as tables for the pixel -> field join.
- The map is **tiles**: lonboard `RasterLayer` (deck TileLayer), `_fetch_tile`
  batches a burst of requests (BATCH_S 0.05), serves the WHOLE VIEW per batch
  (deck caps in-flight tile requests at 6), cuts the PNGs per tile from one
  grid, caches tiles in memory (TILE_CACHE). A state change (year, checkboxes,
  crops-only, selection) rebuilds the layer, REMOVE THEN ADD via `deck.layers`
  (under marimo every lonboard layer has id `undefined`; a direct replacement
  reads to deck as an update and it keeps its loaded tiles). The TMS must carry
  a `boundingBox` (morecantile's stock WebMercatorQuad lacks one); the TMS-less
  path in lonboard 0.16 is dead code (`getTileData` returns null).
- **The lighting patch is REQUIRED**: lonboard's tile mesh fragment shader calls
  `lighting_getLightColor`, ~0.69x on every channel, `opacity` ignored, no
  Python prop reaches it. `tools/patch_lonboard_raster_unlit.py` replaces the
  line; re-run after any install; hard-reload the browser. Without it the
  colours are wrong (a protan-safe palette drawn dark).

## Things that cost a round trip (keep)

- Fields of the World: `confidence` is NULL for the whole US; the STAC items'
  bboxes are wrong (US_CA reports Montana); state extents are embedded in the
  notebook from the files' own row-group stats. The parquet's geometry arrives
  `GEOMETRY('OGC:CRS84')`, cast `::GEOMETRY` for lonboard. Row groups are ~13
  MB and only roughly spatially sorted: a viewport read is 13-40 MB; that was
  the ~10 s stall on a slow link, gone from the map now (raster clip + PMTiles
  outlines), and `cache_httpfs` makes repeats local for the SQL cells.
- The PMTiles outlines: tippecanoe z0-13, layers "2024" / "2025", no id
  (draw-only); the reader is the HRRR counties film's PMTiles v3 + MVT decode
  by copy; segments along a tile's clip line are dropped (no seams) and
  polylines are NOT closed when drawn (a cut piece closed itself with a diagonal).
- The P(field) mask is cached by the Zarr's 512-px inner chunk, in memory and
  as packbits on disk (`$TMPDIR/x-sql-marimo/ftw-mask/`), so a pan reads only
  missing chunks. Tile blobs cache under `$TMPDIR/x-sql-marimo/ftw-tiles/`.
- marimo lessons: underscore-prefixed cell locals are mangled (a helper must be
  defined above its use in the same cell; `con.register(...)` explicitly, the
  replacement scan does not see cell locals); every trait assignment from a
  worker thread goes through `loop.call_soon_threadsafe`; an anywidget's CSS
  classes must be prefixed (marimo's Tailwind owns `.hidden`).
- The FTW modes (clip, disagreement) are decided PER TILE ZOOM (`_ftw_zoom_ok`:
  the view box at that zoom under FTW_BOX_DEG2, i.e. tile z >= 12 at CONUS
  latitudes), never per batch box: the first batch of a view is the whole
  view, a pan's batch is two tiles, and the cache key does not record the
  decision, so a per-batch test mixed clipped and unclipped tiles in one view
  (2026-08-21, "just the fields, sometimes everything, some tiles").
- With fields ON, disagreement's orange class (CDL crop, no FTW field) cannot
  appear (the clip is the same grid); the legend says so.

## Pins and the one trap

- `ipywidgets==8.1.8`, `traitlets==5.15.1` are pinned because that is the pair
  the build was verified under (a fresh resolve took 8.1.9 / 5.16.1; not shown
  to break anything, pinned to keep the verified pair).
- A `_NoTMS` experiment (a TileMatrixSet subclass serializing to null, to reach
  lonboard's TMS-less path) was once left in by mistake: tiles were SERVED (the
  status line showed batches) and the JS DISCARDED them (`getTileData` returns
  null without tileMatrices), so the map was blank with no error anywhere. If
  the map is ever blank while the status shows batches, check the TMS first.
- Verified 2026-08-20 night in this venv: TMS with boundingBox + the unlit
  patch -> tile colours equal the reference (255 -> 255, 150 -> 152).

## Open

- Stephen has not flown the tile build with the patch applied. Judge rendering
  by screenshots, never console errors.
- The notebook still opens with the intro of the x-sql-marimo era; the shape
  he wants is register -> the joins as SQL cells -> the map as an output.
- Picking (which dataset says what at a point): geometric in JS, not deck's.
