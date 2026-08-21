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
- **No DuckDB in the map notebook** (2026-08-21, Stephen: "i want to run xarray
  numpy and lonboard"). The per-field joins as SQL (the fiboa parquet through
  httpfs + `cache_httpfs`, `ST_Contains`, per-field crop / purity, the 2x2,
  `xarray-sql` exposing the CDL 10 m levels and `ftw_4` as tables) live in
  `cdl-ftw-sql.py`, which is GITIGNORED and out of the repo for now (his call);
  it carries its own inline deps, run it with `--sandbox`. `duckdb` and
  `xarray-sql` are out of pyproject.
- The map is **tiles**: lonboard `RasterLayer` (deck TileLayer), `_fetch_tile`
  batches a burst of requests (BATCH_S 0.05), serves the WHOLE VIEW per batch
  (deck caps in-flight tile requests at 6), cuts the PNGs per tile from one
  grid, caches tiles in memory (TILE_CACHE). A state change (year, checkboxes,
  crops-only, selection) rebuilds the layer, REMOVE THEN ADD via `deck.layers`
  (the lonboard JS patch gives each raster layer its own deck id; without it
  every layer under marimo is deck layer "undefined", a replacement reads as
  an update and it keeps its loaded tiles: the old state stayed on screen in
  bands, 2026-08-21 night, with remove-then-add in one run not reaching deck
  as two steps). The TMS must carry
  a `boundingBox` (morecantile's stock WebMercatorQuad lacks one); the TMS-less
  path in lonboard 0.16 is dead code (`getTileData` returns null).
- **The lonboard JS patch is REQUIRED and the notebook applies it itself**: the
  first cell runs `tools/patch_lonboard_raster_unlit.py` (three replacements)
  in whatever environment is executing the notebook, before the Map is created
  (anywidget reads the JS into the Map's `_esm` at creation). THE TRAP OF
  2026-08-21: Stephen runs `uv run marimo edit cdl-ftw.py --sandbox`; the sandbox
  is a fresh uv env from the inline deps, which had `lonboard>=0.16.0` with no
  cap, so it resolved lonboard 0.17.0b1, unpatched, while every headless check
  ran in the patched `.venv` (0.16). Stale tiles on every toggle, blank after a
  flight, 10 s drops: all of it was the unpatched JS in his kernel. Inline deps
  now pin `lonboard>=0.16.0,<0.17` + ipywidgets/traitlets like pyproject. Before
  theorising about a session, `ps -axo command | grep marimo` and look at the
  interpreter path. The three edits: (1) the tile mesh fragment shader calls
  `lighting_getLightColor`, ~0.69x on every channel, `opacity` ignored, no
  Python prop reaches it; without it the colours are wrong (a protan-safe
  palette drawn dark). (2) `getTileData` gives the kernel TEN SECONDS per tile
  request (`timeout:1e4`); past that the JS drops the tile and deck never asks
  again, so a batch over 10 s (a fly-to into a cold region) left the map blank
  until a param change rebuilt the layer (2026-08-21, Champaign). Raised to
  120 s. Keep batches short anyway: the status line's ms is the number. (3)
  the raster layer's deck id is `${this.model.model_id}`, undefined under
  marimo, so every RasterLayer was deck layer "undefined" and a rebuild kept
  the old tiles; patched to a per-instance id when model_id is missing.

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
- The FTW modes (clip, disagreement) work at EVERY zoom: the mask picks the
  coarsest pyramid level within 4/3 of the CDL pixel served (`FTW_LEVELS` 4,
  16, 64, 256 = 40 m .. 2.56 km; all share the origin and 512-px chunks). The
  old 0.35 deg^2 cap ("zoom in for FTW") is gone: it was first applied per
  batch (the whole-view first batch off, a pan's small batch on, one cache key,
  so a view mixed clipped and unclipped tiles), then per tile zoom, and even
  then deck's placeholder tiles (refinementStrategy best-available, not
  exposed by lonboard) flashed the unclipped low-zoom tiles while panning.
- The batch future always resolves (`_run_batch` try/except, CancelledError
  included): a batch neither closed nor resolved would collect every later
  request of its zoom forever. Nothing else is speculative on the serve: the
  data lands in ~0.5 s anywhere in CONUS (cold Champaign CDL read measured),
  so no retry, no heal, no timeouts (a day's worth of those was removed).
- The search runs IN THE CELL RUN, like a toggle: Photon synchronously, camera,
  `HOLD["layer_state"] = None` so the layer is rebuilt in the same run. As a
  background task after the run (fly_to, sleep, rebuild) the frontend never
  got the layer ("Model not found for key", empty map until a toggle).
- A layer is always a NEW `RasterLayer` (`_make_raster` / `_rebuild`): under
  marimo a layer removed from `deck.layers` is closed; re-adding the same
  object draws nothing. `_make_raster` takes `_fetch` / `_render` from HOLD:
  it sits above them in the cell and marimo drops underscore temporaries a
  forward reference does not keep ("NameError: _cell_..._fetch").
- Outlines only from tile z12 (`OUTLINE_ZMIN`): a z5 outline tile holds a
  state's every field; with the clip at every zoom the outlines must not be.
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

- HUD `refresh` button (act "refresh"): `HOLD["layer_state"] = None` in the run,
  i.e. a rebuild like a toggle, the escape hatch if deck stalls. `TILE_ZMIN` 3.
  The camera is clamped to EXTENT + 2 deg in `_on_vs` via `deck.set_view_state`
  (guarded by HOLD["clamping"]). Stephen, 2026-08-21 evening.
- With fields on below ~z9 the coarse P(field) (64x / 256x, >= 0.5) keeps almost
  nothing (z6: 55 px drawn of 6.8 M) and the batch is slow (1.5 s): "clip at
  every zoom" is a blank at low zoom. Not changed; his call (a clip floor was
  offered).

## Open

- Stephen has not flown the tile build with the patch applied. Judge rendering
  by screenshots, never console errors.
- The notebook still opens with the intro of the x-sql-marimo era; the shape
  he wants is register -> the joins as SQL cells -> the map as an output.
- Picking (which dataset says what at a point): geometric in JS, not deck's.
