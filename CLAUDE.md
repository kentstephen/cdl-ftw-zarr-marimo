# CLAUDE.md

Guidance for Claude Code in this repository. Inherits the global rules in
`~/CLAUDE.md` (tone, no em dashes, memory in `.claude/memory/` here,
colorblind-safe encodings: Stephen has trouble seeing red).

## What this is

`cdl-ftw.py`, one marimo notebook (xarray + numpy + lonboard; the per-field
joins as DuckDB SQL are `cdl-ftw-sql.py`, local and gitignored for now): USDA
Cropland Data Layer (icechunk Zarr v3,
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
  Python prop reaches it; without it the colors are wrong (a protan-safe
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
  the ~10 s stall on a slow link, gone from the map (raster clip + PMTiles
  outlines); the SQL notebook's `cache_httpfs` makes repeats local there.
- The PMTiles outlines: tippecanoe z0-13, layers "2024" / "2025", no id
  (draw-only); the reader is the HRRR counties film's PMTiles v3 + MVT decode
  by copy; segments along a tile's clip line are dropped (no seams) and
  polylines are NOT closed when drawn (a cut piece closed itself with a diagonal).
- The P(field) mask is cached by the Zarr's 512-px inner chunk, in memory and
  as packbits on disk (`$TMPDIR/x-sql-marimo/ftw-mask/`), so a pan reads only
  missing chunks. Tile blobs cache under `$TMPDIR/x-sql-marimo/ftw-tiles/`.
- marimo lessons, the 2026-08-24 night set (all found by driving the real
  frontend with playwright; NONE are visible headless): a widget TRAIT update
  from a worker thread (even via call_soon_threadsafe) syncs kernel-side but
  NEVER reaches a live frontend (the in-place polygon table drew once at
  startup, then every pan lost the fills) -> store the data, poke a HUD trait,
  let the HUD's JS answer through ctl, do the assignment in that cell run;
  creating ANY widget mid-session (even in a ctl-triggered run) leaves the
  frontend with "Model not found for key" and the whole deck goes blank ->
  every layer is created in the map cell and lives forever, only traits are
  assigned; TWO vector layers collide on deck id "undefined-0" (model_id is
  undefined under marimo; the JS patch fixed raster ids only) and deck asserts
  -> one vector layer, selection is a color, not a layer. Frontend verification:
  `uv run marimo edit <nb> --headless --port N --no-token` + playwright
  (chromium installed), click run-all at (1558, 924) in a 1600x1000 viewport,
  wait ~75 s cold, screenshot the big canvas; HUD text lives in a shadow root
  (page.inner_text misses it, walk shadowRoots in JS); a second page connecting
  to the session shows a blank deck (reconnect artifact, not a bug).
- marimo lessons: underscore-prefixed cell locals are mangled and dropped
  after the run unless a closure's reference is seen (a helper must be defined
  above its use in the same cell; forward references are not kept); every trait
  assignment from a worker thread goes through `loop.call_soon_threadsafe`; an
  anywidget's CSS classes must be prefixed (marimo's Tailwind owns `.hidden`);
  widget comms are bound to the session stream of the run that opened them,
  so a widget created from a background task never reaches the frontend.
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
  patch -> tile colors equal the reference (255 -> 255, 150 -> 152).

## Controls added 2026-08-21 evening

- HUD `refresh` button (act "refresh"): `HOLD["layer_state"] = None` in the run,
  i.e. a rebuild like a toggle, the escape hatch if deck stalls. `TILE_ZMIN` 3.
  The camera is clamped to EXTENT + 2 deg in `_on_vs` via `deck.set_view_state`
  (guarded by HOLD["clamping"]). Stephen, 2026-08-21 evening.
- With fields on below ~z9 the coarse P(field) (64x / 256x, >= 0.5) keeps almost
  nothing (z6: 55 px drawn of 6.8 M) and the batch is slow (1.5 s): "clip at
  every zoom" is a blank at low zoom. Not changed; his call (a clip floor was
  offered).


## cdl-aef-deck.py (branch cdl-aef-deck, 2026-08-25)

- The NLCD deck notebook (`x-sql-marimo/xsql-aef-nlcd-deck.py`) forked onto
  the CDL and the FTW fields, WITHOUT the hexagons (Stephen: zoomed out you
  see the raster, the hexagon folds were the slow part, "I don't necessarily
  know if I wanna see them"). So: no H3, no DataFusion, no DuckDB; xarray +
  numpy + scipy + pyarrow, runs from this repo's venv. lonboard, morecantile
  and the JS patch are OUT of this notebook (cdl-ftw.py still uses them).
- CHASSIS (third, 2026-08-25 evening, the plan in
  `docs/deck-geoarrow-fields-plan.md`, executed): a `DeckMap` anywidget, the
  HRRR counties film's pinned esm.sh graph (deck.gl 9.3.10 + maplibre 5.24,
  every `?deps` per package identical so ONE @deck.gl/core resolves) with the
  0693f27 build's boot (Carto Positron, interleaved MapboxOverlay, layers
  under `watername_ocean`). Traits: `config` JSON kernel -> browser; `fields`
  / `lines` / `colors` bytes kernel -> browser; `view` (moveend, lon/lat/zoom
  + canvas w/h) and `pick` browser -> kernel; custom messages `tile` (PNG
  back) and `fly`. The HUD strip is unchanged except its canvas click is
  gone (the map picks). History: lonboard RasterLayer build e82a293..3448d2b
  (Stephen's three reports on it: load delay, fields clipped at batch seams,
  lost picking; the seam and the pick are structural there, see the plan);
  the first deck widget with two TileLayers b970d50..0693f27.
- THREE INDEPENDENT LAYERS (Stephen, 2026-08-26: the controls are "a bit
  wonky", "should be able to disable all layers"; "We can keep the polygons. I
  just wanna get rid of the field outlines if I don't want them. I can just go
  to the raster if I wanna look at that and filter for crops only. It's a
  separate product."). The CDL raster (its own switch, crops-only its
  modifier), the painted polygons (the paint buttons: none selected = none),
  the field outlines (its own switch, ctl key `outlines`, was `fields`).
  RETIRED with the one-switch design below: the raster's clip to P(field)
  (`_clip = False`, still threaded through the serve, one line to bring back)
  and the `under` config (the raster now draws under the polygons whenever its
  own switch is on; the JS is `visible: cfg.raster !== false || (cfg.outlines
  !== false && !fieldsOn)`). The outlines are drawn ONCE: the PathLayer when a
  paint is up, the PIL polylines on the raster tiles when it is not, which is
  `_rings_on = _outlines and _paint is None` inside `_rstate` (so toggling the
  outlines under a paint costs no tile refetch). Every combination is now
  reachable, including all-off (basemap only) and outlines with no raster.
  A FIELD PAINT IS THE FIELDS ALONE (Stephen, 2026-08-26, on a shot of
  crops-only CDL + the viridis paint + outlines: "what am i looking at ...
  perceptually baffling", then "if we're looking at agreement we are only
  looking at the intersecting data that is the cdl and the ae in the field
  boundaries ... agreement is only fields for this use case subject to
  change"). Three encodings at full strength in one frame (CDL class hues on
  the raster pixels BETWEEN the fields, viridis on the fields, silver
  boundaries) do not resolve. So `cfg.raster_dim` (map config, DEFAULT 0):
  under the polygons the TileLayer draws at that opacity, and at 0 it is not
  drawn at all (`rasterOn` in the JS), so with a paint up the CDL raster
  switch has no visible effect until you drop the paint. Raise it for a
  backdrop, 1 for the old full strength.
- COLLAPSE / EXPAND (Stephen, 2026-08-26, after two rounds of placement: the
  collapse caret "needs to be at the top right not in the middle ... on the
  same line all the way over to the right as year and the other top params",
  the strip "needs to disapear entirely", and the expand "arrow up should be
  visible above the carto credit"). So: `topRow` (flex 1 1 100 %) holds year,
  the layer switches and the paints with the caret at `margin-left:auto`, and
  analyze and the legend wrap below it; collapse sets `wrap.style.display =
  "none"` (the whole strip, not its parts) and `onFs` re-applies it after its
  cssText rewrites. The expand arrow is appended INTO the map container
  (`deepFind(".maplibregl-map")` across shadow roots, polled every 400 ms for
  ~24 s because the map is another widget, then a fixed bottom-right corner
  fallback), absolute at right 8 / bottom 52 px, which clears the Carto info
  circle it first overlapped; a WHITE arrow (opacity .9, white border, faint
  dark fill) so it reads on the dark basemap. It carries class
  `maplibregl-ctrl` so the map's pointerup pick skips it. All client-side: no ctl, so no kernel run and no re-serve. The
  stale-widget sweep takes it with `[data-aef-expand]`.
- ONE ZOOM RULE FOR THE FIELDS, `_field_floor(vsd)` (Stephen, 2026-08-26:
  "layers get uneselected like color by agreement for no noticable reason.
  check the logic"). FIELD_ZOOM and FIELD_MAX_KM2 were two knobs that
  disagreed: the padded box at camera z11 is 1,170 km2 on a 1400x700 canvas
  (under the 1,500 cap) but 2,383 fullscreen at 2000x1000 and 3,965 at
  2560x1300, so at his fullscreen z11 the paint stayed lit while the area cap
  silently dropped the fields. `_field_floor` returns FIELD_ZOOM, or the zoom
  where the padded box fits the cap when that is higher (the box quarters per
  zoom step: `z + 0.5*log2(km2/cap)`, ceiled to 0.1); `_serve_fields`,
  `_raster_line`, the pick and analyze all speak that ONE number, and the
  separate area branch in `_serve_fields` is gone with it.
- CROPS ONLY IS ON BY DEFAULT (Stephen, 2026-08-26), the kernel default and
  the strip default both. Knock-on he hit at once: with no polygons up
  the map falls back to the CDL raster, and crops-only makes that read as
  crop-colored FIELDS while a paint button is still lit ("you made a change,
  which is confusing color by agreement shows cdl fields"). So `_fields_off`
  now takes a `note` and pushes it through `cfg.note` to the map's own
  top-left overlay ("this is the CDL raster · the field paint needs camera
  zX"), cleared whenever the polygons come on. The strip's status line said it
  already; nobody reads under the map.
- SIMPLIFY FOR VIZ (Stephen, 2026-08-26: the boundaries "look like shit",
  "they're not straight lines", "looks like pixelation thing"). Measured over
  6 z13 tiles in the Delta: 17,142 outline segments, 100 % axis-aligned,
  median 10.3 m, i.e. FTW's 10 m raster vectorized into a staircase. So
  `SIMPLIFY_M = 10.0` and Douglas-Peucker (`_dp`, iterative, both ends pinned;
  `_dp_ring` for the closed rings) at MVT DECODE time in tile units
  (`_eps_units` from the tile row's latitude), on the outline polylines and
  the polygon rings alike, so it is one pass per tile and cached with it.
  0 disables. NOT yet measured: the vertex reduction and the serve cost.
- ONE FIELDS SWITCH (Stephen, 2026-08-25 evening: "just keep one button for
  field boundaries ... have it selected ... when you zoom in, it clips to
  the fields, but it can be unselected ... but that's only past z14,
  otherwise we see all of CDL unless crops are masked"). "fields clip" and
  "outlines" are gone. Below camera FIELD_ZOOM (14): the raw CDL, crops-only
  optional, nothing field-related (the old clip-at-every-zoom blanks are
  gone with it). From z14, switch ON = the raster clipped to P(field) + the
  outlines (no paint) or the painted polygons over the basemap (a paint);
  switch OFF = the raw CDL, drawn UNDER the painted polygons too ("maybe
  there's disconnect with the fields and what alpha earth thinks is there").
  Kernel: `_fields` -> `_clip = _outlines`, gated in `_serve_batch` on tile
  z >= FIELD_TILE_Z (15 = camera 14; tile z = round(camera + 1) with 256-px
  tiles, which is why the status line's z runs one ahead of the camera);
  config `under` shows the TileLayer beneath the polygons.
- Two tiers by CAMERA zoom. Below FIELD_ZOOM (14; z12 and z13 "gave no
  context for most fields of what it could be"): the CDL raster as a deck
  `TileLayer` (`cdl-<rgen>`; a new rgen on a raster state change makes deck
  refetch) whose PNGs the kernel renders ONE BATCH PER VIEW (cdl-ftw's
  `_fetch`: the first request waits BATCH_S, the batch takes the whole view's
  tiles at that zoom, one grid, PNGs cached by `(rstate, z, x, y)`), the
  legend from the batch's class counts. deck fetches the HIDDEN raster
  layer's tiles too, so a raster state change may find every tile cached
  and produce no batch: the status line is composed at display time
  (`_raster_line`: the cached bare line + why the fields are off) and said
  on every raster state change. From camera z14 with a
  field paint on: the FIELDS ARE POLYGONS. `_serve_fields` (an asyncio task,
  settle-debounced SETTLE 0.35 s, coalescing): the field table over the view
  padded by PAD 1.15 (aef-agreement's positional join, unchanged), the FTW
  PMTiles' z13 tiles under that box decoded to closed rings with holes
  (`ftw_tile_polys`: MVT winding, exterior = positive area in tile coords;
  every ring CLIPPED to the tile box, `_clip_ring`, because tippecanoe's
  buffer put ~100 m of every piece in the neighbour tile too and two fills
  read as dark bands), each polygon keyed to its field id by sampling the
  label grid at its centroid (`poly_fids`, vertex-mode fallback), one Arrow
  IPC table (`polys_ipc`: geoarrow.polygon interleaved f64 + fid int32 +
  rgba) on a `GeoArrowPolygonLayer`, the outline polylines a second IPC
  table on a PathLayer. A pan INSIDE the padded box costs nothing (contains
  check); leaving it re-serves (table cached by chunk-aligned box, tiles
  from memory/disk). The CDL TileLayer is hidden while the fields draw
  (unless the fields switch is off).
- HOME is Stephen's box (`HOME_BOX`, Bethel Island to Stockton, from
  boundingbox.klokantech.com) fitted to the canvas: camera z10.71. THEN
  (2026-08-25 night, "i want to start here", a screenshot at tile z11 =
  camera ~z10, the whole Delta from Rio Vista and Antioch to Lodi and
  Stockton, wider than HOME_BOX): the START VIEW IS THE RASTER, that is all
  he asked ("no i asked for the new starting view to be raster and it is");
  nothing is folded there, no AEF cost. Not done (no coding asked): a
  HOME_BOX for that vantage, roughly -121.95..-121.05, 37.75..38.35. He set
  FIELD_ZOOM = 11 himself; a camera-z11 view (~1,300 km2, ~1,700 padded) is
  ~1 GB of AEF cold and sits just above FIELD_MAX_KM2 1500 (2,000 lets it
  through; offered, not done). Do not conflate the two again. RESOLVED
  2026-08-26 by `_field_floor()` below: the two knobs no longer disagree.
- PLAN LOGGED (2026-08-25 night, not built): `docs/wide-prototypes-plan.md`,
  prototypes fitted ONCE at the raster tier from a one-in-nine sample of the
  AEF chunks under the wide view (the CDL is already in hand there), held in
  HOLD, applied to every field serve inside the fit box, refit elsewhere;
  Stephen's "we get crop data before zoom in". Earth Index (Earth Genome, 320 m
  patches) is noted there as a side experiment, not a paint.
- Open question he raised: the prototypes are per served box (view + 15 %),
  so at z14 few crops reach 20 fields; "looking at an area this large would
  be helpful for aef context". Directions offered, none chosen: a fixed
  context box (HOME_BOX) scored once; a much wider PAD at high zoom; a
  "score this area" button. The draw zoom and the context box are coupled
  today.
- Paint switch / highlight / legend isolate = `_recolor`: rgba LUT by field
  id -> one bytes trait of N x 4, recolored in JS, no geometry reload, no
  round trip for tiles. Year change and refresh = a forced re-serve.
- THE PICK is in the browser: pointerup (a press on a map control or one
  that moves > 4 px is not a click), `map.unproject`, bbox reject, even-odd
  over the rings the browser holds -> the polygon's fid; the JS toggles the
  gold PathLayer on every piece with that fid itself, then sets `pick`
  ({i, fid, on, lon, lat, gen}); the kernel's observer writes the story to
  the panel (ignores a pick whose gen is not the current table's). deck's
  GPU picking is not used anywhere (never worked under marimo).
- Acts are applied ONCE per ctl (`n` compared to HOLD["ctl_n"]): a re-run
  of the wiring cell for any other reason does not repeat the last click.
- Verified headless 2026-08-25 evening (`marimo edit --headless` + playwright,
  the Delta at cam z12.6): cold field serve 1.8-2.6 s (table 1.7 s of it:
  cdl 1.1 with the parallel block reads, aef 264 MB from disk 0.1), 973
  polygons 0.9 MB; click -> story + gold in ~1 s; paint/highlight/outlines
  instant; a pan out of the box 0.6 s; year 2023 1.1 s; zoom out to the
  raster and back 0.9 s; no console errors on the last run ("Model not found
  for key" x2 appeared once on an earlier run-all with no visible effect).
  NOT yet run in Stephen's browser: his `uv run marimo edit cdl-aef-deck.py`
  kernel must be restarted to pick this up.
- HIGHLIGHT DISAGREEMENT BELONGS TO ONE PAINT (Stephen, 2026-08-26: "the
  logic of the control panel is fucked. highlight disagreement should not
  carry over to cdl. the selection for highlight disagreement should be next
  to color by agreement"). It now sits immediately after that button in the
  row (`paintKids`, not appended past "AlphaEarth suggests"), `stylePaint()`
  shows it only while that paint is on and unchecks it otherwise (hidden, not
  greyed: his standing rule), and the kernel enforces the same at the source,
  `_inv = ... and _paint == "viridis"`. Before, only the viridis branch of
  `field_fill` read `inv` but the checkbox stayed checked across paint
  switches, which is the carry-over he saw.
- The expand arrow is a WHITE button (white fill, dark glyph, thin dark border
  and a soft halo, full opacity), matching the Carto info control under it;
  the earlier dark-fill/white-glyph version was "hard to see" on the map.
- OPEN, his words, no direction chosen: "the crops only button is confusing".
  Two readings were put to him and he did not pick: (1) it reads as a LAYER
  sitting between "CDL raster" and "field outlines" when it is only a modifier
  of the raster (the polygons are crop fields already), so nest and rename it;
  (2) on Dark Matter it punches the non-crop classes to transparent, so
  pasture, water and developed read as MISSING DATA rather than "not a crop",
  so draw them as one muted tone instead. Do not guess: ask.
- NOT VERIFIED IN A BROWSER (the whole 2026-08-26 set above: the suggests
  paint, the silver outlines, DP simplify, the three independent layers, the
  raster_dim rule, collapse/expand, `_field_floor`, crops-only default). What
  WAS checked: the file parses, both `_esm` modules pass `node --check`, the
  suggests paint and the outline geometry were measured headless (`app.run()`
  from a scratch copy, then `ftw_tile_rings` / `field_table` directly), and an
  earlier playwright pass covered the suggests paint and the silver outlines
  only. Stephen's own kernel needs a restart to pick any of it up. He asked
  NOT to spin up a headless notebook for small changes ("you dont have to
  start a headless notebook for these kinds of changes").
- TODO (Stephen, 2026-08-25 night, not now): on a zoom in with the fields
  switch on, the fields JUDDER from CDL colors to the agreement paint: the
  clipped CDL raster tiles (the raster tier at the new tile zoom) show first,
  then the polygons replace them. "Should just render as agreement": the
  raster must not draw its field-tier tiles while a paint is on and the
  polygons are coming (hide the TileLayer from FIELD_ZOOM whenever a paint
  is on, or keep the last polygon layer up until the new table lands). Also
  from him: the toggle between the layers (raster tier <-> field tier, and
  the paint buttons) is JUMPY and sometimes REVERTS (a switch lands, then
  the previous state comes back). Not reproduced headless; suspects: the
  settle-debounced serve racing a ctl run (`_kick` vs `_refresh` with
  HOLD["pending"]), a stale `last ctl` re-initialising the strip, or a
  raster batch's `_push` landing after `_fields_off`. Reproduce in HIS
  browser first.
- TODO (Stephen, 2026-08-25 night, not now): a better field-boundary STROKE
  for disagreement, "maybe the same color as the boundary": the disagreeing
  fields (agreement < 0.5) outlined with a heavier stroke in the outline
  color rather than, or as well as, the fill ramp. The outlines are a
  PathLayer of the PMTiles polylines with no field id today, so the stroke
  would come from the polygon layer (stroked, getLineColor / getLineWidth
  per polygon from the fid, the tile-clip segments then visible as lines
  across the cut fields) or from a per-fid line table built with the fills.
  Colorblind rule: the stroke must not be a red-vs-green distinction.
- Open on this chassis: the tile-edge pieces (a field across two z13 tiles is
  two polygons with one fid; fills and the pick do not care; the gold outline
  shows the seam as a line across the field), three routes in the plan doc,
  none chosen. A cold region is the AEF read (Champaign: 12 s of a 16 s
  table for 0.02 deg^2). Not done: clusters, the three-voter categorical
  paint, a search zoom floor, an area selection (shift-click set or box).
- Paints, join, strip lessons unchanged from before: CDL / "color by
  agreement" (viridis, bright = agrees, the default) / "AlphaEarth suggests";
  the alpha "agreement" paint stays commented out; no greyed-out buttons;
  gold outline not a white fill; color not colour; the strip initialises
  from the kernel's last ctl.
- "ALPHAEARTH SUGGESTS" IS NOW A CROP MAP (Stephen, 2026-08-26: "the palette
  should be what the CDL is, except it can be a null color if it doesn't have
  that. And it looks like it's all null now"). It used to color ONLY the
  disagreeing fields (the runner-up crop) and paint every agreeing field flat
  grey, so a view where AEF mostly agrees read as all-null. Now `aef_best(ft)`
  gives every field the crop AEF puts it closest to (its own where agreement
  >= 0.5, the runner-up where not, -1 where the field sits out or its crop has
  no prototype in view) and the paint is `CLASS_RGB[best]` at ALPHA_FLAT, the
  null grey only where best < 0. The legend lists the suggested crops with
  "N against CDL" per crop and a "no suggestion" row; the legend isolate and
  the selection panel key off the same array. NOT touched: the CDL palette
  itself, which is the store's own colors except three protan remaps (Cotton
  #FF2525, Apples #B9004F, Dbl Crop Lettuce/Cotton), so grapes are #6F4488.
  Measured over the Delta box at camera z12.6: 860 kept fields, 756 colored
  (657 their own crop, 99 another), 104 null; at z13 in the browser 2,765
  fields, 131 "no suggestion".
- Note for later, not changed: the CDL palette gives Cucumbers, Garlic,
  Cauliflower, Cabbage, Lettuce, Carrots, Broccoli and Watermelons the SAME
  hex #FF6666, which is both indistinguishable between them and red (it clears
  the protan remap only because g = 102 > the 100 cutoff).
- The field OUTLINE is CLASSIC SILVER (192, 192, 192, 235), Stephen
  2026-08-26; it was near-black (40, 40, 40, 200), then briefly a cool
  (206, 210, 216) that read as grey to him. One constant kernel-side (the PIL
  polylines on the clipped raster tiles) and one in the widget JS (the
  PathLayer over the polygons); both must move together.
- THE BASEMAP IS CARTO DARK MATTER (Stephen changed it himself in the JS,
  2026-08-26); it was Positron. Dark Matter carries `watername_ocean`, so
  LABELS_SLOT and the JS `before()` slot are unchanged and the deck layers
  still draw under the labels. Knock-on flagged, not changed: the null grey
  QUIET (150, 150, 150) at alpha 45 was tuned against near-white and is
  nearly invisible on dark.

## Open

- Speed on cdl-ftw.py (lonboard): the per-tile widget round trip (45
  messages, 6 in flight) is the floor on every state change; 512 px tiles
  would cut that by 4. The low-zoom clip floor is undecided.
- cdl-ftw.py: one layer for the Map's life with a reload trigger instead of a
  rebuild per toggle would remove the remove/add flash; not done (cdl-aef-deck
  has that: `rgen` in the TileLayer id).
- Judge rendering by screenshots, never console errors; the status line's ms
  is the serve time, not the browser fill.
- Picking (which dataset says what at a point): geometric in JS, not deck's.
