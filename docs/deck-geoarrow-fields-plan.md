# cdl-aef-deck: the fields as polygons on a deck.gl-geoarrow widget

Planned 2026-08-25 afternoon, EXECUTED the same evening (see the status at
the end). The goal: click a field close in and get its printout, which the
lonboard RasterLayer build does not deliver in Stephen's browser, plus a
selection that can be one field or an area of fields.

## Why leave the lonboard chassis

- deck's GPU picking has never worked under marimo on any chassis tried
  (counties film, crops PolygonLayer, NLCD hexagons, the deck-widget build:
  `onClick` with `layer: null, index: -1`, or a panel that shows once). Every
  working pick was geometric from the click coordinate. lonboard 0.16's
  `Map.on_click` sends only lon/lat as a custom message, no picked object, and
  its callback runs outside a cell run, so it cannot drive the wiring cell.
- The lonboard chassis costs the JS patch (three edits), the model-id
  workaround, and a remove-then-add rebuild per toggle.
- Headless under `marimo edit` (2026-08-25): after a click-triggered cell run
  that does not rebuild the layer, pending tile batches stalled until the next
  interaction (three runs with a basemap click before a wheel zoom served
  nothing for 120 s; one run without the click served z14 in 2.2 s; in the pan
  test the pan's batch appeared only when the next click ran the cell). Not
  pinned to a cause. It fits all three of Stephen's reports (load delay, fields
  cut at old batch seams, a dead click). The click itself works headless:
  gold outline and story at 55 ms, before and after pans.
- Also seen: with the camera at z14 deck asked for the z13 parents as well
  (a 66-tile batch, 8.8 s, on top of the 39-tile z14 batch), placeholder
  tiles the RasterLayer does not let us turn off.

## The chassis

The HRRR counties film's (`x-sql-marimo/xsql-hrrr-counties.py`): a `DeckMap`
anywidget, esm.sh deck.gl 9.3.10 + maplibre with every `?deps` pinned so one
`@deck.gl/core` resolves, geometry as one Arrow IPC bytes trait on a
`GeoArrowPolygonLayer` (`@geoarrow/deck.gl-layers` 0.3.2), per-feature values
as a typed-array bytes trait recolored in JS, `config` JSON kernel -> browser,
and the click picked geometrically in JS on `pointerup` (unproject, bbox
reject, even-odd over the rings the browser holds; a press that moves more
than 4 px or starts on the HUD is not a click). The HUD strip (year, raster,
masks, paints, highlight, outlines, analyze, clear, refresh, search, the
pickable legend) and the status and panel lines stay as they are; only the map
widget changes. `0693f27` is the tile-PNG deck widget of this notebook and
supplies the CDL `TileLayer` over custom messages; it is not the polygon map.

No new Python dependency: pyarrow builds the IPC, the MVT decode exists, no
rasterio (Stephen does not want it), no lonboard on this path.

## Two tiers, as now

- Below tile z13: the CDL raster as a deck `TileLayer`, PNGs rendered by the
  kernel per tile request on the custom-message channel, with the whole-view
  batching moved over from the current `_fetch` (collect a burst for 50 ms,
  serve the whole view from one grid, answer every request). Crops-only and
  the P(field) pyramid clip stay as masks; outlines drawn on the tiles from
  z12 as today.
- From tile z13 (camera ~z12) with a field paint on: the fields as polygons
  from the FTW PMTiles, fetched for the VIEW directly (not through deck's
  tiling): the z13 tiles under the view box by ranged GET, decoded, one Arrow
  table. Measured with the notebook's `ftw_tile_rings` (2026-08-25): a cam
  z12.5 view is 18-24 tiles, ~2,000 rings, ~60k vertices, 71 ms with a warm
  connection, 1.3 s cold including connection setup, 0 ms from the disk cache;
  a cam z11.5 view is 66 tiles, ~7,000 rings, ~200k vertices, 0.2-0.4 s. The
  payload is ~1 MB at z12.5, ~3.5 MB at z11.5 (f64 pairs). The CDL raster is
  not drawn under the fields (kept from the alpha-paint complaint).

## The join stays positional

The field table is unchanged: `ndimage.label` on P(field) >= 0.5 at 10 m over
the batch box, CDL majority and purity by one bincount, last year's majority,
mean AEF vector, the prototype margin score (TAU 0.05), cached by
chunk-aligned box + year. Each polygon gets its field id by sampling the label
grid at a representative point (the ring centroid; if that lands on 0, the
mode of the labels at the vertices nudged 15 m toward the centroid). The
printout is the same `_field_story`, keyed by that id. Per-field rgba from
`field_fill` (CDL, viridis, suggests, the commented alpha paint) becomes a
per-polygon rgba bytes trait, so a paint switch, highlight, or legend isolate
is one small trait, recolored in JS, no tile round trip, no rebuild.

## Picking and selection

- Click: the film's even-odd in JS gives the polygon index; the widget sets a
  `pick` trait `{fid, lon, lat, n}`; the kernel observer writes the story to
  the HUD panel and sets `hit`; the browser draws the gold outline as the
  picked polygons' line color (every piece with that fid). Same field again or
  the basemap clears it, as now.
- An area: once the fields are shapes with ids, a shift-click set or a box is
  a set of ids; the printout for a set is the "analyze what's in view" story
  over that subset (crops by area, agreement distribution, the disagreeing
  fields listed). Not in the first cut.

## The tile-edge problem (open)

tippecanoe clips polygons at the tile boundary (buffer 5/4096, ~6 m at z13),
so a field across two z13 tiles is two pieces, one per tile, and the FTW
PMTiles carry no id. Fills and picking do not care (both go by the sampled
label id). A selection outline would trace the tile edge across a field unless
the segments along the clip line are dropped (the outline reader already does
that for its polylines). Buffer-and-dedupe by id does not fix it: the
neighbour tile holds the other cut piece, not the whole field; only a union of
the pieces gives the whole shape (a dissolve by id, in the kernel).

Routes to whole fields, none chosen:

1. Our own PMTiles from the fiboa parquet, one offline build per state:
   tippecanoe `-pc` (no clipping: every polygon whole in every tile it
   touches; fields are small so the duplication is a few percent) and the
   fiboa id kept. Then fetch the view's tiles, drop duplicate ids, whole fields
   with real FTW ids; the reader stays as it is. The parquet reads happen at
   build time, once. Stephen will not cache the parquet at view time.
2. A boundary tracer over the label image (a page of numpy: edges between
   differing labels chained into rings), whole fields inside the batch box, no
   reads, staircase edges at 10 m (1-2 px at z13). No rasterio.
3. The dissolve of PMTiles pieces by sampled id in the kernel.

## Startup

`HOME` is now zoom 11.5 over the Delta west of Stockton (`cdl-aef-deck.py:269`,
uncommitted): the map opens on the CDL raster at tile z12 in ~1.5 s ("fields
from tile z13 (zoom in)"); one zoom in brings the fields. Verified headless.
The default paint stays "color by agreement".

## Verification

Headless `uv run marimo edit cdl-aef-deck.py --headless --port N --no-token` +
playwright chromium (run-all at (1558, 924) in 1600x1000; HUD text is in
shadow roots; only Chromium is installed, no WebKit/Firefox), then in
Stephen's browser before claiming anything: click a field, printout; pan,
click again; paint switch with no reload; zoom out to the raster and back.

## Order of work, when it is executed

1. `DeckMap` widget: the film's boot (pinned esm.sh graph), the CDL
   `TileLayer` on custom messages with the whole-view batching, `cam` trait on
   moveend, `config` trait.
2. Kernel: polygons for the view from `ftw_tile_rings` kept as closed rings
   with holes (MVT winding), sampled to ids, Arrow IPC (geoarrow polygon,
   interleaved f64) + rgba bytes.
3. `GeoArrowPolygonLayer` in the widget, recolor on `colors` change, gold line
   on `hit`.
4. JS pick on pointerup -> `pick` trait -> kernel story -> panel.
5. HUD acts rerouted (paint/highlight/isolate recolor; year/raster/masks
   re-serve); search; refresh.
6. Headless drive, then Stephen's browser.
7. The tile-edge route (1, 2 or 3 above) once chosen.

## Status (2026-08-25 evening)

Steps 1-6 are in `cdl-aef-deck.py` (one commit's worth of change on the
lonboard build, verified headless with playwright; not yet in Stephen's
browser). Departures from the plan above:

- The polygons are fetched for the PADDED view (PAD 1.15), the same box as
  the field table, and reused while the view stays inside it (contains
  check), rather than per view.
- The rgba paint travels WITH the geometry (an `rgba` column in the IPC
  table) so the first draw has its colors; the `colors` bytes trait only
  carries repaints.
- Every ring is clipped to its tile box (Sutherland-Hodgman in tile coords):
  tippecanoe's buffer (5/256 of the tile, ~100 m at z13, not 5/4096) had put
  the same strip of every field in two tiles and the doubled alpha drew as
  dark bands across the fields.
- The outlines are a second IPC table (paths) on a PathLayer, the existing
  polyline decode with the clip-line segments dropped, so tile edges do not
  draw as outlines.
- The JS owns the hit (gold PathLayer on every piece with the fid) and tells
  the kernel; the kernel only writes the story.
- The CDL block reads in `cdl_window` run in parallel (8 threads).

Step 7 (whole fields across tile edges) is still open; the gold outline of a
field cut by a z13 tile edge shows the seam. The area selection is not
started.
