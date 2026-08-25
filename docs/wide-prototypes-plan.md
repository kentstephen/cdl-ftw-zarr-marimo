# Prototypes fitted at the raster tier (plan, 2026-08-25 night)

Logged, not built. Stephen's observation: "we get crop data before zoom in".

## The problem

The agreement paint scores each field against per-crop PROTOTYPES (the mean
of a crop's field vectors, `score` in cdl-aef-deck.py) fitted from the served
box (view + 15 %). At camera z14 the box is small: most crops have fewer than
MIN_CLASS_FIELDS (20) fields there and are unscored; the ones that pass are
built from a handful of neighbours; a zoom or a pan changes the training set,
so the numbers and the "AlphaEarth suggests" column move with the camera, not
with the model. (docs/aef-earth-genome-notes.md, "fly-around": accuracy
tracks the viewport; fit once, predict in the viewport.)

## The plan

Fit the prototypes ONCE from a wide view, keep them fixed while zooming in.
The wide fit is cheap at the raster tier because the CDL for the whole view
is already in hand (the raster batch's grid and class counts, i.e. the
legend), and a prototype needs a few hundred samples per crop, not every
pixel: sample the AEF chunks instead of reading them all.

1. Split `score` into fit (box -> P, classes, counts per class) and apply
   (V, P -> agree / alt / own / oth). Hold the fit in HOLD["protos"]:
   P (~30 x 64 f32), the classes, counts, the fit box, year, the chunk list.
2. The fit task, after a raster batch lands at camera z < FIELD_ZOOM:
   settle-debounced and coalescing like `_serve_fields`. Take the AEF chunks
   (ACH 256 px = 2.56 km, 4 MB each) under the view, keep one in nine (every
   3rd in x and y; ~31 of ~260 at the z10.7 Delta vantage, ~120 MB cold,
   3-6 s; from disk under 1 s). On those chunks only: FTW mask, label,
   CDL majority, mean AEF vector per field (field_table's steps on a sparse
   chunk set, the same code), then the per-crop mean of field means.
   FIELD-LEVEL prototypes, so "agreement" keeps today's definition.
   MIN_CLASS_FIELDS stays 20, now applied to the wide sample.
3. `_serve_fields`: if the served box is inside the fit box and the year
   matches, apply the wide prototypes; otherwise fit from the current box
   (today's behaviour) and, on the next raster batch elsewhere, refit there.
   No pinned HOME_BOX, no "fit here" button.
4. The status line names the fit: "prototypes: 23 crops · z10.7 view · 31
   chunks". The panel's "suggests" column and the story read from the table
   as today (apply writes agree / alt / own / oth into ft in place).
5. Verify headless: the Delta start view fits before any zoom; a zoom-in to
   an island scores the six-field crops; a pan to the next island keeps the
   numbers; Champaign refits.

Build estimate: 2-3 h kernel-side, no JS. FIELD_MAX_KM2 does not move.

## The tradeoff (accepted)

A Delta-wide prototype asks "does this field look like this crop as grown in
the Delta", the local fit asked "as grown on this island". A correctly
labelled field in an odd corner can read as a disagreement under the wide
fit. The status line says which fit is on. The local scorer can return later
as a second scorer over the same table.

## Not in this plan

- kNN over the kept sample vectors instead of nearest mean (the note's
  suggestion; ~4 MB of samples, drops in behind apply later).
- Earth Index embeddings (source.coop earthgenome/earthindexembeddings:
  SoftCon ViT-S/14, 384-D, one vector per 320 m Major TOM patch, GeoParquet
  points by UTM tile and year). Cheap per km2 (~1000x less than AEF) but a
  320 m patch straddles any field under ~15 ha; a side experiment, not a
  paint, and its point-in-polygon is DuckDB work (cdl-ftw-sql.py or a
  script, not the map notebook). See docs/aef-earth-genome-notes.md.

## Other rulers (discussed 2026-08-25 night, none chosen)

- NLCD in the Delta is three or four classes (cultivated crops, pasture/hay,
  herbaceous wetland, water): nothing to say inside a field; its ag classes
  have drawn on CDL, so not an independent ruler on farmland. Its place is
  the control row of the note's 2x2 on a mixed landscape (built / forest /
  water), or a conversion map (pasture and wetland called crop).
- A second field-level ruler for the Delta: DWR / Land IQ Statewide Crop
  Mapping (field polygons with crop type, ground-verified, independent of
  CDL, 2014-2023, CNRA open data). California Pesticide Use Reports (crop by
  PLSS section) as a coarser third.
- CDL confidence layer (0-100 per pixel, NASS, on the CDL grid): "CDL was
  unsure here" vs "AEF disagrees where CDL was sure". NOT in the chill
  icechunk store (both groups carry `crop_type` only, checked 2026-08-25);
  reachable as the `confidence` band of Earth Engine's `USDA/NASS/CDL` or as
  NASS's separate per-year zip; 10 m-era (2024+) availability unchecked.
  Per-field purity in `field_table` is a partial proxy already. Hosting if
  taken: the Delta window is ~2 MB/year, a local file in the cache dir; a
  region as a COG in a bucket of Stephen's; CONUS as a `confidence` variable
  upstream in chill's store (processing code colinahill/usda_cropland_data).
- Others: ESA WorldCereal (10 m, thin class set, global); AAFC Annual Crop
  Inventory (Canada); Dynamic World / Esri IO / WorldCover (Sentinel
  classifiers, partly circular with AEF); OpenET (field-level ET, a
  continuous target AEF may encode that CDL has no word for); gSSURGO soils
  (stationary, a candidate for the sub-region effect on prototypes); FSA
  CLU boundaries (not public since 2008).
- The Fused CDL UDF's "google api" is the Earth Engine STAC JSON
  (earthengine-stac/catalog/USDA/USDA_NASS_CDL.json): class names and
  colors only, which the icechunk store's attrs already carry.
