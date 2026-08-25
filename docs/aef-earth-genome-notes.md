# Comparing AEF and Earth Index embeddings against CDL / NLCD

Working notes. The experiment: use CDL (and NLCD) as a ruler to compare two
frozen embedding models, at field-polygon level.

---

## What the experiment actually is

A **linear probe**. Not making a crop map — using CDL to measure how much
crop-relevant information each encoder packed into its vectors, and whether a
cheap classifier can get at it. The map is a byproduct; the number is the result.

**Why a second model.** One accuracy number is uninterpretable. AEF gets 88% on
CDL — is that good? No reference point. Add Earth Index:

- EI gets 71% → AEF is meaningfully better at this task.
- EI gets 87% → the signal is just *in* Sentinel-2, and any encoder finds it.

Same number, opposite conclusion. The **gap** is the finding, not the winner.

**Steering what the gap means.** AEF encodes annual time series; Earth Index
(SoftCon/DINOv2 ViT-S/14) sees essentially one composite. So the AEF–EI gap on
CDL ≈ *what does watching the crop grow buy you over just looking at it?*

Run NLCD as the control:

| | CDL (phenology-dependent) | NLCD (snapshot-decidable) |
|---|---|---|
| **AEF** | expect large lead | expect small lead |
| **Earth Index** | | |

If the lead is large on crops and small on built/forest/water → consistent with
the time-series story. If it's the same size on both → the advantage comes from
something else and the phenology explanation is wrong. The pattern across all
four cells says more than any single cell.

---

## Rosetta Stone framing

The stone's value isn't that the Greek was *correct* — it's that the same
content appears in all three scripts.

For an A-vs-B comparison, **CDL's accuracy barely matters**. Whatever CDL gets
wrong, it gets wrong identically for both encoders, so systematic label error
cancels in the differential.

This reframes the leakage worry: contamination only breaks the comparison if
it's **asymmetric**. And it would be — SoftCon is self-supervised on imagery
with no labels at all, while AEF was trained with sparse label supervision
possibly including NLCD. So:

- **CDL is the better stone** — less likely to have been in the room.
- NLCD tilts the field toward AEF for reasons unrelated to representation quality.
- (The NLCD-as-AEF-training-target claim is from a secondary blog source, not
  confirmed against the paper. Worth verifying before leaning on it.)

**Where the metaphor gives out:** the stone's three scripts encode the same
content. Embeddings encode a *superset* — CDL is a lossy projection of what's in
them. Failure to decode CDL isn't proof of a worse representation; it might be
encoding something CDL has no word for. It's a probe along one axis, not a ranking.

---

## Why field boundaries make this good rather than just decent

1. **Support mismatch dissolves.** AEF is per-pixel 10 m; Earth Index is
   patch-based (ViT chips). No clean reconciliation at pixel level. At field
   level: mean-pool AEF over the polygon and renormalize, average the EI patches
   whose centers fall inside → a 64-vector and a 384-vector describing the same
   object. One label per field from the CDL mode. No resampling argument to have.
2. **Label noise drops near zero.** CDL error concentrates at edges and mixed
   pixels. Mode over a field interior is effectively ground truth. Matters
   specifically because if label noise is large relative to the AEF–EI gap, the
   gap can't be resolved at all.
3. **N becomes honest.** 10,000 pixels in one field is *one* observation.
   Pixel-level probes produce absurdly inflated confidence. Field-level N is the
   real statistical power — typically two orders of magnitude below the pixel count.
4. **Blocking gets easy and necessary.** Block by **section or county, not by
   field** — adjacent fields are heavily correlated, and both encoders are
   patch-based, so a model can partly read a field's identity off its neighbors.
   Without blocking you measure spatial autocorrelation, not representation quality.

---

## Protocol

Hold everything fixed except the feature block (64 cols vs 384 cols):
same folds, same spatial blocking, same classifier.

- **Linear probe** — the standard.
- **kNN probe** — no training, tests the metric structure directly rather than
  what a classifier can carve out of it. Run both.
- **Label-efficiency curve** — accuracy at 10 / 50 / 200 / 1000 samples per
  class. AEF's whole pitch is low-shot; this curve is where you'd see it or not.

### Erosion test

Erode the field polygon progressively — full field, 20 m in, 40 m in — and watch
accuracy:

- **Holds** → the signal is the field.
- **Climbs** → you were being hurt by edge contamination.
- **Falls** → a meaningful share of the "crop signal" was the *surrounding
  landscape*. Real and slightly uncomfortable finding about patch-based encoders.

### Label-free test (free with boundaries)

Measure embedding discontinuity **across** a field edge vs. **within** the field.
No labels at all. Directly asks whether either encoder resolves agricultural
parcel structure or smears across it. AEF should do well by construction;
whether SoftCon does is an actual open question.

### Cross-model agreement (also label-free)

Can't compare 64-D and 384-D directly — compare **similarity structures**.
Sample N fields, build the pairwise similarity matrix in each space, correlate
the two matrices (RSA, Mantel, or CKA).

Localized version, which gives a map: for each field, take its k nearest
neighbors in AEF space and in EI space, score the rank agreement.

- High agreement → both models independently group this place with the same set
  of places. Probably real.
- Low agreement → one encoder sees structure the other flattens. **These
  disagreement spots are where the interesting stuff lives.**

---

## Fly-around / interactive viewing

kNN is the right tool — cheap, no training step, refits instantly as you move.

**The trap is the opposite of the intuitive one.** Zooming *in* doesn't make you
miss things, it makes the problem fake-easy:

- One Iowa county → two classes, big rectangles, one climate. kNN hits 95% and
  it means nothing; the nearest neighbors are literally the fields next door.
- Whole Corn Belt → number drops, not because the embeddings got worse but
  because you added classes and regional variation.

Accuracy tracks the **viewport**, not the model. Two numbers from two zoom
levels are not comparable.

**Fix:** stop letting the view define the training set.

- Fit the kNN **once**, on a big stratified sample across the whole region — all
  classes, spread geographically.
- Then fly around and only **predict** inside the viewport.
- Model is fixed; only the location changes. Numbers stay comparable, and you
  see where the model struggles instead of watching the difficulty knob turn.

**Two more:**

- **Don't display an accuracy number at all.** Display the *disagreement* —
  color each field by whether the prediction matched CDL. A scalar lies about
  zoom; a spatial pattern doesn't. And you're looking for *where* it breaks
  anyway, which is a map question.
- **If you want a number, use balanced accuracy per class.** A viewport that's
  90% corn will flatter you badly on overall accuracy.

Zoomed way out you eventually hit a rendering/sampling limit, but field-level
(vs. pixel-level) buys a lot of room — a whole state before that binds.

---

## Data

| Dataset | Notes | Link |
|---|---|---|
| AEF / Satellite Embedding V1 | 64-D, 10 m, annual 2017–2025, unit-length, linearly composable | [EE catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) |
| AEF COGs (source.coop) | by UTM zone, for local workflows | [source.coop/tge-labs/aef](https://source.coop/tge-labs/aef) |
| Earth Index embeddings | 384-D, global 2024, SoftCon weights on DINOv2 ViT-S/14 (TUM), parquet by UTM tile | [source.coop/earthgenome/earthindexembeddings](https://source.coop/earthgenome/earthindexembeddings) |
| Earth Genome S2 composite | 12 bands, cloud-free, 2023 — reference mosaic, *not* a label source | [announcement](https://medium.com/earthrisemedia/announcing-public-access-to-our-global-cloud-free-imagery-archive-bb21311abb69) |

**Year alignment:** EI is 2024 → use AEF 2024. The mosaic is 2023; accept the
offset on the backdrop only.

**Both AEF and EI are on source.coop**, so this is a local DuckDB + rasterio job,
no Earth Engine round-trip. Grab one UTM tile and prototype there.

### Sidebar: Overture / raster LULC as alternate targets

- Overture `base/land_use` is a translation of the OSM `landuse` tag — land
  *use*, not cover. Cemetery/golf course/park are near-identical in embedding
  space. Much of the "disagreement" is ontological, not error. `base/land_cover`
  is the fairer comparison.
- Esri/Impact Observatory 10m Annual LULC: 2017–2025, same 10 m S2 grid as AEF.
  Cleanest raster target, but only 9 classes (Rangeland is a catch-all).
- Dynamic World ships **per-pixel class probabilities** — lets you separate
  "both confident and disagreeing" (interesting) from "uncertain and disagreeing"
  (boring mixed pixel).
- **Circularity caveat:** AEF ingests S2/Landsat; IO and DW are S2 classifiers.
  Agreement is partly baked in. You're testing information retention under
  compression, not accuracy.