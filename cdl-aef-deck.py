# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xarray",
#     "zarr>=3",
#     "icechunk",
#     "obstore",
#     "numpy",
#     "scipy",
#     "anywidget>=0.9",
#     "pyarrow>=25.0.0",
#     "pillow==12.3.0",
#     "ipywidgets==8.1.8",
#     "traitlets==5.15.1",
# ]
# ///
"""CDL, backed or not by AlphaEarth, on the Fields of the World: the deck.gl build.

Zoomed out, at ANY zoom, the map is the Cropland Data Layer as a picture:
tiles the kernel renders from the icechunk store's majority pyramid (30 m
2008-2025, 10 m 2024-2025), the same read as cdl-ftw.py, with a crops-only
mask and the P(field) clip as toggles. Zoomed in (camera z12+, a field
paint on), the unit becomes THE FIELD:

  1. The fields in view are the connected components of FTW P(field) >= 0.5
     at 10 m (scipy.ndimage.label on one window of the probability Zarr).
  2. Each field gets its CDL majority crop (the year's CDL sampled onto the
     field grid through the closed-form Albers forward), its purity, LAST
     year's majority (the rotation voter), and its mean AlphaEarth vector
     (the year's embeddings at 20 m stride, one bincount per band).
  3. Per view, every crop with enough fields gets a PROTOTYPE (the mean of
     its fields' unit vectors). A field's agreement is the sigmoid of the
     margin between the cosine to its own crop's prototype and the cosine
     to the best other prototype (the NLCD deck notebook's score, on fields
     instead of hexagons). The runner-up is what "AlphaEarth suggests",
     relative to the crops in THIS view, not a classification.
  4. Paints, one at a time: CDL (each field its majority crop's color);
     color by agreement (viridis, BRIGHT = agrees, dark = a lead);
     AlphaEarth suggests (every field in the CDL color of the crop AEF puts
     it closest to, grey where AEF has no prototype for its crop). Highlight
     disagreement reverses the
     ramp. (An "agreement" paint, CDL color with alpha by agreement, is
     commented out in the strip: near-binary scores made it read as plain
     CDL; the code path stays in field_fill.) The CDL raster does not draw
     under the fields (a faded field fades to the basemap). A click outlines
     the field in gold; the same field again, or the basemap, clears it.

The map is a deck.gl 9.3.10 anywidget (the HRRR counties film's pinned
esm.sh graph inside maplibre, docs/deck-geoarrow-fields-plan.md). Zoomed
out, a TileLayer whose PNGs the kernel renders, ONE batch per view (cdl-ftw's
serve over custom messages). From camera z12 with a field paint on, the
FIELDS ARE POLYGONS: the FTW PMTiles' z13 tiles under the padded view,
decoded to closed rings, each polygon keyed to its field id by sampling the
label grid, shipped as one GeoArrow IPC table to a GeoArrowPolygonLayer; a
paint switch is a bytes trait of colors, no reload; a pan inside the box
costs nothing. The click is picked geometrically IN THE BROWSER (deck's GPU
picking never worked under marimo) and the kernel tells the field's story.
The lonboard RasterLayer build is in history (e82a293..3448d2b); the first
deck-widget build with two TileLayers at b970d50..0693f27. No SQL: the joins
are positional (every dataset is a raster on a known grid; the field id image
from ndimage.label is the key and np.bincount is the groupby), see the
"join" cell.

Data (all anonymous on source.coop): the CDL icechunk repo
(chill/usda-cropland-data-layer), FTW P(field) Zarr + per-state PMTiles
outlines (tge-labs/ftw-global-data), the AlphaEarth mosaic (tge-labs/
aef-mosaic; "The AlphaEarth Foundations Satellite Embedding dataset is
produced by Google and Google DeepMind", CC-BY 4.0).

Run from THIS repo's venv:  uv sync && uv run marimo edit cdl-aef-deck.py

In the cloud, no install (molab: github.com -> molab.marimo.io/github, the
branch in the path, so this link follows the branch and must be repointed at
main when it merges; the inline deps above are the environment, every read is
anonymous HTTPS, and the caches under $TMPDIR start empty, so every serve there
is cold):

  https://molab.marimo.io/github/kentstephen/cdl-ftw-zarr-marimo/blob/cdl-aef-deck/cdl-aef-deck.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import asyncio
    import gzip
    import io
    import json
    import math
    import os
    import struct
    import tempfile
    import threading
    import time
    import urllib.parse
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from PIL import Image, ImageDraw
    from scipy import ndimage

    import anywidget
    import traitlets
    import obstore
    import icechunk
    import xarray as xr
    import zarr
    from obstore.store import S3Store

    import pyarrow as pa
    import pyarrow.ipc as pa_ipc

    import marimo as mo

    return (
        Image,
        ImageDraw,
        S3Store,
        ThreadPoolExecutor,
        anywidget,
        asyncio,
        gzip,
        icechunk,
        io,
        json,
        math,
        mo,
        ndimage,
        np,
        obstore,
        os,
        pa,
        pa_ipc,
        struct,
        tempfile,
        threading,
        time,
        traitlets,
        urllib,
        xr,
        zarr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # CDL, backed or not by AlphaEarth, on the Fields of the World

    Zoomed out: the **Cropland Data Layer** as a picture, at any zoom (the
    store's majority pyramid), with *crops only* and the *fields* clip as
    masks. Zoomed in past **z12** with a field paint on, each **Fields of the
    World** field gets its CDL majority crop, last year's crop, and its mean
    **AlphaEarth** vector; per view every crop with enough fields gets a
    prototype, and a field's *agreement* is how much closer its vector sits to
    its own crop's prototype than to the best other one.

    - **CDL** paint: each field its majority crop's color.
    - **color by agreement** paint: viridis on the agreement (bright = agrees, dark = a lead).
    - **AlphaEarth suggests** paint: every field in the CDL color of the crop
      AlphaEarth puts it closest to (relative to this view): its own where
      AlphaEarth backs it, another crop's where it does not. Grey where
      AlphaEarth has no prototype for its crop.

    Click a field for its story; *analyze what's in view* for the per-crop table.

    Run it in the cloud, nothing to install:
    [open this notebook in molab](https://molab.marimo.io/github/kentstephen/cdl-ftw-zarr-marimo/blob/cdl-aef-deck/cdl-aef-deck.py).

    | leg | source | read |
    |---|---|---|
    | crops | CDL icechunk repo, 30 m 2008-2025 + 10 m 2024-2025, majority pyramid, EPSG:5070 | xarray window + closed-form Albers |
    | fields | FTW P(field) Zarr, 10 m + pyramid; per-state PMTiles outlines | xarray window, ndimage.label; ranged GETs + MVT decode |
    | embeddings | `tge-labs/aef-mosaic`, 10 m, 64 x int8, 2017-2025, no pyramid | xarray window by 256-px chunk, cached |
    | score | per-view prototypes, sigmoid margin | numpy |
    """)
    return


@app.cell
def _(math, os, tempfile):
    # ---- constants ----------------------------------------------------------
    SC_BUCKET = "us-west-2.opendata.source.coop"
    AEF_ZARR = "tge-labs/aef-mosaic/"
    CDL_BUCKET = "chill"
    CDL_PREFIX = "usda-cropland-data-layer/v0.1.0.icechunk"
    CDL_ENDPOINT = "https://data.source.coop"
    FTW_ZARR = "tge-labs/ftw-global-data/predictions/zarr/alpha/global.zarr/"
    FTW_VEC = (
        "tge-labs/ftw-global-data/predictions/vectors/alpha/"
        "results-by-admin-conf/admin:country_code=US/"
    )

    # AlphaEarth mosaic grid (EPSG:4326)
    AEF_RES = 8.983111749910169e-05
    AEF_Y0 = 83.68570533713473
    AEF_X0 = -180.0
    AEF_SHAPE = (1859584, 4009984)
    AEF_YEARS = list(range(2017, 2026))
    ACH = 256                 # AEF inner chunk (px)
    AEF_MEM_CHUNKS = 192

    # FTW grid (EPSG:4326)
    FTW_RES = 8.98311982e-05
    FTW_Y0 = 83.748345
    FTW_YEARS = (2024, 2025)
    FTW_LEVELS = [4, 16, 64, 256]   # P(field) pyramid levels for the raster clip
    FTW_TILE_ZMAX = 13              # the per-state PMTiles' top zoom

    # CDL pyramids: pixel = 30*k (30m group) or 10*k (10m group, 2024-2025)
    LEVELS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    LEVELS10 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    CDL_YEARS = list(range(2008, 2026))
    YEARS = [y for y in CDL_YEARS if y in AEF_YEARS]   # 2017-2025: both sides
    YEAR0 = 2024
    PX_PER = 1.4              # level floor: largest k with pixel <= PX_PER image px

    # the tile serve (cdl-ftw.py's)
    TILE_PX = 256             # deck asks for z/x/y, the kernel answers 256-px PNGs
    BATCH_S = 0.05            # how long the first request of a burst waits
    TILE_CACHE = 3000         # PNG tiles kept
    TILE_ZMIN, TILE_ZMAX = 3, 15
    AEF_ZMIN = 13             # the PMTiles zoom the field polygons are read at (their top)
    MARGIN = 0.15             # raster batch: view box slack beyond the viewport
    # THE TWO KNOBS FOR WHERE THE FIELDS SHOW (Stephen, 2026-08-25 night: "we
    # need to be able to zoom out and look at it").
    # FIELD_ZOOM feels REVERSED: it is the camera zoom the fields START at, so
    # a SMALLER number shows the fields from FURTHER OUT (12 = from camera z12
    # and every zoom closer in); a bigger number makes you zoom in more. The
    # status line prints the TILE zoom, one more than the camera.
    # FIELD_MAX_KM2 caps the box the kernel folds (view + 15 %); a view above
    # it says "zoom in for the fields". Camera z13 on a 1400-px canvas is
    # ~70 km2 padded, z12 ~290, z11.5 ~600, z11 ~1,200; the AEF read is
    # ~0.6 MB per km2 cold, then the disk cache.
    FIELD_ZOOM = 11.0
    PAD = 1.15                # the field box beyond the camera footprint (pans inside it cost nothing)
    FIELD_MAX_KM2 = 1500.0
    FIELD_TILE_Z = int(round(FIELD_ZOOM)) + 1   # the tile zoom of camera FIELD_ZOOM (256-px tiles): the raster's clip
    #                                             and outlines apply from here, nothing field-related below
    SETTLE = 0.35             # seconds the camera must rest before a field serve
    LABELS_SLOT = "watername_ocean"   # the basemap style layer the deck layers draw under
    #                                   (Positron and Dark Matter both carry it)
    MIN_FIELD_PX = 12         # ~0.3 ac at 10 m: smaller components sit out
    # FTW's polygons are its 10 m raster vectorized, so every boundary is a
    # staircase of 10 m steps (measured: 100 % of the outline segments are
    # axis-aligned, median 10.3 m) and zoomed in the fields read as jagged.
    # Douglas-Peucker for the VIZ, in tile units at decode time (cached with
    # the tile, so it costs one pass per tile): a straight edge comes back a
    # straight line, real curvature stays. 0 disables. Stephen, 2026-08-26.
    SIMPLIFY_M = 10.0
    MIN_CROP_FRAC = 0.3       # a field is a CROP field if >= this much is a crop class
    MIN_CLASS_FIELDS = 20     # a crop needs this many fields in view for a prototype
    TAU = 0.05                # the sigmoid's scale on the cosine margin (0.02 on hexagon
    #                           means saturated on field means: p50 1.00)
    PANEL_MIN_AC = 10.0       # the "most surprising" list ignores scraps
    ACRES_PER_KM2 = 247.10538

    # paint
    ALPHA_MIN, ALPHA_MAX = 30, 235
    ALPHA_FLAT = 220
    ALPHA_RAMP = 225
    DIM_ALPHA = 22
    QUIET = (150, 150, 150)   # unscored / agreeing-under-suggests fields
    AGREE_CMAP = "viridis"
    RAMPS = {
        "viridis": "440154470d6048186a482374472e7c4538824241863e4a893a548c365d8d32658e2e6d8e2b758e287d8e25848e228c8d1f948c1e9c8920a38625ab822eb37c3aba7648c16e58c7656ccd5a7fd34e93d741a8db34c0df25d5e21aeae51afde725",
        "cividis": "00224e00285b002e6a0533711c396f293f6e33446d3c4a6c45506c4d556c555b6d5c616e6467706b6d72727274787877807f78888578908b78979177a09875a89e73b0a571b9ab6dc2b369cbb965d3c05fdcc859e6d051efd748f8df3cfee838",
    }
    OUTLINE = (192, 192, 192, 255)   # classic silver, full alpha (Stephen, 2026-08-26)

    VIEW_W, VIEW_H = 1400, 700
    EXTENT = [-125.0, 24.0, -66.5, 49.8]
    # the opening view: Stephen's box over the Delta (Bethel Island to Stockton,
    # 2026-08-25, boundingbox.klokantech.com), fitted to the canvas
    HOME_BOX = (-121.604644, 37.887747, -121.2336, 38.111648)
    _hw, _hs, _he, _hn = HOME_BOX
    _my = lambda lat: math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    _zx = math.log2(2 * math.pi * (VIEW_W / 512) / math.radians(_he - _hw))
    _zy = math.log2(2 * math.pi * (VIEW_H / 512) / (_my(_hn) - _my(_hs)))
    HOME = {"longitude": (_hw + _he) / 2,
            "latitude": math.degrees(2 * math.atan(math.exp((_my(_hn) + _my(_hs)) / 2)) - math.pi / 2),
            "zoom": round(min(_zx, _zy) - 0.05, 2)}
    CACHE_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo")

    HOLD: dict = {}
    return (
        ACH,
        ACRES_PER_KM2,
        AEF_MEM_CHUNKS,
        AEF_RES,
        AEF_SHAPE,
        AEF_X0,
        AEF_Y0,
        AEF_ZARR,
        AEF_ZMIN,
        AGREE_CMAP,
        ALPHA_FLAT,
        ALPHA_MAX,
        ALPHA_MIN,
        ALPHA_RAMP,
        BATCH_S,
        CACHE_DIR,
        CDL_BUCKET,
        CDL_ENDPOINT,
        CDL_PREFIX,
        CDL_YEARS,
        DIM_ALPHA,
        EXTENT,
        FIELD_MAX_KM2,
        FIELD_TILE_Z,
        FIELD_ZOOM,
        FTW_LEVELS,
        FTW_RES,
        FTW_TILE_ZMAX,
        FTW_VEC,
        FTW_Y0,
        FTW_YEARS,
        FTW_ZARR,
        HOLD,
        HOME,
        LABELS_SLOT,
        LEVELS,
        LEVELS10,
        MARGIN,
        MIN_CLASS_FIELDS,
        MIN_CROP_FRAC,
        MIN_FIELD_PX,
        OUTLINE,
        PAD,
        PX_PER,
        QUIET,
        RAMPS,
        SC_BUCKET,
        SETTLE,
        SIMPLIFY_M,
        TAU,
        TILE_CACHE,
        TILE_PX,
        TILE_ZMAX,
        TILE_ZMIN,
        VIEW_H,
        VIEW_W,
        YEAR0,
        YEARS,
    )


@app.cell
def _(
    AEF_ZARR,
    CDL_BUCKET,
    CDL_ENDPOINT,
    CDL_PREFIX,
    FTW_LEVELS,
    FTW_ZARR,
    LEVELS,
    LEVELS10,
    S3Store,
    SC_BUCKET,
    icechunk,
    xr,
    zarr,
):
    # ---- open the stores (cdl-ftw.py + aef-agreement.py, by copy) ---------
    _aef_store = zarr.storage.ObjectStore(
        S3Store(bucket=SC_BUCKET, region="us-west-2", skip_signature=True, prefix=AEF_ZARR),
        read_only=True,
    )
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        AEF_DS = xr.open_zarr(_aef_store, chunks=None, consolidated=False)

    _ftw_store = zarr.storage.ObjectStore(
        S3Store(bucket=SC_BUCKET, region="us-west-2", skip_signature=True, prefix=FTW_ZARR),
        read_only=True,
    )
    FTW_ROOT = xr.open_zarr(_ftw_store, chunks=None, consolidated=False)
    FTW_DS = {}
    for _k in FTW_LEVELS:
        FTW_DS[_k] = xr.open_zarr(_ftw_store, group=f"{_k}x", chunks=None, consolidated=False)

    _storage = icechunk.s3_storage(
        bucket=CDL_BUCKET, prefix=CDL_PREFIX, endpoint_url=CDL_ENDPOINT,
        region="us-east-1", anonymous=True, force_path_style=True,
    )
    _repo = icechunk.Repository.open(_storage)
    _session = _repo.readonly_session("main")
    # every pyramid level as an xarray Dataset: DS[k] (30 m group), DS10[k] (10 m group)
    DS = {}
    for _k in LEVELS:
        DS[_k] = xr.open_zarr(_session.store, group=("30m" if _k == 1 else f"30m/{_k}x"), chunks=None)
    DS10 = {}
    for _k in LEVELS10:
        DS10[_k] = xr.open_zarr(_session.store, group=("10m" if _k == 1 else f"10m/{_k}x"), chunks=None)

    # ---- the classes, from the store's own attrs; protan-safe default palette
    # (red-dominant classes remapped onto a blue/purple cycle: cotton #FF2525
    # next to soybean green fails for Stephen)
    _at = DS[1]["crop_type"].attrs
    _names, _colors = _at["class_names"], _at["class_colors"]

    def _noncrop(name):
        if name.startswith("Developed"):
            return True
        return name in {
            "Background", "Clouds/No Data", "Water", "Open Water",
            "Perennial Ice/Snow", "Barren", "Forest", "Deciduous Forest",
            "Evergreen Forest", "Mixed Forest", "Shrubland",
            "Grassland/Pasture", "Grass/Pasture", "Woody Wetlands",
            "Herbaceous Wetlands", "Wetlands", "Nonag/Undefined",
        }

    def _rgb(hexs):
        return int(hexs[1:3], 16), int(hexs[3:5], 16), int(hexs[5:7], 16)

    _SAFE_CYCLE = ["#3F6BD6", "#8E44AD", "#00B8D4", "#D633C4",
                   "#5C6BC0", "#0091EA", "#7C4DFF", "#6A1B9A"]
    _i = 0
    CLASSES = {}   # code -> (name, hex, (r, g, b), noncrop)
    for _code in sorted(_names, key=int):
        _nm, _hx = _names[_code], _colors[_code]
        _r, _g, _b = _rgb(_hx)
        _safe = _hx
        if _r >= 170 and _g <= 100 and _b <= 110:
            _safe = _SAFE_CYCLE[_i % len(_SAFE_CYCLE)]
            _i += 1
        CLASSES[int(_code)] = (_nm, _safe, _rgb(_safe), _noncrop(_nm))
    NONCROP_CODES = sorted(c for c, v in CLASSES.items() if v[3])
    return AEF_DS, CLASSES, DS, DS10, FTW_DS, FTW_ROOT, NONCROP_CODES


@app.cell
def _(
    AGREE_CMAP,
    CLASSES,
    MARGIN,
    NONCROP_CODES,
    PAD,
    RAMPS,
    VIEW_H,
    VIEW_W,
    math,
    np,
):
    # ---- pure helpers -------------------------------------------------------
    def tile_box(z, x, y):
        """Web Mercator tile -> lon/lat (W, S, E, N)."""
        n = 2 ** z
        W = x / n * 360.0 - 180.0
        E = (x + 1) / n * 360.0 - 180.0
        N = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        S = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        return W, S, E, N

    def _lat_to_y(lat):
        r = math.radians(lat)
        return (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2

    def _y_to_lat(y):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y))))

    def view_to_bbox(vs):
        """The flat camera footprint (W, S, E, N); the widget reports its canvas
        size (`w`, `h`) with every move, the constants are the seed."""
        world = 512 * (2 ** vs["zoom"])
        w, h = vs.get("w") or VIEW_W, vs.get("h") or VIEW_H
        half_lon = 360.0 * w / world / 2
        yc, half_y = _lat_to_y(vs["latitude"]), h / world / 2
        return (vs["longitude"] - half_lon, _y_to_lat(yc + half_y),
                vs["longitude"] + half_lon, _y_to_lat(yc - half_y))

    def pad_box(b, f=PAD):
        dx, dy = (b[2] - b[0]) * (f - 1) / 2, (b[3] - b[1]) * (f - 1) / 2
        return (max(-179.9, b[0] - dx), max(-85.0, b[1] - dy),
                min(179.9, b[2] + dx), min(85.0, b[3] + dy))

    def bbox4326(vs):
        """The raster batch's view box: the footprint with MARGIN slack."""
        return pad_box(view_to_bbox(vs), 1 + MARGIN)

    def box_km2(b):
        w = (b[2] - b[0]) * 111.32 * math.cos(math.radians((b[1] + b[3]) / 2))
        return abs(w * (b[3] - b[1]) * 110.574)

    def contains(outer, inner):
        return (outer[0] <= inner[0] and outer[1] <= inner[1]
                and outer[2] >= inner[2] and outer[3] >= inner[3])

    def albers_xy(lon, lat):
        """EPSG:5070 forward (Albers equal-area conic on GRS80), closed form in
        numpy; verified to the millimetre against ST_Transform (2026-08-20)."""
        a = 6378137.0
        e2 = 0.00669438002290
        e = math.sqrt(e2)
        lat0, lon0 = math.radians(23.0), math.radians(-96.0)
        lat1, lat2 = math.radians(29.5), math.radians(45.5)

        def m(p):
            return np.cos(p) / np.sqrt(1 - e2 * np.sin(p) ** 2)

        def q(p):
            sp = np.sin(p)
            return (1 - e2) * (sp / (1 - e2 * sp * sp)
                               - (1 / (2 * e)) * np.log((1 - e * sp) / (1 + e * sp)))

        n = (m(lat1) ** 2 - m(lat2) ** 2) / (q(lat2) - q(lat1))
        C = m(lat1) ** 2 + n * q(lat1)
        rho0 = a * np.sqrt(C - n * q(lat0)) / n
        lon = np.radians(lon)
        lat = np.radians(lat)
        rho = a * np.sqrt(C - n * q(lat)) / n
        th = n * (lon - lon0)
        return rho * np.sin(th), rho0 - rho * np.cos(th)

    def albers_box(W, S, E, N):
        """Albers box of a lon/lat box: densified boundary, clamped to the array."""
        lons = np.linspace(W, E, 9)
        lats = np.linspace(S, N, 9)
        bl = np.concatenate([lons, lons, np.full(9, W), np.full(9, E)])
        bt = np.concatenate([np.full(9, S), np.full(9, N), lats, lats])
        X, Y = albers_xy(bl, bt)
        _X0, _Y0, _X1, _Y1 = -2417835.0, 158265.0, 2387295.0, 3321225.0
        return (max(float(X.min()), _X0), max(float(Y.min()), _Y0),
                min(float(X.max()), _X1), min(float(Y.max()), _Y1))

    # the class LUTs (256 entries by code)
    CLASS_RGB = np.full((256, 3), 136, dtype=np.uint8)
    for _c, (_nm, _hx, _rgb, _nc) in CLASSES.items():
        CLASS_RGB[_c] = _rgb
    NONCROP = np.zeros(256, dtype=bool)
    NONCROP[[0, 81, *NONCROP_CODES]] = True

    # the agreement ramp: AGREE_CMAP's stops interpolated to a 256-entry LUT
    # (index 0 = 0 % agreement, 255 = 100 %; viridis: bright = agrees)
    _hx = RAMPS[AGREE_CMAP]
    _stops = np.array([[int(_hx[i + j:i + j + 2], 16) for j in (0, 2, 4)] for i in range(0, len(_hx), 6)], np.float64)
    AGREE_LUT = np.stack(
        [np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_stops)), _stops[:, k]) for k in range(3)], 1
    ).round().astype(np.uint8)
    RAMP_HEX = ["#%02x%02x%02x" % tuple(int(v) for v in AGREE_LUT[i]) for i in range(0, 256, 17)]
    return (
        AGREE_LUT,
        CLASS_RGB,
        NONCROP,
        RAMP_HEX,
        albers_box,
        albers_xy,
        bbox4326,
        box_km2,
        contains,
        pad_box,
        tile_box,
        view_to_bbox,
    )


@app.cell
def _(
    CACHE_DIR,
    FTW_TILE_ZMAX,
    FTW_VEC,
    S3Store,
    SC_BUCKET,
    SIMPLIFY_M,
    ThreadPoolExecutor,
    gzip,
    math,
    np,
    obstore,
    os,
    struct,
    threading,
):
    # ---- FTW field OUTLINES from the per-state PMTiles (cdl-ftw.py's, by copy):
    # PMTiles v3 directory walk over ranged GETs, MVT decode by hand, segments
    # along a tile's clip line dropped, polylines never closed. Blobs cached on
    # disk, decoded rings in memory.
    _pm = S3Store(bucket=SC_BUCKET, region="us-west-2", skip_signature=True)
    _TILE_DIR = os.path.join(CACHE_DIR, "ftw-tiles")
    _arch, _arch_lock = {}, threading.Lock()
    _mem, _mem_lock = {}, threading.Lock()
    _tpool = ThreadPoolExecutor(max_workers=16)

    def _rng(path, a, b):
        return bytes(memoryview(obstore.get_range(_pm, path, start=a, end=b + 1)))

    def _varint(buf, i):
        r = sh = 0
        while True:
            c = buf[i]
            i += 1
            r |= (c & 0x7F) << sh
            if not c & 0x80:
                return r, i
            sh += 7

    def _parse_dir(buf):
        n, i = _varint(buf, 0)
        ids, last = [0] * n, 0
        for k in range(n):
            v, i = _varint(buf, i)
            last += v
            ids[k] = last
        runs = [0] * n
        for k in range(n):
            runs[k], i = _varint(buf, i)
        lens = [0] * n
        for k in range(n):
            lens[k], i = _varint(buf, i)
        offs = [0] * n
        for k in range(n):
            v, i = _varint(buf, i)
            offs[k] = (offs[k - 1] + lens[k - 1]) if v == 0 and k > 0 else v - 1
        return list(zip(ids, offs, lens, runs))

    def _tile_id(z, x, y):
        acc = sum((1 << t) * (1 << t) for t in range(z))
        n = 1 << z
        d, sq = 0, n >> 1
        while sq > 0:
            rx = 1 if x & sq else 0
            ry = 1 if y & sq else 0
            d += sq * sq * ((3 * rx) ^ ry)
            if ry == 0:
                if rx == 1:
                    x, y = sq - 1 - x, sq - 1 - y
                x, y = y, x
            sq >>= 1
        return acc + d

    def _find(entries, tid):
        lo, hi = 0, len(entries) - 1
        while lo <= hi:
            m = (lo + hi) // 2
            if tid < entries[m][0]:
                hi = m - 1
            elif tid > entries[m][0]:
                lo = m + 1
            else:
                return entries[m]
        if hi >= 0 and (entries[hi][3] == 0 or tid - entries[hi][0] < entries[hi][3]):
            return entries[hi]
        return None

    def _open(st):
        with _arch_lock:
            a = _arch.get(st)
            if a is None:
                path = f"{FTW_VEC}US_{st}.pmtiles"
                hdr = _rng(path, 0, 126)
                assert hdr[:7] == b"PMTiles" and hdr[7] == 3, "not a PMTiles v3 archive"
                rd_off, rd_len, _, _, ld_off, _, td_off, _ = struct.unpack("<8Q", hdr[8:72])
                root = _parse_dir(gzip.decompress(_rng(path, rd_off, rd_off + rd_len - 1)))
                a = _arch[st] = {"path": path, "root": root, "ld": ld_off, "td": td_off,
                                 "leaf": {}, "maxz": hdr[101]}
            return a

    def _blob(st, z, x, y):
        fp = os.path.join(_TILE_DIR, st, str(z), str(x), f"{y}.mvt")
        if os.path.exists(fp):
            with open(fp, "rb") as f:
                b = f.read()
            return b or None
        a = _open(st)
        tid, ents = _tile_id(z, x, y), a["root"]
        blob = b""
        for _ in range(4):
            e = _find(ents, tid)
            if e is None:
                break
            if e[3] == 0:
                lk = (e[1], e[2])
                with _arch_lock:
                    if lk not in a["leaf"]:
                        a["leaf"][lk] = _parse_dir(gzip.decompress(
                            _rng(a["path"], a["ld"] + e[1], a["ld"] + e[1] + e[2] - 1)))
                ents = a["leaf"][lk]
                continue
            blob = _rng(a["path"], a["td"] + e[1], a["td"] + e[1] + e[2] - 1)
            break
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        tmp = f"{fp}.{threading.get_ident()}.tmp"
        with open(tmp, "wb") as f:
            f.write(blob)
        os.replace(tmp, fp)
        return blob or None

    def _fields_pb(buf):
        i, n = 0, len(buf)
        while i < n:
            key, i = _varint(buf, i)
            f, w = key >> 3, key & 0x7
            if w == 0:
                v, i = _varint(buf, i)
            elif w == 2:
                ln, i = _varint(buf, i)
                v = buf[i:i + ln]
                i += ln
            elif w == 5:
                v = buf[i:i + 4]
                i += 4
            elif w == 1:
                v = buf[i:i + 8]
                i += 8
            else:
                raise ValueError(f"wire type {w}")
            yield f, w, v

    def _rings(geom):
        rings, ring = [], None
        x = y = 0
        i, n = 0, len(geom)
        while i < n:
            cmd, i = _varint(geom, i)
            op, count = cmd & 0x7, cmd >> 3
            if op == 1:
                for _ in range(count):
                    dx, i = _varint(geom, i)
                    dy, i = _varint(geom, i)
                    x += (dx >> 1) ^ -(dx & 1)
                    y += (dy >> 1) ^ -(dy & 1)
                    ring = [(x, y)]
                    rings.append(ring)
            elif op == 2:
                for _ in range(count):
                    dx, i = _varint(geom, i)
                    dy, i = _varint(geom, i)
                    x += (dx >> 1) ^ -(dx & 1)
                    y += (dy >> 1) ^ -(dy & 1)
                    ring.append((x, y))
            elif op == 7:
                ring.append(ring[0])
            else:
                raise ValueError(f"geometry op {op}")
        return rings

    def _eps_units(z, y, extent):
        """SIMPLIFY_M in this tile's own units (MVT integers): the tile row's
        centre latitude gives the ground metres per unit. 0 = no simplify."""
        if SIMPLIFY_M <= 0:
            return 0.0
        n = 1 << z
        lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (y + 0.5) / n))))
        m_per_unit = 40075016.686 * math.cos(math.radians(lat)) / (n * extent)
        return SIMPLIFY_M / max(m_per_unit, 1e-9)

    def _dp(a, eps):
        """Douglas-Peucker on an (n, 2) array with both ends pinned, iterative
        (the recursive form goes deep on a 10 m staircase)."""
        n = len(a)
        if eps <= 0 or n < 3:
            return a
        keep = np.zeros(n, dtype=bool)
        keep[0] = keep[n - 1] = True
        e2, stack = eps * eps, [(0, n - 1)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            p, d = a[i], a[j] - a[i]
            seg = a[i + 1:j] - p
            L2 = float(d[0] * d[0] + d[1] * d[1])
            if L2 <= 0.0:
                dist2 = seg[:, 0] ** 2 + seg[:, 1] ** 2
            else:
                t = np.clip((seg @ d) / L2, 0.0, 1.0)
                off = seg - t[:, None] * d
                dist2 = off[:, 0] ** 2 + off[:, 1] ** 2
            k = int(dist2.argmax())
            if dist2[k] > e2:
                m = i + 1 + k
                keep[m] = True
                stack.append((i, m))
                stack.append((m, j))
        return a[keep]

    def _dp_ring(a, eps):
        """_dp on a CLOSED ring (the first point repeated last, so it stays
        closed); None if too little is left to fill."""
        b = _dp(a, eps)
        return b if len(b) >= 4 else None

    def _decode(blob, year, z, x, y):
        if blob[:2] == b"\x1f\x8b":
            blob = gzip.decompress(blob)
        want = str(year)
        out = []
        n = 1 << z
        for f, _w, v in _fields_pb(blob):
            if f != 3:
                continue
            name, extent, feats = None, 4096, []
            for lf, _lw, lv in _fields_pb(v):
                if lf == 1:
                    name = lv.decode("utf-8")
                elif lf == 2:
                    feats.append(lv)
                elif lf == 5:
                    extent = lv
            if name != want:
                continue
            for fv in feats:
                gtype, geom = 0, b""
                for ff, _fw, fvv in _fields_pb(fv):
                    if ff == 3:
                        gtype = fvv
                    elif ff == 4:
                        geom = fvv
                if gtype != 3:
                    continue
                for ring in _rings(geom):
                    if len(ring) < 2:
                        continue
                    a = np.asarray(ring, dtype=np.float64)
                    ax, ay = a[:-1], a[1:]
                    keep = ~(((ax[:, 0] < 0) & (ay[:, 0] < 0)) | ((ax[:, 0] > extent) & (ay[:, 0] > extent))
                             | ((ax[:, 1] < 0) & (ay[:, 1] < 0)) | ((ax[:, 1] > extent) & (ay[:, 1] > extent)))
                    idx = np.flatnonzero(keep)
                    if not len(idx):
                        continue
                    cuts = np.flatnonzero(np.diff(idx) > 1) + 1
                    eps = _eps_units(z, y, extent)
                    for run in np.split(idx, cuts):
                        pts = _dp(a[run[0]:run[-1] + 2], eps)
                        if len(pts) < 2:
                            continue
                        lon = (x + pts[:, 0] / extent) / n * 360.0 - 180.0
                        lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + pts[:, 1] / extent) / n))))
                        out.append(np.column_stack([lon, lat]))
        return out

    def _tile(st, z, x, y, year):
        key = (st, z, x, y, year)
        with _mem_lock:
            v = _mem.get(key)
        if v is not None:
            return v, False
        b = _blob(st, z, x, y)
        v = _decode(b, year, z, x, y) if b else []
        with _mem_lock:
            _mem[key] = v
            if len(_mem) > 4000:
                for _k in list(_mem)[:500]:
                    _mem.pop(_k, None)
        return v, True

    def ftw_tile_rings(states, year, W, S, E, N, z):
        """Every outline polyline (lon/lat arrays) of the states' PMTiles tiles
        at zoom z under the box. Returns (rings, tiles asked, tiles fetched)."""
        z = min(z, FTW_TILE_ZMAX)
        n = 1 << z

        def tx(lon):
            return min(n - 1, max(0, int((lon + 180) / 360 * n)))

        def ty(lat):
            lat = max(-85.05, min(85.05, lat))
            return min(n - 1, max(0, int((1 - math.log(math.tan(math.radians(lat))
                                                     + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)))

        jobs = [(st, z, x, y, year) for st in states
                for x in range(tx(W), tx(E) + 1) for y in range(ty(N), ty(S) + 1)]
        res = list(_tpool.map(lambda j: _tile(*j), jobs))
        rings = [r for v, _ in res for r in v]
        return rings, len(jobs), sum(1 for _, miss in res if miss)

    # ---- the same tiles as POLYGONS: closed rings, holes attached to their
    # exterior by MVT winding (exterior = positive shoelace area in tile coords,
    # y down). tippecanoe clips at the tile edge (buffer 5/4096), so a field
    # across two tiles is two pieces; fills and the pick go by the sampled
    # field id, so the pieces do not matter to them (the tile-edge route for
    # whole fields is open: docs/deck-geoarrow-fields-plan.md).
    _pmem, _pmem_lock = {}, threading.Lock()

    def _clip_ring(a, extent):
        """Sutherland-Hodgman of a closed ring against [0, extent]^2 (tile
        coords): tippecanoe's buffer (~2 % of the tile) put the same piece in
        two tiles, and two fills at alpha 220 read as a dark band. None if
        nothing is left."""
        for axis, bound, keep_ge in ((0, 0.0, True), (0, float(extent), False), (1, 0.0, True), (1, float(extent), False)):
            if len(a) < 4:
                return None
            p, q = a[:-1], a[1:]
            vp, vq = p[:, axis], q[:, axis]
            ip = (vp >= bound) if keep_ge else (vp <= bound)
            iq = (vq >= bound) if keep_ge else (vq <= bound)
            if ip.all() and iq.all():
                continue
            cross = ip != iq
            dv = vq - vp
            t = np.where(cross, (bound - vp) / np.where(dv == 0, 1.0, dv), 0.0)
            X = p + t[:, None] * (q - p)
            X[:, axis] = bound
            cnt = ip.astype(np.int64) + cross
            off = np.cumsum(cnt) - cnt
            out = np.empty((int(cnt.sum()), 2), dtype=np.float64)
            out[off[ip]] = p[ip]
            out[off[cross] + ip[cross]] = X[cross]
            if len(out) < 3:
                return None
            a = np.vstack([out, out[:1]])
        return a if len(a) >= 4 else None

    def _decode_polys(blob, year, z, x, y):
        if blob[:2] == b"\x1f\x8b":
            blob = gzip.decompress(blob)
        want = str(year)
        out = []
        n = 1 << z
        for f, _w, v in _fields_pb(blob):
            if f != 3:
                continue
            name, extent, feats = None, 4096, []
            for lf, _lw, lv in _fields_pb(v):
                if lf == 1:
                    name = lv.decode("utf-8")
                elif lf == 2:
                    feats.append(lv)
                elif lf == 5:
                    extent = lv
            if name != want:
                continue
            for fv in feats:
                gtype, geom = 0, b""
                for ff, _fw, fvv in _fields_pb(fv):
                    if ff == 3:
                        gtype = fvv
                    elif ff == 4:
                        geom = fvv
                if gtype != 3:
                    continue
                poly = None
                for ring in _rings(geom):
                    if len(ring) < 4:
                        continue
                    a = np.asarray(ring, dtype=np.float64)
                    if a[0, 0] != a[-1, 0] or a[0, 1] != a[-1, 1]:
                        a = np.vstack([a, a[:1]])
                    area = 0.5 * float(np.sum(a[:-1, 0] * a[1:, 1] - a[1:, 0] * a[:-1, 1]))
                    if area == 0.0:
                        continue
                    a = _clip_ring(a, extent)
                    if a is not None:
                        a = _dp_ring(a, _eps_units(z, y, extent))
                    if a is None:
                        if area > 0:
                            poly = None   # an exterior wholly in the buffer: its holes go too
                        continue
                    lon = (x + a[:, 0] / extent) / n * 360.0 - 180.0
                    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + a[:, 1] / extent) / n))))
                    ll = np.column_stack([lon, lat])
                    if area > 0 or poly is None:
                        poly = [ll]
                        out.append(poly)
                    else:
                        poly.append(ll)
        return out

    def _tile_polys(st, z, x, y, year):
        key = (st, z, x, y, year)
        with _pmem_lock:
            v = _pmem.get(key)
        if v is not None:
            return v
        b = _blob(st, z, x, y)
        v = _decode_polys(b, year, z, x, y) if b else []
        with _pmem_lock:
            _pmem[key] = v
            if len(_pmem) > 2000:
                for _k in list(_pmem)[:300]:
                    _pmem.pop(_k, None)
        return v

    def ftw_tile_polys(states, year, W, S, E, N, z=FTW_TILE_ZMAX):
        """Every field polygon (list of closed lon/lat rings, exterior first)
        of the states' PMTiles tiles at zoom z under the box."""
        z = min(z, FTW_TILE_ZMAX)
        n = 1 << z

        def tx(lon):
            return min(n - 1, max(0, int((lon + 180) / 360 * n)))

        def ty(lat):
            lat = max(-85.05, min(85.05, lat))
            return min(n - 1, max(0, int((1 - math.log(math.tan(math.radians(lat))
                                                     + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)))

        jobs = [(st, z, x, y, year) for st in states
                for x in range(tx(W), tx(E) + 1) for y in range(ty(N), ty(S) + 1)]
        return [p for v in _tpool.map(lambda j: _tile_polys(*j), jobs) for p in v]

    # the FTW state partitions: each file's extent from its OWN row-group stats
    # (the STAC bboxes are wrong; US_CA reports Montana). Non-CONUS rows kept;
    # CDL has no pixels there.
    STATES = [
        ("AB", -113.4609, 48.8716, -109.9513, 49.1153), ("AK", -179.1069, 51.2673, 178.5722, 71.3595),
        ("AL", -88.4692, 30.2366, -84.9303, 35.0198), ("AR", -94.6086, 32.9912, -89.6512, 36.5159),
        ("AZ", -114.8428, 31.3059, -108.9772, 37.1642), ("BC", -136.9020, 48.9859, -115.0546, 59.6770),
        ("BCN", -116.1912, 32.4933, -114.7463, 32.7504), ("CA", -124.3523, 32.5401, -114.1433, 42.1078),
        ("CHH", -108.7570, 29.0018, -103.3053, 31.7885), ("CO", -109.2281, 36.8565, -101.9949, 41.0663),
        ("COA", -103.3081, 28.9751, -101.2998, 29.6612), ("CT", -73.6412, 41.1286, -71.7891, 42.0537),
        ("DE", -75.7910, 38.4468, -75.0627, 39.8389), ("FL", -87.6050, 24.6337, -80.0375, 31.0112),
        ("GA", -85.6023, 30.3786, -80.8461, 34.9940), ("HI", -171.7315, 18.9141, -154.8429, 25.7605),
        ("IA", -96.6383, 40.3755, -90.1597, 43.5292), ("ID", -117.2075, 41.8476, -111.0439, 49.0006),
        ("IL", -91.5112, 36.9812, -87.4950, 42.5224), ("IN", -88.0956, 37.7752, -84.7778, 41.7762),
        ("KS", -102.1802, 36.8925, -94.5901, 40.0618), ("KY", -89.5650, 36.4889, -82.3231, 39.1427),
        ("LA", -94.0409, 29.1032, -89.1778, 33.0262), ("MA", -73.4565, 41.2416, -69.9653, 42.8878),
        ("MB", -101.3629, 48.9465, -95.3080, 49.0306), ("MD", -79.4903, 37.9769, -75.0799, 39.7319),
        ("ME", -71.0137, 43.1226, -67.0023, 47.4349), ("MI", -90.2135, 41.6930, -82.4660, 47.3937),
        ("MN", -97.2376, 43.4865, -90.0070, 49.3549), ("MO", -95.7638, 35.9749, -89.1052, 40.6164),
        ("MS", -91.6424, 30.2577, -88.1318, 35.0043), ("MT", -116.0404, 44.4582, -103.9331, 49.1742),
        ("NB", -67.7911, 46.1704, -67.7640, 47.0352), ("NC", -84.3100, 33.8565, -75.6323, 36.5740),
        ("ND", -104.0996, 45.8636, -96.5552, 49.0298), ("NE", -104.2059, 39.9444, -95.3097, 43.1038),
        ("NH", -72.5293, 42.6948, -70.7183, 45.1750), ("NJ", -75.5626, 38.9404, -73.9993, 41.3522),
        ("NM", -109.1490, 31.3281, -102.7869, 37.1542), ("NV", -120.1432, 35.0057, -113.7718, 42.1525),
        ("NY", -79.7662, 40.6174, -72.1221, 45.0234), ("OH", -84.8425, 38.4374, -80.5134, 41.9528),
        ("OK", -103.0702, 33.6282, -94.4282, 37.1493), ("OR", -124.5325, 41.7591, -116.5060, 46.1685),
        ("PA", -80.5300, 39.7032, -74.7718, 42.2674), ("QC", -74.4936, 44.9868, -69.0288, 47.4349),
        ("RI", -71.8374, 41.1601, -71.1195, 42.0213), ("SC", -83.2788, 32.0489, -78.6307, 35.1982),
        ("SD", -104.1481, 42.4952, -96.4253, 46.0128), ("SK", -110.0217, 48.8504, -101.3551, 49.1742),
        ("SON", -114.8428, 31.3059, -108.7519, 32.5818), ("TN", -90.3186, 34.9703, -81.7296, 36.6666),
        ("TX", -106.6500, 25.8412, -93.6194, 36.6163), ("UT", -114.2245, 36.8858, -108.9914, 42.1293),
        ("VA", -83.6007, 36.5346, -75.3106, 39.4304), ("VT", -73.4202, 42.7267, -71.5183, 45.0286),
        ("WA", -124.3931, 45.5561, -116.9255, 49.0109), ("WI", -92.8177, 42.4731, -86.8788, 46.9017),
        ("WV", -82.6197, 37.2515, -77.7529, 40.6241), ("WY", -111.1497, 40.8559, -103.8754, 45.1036),
        ("YT", -141.0438, 60.0153, -139.0725, 69.6589),
    ]

    def states_in(W, S, E, N):
        return [s for s, w, s_, e, n in STATES if w < E and e > W and s_ < N and n > S]

    return ftw_tile_polys, ftw_tile_rings, states_in


@app.cell
def _(
    CACHE_DIR,
    CLASS_RGB,
    DS,
    DS10,
    EXTENT,
    FTW_DS,
    FTW_LEVELS,
    FTW_RES,
    FTW_Y0,
    FTW_YEARS,
    HOLD: dict,
    Image,
    ImageDraw,
    LEVELS,
    LEVELS10,
    NONCROP,
    OUTLINE,
    PX_PER,
    TILE_PX,
    ThreadPoolExecutor,
    albers_box,
    albers_xy,
    io,
    math,
    np,
    os,
    threading,
    tile_box,
):
    # ---- the CDL raster: tiles the kernel renders from the majority pyramid
    # (cdl-ftw.py's serve, per tile instead of per batch: the level's windows
    # are cached in 1024-px BLOCKS per (group, level, year), so a burst of
    # tiles over one area reads each block once). A tile is the level whose
    # pixel is within PX_PER of the tile's ground resolution, sampled at the
    # 256 output pixel centres through the closed-form Albers forward (the
    # store is EPSG:5070, the tiles 3857): nearest neighbour in numpy.
    _BLK = 1024
    _blocks = {}                       # (group, k, year, by, bx) -> uint8 block
    _held = {"bytes": 0}
    _BUDGET = 512 * 1024 * 1024
    _lock = threading.Lock()
    _meta = {}                         # (group, k) -> (x_left_edge, y_top_edge, pix, W, H)
    _bpool = ThreadPoolExecutor(max_workers=8)   # a window's blocks are read in parallel

    def _grp(year):
        return ("10m", DS10, LEVELS10, 10.0) if year in FTW_YEARS else ("30m", DS, LEVELS, 30.0)

    def _level_meta(g, ds, k, base):
        key = (g, k)
        m = _meta.get(key)
        if m is None:
            da = ds[k]["crop_type"]
            pix = base * k
            x0 = float(da.x.values[0]) - pix / 2
            y1 = float(da.y.values[0]) + pix / 2
            m = _meta[key] = (x0, y1, pix, int(da.sizes["x"]), int(da.sizes["y"]))
        return m

    def _block(g, ds, k, year, by, bx):
        key = (g, k, year, by, bx)
        with _lock:
            b = _blocks.get(key)
        if b is not None:
            return b
        da = ds[k]["crop_type"].sel(year=year)
        _x0, _y1, _pix, W, H = _level_meta(g, ds, k, 10.0 if g == "10m" else 30.0)
        r0, c0 = by * _BLK, bx * _BLK
        a = np.asarray(da.isel(y=slice(r0, min(H, r0 + _BLK)), x=slice(c0, min(W, c0 + _BLK))).values, dtype=np.uint8)
        with _lock:
            _blocks[key] = a
            _held["bytes"] += a.nbytes
            while _held["bytes"] > _BUDGET and len(_blocks) > 1:
                k0 = next(iter(_blocks))
                if k0 == key:
                    break
                _held["bytes"] -= _blocks.pop(k0).nbytes
        return a

    def cdl_window(year, k, c0, r0, c1, r1):
        """uint8 (r1-r0, c1-c0) of the level's crop_type over a pixel window,
        assembled from cached blocks."""
        g, ds, _lv, base = _grp(year)
        _x0, _y1, _pix, W, H = _level_meta(g, ds, k, base)
        c0, r0, c1, r1 = max(0, c0), max(0, r0), min(W, c1), min(H, r1)
        out = np.zeros((max(0, r1 - r0), max(0, c1 - c0)), dtype=np.uint8)
        if out.size == 0:
            return out, c0, r0
        want = [(by, bx) for by in range(r0 // _BLK, (r1 - 1) // _BLK + 1)
                for bx in range(c0 // _BLK, (c1 - 1) // _BLK + 1)]
        got = list(_bpool.map(lambda b: _block(g, ds, k, year, *b), want))
        for (by, bx), a in zip(want, got):
            sr, sc = by * _BLK, bx * _BLK
            rr0, cc0 = max(r0, sr), max(c0, sc)
            rr1, cc1 = min(r1, sr + a.shape[0]), min(c1, sc + a.shape[1])
            if rr1 <= rr0 or cc1 <= cc0:
                continue
            out[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0] = a[rr0 - sr:rr1 - sr, cc0 - sc:cc1 - sc]
        return out, c0, r0

    def level_for(year, z, lat):
        """The pyramid level for a tile zoom: the largest k with pixel <= PX_PER
        ground metres per output pixel."""
        _g, _ds, lv, base = _grp(year)
        mpp = 156543.03392 * math.cos(math.radians(lat)) / 2 ** z * (256 / TILE_PX)
        want = max(mpp * PX_PER / base, 1.0)
        ks = [k for k in lv if k <= want]
        return ks[-1] if ks else lv[0]

    def cdl_codes_at(year, LON, LAT, k=1):
        """CDL codes (uint8, LON.shape) at lon/lat points from level k."""
        g, ds, _lv, base = _grp(year)
        x0, y1, pix, W, H = _level_meta(g, ds, k, base)
        bx0, by0, bx1, by1 = albers_box(float(LON.min()), float(LAT.min()), float(LON.max()), float(LAT.max()))
        c0, c1 = int((bx0 - x0) / pix), int(math.ceil((bx1 - x0) / pix)) + 1
        r0, r1 = int((y1 - by1) / pix), int(math.ceil((y1 - by0) / pix)) + 1
        grid, c0, r0 = cdl_window(year, k, c0, r0, c1, r1)
        code = np.zeros(LON.shape, dtype=np.uint8)
        if grid.size == 0:
            return code
        X, Y = albers_xy(LON, LAT)
        jx = ((X - x0) / pix).astype(np.int64) - c0
        jy = ((y1 - Y) / pix).astype(np.int64) - r0
        ok = (jx >= 0) & (jx < grid.shape[1]) & (jy >= 0) & (jy < grid.shape[0])
        code[ok] = grid[jy[ok], jx[ok]]
        return code

    # ---- the P(field) clip for the raster: the pyramid level within 4/3 of
    # the pixel served, chunk-cached (cdl-ftw.py's _ftw_mask, by copy) ---------
    _CH = 512
    _MASK_DIR = os.path.join(CACHE_DIR, "ftw-mask")

    def ftw_mask(fyear, px_m, W, S, E, N):
        """Dense boolean of P(field) >= 0.5 over the box from the coarsest
        pyramid level whose cell is within 4/3 of px_m: (mask, ix0, iy0, res)."""
        f = max(l for l in FTW_LEVELS if 10 * l <= max(px_m * 4 / 3, 40))
        res = FTW_RES * f
        ix0 = int(math.floor((W + 180.0) / res))
        ix1 = int(math.floor((E + 180.0) / res))
        iy0 = int(math.floor((FTW_Y0 - N) / res))
        iy1 = int(math.floor((FTW_Y0 - S) / res))
        cx0, cx1, cy0, cy1 = ix0 // _CH, ix1 // _CH, iy0 // _CH, iy1 // _CH
        cache = HOLD.setdefault("fchunks", {})
        missing = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                if (fyear, f, cx, cy) in cache:
                    continue
                fp = os.path.join(_MASK_DIR, f"{f}x", str(fyear), f"{cx}_{cy}.npy")
                if os.path.exists(fp):
                    cache[(fyear, f, cx, cy)] = np.unpackbits(np.load(fp)).reshape(_CH, _CH).astype(bool)
                else:
                    missing.append((cx, cy))
        if missing:
            mx0, mx1 = min(c[0] for c in missing), max(c[0] for c in missing)
            my0, my1 = min(c[1] for c in missing), max(c[1] for c in missing)
            lon0, lon1 = mx0 * _CH * res - 180.0, (mx1 + 1) * _CH * res - 180.0
            lat1, lat0 = FTW_Y0 - my0 * _CH * res, FTW_Y0 - (my1 + 1) * _CH * res
            da = FTW_DS[f]["variables"].sel(time=f"{fyear}-01-01", band="field").sel(
                x=slice(lon0, lon1), y=slice(lat1, lat0))
            vals = np.asarray(da.values) >= 0.5
            big = np.zeros(((my1 - my0 + 1) * _CH, (mx1 - mx0 + 1) * _CH), dtype=bool)
            if vals.size:
                gx = np.floor((da.x.values + 180.0) / res).astype(np.int64) - mx0 * _CH
                gy = np.floor((FTW_Y0 - da.y.values) / res).astype(np.int64) - my0 * _CH
                okx = (gx >= 0) & (gx < big.shape[1])
                oky = (gy >= 0) & (gy < big.shape[0])
                big[np.ix_(gy[oky], gx[okx])] = vals[oky][:, okx]
            for cx in range(mx0, mx1 + 1):
                for cy in range(my0, my1 + 1):
                    piece = big[(cy - my0) * _CH:(cy - my0 + 1) * _CH, (cx - mx0) * _CH:(cx - mx0 + 1) * _CH].copy()
                    cache[(fyear, f, cx, cy)] = piece
                    fp = os.path.join(_MASK_DIR, f"{f}x", str(fyear), f"{cx}_{cy}.npy")
                    try:
                        os.makedirs(os.path.dirname(fp), exist_ok=True)
                        tmp = f"{fp}.{threading.get_ident()}.tmp.npy"
                        np.save(tmp, np.packbits(piece))
                        os.replace(tmp, fp)
                    except Exception:
                        pass
            if len(cache) > 600:
                for _k in list(cache)[:100]:
                    cache.pop(_k, None)
        mask = np.zeros(((cy1 - cy0 + 1) * _CH, (cx1 - cx0 + 1) * _CH), dtype=bool)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                mask[(cy - cy0) * _CH:(cy - cy0 + 1) * _CH, (cx - cx0) * _CH:(cx - cx0 + 1) * _CH] = cache[(fyear, f, cx, cy)]
        return mask, cx0 * _CH, cy0 * _CH, res

    _blank = {"png": None}

    def blank_png():
        if _blank["png"] is None:
            buf = io.BytesIO()
            Image.new("RGBA", (1, 1), (0, 0, 0, 0)).save(buf, format="PNG")
            _blank["png"] = buf.getvalue()
        return _blank["png"]

    def cdl_tile_png(year, crops_only, fields, sel, z, x, y, rings=None):
        """(PNG bytes, counts by code) for Web Mercator tile (z, x, y): the CDL
        from the level the zoom deserves; nodata alpha 0; crops_only drops the
        non-crop classes; fields drops pixels outside P(field) >= 0.5; sel
        isolates classes (after the counts); `rings` (dict rings/bounds) are
        drawn on top with PIL."""
        W, S, E, N = tile_box(z, x, y)
        zero = np.zeros(256, dtype=np.int64)
        if E < EXTENT[0] or W > EXTENT[2] or N < EXTENT[1] or S > EXTENT[3]:
            return blank_png(), zero
        T = TILE_PX
        lons = W + (np.arange(T) + 0.5) * (E - W) / T
        lats = N - (np.arange(T) + 0.5) * (N - S) / T
        LON, LAT = np.meshgrid(lons, lats)
        k = level_for(year, z, (N + S) / 2)
        code = cdl_codes_at(year, LON, LAT, k)
        if not code.any():
            return blank_png(), zero
        if crops_only:
            code[NONCROP[code]] = 0
        if fields:
            fyear = year if year in FTW_YEARS else FTW_YEARS[0]
            _g, _ds, _lv, base = _grp(year)
            mask, fx0, fy0, res = ftw_mask(fyear, base * k, W, S, E, N)
            fx = np.floor((LON + 180.0) / res).astype(np.int64) - fx0
            fy = np.floor((FTW_Y0 - LAT) / res).astype(np.int64) - fy0
            inb = (fx >= 0) & (fx < mask.shape[1]) & (fy >= 0) & (fy < mask.shape[0])
            field = np.zeros((T, T), dtype=bool)
            field[inb] = mask[fy[inb], fx[inb]]
            code[~field] = 0
        counts = np.bincount(code[code > 0], minlength=256)
        if sel:
            keep = np.zeros(256, dtype=bool)
            keep[list(sel)] = True
            code = np.where(keep[code], code, 0)
        out = np.zeros((T, T, 4), dtype=np.uint8)
        out[..., :3] = CLASS_RGB[code]
        out[..., 3] = np.where(code > 0, 255, 0)
        img = Image.fromarray(out, "RGBA")
        if rings:
            rb = rings["bounds"]
            hit = np.flatnonzero((rb[:, 0] < E) & (rb[:, 2] > W) & (rb[:, 1] < N) & (rb[:, 3] > S))
            if len(hit):
                d = ImageDraw.Draw(img)
                sx, sy = T / (E - W), T / (N - S)
                for hi in hit:
                    r = rings["rings"][hi]
                    pts = list(zip((r[:, 0] - W) * sx, (N - r[:, 1]) * sy))
                    if len(pts) >= 2:
                        d.line(pts, fill=OUTLINE, width=1)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=4)
        return buf.getvalue(), counts

    def rings_png(z, x, y, rings):
        """A transparent tile with only the outline polylines (raster off)."""
        W, S, E, N = tile_box(z, x, y)
        T = TILE_PX
        img = Image.new("RGBA", (T, T), (0, 0, 0, 0))
        rb = rings["bounds"]
        hit = np.flatnonzero((rb[:, 0] < E) & (rb[:, 2] > W) & (rb[:, 1] < N) & (rb[:, 3] > S))
        if len(hit):
            d = ImageDraw.Draw(img)
            sx, sy = T / (E - W), T / (N - S)
            for hi in hit:
                r = rings["rings"][hi]
                pts = list(zip((r[:, 0] - W) * sx, (N - r[:, 1]) * sy))
                if len(pts) >= 2:
                    d.line(pts, fill=OUTLINE, width=1)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False, compress_level=4)
        return buf.getvalue()

    return blank_png, cdl_codes_at, cdl_tile_png, rings_png


@app.cell
def _(
    ACH,
    ACRES_PER_KM2,
    AEF_DS,
    AEF_MEM_CHUNKS,
    AEF_RES,
    AEF_SHAPE,
    AEF_X0,
    AEF_Y0,
    CACHE_DIR,
    CDL_YEARS,
    CLASSES,
    FTW_RES,
    FTW_ROOT,
    FTW_Y0,
    FTW_YEARS,
    HOLD: dict,
    MIN_CLASS_FIELDS,
    MIN_CROP_FRAC,
    MIN_FIELD_PX,
    NONCROP,
    TAU,
    cdl_codes_at,
    math,
    ndimage,
    np,
    os,
    threading,
    time,
):
    # ---- THE JOIN. Every dataset is a raster on a known grid, so the join is
    # positional: the FTW 10 m grid is the frame; ndimage.label turns P(field)
    # >= 0.5 into a field-id image (the key); the CDL is sampled onto that
    # grid through the Albers forward (index arithmetic, nearest); AlphaEarth
    # shares the lat/lon pitch (a floor-divide); np.bincount over the field id
    # is the GROUP BY (one call for the CDL class histogram of every field,
    # 64 calls for the mean embedding). aef-agreement.py's field table with
    # the NLCD deck notebook's prototype-margin score in place of the kNN.

    # AEF chunk cache + dequant (aef-similarity.py's) --------------------------
    _AEF_DIR = os.path.join(CACHE_DIR, "aef-emb")
    _DQ = np.zeros(256, dtype=np.float32)
    _qv = np.arange(-128, 128, dtype=np.float32)
    _DQ[(np.arange(-128, 128) & 0xFF)] = (np.abs(_qv) / 127.5) ** 2 * np.sign(_qv)
    _DQ[128] = 0.0   # nodata -128 -> 0 weight (valid tracked separately)

    def _aef_fetch(year, missing, needed):
        mem = HOLD.setdefault("aef_chunks", {})
        mx0, mx1 = min(c[0] for c in missing), max(c[0] for c in missing)
        my0, my1 = min(c[1] for c in missing), max(c[1] for c in missing)
        ya, yb = my0 * ACH, min((my1 + 1) * ACH, AEF_SHAPE[0])
        xa, xb = mx0 * ACH, min((mx1 + 1) * ACH, AEF_SHAPE[1])
        big = AEF_DS.embeddings.sel(time=year).isel(y=slice(ya, yb), x=slice(xa, xb)).values
        for cx in range(mx0, mx1 + 1):
            for cy in range(my0, my1 + 1):
                piece = np.full((64, ACH, ACH), -128, dtype=np.int8)
                sy, sx = (cy - my0) * ACH, (cx - mx0) * ACH
                part = big[:, sy:sy + ACH, sx:sx + ACH]
                piece[:, :part.shape[1], :part.shape[2]] = part
                mem[(year, cx, cy)] = piece
                fp = os.path.join(_AEF_DIR, str(year), f"{cx}_{cy}.npy")
                try:
                    os.makedirs(os.path.dirname(fp), exist_ok=True)
                    tmp = f"{fp}.{threading.get_ident()}.tmp.npy"
                    np.save(tmp, piece)
                    os.replace(tmp, fp)
                except Exception:
                    pass
        if len(mem) > AEF_MEM_CHUNKS:
            # never evict what the current request needs (the 2026-08-24 race)
            for _k in [k for k in list(mem) if k not in needed][:AEF_MEM_CHUNKS // 4]:
                mem.pop(_k, None)

    def _aef_chunks(year, cx0, cx1, cy0, cy1):
        mem = HOLD.setdefault("aef_chunks", {})
        needed = {(year, cx, cy) for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)}
        missing = []
        for key in needed:
            if key in mem:
                mem[key] = mem.pop(key)
                continue
            fp = os.path.join(_AEF_DIR, str(year), f"{key[1]}_{key[2]}.npy")
            if os.path.exists(fp):
                mem[key] = np.load(fp)
            else:
                missing.append((key[1], key[2]))
        if missing:
            _aef_fetch(year, missing, needed)
        return {k: mem[k] for k in needed}

    def aef_mosaic(year, W, S, E, N):
        """int8 (64, H, W) chunk-aligned over the box + (ix0, iy0)."""
        ix0 = max(int((W - AEF_X0) / AEF_RES), 0)
        ix1 = min(int((E - AEF_X0) / AEF_RES), AEF_SHAPE[1] - 1)
        iy0 = max(int((AEF_Y0 - N) / AEF_RES), 0)
        iy1 = min(int((AEF_Y0 - S) / AEF_RES), AEF_SHAPE[0] - 1)
        cx0, cx1, cy0, cy1 = ix0 // ACH, ix1 // ACH, iy0 // ACH, iy1 // ACH
        mem = _aef_chunks(year, cx0, cx1, cy0, cy1)
        mos = np.full((64, (cy1 - cy0 + 1) * ACH, (cx1 - cx0 + 1) * ACH), -128, dtype=np.int8)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                mos[:, (cy - cy0) * ACH:(cy - cy0 + 1) * ACH, (cx - cx0) * ACH:(cx - cx0 + 1) * ACH] = mem[(year, cx, cy)]
        return mos, cx0 * ACH, cy0 * ACH

    # the FTW 10 m mask, chunk-cached (aef-agreement.py's) ---------------------
    _CH = 512
    _MASK_DIR = os.path.join(CACHE_DIR, "ftw-mask")

    def ftw10(fyear, W, S, E, N):
        """P(field) >= 0.5 at 10 m over the box, chunk-aligned: (mask, fx0, fy0).
        Its connected components are THE FIELDS."""
        res = FTW_RES
        ix0 = int(math.floor((W + 180.0) / res))
        ix1 = int(math.floor((E + 180.0) / res))
        iy0 = int(math.floor((FTW_Y0 - N) / res))
        iy1 = int(math.floor((FTW_Y0 - S) / res))
        cx0, cx1, cy0, cy1 = ix0 // _CH, ix1 // _CH, iy0 // _CH, iy1 // _CH
        cache = HOLD.setdefault("fchunks10", {})
        missing = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                if (fyear, cx, cy) in cache:
                    continue
                fp = os.path.join(_MASK_DIR, "1x", str(fyear), f"{cx}_{cy}.npy")
                if os.path.exists(fp):
                    cache[(fyear, cx, cy)] = np.unpackbits(np.load(fp)).reshape(_CH, _CH).astype(bool)
                else:
                    missing.append((cx, cy))
        if missing:
            mx0, mx1 = min(c[0] for c in missing), max(c[0] for c in missing)
            my0, my1 = min(c[1] for c in missing), max(c[1] for c in missing)
            lon0, lon1 = mx0 * _CH * res - 180.0, (mx1 + 1) * _CH * res - 180.0
            lat1, lat0 = FTW_Y0 - my0 * _CH * res, FTW_Y0 - (my1 + 1) * _CH * res
            da = FTW_ROOT["variables"].sel(time=f"{fyear}-01-01", band="field").sel(
                x=slice(lon0, lon1), y=slice(lat1, lat0))
            vals = np.asarray(da.values) >= 0.5
            big = np.zeros(((my1 - my0 + 1) * _CH, (mx1 - mx0 + 1) * _CH), dtype=bool)
            if vals.size:
                gx = np.floor((da.x.values + 180.0) / res).astype(np.int64) - mx0 * _CH
                gy = np.floor((FTW_Y0 - da.y.values) / res).astype(np.int64) - my0 * _CH
                okx = (gx >= 0) & (gx < big.shape[1])
                oky = (gy >= 0) & (gy < big.shape[0])
                big[np.ix_(gy[oky], gx[okx])] = vals[oky][:, okx]
            for cx in range(mx0, mx1 + 1):
                for cy in range(my0, my1 + 1):
                    piece = big[(cy - my0) * _CH:(cy - my0 + 1) * _CH, (cx - mx0) * _CH:(cx - mx0 + 1) * _CH].copy()
                    cache[(fyear, cx, cy)] = piece
                    fp = os.path.join(_MASK_DIR, "1x", str(fyear), f"{cx}_{cy}.npy")
                    try:
                        os.makedirs(os.path.dirname(fp), exist_ok=True)
                        tmp = f"{fp}.{threading.get_ident()}.tmp.npy"
                        np.save(tmp, np.packbits(piece))
                        os.replace(tmp, fp)
                    except Exception:
                        pass
            if len(cache) > 400:
                for _k in list(cache)[:80]:
                    cache.pop(_k, None)
        mask = np.zeros(((cy1 - cy0 + 1) * _CH, (cx1 - cx0 + 1) * _CH), dtype=bool)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                mask[(cy - cy0) * _CH:(cy - cy0 + 1) * _CH, (cx - cx0) * _CH:(cx - cx0 + 1) * _CH] = cache[(fyear, cx, cy)]
        return mask, cx0 * _CH, cy0 * _CH

    def score(cls, V, ok):
        """The NLCD deck notebook's score on any unit: per class with >=
        MIN_CLASS_FIELDS scored units the prototype is the mean unit vector;
        agreement = sigmoid((own cosine - best other cosine) / TAU); alt is
        the runner-up class (what AlphaEarth suggests, relative to the view).
        Returns (agree, alt, own_cos, alt_cos, prototype classes)."""
        n = len(cls)
        agree = np.full(n, np.nan, dtype=np.float32)
        alt = np.full(n, -1, dtype=np.int64)
        own = np.full(n, np.nan, dtype=np.float32)
        oth = np.full(n, np.nan, dtype=np.float32)
        ids = np.flatnonzero(ok)
        if len(ids) == 0:
            return agree, alt, own, oth, np.zeros(0, np.int64)
        present, counts = np.unique(cls[ids], return_counts=True)
        protos = present[counts >= MIN_CLASS_FIELDS]
        if len(protos) < 2:
            return agree, alt, own, oth, protos
        P = np.stack([V[ids][cls[ids] == c].mean(0) for c in protos])
        P /= np.maximum(np.linalg.norm(P, axis=1), 1e-9)[:, None]
        cos = V[ids] @ P.T
        has = np.isin(cls[ids], protos)
        idx = np.where(has, np.searchsorted(protos, cls[ids]), 0)
        rows = np.arange(len(ids))
        o = cos[rows, idx]
        other = cos.copy()
        other[rows, idx] = -np.inf
        ai = other.argmax(1)
        a = other[rows, ai]
        m = o - a
        sc = ids[has]
        agree[sc] = 1.0 / (1.0 + np.exp(-m[has] / TAU))
        alt[sc] = protos[ai[has]]
        own[sc] = o[has]
        oth[sc] = a[has]
        return agree, alt, own, oth, protos

    def field_table(year, W, S, E, N):
        """The fields over the box: labels, CDL majority (this year and last),
        purity, mean AEF vector, agreement. CACHED by the chunk-aligned box and
        year (a toggle or a same-box zoom pays nothing)."""
        fyear = year if year in FTW_YEARS else FTW_YEARS[0]
        ck = (year, int(math.floor((W + 180.0) / FTW_RES)) // _CH, int(math.floor((E + 180.0) / FTW_RES)) // _CH,
              int(math.floor((FTW_Y0 - N) / FTW_RES)) // _CH, int(math.floor((FTW_Y0 - S) / FTW_RES)) // _CH)
        fc = HOLD.setdefault("ftab_cache", {})
        hit = fc.get(ck)
        if hit is not None:
            return hit
        lap = {"t": time.time()}
        timing = {}

        def tick(name):
            now = time.time()
            timing[name] = now - lap["t"]
            lap["t"] = now

        mask, fx0, fy0 = ftw10(fyear, W, S, E, N)
        tick("ftw")
        lab, nlab = ndimage.label(mask)
        lab = lab.astype(np.int32)
        sizes = np.bincount(lab.ravel(), minlength=nlab + 1)
        h, w = lab.shape
        lonv = (fx0 + np.arange(w) + 0.5) * FTW_RES - 180.0
        latv = FTW_Y0 - (fy0 + np.arange(h) + 0.5) * FTW_RES
        LON, LAT = np.meshgrid(lonv, latv)
        tick("label")

        def majority(codes):
            codes_crop = np.where(NONCROP[codes], 0, codes)
            pair = lab.astype(np.int64) * 256 + codes_crop
            pc = np.bincount(pair.ravel(), minlength=(nlab + 1) * 256).reshape(nlab + 1, 256)
            pc[:, 0] = 0
            return pc.argmax(1).astype(np.uint8), pc.max(1), pc.sum(1)

        codes = cdl_codes_at(year, LON, LAT, 1)
        maj, crop_px, crop_tot = majority(codes)
        prev_year = year - 1 if (year - 1) in CDL_YEARS else None
        if prev_year is not None:
            prev, _pp, _pt = majority(cdl_codes_at(prev_year, LON, LAT, 1))
        else:
            prev = np.zeros(nlab + 1, dtype=np.uint8)
        tick("cdl")
        # mean AEF vector per field: the field labels mapped ONTO the AEF grid
        # (nearest) at 20 m stride, one weighted bincount per band
        mos, ax0, ay0 = aef_mosaic(year, float(lonv[0]), float(latv[-1]), float(lonv[-1]), float(latv[0]))
        tick("aef read")
        alon = AEF_X0 + (ax0 + np.arange(0, mos.shape[2], 2) + 0.5) * AEF_RES
        alat = AEF_Y0 - (ay0 + np.arange(0, mos.shape[1], 2) + 0.5) * AEF_RES
        fj = np.clip(((alon + 180.0) / FTW_RES).astype(np.int64) - fx0, 0, w - 1)
        fi = np.clip(((FTW_Y0 - alat) / FTW_RES).astype(np.int64) - fy0, 0, h - 1)
        lab_a = lab[np.ix_(fi, fj)]
        q = mos[:, ::2, ::2]
        valid = q[0] != -128
        la = np.where(valid, lab_a, 0).ravel()
        counts = np.bincount(la, minlength=nlab + 1)
        V = np.zeros((nlab + 1, 64), dtype=np.float32)
        for b in range(64):
            V[:, b] = np.bincount(la, weights=_DQ[q[b].view(np.uint8)].ravel(), minlength=nlab + 1)
        okv = counts > 0
        V[okv] /= counts[okv, None]
        nrm = np.linalg.norm(V, axis=1)
        V[nrm > 0] /= nrm[nrm > 0, None]
        tick("aef fold")
        kept = (okv & (sizes >= MIN_FIELD_PX) & (maj > 0)
                & (crop_tot >= MIN_CROP_FRAC * np.maximum(sizes, 1)) & (nrm > 0))
        kept[0] = False
        agree, alt, own, oth, protos = score(maj.astype(np.int64), V, kept)
        tick("score")
        latm = math.radians((S + N) / 2)
        pxa = ((FTW_RES * 111.32 * math.cos(latm)) * (FTW_RES * 110.574)) * ACRES_PER_KM2
        purity = np.where(crop_tot > 0, crop_px / np.maximum(crop_tot, 1), 0.0).astype(np.float32)
        scored = ~np.isnan(agree)
        ft = {"lab": lab, "fx0": fx0, "fy0": fy0, "nlab": nlab, "maj": maj, "prev": prev,
              "purity": purity, "agree": agree, "alt": alt, "own": own, "oth": oth,
              "sizes": sizes, "kept": kept, "scored": scored, "protos": protos,
              "pxa": pxa, "fyear": fyear, "year": year, "prev_year": prev_year,
              "box": (float(lonv[0]), float(latv[-1]), float(lonv[-1]), float(latv[0])),
              "nfields": int(kept.sum()), "nscored": int(scored.sum()),
              "timing": timing, "mos_mb": mos.nbytes / 1e6}
        fc[ck] = ft
        if len(fc) > 6:
            for _k in list(fc)[:2]:
                fc.pop(_k, None)
        return ft

    def cname(code):
        return CLASSES.get(int(code), (f"code {code}", "#888", (136, 136, 136), True))[0]

    return cname, field_table


@app.cell
def _(
    AGREE_LUT,
    ALPHA_FLAT,
    ALPHA_MAX,
    ALPHA_MIN,
    ALPHA_RAMP,
    CLASSES,
    CLASS_RGB,
    DIM_ALPHA,
    FTW_RES,
    FTW_Y0,
    QUIET,
    RAMP_HEX,
    cname,
    io,
    math,
    np,
    pa,
    pa_ipc,
):
    # ---- the field paints: one rgba LUT by field id; the polygons keyed to
    # the field ids and shipped as Arrow
    def aef_best(ft):
        """The crop AlphaEarth puts each field closest to: the field's own CDL
        crop where AlphaEarth backs it (agreement >= 0.5), the runner-up crop
        where it does not, and -1 where AlphaEarth has nothing to say (the
        field sits out, or its crop has no prototype in this view)."""
        maj, alt, agree, kept, scored = ft["maj"], ft["alt"], ft["agree"], ft["kept"], ft["scored"]
        best = np.where(np.nan_to_num(agree, nan=1.0) < 0.5, alt, maj.astype(np.int64))
        return np.where(kept & scored & (best >= 0), best, -1).astype(np.int64)

    def field_fill(ft, paint, sel, inv=False):
        """(nlab+1, 4) uint8 rgba by field id for a paint (cdl, agreement,
        viridis, suggests). Fields that sit out (tiny, non-crop, no embedding)
        are faint grey; id 0 is transparent. The picked field is NOT here: it
        keeps its color and gets a gold outline in the tile."""
        n = ft["nlab"] + 1
        maj, alt, agree, kept, scored = ft["maj"], ft["alt"], ft["agree"], ft["kept"], ft["scored"]
        rgba = np.zeros((n, 4), dtype=np.uint8)
        rgba[1:, :3] = QUIET
        rgba[1:, 3] = 45
        a01 = np.clip(np.nan_to_num(agree, nan=0.0), 0, 1)
        if paint == "cdl":
            rgba[kept, :3] = CLASS_RGB[maj[kept]]
            rgba[kept, 3] = ALPHA_FLAT
            key = maj.astype(np.int64)
        elif paint == "viridis":
            idx = (a01 * 255).astype(np.int64)
            lut = AGREE_LUT[255 - idx] if inv else AGREE_LUT[idx]
            rgba[kept, :3] = np.where(scored[kept, None], lut[kept], 128)
            rgba[kept, 3] = ALPHA_RAMP
            key = maj.astype(np.int64)
        elif paint == "agreement":
            rgba[kept, :3] = CLASS_RGB[maj[kept]]
            t = (1 - a01) if inv else a01
            al = (ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * t).astype(np.uint8)
            al = np.where(scored, al, ALPHA_MIN if inv else ALPHA_MAX).astype(np.uint8)
            rgba[kept, 3] = al[kept]
            key = maj.astype(np.int64)
        else:  # "suggests": every field in the CDL color of the crop AEF puts
            # it closest to (its own where AEF backs it, the runner-up where it
            # does not); a field AEF cannot speak to keeps the null grey
            key = aef_best(ft)
            has = key >= 0
            rgba[has, :3] = CLASS_RGB[key[has]]
            rgba[has, 3] = ALPHA_FLAT
        if sel:
            keep = np.isin(key, list(sel))
            rgba[kept & ~keep, 3] = DIM_ALPHA
        rgba[0] = 0
        return rgba

    def legend_for(ft, paint, inv=False):
        maj, alt, agree, kept, scored = ft["maj"], ft["alt"], ft["agree"], ft["kept"], ft["scored"]
        items = []
        if paint == "viridis" and ft["nscored"]:
            items.append({"ramp": RAMP_HEX, "cmap": "viridis",
                          "lo": "agrees" if inv else "disagrees", "hi": "disagrees" if inv else "agrees"})
        if paint == "suggests":
            best = aef_best(ft)
            has = best >= 0
            tot = max(1, int(has.sum()))
            codes, nn = np.unique(best[has], return_counts=True)
            for code, cnt in sorted(zip(codes, nn), key=lambda t: -t[1]):
                if int(code) not in CLASSES:
                    continue
                vs = int((has & (best == code) & (maj != code)).sum())
                items.append({"code": int(code), "name": cname(code), "hex": CLASSES[int(code)][1],
                              "pct": round(100 * int(cnt) / tot, 1), "p50": "",
                              "note": f"{int(cnt):,}" + (f" · {vs:,} against CDL" if vs else "")})
            null = int((kept & ~has).sum())
            if null:
                items.append({"code": -1, "name": "no suggestion", "hex": "#%02x%02x%02x" % QUIET,
                              "pct": round(100 * null / max(1, int(kept.sum())), 1), "p50": "",
                              "note": f"{null:,} of {int(kept.sum()):,} fields"})
            return items
        tot = max(1, int(kept.sum()))
        codes, nn = np.unique(maj[kept], return_counts=True)
        for code, cnt in sorted(zip(codes, nn), key=lambda t: -t[1]):
            if int(code) not in CLASSES:
                continue
            a = agree[kept & (maj == code) & scored]
            items.append({"code": int(code), "name": cname(code), "hex": CLASSES[int(code)][1],
                          "pct": round(100 * int(cnt) / tot, 1),
                          "p50": f"{np.median(a):.2f}" if len(a) else "none",
                          "note": "" if len(a) else "(unscored)"})
        return items

    def poly_fids(ft, polys):
        """The field id of every polygon: the label grid sampled at the
        exterior ring's centroid; if that lands on 0 (a thin field, a hole),
        the mode of the labels at the vertices nudged ~15 m toward it."""
        n = len(polys)
        fids = np.zeros(n, dtype=np.int32)
        if n == 0:
            return fids
        lab, fx0, fy0 = ft["lab"], ft["fx0"], ft["fy0"]
        H, W = lab.shape

        def at(lon, lat):
            gx = np.floor((lon + 180.0) / FTW_RES).astype(np.int64) - fx0
            gy = np.floor((FTW_Y0 - lat) / FTW_RES).astype(np.int64) - fy0
            ok = (gx >= 0) & (gx < W) & (gy >= 0) & (gy < H)
            out = np.zeros(len(lon), dtype=np.int32)
            out[ok] = lab[gy[ok], gx[ok]]
            return out

        ext = [p[0][:-1] for p in polys]
        cen = np.array([[r[:, 0].mean(), r[:, 1].mean()] for r in ext])
        fids[:] = at(cen[:, 0], cen[:, 1])
        step_lat = 15.0 / 110574.0
        for i in np.flatnonzero(fids == 0):
            r, c = ext[i], cen[i]
            d = c - r
            dist = np.hypot(d[:, 0] * math.cos(math.radians(c[1])), d[:, 1])
            f = np.minimum(1.0, step_lat / np.maximum(dist, 1e-12))
            pts = r + d * f[:, None]
            v = at(pts[:, 0], pts[:, 1])
            v = v[v > 0]
            if len(v):
                fids[i] = np.bincount(v).argmax()
        return fids

    def polys_ipc(polys, fids, rgba):
        """One Arrow IPC stream: `geometry` (geoarrow.polygon, interleaved
        f64), `fid` (int32) and `rgba` (uint8 x 4, the paint at build time)."""
        ring_len, nrings, xs = [], [], []
        for p in polys:
            nrings.append(len(p))
            for r in p:
                ring_len.append(len(r))
                xs.append(r)
        xy = np.concatenate(xs) if xs else np.zeros((0, 2))
        ring_off = np.concatenate([[0], np.cumsum(ring_len)]).astype(np.int32)
        poly_off = np.concatenate([[0], np.cumsum(nrings)]).astype(np.int32)
        coords = pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(xy, dtype=np.float64).ravel(), pa.float64()), 2)
        rings = pa.ListArray.from_arrays(pa.array(ring_off, pa.int32()), coords)
        geom = pa.ListArray.from_arrays(pa.array(poly_off, pa.int32()), rings)
        col = np.ascontiguousarray(rgba[fids], dtype=np.uint8)
        colors = pa.FixedSizeListArray.from_arrays(pa.array(col.ravel(), pa.uint8()), 4)
        schema = pa.schema([
            pa.field("geometry", geom.type, False, metadata={"ARROW:extension:name": "geoarrow.polygon"}),
            pa.field("fid", pa.int32(), False),
            pa.field("rgba", colors.type, False),
        ])
        tbl = pa.Table.from_arrays([geom, pa.array(fids, pa.int32()), colors], schema=schema)
        sink = io.BytesIO()
        with pa_ipc.new_stream(sink, tbl.schema) as w:
            w.write_table(tbl)
        return sink.getvalue()

    def lines_ipc(rings):
        """One Arrow IPC stream of the outline polylines: `path`
        (list<fixed_size_list<f64, 2>>)."""
        lens = [len(r) for r in rings]
        xy = np.concatenate(rings) if rings else np.zeros((0, 2))
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
        coords = pa.FixedSizeListArray.from_arrays(pa.array(np.ascontiguousarray(xy, dtype=np.float64).ravel(), pa.float64()), 2)
        path = pa.ListArray.from_arrays(pa.array(off, pa.int32()), coords)
        tbl = pa.Table.from_arrays([path], schema=pa.schema([pa.field("path", path.type, False)]))
        sink = io.BytesIO()
        with pa_ipc.new_stream(sink, tbl.schema) as w:
            w.write_table(tbl)
        return sink.getvalue()

    return aef_best, field_fill, legend_for, lines_ipc, poly_fids, polys_ipc


@app.cell
def _(anywidget, traitlets):
    class HudControls(anywidget.AnyWidget):
        """The strip under the map (aef-agreement.py's skeleton: the fullscreen
        dock; the click is the map widget's own now) with this
        notebook's controls, THREE INDEPENDENT LAYERS (Stephen, 2026-08-26:
        "should be able to disable all layers"): the CDL raster (its switch,
        crops only its modifier), the painted polygons (the paint buttons:
        CDL / color by agreement / AlphaEarth suggests, one at a time, click
        again for none), the field outlines (its own switch, no longer the
        raster's clip). Plus highlight disagreement, analyze, refresh, search,
        the pickable legend, panel and status lines, and the collapse button
        (top right; the expand button sits at the screen's bottom right).
        Every commit re-runs the wiring cell (marimo), where the acts happen
        IN the cell run."""

        ctl = traitlets.Unicode("").tag(sync=True)
        years = traitlets.Unicode("[]").tag(sync=True)
        year0 = traitlets.Unicode("2024").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)
        # is the CDL raster actually DRAWN right now (its switch on, and not
        # hidden under a field paint)? "crops only" is a modifier of that
        # raster and nothing else, so it exists only while this is true, the
        # way highlight disagreement exists only under color by agreement.
        rasteron = traitlets.Bool(True).tag(sync=True)

        _esm = r"""
        function render({ model, el }) {
          const box = document.createElement("div");
          box.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;gap:.6rem 1rem;" +
            "font:13px ui-sans-serif,system-ui,sans-serif;padding:.35rem 0 0;" +
            "user-select:none;width:100%";
          const btnCss =
            "font:13px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
            "padding:.2rem .6rem;border-radius:5px;border:1px solid " +
            "rgba(127,127,127,.45);background:transparent;color:inherit";
          const onCss = (b, on) => {
            b.style.borderColor = on ? "#2b6cb0" : "rgba(127,127,127,.45)";
            b.style.fontWeight = on ? "600" : "400";
          };
          // the strip's state starts from the LAST ctl the kernel holds (a page
          // reload rebuilds this JS, the kernel keeps serving the old switches)
          let last = {};
          try { last = JSON.parse(model.get("ctl") || "{}") || {}; } catch (e) { last = {}; }
          const has = (k) => Object.prototype.hasOwnProperty.call(last, k);
          let paint = has("paint") ? last.paint : "viridis";
          let raster = has("raster") ? !!last.raster : true;
          let crops = has("crops") ? !!last.crops : true;
          let outlinesOn = has("outlines") ? !!last.outlines : (has("fields") ? !!last.fields : true);
          let open = true;
          const sel = new Set(Array.isArray(last.sel) ? last.sel : []);
          let seq = has("n") ? (last.n | 0) : 0;
          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({
              act: act, paint: paint, sel: Array.from(sel), inv: inv.checked,
              raster: raster, crops: crops, outlines: outlinesOn,
              year: parseInt(yearSel.value, 10), n: ++seq }, extra || {})));
            model.save_changes();
          };
          const mkChk = (text, title, init, onchange) => {
            const lab = document.createElement("label");
            lab.style.cssText = "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
            const c = document.createElement("input");
            c.type = "checkbox"; c.checked = init;
            lab.appendChild(c); lab.appendChild(document.createTextNode(text));
            lab.title = title;
            c.addEventListener("change", () => onchange(c.checked));
            return [lab, c];
          };
          const yearBox = document.createElement("span");
          yearBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const yl = document.createElement("span"); yl.textContent = "year";
          const yearSel = document.createElement("select");
          yearSel.style.cssText = "font:13px ui-sans-serif,system-ui,sans-serif;padding:.1rem .3rem;border-radius:4px;border:1px solid rgba(127,127,127,.45);background:transparent;color:inherit";
          let yrs = [];
          try { yrs = JSON.parse(model.get("years") || "[]"); } catch (e) { yrs = []; }
          yrs.forEach((y) => { const o = document.createElement("option"); o.value = String(y); o.textContent = String(y); yearSel.appendChild(o); });
          yearSel.value = String(has("year") ? last.year : (model.get("year0") || yrs[yrs.length - 1] || ""));
          yearSel.addEventListener("change", () => send("set"));
          yearBox.append(yl, yearSel);
          const rasterBox = document.createElement("span");
          rasterBox.style.cssText = "display:inline-flex;gap:.6rem;align-items:center";
          const [rasLab] = mkChk("CDL raster", "the Cropland Data Layer as tiles, at any zoom (not under the fields)", raster, (v) => { raster = v; send("set"); });
          const [cropLab] = mkChk("crops only", "raster: drop the non-crop classes", crops, (v) => { crops = v; send("set"); });
          const [fldLab] = mkChk("field outlines", "the Fields of the World boundaries as silver lines (from camera z11); independent of the raster and of the paint", outlinesOn, (v) => { outlinesOn = v; send("set"); });
          rasterBox.append(rasLab, cropLab, fldLab);
          // crops only MODIFIES THE RASTER, so it is there only while the
          // raster is on screen: gone with the switch off, and gone under a
          // field paint (the paint is the fields alone, raster_dim 0), the
          // same rule as highlight disagreement under color by agreement.
          // Hidden, not greyed; the checkbox keeps its value for its return.
          const styleCrops = () => {
            cropLab.style.display = model.get("rasteron") === false ? "none" : "inline-flex";
          };
          model.on("change:rasteron", styleCrops);
          styleCrops();
          const paintBox = document.createElement("span");
          paintBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const pl = document.createElement("span");
          pl.textContent = "fields";
          const mkPaint = (key, text, title) => {
            const b = document.createElement("button");
            b.textContent = text; b.title = title; b.style.cssText = btnCss;
            b.onclick = () => { paint = paint === key ? null : key; sel.clear(); stylePaint(); send("set"); renderLegend(); };
            return [key, b];
          };
          const paintBtns = [
            mkPaint("cdl", "CDL", "each field its CDL majority crop's color (from camera z11); click again to hide"),
            // "agreement" (CDL color, alpha by agreement) is OUT for now (Stephen,
            // 2026-08-25: near-binary scores made it read as plain CDL):
            // mkPaint("agreement", "agreement", "the CDL color, alpha follows how well AlphaEarth backs the crop; click again to hide"),
            mkPaint("viridis", "color by agreement", "viridis on the agreement value: bright = agrees, dark = a lead (from camera z11); highlight disagreement reverses; click again to hide"),
            mkPaint("suggests", "AlphaEarth suggests", "every field in the CDL color of the crop AlphaEarth puts it closest to, relative to this view: its own where AlphaEarth backs it, another crop's where it does not; grey where AlphaEarth has no prototype for its crop (from camera z11); click again to hide"),
          ];
          const [invLab, inv] = mkChk("highlight disagreement", "color by agreement: reverse the ramp (bright = disagrees)", has("inv") ? !!last.inv : false, () => send("set"));
          // it modifies ONE paint, so it sits with that paint and exists only
          // while it is on: no carry-over to CDL or to suggests (Stephen,
          // 2026-08-26: "highlight disagreement should not carry over to cdl.
          // the selection ... should be next to color by agreement")
          const stylePaint = () => {
            paintBtns.forEach(([k, b]) => onCss(b, k === paint));
            invLab.style.display = paint === "viridis" ? "inline-flex" : "none";
            if (paint !== "viridis") inv.checked = false;
          };
          stylePaint();
          const paintKids = [pl];
          paintBtns.forEach(([k, b]) => { paintKids.push(b); if (k === "viridis") paintKids.push(invLab); });
          paintBox.append(...paintKids);
          const legendBox = document.createElement("div");
          legendBox.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;" +
            "gap:.15rem .7rem;flex:1 1 100%;min-width:14rem;font-size:13px";
          const renderLegend = () => {
            let items = [];
            try { items = JSON.parse(model.get("legend") || "[]"); }
            catch (e) { items = []; }
            legendBox.innerHTML = "";
            if (sel.size) {
              const x = document.createElement("button");
              x.textContent = "× all";
              x.style.cssText =
                "font:11px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
                "padding:.05rem .35rem;border-radius:4px;border:1px solid " +
                "#2b6cb0;background:transparent;color:inherit";
              x.onclick = () => { sel.clear(); send("set"); renderLegend(); };
              legendBox.appendChild(x);
            }
            items.forEach((it) => {
              if (it.ramp) {
                const r = document.createElement("span");
                r.style.cssText = "display:inline-flex;align-items:center;gap:.35rem;font:12px ui-sans-serif,system-ui,sans-serif";
                r.title = it.cmap + ": color by agreement";
                r.innerHTML =
                  '<span style="opacity:.75">' + it.lo + '</span>' +
                  '<span style="display:inline-block;width:9rem;height:10px;border-radius:2px;' +
                  "background:linear-gradient(to right," + it.ramp.join(",") + ')"></span>' +
                  '<span style="opacity:.75">' + it.hi + '</span>';
                legendBox.appendChild(r);
                return;
              }
              const b = document.createElement("button");
              const on = sel.has(it.code);
              b.style.cssText =
                "display:inline-flex;align-items:center;gap:.3rem;" +
                "font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
                "padding:.05rem .35rem;border-radius:4px;background:transparent;" +
                "color:inherit;border:1px solid " +
                (on ? "#2b6cb0" : "transparent") + (on ? ";font-weight:600" : "");
              b.title = it.pct + "%" + (it.p50 ? " · agreement p50 " + it.p50 : "") + " · click to isolate";
              b.innerHTML =
                '<span style="width:10px;height:10px;border-radius:2px;' +
                "background:" + it.hex + ';display:inline-block"></span>' +
                it.name + (it.note ? ' <span style="opacity:.6">' + it.note + "</span>" : "");
              b.onclick = () => {
                if (sel.has(it.code)) sel.delete(it.code); else sel.add(it.code);
                send("set"); renderLegend();
              };
              legendBox.appendChild(b);
            });
          };
          model.on("change:legend", renderLegend);
          renderLegend();
          const anBox = document.createElement("span");
          anBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const anB = document.createElement("button");
          anB.textContent = "analyze what's in view"; anB.style.cssText = btnCss;
          anB.title = "per crop in view: fields, acres, agreement, what AlphaEarth suggests, last year's crop";
          anB.onclick = () => send("analyze");
          const clB = document.createElement("button");
          clB.textContent = "× clear"; clB.style.cssText = btnCss;
          clB.onclick = () => send("clear");
          const rfB = document.createElement("button");
          rfB.textContent = "refresh"; rfB.style.cssText = btnCss; rfB.title = "rebuild the tile layer";
          rfB.onclick = () => send("refresh");
          const search = document.createElement("input");
          search.type = "search";
          search.placeholder = "find a place…";
          search.title = "Photon geocoder: Enter flies to the first hit";
          search.style.cssText =
            "width:11rem;font:13px ui-sans-serif,system-ui,sans-serif;" +
            "padding:.15rem .45rem;border:1px solid rgba(127,127,127,.45);" +
            "border-radius:4px;background:transparent;color:inherit";
          search.addEventListener("keydown", (e) => {
            const q = search.value.trim();
            if (e.key === "Enter" && q) { e.preventDefault(); send("search", { q: q }); }
          });
          anBox.append(anB, clB, rfB, search);
          // collapse (top right of the strip) / expand (bottom right of the
          // screen). Client-side only: no ctl, so no kernel run and no re-serve.
          const sqCss =
            "font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
            "width:1.5rem;height:1.5rem;line-height:1;padding:0;border-radius:5px;" +
            "border:1px solid rgba(127,127,127,.45);color:inherit;opacity:.6";
          const colB = document.createElement("button");
          colB.textContent = "\u25be"; colB.title = "hide the controls";
          colB.style.cssText = sqCss + ";margin-left:auto;flex:0 0 auto;background:transparent";
          const expB = document.createElement("button");
          expB.textContent = "\u25b4"; expB.title = "show the controls";
          expB.dataset.aefExpand = "1";
          expB.className = "maplibregl-ctrl";   // the map's pick handler skips its controls
          expB.style.cssText =
            sqCss + ";position:absolute;right:8px;bottom:52px;z-index:6;display:none;" +
            "background:#fff;color:#222;border-color:rgba(0,0,0,.2);opacity:1;" +
            "box-shadow:0 0 0 2px rgba(0,0,0,.1)";
          // the expand arrow belongs to the MAP, just above the Carto credit. The
          // map is another widget and may not be in the DOM yet, so poll briefly
          // for its container; failing that, the page's bottom right corner.
          const deepFind = (sel) => {
            const walk = (r) => {
              for (const n of r.querySelectorAll("*")) {
                if (n.matches && n.matches(sel)) return n;
                if (n.shadowRoot) { const h = walk(n.shadowRoot); if (h) return h; }
              }
              return null;
            };
            return walk(document);
          };
          let tries = 0;
          const dock = () => {
            const m = deepFind(".maplibregl-map");
            if (m) { m.appendChild(expB); return true; }
            if (++tries > 60) {
              expB.style.position = "fixed";
              expB.style.right = ".9rem";
              expB.style.bottom = ".9rem";
              expB.style.zIndex = "60";
              document.body.appendChild(expB);
              return true;
            }
            return false;
          };
          if (!dock()) { const iv = setInterval(() => { if (dock()) clearInterval(iv); }, 400); }
          const setOpen = (v) => {
            open = v;
            wrap.style.display = v ? "" : "none";   // the strip goes entirely
            expB.style.display = v ? "none" : "block";
          };
          colB.onclick = () => setOpen(false);
          expB.onclick = () => setOpen(true);
          [colB, expB].forEach((b) => {
            b.onmouseenter = () => { b.style.opacity = "1"; };
            b.onmouseleave = () => { b.style.opacity = b === expB ? "1" : ".6"; };
          });
          // ROW ONE is year + the layer switches + the paints, with the collapse
          // button hard right ON THAT LINE (Stephen, 2026-08-26); analyze and the
          // legend wrap below it.
          const topRow = document.createElement("div");
          topRow.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;gap:.6rem 1rem;flex:1 1 100%";
          topRow.append(yearBox, rasterBox, paintBox, colB);
          box.append(topRow, anBox, legendBox);
          const panel = document.createElement("div");
          panel.style.cssText = "font:13.5px ui-sans-serif,system-ui,sans-serif;padding:.25rem 0";
          const status = document.createElement("div");
          status.style.cssText =
            "font:13px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.2rem 0;min-height:1.2em;white-space:pre-line";
          const wrap = document.createElement("div");
          wrap.style.cssText = "width:100%;box-sizing:border-box";
          wrap.dataset.aefStrip = "1";
          wrap.append(box, panel, status);
          const killOld = (root) => {
            if (!root || !root.querySelectorAll) return;
            root.querySelectorAll("[data-aef-strip]").forEach((w) => {
              if (w !== wrap) { w.dataset.dead = "1"; w.remove(); }
            });
            root.querySelectorAll("[data-aef-expand]").forEach((b) => {
              if (b !== expB) b.remove();
            });
            root.querySelectorAll("*").forEach((n) => { if (n.shadowRoot) killOld(n.shadowRoot); });
          };
          killOld(document);
          el.appendChild(wrap);
          setOpen(open);
          const realFs = () => {
            let fe = document.fullscreenElement;
            while (fe && fe.shadowRoot && fe.shadowRoot.fullscreenElement)
              fe = fe.shadowRoot.fullscreenElement;
            return fe;
          };
          const onFs = () => {
            if (wrap.dataset.dead || !el.isConnected) {
              wrap.remove();
              document.removeEventListener("fullscreenchange", onFs);
              return;
            }
            const fe = realFs();
            if (fe && fe !== el && !el.contains(fe)) {
              if (getComputedStyle(fe).position === "static") fe.style.position = "relative";
              wrap.style.cssText =
                "position:absolute;left:0;right:0;bottom:0;z-index:30;" +
                "background:rgba(255,255,255,.94);color:#111;box-sizing:border-box;" +
                "padding:.6rem 1.4rem .7rem;box-shadow:0 -1px 4px rgba(0,0,0,.18)";
              fe.appendChild(wrap);
            } else {
              wrap.style.cssText = "width:100%;box-sizing:border-box";
              el.appendChild(wrap);
            }
            setOpen(open);   // the cssText rewrites above drop a collapsed strip's display
          };
          document.addEventListener("fullscreenchange", onFs);
          const paintS = () => { status.textContent = model.get("status") || ""; };
          model.on("change:status", paintS);
          paintS();
          const paintP = () => { panel.innerHTML = model.get("panel") || ""; };
          model.on("change:panel", paintP);
          paintP();
          return () => {
            document.removeEventListener("fullscreenchange", onFs);
            wrap.remove();
            expB.remove();
          };
        }
        export default { render };
        """

    return (HudControls,)


@app.cell
def _(anywidget, asyncio, traitlets):
    class DeckMap(anywidget.AnyWidget):
        """The map: maplibre (Carto Positron, interleaved) with deck.gl 9.3.10
        from esm.sh drawing INSIDE it under the label layers (the HRRR counties
        film's pinned esm.sh graph; the 0693f27 build's boot). Layers:

        - `cdl-<rgen>`: a TileLayer whose PNGs the kernel renders on request
          (anywidget custom messages, one batch per view on the kernel side);
          a new generation id makes deck refetch. Hidden while the fields draw.
        - `fields`: a GeoArrowPolygonLayer over ONE Arrow IPC table in the
          `fields` bytes trait (geoarrow.polygon, interleaved f64, plus `fid`
          int32 and `rgba` uint8x4); `colors` (bytes, N x 4) repaints it
          without a geometry reload; `lines` (an IPC table of paths) are the
          outline polylines, a PathLayer; the picked field's pieces get a gold
          PathLayer.
        - the click is picked HERE, geometrically (pointerup, a press that
          moves > 4 px or starts on a map control is not a click; bbox reject
          then even-odd over the rings the browser holds), because deck's
          GPU picking has never worked under marimo on any chassis tried.

        Kernel -> browser: `config` (JSON), the three bytes traits, custom
        messages `tile` and `fly`. Browser -> kernel: `view` (lon/lat/zoom +
        canvas w/h on every moveend) and `pick` ({i, fid, on, lon, lat, gen})."""

        config = traitlets.Unicode("{}").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        pick = traitlets.Unicode("").tag(sync=True)
        fields = traitlets.Bytes(b"").tag(sync=True)
        colors = traitlets.Bytes(b"").tag(sync=True)
        lines = traitlets.Bytes(b"").tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tile_fn = None  # async (z, x, y) -> PNG bytes | None; the wiring sets it
            self.on_msg(self._on_custom)

        def _on_custom(self, widget, content, buffers):
            if not isinstance(content, dict) or content.get("kind") != "tile":
                return
            try:
                asyncio.get_running_loop().create_task(self._tile(content))
            except RuntimeError:
                self.send({"kind": "tile", "id": content.get("id"), "empty": True})

        async def _tile(self, c):
            try:
                png = await self.tile_fn(int(c["z"]), int(c["x"]), int(c["y"])) if self.tile_fn else None
            except Exception:
                png = None
            if png is None:
                self.send({"kind": "tile", "id": c["id"], "empty": True})
            else:
                self.send({"kind": "tile", "id": c["id"]}, buffers=[png])

        _esm = r"""
        // every deck import pins the same versions AND the same ?deps= per package
        // (esm.sh hashes a module by its deps list), so the whole graph resolves to
        // ONE @deck.gl/core (the HRRR counties film's strings).
        import maplibregl from "https://esm.sh/maplibre-gl@5.24.0";
        import {MapboxOverlay} from "https://esm.sh/@deck.gl/mapbox@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {BitmapLayer, PathLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import {GeoArrowPolygonLayer} from "https://esm.sh/@geoarrow/deck.gl-layers@0.3.2?deps=@deck.gl/aggregation-layers@9.3.10,@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/geo-layers@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";
        import * as arrow from "https://esm.sh/apache-arrow@18.1.0";

        const STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";
        const OUTLINE = [192, 192, 192, 255];
        const GOLD = [255, 200, 40, 255];

        function bytesOf(v) {
          if (!v) return null;
          if (v instanceof DataView) return new Uint8Array(v.buffer, v.byteOffset, v.byteLength);
          if (v instanceof ArrayBuffer) return new Uint8Array(v);
          if (v.buffer) return new Uint8Array(v.buffer, v.byteOffset || 0, v.byteLength);
          return null;
        }

        function render({model, el}) {
          let cfg = {};
          try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
          const css = document.createElement("link");
          css.rel = "stylesheet"; css.href = "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.css";
          const root = document.createElement("div");
          root.style.cssText = "position:relative;width:100%";
          const mapEl = document.createElement("div");
          mapEl.style.cssText = "width:100%;height:" + (cfg.height || 700) + "px;background:#f4f2ee";
          const note = document.createElement("div");
          note.style.cssText = "position:absolute;left:8px;top:8px;z-index:5;font:11px ui-monospace,Menlo,monospace;" +
            "color:#333;background:rgba(255,255,255,.85);padding:2px 6px;border-radius:3px;pointer-events:none;display:none";
          root.append(mapEl, note);
          el.append(css, root);
          const say = (t) => { note.textContent = t; note.style.display = t ? "block" : "none"; };

          let seq = 0, map = null, overlay = null;

          // ---- tiles: ask the kernel, get a PNG back on the custom-message channel
          const pending = new Map();
          let tseq = 0;
          model.on("msg:custom", (msg, buffers) => {
            if (msg && msg.kind === "fly" && map) {
              map.flyTo({center: [msg.lon, msg.lat], zoom: msg.zoom, duration: msg.duration || 2000});
              return;
            }
            if (!msg || msg.kind !== "tile") return;
            const p = pending.get(msg.id);
            if (!p) return;
            pending.delete(msg.id);
            if (msg.empty || !buffers || !buffers.length) { p.resolve(null); return; }
            const u8 = bytesOf(buffers[0]);
            createImageBitmap(new Blob([u8], {type: "image/png"})).then(p.resolve, () => p.resolve(null));
          });
          const tileFn = ({index, signal}) => new Promise((resolve) => {
            const id = ++tseq;
            pending.set(id, {resolve});
            model.send({kind: "tile", id, x: index.x, y: index.y, z: index.z});
            if (signal) signal.addEventListener("abort", () => { pending.delete(id); resolve(null); });
          });
          const sub = (p) => {
            if (!p.data) return null;
            const {west, south, east, north} = p.tile.bbox;
            return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]});
          };

          // ---- the fields: one Arrow table, indexed for the geometric pick
          let table = null, N = 0, fids = null, geo = null, colors = null, lines = [], hit = 0;
          function indexGeometry() {
            const d = table.getChild("geometry").data[0];       // polygon: list<ring>
            const ringD = d.children[0];                        // ring: list<coord>
            const coordD = ringD.children[0];                   // coord: fixed_size_list<f64, 2>
            const xy = coordD.children[0].values;               // interleaved x y
            const polyOff = d.valueOffsets, ringOff = ringD.valueOffsets;
            const bbox = new Float64Array(N * 4), polys = new Array(N);
            for (let i = 0; i < N; i++) {
              let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
              const rings = [];
              for (let r = polyOff[d.offset + i]; r < polyOff[d.offset + i + 1]; r++) {
                const s = ringOff[r], e = ringOff[r + 1];
                rings.push([s, e]);
                for (let c = s; c < e; c++) { const x = xy[2 * c], y = xy[2 * c + 1]; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
              }
              bbox[4 * i] = x0; bbox[4 * i + 1] = y0; bbox[4 * i + 2] = x1; bbox[4 * i + 3] = y1;
              polys[i] = rings;
            }
            geo = {xy, bbox, polys};
          }
          function fieldAt(lng, lat) {
            if (!geo) return -1;
            const {xy, bbox, polys} = geo;
            for (let i = 0; i < N; i++) {
              if (lng < bbox[4 * i] || lng > bbox[4 * i + 2] || lat < bbox[4 * i + 1] || lat > bbox[4 * i + 3]) continue;
              let inside = false;
              for (const [s, e] of polys[i]) {
                for (let a = s, b = e - 1; a < e; b = a++) {
                  const xa = xy[2 * a], ya = xy[2 * a + 1], xb = xy[2 * b], yb = xy[2 * b + 1];
                  if ((ya > lat) !== (yb > lat) && lng < (xb - xa) * (lat - ya) / (yb - ya) + xa) inside = !inside;
                }
              }
              if (inside) return i;
            }
            return -1;
          }
          function loadFields() {
            const u8 = bytesOf(model.get("fields"));
            table = null; N = 0; fids = null; geo = null; hit = 0;
            if (!u8 || !u8.length) return;
            try {
              table = arrow.tableFromIPC(u8);
              N = table.numRows;
              fids = table.getChild("fid").toArray();
              indexGeometry();
            } catch (e) { table = null; N = 0; geo = null; say("fields: " + e.message); }
          }
          function loadColors() {
            const u8 = bytesOf(model.get("colors"));
            colors = (u8 && N && u8.length === N * 4) ? u8.slice() : null;
          }
          function loadLines() {
            lines = [];
            const u8 = bytesOf(model.get("lines"));
            if (!u8 || !u8.length) return;
            try {
              const t = arrow.tableFromIPC(u8);
              const d = t.getChild("path").data[0];
              const off = d.valueOffsets, xy = d.children[0].children[0].values;
              for (let i = 0; i < t.numRows; i++) lines.push(xy.subarray(2 * off[d.offset + i], 2 * off[d.offset + i + 1]));
            } catch (e) { lines = []; say("lines: " + e.message); }
          }
          function colorVector() {
            if (!colors) return table.getChild("rgba");
            const child = arrow.makeData({type: new arrow.Uint8(), data: colors});
            const data = arrow.makeData({type: new arrow.FixedSizeList(4, new arrow.Field("c", new arrow.Uint8(), false)), length: N, nullCount: 0, child});
            return arrow.makeVector(data);
          }
          function hitPaths() {
            const out = [];
            if (!geo || !fids || !(hit > 0)) return out;
            for (let i = 0; i < N; i++) if (fids[i] === hit) for (const [s, e] of geo.polys[i]) out.push(geo.xy.subarray(2 * s, 2 * e));
            return out;
          }

          const before = () => cfg.labels_slot || "watername_ocean";
          function layers() {
            const out = [];
            const fieldsOn = !!(cfg.fields_on && table && N);
            // the raster under a paint: cfg.raster_dim (0 = not drawn at all,
            // the agreement reading is the fields alone)
            const rdim = cfg.raster_dim == null ? 0 : cfg.raster_dim;
            const rasterOn = cfg.raster !== false && !(fieldsOn && rdim <= 0);
            out.push(new TileLayer({
              id: "cdl-" + (cfg.rgen || 0),
              getTileData: tileFn,
              tileSize: cfg.tile || 256,
              minZoom: cfg.tile_zmin || 3, maxZoom: cfg.tile_zmax || 15,
              extent: cfg.extent || null,
              // its own switch: the raster draws under the painted polygons too.
              // With the raster off it stays up only to carry the outline tiles,
              // which it does not when a paint is on (the PathLayer has them)
              // its own switch; under the polygons it is at most a BACKDROP, never
              // a second reading (Stephen, 2026-08-26, on CDL hues next to the
              // viridis paint: "perceptually baffling"). With the raster off the
              // layer stays up only to carry the outline tiles, which it does not
              // when a paint is on (the PathLayer has them).
              visible: rasterOn || (cfg.outlines !== false && !fieldsOn),
              opacity: fieldsOn ? rdim : 1,
              refinementStrategy: "no-overlap",
              beforeId: before(),
              renderSubLayers: sub,
            }));
            if (fieldsOn) {
              out.push(new GeoArrowPolygonLayer({
                id: "fields",
                data: table,
                getPolygon: table.getChild("geometry"),
                getFillColor: colorVector(),
                filled: true,
                stroked: false,
                pickable: false,
                _validate: false,
                beforeId: before(),
              }));
              if (cfg.outlines !== false && lines.length) out.push(new PathLayer({
                id: "outlines",
                data: lines,
                getPath: (d) => d,
                positionFormat: "XY",
                getColor: OUTLINE,
                getWidth: 1,
                widthUnits: "pixels",
                widthMinPixels: 1,
                pickable: false,
                beforeId: before(),
              }));
              const hp = hitPaths();
              if (hp.length) out.push(new PathLayer({
                id: "hit",
                data: hp,
                getPath: (d) => d,
                positionFormat: "XY",
                getColor: GOLD,
                getWidth: 3,
                widthUnits: "pixels",
                widthMinPixels: 3,
                pickable: false,
                beforeId: before(),
              }));
            }
            return out;
          }
          function update() { if (overlay) { try { overlay.setProps({layers: layers()}); } catch (e) { say("layers: " + e.message); } } }

          function sendView() {
            if (!map) return;
            const c = map.getCenter();
            model.set("view", JSON.stringify({
              longitude: c.lng, latitude: c.lat, zoom: map.getZoom(),
              w: mapEl.clientWidth, h: mapEl.clientHeight, n: ++seq,
            }));
            model.save_changes();
          }
          function sendPick(i, fid, ll) {
            model.set("pick", JSON.stringify({i, fid, on: hit > 0, lon: ll.lng, lat: ll.lat, gen: cfg.fgen || 0, n: ++seq}));
            model.save_changes();
          }

          function boot() {
            const home = cfg.home || {longitude: -96, latitude: 38.5, zoom: 4};
            map = new maplibregl.Map({
              container: mapEl, style: STYLE,
              center: [home.longitude, home.latitude], zoom: home.zoom,
              attributionControl: {compact: true},
            });
            map.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
            map.addControl(new maplibregl.FullscreenControl(), "top-right");
            overlay = new MapboxOverlay({
              interleaved: true,
              layers: [],
              onError: (e) => say("deck: " + (e && e.message ? e.message : e)),
            });
            map.addControl(overlay);
            // the pick, explicit on pointerup (the film's): the press must not
            // start on a map control and must not move more than 4 px
            let down = null;
            mapEl.addEventListener("pointerdown", (ev) => {
              down = (ev.target.closest && ev.target.closest(".maplibregl-ctrl")) ? null : [ev.clientX, ev.clientY];
            }, true);
            mapEl.addEventListener("pointerup", (ev) => {
              if (!down) return;
              const moved = Math.hypot(ev.clientX - down[0], ev.clientY - down[1]); down = null;
              if (moved > 4 || !map) return;
              const r = mapEl.getBoundingClientRect();
              let ll = null;
              try { ll = map.unproject([ev.clientX - r.left, ev.clientY - r.top]); }
              catch (e) { say("unproject: " + e.message); return; }
              if (!cfg.fields_on || !geo) { sendPick(-1, 0, ll); return; }
              const i = fieldAt(ll.lng, ll.lat);
              const f = i >= 0 ? fids[i] : 0;
              // the same field again, or the basemap, clears
              hit = (f > 0 && f !== hit) ? f : 0;
              update();
              sendPick(i, f, ll);
            }, true);
            if (cfg.debug) window.__aef = {overlay, map, model, get cfg() { return cfg; }, get geo() { return geo; }, get n() { return N; }};
            map.on("load", () => { update(); sendView(); });
            map.on("moveend", sendView);
            map.on("error", (e) => { if (e && e.error && e.error.message) say("map: " + e.error.message); });
            new ResizeObserver(() => { try { map.resize(); } catch (e) {} }).observe(mapEl);
            document.addEventListener("fullscreenchange", () => { setTimeout(() => { try { map.resize(); } catch (e) {} }, 50); });
          }

          model.on("change:config", () => {
            const was = cfg;
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (cfg.height && cfg.height !== was.height && !document.fullscreenElement) mapEl.style.height = cfg.height + "px";
            if (cfg.note !== was.note) say(cfg.note || "");
            update();
          });
          model.on("change:fields", () => { loadFields(); loadColors(); update(); });
          model.on("change:colors", () => { loadColors(); update(); });
          model.on("change:lines", () => { loadLines(); update(); });
          loadFields(); loadColors(); loadLines();
          try { boot(); if (cfg.note) say(cfg.note); }
          catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { try { map && map.remove(); } catch (e) {} };
        }
        export default {render};
        """

    return (DeckMap,)


@app.cell
def _(
    DeckMap,
    EXTENT,
    HOLD: dict,
    HOME,
    LABELS_SLOT,
    TILE_PX,
    TILE_ZMAX,
    TILE_ZMIN,
    json,
):
    # ---- map cell: builds the map ONCE, empty; must never re-run --------------
    deck = DeckMap(config=json.dumps({
        "height": 700, "home": dict(HOME), "raster": True, "outlines": True,
        # the raster's opacity UNDER the painted polygons. 0: a field paint is a
        # reading of CDL x AlphaEarth INSIDE the boundaries, so the raster is not
        # part of it (Stephen, 2026-08-26: "agreement is only fields for this use
        # case subject to change"). Raise it for a backdrop, 1 for full strength.
        "raster_dim": 0.0,
        "labels_slot": LABELS_SLOT, "tile": TILE_PX, "extent": EXTENT,
        "tile_zmin": TILE_ZMIN, "tile_zmax": TILE_ZMAX,
        "fields_on": False, "rgen": 0, "fgen": 0, "debug": True,
    }))
    HOLD.update({
        "ft": None, "box": None, "fids": None, "hit": None, "vs": None,
        "busy": False, "pending": None, "task": None, "loop": None,
        "st": None, "rstate": None, "rgen": 0, "fgen": 0, "ctl_n": None,
        "tiles": {}, "batch": None, "h_view": None, "h_pick": None,
        "sel_html": "", "last_status": "", "last_by_state": {},
    })
    deck
    return (deck,)


@app.cell
def _(HudControls, YEAR0, YEARS, json, mo):
    hud = mo.ui.anywidget(HudControls(years=json.dumps(YEARS), year0=str(YEAR0)))
    hud
    return (hud,)


@app.cell
def _(
    AEF_ZMIN,
    BATCH_S,
    CLASSES,
    FIELD_MAX_KM2,
    FIELD_TILE_Z,
    FIELD_ZOOM,
    FTW_YEARS,
    HOLD: dict,
    HOME,
    SETTLE,
    TILE_CACHE,
    ThreadPoolExecutor,
    VIEW_W,
    YEAR0,
    YEARS,
    aef_best,
    asyncio,
    bbox4326,
    blank_png,
    box_km2,
    cdl_tile_png,
    cname,
    contains,
    deck,
    field_fill,
    field_table,
    ftw_tile_polys,
    ftw_tile_rings,
    hud,
    json,
    legend_for,
    lines_ipc,
    math,
    np,
    pad_box,
    poly_fids,
    polys_ipc,
    rings_png,
    states_in,
    tile_box,
    time,
    urllib,
    view_to_bbox,
):
    # ---- wiring cell: re-runs on every HUD commit; the map cell never re-runs.
    # THREE INDEPENDENT LAYERS: the CDL raster (its switch, crops-only its
    # modifier, on by default), the painted polygons (the paint buttons, none
    # selected = none, and under a paint the raster is not drawn: cfg
    # raster_dim), the field outlines (its own switch, from the field tier).
    # The fields come on at _field_floor(): FIELD_ZOOM, or higher on a canvas
    # big enough that the padded box would blow FIELD_MAX_KM2 there.
    # The HUD's acts happen IN the run (state, recolor, search); the camera and
    # the pick arrive as widget traits and are handled by observers (re-bound
    # every run); the field serve is an asyncio task on the kernel loop (the
    # fold on a worker thread, every trait assignment back on the loop); the
    # CDL tiles are pulled by deck through the widget's custom messages and
    # served ONE BATCH PER VIEW (cdl-ftw.py's serve).
    try:
        _c = json.loads(hud.widget.ctl or "{}")
    except Exception:
        _c = {}
    _year = int(_c.get("year", YEAR0))
    if _year not in YEARS:
        _year = YEAR0
    _paint = _c.get("paint", "viridis") if "paint" in _c else "viridis"
    if _paint not in ("cdl", "agreement", "viridis", "suggests"):
        _paint = None
    _raster = bool(_c.get("raster", True))
    _crops = bool(_c.get("crops", True))
    # THREE INDEPENDENT LAYERS (Stephen, 2026-08-26: "should be able to disable
    # all layers"): the CDL raster (its own switch, crops-only its modifier),
    # the painted polygons (the paint buttons: none selected = no polygons),
    # the field outlines (this switch). The raster's CLIP to P(field) is retired
    # with the old one-switch design (the CDL paint draws the same picture as
    # polygons, and the raster is "a separate product", his words); _clip is
    # still threaded through the serve, so it is one line to bring back.
    _outlines = bool(_c.get("outlines", _c.get("fields", True)))
    _clip = False
    _inv = bool(_c.get("inv", False)) and _paint == "viridis"   # it modifies that paint only
    _sel = tuple(sorted(int(v) for v in (_c.get("sel") or [])))
    _fyear = _year if _year in FTW_YEARS else FTW_YEARS[0]
    # an act is applied ONCE (a re-run of this cell for any other reason must
    # not repeat the last click of the strip)
    _fresh = _c.get("n") != HOLD.get("ctl_n")
    HOLD["ctl_n"] = _c.get("n")
    _act = _c.get("act", "set") if _fresh else "set"
    _q = str(_c.get("q", "")).strip()
    _was = HOLD.get("st") or {}
    _st = {"year": _year, "paint": _paint, "raster": _raster, "crops": _crops, "clip": _clip,
           "outlines": _outlines, "inv": _inv, "sel": _sel, "fyear": _fyear}
    HOLD["st"] = _st

    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass
    _pool = HOLD.setdefault("pool", ThreadPoolExecutor(max_workers=4))

    def _say(msg):
        try:
            hud.widget.status = msg
        except Exception:
            pass

    def _sync_crops(c):
        """Tell the strip whether the CDL raster is on screen, so "crops only"
        (its modifier and nothing else) can go with it: the switch off, or a
        field paint drawing the polygons with raster_dim 0 (the raster not drawn
        at all). Stephen, 2026-08-26, the same shape as highlight disagreement
        belonging to color by agreement."""
        try:
            dim = float(c.get("raster_dim") or 0)
        except (TypeError, ValueError):
            dim = 0.0
        drawn = bool(c.get("raster", True)) and not (bool(c.get("fields_on")) and dim <= 0)
        try:
            hud.widget.rasteron = drawn
        except Exception:
            pass

    def _cfg(**kw):
        c = json.loads(deck.config or "{}")
        c.update(kw)
        deck.config = json.dumps(c)
        _sync_crops(c)

    def _cfg_get(k, default=None):
        try:
            return json.loads(deck.config or "{}").get(k, default)
        except Exception:
            return default

    def _vsd(vs):
        if vs is None:
            return dict(HOME)
        if isinstance(vs, str):
            try:
                vs = json.loads(vs)
            except Exception:
                return dict(HOME)
        try:
            out = {"longitude": float(vs["longitude"]), "latitude": float(vs["latitude"]), "zoom": float(vs["zoom"])}
        except Exception:
            return dict(HOME)
        if vs.get("w") and vs.get("h"):
            out["w"], out["h"] = float(vs["w"]), float(vs["h"])
        return out

    def _chip(code):
        return (f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;"
                f"background:{CLASSES.get(int(code), ('', '#888'))[1]};margin-right:.35rem;vertical-align:-1px'></span>")

    # ---- the panel pieces ---------------------------------------------------
    def _analyze_html(ft):
        maj, prev, alt, agree, kept, scored, sizes = ft["maj"], ft["prev"], ft["alt"], ft["agree"], ft["kept"], ft["scored"], ft["sizes"]
        n = max(1, int(kept.sum()))
        a_ok = agree[scored]
        head = (f"<b>year {ft['year']}</b> · {n:,} crop fields · {sizes[kept].sum() * ft['pxa']:,.0f} ac · "
                + (f"agreement p50 {np.median(a_ok):.2f}, {(a_ok < 0.5).mean() * 100:.0f}% below 0.5" if len(a_ok) else "unscored"))
        td = "padding:.1rem .6rem .1rem 0;white-space:nowrap"
        th = "padding:.1rem .6rem .1rem 0;text-align:left;opacity:.6;font-weight:500"
        rows = []
        codes, counts = np.unique(maj[kept], return_counts=True)
        for code, cnt in sorted(zip(codes, counts), key=lambda t: -t[1]):
            m = kept & (maj == code)
            a = agree[m & scored]
            dis = m & scored & (agree < 0.5)
            sug = ""
            if dis.any():
                u, c = np.unique(alt[dis], return_counts=True)
                sug = cname(u[c.argmax()])
            last = ""
            if ft["prev_year"] is not None:
                u, c = np.unique(prev[m], return_counts=True)
                last = cname(u[c.argmax()]) if u[c.argmax()] > 0 else "(non-crop)"
            rows.append(
                f"<tr><td style='{td}'>{_chip(code)}{cname(code)}</td><td style='{td};text-align:right'>{int(cnt):,}</td>"
                f"<td style='{td};text-align:right'>{sizes[m].sum() * ft['pxa']:,.0f} ac</td>"
                + (f"<td style='{td};text-align:right'>{np.median(a):.2f}</td><td style='{td};text-align:right'>{(a < 0.5).mean() * 100:.0f}%</td>"
                   if len(a) else f"<td style='{td}' colspan=2><span style='opacity:.6'>unscored</span></td>")
                + f"<td style='{td};opacity:.75'>{sug}</td><td style='{td};opacity:.75'>{last}</td></tr>")
        table = (f"<table style='border-collapse:collapse;font-size:13px;margin:.2rem 0'><tr><th style='{th}'>CDL crop</th><th style='{th}'>fields</th>"
                 f"<th style='{th}'>area</th><th style='{th}'>agreement p50</th><th style='{th}'>below 0.5</th>"
                 f"<th style='{th}' title='the crop whose per-view AlphaEarth prototype the disagreeing fields sit closest to; a suggestion relative to this view'>AlphaEarth usually suggests</th>"
                 f"<th style='{th}'>last year usually</th></tr>" + "".join(rows) + "</table>")
        return head + table

    def _selection_panel(ft):
        st = HOLD["st"]
        if not st["sel"]:
            return ""
        maj, alt, agree, kept, scored = ft["maj"], ft["alt"], ft["agree"], ft["kept"], ft["scored"]
        parts = []
        for code in st["sel"]:
            m = kept & ((aef_best(ft) == code) if st["paint"] == "suggests" else (maj == code))
            if not m.any():
                continue
            a = agree[m & scored]
            s = f"<b>{cname(code) if code >= 0 else 'no suggestion'}</b>: {int(m.sum()):,} fields"
            if len(a):
                s += f", agreement p50 {np.median(a):.2f}, {(a < 0.5).mean() * 100:.0f}% below 0.5"
            parts.append(s)
        return " · ".join(parts)

    def _field_story(ft, fid, lon, lat):
        maj, prev, alt, agree, own, oth, purity, sizes = ft["maj"], ft["prev"], ft["alt"], ft["agree"], ft["own"], ft["oth"], ft["purity"], ft["sizes"]
        where = f" at {lat:.4f}, {lon:.4f}"
        ac = sizes[fid] * ft["pxa"]
        if not ft["kept"][fid]:
            why = "too small" if sizes[fid] < 12 else ("not a crop field by the CDL" if maj[fid] == 0 else "no embedding")
            return f"<span style='opacity:.8'>field {fid}{where}: {ac:,.1f} ac, sits out ({why})</span>"
        s = f"{_chip(maj[fid])}<b>{cname(maj[fid])}</b>{where}: {ac:,.1f} ac, CDL purity {purity[fid]:.2f}"
        if ft["prev_year"] is not None:
            s += f"; {ft['prev_year']}: {cname(prev[fid]) if prev[fid] > 0 else 'non-crop'}"
        if ft["scored"][fid]:
            s += f"; agreement <b>{agree[fid]:.2f}</b> (cos own {own[fid]:.3f}, best other {oth[fid]:.3f})"
            if agree[fid] < 0.5 and alt[fid] >= 0:
                s += f"; AlphaEarth suggests it could be <i>{cname(alt[fid])}</i> (relative to this view)"
        else:
            s += "; unscored (its crop has too few fields in view for a prototype)"
        return s

    def _panel(ft):
        return HOLD.get("sel_html", "") + (_selection_panel(ft) if ft is not None else "")

    def _set_panel(html):
        try:
            hud.widget.panel = html
        except Exception:
            pass

    # ---- the CDL raster: one batch per view, PNGs cached by the raster state
    # the raster tiles carry the outlines only when the polygons are NOT up
    # (with a paint on, the polygon layer's PathLayer draws them): as part of
    # the raster state, toggling the outlines under a paint costs no refetch
    _rings_on = _outlines and _paint is None
    _rstate = (_year, _raster, _crops, _clip, _sel, _rings_on)
    _tiles = HOLD["tiles"]

    def _rings_for(fyear, W, S, E, N, z):
        rings, asked, fetched = ftw_tile_rings(states_in(W, S, E, N), fyear, W, S, E, N, z)
        rb = (np.array([[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()] for r in rings])
              if rings else np.zeros((0, 4)))
        return {"rings": rings, "bounds": rb, "n": len(rings), "tiles": asked, "fetched": fetched}

    def _serve_batch(z, state, keys):
        """ONE batch: the whole view's raster tiles at zoom z (worker thread)."""
        t0 = time.time()
        year, raster, crops, clip, sel, rings_on = state
        fyear = year if year in FTW_YEARS else FTW_YEARS[0]
        boxes = [tile_box(z, x, y) for (_st, _z, x, y) in keys]
        W, S, E, N = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes))
        # the fields switch works from the field tier only: below it the raw CDL
        # (crops-only optional), "otherwise we see all of CDL unless crops are masked"
        tier = z >= FIELD_TILE_Z
        rings = _rings_for(fyear, W, S, E, N, z) if (rings_on and tier) else None
        clip = clip and tier
        counts = np.zeros(256, dtype=np.int64)
        pngs = []
        for (_st, _z, x, y) in keys:
            if raster:
                png, c = cdl_tile_png(year, crops, clip, set(sel), z, x, y, rings)
                counts += c
            else:
                png = rings_png(z, x, y, rings) if rings else blank_png()
            pngs.append(png)
        tot = max(1, int(counts.sum()))
        legend = [{"code": int(c), "name": cname(c), "hex": CLASSES[int(c)][1], "pct": round(100 * counts[c] / tot, 1), "p50": "", "note": ""}
                  for c in np.argsort(-counts)[:24] if counts[c] > 0 and int(c) in CLASSES]
        what = ("CDL raster" if raster else "no raster") + (" · crops only" if crops else "") + (" · outlines" if rings is not None else "")
        line = f"z{z} · {len(keys)} tiles · year {year} · {what} · {int((time.time() - t0) * 1000)} ms"
        for key, png in zip(keys, pngs):
            _tiles[key] = png
        if len(_tiles) > TILE_CACHE:
            for _k in list(_tiles)[:TILE_CACHE // 4]:
                _tiles.pop(_k, None)

        def _push():
            HOLD["last_by_state"][state] = line
            if HOLD["ft"] is not None and _cfg_get("fields_on"):
                return   # the field legend and status own the strip
            _say(_raster_line())
            try:
                hud.widget.legend = json.dumps(legend)
            except Exception:
                pass

        _loop = HOLD.get("loop")
        if _loop is not None:
            _loop.call_soon_threadsafe(_push)
        else:
            _push()

    def _view_tiles(z, state):
        vs = _vsd(HOLD.get("vs"))
        W, S, E, N = bbox4326(vs)
        n = 1 << z

        def tx(lon):
            return min(n - 1, max(0, int((lon + 180.0) / 360.0 * n)))

        def ty(lat):
            lat = max(-85.05, min(85.05, lat))
            return min(n - 1, max(0, int((1 - math.log(math.tan(math.radians(lat))
                                                     + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)))

        xs, ys = range(tx(W), tx(E) + 1), range(ty(N), ty(S) + 1)
        if len(xs) * len(ys) > 80:
            return set()
        return {(state, z, x, y) for x in xs for y in ys}

    async def _run_batch(b):
        try:
            await asyncio.sleep(BATCH_S)
            b["closed"] = True
            b["keys"] |= {k for k in _view_tiles(b["z"], b["state"]) if k not in _tiles}
            keys = sorted(b["keys"])
            await asyncio.get_running_loop().run_in_executor(_pool, _serve_batch, b["z"], b["state"], keys)
            if not b["fut"].done():
                b["fut"].set_result(True)
        except asyncio.CancelledError:
            b["closed"] = True
            if not b["fut"].done():
                b["fut"].set_exception(RuntimeError("batch cancelled"))
            raise
        except Exception as _e:
            b["closed"] = True
            if not b["fut"].done():
                b["fut"].set_exception(_e)
            _say(f"serve error: {type(_e).__name__}: {_e}")

    async def _tile_fn(z, x, y):
        state = HOLD["rstate"]
        key = (state, z, x, y)
        if key in _tiles:
            return _tiles[key]
        loop = asyncio.get_running_loop()
        HOLD["loop"] = loop
        b = HOLD.get("batch")
        if b is None or b["closed"] or b["z"] != z or b["state"] != state:
            b = {"z": z, "state": state, "keys": set(), "closed": False, "fut": loop.create_future()}
            b["fut"].add_done_callback(lambda f: f.cancelled() or f.exception())
            b["task"] = loop.create_task(_run_batch(b))
            HOLD["batch"] = b
        b["keys"].add(key)
        await asyncio.shield(b["fut"])
        return _tiles.get(key)

    deck.tile_fn = _tile_fn

    def _field_floor(vsd):
        """The camera zoom the fields can ACTUALLY come on at: FIELD_ZOOM, or
        higher when the canvas is big enough that the padded box would blow
        FIELD_MAX_KM2 there (the box quarters per zoom step). Two knobs used to
        disagree: at FIELD_ZOOM 11 a 1400x700 canvas is 1,170 km2 (under the
        1,500 cap) but a fullscreen 2000x1000 is 2,380 and 2560x1300 is 3,970,
        so a paint sat lit with nothing drawn and the fields looked like they
        unselected themselves (Stephen, 2026-08-26: "layers get uneselected ...
        for no noticable reason"). One rule, one message."""
        km2 = box_km2(pad_box(view_to_bbox(vsd)))
        if km2 <= 0:
            return FIELD_ZOOM
        need = vsd["zoom"] + 0.5 * math.log2(km2 / FIELD_MAX_KM2)
        return max(FIELD_ZOOM, math.ceil(need * 10) / 10)

    def _raster_line():
        """The raster tier's status: the last batch line of the current raster
        state (composed at display time; the cached line carries no note) plus
        why the fields are not on."""
        st = HOLD["st"]
        line = HOLD["last_by_state"].get(HOLD["rstate"]) or (
            f"year {st['year']} · " + ("CDL raster" if st["raster"] else "nothing on") + " · loading …")
        if st["paint"] is None:
            return line + " · pick a field paint"
        _v = _vsd(HOLD.get("vs"))
        _fl = _field_floor(_v)
        if _v["zoom"] < _fl:
            return line + f" · fields from camera z{_fl:.1f} (zoom in)"
        return line

    def _raster_changed():
        HOLD["rstate"] = _rstate
        HOLD["batch"] = None
        HOLD["rgen"] += 1
        _cfg(raster=_raster, outlines=_outlines, rgen=HOLD["rgen"])

    # ---- the fields: the table over the padded view, the polygons for it -----
    def _fields_off(msg=None, note=""):
        """No polygons: the map falls back to the CDL raster, which with a paint
        button lit reads as "color by agreement shows cdl fields" (Stephen,
        2026-08-26). So say it IN the map, not only in the strip below it."""
        if _cfg_get("fields_on") or _cfg_get("note", "") != note:
            _cfg(fields_on=False, note=note)
        try:
            hud.widget.legend = "[]"
        except Exception:
            pass
        _say(msg or _raster_line())

    def _recolor():
        """A paint switch, the ramp reversed, or a legend isolate: one rgba per
        polygon from the held table's LUT, no geometry reload."""
        ft, fids = HOLD["ft"], HOLD["fids"]
        if ft is None or fids is None:
            return
        st = HOLD["st"]
        rgba = field_fill(ft, st["paint"], set(st["sel"]), st["inv"])
        deck.colors = np.ascontiguousarray(rgba[fids]).tobytes()
        try:
            hud.widget.legend = json.dumps(legend_for(ft, st["paint"], st["inv"]))
        except Exception:
            pass

    def _status_for(ft, extra=""):
        t = ft["timing"]
        a = ft["agree"][ft["scored"]]
        sc = (f"agreement p50 {np.median(a):.2f} · {(a < 0.5).mean() * 100:.0f}% below 0.5"
              if len(a) else f"unscored (a crop needs 20 fields in view for a prototype; {len(ft['protos'])} have)")
        return (f"year {ft['year']} · {ft['nfields']:,} crop fields ({ft['nlab']:,} components) · {ft['nscored']:,} scored · {sc}"
                + (f" · FTW {ft['fyear']} footprint" if ft["fyear"] != ft["year"] else "")
                + f"\nftw {t.get('ftw', 0):.1f} s · cdl {t.get('cdl', 0):.1f} · aef read {t.get('aef read', 0):.1f} s ({ft['mos_mb']:.0f} MB) · fold {t.get('aef fold', 0):.1f}" + extra)

    async def _serve_fields(vs, force=False):
        st = HOLD["st"]
        vsd = _vsd(vs)
        z = vsd["zoom"]
        if st["paint"] is None:
            _fields_off()
            return
        view = view_to_bbox(vsd)
        box = pad_box(view)
        floor = _field_floor(vsd)
        if z < floor:
            msg = f"zoom {z:.1f} · fields from camera z{floor:.1f} (zoom in)"
            if floor > FIELD_ZOOM:
                msg += f" · {box_km2(box):,.0f} km² in view here, the fold caps at {FIELD_MAX_KM2:g}"
            _fields_off(msg, note=f"this is the CDL raster · the field paint needs camera z{floor:.1f}")
            return
        if (not force and HOLD["ft"] is not None and HOLD["box"] is not None
                and contains(HOLD["box"], view) and HOLD["ft"]["year"] == st["year"]):
            if not _cfg_get("fields_on"):
                _cfg(fields_on=True, outlines=st["outlines"], note="")
                _recolor()
                _say(HOLD.get("last_status", ""))
            return
        t0 = time.time()
        _say(f"year {st['year']} · folding {box_km2(box):,.0f} km² of fields …")
        loop = asyncio.get_running_loop()
        yr = st["year"]
        ft = await loop.run_in_executor(_pool, lambda: field_table(yr, *box))
        fy = ft["fyear"]
        t1 = time.time()
        polys, lines = await loop.run_in_executor(
            _pool, lambda: (ftw_tile_polys(states_in(*box), fy, *box), _rings_for(fy, *box, AEF_ZMIN)))
        t2 = time.time()
        fids = poly_fids(ft, polys)
        rgba = field_fill(ft, st["paint"], set(st["sel"]), st["inv"])
        ipc = polys_ipc(polys, fids, rgba)
        lipc = lines_ipc(lines["rings"])
        t3 = time.time()
        HOLD["ft"], HOLD["box"], HOLD["fids"], HOLD["hit"], HOLD["sel_html"] = ft, box, fids, None, ""
        HOLD["fgen"] += 1
        deck.colors = b""
        deck.lines = lipc
        deck.fields = ipc
        _cfg(fields_on=True, outlines=st["outlines"], note="", fgen=HOLD["fgen"])
        try:
            hud.widget.legend = json.dumps(legend_for(ft, st["paint"], st["inv"]))
        except Exception:
            pass
        _set_panel(_panel(ft))
        HOLD["last_status"] = _status_for(
            ft, f"\n{len(polys):,} polygons ({int((fids > 0).sum()):,} on a field) · {lines['n']:,} outline pieces · "
                f"{len(ipc) / 1e6:.1f} MB · table {t1 - t0:.1f} s · tiles {t2 - t1:.1f} · arrow {t3 - t2:.1f} · {time.time() - t0:.1f} s")
        _say(HOLD["last_status"])

    async def _refresh(vs):
        """Settle-debounced, coalescing serve (the camera must rest SETTLE s)."""
        if HOLD["busy"]:
            HOLD["pending"] = vs
            return
        HOLD["busy"] = True
        try:
            while True:
                await asyncio.sleep(SETTLE)
                if HOLD["pending"] is not None:
                    vs, HOLD["pending"] = HOLD["pending"], None
                    continue
                await _serve_fields(vs)
                vs = HOLD["pending"]
                if vs is None:
                    return
                HOLD["pending"] = None
        except Exception as exc:
            _say(f"fields failed: {type(exc).__name__}: {exc}")
            raise
        finally:
            HOLD["busy"], HOLD["pending"] = False, None

    def _spawn(coro):
        try:
            return asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            loop = HOLD.get("loop")
            return asyncio.run_coroutine_threadsafe(coro, loop) if loop else None

    def _kick(force=False):
        vs = HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)
        if HOLD["busy"]:
            HOLD["pending"] = vs
        else:
            HOLD["task"] = _spawn(_serve_fields(vs, force=True) if force else _refresh(vs))

    # ---- the camera: every moveend --------------------------------------------
    def _on_view(change):
        vs = change["new"]
        if not vs:
            return
        HOLD["vs"] = _vsd(vs)
        if HOLD["busy"]:
            HOLD["pending"] = vs
            return
        HOLD["task"] = _spawn(_refresh(vs))

    if HOLD.get("h_view") is not None:
        try:
            deck.unobserve(HOLD["h_view"], names="view")
        except ValueError:
            pass
    deck.observe(_on_view, names="view")
    HOLD["h_view"] = _on_view

    # ---- the pick: the browser found the polygon; the kernel tells its story
    def _on_pick(change):
        try:
            p = json.loads(change["new"] or "{}")
        except Exception:
            return
        ft = HOLD["ft"]
        lon, lat = p.get("lon"), p.get("lat")
        if lon is None or lat is None:
            return
        if ft is None or not _cfg_get("fields_on"):
            _say(f"no fields on at {lat:.4f}, {lon:.4f}: pick a field paint and zoom in "
                 f"(camera z{_field_floor(_vsd(HOLD.get('vs'))):.1f})")
            return
        if int(p.get("gen", -1)) != HOLD["fgen"]:
            return   # a click on the previous table
        fid = int(p.get("fid") or 0)
        if fid > 0 and p.get("on"):
            HOLD["hit"] = fid
            HOLD["sel_html"] = (_field_story(ft, fid, lon, lat)
                                + "<hr style='border:none;border-top:1px solid rgba(127,127,127,.25);margin:.3rem 0'>")
        else:
            HOLD["hit"], HOLD["sel_html"] = None, ""
            if fid == 0:
                _say(f"no field at {lat:.4f}, {lon:.4f}" + (" (a polygon off the 10 m grid)" if int(p.get("i", -1)) >= 0 else ""))
        _set_panel(_panel(ft))

    if HOLD.get("h_pick") is not None:
        try:
            deck.unobserve(HOLD["h_pick"], names="pick")
        except ValueError:
            pass
    deck.observe(_on_pick, names="pick")
    HOLD["h_pick"] = _on_pick

    # ---- the search: Photon, then a fly ---------------------------------------
    def _photon_first(query, vs):
        params = {"q": query, "limit": 1, "lang": "en"}
        if isinstance(vs, dict) and vs.get("longitude") is not None:
            params["lon"] = round(vs["longitude"], 4)
            params["lat"] = round(vs["latitude"], 4)
        url = "https://photon.komoot.io/api/?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "cdl-ftw-zarr-marimo cdl aef deck notebook"})
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.load(r)
        feats = data.get("features") or []
        if not feats:
            return None
        f = feats[0]
        p = f.get("properties", {})
        lon, lat = f["geometry"]["coordinates"][:2]
        name = ", ".join(str(v) for v in (p.get("name"), p.get("city"), p.get("state")) if v) or query
        return name, float(lon), float(lat), p.get("extent")

    async def _search(q):
        vs = _vsd(HOLD.get("vs"))
        try:
            hit = await asyncio.get_running_loop().run_in_executor(_pool, _photon_first, q, vs)
        except Exception as e:
            _say(f"search error: {type(e).__name__}: {e}")
            return
        if hit is None:
            _say(f"no match: {q}")
            return
        name, lon, lat, ext = hit
        w = vs.get("w") or VIEW_W
        if ext and len(ext) == 4:
            span = max(abs(ext[2] - ext[0]), abs(ext[1] - ext[3]) * 2, 0.01)
            zoom = math.log2(360.0 * (w / 512) / span) - 0.3
        else:
            zoom = 12.5
        zoom = max(3.5, min(15.0, zoom))
        deck.send({"kind": "fly", "lon": lon, "lat": lat, "zoom": zoom, "duration": 2000})
        _say(f"→ {name} · zoom {zoom:.1f}")

    # ---- the acts of THIS run ---------------------------------------------------
    if _act == "clear":
        HOLD["sel_html"] = ""
        _set_panel(_panel(HOLD["ft"]))
    if _act == "analyze":
        ft = HOLD["ft"]
        _set_panel(_analyze_html(ft) if ft is not None and _paint is not None and _cfg_get("fields_on")
                   else f"<span style='opacity:.7'>no fields in view (pick a field paint and zoom in, "
                        f"camera z{_field_floor(_vsd(HOLD.get('vs'))):.1f})</span>")
    if _act == "search" and _q:
        _say(f"searching: {_q}")
        HOLD["stask"] = _spawn(_search(_q))
    if _act == "refresh":
        _tiles.clear()
        HOLD["last_by_state"].clear()
        _raster_changed()
        _kick(force=True)
    elif HOLD.get("rstate") != _rstate:
        _raster_changed()
        if not _cfg_get("fields_on"):
            _say(_raster_line())   # the tiles may all be cached: no batch, no line otherwise

    if _act != "refresh":
        if not _was or _was["year"] != _year:
            _kick(force=True)
        elif _was["paint"] != _paint:
            if _paint is None:
                _fields_off()
            elif HOLD["ft"] is None:
                _kick(force=True)
            else:
                _kick()
                _recolor()
        elif _paint is not None and HOLD["ft"] is not None:
            if (_was["inv"], _was["sel"]) != (_inv, _sel):
                _recolor()
                _set_panel(_panel(HOLD["ft"]))
            if _was["outlines"] != _outlines:
                _cfg(outlines=_outlines)
    _sync_crops(json.loads(deck.config or "{}"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Under the map

    The table below is the LAST batch's fields, per crop (press the button
    after the map settles with a field paint on).
    """)
    return


@app.cell
def _(mo):
    tables_btn = mo.ui.run_button(label="table for the current view")
    tables_btn
    return (tables_btn,)


@app.cell
def _(HOLD: dict, cname, mo, np, tables_btn):
    mo.stop(not tables_btn.value or HOLD.get("ft") is None, mo.md("*no fields folded yet*"))
    _ft = HOLD["ft"]
    _maj, _alt, _agree, _kept, _scored, _sizes, _prev = _ft["maj"], _ft["alt"], _ft["agree"], _ft["kept"], _ft["scored"], _ft["sizes"], _ft["prev"]
    _rows = []
    for _code, _cnt in zip(*np.unique(_maj[_kept], return_counts=True)):
        _m = _kept & (_maj == _code)
        _a = _agree[_m & _scored]
        _dis = _m & _scored & (_agree < 0.5)
        _u, _cc = np.unique(_alt[_dis], return_counts=True) if _dis.any() else (np.array([]), np.array([]))
        _pu, _pc = np.unique(_prev[_m], return_counts=True)
        _rows.append({
            "crop": cname(_code), "fields": int(_cnt), "acres": round(float(_sizes[_m].sum() * _ft["pxa"]), 1),
            "agree_p50": round(float(np.median(_a)), 3) if len(_a) else None,
            "pct_below_half": round(float((_a < 0.5).mean() * 100), 1) if len(_a) else None,
            "aef_usually_suggests": cname(_u[_cc.argmax()]) if len(_u) else "",
            "last_year_usually": (cname(_pu[_pc.argmax()]) if _pu[_pc.argmax()] > 0 else "(non-crop)") if _ft["prev_year"] else "",
        })
    _rows.sort(key=lambda r: -r["fields"])
    per_crop = mo.ui.table(_rows, selection=None)
    per_crop
    return


if __name__ == "__main__":
    app.run()
