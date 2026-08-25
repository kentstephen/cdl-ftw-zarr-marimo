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
#     "traitlets",
#     "pillow",
# ]
# ///
"""CDL, backed or not by AlphaEarth, on the Fields of the World: the deck.gl build.

Zoomed out, at ANY zoom, the map is the Cropland Data Layer as a picture:
tiles the kernel renders from the icechunk store's majority pyramid (30 m
2008-2025, 10 m 2024-2025), the same read as cdl-ftw.py, with a crops-only
mask and the P(field) clip as toggles. Zoomed in (camera z12.5+, a field
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
     AlphaEarth suggests (a disagreeing field takes the runner-up crop's
     color, agreeing fields go quiet). Highlight disagreement reverses the
     ramp. (An "agreement" paint, CDL color with alpha by agreement, is
     commented out in the strip: near-binary scores made it read as plain
     CDL; the code path stays in field_fill.) The CDL raster does not draw
     under the fields (a faded field fades to the basemap). A click outlines
     the field in gold; the same field again, or the basemap, clears it.

The map is its OWN deck.gl widget (the NLCD deck notebook's chassis, hexagon
layers removed): maplibre + deck interleaved under the labels, two TileLayers
whose tiles the kernel renders on request over anywidget custom messages (the
CDL raster; the field paint from z12), real layer ids, a click that is deck's
pick (lon/lat) answered by the kernel from the field grid. No lonboard, so no
JS patch. No SQL: the joins are positional (every dataset is a raster on a
known grid; the field id image from ndimage.label is the key and np.bincount
is the groupby), see the "join" cell.

Data (all anonymous on source.coop): the CDL icechunk repo
(chill/usda-cropland-data-layer), FTW P(field) Zarr + per-state PMTiles
outlines (tge-labs/ftw-global-data), the AlphaEarth mosaic (tge-labs/
aef-mosaic; "The AlphaEarth Foundations Satellite Embedding dataset is
produced by Google and Google DeepMind", CC-BY 4.0).

Run from THIS repo's venv:  uv sync && uv run marimo edit cdl-aef-deck.py
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
    masks. Zoomed in past **z12.5** with a field paint on, each **Fields of the
    World** field gets its CDL majority crop, last year's crop, and its mean
    **AlphaEarth** vector; per view every crop with enough fields gets a
    prototype, and a field's *agreement* is how much closer its vector sits to
    its own crop's prototype than to the best other one.

    - **CDL** paint: each field its majority crop's color.
    - **color by agreement** paint: viridis on the agreement (bright = agrees, dark = a lead).
    - **AlphaEarth suggests** paint: a disagreeing field takes the color of the
      crop AlphaEarth puts it closest to (relative to this view); agreeing fields
      go quiet grey.

    Click a field for its story; *analyze what's in view* for the per-crop table.

    | leg | source | read |
    |---|---|---|
    | crops | CDL icechunk repo, 30 m 2008-2025 + 10 m 2024-2025, majority pyramid, EPSG:5070 | xarray window + closed-form Albers |
    | fields | FTW P(field) Zarr, 10 m + pyramid; per-state PMTiles outlines | xarray window, ndimage.label; ranged GETs + MVT decode |
    | embeddings | `tge-labs/aef-mosaic`, 10 m, 64 x int8, 2017-2025, no pyramid | xarray window by 256-px chunk, cached |
    | score | per-view prototypes, sigmoid margin | numpy |
    """)
    return


@app.cell
def _(os, tempfile):
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

    # the field tier
    FIELD_ZOOM = 12.5         # camera zoom from which a field paint folds the view
    FIELD_MAX_KM2 = 450.0     # padded box cap (AEF is a native 10 m read: ~0.6 MB / km2)
    PAD = 1.15                # fold box beyond the camera footprint
    SETTLE = 0.35             # seconds the camera must rest before a fold
    MIN_FIELD_PX = 12         # ~0.3 ac at 10 m: smaller components sit out
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
    OUTLINE = (40, 40, 40, 200)

    RASTER_TILE = 256
    VIEW_W, VIEW_H = 1400, 720
    EXTENT = [-125.0, 24.0, -66.5, 49.8]
    LABELS_SLOT = "watername_ocean"
    HOME = {"longitude": -121.45, "latitude": 37.95, "zoom": 12.8}  # the Delta west of Stockton
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
        AGREE_CMAP,
        ALPHA_FLAT,
        ALPHA_MAX,
        ALPHA_MIN,
        ALPHA_RAMP,
        CACHE_DIR,
        CDL_BUCKET,
        CDL_ENDPOINT,
        CDL_PREFIX,
        CDL_YEARS,
        DIM_ALPHA,
        EXTENT,
        FIELD_MAX_KM2,
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
        MIN_CLASS_FIELDS,
        MIN_CROP_FRAC,
        MIN_FIELD_PX,
        OUTLINE,
        PAD,
        PANEL_MIN_AC,
        PX_PER,
        QUIET,
        RAMPS,
        RASTER_TILE,
        SC_BUCKET,
        SETTLE,
        TAU,
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
def _(AGREE_CMAP, CLASSES, NONCROP_CODES, PAD, RAMPS, VIEW_H, VIEW_W, math, np):
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
                    for run in np.split(idx, cuts):
                        pts = a[run[0]:run[-1] + 2]
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

    return ftw_tile_rings, states_in


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
    HOLD,
    Image,
    LEVELS,
    LEVELS10,
    NONCROP,
    PX_PER,
    RASTER_TILE,
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
        for by in range(r0 // _BLK, (r1 - 1) // _BLK + 1):
            for bx in range(c0 // _BLK, (c1 - 1) // _BLK + 1):
                a = _block(g, ds, k, year, by, bx)
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
        mpp = 156543.03392 * math.cos(math.radians(lat)) / 2 ** z * (256 / RASTER_TILE)
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

    def cdl_tile_png(year, crops_only, fields, sel, z, x, y):
        """PNG bytes for Web Mercator tile (z, x, y): the CDL from the level the
        zoom deserves; nodata alpha 0; crops_only drops the non-crop classes;
        fields drops pixels outside P(field) >= 0.5; sel isolates classes."""
        W, S, E, N = tile_box(z, x, y)
        if E < EXTENT[0] or W > EXTENT[2] or N < EXTENT[1] or S > EXTENT[3]:
            return blank_png()
        T = RASTER_TILE
        lons = W + (np.arange(T) + 0.5) * (E - W) / T
        lats = N - (np.arange(T) + 0.5) * (N - S) / T
        LON, LAT = np.meshgrid(lons, lats)
        k = level_for(year, z, (N + S) / 2)
        code = cdl_codes_at(year, LON, LAT, k)
        if not code.any():
            return blank_png()
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
        if sel:
            keep = np.zeros(256, dtype=bool)
            keep[list(sel)] = True
            code = np.where(keep[code], code, 0)
        out = np.zeros((T, T, 4), dtype=np.uint8)
        out[..., :3] = CLASS_RGB[code]
        out[..., 3] = np.where(code > 0, 255, 0)
        buf = io.BytesIO()
        Image.fromarray(out, "RGBA").save(buf, format="PNG", optimize=False, compress_level=4)
        return buf.getvalue()

    return blank_png, cdl_codes_at, cdl_tile_png


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
    HOLD,
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
    EXTENT,
    FTW_RES,
    FTW_Y0,
    Image,
    ImageDraw,
    OUTLINE,
    QUIET,
    RAMP_HEX,
    RASTER_TILE,
    blank_png,
    cname,
    io,
    ndimage,
    np,
    tile_box,
):
    # ---- the field paints: one rgba LUT by field id, and the tiles drawn from it
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
        else:  # "suggests": the runner-up crop's color where AEF disagrees
            dis = scored & (agree < 0.5) & (alt >= 0)
            quiet = kept & ~dis
            rgba[quiet, :3] = QUIET
            rgba[quiet, 3] = np.where(scored[quiet], 90, 45)
            rgba[dis, :3] = CLASS_RGB[alt[dis]]
            rgba[dis, 3] = ALPHA_FLAT
            key = alt
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
            dis = scored & (agree < 0.5) & (alt >= 0)
            tot = max(1, int(dis.sum()))
            codes, nn = np.unique(alt[dis], return_counts=True)
            for code, cnt in sorted(zip(codes, nn), key=lambda t: -t[1]):
                items.append({"code": int(code), "name": cname(code), "hex": CLASSES[int(code)][1],
                              "pct": round(100 * int(cnt) / tot, 1), "p50": "", "note": f"{int(cnt):,}"})
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

    GOLD = (255, 200, 40, 255)
    HIT_PAD, HIT_W = 6, 3   # lattice pad (px) so the outline erosion sees past the seam; outline width

    def field_tile_png(ft, rgba, rings, z, x, y, hit=None):
        """PNG for tile (z, x, y) from the field-id image and the paint LUT;
        outline polylines drawn on top with PIL; the picked field (`hit`) gets
        a gold outline, its fill unchanged."""
        W, S, E, N = tile_box(z, x, y)
        bW, bS, bE, bN = ft["box"]
        if E < bW or W > bE or N < bS or S > bN or E < EXTENT[0] or W > EXTENT[2]:
            return blank_png()
        T, P = RASTER_TILE, HIT_PAD
        lons = W + (np.arange(-P, T + P) + 0.5) * (E - W) / T
        lats = N - (np.arange(-P, T + P) + 0.5) * (N - S) / T
        jx = np.floor((lons + 180.0) / FTW_RES).astype(np.int64) - ft["fx0"]
        jy = np.floor((FTW_Y0 - lats) / FTW_RES).astype(np.int64) - ft["fy0"]
        lab = ft["lab"]
        okx = (jx >= 0) & (jx < lab.shape[1])
        oky = (jy >= 0) & (jy < lab.shape[0])
        ids = np.zeros((T + 2 * P, T + 2 * P), dtype=np.int32)
        if okx.any() and oky.any():
            ids[np.ix_(oky, okx)] = lab[np.ix_(jy[oky], jx[okx])]
        out = rgba[ids]
        if hit is not None and hit > 0:
            hm = ids == hit
            if hm.any():
                edge = hm & ~ndimage.binary_erosion(hm, iterations=HIT_W, border_value=1)
                out[edge] = GOLD
        out = out[P:P + T, P:P + T]
        img = Image.fromarray(np.ascontiguousarray(out), "RGBA")
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
        return buf.getvalue()

    return field_fill, field_tile_png, legend_for


@app.cell
def _(anywidget, traitlets):
    class HudControls(anywidget.AnyWidget):
        """The strip under the map (the NLCD deck notebook's, reworked): the
        raster switches (CDL raster, crops only, fields clip), the field paint
        (one of CDL / agreement / AlphaEarth suggests, or none), its modifiers
        (highlight disagreement, color by agreement, outlines), the year, the
        pickable legend, analyze, labels, search; panel and status lines. The
        one element docks into the map's fullscreen."""

        ctl = traitlets.Unicode("").tag(sync=True)
        years = traitlets.Unicode("[]").tag(sync=True)
        year0 = traitlets.Unicode("2024").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)

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
          // reload or a re-render rebuilds this JS, the kernel keeps serving the
          // old switches: "fields clip comes on without selecting", 2026-08-25)
          let last = {};
          try { last = JSON.parse(model.get("ctl") || "{}") || {}; } catch (e) { last = {}; }
          const has = (k) => Object.prototype.hasOwnProperty.call(last, k);
          let paint = has("paint") ? last.paint : "viridis";
          let raster = has("raster") ? !!last.raster : true;
          let crops = has("crops") ? !!last.crops : false;
          let clip = has("clip") ? !!last.clip : false;
          let outlines = has("outlines") ? !!last.outlines : true;
          const sel = new Set(Array.isArray(last.sel) ? last.sel : []);
          let seq = has("n") ? (last.n | 0) : 0;
          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({
              act: act, paint: paint, sel: Array.from(sel), inv: inv.checked,
              raster: raster, crops: crops, clip: clip, outlines: outlines, labels: labelsOn,
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
          // the year: both sides have it (2017-2025); the CDL is 10 m for 2024-2025
          const yearBox = document.createElement("span");
          yearBox.style.cssText = "display:inline-flex;gap:.3rem;align-items:center";
          const yl = document.createElement("span"); yl.textContent = "year";
          const yearSel = document.createElement("select");
          yearSel.style.cssText = "font:13px ui-sans-serif,system-ui,sans-serif;padding:.1rem .3rem;border-radius:4px;border:1px solid rgba(127,127,127,.45);background:transparent;color:inherit";
          let yrs = [];
          try { yrs = JSON.parse(model.get("years") || "[]"); } catch (e) { yrs = []; }
          yrs.forEach((y) => { const o = document.createElement("option"); o.value = String(y); o.textContent = String(y); yearSel.appendChild(o); });
          yearSel.value = String(has("year") ? last.year : (model.get("year0") || yrs[yrs.length - 1] || ""));
          yearSel.addEventListener("change", () => send("year"));
          yearBox.append(yl, yearSel);
          // the raster: on/off and its two masks
          const rasterBox = document.createElement("span");
          rasterBox.style.cssText = "display:inline-flex;gap:.6rem;align-items:center";
          const [rasLab] = mkChk("CDL raster", "the Cropland Data Layer as tiles, at any zoom", raster, (v) => { raster = v; send("raster"); });
          const [cropLab] = mkChk("crops only", "raster: drop the non-crop classes", crops, (v) => { crops = v; send("raster"); });
          const [clipLab] = mkChk("fields clip", "raster: only pixels inside a P(field) >= 0.5 cell (the FTW pyramid, at every zoom)", clip, (v) => { clip = v; send("raster"); });
          rasterBox.append(rasLab, cropLab, clipLab);
          // the field paint: one at a time, click the one that is on to hide it
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
            mkPaint("cdl", "CDL", "each field its CDL majority crop's color (z12.5+); click again to hide"),
            // "agreement" (CDL color, alpha by agreement) is OUT for now (Stephen,
            // 2026-08-25: the score is near-binary, 92% of fields at full alpha, the
            // paint read as plain CDL; color by agreement does the job on its own):
            // mkPaint("agreement", "agreement", "the CDL color, alpha follows how well AlphaEarth backs the crop (z12.5+); click again to hide"),
            mkPaint("viridis", "color by agreement", "viridis on the agreement value: bright = agrees, dark = a lead (z12.5+); highlight disagreement reverses; click again to hide"),
            mkPaint("suggests", "AlphaEarth suggests", "a disagreeing field takes the color of the crop AlphaEarth puts it closest to, relative to this view; agreeing fields go quiet (z12.5+); click again to hide"),
          ];
          const [invLab, inv] = mkChk("highlight disagreement", "color by agreement: reverse the ramp (bright = disagrees)", has("inv") ? !!last.inv : false, () => send("set"));
          const [outLab] = mkChk("outlines", "field outlines from the FTW PMTiles, drawn on the field tiles", outlines, (v) => { outlines = v; send("set"); });
          const stylePaint = () => { paintBtns.forEach(([k, b]) => onCss(b, k === paint)); };
          stylePaint();
          paintBox.append(pl, ...paintBtns.map(([, b]) => b), invLab, outLab);
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
              b.title = it.pct + "% of fields" + (it.p50 ? " · agreement p50 " + it.p50 : "");
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
          anBox.append(anB, clB);
          let labelsOn = has("labels") ? !!last.labels : true;
          const lbB = document.createElement("button");
          lbB.textContent = "labels"; lbB.style.cssText = btnCss; lbB.title = "place labels over the map, on or off";
          const styleLb = () => onCss(lbB, labelsOn);
          styleLb();
          lbB.onclick = () => { labelsOn = !labelsOn; styleLb(); send("labels", { labels: labelsOn }); };
          anBox.append(lbB);
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
          anBox.append(search);
          box.append(yearBox, rasterBox, paintBox, anBox, legendBox);
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
            root.querySelectorAll("*").forEach((n) => { if (n.shadowRoot) killOld(n.shadowRoot); });
          };
          killOld(document);
          el.appendChild(wrap);
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
          };
        }
        export default { render };
        """

    return (HudControls,)


@app.cell
def _(anywidget, asyncio, traitlets):
    class DeckMap(anywidget.AnyWidget):
        """The map: maplibre (Carto Positron, interleaved) with deck.gl 9.3.10
        from esm.sh drawing INSIDE it under the label layers (the NLCD deck
        notebook's chassis, hexagon layers removed). Two TileLayers with real
        ids whose tiles the kernel renders on request (anywidget custom
        messages, PNG bytes back): `cdl-<rgen>` at any zoom and
        `fields-<fgen>` from tile z13 inside the folded box; a new generation
        id makes deck refetch (the kernel's state changed). Kernel -> browser:
        `config` (JSON). Browser -> kernel: `view` (lon/lat/zoom + canvas w/h
        on every moveend) and `pick` (the click's lon/lat)."""

        config = traitlets.Unicode("{}").tag(sync=True)
        view = traitlets.Unicode("").tag(sync=True)
        pick = traitlets.Unicode("").tag(sync=True)

        def __init__(self, **kw):
            super().__init__(**kw)
            self.tile_fn = None  # async (layer, z, x, y) -> PNG bytes; the wiring sets it
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
                png = await self.tile_fn(c.get("layer"), int(c["z"]), int(c["x"]), int(c["y"])) if self.tile_fn else None
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
        import {BitmapLayer} from "https://esm.sh/@deck.gl/layers@9.3.10?deps=@deck.gl/core@9.3.10,apache-arrow@18.1.0";
        import {TileLayer} from "https://esm.sh/@deck.gl/geo-layers@9.3.10?deps=@deck.gl/core@9.3.10,@deck.gl/extensions@9.3.10,@deck.gl/layers@9.3.10,@deck.gl/mesh-layers@9.3.10,apache-arrow@18.1.0";

        const STYLES = {
          labels: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        };

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
          mapEl.style.cssText = "width:100%;height:" + (cfg.height || 720) + "px;background:#f4f2ee";
          const note = document.createElement("div");
          note.style.cssText = "position:absolute;left:8px;top:8px;z-index:5;font:11px ui-monospace,Menlo,monospace;" +
            "color:#333;background:rgba(255,255,255,.85);padding:2px 6px;border-radius:3px;pointer-events:none;display:none";
          root.append(mapEl, note);
          el.append(css, root);
          const say = (t) => { note.textContent = t; note.style.display = t ? "block" : "none"; };

          let seq = 0, map = null, overlay = null;

          // tiles: ask the kernel, get a PNG back on the custom-message channel
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
          const tileFn = (layer) => ({index, signal}) => new Promise((resolve) => {
            const id = ++tseq;
            pending.set(id, {resolve});
            model.send({kind: "tile", id, layer, x: index.x, y: index.y, z: index.z});
            if (signal) signal.addEventListener("abort", () => { pending.delete(id); resolve(null); });
          });
          const sub = (p) => {
            if (!p.data) return null;
            const {west, south, east, north} = p.tile.bbox;
            return new BitmapLayer(p, {data: null, image: p.data, bounds: [west, south, east, north]});
          };

          function layers() {
            const out = [];
            out.push(new TileLayer({
              id: "cdl-" + (cfg.rgen || 0),
              getTileData: tileFn("cdl"),
              tileSize: cfg.tile || 256,
              minZoom: 3, maxZoom: 15,
              extent: cfg.extent || null,
              // off while the fields draw: a faded field must fade to the basemap,
              // not to the same CDL color underneath
              visible: cfg.raster !== false && !cfg.fields_on,
              refinementStrategy: "no-overlap",
              beforeId: cfg.labels_slot || "watername_ocean",
              renderSubLayers: sub,
            }));
            if (cfg.fields_on && cfg.fbox) out.push(new TileLayer({
              id: "fields-" + (cfg.fgen || 0),
              getTileData: tileFn("fields"),
              tileSize: cfg.tile || 256,
              minZoom: 12, maxZoom: 16,
              extent: cfg.fbox,
              visible: true,
              refinementStrategy: "no-overlap",
              beforeId: cfg.labels_slot || "watername_ocean",
              renderSubLayers: sub,
            }));
            return out;
          }
          function update() { if (overlay) overlay.setProps({layers: layers()}); }

          function labels(on) {
            if (!map || !map.isStyleLoaded()) return;
            const st = map.getStyle();
            if (!st || !st.layers) return;
            st.layers.forEach((l) => {
              if (l.layout && l.layout["text-field"] !== undefined)
                map.setLayoutProperty(l.id, "visibility", on ? "visible" : "none");
            });
          }

          function sendView() {
            if (!map) return;
            const c = map.getCenter();
            model.set("view", JSON.stringify({
              longitude: c.lng, latitude: c.lat, zoom: map.getZoom(),
              w: mapEl.clientWidth, h: mapEl.clientHeight, n: ++seq,
            }));
            model.save_changes();
          }

          function boot() {
            const home = cfg.home || {longitude: -96, latitude: 38.5, zoom: 4};
            map = new maplibregl.Map({
              container: mapEl, style: STYLES.labels,
              center: [home.longitude, home.latitude], zoom: home.zoom,
              attributionControl: {compact: true},
            });
            map.addControl(new maplibregl.NavigationControl({showCompass: false}), "top-right");
            map.addControl(new maplibregl.FullscreenControl(), "top-right");
            overlay = new MapboxOverlay({
              interleaved: true,
              layers: [],
              onClick: (info) => {
                // the click's lon/lat; the kernel answers from the field grid
                model.set("pick", JSON.stringify({
                  lon: info && info.coordinate ? info.coordinate[0] : null,
                  lat: info && info.coordinate ? info.coordinate[1] : null, n: ++seq}));
                model.save_changes();
              },
              onError: (e) => say("deck: " + (e && e.message ? e.message : e)),
            });
            map.addControl(overlay);
            if (cfg.debug) window.__aef = {overlay, map, model, get cfg() { return cfg; }};
            map.on("load", () => { labels(cfg.labels !== false); update(); sendView(); });
            map.on("moveend", sendView);
            map.on("error", (e) => { if (e && e.error && e.error.message) say("map: " + e.error.message); });
            new ResizeObserver(() => { try { map.resize(); } catch (e) {} }).observe(mapEl);
            document.addEventListener("fullscreenchange", () => { setTimeout(() => { try { map.resize(); } catch (e) {} }, 50); });
          }

          model.on("change:config", () => {
            const was = cfg;
            try { cfg = JSON.parse(model.get("config") || "{}"); } catch (e) { cfg = {}; }
            if (cfg.height && cfg.height !== was.height && !document.fullscreenElement) mapEl.style.height = cfg.height + "px";
            if (cfg.labels !== was.labels) labels(cfg.labels !== false);
            update();
          });
          try { boot(); }
          catch (e) { say("boot: " + e.message); console.error(e); }
          return () => { try { map && map.remove(); } catch (e) {} };
        }
        export default {render};
        """

    return (DeckMap,)


@app.cell
def _(DeckMap, EXTENT, HOLD, HOME, LABELS_SLOT, RASTER_TILE, YEAR0, json):
    # ---- the map: built ONCE, empty; never re-runs for a parameter -----------------
    deck = DeckMap(config=json.dumps({
        "height": 720, "home": dict(HOME), "raster": True, "labels": True,
        "labels_slot": LABELS_SLOT, "tile": RASTER_TILE, "extent": EXTENT,
        "fields_on": False, "fbox": None, "rgen": 0, "fgen": 0,
    }))
    HOLD.update({
        "ft": None, "box": None, "vs": None, "busy": False, "pending": None, "task": None, "loop": None,
        "year": YEAR0, "paint": "viridis", "raster": True, "crops": False, "clip": False,
        "outlines": True, "inv": False, "labels": True,
        "sel": set(), "hit": None, "rings": None, "rgba": None, "rgen": 0, "fgen": 0,
        "tiles": {}, "h_cam": None, "h_ctl": None, "h_pick": None,
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
    CLASSES,
    FIELD_MAX_KM2,
    FIELD_ZOOM,
    FTW_TILE_ZMAX,
    HOLD,
    HOME,
    SETTLE,
    ThreadPoolExecutor,
    VIEW_W,
    asyncio,
    box_km2,
    cdl_tile_png,
    cname,
    contains,
    deck,
    field_fill,
    field_table,
    field_tile_png,
    ftw_tile_rings,
    hud,
    json,
    legend_for,
    math,
    np,
    pad_box,
    states_in,
    time,
    urllib,
    view_to_bbox,
):
    # ---- wiring: the camera loop, the tiles, the strip. Re-runs freely
    # (un-observes its old handlers first); the map cell never re-runs.
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

    def _cfg(**kw):
        c = json.loads(deck.config or "{}")
        c.update(kw)
        deck.config = json.dumps(c)

    def _vsd(vs):
        if vs is None:
            return dict(HOME)
        if isinstance(vs, str):
            try:
                vs = json.loads(vs)
            except Exception:
                return dict(HOME)
        out = {"longitude": float(vs["longitude"]), "latitude": float(vs["latitude"]), "zoom": float(vs["zoom"])}
        if vs.get("w") and vs.get("h"):
            out["w"], out["h"] = float(vs["w"]), float(vs["h"])
        return out

    # ---- tiles: the widget asks (layer, z, x, y); rendered on a worker thread,
    # cached by the state that produced them
    _tiles = HOLD["tiles"]

    async def _tile_fn(layer, z, x, y):
        if layer == "cdl":
            key = ("cdl", HOLD["year"], HOLD["crops"], HOLD["clip"], tuple(sorted(HOLD["sel"])), z, x, y)
            if key not in _tiles:
                yr, cr, cl, sl = HOLD["year"], HOLD["crops"], HOLD["clip"], set(HOLD["sel"])
                png = await asyncio.get_running_loop().run_in_executor(
                    _pool, lambda: cdl_tile_png(yr, cr, cl, sl, z, x, y))
                _tiles[key] = png
        else:
            ft, rgba, rings = HOLD["ft"], HOLD["rgba"], HOLD["rings"]
            if ft is None or rgba is None:
                return None
            key = ("f", HOLD["fgen"], z, x, y)
            if key not in _tiles:
                png = await asyncio.get_running_loop().run_in_executor(
                    _pool, lambda: field_tile_png(ft, rgba, rings if HOLD["outlines"] else None, z, x, y, HOLD["hit"]))
                _tiles[key] = png
        if len(_tiles) > 4000:
            for _k in list(_tiles)[:800]:
                _tiles.pop(_k, None)
        return _tiles[key]

    deck.tile_fn = _tile_fn

    def _raster_changed():
        HOLD["rgen"] += 1
        _cfg(raster=HOLD["raster"], rgen=HOLD["rgen"])

    def _fields_off(msg):
        if HOLD["ft"] is not None or json.loads(deck.config).get("fields_on"):
            _cfg(fields_on=False)
        HOLD["ft"], HOLD["box"], HOLD["hit"], HOLD["rgba"] = None, None, None, None
        try:
            hud.widget.legend = "[]"
        except Exception:
            pass
        _say(msg)

    def _paint():
        """The paint LUT for the held field table; a new generation makes the
        widget refetch the field tiles (ms each: a LUT lookup and a PNG)."""
        ft = HOLD["ft"]
        if ft is None:
            return
        HOLD["rgba"] = field_fill(ft, HOLD["paint"], HOLD["sel"], HOLD["inv"])
        HOLD["fgen"] += 1
        _cfg(fields_on=True, fbox=list(ft["box"]), fgen=HOLD["fgen"])
        try:
            hud.widget.legend = json.dumps(legend_for(ft, HOLD["paint"], HOLD["inv"]))
        except Exception:
            pass

    def _rings_for(ft):
        """Outline polylines over the field box (the PMTiles at z13), with
        their bounds for the per-tile cull."""
        W, S, E, N = ft["box"]
        rings, asked, fetched = ftw_tile_rings(states_in(W, S, E, N), ft["fyear"], W, S, E, N, FTW_TILE_ZMAX)
        rb = (np.array([[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()] for r in rings])
              if rings else np.zeros((0, 4)))
        return {"rings": rings, "bounds": rb, "n": len(rings), "tiles": asked, "fetched": fetched}

    def _status_for(ft, extra=""):
        t = ft["timing"]
        a = ft["agree"][ft["scored"]]
        sc = (f"agreement p50 {np.median(a):.2f} · {(a < 0.5).mean() * 100:.0f}% below 0.5"
              if len(a) else f"unscored (a crop needs 20 fields in view for a prototype; {len(ft['protos'])} have)")
        return (f"year {ft['year']} · {ft['nfields']:,} crop fields ({ft['nlab']:,} components) · {ft['nscored']:,} scored · {sc}"
                f"\nftw {t.get('ftw', 0):.1f} s · label {t.get('label', 0):.1f} · cdl {t.get('cdl', 0):.1f} · "
                f"aef read {t.get('aef read', 0):.1f} s ({ft['mos_mb']:.0f} MB) · fold {t.get('aef fold', 0):.1f} · score {t.get('score', 0):.2f}" + extra)

    async def _serve(vs, force=False):
        vsd = _vsd(vs)
        z = vsd["zoom"]
        if HOLD["paint"] is None:
            _fields_off(f"zoom {z:.1f} · CDL raster only · pick a field paint")
            return
        if z < FIELD_ZOOM:
            _fields_off(f"zoom {z:.1f} · CDL raster · zoom in past {FIELD_ZOOM:g} for the fields")
            return
        view = view_to_bbox(vsd)
        box = pad_box(view)
        if box_km2(box) > FIELD_MAX_KM2:
            _fields_off(f"zoom {z:.1f} · view is {box_km2(box):,.0f} km² (> {FIELD_MAX_KM2:g}); zoom in for the fields")
            return
        if not force and HOLD["ft"] is not None and HOLD["box"] is not None and contains(HOLD["box"], view) and HOLD["ft"]["year"] == HOLD["year"]:
            return
        t0 = time.time()
        _say(f"year {HOLD['year']} · folding {box_km2(box):,.0f} km² of fields…")
        loop = asyncio.get_running_loop()
        yr = HOLD["year"]
        ft = await loop.run_in_executor(_pool, lambda: field_table(yr, *box))
        HOLD["ft"], HOLD["box"], HOLD["hit"] = ft, ft["box"], None
        HOLD["rings"] = await loop.run_in_executor(_pool, lambda: _rings_for(ft))
        _paint()
        HOLD["last_status"] = _status_for(ft, f"\noutlines {HOLD['rings']['n']:,} rings from {HOLD['rings']['tiles']} tiles · {time.time() - t0:.1f} s")
        _say(HOLD["last_status"])

    async def refresh(vs):
        """Settle-debounced, coalescing fold."""
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
                await _serve(vs)
                vs = HOLD["pending"]
                if vs is None:
                    return
                HOLD["pending"] = None
        except Exception as exc:
            _say(f"failed: {type(exc).__name__}: {exc}")
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
            HOLD["task"] = _spawn(_serve(vs, force=force) if force else refresh(vs))

    def _on_camera(change):
        vs = change["new"]
        if not vs:
            return
        HOLD["vs"] = vs
        if HOLD["busy"]:
            HOLD["pending"] = vs
            return
        HOLD["task"] = _spawn(refresh(vs))

    if HOLD.get("h_cam") is not None:
        try:
            deck.unobserve(HOLD["h_cam"], names="view")
        except ValueError:
            pass
    deck.observe(_on_camera, names="view")
    HOLD["h_cam"] = _on_camera

    def _chip(code):
        return (f"<span style='display:inline-block;width:10px;height:10px;border-radius:2px;"
                f"background:{CLASSES.get(int(code), ('', '#888'))[1]};margin-right:.35rem;vertical-align:-1px'></span>")

    def _analyze_html(ft):
        """Per crop in view: fields, acres, agreement p50, share below 0.5, what
        AlphaEarth usually suggests, and what last year's CDL usually said."""
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
        if not HOLD["sel"]:
            return ""
        maj, alt, agree, kept, scored = ft["maj"], ft["alt"], ft["agree"], ft["kept"], ft["scored"]
        parts = []
        for code in sorted(HOLD["sel"]):
            m = kept & ((alt == code) & scored & (agree < 0.5) if HOLD["paint"] == "suggests" else (maj == code))
            if not m.any():
                continue
            a = agree[m & scored]
            s = f"<b>{cname(code)}</b>: {int(m.sum()):,} fields"
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
        s = f"<b>{cname(maj[fid])}</b>{where}: {ac:,.1f} ac, CDL purity {purity[fid]:.2f}"
        if ft["prev_year"] is not None:
            s += f"; {ft['prev_year']}: {cname(prev[fid]) if prev[fid] > 0 else 'non-crop'}"
        if ft["scored"][fid]:
            s += f"; agreement <b>{agree[fid]:.2f}</b> (cos own {own[fid]:.3f}, best other {oth[fid]:.3f})"
            if agree[fid] < 0.5 and alt[fid] >= 0:
                s += f"; AlphaEarth suggests it could be <i>{cname(alt[fid])}</i> (relative to this view)"
        else:
            s += "; unscored (its crop has too few fields in view for a prototype)"
        return s

    def _on_pick(change):
        ft = HOLD["ft"]
        try:
            p = json.loads(change["new"] or "{}")
        except Exception:
            return
        if ft is None or p.get("lon") is None:
            return
        lon, lat = float(p["lon"]), float(p["lat"])
        jx = int(math.floor((lon + 180.0) / 8.98311982e-05)) - ft["fx0"]
        jy = int(math.floor((83.748345 - lat) / 8.98311982e-05)) - ft["fy0"]
        lab = ft["lab"]
        fid = int(lab[jy, jx]) if 0 <= jx < lab.shape[1] and 0 <= jy < lab.shape[0] else 0
        try:
            if fid == 0:
                HOLD["hit"] = None
                hud.widget.panel = _selection_panel(ft) or f"<span style='opacity:.7'>no field at {lat:.4f}, {lon:.4f}</span>"
            else:
                HOLD["hit"] = fid if HOLD["hit"] != fid else None
                hud.widget.panel = _field_story(ft, fid, lon, lat)
        except Exception as e:
            hud.widget.panel = f"<span style='opacity:.7'>click: {e}</span>"
        _paint()

    if HOLD.get("h_pick") is not None:
        try:
            deck.unobserve(HOLD["h_pick"], names="pick")
        except ValueError:
            pass
    deck.observe(_on_pick, names="pick")
    HOLD["h_pick"] = _on_pick

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
            hit = await asyncio.get_running_loop().run_in_executor(None, _photon_first, q, vs)
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
            zoom = 13.0
        zoom = max(3.5, min(14.5, zoom))
        deck.send({"kind": "fly", "lon": lon, "lat": lat, "zoom": zoom, "duration": 2000})
        _say(f"→ {name} · zoom {zoom:.1f}")

    def _on_ctl(change):
        try:
            c = json.loads(change["new"] or "{}")
        except Exception:
            return
        act = c.get("act", "set")
        paint_was, year_was = HOLD["paint"], HOLD["year"]
        HOLD["paint"] = c.get("paint", None)
        sel_was = set(HOLD["sel"])
        HOLD["sel"] = {int(x) for x in c.get("sel", [])}
        HOLD["inv"] = bool(c.get("inv", False))
        HOLD["outlines"] = bool(c.get("outlines", True))
        raster_was = (HOLD["raster"], HOLD["crops"], HOLD["clip"])
        HOLD["raster"] = bool(c.get("raster", True))
        HOLD["crops"] = bool(c.get("crops", False))
        HOLD["clip"] = bool(c.get("clip", False))
        try:
            HOLD["year"] = int(c.get("year", HOLD["year"]))
        except (TypeError, ValueError):
            pass
        if act == "clear":
            hud.widget.panel = ""
            return
        if act == "labels":
            HOLD["labels"] = bool(c.get("labels", True))
            _cfg(labels=HOLD["labels"])
            return
        if act == "analyze":
            ft = HOLD["ft"]
            hud.widget.panel = _analyze_html(ft) if ft is not None else f"<span style='opacity:.7'>no fields in view (zoom in past {FIELD_ZOOM:g} with a field paint on)</span>"
            return
        if act == "search":
            q = str(c.get("q") or "").strip()
            if q:
                _say(f"searching: {q}")
                HOLD["stask"] = _spawn(_search(q))
            return
        if HOLD["year"] != year_was:
            # both layers change: the raster refetches, the fields refold
            _raster_changed()
            HOLD["ft"], HOLD["box"], HOLD["hit"] = None, None, None
            _kick(force=True)
            return
        # the raster refetches when its switches change; the legend's selection
        # isolates classes on the raster too
        if (HOLD["raster"], HOLD["crops"], HOLD["clip"]) != raster_was or HOLD["sel"] != sel_was:
            _raster_changed()
        if HOLD["paint"] != paint_was:
            if HOLD["paint"] is None:
                _fields_off(f"CDL raster only · pick a field paint")
                return
            if HOLD["ft"] is None:
                _kick(force=True)
                return
        ft = HOLD["ft"]
        if ft is not None:
            hud.widget.panel = _selection_panel(ft)
            _paint()
            _say(HOLD.get("last_status", ""))

    if HOLD.get("h_ctl") is not None:
        try:
            hud.widget.unobserve(HOLD["h_ctl"], names="ctl")
        except ValueError:
            pass
    hud.widget.observe(_on_ctl, names="ctl")
    HOLD["h_ctl"] = _on_ctl

    # the opening fold (or a repaint if a field table already exists after a re-run)
    if HOLD["ft"] is None:
        HOLD["task"] = _spawn(_serve(HOLD["vs"] if HOLD["vs"] is not None else dict(HOME)))
    else:
        _paint()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Under the map

    The table below is the CURRENT view's fields, per crop (press the button
    after the map settles).
    """)
    return


@app.cell
def _(mo):
    tables_btn = mo.ui.run_button(label="table for the current view")
    tables_btn
    return (tables_btn,)


@app.cell
def _(HOLD, cname, mo, np, tables_btn):
    mo.stop(not tables_btn.value or HOLD["ft"] is None, mo.md("*no fields folded yet*"))
    _ft = HOLD["ft"]
    _maj, _alt, _agree, _kept, _scored, _sizes, _prev = _ft["maj"], _ft["alt"], _ft["agree"], _ft["kept"], _ft["scored"], _ft["sizes"], _ft["prev"]
    _rows = []
    for _code, _cnt in zip(*np.unique(_maj[_kept], return_counts=True)):
        _m = _kept & (_maj == _code)
        _a = _agree[_m & _scored]
        _dis = _m & _scored & (_agree < 0.5)
        _u, _c = np.unique(_alt[_dis], return_counts=True) if _dis.any() else (np.array([]), np.array([]))
        _pu, _pc = np.unique(_prev[_m], return_counts=True)
        _rows.append({
            "crop": cname(_code), "fields": int(_cnt), "acres": round(float(_sizes[_m].sum() * _ft["pxa"]), 1),
            "agree_p50": round(float(np.median(_a)), 3) if len(_a) else None,
            "pct_below_half": round(float((_a < 0.5).mean() * 100), 1) if len(_a) else None,
            "aef_usually_suggests": cname(_u[_c.argmax()]) if len(_u) else "",
            "last_year_usually": (cname(_pu[_pc.argmax()]) if _pu[_pc.argmax()] > 0 else "(non-crop)") if _ft["prev_year"] else "",
        })
    _rows.sort(key=lambda r: -r["fields"])
    per_crop = mo.ui.table(_rows, selection=None)
    per_crop
    return (per_crop,)


if __name__ == "__main__":
    app.run()
