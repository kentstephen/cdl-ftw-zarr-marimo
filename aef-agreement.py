# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xarray",
#     "zarr>=3",
#     "icechunk",
#     "obstore",
#     "pyarrow>=25.0.0",
#     "numpy",
#     "scipy",
#     "anywidget>=0.9",
#     "lonboard>=0.16.0,<0.17",
#     "arro3-core",
#     "pillow==12.3.0",
#     "morecantile==7.0.3",
#     "ipywidgets==8.1.8",
#     "traitlets==5.15.1",
# ]
# ///
"""The agreement map: does each field's AlphaEarth look-alikes carry its CDL label?

The click-a-field notebook (aef-similarity.py) grown to every field at once
(branch aef-similarity, 2026-08-24, Stephen: "ok lets build the agreement
map"). Per batch of the view:

  1. The FIELDS in view are the connected components of FTW P(field) >= 0.5
     at 10 m (scipy.ndimage.label on one windowed read of the root group).
  2. Each field gets its mean AlphaEarth vector (the year's embeddings,
     sampled at 20 m stride over the field, unit-normalised) and its CDL
     majority crop (the year's CDL, 10 m group for 2024-2025 else 30 m).
  3. For each crop field: of its K=10 nearest embedding look-alikes AMONG
     THE FIELDS IN VIEW (cosine, a plain V @ V.T), how many carry its CDL
     label? That fraction is the field's agreement.
  4. Paint: DARK = agrees (AlphaEarth's look-alikes and CDL tell the same
     story; boring, correct). BRIGHT YELLOW = disagrees (a young orchard,
     a CDL mislabel, a double-crop, something odd: every bright field is a
     question). Non-crop-majority and tiny fields sit out (faint grey).

The panel lists the most surprising fields in view (what CDL calls them vs
what their look-alikes are); CLICK any field for its full story (CDL class
and purity, acres, its look-alikes' labels). The neighbor pool is the view,
so the score is "agreement among the fields on screen": local, honest, and
recomputed per batch (a pan can nudge scores; tiles from different batches
may differ slightly at seams).

Everything else is aef-similarity.py's, by copy: the AEF chunk cache, the
CDL anchor machinery, the PMTiles outlines, the tile-batch serve, the HUD
with the canvas click (NOT lonboard's on_click), the lonboard JS patch in
the first cell, the z13 similarity floor (no pyramid in the mosaic).

Run from THIS repo's venv:
  uv sync && uv run marimo edit aef-agreement.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    # ---- the lonboard JS patch, applied in the RUNNING environment ----------
    import importlib.util as _ilu
    import os as _os

    _here = _os.path.dirname(_os.path.abspath(__file__)) if "__file__" in globals() else _os.getcwd()
    _tool = _os.path.join(_here, "tools", "patch_lonboard_raster_unlit.py")
    LONBOARD_PATCHED = False
    if _os.path.exists(_tool):
        _spec = _ilu.spec_from_file_location("patch_lonboard_raster_unlit", _tool)
        _mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        LONBOARD_PATCHED = _mod.main() == 0
    else:
        print(f"patch tool not found at {_tool}; lonboard runs unpatched")
    return (LONBOARD_PATCHED,)


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
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from PIL import Image, ImageDraw
    from scipy import ndimage

    import anywidget
    import obstore
    import icechunk
    import xarray as xr
    import zarr
    import traitlets
    import urllib.parse
    import urllib.request

    import morecantile
    import pyarrow as pa
    from arro3.core import Table as A3Table
    from lonboard import Map, SolidPolygonLayer, RasterLayer
    from lonboard.raster import EncodedImage
    from lonboard.basemap import CartoStyle, MaplibreBasemap
    from obstore.store import S3Store

    import marimo as mo

    return (
        A3Table,
        CartoStyle,
        EncodedImage,
        Image,
        ImageDraw,
        Map,
        MaplibreBasemap,
        RasterLayer,
        S3Store,
        SolidPolygonLayer,
        ThreadPoolExecutor,
        anywidget,
        asyncio,
        gzip,
        icechunk,
        io,
        json,
        math,
        mo,
        morecantile,
        ndimage,
        np,
        obstore,
        os,
        pa,
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
    # The agreement map

    **What you are looking at.** Every field on screen gets one score, from
    one question. Take the field's AlphaEarth fingerprint (64 numbers
    summarizing its whole year as the satellites saw it), find the **10
    fields in view whose year looked most like it**, and check their crop
    labels in CDL: **how many of the 10 grow what CDL says THIS field
    grows?** That fraction is the color:

    - **Bright yellow = 10/10.** Everything checks out: the field looks
      like its label. Three datasets that have never met (AlphaEarth = raw
      satellite story, CDL = USDA's labels, FTW = where the fields are)
      tell one story. Most of the map should be bright; that is the two
      datasets corroborating each other.
    - **Dark purple = 0/10.** This field's look-alikes all grow something
      ELSE. Either CDL is wrong here (a mislabel, a just-planted orchard
      still filed under its old crop), or the field genuinely had a
      different year than its label implies (failed, fallow, double-crop).
      **The dark fields are the leads**: each one is a specific place where
      somebody's data is wrong in an interesting way.
    - **Faint grey** sits out: too small, or not cropland per CDL.

    The panel under the map always shows the count, the median score, and
    the worst offenders in view (what CDL calls them vs what their
    look-alikes grow). **Click any field** to get its story. Paint from
    camera ~z12; the look-alike pool is the fields on screen, so scores are
    local and recompute as you move.
    """)
    return


@app.cell
def _():
    # ---- constants (aef-similarity.py's, plus the agreement knobs) ----------
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

    AEF_RES = 8.983111749910169e-05
    AEF_Y0 = 83.68570533713473
    AEF_X0 = -180.0
    AEF_SHAPE = (1859584, 4009984)
    AEF_YEARS = list(range(2017, 2026))
    YEAR0 = 2024

    FTW_RES = 8.98311982e-05
    FTW_Y0 = 83.748345
    FTW_YEARS = (2024, 2025)

    ACH = 256                 # AEF inner chunk (px)
    AEF_MEM_CHUNKS = 192
    K_NBR = 10                # look-alikes per field
    MIN_FIELD_PX = 12         # ~0.3 ac at 10 m: smaller components sit out
    MIN_CROP_FRAC = 0.3       # a field is a CROP field if >= this much of it
    #                           carries a CDL crop class
    PANEL_MIN_AC = 10.0       # the "most surprising" list ignores scraps
    ACRES_PER_KM2 = 247.10538

    TILE_PX = 256
    BATCH_S = 0.05
    TILE_CACHE = 3000
    AEF_ZMIN, TILE_ZMAX = 13, 15
    VIEW_ZMIN = 3.0
    EXTENT = [-125.0, 24.0, -66.5, 49.8]
    FTW_TILE_ZMAX = 13
    OUTLINE_ZMIN = 12
    MARGIN = 0.35
    VIEW_W, VIEW_H = 1400, 700
    HOME = {"longitude": -121.45, "latitude": 37.95, "zoom": 12.5}

    HOLD: dict = {}
    return (
        ACH,
        ACRES_PER_KM2,
        AEF_MEM_CHUNKS,
        AEF_RES,
        AEF_SHAPE,
        AEF_X0,
        AEF_Y0,
        AEF_YEARS,
        AEF_ZARR,
        AEF_ZMIN,
        BATCH_S,
        CDL_BUCKET,
        CDL_ENDPOINT,
        CDL_PREFIX,
        EXTENT,
        FTW_RES,
        FTW_TILE_ZMAX,
        FTW_VEC,
        FTW_Y0,
        FTW_YEARS,
        FTW_ZARR,
        HOLD,
        HOME,
        K_NBR,
        MARGIN,
        MIN_CROP_FRAC,
        MIN_FIELD_PX,
        OUTLINE_ZMIN,
        PANEL_MIN_AC,
        SC_BUCKET,
        TILE_CACHE,
        TILE_PX,
        TILE_ZMAX,
        VIEW_H,
        VIEW_W,
        VIEW_ZMIN,
        YEAR0,
    )


@app.cell
def _(
    AEF_ZARR,
    CDL_BUCKET,
    CDL_ENDPOINT,
    CDL_PREFIX,
    FTW_ZARR,
    S3Store,
    SC_BUCKET,
    icechunk,
    xr,
    zarr,
):
    # ---- open the stores (aef-similarity.py's cell, FTW root only) ----------
    _aef_store = zarr.storage.ObjectStore(
        S3Store(bucket=SC_BUCKET, region="us-west-2", skip_signature=True,
                prefix=AEF_ZARR),
        read_only=True,
    )
    import warnings as _warnings
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        AEF_DS = xr.open_zarr(_aef_store, chunks=None, consolidated=False)

    _ftw_store = zarr.storage.ObjectStore(
        S3Store(bucket=SC_BUCKET, region="us-west-2", skip_signature=True,
                prefix=FTW_ZARR),
        read_only=True,
    )
    FTW_ROOT = xr.open_zarr(_ftw_store, chunks=None, consolidated=False)

    _storage = icechunk.s3_storage(
        bucket=CDL_BUCKET,
        prefix=CDL_PREFIX,
        endpoint_url=CDL_ENDPOINT,
        region="us-east-1",
        anonymous=True,
        force_path_style=True,
    )
    _repo = icechunk.Repository.open(_storage)
    _session = _repo.readonly_session("main")
    CDL30 = xr.open_zarr(_session.store, group="30m", chunks=None)
    CDL10 = xr.open_zarr(_session.store, group="10m", chunks=None)

    _at = CDL30["crop_type"].attrs
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
    CLASSES = {}
    for _code in sorted(_names, key=int):
        _nm, _hx = _names[_code], _colors[_code]
        _r, _g, _b = _rgb(_hx)
        _safe = _hx
        if _r >= 170 and _g <= 100 and _b <= 110:
            _safe = _SAFE_CYCLE[_i % len(_SAFE_CYCLE)]
            _i += 1
        CLASSES[int(_code)] = (_nm, _safe, _noncrop(_nm))
    NONCROP_CODES = sorted(c for c, v in CLASSES.items() if v[2])
    return AEF_DS, CDL10, CDL30, CLASSES, FTW_ROOT, NONCROP_CODES


@app.cell
def _(MARGIN, VIEW_H, VIEW_W, math, np):
    # ---- pure helpers (aef-similarity.py's) ---------------------------------
    def tile_box(z, x, y):
        n = 2 ** z
        W = x / n * 360.0 - 180.0
        E = (x + 1) / n * 360.0 - 180.0
        N = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
        S = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
        return W, S, E, N

    def bbox4326(vs):
        span = 360.0 / (512 * 2 ** vs["zoom"])
        dlon = VIEW_W * span * (1 + MARGIN) / 2
        dlat = VIEW_H * span * math.cos(math.radians(vs["latitude"])) * (1 + MARGIN) / 2
        return (vs["longitude"] - dlon, vs["latitude"] - dlat,
                vs["longitude"] + dlon, vs["latitude"] + dlat)

    def unproject(vs, px, py, w, h):
        world = 512 * 2 ** vs["zoom"]
        lon = vs["longitude"] + (px - w / 2) * 360.0 / world
        lat0 = math.radians(vs["latitude"])
        uy = (1 - math.log(math.tan(lat0) + 1 / math.cos(lat0)) / math.pi) / 2
        uy = uy + (py - h / 2) / world
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * uy))))
        return lon, lat

    def albers_xy(lon, lat):
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
        lons = np.linspace(W, E, 9)
        lats = np.linspace(S, N, 9)
        bl = np.concatenate([lons, lons, np.full(9, W), np.full(9, E)])
        bt = np.concatenate([np.full(9, S), np.full(9, N), lats, lats])
        X, Y = albers_xy(bl, bt)
        _X0, _Y0, _X1, _Y1 = -2417835.0, 158265.0, 2387295.0, 3321225.0
        return (max(float(X.min()), _X0), max(float(Y.min()), _Y0),
                min(float(X.max()), _X1), min(float(Y.max()), _Y1))

    # the ramp (Stephen, 2026-08-24, after a PuOr diverging detour): viridis,
    # BRIGHT = HIGH agreement (the datasets corroborate each other), DARK =
    # low (someone is wrong here; the dark fields are the leads). Index 0 =
    # 0% agreement, 255 = 100%.
    _VIR = np.array([
        (68, 1, 84), (72, 26, 108), (71, 47, 125), (65, 68, 135),
        (57, 86, 140), (49, 104, 142), (42, 120, 142), (35, 136, 142),
        (31, 152, 139), (34, 168, 132), (53, 183, 121), (84, 197, 104),
        (122, 209, 81), (165, 219, 54), (210, 226, 27), (253, 231, 37),
    ], dtype=np.float64)
    _t = np.linspace(0, 1, 256)
    _a = np.linspace(0, 1, len(_VIR))
    AGREE_LUT = np.stack([np.interp(_t, _a, _VIR[:, i]) for i in range(3)],
                         axis=1).astype(np.uint8)
    return AGREE_LUT, albers_box, albers_xy, bbox4326, tile_box, unproject


@app.cell
def _(
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
    tempfile,
    threading,
):
    # ---- FTW field OUTLINES from the per-state PMTiles (by copy) ------------
    _pm = S3Store(bucket=SC_BUCKET, region="us-west-2", skip_signature=True)
    _TILE_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "ftw-tiles")
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

    # ---- the same tiles as CLOSED POLYGON RINGS (the vector fill, Stephen
    # 2026-08-24 night: "lets just use the pmtiles"): no clip-segment
    # dropping here (a fill needs the ring closed, tile edge included; the
    # seam problem is the STROKE's, and strokes stay with the raster
    # outlines). Same blobs, own decode + memory cache.
    _memp, _memp_lock = {}, threading.Lock()

    def _decode_poly(blob, year, z, x, y):
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
                    if len(ring) < 4:
                        continue
                    a = np.asarray(ring, dtype=np.float64)
                    if not np.array_equal(a[0], a[-1]):
                        a = np.vstack([a, a[:1]])
                    lon = (x + a[:, 0] / extent) / n * 360.0 - 180.0
                    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + a[:, 1] / extent) / n))))
                    out.append(np.column_stack([lon, lat]))
        return out

    def _tile_poly(st, z, x, y, year):
        key = (st, z, x, y, year)
        with _memp_lock:
            v = _memp.get(key)
        if v is not None:
            return v
        b = _blob(st, z, x, y)
        v = _decode_poly(b, year, z, x, y) if b else []
        with _memp_lock:
            _memp[key] = v
            if len(_memp) > 2000:
                for _k in list(_memp)[:400]:
                    _memp.pop(_k, None)
        return v

    def ftw_tile_polys(states, year, W, S, E, N, z):
        """Closed rings (lon/lat, first point repeated) over the box."""
        n = 1 << z

        def tx(lon):
            return min(n - 1, max(0, int((lon + 180) / 360 * n)))

        def ty(lat):
            lat = max(-85.05, min(85.05, lat))
            return min(n - 1, max(0, int((1 - math.log(math.tan(math.radians(lat))
                                                     + 1 / math.cos(math.radians(lat))) / math.pi) / 2 * n)))

        jobs = [(st, z, x, y, year) for st in states
                for x in range(tx(W), tx(E) + 1) for y in range(ty(N), ty(S) + 1)]
        res = list(_tpool.map(lambda j: _tile_poly(*j), jobs))
        return [r for v in res for r in v]

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
    _ = FTW_TILE_ZMAX
    return STATES, ftw_tile_polys, ftw_tile_rings


@app.cell
def _(anywidget, traitlets):
    class HudControls(anywidget.AnyWidget):
        """aef-similarity.py's strip for the agreement map: year slider,
        refresh, search, legend (the ramp), panel (the surprising fields /
        a clicked field's story), status; the canvas CLICK through ctl; the
        fullscreen dock."""

        ctl = traitlets.Unicode("").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        poke = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("").tag(sync=True)
        panel = traitlets.Unicode("").tag(sync=True)

        _esm = r"""
        function render({ model, el }) {
          const box = document.createElement("div");
          box.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;gap:.9rem;" +
            "font:12px ui-sans-serif,system-ui,sans-serif;padding:.2rem 0 0;" +
            "user-select:none";
          const btnCss =
            "font:12px ui-sans-serif,system-ui,sans-serif;cursor:pointer;" +
            "padding:.1rem .45rem;border-radius:4px;border:1px solid " +
            "rgba(127,127,127,.45);background:transparent;color:inherit";
          const yl = document.createElement("span");
          yl.textContent = "year";
          const range = document.createElement("input");
          range.type = "range";
          range.min = "2017"; range.max = "2025"; range.step = "1";
          range.value = "2024";
          range.style.cssText = "width:9rem";
          const yv = document.createElement("span");
          yv.style.cssText = "font-weight:600;font-variant-numeric:tabular-nums";
          yv.textContent = range.value;
          const labC = document.createElement("label");
          labC.style.cssText =
            "display:inline-flex;align-items:center;gap:.35rem;cursor:pointer";
          const cdl = document.createElement("input");
          cdl.type = "checkbox"; cdl.checked = false;
          labC.appendChild(cdl);
          labC.appendChild(document.createTextNode("show CDL"));
          labC.title = "color the fields by their CDL crop instead of the score";
          const rfr = document.createElement("button");
          rfr.textContent = "refresh";
          rfr.title = "rebuild the tile layer";
          rfr.style.cssText = btnCss;
          const search = document.createElement("input");
          search.type = "search";
          search.placeholder = "find a place…";
          search.style.cssText =
            "width:11rem;font:12px ui-sans-serif,system-ui,sans-serif;" +
            "padding:.15rem .45rem;border:1px solid rgba(127,127,127,.45);" +
            "border-radius:4px;background:transparent;color:inherit";
          const legendBox = document.createElement("div");
          legendBox.style.cssText =
            "display:flex;flex-wrap:wrap;align-items:center;gap:.6rem;" +
            "flex:1;min-width:14rem";
          const renderLegend = () => {
            legendBox.innerHTML = model.get("legend") || "";
          };
          model.on("change:legend", renderLegend);
          renderLegend();
          box.append(yl, range, yv, labC, rfr, search, legendBox);
          const status = document.createElement("div");
          status.style.cssText =
            "font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.15rem 0;min-height:1.2em";
          const panel = document.createElement("div");
          panel.style.cssText =
            "font:13px ui-sans-serif,system-ui,sans-serif;padding:.15rem 0";
          const wrap = document.createElement("div");
          wrap.dataset.aefStrip = "1";
          wrap.append(box, panel, status);
          const killOld = (root) => {
            if (!root || !root.querySelectorAll) return;
            root.querySelectorAll("[data-aef-strip]").forEach((w) => {
              if (w !== wrap) { w.dataset.dead = "1"; w.remove(); }
            });
            root.querySelectorAll("*").forEach((n) => {
              if (n.shadowRoot) killOld(n.shadowRoot);
            });
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
              if (getComputedStyle(fe).position === "static")
                fe.style.position = "relative";
              wrap.style.cssText =
                "position:absolute;left:0;right:0;bottom:0;z-index:30;" +
                "background:rgba(255,255,255,.94);color:#111;" +
                "padding:.5rem 1.2rem;box-shadow:0 -1px 4px rgba(0,0,0,.18)";
              fe.appendChild(wrap);
            } else {
              wrap.style.cssText = "";
              el.appendChild(wrap);
            }
          };
          document.addEventListener("fullscreenchange", onFs);
          let seq = 0, deb = null;
          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({
              act: act, year: +range.value, cdl: cdl.checked,
              n: ++seq }, extra || {})));
            model.save_changes();
          };
          const commit = () => {
            clearTimeout(deb);
            deb = setTimeout(() => send("set"), 250);
          };
          range.addEventListener("input", () => { yv.textContent = range.value; });
          range.addEventListener("change", commit);
          cdl.addEventListener("change", commit);
          rfr.addEventListener("click", () => send("refresh"));
          search.addEventListener("keydown", (e) => {
            const q = search.value.trim();
            if (e.key === "Enter" && q) send("search", { q: q });
          });
          let downAt = null;
          const onDown = (e) => { downAt = [e.clientX, e.clientY]; };
          const onClick = (e) => {
            if (wrap.dataset.dead || !el.isConnected) return;
            const path = e.composedPath ? e.composedPath() : [e.target];
            const cv = path.find((n) => n && n.tagName === "CANVAS");
            if (!cv) return;
            if (downAt && Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 5) return;
            const r = cv.getBoundingClientRect();
            if (!r.width || !r.height) return;
            send("click", {
              px: e.clientX - r.left, py: e.clientY - r.top,
              w: r.width, h: r.height });
          };
          document.addEventListener("pointerdown", onDown, true);
          document.addEventListener("click", onClick, true);
          const paintS = () => { status.textContent = model.get("status") || ""; };
          model.on("change:status", paintS);
          paintS();
          const paintP = () => { panel.innerHTML = model.get("panel") || ""; };
          model.on("change:panel", paintP);
          paintP();
          // the polygon roundtrip: the serve thread cannot hand a new widget
          // (or a table update) to the frontend under marimo; it stores the
          // built polygons, sets `poke`, and THIS reply turns the update
          // into a cell run, where creating the layer works (2026-08-24
          // night: the in-place table push drew once at startup and never
          // again; every pan lost the fills)
          model.on("change:poke", () => {
            if (model.get("poke")) send("polys");
          });
          const hideBbox = (root) => {
            if (!root || !root.querySelectorAll) return;
            root.querySelectorAll("button[aria-label]").forEach((b) => {
              const a = b.getAttribute("aria-label");
              if (a === "Select BBox" || a === "Cancel drawing" ||
                  a === "Clear bounding box") {
                const holder = b.closest("div[style*='absolute']") || b;
                holder.style.display = "none";
              }
            });
            root.querySelectorAll("*").forEach((n) => {
              if (n.shadowRoot) hideBbox(n.shadowRoot);
            });
          };
          const bboxTimer = setInterval(() => hideBbox(document), 1000);
          return () => {
            document.removeEventListener("fullscreenchange", onFs);
            document.removeEventListener("pointerdown", onDown, true);
            document.removeEventListener("click", onClick, true);
            clearInterval(bboxTimer);
            wrap.remove();
          };
        }
        export default { render };
        """

    return (HudControls,)


@app.cell
def _(
    CartoStyle,
    HOLD: dict,
    HOME,
    LONBOARD_PATCHED,
    Map,
    MaplibreBasemap,
    SolidPolygonLayer,
    np,
    pa,
):
    # ---- map cell: builds the Map, must never re-run ------------------------
    # The POLYGON layer is created HERE, once, and NEVER removed from
    # deck.layers (under marimo a removed layer is closed; a widget created
    # outside a cell run never reaches the frontend). The serve only assigns
    # its table + fill colours in place; the raster layer above it keeps the
    # seam-free outlines and the z12 view.
    _ = LONBOARD_PATCHED
    HOLD["layer_state"] = None

    def _dummy_table():
        flat = np.array([[-140.0, 20.0], [-139.999, 20.0],
                         [-139.999, 20.001], [-140.0, 20.0]])
        coords = pa.FixedSizeListArray.from_arrays(
            pa.array(flat.ravel(), type=pa.float64()), 2)
        rings = pa.ListArray.from_arrays(pa.array([0, 4], type=pa.int32()), coords)
        polys = pa.ListArray.from_arrays(pa.array([0, 1], type=pa.int32()), rings)
        schema = pa.schema([pa.field("geometry", polys.type, metadata={
            b"ARROW:extension:name": b"geoarrow.polygon"})])
        return pa.Table.from_arrays([polys], schema=schema)

    # ONE polygon layer, created here and never replaced. Creating any
    # widget mid-session leaves the frontend with "Model not found for key"
    # and the deck goes blank; a SECOND vector layer collides on deck id
    # "undefined-0" (model_id is undefined under marimo and the JS patch
    # only fixed raster ids) and deck asserts. Both found with playwright,
    # 2026-08-24 night. Only its table / colour traits are ever assigned,
    # and only IN a cell run; the click highlight is a colour, not a layer.
    HOLD["polys"] = SolidPolygonLayer(
        table=_dummy_table(),
        filled=True,
        get_fill_color=np.zeros((1, 4), dtype=np.uint8),
        pickable=False,
    )
    deck = Map(
        layers=[],
        basemap=MaplibreBasemap(style=CartoStyle.Positron),
        view_state=HOME,
        height=700,
        show_side_panel=False,
    )
    deck
    return (deck,)


@app.cell
def _(HudControls, mo):
    hud = mo.ui.anywidget(HudControls())
    hud
    return (hud,)


@app.cell
def _(
    A3Table,
    ACH,
    ACRES_PER_KM2,
    AEF_DS,
    AEF_MEM_CHUNKS,
    AEF_RES,
    AEF_SHAPE,
    AEF_X0,
    AEF_Y0,
    AEF_YEARS,
    AEF_ZMIN,
    AGREE_LUT,
    BATCH_S,
    CDL10,
    CDL30,
    CLASSES,
    EXTENT,
    EncodedImage,
    FTW_RES,
    FTW_ROOT,
    FTW_TILE_ZMAX,
    FTW_Y0,
    FTW_YEARS,
    HOLD: dict,
    HOME,
    Image,
    ImageDraw,
    K_NBR,
    MIN_CROP_FRAC,
    MIN_FIELD_PX,
    NONCROP_CODES,
    OUTLINE_ZMIN,
    PANEL_MIN_AC,
    RasterLayer,
    STATES,
    SolidPolygonLayer,
    TILE_CACHE,
    TILE_PX,
    TILE_ZMAX,
    VIEW_W,
    VIEW_ZMIN,
    YEAR0,
    albers_box,
    albers_xy,
    asyncio,
    bbox4326,
    deck,
    ftw_tile_polys,
    ftw_tile_rings,
    hud,
    io,
    json,
    math,
    morecantile,
    ndimage,
    np,
    os,
    pa,
    tempfile,
    threading,
    tile_box,
    time,
    unproject,
    urllib,
):
    # ---- wiring cell: re-runs on every HUD commit ---------------------------
    try:
        _c = json.loads(hud.widget.ctl or "{}")
    except Exception:
        _c = {}
    _year = int(_c.get("year", YEAR0))
    if _year not in AEF_YEARS:
        _year = YEAR0
    _cdl_mode = bool(_c.get("cdl", False))
    _act = _c.get("act", "set")
    _q = str(_c.get("q", "")).strip()
    _NONCROP = np.zeros(256, dtype=bool)
    _NONCROP[[0, 81, *NONCROP_CODES]] = True
    _CLASS_RGB = np.full((256, 3), 136, dtype=np.uint8)
    for _cc, (_nm2, _hx2, _nc2) in CLASSES.items():
        _CLASS_RGB[_cc] = (int(_hx2[1:3], 16), int(_hx2[3:5], 16),
                           int(_hx2[5:7], 16))

    try:
        HOLD["loop"] = asyncio.get_running_loop()
    except RuntimeError:
        pass

    def _say(msg):
        try:
            hud.widget.status = msg
        except Exception:
            pass

    def _vsd(vs):
        if vs is None:
            return None
        if isinstance(vs, dict):
            d = {k: vs.get(k) for k in ("longitude", "latitude", "zoom")}
        else:
            d = {k: getattr(vs, k, None) for k in ("longitude", "latitude", "zoom")}
        return d if None not in d.values() else None

    # ---- AEF chunk cache + dequant (aef-similarity.py's) --------------------
    _AEF_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "aef-emb")
    _DQ = np.zeros(256, dtype=np.float32)
    _qv = np.arange(-128, 128, dtype=np.float32)
    _DQ[(np.arange(-128, 128) & 0xFF)] = (np.abs(_qv) / 127.5) ** 2 * np.sign(_qv)
    _DQ[128] = 0.0   # nodata -128 -> 0 weight here (valid tracked separately)

    def _aef_fetch(year, missing, needed):
        mem = HOLD.setdefault("aef_chunks", {})
        mx0, mx1 = min(c[0] for c in missing), max(c[0] for c in missing)
        my0, my1 = min(c[1] for c in missing), max(c[1] for c in missing)
        ya, yb = my0 * ACH, min((my1 + 1) * ACH, AEF_SHAPE[0])
        xa, xb = mx0 * ACH, min((mx1 + 1) * ACH, AEF_SHAPE[1])
        big = AEF_DS.embeddings.sel(time=year).isel(
            y=slice(ya, yb), x=slice(xa, xb)).values
        for cx in range(mx0, mx1 + 1):
            for cy in range(my0, my1 + 1):
                piece = np.full((64, ACH, ACH), -128, dtype=np.int8)
                sy = (cy - my0) * ACH
                sx = (cx - mx0) * ACH
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
            # NEVER evict what the current request needs: an eviction here
            # once dropped a chunk the caller had just checked off as
            # present, and every batch in the area died on the KeyError
            # until a restart ("nothing happens when I move around",
            # 2026-08-24 night; the cap only trips after real panning,
            # which is why headless never saw it)
            for _k in [k for k in list(mem) if k not in needed][:AEF_MEM_CHUNKS // 4]:
                mem.pop(_k, None)

    def _aef_chunks(year, cx0, cx1, cy0, cy1):
        """The range's chunks as a SNAPSHOT dict {key: int8 array}: safe
        against any later eviction. Hits are re-inserted (LRU-ish)."""
        mem = HOLD.setdefault("aef_chunks", {})
        needed = {(year, cx, cy)
                  for cx in range(cx0, cx1 + 1) for cy in range(cy0, cy1 + 1)}
        missing = []
        for key in needed:
            if key in mem:
                mem[key] = mem.pop(key)   # refresh recency
                continue
            fp = os.path.join(_AEF_DIR, str(year), f"{key[1]}_{key[2]}.npy")
            if os.path.exists(fp):
                mem[key] = np.load(fp)
            else:
                missing.append((key[1], key[2]))
        if missing:
            _aef_fetch(year, missing, needed)
        return {k: mem[k] for k in needed}

    def _aef_mosaic(year, W, S, E, N):
        """int8 (64, H, W) chunk-aligned over the box + (ix0, iy0)."""
        ix0 = max(int((W - AEF_X0) / AEF_RES), 0)
        ix1 = min(int((E - AEF_X0) / AEF_RES), AEF_SHAPE[1] - 1)
        iy0 = max(int((AEF_Y0 - N) / AEF_RES), 0)
        iy1 = min(int((AEF_Y0 - S) / AEF_RES), AEF_SHAPE[0] - 1)
        cx0, cx1, cy0, cy1 = ix0 // ACH, ix1 // ACH, iy0 // ACH, iy1 // ACH
        mem = _aef_chunks(year, cx0, cx1, cy0, cy1)
        mos = np.full((64, (cy1 - cy0 + 1) * ACH, (cx1 - cx0 + 1) * ACH), -128,
                      dtype=np.int8)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                mos[:, (cy - cy0) * ACH:(cy - cy0 + 1) * ACH,
                    (cx - cx0) * ACH:(cx - cx0 + 1) * ACH] = mem[(year, cx, cy)]
        return mos, cx0 * ACH, cy0 * ACH

    # ---- the FTW 10 m mask, chunk-cached (the 4x machinery at the root) -----
    _CH = 512
    _MASK_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "ftw-mask")

    def _ftw10(fyear, W, S, E, N):
        """P(field) >= 0.5 at 10 m over the box, chunk-aligned: (mask,
        fx0, fy0). The unit of the whole notebook: its connected components
        are THE FIELDS."""
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
                    cache[(fyear, cx, cy)] = np.unpackbits(
                        np.load(fp)).reshape(_CH, _CH).astype(bool)
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
                    piece = big[(cy - my0) * _CH:(cy - my0 + 1) * _CH,
                                (cx - mx0) * _CH:(cx - mx0 + 1) * _CH].copy()
                    cache[(fyear, cx, cy)] = piece
                    fp = os.path.join(_MASK_DIR, "1x", str(fyear), f"{cx}_{cy}.npy")
                    try:
                        os.makedirs(os.path.dirname(fp), exist_ok=True)
                        tmp = f"{fp}.{threading.get_ident()}.tmp"
                        np.save(tmp, np.packbits(piece))
                        os.replace(tmp + ".npy" if not tmp.endswith(".npy") else tmp, fp)
                    except Exception:
                        pass
            if len(cache) > 400:
                for _k in list(cache)[:80]:
                    cache.pop(_k, None)
        mask = np.zeros(((cy1 - cy0 + 1) * _CH, (cx1 - cx0 + 1) * _CH), dtype=bool)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                mask[(cy - cy0) * _CH:(cy - cy0 + 1) * _CH,
                     (cx - cx0) * _CH:(cx - cx0 + 1) * _CH] = cache[(fyear, cx, cy)]
        return mask, cx0 * _CH, cy0 * _CH

    def _cdl_codes(year, LON, LAT):
        ds = CDL10 if year in (2024, 2025) else CDL30
        pix = 10.0 if year in (2024, 2025) else 30.0
        x0, y0, x1, y1 = albers_box(float(LON.min()), float(LAT.min()),
                                    float(LON.max()), float(LAT.max()))
        da = ds.crop_type.sel(year=year).sel(x=slice(x0, x1), y=slice(y1, y0))
        g = np.asarray(da.values)
        code = np.zeros(LON.shape, dtype=np.uint8)
        if not g.size:
            return code
        gx0 = float(da.x.values[0]) - pix / 2
        gy1 = float(da.y.values[0]) + pix / 2
        X, Y = albers_xy(LON, LAT)
        jx = ((X - gx0) / pix).astype(np.int64)
        jy = ((gy1 - Y) / pix).astype(np.int64)
        ok = (jx >= 0) & (jx < g.shape[1]) & (jy >= 0) & (jy < g.shape[0])
        code[ok] = g[jy[ok], jx[ok]]
        return code

    def _chip(hx):
        return ('<span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:2px;background:{hx};margin-right:4px;'
                'vertical-align:-1px"></span>')

    def _cname(code):
        return CLASSES.get(int(code), (f"code {code}", "#888", True))

    # ---- THE FIELD TABLE: one batch's fields, vectors, labels, agreement ----
    def _field_table(year, W, S, E, N):
        """Label the FTW fields over the box, give each its mean AEF vector
        and CDL majority, score each crop field by its K nearest look-alikes.
        Returns the dict the tiles, the panel and the click all read.
        CACHED by the chunk-aligned box: a toggle or a same-box zoom pays
        nothing (Stephen, 2026-08-24 night: toggling must not recompute)."""
        fyear = year if year in FTW_YEARS else FTW_YEARS[0]
        _ck = (year, int(math.floor((W + 180.0) / FTW_RES)) // _CH,
               int(math.floor((E + 180.0) / FTW_RES)) // _CH,
               int(math.floor((FTW_Y0 - N) / FTW_RES)) // _CH,
               int(math.floor((FTW_Y0 - S) / FTW_RES)) // _CH)
        _fc = HOLD.setdefault("ftab_cache", {})
        hitft = _fc.get(_ck)
        if hitft is not None:
            return hitft
        mask, fx0, fy0 = _ftw10(fyear, W, S, E, N)
        lab, nlab = ndimage.label(mask)
        lab = lab.astype(np.int32)
        sizes = np.bincount(lab.ravel(), minlength=nlab + 1)
        h, w = lab.shape
        lonv = (fx0 + np.arange(w) + 0.5) * FTW_RES - 180.0
        latv = FTW_Y0 - (fy0 + np.arange(h) + 0.5) * FTW_RES
        LON, LAT = np.meshgrid(lonv, latv)
        codes = _cdl_codes(year, LON, LAT)
        # CDL majority among CROP classes per field (a field is a crop field
        # if enough of it carries a crop class; the rest sit out)
        codes_crop = np.where(_NONCROP[codes], 0, codes)
        pair = lab.astype(np.int64) * 256 + codes_crop
        pc = np.bincount(pair.ravel(), minlength=(nlab + 1) * 256
                         ).reshape(nlab + 1, 256)
        pc[:, 0] = 0
        maj = pc.argmax(1).astype(np.uint8)
        crop_px = pc.max(1)
        crop_tot = pc.sum(1)
        # mean AEF vector per field: the field labels mapped ONTO the AEF
        # grid once (nearest), then one weighted bincount per band at 20 m
        # stride (the mean does not need every 10 m sample)
        mos, ax0, ay0 = _aef_mosaic(year, float(lonv[0]), float(latv[-1]),
                                    float(lonv[-1]), float(latv[0]))
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
            V[:, b] = np.bincount(la, weights=_DQ[q[b].view(np.uint8)].ravel(),
                                  minlength=nlab + 1)
        ok = counts > 0
        V[ok] /= counts[ok, None]
        nrm = np.linalg.norm(V, axis=1)
        kept = (ok & (sizes >= MIN_FIELD_PX) & (maj > 0)
                & (crop_tot >= MIN_CROP_FRAC * np.maximum(sizes, 1))
                & (nrm > 0))
        kept[0] = False
        V[nrm > 0] /= nrm[nrm > 0, None]
        ids = np.flatnonzero(kept)
        agree = np.full(nlab + 1, np.nan, dtype=np.float32)
        nbr_tally = {}
        if len(ids) > K_NBR:
            Vk = V[ids]
            S_ = Vk @ Vk.T
            np.fill_diagonal(S_, -2.0)
            kk = min(K_NBR, len(ids) - 1)
            nb = np.argpartition(-S_, kk, axis=1)[:, :kk]
            nb_maj = maj[ids][nb]
            same = nb_maj == maj[ids][:, None]
            agree[ids] = same.mean(1)
            for i, fid in enumerate(ids):
                t = {}
                for c in nb_maj[i]:
                    t[int(c)] = t.get(int(c), 0) + 1
                nbr_tally[int(fid)] = sorted(t.items(), key=lambda x: -x[1])
        latm = math.radians((S + N) / 2)
        pxa = ((FTW_RES * 111.32 * math.cos(latm))
               * (FTW_RES * 110.574)) * ACRES_PER_KM2
        # the paint LUT: agreement through viridis (bright = agrees, dark =
        # low agreement, the leads); the rest faint grey; nothing at label 0
        rgba = np.zeros((nlab + 1, 4), dtype=np.uint8)
        rgba[1:, :3] = 150
        rgba[1:, 3] = 45
        rgba_cdl = rgba.copy()
        if len(ids):
            idx = (np.clip(agree[ids], 0, 1) * 255).astype(np.uint8)
            rgba[ids, :3] = AGREE_LUT[idx]
            rgba[ids, 3] = 220
            # the CDL view: the same fields by their majority crop's colour
            # (the store's palette, red-dominant classes remapped)
            rgba_cdl[ids, :3] = _CLASS_RGB[maj[ids]]
            rgba_cdl[ids, 3] = 220
        ft = {"lab": lab, "fx0": fx0, "fy0": fy0, "maj": maj,
              "agree": agree, "sizes": sizes, "crop_px": crop_px,
              "crop_tot": crop_tot, "kept": kept, "nbr": nbr_tally,
              "pxa": pxa, "rgba": rgba, "rgba_cdl": rgba_cdl,
              "fyear": fyear, "year": year, "nfields": int(len(ids))}
        _fc[_ck] = ft
        if len(_fc) > 6:
            for _k in list(_fc)[:2]:
                _fc.pop(_k, None)
        return ft

    # ---- state + tile serve -------------------------------------------------
    _state = (_year, _cdl_mode)
    _tiles = HOLD.setdefault("tiles", {})

    def _states_in(W, S, E, N):
        return [st for st, xmin, ymin, xmax, ymax in STATES
                if xmax > W and xmin < E and ymax > S and ymin < N]

    def _poly_table(keep_r):
        """geoarrow.polygon pyarrow table from a list of closed rings."""
        flat = np.concatenate(keep_r)
        ro = np.zeros(len(keep_r) + 1, dtype=np.int32)
        ro[1:] = np.cumsum([len(r) for r in keep_r])
        po = np.arange(len(keep_r) + 1, dtype=np.int32)
        coords = pa.FixedSizeListArray.from_arrays(
            pa.array(flat.ravel(), type=pa.float64()), 2)
        ringsarr = pa.ListArray.from_arrays(pa.array(ro), coords)
        polysarr = pa.ListArray.from_arrays(pa.array(po), ringsarr)
        schema = pa.schema([pa.field("geometry", polysarr.type, metadata={
            b"ARROW:extension:name": b"geoarrow.polygon"})])
        return pa.Table.from_arrays([polysarr], schema=schema)

    def _push_polys(ft, W, S, E, N):
        """The vector fill: closed rings from the PMTiles joined to the
        batch's fields (each ring's vertex mean into the label image). This
        runs on the SERVE THREAD, which under marimo cannot hand the
        frontend a new widget or an updated table (the in-place push of
        2026-08-24 drew once at startup and never again: fills vanished on
        every pan, seen in playwright). So it only STORES the built data and
        pokes the HUD; the HUD's JS replies through ctl, and the "polys" act
        below rebuilds the layer IN A CELL RUN."""
        sig = (ft["year"], _cdl_mode, ft["fx0"], ft["fy0"], ft["lab"].shape)
        if HOLD.get("polys_sig") == sig:
            return
        rings = ftw_tile_polys(_states_in(W, S, E, N), ft["fyear"],
                               W, S, E, N, min(13, FTW_TILE_ZMAX))
        lab = ft["lab"]
        rgba = ft["rgba_cdl" if _cdl_mode else "rgba"]
        keep_r, cols, fids = [], [], []
        for r in rings:
            m = r.mean(axis=0)
            gx = int((m[0] + 180.0) / FTW_RES) - ft["fx0"]
            gy = int((FTW_Y0 - m[1]) / FTW_RES) - ft["fy0"]
            fid = 0
            if 0 <= gy < lab.shape[0] and 0 <= gx < lab.shape[1]:
                fid = int(lab[gy, gx])
            if fid == 0 or rgba[fid][3] < 200:   # sit-outs stay outline-only
                continue
            keep_r.append(r)
            fids.append(fid)
            c = rgba[fid].copy()
            c[3] = 255
            cols.append(c)
        if not keep_r:
            return
        HOLD["polys_data"] = {
            "tbl": _poly_table(keep_r),
            "cols": np.asarray(cols, dtype=np.uint8),
            "fids": np.asarray(fids, dtype=np.int64),
            "rings": keep_r,
            "ft_key": (ft["fx0"], ft["fy0"], ft["lab"].shape),
            "sig": sig,
        }

        def _pp():
            try:
                hud.widget.poke = f"{time.time():.4f}"
            except Exception as _e:
                _say(f"polys poke error: {type(_e).__name__}: {_e}")

        _loop = HOLD.get("loop")
        if _loop is not None:
            _loop.call_soon_threadsafe(_pp)
        else:
            _pp()

    def _panel_html(ft):
        if _cdl_mode:
            # the CDL view: what grows in view, by acres
            ids = np.flatnonzero(ft["kept"])
            if not len(ids):
                return "no crop fields in view"
            ac = np.bincount(ft["maj"][ids], weights=ft["sizes"][ids],
                             minlength=256) * ft["pxa"]
            order = [int(c) for c in np.argsort(-ac) if ac[c] > 0][:8]
            parts = [f"{_chip(_cname(c)[1])}{_cname(c)[0]} {ac[c] / 1e3:,.1f}k ac"
                     for c in order]
            return (f"<b>CDL {ft['year']} in view</b> (each field its "
                    f"majority crop): " + " · ".join(parts))
        rows = []
        ids = np.flatnonzero(ft["kept"] & np.isfinite(ft["agree"]))
        if len(ids):
            ac = ft["sizes"][ids] * ft["pxa"]
            order = ids[np.argsort(ft["agree"][ids])]
            for fid in order:
                if ft["sizes"][fid] * ft["pxa"] < PANEL_MIN_AC:
                    continue
                if ft["agree"][fid] >= 0.5:
                    break
                nm, hx, _nc = _cname(ft["maj"][fid])
                t = ft["nbr"].get(int(fid), [])
                tt = ", ".join(f"{_cname(c)[0]} {n}/{K_NBR}" for c, n in t[:2])
                rows.append(f"{_chip(hx)}CDL says {nm} "
                            f"({ft['sizes'][fid] * ft['pxa']:,.0f} ac), "
                            f"look-alikes say {tt}")
                if len(rows) >= 4:
                    break
            med = float(np.nanmedian(ft["agree"][ids]))
            head = (f"<b>{ft['nfields']} fields scored</b> · each: of the "
                    f"{K_NBR} fields in view that look most like it "
                    f"(AlphaEarth), how many grow its CDL crop? · median "
                    f"{100 * med:.0f}%")
            body = ("<div style='margin-top:2px'><b>the dark fields "
                    "(leads):</b> " + " · ".join(rows) + "</div>") if rows else \
                   ("<div style='margin-top:2px;opacity:.6'>no dark fields: "
                    "AlphaEarth and CDL agree everywhere in view</div>")
            return head + body
        return "no crop fields in view"

    def _serve_batch(z, keys):
        t0 = time.time()
        boxes = [tile_box(z, x, y) for (_st, _z, x, y) in keys]
        W = min(b[0] for b in boxes)
        S = min(b[1] for b in boxes)
        E = max(b[2] for b in boxes)
        N = max(b[3] for b in boxes)
        ft = None
        note = ""
        if z >= AEF_ZMIN:
            ft = _field_table(_year, W, S, E, N)
            HOLD["ftab"] = ft
            _push_polys(ft, W, S, E, N)
        else:
            note = f" · zoom in for the agreement paint (from z{AEF_ZMIN})"
        rings, nt = None, 0
        if z >= OUTLINE_ZMIN:
            _fy = _year if _year in FTW_YEARS else FTW_YEARS[0]
            rings, nt, _nm = ftw_tile_rings(
                _states_in(W, S, E, N), _fy, W, S, E, N, min(z, FTW_TILE_ZMAX))
        # tiles: nearest sample of the per-field colour through the label grid
        line_col = (40, 40, 40, 160)
        if rings:
            rb = np.array([[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()]
                           for r in rings]) if len(rings) else np.zeros((0, 4))
        # the tiles carry ONLY the outlines now (2026-08-24 night: the fill
        # is the polygon layer; the raster keeps the seam-free strokes and
        # the z12 view)
        pngs = []
        for (tW, tS, tE, tN) in boxes:
            out = np.zeros((TILE_PX, TILE_PX, 4), dtype=np.uint8)
            img = Image.fromarray(out, "RGBA")
            if rings:
                hit = np.flatnonzero((rb[:, 0] < tE) & (rb[:, 2] > tW)
                                     & (rb[:, 1] < tN) & (rb[:, 3] > tS))
                if len(hit):
                    d = ImageDraw.Draw(img)
                    sx, sy = TILE_PX / (tE - tW), TILE_PX / (tN - tS)
                    for hi in hit:
                        r = rings[hi]
                        pts = list(zip((r[:, 0] - tW) * sx, (tN - r[:, 1]) * sy))
                        if len(pts) >= 2:
                            d.line(pts, fill=line_col, width=1)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=False, compress_level=4)
            pngs.append(buf.getvalue())
        for key, png in zip(keys, pngs):
            if key[0] == _state:
                _tiles[key] = EncodedImage(data=png, media_type="image/png")
        if len(_tiles) > TILE_CACHE:
            for _k in list(_tiles)[:TILE_CACHE // 4]:
                _tiles.pop(_k, None)
        if ft is not None:
            line = (f"z{z} · {len(keys)} tiles · {ft['nfields']} crop fields · "
                    f"CDL + AEF {_year}"
                    + (f" · FTW {ft['fyear']} footprint" if ft["fyear"] != _year else "")
                    + f" · {int((time.time() - t0) * 1000)} ms")
            phtml = _panel_html(ft)
        else:
            line = (f"z{z} · {len(keys)} tiles{note} · "
                    f"{int((time.time() - t0) * 1000)} ms")
            phtml = None

        def _push():
            HOLD.setdefault("last_by_state", {})[_state] = line
            _say(line)
            if phtml is not None:
                HOLD["batch_html"] = phtml
                try:
                    hud.widget.panel = (HOLD.get("sel_html", "")
                                        + HOLD["batch_html"])
                except Exception:
                    pass

        _loop = HOLD.get("loop")
        if _loop is not None:
            _loop.call_soon_threadsafe(_push)
        else:
            _push()

    def _view_tiles(z, state=None):
        vs = _vsd(HOLD.get("vs")) or dict(HOME)
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
        return {(state or _state, z, x, y) for x in xs for y in ys}

    def _make_raster():
        _tms0 = morecantile.tms.get("WebMercatorQuad")
        _m = 20037508.342789244
        _tms = _tms0.model_copy(update={"boundingBox": morecantile.models.TMSBoundingBox(
            lowerLeft=(-_m, -_m), upperRight=(_m, _m), crs=_tms0.crs)})
        return RasterLayer(
            _tile_matrix_set=_tms,
            _crs=_tms.crs,
            _fetch_tile=HOLD["fetch"],
            _render_tile=HOLD["render"],
            min_zoom=AEF_ZMIN - 1,
            max_zoom=TILE_ZMAX,
            extent=EXTENT,
            _tile_size=TILE_PX,
            debounce_time=30,
            opacity=1.0,
            pickable=False,
        )

    def _legend_html():
        if _cdl_mode:
            return ('<span style="opacity:.7">fields coloured by their CDL '
                    'majority crop (the panel lists them) · untick to see '
                    'the agreement score</span>')
        stops = ", ".join(f"rgb({r},{g},{b}) {i / 15 * 100:.0f}%"
                          for i, (r, g, b) in enumerate(
                              AGREE_LUT[np.linspace(0, 255, 16).astype(int)]))
        bar = (f'<span style="display:inline-block;width:9rem;height:10px;'
               f'border-radius:3px;background:linear-gradient(90deg,{stops})"></span>')
        return (f'<span style="opacity:.8">disagrees</span>{bar}'
                f'<span style="opacity:.8">agrees</span>'
                f'<span style="opacity:.6">bright = its look-alikes grow what '
                f'CDL says it grows · dark = they don\'t (a lead) · '
                f'click a field for its story</span>')

    def _compose_layers():
        """polygons under, raster outlines on top. NO empty step (a removed
        layer is closed under marimo; the raster's per-instance deck id
        makes replacement a new layer). polys is the map cell's immortal
        widget."""
        deck.layers = [HOLD["polys"], HOLD["raster"]]

    def _rebuild():
        HOLD["batch"] = None
        HOLD["raster"] = _make_raster()
        HOLD["layer_state"] = _state
        HOLD["polys_sig"] = None    # colours may be stale for the new state
        _compose_layers()
        try:
            hud.widget.legend = _legend_html()
        except Exception:
            pass
        _last = HOLD.get("last_by_state", {}).get(_state)
        if _last is not None:
            _say(_last + " · from cache")
        else:
            _say(f"CDL + AlphaEarth {_year} · scoring the fields in view "
                 f"(paint from camera ~z12) · loading …")

    async def _run_batch(b):
        try:
            await asyncio.sleep(BATCH_S)
            b["closed"] = True
            b["keys"] |= {k for k in b["view"](b["z"], b["state"]) if k not in _tiles}
            keys = sorted(b["keys"])
            await asyncio.get_running_loop().run_in_executor(None, b["serve"], b["z"], keys)
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

    async def _fetch(x, y, z):
        key = (_state, z, x, y)
        hit = _tiles.get(key)
        if hit is not None:
            return hit
        loop = asyncio.get_running_loop()
        HOLD["loop"] = loop
        b = HOLD.get("batch")
        if b is None or b["closed"] or b["z"] != z or b["state"] != _state:
            b = {"z": z, "state": _state, "keys": set(), "closed": False,
                 "fut": loop.create_future(),
                 "serve": _serve_batch, "view": _view_tiles}
            b["fut"].add_done_callback(lambda f: f.cancelled() or f.exception())
            b["task"] = loop.create_task(_run_batch(b))
            HOLD["batch"] = b
        b["keys"].add(key)
        await asyncio.shield(b["fut"])
        return _tiles.get(key)

    def _render(tile):
        return tile

    HOLD["fetch"], HOLD["render"] = _fetch, _render

    def _on_vs(change):
        try:
            vs = _vsd(change.new)
            if vs is None:
                return
            HOLD["vs"] = vs
            W0, S0, E0, N0 = EXTENT
            M = 2.0
            lon = min(max(vs["longitude"], W0 - M), E0 + M)
            lat = min(max(vs["latitude"], S0 - M), N0 + M)
            zoom = max(vs["zoom"], VIEW_ZMIN)
            if ((lon, lat, zoom) != (vs["longitude"], vs["latitude"], vs["zoom"])
                    and not HOLD.get("clamping")):
                HOLD["clamping"] = True
                try:
                    deck.set_view_state(longitude=lon, latitude=lat, zoom=zoom)
                finally:
                    HOLD["clamping"] = False
        except Exception as _e:
            _say(f"camera error: {type(_e).__name__}: {_e}")

    _old = HOLD.get("h_vs")
    if _old is not None:
        try:
            deck.unobserve(_old, names="view_state")
        except Exception:
            pass
    deck.observe(_on_vs, names="view_state")
    HOLD["h_vs"] = _on_vs

    # ---- the acts -----------------------------------------------------------
    if _act == "polys":
        # the serve thread's reply: update the ONE polygon layer HERE, in a
        # cell run, in place (a trait assignment from the serve thread never
        # reached a live frontend; creating a new widget mid-session blanked
        # the whole deck: both found with playwright, 2026-08-24 night)
        _pd = HOLD.get("polys_data")
        if _pd is not None and HOLD.get("polys_sig") != _pd["sig"]:
            try:
                _lay = HOLD["polys"]
                with _lay.hold_trait_notifications():
                    _lay.table = A3Table.from_arrow(_pd["tbl"])
                    _lay.get_fill_color = _pd["cols"]
                HOLD["polys_sig"] = _pd["sig"]
            except Exception as _e:
                _say(f"polys error: {type(_e).__name__}: {_e}")

    if _act == "click":
        # a click INSPECTS: the field's story into the panel, and the field
        # itself LIT on the map (Stephen: "select the field visually"): its
        # rings from the stored polygon data as a white overlay layer,
        # created here in the cell run
        try:
            _vs = _vsd(HOLD.get("vs")) or dict(HOME)
            _lon, _lat = unproject(_vs, float(_c["px"]), float(_c["py"]),
                                   float(_c["w"]), float(_c["h"]))
            ft = HOLD.get("ftab")
            if ft is None:
                _say("zoom in first: the fields are scored from camera ~z12")
            else:
                gx = int((_lon + 180.0) / FTW_RES) - ft["fx0"]
                gy = int((FTW_Y0 - _lat) / FTW_RES) - ft["fy0"]
                fid = 0
                if 0 <= gy < ft["lab"].shape[0] and 0 <= gx < ft["lab"].shape[1]:
                    fid = int(ft["lab"][gy, gx])
                _pd = HOLD.get("polys_data")
                _pd_ok = (_pd is not None and HOLD.get("polys_sig") == _pd["sig"]
                          and _pd["ft_key"] == (ft["fx0"], ft["fy0"], ft["lab"].shape))
                if fid == 0:
                    HOLD["sel_html"] = ""
                    if _pd_ok:
                        try:
                            HOLD["polys"].get_fill_color = _pd["cols"]
                        except Exception:
                            pass
                    _say(f"no FTW field at {_lat:.4f}, {_lon:.4f}")
                else:
                    # light the field (Stephen: "select the field visually"):
                    # the selection is a COLOUR on the one polygon layer (a
                    # second layer collides on deck id under marimo): the
                    # clicked field's polygons go white, everything else
                    # keeps its colour
                    if _pd_ok:
                        _m = _pd["fids"] == fid
                        if _m.any():
                            try:
                                _c2 = _pd["cols"].copy()
                                _c2[_m] = (255, 255, 255, 255)
                                HOLD["polys"].get_fill_color = _c2
                            except Exception as _e:
                                _say(f"highlight error: {type(_e).__name__}: {_e}")
                    nm, hx, _nc = _cname(ft["maj"][fid])
                    ac = ft["sizes"][fid] * ft["pxa"]
                    if ft["kept"][fid] and np.isfinite(ft["agree"][fid]):
                        ag = ft["agree"][fid]
                        t = ft["nbr"].get(fid, [])
                        tt = " · ".join(
                            f"{_chip(_cname(c)[1])}{_cname(c)[0]} {n}/{K_NBR}"
                            for c, n in t[:4])
                        pur = 100 * ft["crop_px"][fid] / max(ft["crop_tot"][fid], 1)
                        HOLD["sel_html"] = (
                            f"this field: {_chip(hx)}<b>{nm}</b> per CDL "
                            f"{ft['year']} ({pur:.0f}% of its crop pixels) · "
                            f"{ac:,.0f} ac · agreement "
                            f"<b>{100 * ag:.0f}%</b>: its look-alikes are {tt}"
                            f"<hr style='border:none;border-top:1px solid "
                            f"rgba(127,127,127,.25);margin:.3rem 0'>")
                    else:
                        HOLD["sel_html"] = (
                            f"this field sits out ({ac:,.0f} ac: "
                            + ("too small" if ft["sizes"][fid] < MIN_FIELD_PX
                               else "not a crop field by CDL")
                            + ")<hr style='border:none;border-top:1px solid "
                            "rgba(127,127,127,.25);margin:.3rem 0'>")
                    try:
                        hud.widget.panel = (HOLD["sel_html"]
                                            + HOLD.get("batch_html", ""))
                    except Exception:
                        pass
        except Exception as _e:
            _say(f"click error: {type(_e).__name__}: {_e}")

    if _act == "refresh":
        HOLD["layer_state"] = None

    def _photon_first(query, vs):
        _params = {"q": query, "limit": 1, "lang": "en"}
        if isinstance(vs, dict) and vs.get("longitude") is not None:
            _params["lon"] = round(vs["longitude"], 4)
            _params["lat"] = round(vs["latitude"], 4)
        _url = "https://photon.komoot.io/api/?" + urllib.parse.urlencode(_params)
        _req = urllib.request.Request(
            _url, headers={"User-Agent": "cdl-ftw-zarr-marimo aef notebook"}
        )
        with urllib.request.urlopen(_req, timeout=10) as _r:
            _data = json.load(_r)
        _feats = _data.get("features") or []
        if not _feats:
            return None
        _f = _feats[0]
        _p = _f.get("properties", {})
        _lon, _lat = _f["geometry"]["coordinates"][:2]
        _name = ", ".join(
            str(v) for v in (_p.get("name"), _p.get("city"), _p.get("state")) if v
        ) or query
        return _name, _lon, _lat, _p.get("extent")

    if _act == "search" and _q:
        try:
            _hit = _photon_first(_q, _vsd(HOLD.get("vs")))
        except Exception as _e:
            _hit = None
            _say(f"search error: {type(_e).__name__}: {_e}")
        if _hit is None:
            _say(f"no match: {_q}")
        else:
            _name, _lon, _lat, _ext = _hit
            if _ext and len(_ext) == 4:
                _span = max(abs(_ext[2] - _ext[0]), abs(_ext[1] - _ext[3]) * 2, 0.01)
                _zoom = math.log2(360.0 * (VIEW_W / 512) / _span) - 0.3
            else:
                _zoom = 12.5
            _zoom = max(3.5, min(13.5, _zoom))
            _nvs = {"longitude": _lon, "latitude": _lat, "zoom": _zoom}
            HOLD["vs"] = _nvs
            deck.fly_to(longitude=_lon, latitude=_lat, zoom=_zoom, duration=2000)
            _say(f"→ {_name}")
            HOLD["layer_state"] = None

    if HOLD.get("layer_state") != _state or HOLD.get("raster") is None:
        _rebuild()
    return


if __name__ == "__main__":
    app.run()
