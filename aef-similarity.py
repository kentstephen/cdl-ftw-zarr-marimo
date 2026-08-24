# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xarray",
#     "zarr>=3",
#     "obstore",
#     "pyarrow>=25.0.0",
#     "numpy",
#     "anywidget>=0.9",
#     "lonboard>=0.16.0,<0.17",
#     "arro3-core",
#     "pillow==12.3.0",
#     "morecantile==7.0.3",
#     "ipywidgets==8.1.8",
#     "traitlets==5.15.1",
# ]
# ///
"""Click a field: AlphaEarth similarity over Fields of the World (CONUS first).

The first AEF notebook (branch aef-similarity, 2026-08-24). Click a field on
the map; the kernel flood-fills the FTW P(field) >= 0.5 grid at 10 m around the
click into that field's pixels, averages the AlphaEarth embedding (64-dim,
10 m, annual) over them into one unit vector, and repaints the view as
per-pixel cosine similarity to it (viridis, protan-safe). A click off any
field uses the 3x3 mean at the point. The year slider picks WHICH YEAR'S
embeddings the reference is compared against: the reference keeps the year it
was clicked in, so 2024's field vs 2020's embeddings is a cross-year question,
stated in the status line.

WHICH DATA, FROM WHERE (all anonymous S3 on us-west-2.opendata.source.coop):

  AlphaEarth Foundations annual embeddings, 2017-2025, 10 m, EPSG:4326
    tge-labs/aef-mosaic  (plain Zarr v3, sharded)
    embeddings(time, band, y, x) int8, shards (1, 64, 4096, 4096) with INNER
    chunks (1, 64, 256, 256): any pixel read pulls all 64 bands of its 256-px
    chunk (~4 MB raw), which is what a similarity needs anyway. Dequantize
    (v / 127.5)^2 * sign(v), nodata -128. Vectors are ~unit norm (0.9989
    measured at home). NO PYRAMID: similarity tiles exist from tile z13
    (camera ~z12); below that the layer requests nothing (min_zoom).
  FTW P(field) probabilities, 2024 + 2025, 10 m, EPSG:4326
    tge-labs/ftw-global-data/predictions/zarr/alpha/global.zarr (root group,
    full resolution: the flood fill; the map's clip machinery is not here)
  FTW field outlines, one PMTiles per US state (tippecanoe z0-13)
    same account, ranged GETs + hand-rolled PMTiles v3 + MVT decode, copied
    from cdl-ftw.py (itself the HRRR counties film's, by copy)

The tile serve, the batch machinery, the HUD strip, the rebuild-as-new-layer
rule and the lonboard JS patch are cdl-ftw.py's, trimmed (no CDL here, no
DuckDB, no analyze). THE CLICK IS NOT lonboard's on_click (never worked under
marimo in the last notebook, Stephen 2026-08-24): the HUD's JS listens for
plain clicks on the map CANVAS (composedPath through the shadow roots, a drag
guard), sends the pixel + canvas rect through the proven `ctl` trait, and the
kernel unprojects with the view state it already tracks. Everything that
changes the map happens IN a cell run, like a toggle.

Run from THIS repo's venv (the first cell applies the lonboard JS patch to
whatever environment is executing the notebook):
  uv sync && uv run marimo edit aef-similarity.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    # ---- the lonboard JS patch, applied here in the RUNNING environment
    # (cdl-ftw.py's first cell, same reasons: --sandbox builds a fresh env and
    # an unpatched lonboard cost a day on 2026-08-21). Idempotent.
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

    import anywidget
    import obstore
    import xarray as xr
    import zarr
    import traitlets
    import urllib.parse
    import urllib.request

    import morecantile
    from lonboard import Map, RasterLayer
    from lonboard.raster import EncodedImage
    from lonboard.basemap import CartoStyle, MaplibreBasemap
    from obstore.store import S3Store

    import marimo as mo

    return (
        CartoStyle,
        EncodedImage,
        Image,
        ImageDraw,
        Map,
        MaplibreBasemap,
        RasterLayer,
        S3Store,
        ThreadPoolExecutor,
        anywidget,
        asyncio,
        gzip,
        io,
        json,
        math,
        mo,
        morecantile,
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
    # Click a field: AlphaEarth similarity

    **Zoom to farmland (similarity lives from camera ~z12), click a field.**
    The kernel flood-fills that field out of the FTW P(field) grid at 10 m,
    averages the **AlphaEarth Foundations** embedding (64 numbers per 10 m
    pixel, one set per year 2017-2025, everything Sentinel and more saw that
    year compressed into a vector) over the field's pixels, and repaints the
    view as **cosine similarity** to it: bright viridis = "looks like the
    field you clicked". A click off any field compares against that point's
    own vector. The year slider picks which year's embeddings are compared;
    the reference keeps its click year, so sliding is a cross-year question.

    | data | what | how it is read |
    |---|---|---|
    | AlphaEarth mosaic `embeddings(time, band, y, x)`, int8, 10 m | the vectors | plain Zarr v3; 256-px inner chunks cached in memory and on disk |
    | FTW P(field), 10 m, 2024-2025 | the field a click lands in | one 512-px window + flood fill per click |
    | FTW per-state PMTiles | field outlines from z12 | ranged GETs + MVT decode (cdl-ftw.py's, by copy) |

    No CDL in this notebook yet: this is the debug view for the pairing
    (the agreement map is next). CONUS only for now, same camera clamp.
    """)
    return


@app.cell
def _():
    # ---- constants ----------------------------------------------------------
    SC_BUCKET = "us-west-2.opendata.source.coop"
    AEF_ZARR = "tge-labs/aef-mosaic/"
    FTW_ZARR = "tge-labs/ftw-global-data/predictions/zarr/alpha/global.zarr/"
    FTW_VEC = (
        "tge-labs/ftw-global-data/predictions/vectors/alpha/"
        "results-by-admin-conf/admin:country_code=US/"
    )

    # the AEF mosaic's own affine (read from its zarr.json attrs, 2026-08-24);
    # NOT the FTW grid: same nominal 10 m but a different y origin, so each is
    # sampled by its own transform, never by shared indices
    AEF_RES = 8.983111749910169e-05
    AEF_Y0 = 83.68570533713473
    AEF_X0 = -180.0
    AEF_SHAPE = (1859584, 4009984)   # (y, x)
    AEF_YEARS = list(range(2017, 2026))
    YEAR0 = 2024                     # embeddings year at open (FTW's first year)

    FTW_RES = 8.98311982e-05         # the FTW grid (cdl-ftw.py's constants)
    FTW_Y0 = 83.748345
    FTW_YEARS = (2024, 2025)

    ACH = 256                        # AEF inner chunk (px): the cache unit
    AEF_MEM_CHUNKS = 192             # int8 chunks in memory (4 MB each, ~0.8 GB)
    SIM_LO = 0.4                     # ramp floor: cosine 0.4 -> dark, 1.0 -> bright

    TILE_PX = 256
    BATCH_S = 0.05
    TILE_CACHE = 3000
    AEF_ZMIN, TILE_ZMAX = 13, 15     # NO PYRAMID: no similarity below tile z13
    # (camera ~z12; a z12-camera view is ~15-20 inner chunks, 60-80 MB raw,
    # the practical floor; one zoom out quadruples it)
    VIEW_ZMIN = 3.0
    EXTENT = [-125.0, 24.0, -66.5, 49.8]   # CONUS, as in cdl-ftw.py
    FTW_TILE_ZMAX = 13
    OUTLINE_ZMIN = 12
    MARGIN = 0.35
    VIEW_W, VIEW_H = 1400, 700
    HOME = {"longitude": -121.45, "latitude": 37.95, "zoom": 12.5}  # the Delta

    HOLD: dict = {}
    return (
        ACH,
        AEF_MEM_CHUNKS,
        AEF_RES,
        AEF_SHAPE,
        AEF_X0,
        AEF_Y0,
        AEF_YEARS,
        AEF_ZARR,
        AEF_ZMIN,
        BATCH_S,
        EXTENT,
        FTW_RES,
        FTW_TILE_ZMAX,
        FTW_VEC,
        FTW_Y0,
        FTW_YEARS,
        FTW_ZARR,
        HOLD,
        HOME,
        MARGIN,
        OUTLINE_ZMIN,
        SC_BUCKET,
        SIM_LO,
        TILE_CACHE,
        TILE_PX,
        TILE_ZMAX,
        VIEW_H,
        VIEW_W,
        VIEW_ZMIN,
        YEAR0,
    )


@app.cell
def _(AEF_ZARR, FTW_ZARR, S3Store, SC_BUCKET, xr, zarr):
    # ---- open both stores ----------------------------------------------------
    # AEF: the whole mosaic as one dataset; every read below is chunk-aligned
    # isel windows, so the shard never expands whole (the inner chunks do).
    _aef_store = zarr.storage.ObjectStore(
        S3Store(bucket=SC_BUCKET, region="us-west-2", skip_signature=True,
                prefix=AEF_ZARR),
        read_only=True,
    )
    import warnings as _warnings
    with _warnings.catch_warnings():
        # the repo keeps a .checkpoint dir and a README next to the arrays
        _warnings.simplefilter("ignore")
        AEF_DS = xr.open_zarr(_aef_store, chunks=None, consolidated=False)

    # FTW: the ROOT group only (full 10 m): the per-click flood fill reads one
    # 512-px window. The map-clip pyramid machinery of cdl-ftw.py is not here.
    _ftw_store = zarr.storage.ObjectStore(
        S3Store(bucket=SC_BUCKET, region="us-west-2", skip_signature=True,
                prefix=FTW_ZARR),
        read_only=True,
    )
    FTW_ROOT = xr.open_zarr(_ftw_store, chunks=None, consolidated=False)
    return AEF_DS, FTW_ROOT


@app.cell
def _(MARGIN, VIEW_H, VIEW_W, math, np):
    # ---- pure helpers (defined above their use; marimo drops underscore
    # temporaries a forward reference does not keep) ---------------------------
    def tile_box(z, x, y):
        """Web Mercator tile -> lon/lat (W, S, E, N)."""
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
        """Canvas pixel -> lon/lat under the deck camera (512-px world tiles,
        the same scale bbox4326 uses). The click path: the HUD's JS sends the
        pixel and the canvas rect; the kernel does the math with the view
        state it already tracks (lonboard's on_click never worked here)."""
        world = 512 * 2 ** vs["zoom"]
        lon = vs["longitude"] + (px - w / 2) * 360.0 / world
        lat0 = math.radians(vs["latitude"])
        uy = (1 - math.log(math.tan(lat0) + 1 / math.cos(lat0)) / math.pi) / 2
        uy = uy + (py - h / 2) / world
        lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * uy))))
        return lon, lat

    # viridis, 16 anchors -> 256 rows (protan-safe: a luminance ramp)
    _VIR = np.array([
        (68, 1, 84), (72, 26, 108), (71, 47, 125), (65, 68, 135),
        (57, 86, 140), (49, 104, 142), (42, 120, 142), (35, 136, 142),
        (31, 152, 139), (34, 168, 132), (53, 183, 121), (84, 197, 104),
        (122, 209, 81), (165, 219, 54), (210, 226, 27), (253, 231, 37),
    ], dtype=np.float64)
    _t = np.linspace(0, 1, 256)
    _a = np.linspace(0, 1, len(_VIR))
    VIRIDIS = np.stack([np.interp(_t, _a, _VIR[:, i]) for i in range(3)],
                       axis=1).astype(np.uint8)
    return VIRIDIS, bbox4326, tile_box, unproject


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
    # ---- FTW field OUTLINES from the per-state PMTiles: cdl-ftw.py's cell,
    # by copy (which is itself the HRRR counties film's, by copy). Raw tiles
    # cached on disk (the same dir cdl-ftw uses, shared), polylines in memory.
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
        """One tile -> polylines (lon/lat) of the year's layer; clip-line
        segments dropped (no seams), polylines NOT closed (cdl-ftw's rule)."""
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

    # state extents for picking archives by view (cdl-ftw.py's, from the
    # files' own row-group stats; the STAC bboxes are wrong)
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
    _ = FTW_TILE_ZMAX  # documented cap; callers clamp z themselves
    return STATES, ftw_tile_rings


@app.cell
def _(anywidget, traitlets):
    class HudControls(anywidget.AnyWidget):
        """Controls + status UNDER the map, cdl-ftw.py's strip trimmed to this
        notebook: year slider (2017-2025, the EMBEDDINGS year), clear, refresh,
        search, a legend area (the viridis ramp + the reference's description).
        Plus THE CLICK: a capture-phase listener on the document finds the map
        canvas in the event's composedPath (shadow roots included), guards
        against drags, and sends the canvas pixel + rect through `ctl`. Proven
        trait types only; commits on change."""

        ctl = traitlets.Unicode("").tag(sync=True)
        status = traitlets.Unicode("").tag(sync=True)
        legend = traitlets.Unicode("").tag(sync=True)

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
          yl.textContent = "embeddings year";
          const range = document.createElement("input");
          range.type = "range";
          range.min = "2017"; range.max = "2025"; range.step = "1";
          range.value = "2024";
          range.style.cssText = "width:9rem";
          const yv = document.createElement("span");
          yv.style.cssText = "font-weight:600;font-variant-numeric:tabular-nums";
          yv.textContent = range.value;
          const clr = document.createElement("button");
          clr.textContent = "clear";
          clr.title = "drop the reference; outlines only";
          clr.style.cssText = btnCss;
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
          box.append(yl, range, yv, clr, rfr, search, legendBox);
          const status = document.createElement("div");
          status.style.cssText =
            "font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;" +
            "opacity:.85;padding:.15rem 0;min-height:1.2em";
          const wrap = document.createElement("div");
          wrap.dataset.aefStrip = "1";
          wrap.append(box, status);
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
          let seq = 0, deb = null;
          const send = (act, extra) => {
            model.set("ctl", JSON.stringify(Object.assign({
              act: act, year: +range.value, n: ++seq }, extra || {})));
            model.save_changes();
          };
          const commit = () => {
            clearTimeout(deb);
            deb = setTimeout(() => send("set"), 250);
          };
          range.addEventListener("input", () => { yv.textContent = range.value; });
          range.addEventListener("change", commit);
          clr.addEventListener("click", () => send("clear"));
          rfr.addEventListener("click", () => send("refresh"));
          search.addEventListener("keydown", (e) => {
            const q = search.value.trim();
            if (e.key === "Enter" && q) send("search", { q: q });
          });
          // THE CLICK: capture on the document, take only plain clicks whose
          // composedPath crosses a CANVAS (the deck/maplibre canvases share
          // the map's rect), send the canvas pixel + rect; the kernel
          // unprojects with its view state. Drag-guarded.
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
          // hide lonboard's draw-box tool (rendered unconditionally in 0.16)
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
def _(CartoStyle, HOLD: dict, HOME, LONBOARD_PATCHED, Map, MaplibreBasemap):
    # ---- map cell: builds the Map, must never re-run (cdl-ftw.py's rule).
    _ = LONBOARD_PATCHED
    HOLD["layer_state"] = None
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
    ACH,
    AEF_DS,
    AEF_MEM_CHUNKS,
    AEF_RES,
    AEF_SHAPE,
    AEF_X0,
    AEF_Y0,
    AEF_YEARS,
    AEF_ZMIN,
    BATCH_S,
    EXTENT,
    EncodedImage,
    FTW_ROOT,
    FTW_RES,
    FTW_TILE_ZMAX,
    FTW_Y0,
    FTW_YEARS,
    HOLD: dict,
    HOME,
    Image,
    ImageDraw,
    OUTLINE_ZMIN,
    RasterLayer,
    SIM_LO,
    STATES,
    TILE_CACHE,
    TILE_PX,
    TILE_ZMAX,
    VIEW_W,
    VIEW_ZMIN,
    VIRIDIS,
    YEAR0,
    asyncio,
    bbox4326,
    deck,
    ftw_tile_rings,
    hud,
    io,
    json,
    math,
    morecantile,
    np,
    os,
    tempfile,
    threading,
    tile_box,
    time,
    unproject,
    urllib,
):
    # ---- wiring cell: re-runs on every HUD commit; everything that changes
    # the map happens IN this run (cdl-ftw.py's law) -------------------------
    try:
        _c = json.loads(hud.widget.ctl or "{}")
    except Exception:
        _c = {}
    _year = int(_c.get("year", YEAR0))
    if _year not in AEF_YEARS:
        _year = YEAR0
    _act = _c.get("act", "set")
    _q = str(_c.get("q", "")).strip()

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

    # ---- the AEF chunk cache: int8 (64, 256, 256) per (year, cx, cy), in
    # memory and on disk (raw .npy, 4 MB), plus the per-chunk pixel NORMS
    # (float32, computed once: the reference changes, the norms do not) and
    # the per-(reference, chunk) SIMILARITY (float16). The dequantization is
    # a 256-entry lookup: (v / 127.5)^2 * sign(v), nodata -128 -> NaN.
    _AEF_DIR = os.path.join(tempfile.gettempdir(), "x-sql-marimo", "aef-emb")
    _DQ = np.zeros(256, dtype=np.float32)
    _qv = np.arange(-128, 128, dtype=np.float32)
    _DQ[(np.arange(-128, 128) & 0xFF)] = (np.abs(_qv) / 127.5) ** 2 * np.sign(_qv)
    _DQ[128] = np.nan   # int8 -128 viewed as uint8

    def _dequant(q):
        """int8 (64, h, w) -> float32; nodata -128 -> NaN on every band."""
        return _DQ[q.view(np.uint8)]

    def _aef_fetch(year, cx0, cx1, cy0, cy1, missing):
        """One chunk-aligned isel read over the missing chunks' bounding box,
        split into the cache (the _ftw_mask pattern). Blocking."""
        mem = HOLD.setdefault("aef_chunks", {})
        mx0, mx1 = min(c[0] for c in missing), max(c[0] for c in missing)
        my0, my1 = min(c[1] for c in missing), max(c[1] for c in missing)
        ya, yb = my0 * ACH, min((my1 + 1) * ACH, AEF_SHAPE[0])
        xa, xb = mx0 * ACH, min((mx1 + 1) * ACH, AEF_SHAPE[1])
        big = AEF_DS.embeddings.sel(time=year).isel(
            y=slice(ya, yb), x=slice(xa, xb)).values   # (64, H, W) int8
        for cx in range(mx0, mx1 + 1):
            for cy in range(my0, my1 + 1):
                # the bounding read fetched every chunk of the range; keep all
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
            for _k in list(mem)[:AEF_MEM_CHUNKS // 4]:
                mem.pop(_k, None)

    def _aef_chunks(year, cx0, cx1, cy0, cy1):
        """Every chunk of the range in memory; disk then network for misses."""
        mem = HOLD.setdefault("aef_chunks", {})
        missing = []
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                if (year, cx, cy) in mem:
                    continue
                fp = os.path.join(_AEF_DIR, str(year), f"{cx}_{cy}.npy")
                if os.path.exists(fp):
                    mem[(year, cx, cy)] = np.load(fp)
                else:
                    missing.append((cx, cy))
        if missing:
            _aef_fetch(year, cx0, cx1, cy0, cy1, missing)
        return mem

    def _chunk_sim(year, cx, cy, ref, seq):
        """Similarity image of one chunk vs the unit reference: float16
        (256, 256), NaN where nodata. Cached per (seq, year, chunk)."""
        sims = HOLD.setdefault("aef_sims", {})
        key = (seq, year, cx, cy)
        s = sims.get(key)
        if s is not None:
            return s
        q = HOLD["aef_chunks"][(year, cx, cy)]
        v = _dequant(q)                       # (64, 256, 256), NaN at nodata
        dot = np.einsum("k,kij->ij", ref, v)
        norms = HOLD.setdefault("aef_norms", {})
        n = norms.get((year, cx, cy))
        if n is None:
            n = np.sqrt(np.einsum("kij,kij->ij", v, v))
            n[n == 0] = 1.0
            norms[(year, cx, cy)] = n
            if len(norms) > 2000:
                for _k in list(norms)[:400]:
                    norms.pop(_k, None)
        s = (dot / n).astype(np.float16)
        sims[key] = s
        if len(sims) > 4000:
            for _k in list(sims)[:800]:
                sims.pop(_k, None)
        return s

    def _sim_mosaic(year, ref, seq, W, S, E, N):
        """Similarity over the box, assembled chunk-aligned: (float32 mosaic
        with NaN holes, ix0, iy0). Indices are the AEF grid's own."""
        ix0 = max(int((W - AEF_X0) / AEF_RES), 0)
        ix1 = min(int((E - AEF_X0) / AEF_RES), AEF_SHAPE[1] - 1)
        iy0 = max(int((AEF_Y0 - N) / AEF_RES), 0)
        iy1 = min(int((AEF_Y0 - S) / AEF_RES), AEF_SHAPE[0] - 1)
        cx0, cx1, cy0, cy1 = ix0 // ACH, ix1 // ACH, iy0 // ACH, iy1 // ACH
        _aef_chunks(year, cx0, cx1, cy0, cy1)
        mos = np.full(((cy1 - cy0 + 1) * ACH, (cx1 - cx0 + 1) * ACH), np.nan,
                      dtype=np.float32)
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                mos[(cy - cy0) * ACH:(cy - cy0 + 1) * ACH,
                    (cx - cx0) * ACH:(cx - cx0 + 1) * ACH] = \
                    _chunk_sim(year, cx, cy, ref, seq).astype(np.float32)
        return mos, cx0 * ACH, cy0 * ACH

    def _render_tiles(mos, ix0, iy0, boxes, rings, px=TILE_PX):
        """PNGs per tile from the similarity mosaic (nearest per output
        pixel; viridis on [SIM_LO, 1]; NaN transparent) + outline polylines
        (PIL, not closed). mos may be None: outlines only."""
        line = (40, 40, 40, 210)
        if rings:
            rb = np.array([[r[:, 0].min(), r[:, 1].min(), r[:, 0].max(), r[:, 1].max()]
                           for r in rings]) if len(rings) else np.zeros((0, 4))
        pngs = []
        ndrawn = 0
        for (W, S, E, N) in boxes:
            out = np.zeros((px, px, 4), dtype=np.uint8)
            if mos is not None:
                lons = W + (np.arange(px) + 0.5) * (E - W) / px
                lats = N - (np.arange(px) + 0.5) * (N - S) / px
                LON, LAT = np.meshgrid(lons, lats)
                jx = ((LON - AEF_X0) / AEF_RES).astype(np.int64) - ix0
                jy = ((AEF_Y0 - LAT) / AEF_RES).astype(np.int64) - iy0
                ok = (jx >= 0) & (jx < mos.shape[1]) & (jy >= 0) & (jy < mos.shape[0])
                sim = np.full((px, px), np.nan, dtype=np.float32)
                sim[ok] = mos[jy[ok], jx[ok]]
                valid = np.isfinite(sim)
                t = np.zeros((px, px), dtype=np.float32)
                t[valid] = np.clip((sim[valid] - SIM_LO) / (1.0 - SIM_LO), 0.0, 1.0)
                idx = (t * 255).astype(np.uint8)
                out[..., :3] = VIRIDIS[idx]
                out[..., 3] = np.where(valid, 220, 0)
                ndrawn += int(valid.sum())
            img = Image.fromarray(out, "RGBA")
            if rings:
                hit = np.flatnonzero((rb[:, 0] < E) & (rb[:, 2] > W) & (rb[:, 1] < N) & (rb[:, 3] > S))
                if len(hit):
                    d = ImageDraw.Draw(img)
                    sx, sy = px / (E - W), px / (N - S)
                    for hi in hit:
                        r = rings[hi]
                        pts = list(zip((r[:, 0] - W) * sx, (N - r[:, 1]) * sy))
                        if len(pts) >= 2:
                            d.line(pts, fill=line, width=1)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=False, compress_level=4)
            pngs.append(buf.getvalue())
        return pngs, ndrawn

    # ---- the reference vector from a click ----------------------------------
    def _flood(mask, sy, sx):
        """Connected component of mask containing (sy, sx): numpy dilation
        until stable (4-connected). The window is 512 px; capped."""
        comp = np.zeros_like(mask)
        comp[sy, sx] = True
        prev = 1
        for _ in range(800):
            grown = comp.copy()
            grown[1:, :] |= comp[:-1, :]
            grown[:-1, :] |= comp[1:, :]
            grown[:, 1:] |= comp[:, :-1]
            grown[:, :-1] |= comp[:, 1:]
            grown &= mask
            s = int(grown.sum())
            if s == prev:
                break
            comp, prev = grown, s
        return comp

    def _ref_from_click(lon, lat, year):
        """(unit vector float32(64), description) for a click: the FTW field's
        mean AEF vector when the click lands on P(field) >= 0.5, else the 3x3
        mean at the point. Blocking (two windowed reads)."""
        fyear = year if year in FTW_YEARS else FTW_YEARS[0]
        half = 256
        fx = int((lon + 180.0) / FTW_RES)
        fy = int((FTW_Y0 - lat) / FTW_RES)
        da = FTW_ROOT["variables"].sel(time=f"{fyear}-01-01", band="field").isel(
            y=slice(max(fy - half, 0), fy + half),
            x=slice(max(fx - half, 0), fx + half))
        prob = np.asarray(da.values)
        xs, ys = da.x.values, da.y.values
        sx = int(np.abs(xs - lon).argmin()) if len(xs) else 0
        sy = int(np.abs(ys - lat).argmin()) if len(ys) else 0
        comp = None
        if prob.size and prob[sy, sx] >= 0.5:
            comp = _flood(prob >= 0.5, sy, sx)
        # the AEF window over the component's bbox (or 3x3 at the point)
        if comp is not None and comp.any():
            rr, cc = np.nonzero(comp)
            lo_lon, hi_lon = float(xs[cc.min()]), float(xs[cc.max()])
            lo_lat, hi_lat = float(ys[rr.max()]), float(ys[rr.min()])
            plon, plat = xs[cc], ys[rr]
            desc = f"field of {len(rr)} px (FTW {fyear}) at {lat:.4f}, {lon:.4f}"
        else:
            lo_lon = hi_lon = lon
            lo_lat = hi_lat = lat
            plon = np.array([lon])
            plat = np.array([lat])
            desc = f"point at {lat:.4f}, {lon:.4f} (no FTW field here)"
        pad = 2 * AEF_RES
        ax0 = max(int((lo_lon - pad - AEF_X0) / AEF_RES), 0)
        ax1 = min(int((hi_lon + pad - AEF_X0) / AEF_RES), AEF_SHAPE[1] - 1)
        ay0 = max(int((AEF_Y0 - hi_lat - pad) / AEF_RES), 0)
        ay1 = min(int((AEF_Y0 - lo_lat + pad) / AEF_RES), AEF_SHAPE[0] - 1)
        _aef_chunks(year, ax0 // ACH, ax1 // ACH, ay0 // ACH, ay1 // ACH)
        mem = HOLD["aef_chunks"]
        w = np.full((64, ay1 - ay0 + 1, ax1 - ax0 + 1), -128, dtype=np.int8)
        for cx in range(ax0 // ACH, ax1 // ACH + 1):
            for cy in range(ay0 // ACH, ay1 // ACH + 1):
                ch = mem[(year, cx, cy)]
                gy0, gx0 = cy * ACH, cx * ACH
                ya, yb = max(ay0, gy0), min(ay1 + 1, gy0 + ACH)
                xa, xb = max(ax0, gx0), min(ax1 + 1, gx0 + ACH)
                w[:, ya - ay0:yb - ay0, xa - ax0:xb - ax0] = \
                    ch[:, ya - gy0:yb - gy0, xa - gx0:xb - gx0]
        v = _dequant(w)
        jx = np.clip(((plon - AEF_X0) / AEF_RES).astype(np.int64) - ax0, 0, v.shape[2] - 1)
        jy = np.clip(((AEF_Y0 - plat) / AEF_RES).astype(np.int64) - ay0, 0, v.shape[1] - 1)
        if comp is None or not comp.any():
            # 3x3 around the point
            j0x, j0y = int(jx[0]), int(jy[0])
            sel = v[:, max(j0y - 1, 0):j0y + 2, max(j0x - 1, 0):j0x + 2]
            vec = np.nanmean(sel.reshape(64, -1), axis=1)
        else:
            vec = np.nanmean(v[:, jy, jx], axis=1)
        if not np.all(np.isfinite(vec)):
            return None, "no AlphaEarth data here"
        nrm = float(np.linalg.norm(vec))
        if nrm == 0:
            return None, "empty vector here"
        return (vec / nrm).astype(np.float32), desc

    # ---- state + tile serve --------------------------------------------------
    _ref = HOLD.get("ref")          # (vec, seq, desc, click_year) or None
    _seq = _ref[1] if _ref else 0
    _state = (_year, _seq)
    _tiles = HOLD.setdefault("tiles", {})

    def _states_in(W, S, E, N):
        return [st for st, xmin, ymin, xmax, ymax in STATES
                if xmax > W and xmin < E and ymax > S and ymin < N]

    def _serve_batch(z, keys):
        """Blocking (worker thread): chunks + similarity + PNGs for one batch,
        status pushed on the loop thread (cdl-ftw.py's shape)."""
        t0 = time.time()
        boxes = [tile_box(z, x, y) for (_st, _z, x, y) in keys]
        W = min(b[0] for b in boxes)
        S = min(b[1] for b in boxes)
        E = max(b[2] for b in boxes)
        N = max(b[3] for b in boxes)
        ref = HOLD.get("ref")
        mos, ix0, iy0 = None, 0, 0
        note = ""
        if ref is not None and z < AEF_ZMIN:
            # z12 exists for outlines only: a z12 whole-view is hundreds of
            # 4 MB chunks (no pyramid); one zoom in and similarity appears
            note = f" · zoom in for similarity (from z{AEF_ZMIN})"
        if ref is not None and ref[1] == _seq and z >= AEF_ZMIN:
            mos, ix0, iy0 = _sim_mosaic(_year, ref[0], _seq, W, S, E, N)
        rings = None
        nt = 0
        if z >= OUTLINE_ZMIN:
            _fy = _year if _year in FTW_YEARS else FTW_YEARS[0]
            rings, nt, _nm = ftw_tile_rings(
                _states_in(W, S, E, N), _fy, W, S, E, N, min(z, FTW_TILE_ZMAX))
        pngs, ndrawn = _render_tiles(mos, ix0, iy0, boxes, rings)
        for key, png in zip(keys, pngs):
            if key[0] == _state:
                _tiles[key] = EncodedImage(data=png, media_type="image/png")
        if len(_tiles) > TILE_CACHE:
            for _k in list(_tiles)[:TILE_CACHE // 4]:
                _tiles.pop(_k, None)
        if ref is not None:
            line = (f"z{z} · {len(keys)} tiles · sim vs {ref[2]} "
                    f"(clicked in {ref[3]}) on {_year} embeddings · "
                    f"{ndrawn:,} px{note} · {int((time.time() - t0) * 1000)} ms")
        else:
            line = (f"z{z} · {len(keys)} tiles · outlines only · click a "
                    f"field · {int((time.time() - t0) * 1000)} ms")

        def _push():
            HOLD.setdefault("last_by_state", {})[_state] = line
            _say(line)

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
        """A NEW RasterLayer every rebuild (marimo closes a removed layer;
        re-adding the same object draws nothing). TMS must carry its
        boundingBox (cdl-ftw.py's rule)."""
        _tms0 = morecantile.tms.get("WebMercatorQuad")
        _m = 20037508.342789244
        _tms = _tms0.model_copy(update={"boundingBox": morecantile.models.TMSBoundingBox(
            lowerLeft=(-_m, -_m), upperRight=(_m, _m), crs=_tms0.crs)})
        return RasterLayer(
            _tile_matrix_set=_tms,
            _crs=_tms.crs,
            _fetch_tile=HOLD["fetch"],
            _render_tile=HOLD["render"],
            min_zoom=AEF_ZMIN - 1,   # z12: outlines appear one step before sim
            max_zoom=TILE_ZMAX,
            extent=EXTENT,
            _tile_size=TILE_PX,
            debounce_time=30,
            opacity=1.0,
            pickable=False,
        )

    def _legend_html():
        ref = HOLD.get("ref")
        stops = ", ".join(f"rgb({r},{g},{b}) {i / 15 * 100:.0f}%"
                          for i, (r, g, b) in enumerate(
                              VIRIDIS[np.linspace(0, 255, 16).astype(int)]))
        bar = (f'<span style="display:inline-block;width:9rem;height:10px;'
               f'border-radius:3px;background:linear-gradient(90deg,{stops})"></span>')
        if ref is None:
            return ('<span style="opacity:.7">click a field on the map '
                    '(similarity from camera ~z12)</span>')
        return (f'<span style="opacity:.8">cosine {SIM_LO:.1f}</span>{bar}'
                f'<span style="opacity:.8">1.0</span>'
                f'<span style="opacity:.7">· {ref[2]}</span>')

    def _rebuild():
        HOLD["batch"] = None
        HOLD["raster"] = _make_raster()
        HOLD["layer_state"] = _state
        deck.layers = []
        deck.layers = [HOLD["raster"]]
        try:
            hud.widget.legend = _legend_html()
        except Exception:
            pass
        _last = HOLD.get("last_by_state", {}).get(_state)
        if _last is not None:
            _say(_last + " · from cache")
        else:
            ref = HOLD.get("ref")
            _say(f"{_year} embeddings · "
                 + (f"sim vs {ref[2]} · loading …" if ref else
                    "click a field (similarity from camera ~z12) · loading …"))

    async def _run_batch(b):
        # the batch future ALWAYS resolves; everything the batch needs is IN b
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

    # ---- the acts, all IN THIS RUN ------------------------------------------
    if _act == "click":
        try:
            _vs = _vsd(HOLD.get("vs")) or dict(HOME)
            _lon, _lat = unproject(_vs, float(_c["px"]), float(_c["py"]),
                                   float(_c["w"]), float(_c["h"]))
            _say(f"reading the field at {_lat:.4f}, {_lon:.4f} …")
            _vec, _desc = _ref_from_click(_lon, _lat, _year)
            if _vec is None:
                _say(_desc)
            else:
                _seq = HOLD.get("ref_seq", 0) + 1
                HOLD["ref_seq"] = _seq
                HOLD["ref"] = (_vec, _seq, _desc, _year)
                _state = (_year, _seq)
                HOLD["layer_state"] = None
        except Exception as _e:
            _say(f"click error: {type(_e).__name__}: {_e}")

    if _act == "clear":
        HOLD["ref"] = None
        _seq = 0
        _state = (_year, 0)
        HOLD["layer_state"] = None

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
