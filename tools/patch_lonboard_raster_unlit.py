"""Three patches to lonboard's shipped JS: turn OFF deck's default lighting on
the raster tile mesh, raise the raster tile request timeout (10 s -> 120 s), and
give each raster layer its own deck id under marimo (model_id is undefined there).

lonboard 0.16 renders every RasterLayer tile through a mesh sub-layer whose
fragment shader ends in `lighting_getLightColor(...)`: deck's default material
(ambient 0.35) plus the two default directional lights on a flat mesh come to
~0.69, so every tile is drawn ~0.69x darker on every channel and the layer's
`opacity` never reaches it (measured 2026-08-20: 255 -> 176, 100 -> 71). No
Python-side prop fixes it (lonboard's Map has no `effects` trait), so this
script replaces that one line in the shipped JS bundle with the unlit colour.

Re-run after ANY install (`uv sync`, a lonboard upgrade, `--sandbox`), then
RESTART THE KERNEL: anywidget reads this file into the Map widget's `_esm`
when the Map is created, so a browser reload changes nothing. Idempotent.
uv installs by HARDLINK from its cache: the file is rewritten through a temp
file + os.replace so the cached wheel is left alone (the link is broken, not
edited in place).

    uv run python tools/patch_lonboard_raster_unlit.py
"""
from __future__ import annotations

import os
import re
import sys
import tempfile
from pathlib import Path

import lonboard

OLD = ("vec3 lightColor = lighting_getLightColor(color.rgb, cameraPosition, "
       "position_commonspace.xyz, normal);")
NEW = "vec3 lightColor = color.rgb; /* unlit: patch_lonboard_raster_unlit */"

# Second patch (2026-08-21): the raster layer's getTileData gives the kernel
# TEN SECONDS per tile request (`timeout:1e4`); past that the JS drops the
# tile as an error and deck never asks for it again, so any batch over 10 s
# (a fly-to into a cold region: mask chunks, CDL chunks, outlines all new)
# left the map blank until a param change rebuilt the layer. Raised to 120 s,
# the Python side's own wait. Matched by the request's shape, not the
# minified names (the geocoder control has its own 1e4, left alone).
TIMEOUT_RE = re.compile(
    r"(tile:\{index:\{x:\w+,y:\w+,z:\w+\}\}\},\w+,\{signal:\w+,timeout:)1e4\b")
TIMEOUT_NEW = r"\g<1>12e4"

# Third patch (2026-08-21, night): the raster layer's deck id is
# `${this.model.model_id}` and under marimo model_id is undefined, so EVERY
# RasterLayer is deck layer "undefined": a rebuilt layer (year, checkbox, search)
# reads to deck as an update of the same TileLayer, which keeps its loaded
# tiles and only fetches the ones it lacks (the old state stayed on screen in
# bands; the remove-then-add in one cell run does not reach deck as two steps).
# With model_id missing, each JS layer instance gets its own random id, so a
# rebuild is a new TileLayer and the old one is finalised.
# And a per-instance `updateTriggers.getTileData`: deck's TileLayer reloads
# EVERY tile when that trigger changes (updateState -> tileset.reloadAll()), so
# a rebuilt layer refetches through its own getTileData even if deck matches
# it to the old one by id (a click on fields at the home view changed nothing
# until a zoom, 2026-08-21). NOT via `data`: a string data is a URL that
# deck's base layer fetches.
ID_OLD = "layerProps(){return{id:`${this.model.model_id}`,data:null,"
ID_MID = ("layerProps(){return{id:`${this.model.model_id??(this._lbid??="
          "Math.random().toString(36).slice(2))}`,data:null,")
ID_BAD = ("layerProps(){return{id:`${this.model.model_id??(this._lbid??="
          "Math.random().toString(36).slice(2))}`,"
          "data:(this._lbid??=Math.random().toString(36).slice(2)),")
ID_NEW = ("layerProps(){return{id:`${this.model.model_id??(this._lbid??="
          "Math.random().toString(36).slice(2))}`,data:null,"
          "updateTriggers:{getTileData:(this._lbid??=Math.random().toString(36).slice(2))},")


def main() -> int:
    js = Path(lonboard.__file__).parent / "static" / "index.js"
    src = js.read_text()
    out = src
    if NEW in out:
        print("unlit shader: already patched")
    else:
        n = out.count(OLD)
        if n == 0:
            print(f"lit shader line not found: {js}")
            print("lonboard", lonboard.__version__, "(the bundle changed; update OLD)")
            return 1
        # lonboard 0.16 carries the line TWICE: deck's SimpleMeshLayer shader
        # and the raster tile mesh's copy of it. Both go unlit (nothing here
        # wants a lit mesh; the archived SurfaceLayer patch wanted the same).
        out = out.replace(OLD, NEW)
        print(f"unlit shader: replacing {n} occurrence(s)")
    out, m = TIMEOUT_RE.subn(TIMEOUT_NEW, out)
    if m:
        print(f"tile timeout: 10 s -> 120 s ({m} occurrence)")
    elif "timeout:12e4" in out:
        print("tile timeout: already patched")
    else:
        print("tile timeout: getTileData request not found (bundle changed; update TIMEOUT_RE)")
        return 1
    if ID_NEW in out:
        print("raster layer id + updateTriggers: already patched")
    elif ID_BAD in out:
        out = out.replace(ID_BAD, ID_NEW)
        print("raster layer: data token -> updateTriggers.getTileData (a string data is a URL to deck)")
    elif ID_MID in out:
        out = out.replace(ID_MID, ID_NEW)
        print("raster layer updateTriggers: per instance (id patch was already in)")
    elif ID_OLD in out:
        out = out.replace(ID_OLD, ID_NEW)
        print("raster layer id + updateTriggers: per instance")
    else:
        print("raster layer id: layerProps not found (bundle changed; update ID_OLD)")
        return 1
    if out == src:
        print(f"already patched: {js}")
        return 0
    fd, tmp = tempfile.mkstemp(dir=js.parent, prefix="index.js.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(out)
    os.replace(tmp, js)   # new inode: the uv cache's hardlinked copy is untouched
    print(f"patched: {js}  (lonboard {lonboard.__version__}); restart the kernel")
    return 0


if __name__ == "__main__":
    sys.exit(main())
