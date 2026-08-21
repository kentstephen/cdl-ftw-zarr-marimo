"""Two patches to lonboard's shipped JS: turn OFF deck's default lighting on
the raster tile mesh, and raise the raster tile request timeout (10 s -> 120 s).

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
