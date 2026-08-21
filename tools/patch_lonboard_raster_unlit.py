"""Turn OFF deck's default lighting on lonboard's raster tile mesh.

lonboard 0.16 renders every RasterLayer tile through a mesh sub-layer whose
fragment shader ends in `lighting_getLightColor(...)`: deck's default material
(ambient 0.35) plus the two default directional lights on a flat mesh come to
~0.69, so every tile is drawn ~0.69x darker on every channel and the layer's
`opacity` never reaches it (measured 2026-08-20: 255 -> 176, 100 -> 71). No
Python-side prop fixes it (lonboard's Map has no `effects` trait), so this
script replaces that one line in the shipped JS bundle with the unlit colour.

Re-run after ANY install (`uv sync`, a lonboard upgrade, `--sandbox`), then
hard-reload the browser (the widget JS is cached client side). Idempotent.
uv installs by HARDLINK from its cache: the file is rewritten through a temp
file + os.replace so the cached wheel is left alone (the link is broken, not
edited in place).

    uv run python tools/patch_lonboard_raster_unlit.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import lonboard

OLD = ("vec3 lightColor = lighting_getLightColor(color.rgb, cameraPosition, "
       "position_commonspace.xyz, normal);")
NEW = "vec3 lightColor = color.rgb; /* unlit: patch_lonboard_raster_unlit */"


def main() -> int:
    js = Path(lonboard.__file__).parent / "static" / "index.js"
    src = js.read_text()
    if NEW in src:
        print(f"already patched: {js}")
        return 0
    n = src.count(OLD)
    if n != 1:
        print(f"expected the lit shader line exactly once, found {n}: {js}")
        print("lonboard", lonboard.__version__, "(the bundle changed; update OLD)")
        return 1
    out = src.replace(OLD, NEW)
    fd, tmp = tempfile.mkstemp(dir=js.parent, prefix="index.js.", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(out)
    os.replace(tmp, js)   # new inode: the uv cache's hardlinked copy is untouched
    print(f"patched: {js}  (lonboard {lonboard.__version__}); hard-reload the browser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
