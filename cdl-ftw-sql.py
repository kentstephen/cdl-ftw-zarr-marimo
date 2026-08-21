# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo",
#     "xarray-sql[duckdb]==0.4.0rc1",
#     "duckdb>=1.5.5",
#     "xarray",
#     "zarr>=3",
#     "icechunk",
#     "obstore",
#     "pyarrow>=25.0.0",
#     "numpy",
# ]
# ///
"""The CDL x Fields of the World joins, as plain SQL on a box.

Moved out of cdl-ftw.py (the map notebook) on 2026-08-21: the map is xarray +
numpy tiles and never touched these; here they are as statements on one DuckDB
connection, each leaving a table for the next.

  CDL crop_type (icechunk Zarr v3, the 10 m group, 2024-2025)  -> xql.register -> cdl10_<k>
  FTW P(field) (plain Zarr v3, the 4x level, 40 m)              -> xql.register -> ftw_4
  FTW field polygons (fiboa GeoParquet, one file per US state)  -> read_parquet over httpfs
                                                                   (+ cache_httpfs on disk)

Type a box (lon/lat W, S, E, N; the default is the map notebook's opening view,
the Delta west of Stockton), pick FTW's year, press the button.

  uv run marimo edit cdl-ftw-sql.py
"""

import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full", sql_output="native")


@app.cell
def _():
    import math
    import os
    import tempfile

    import duckdb
    import icechunk
    import xarray as xr
    import zarr
    import xarray_sql as xql
    from obstore.store import S3Store

    import marimo as mo

    return S3Store, duckdb, icechunk, math, mo, os, tempfile, xql, xr, zarr


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # CDL x Fields of the World: the joins as SQL

    | source | how it is read | what it gives here |
    |---|---|---|
    | CDL `crop_type(year, y, x)`, the 10 m group (2024-2025) + majority pyramid, EPSG:5070 | icechunk Zarr v3 -> `xql.register` (xarray-sql DuckDB backend): tables `cdl10_1` .. `cdl10_32` | the crop of every pixel centre |
    | FTW softmax P(field), the 4x level (40 m), EPSG:4326 | plain Zarr v3 -> `xql.register`: table `ftw_4` | the field / not-field side of the 2x2 |
    | FTW field polygons, one fiboa GeoParquet per US state, both years in the file | `read_parquet(...)` over httpfs (+ `cache_httpfs` on disk), the `bbox` struct prunes row groups | the fields: `ST_Contains` of the pixel centres |

    FTW's `confidence` column is NULL for the whole US, so nothing here claims a
    per-field confidence. The state files' extents are embedded (the STAC items'
    bboxes are wrong: US_CA reports Montana).
    """)
    return


@app.cell
def _():
    # ---- constants ----------------------------------------------------------
    BUCKET = "chill"
    PREFIX = "usda-cropland-data-layer/v0.1.0.icechunk"
    ENDPOINT = "https://data.source.coop"

    FTW_BUCKET = "us-west-2.opendata.source.coop"
    FTW_VEC = (
        "tge-labs/ftw-global-data/predictions/vectors/alpha/"
        "results-by-admin-conf/admin:country_code=US/"
    )
    FTW_ZARR = "tge-labs/ftw-global-data/predictions/zarr/alpha/global.zarr/"
    FTW_RES = 8.98311982e-05      # degrees per 10 m pixel at the root
    FTW_Y0 = 83.748345            # top edge of the grid
    FTW_YEARS = (2024, 2025)

    LEVELS10 = [1, 2, 4, 8, 16, 32]   # the 10 m group's ladder, as far as the box picker goes
    HTTPFS_CACHE_DIR = "x-sql-marimo/duckdb-httpfs-cache"   # under the OS tmp dir:
    # DuckDB's cache_httpfs community extension writes every byte range it
    # fetches over httpfs (the FTW parquet row groups, ~13 MB each) to disk, so
    # a place touched once is read locally afterwards, on any connection and
    # across restarts (shared with the map notebook). None disables.

    # the map notebook's opening view, as its padded box (bbox4326 of HOME)
    HOME = {"longitude": -121.45, "latitude": 37.95, "zoom": 12.0}
    VIEW_W, VIEW_H, MARGIN = 1400, 700, 0.35
    return (
        BUCKET,
        ENDPOINT,
        FTW_BUCKET,
        FTW_RES,
        FTW_VEC,
        FTW_Y0,
        FTW_YEARS,
        FTW_ZARR,
        HOME,
        HTTPFS_CACHE_DIR,
        LEVELS10,
        MARGIN,
        PREFIX,
        VIEW_H,
        VIEW_W,
    )


@app.cell
def _(
    BUCKET,
    ENDPOINT,
    FTW_BUCKET,
    FTW_ZARR,
    HTTPFS_CACHE_DIR,
    LEVELS10,
    PREFIX,
    S3Store,
    duckdb,
    icechunk,
    os,
    tempfile,
    xql,
    xr,
    zarr,
):
    # ---- open both stores, register the levels as DuckDB tables -------------
    storage = icechunk.s3_storage(
        bucket=BUCKET,
        prefix=PREFIX,
        endpoint_url=ENDPOINT,
        region="us-east-1",
        anonymous=True,
        force_path_style=True,
    )
    _repo = icechunk.Repository.open(storage)
    _session = _repo.readonly_session("main")

    con = duckdb.connect()
    con.sql(
        "INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;"
        " SET s3_region='us-west-2'; SET s3_url_style='path';"
    )
    if HTTPFS_CACHE_DIR:
        _d = os.path.join(tempfile.gettempdir(), HTTPFS_CACHE_DIR)
        os.makedirs(_d, exist_ok=True)
        con.sql(
            "INSTALL cache_httpfs FROM community; LOAD cache_httpfs;"
            f" SET cache_httpfs_cache_directory='{_d}';"
        )

    # the 10 m group (2024-2025, FTW's own resolution): native + majority pyramid
    DS10 = {}
    for _k in LEVELS10:
        _grp = "10m" if _k == 1 else f"10m/{_k}x"
        DS10[_k] = xr.open_zarr(_session.store, group=_grp, chunks=None)
        # 2048^2 blocks so the x/y predicates prune fragments
        xql.register(con, f"cdl10_{_k}", DS10[_k],
                     chunks={"year": 1, "y": 2048, "x": 2048})

    # FTW P(field), the 4x level (40 m). Blocks = the level's INNER chunk (512),
    # never the shard (4096): a shard-sized block expands whole and the window
    # predicate cannot prune inside it (19.5 s vs 1.2 s for the same 20 km box).
    _ftw_store = zarr.storage.ObjectStore(
        S3Store(bucket=FTW_BUCKET, region="us-west-2", skip_signature=True,
                prefix=FTW_ZARR),
        read_only=True,
    )
    FTW4 = xr.open_zarr(_ftw_store, group="4x", chunks=None, consolidated=False)
    xql.register(con, "ftw_4", FTW4, chunks={"time": 1, "band": 3, "y": 512, "x": 512})

    # ---- classes table from the CDL store's own attrs ----------------------
    _at = DS10[1]["crop_type"].attrs
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

    con.sql("CREATE TABLE classes(code UTINYINT, name VARCHAR, hex VARCHAR, noncrop BOOLEAN)")
    con.executemany(
        "INSERT INTO classes VALUES (?,?,?,?)",
        [(int(c), _names[c], _colors[c], _noncrop(_names[c])) for c in sorted(_names, key=int)],
    )

    # ---- the FTW state partitions: each file's extent from its OWN row-group
    # stats (parquet_metadata over all 60 files, 3.7 s on 2026-08-20; embedded
    # so open costs nothing). Non-CONUS rows (AK, HI, the Canadian and Mexican
    # border fragments) are kept; CDL has no pixels there.
    _STATES = [
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
    con.sql(
        "CREATE TABLE ftw_states(st VARCHAR, xmin DOUBLE, ymin DOUBLE,"
        " xmax DOUBLE, ymax DOUBLE)"
    )
    con.executemany("INSERT INTO ftw_states VALUES (?,?,?,?,?)", _STATES)
    return (con,)


@app.cell
def _(FTW_BUCKET, FTW_VEC, HOME, MARGIN, VIEW_H, VIEW_W, math):
    # ---- helpers: the box in Albers, the state files that meet it -----------
    def bbox4326(vs):
        """The map notebook's padded view box for a camera (its default box)."""
        span = 360.0 / (512 * 2 ** vs["zoom"])
        dlon = VIEW_W * span * (1 + MARGIN) / 2
        dlat = VIEW_H * span * math.cos(math.radians(vs["latitude"])) * (1 + MARGIN) / 2
        return (vs["longitude"] - dlon, vs["latitude"] - dlat,
                vs["longitude"] + dlon, vs["latitude"] + dlat)

    def to5070(c, lon0, lat0, lon1, lat1):
        # densified box boundary, clamped to the array's Albers bbox (an
        # EPSG:5070 parallel bows; corner-only min clips the Gulf coast)
        _N = 8
        pts = []
        for _i in range(_N + 1):
            _t = _i / _N
            _lon = lon0 + (lon1 - lon0) * _t
            _lat = lat0 + (lat1 - lat0) * _t
            pts += [(_lon, lat0), (_lon, lat1), (lon0, _lat), (lon1, _lat)]
        vals = ", ".join(f"({a}, {b})" for a, b in pts)
        rows = c.sql(
            f"""SELECT ST_X(p), ST_Y(p) FROM (
                  SELECT ST_Transform(ST_Point(lon, lat), 'EPSG:4326', 'EPSG:5070',
                                      always_xy := true) AS p
                  FROM (VALUES {vals}) AS t(lon, lat))"""
        ).fetchall()
        xs = [r[0] for r in rows]
        ys = [r[1] for r in rows]
        _X0, _Y0, _X1, _Y1 = -2417835.0, 158265.0, 2387295.0, 3321225.0
        return (max(min(xs), _X0), max(min(ys), _Y0),
                min(max(xs), _X1), min(max(ys), _Y1))

    def ftw_files(c, W, S, E, N):
        """The state parquet files whose extent meets the box."""
        sts = [r[0] for r in c.sql(
            f"""SELECT st FROM ftw_states
                WHERE xmax > {W} AND xmin < {E} AND ymax > {S} AND ymin < {N}
                ORDER BY st"""
        ).fetchall()]
        return [f"s3://{FTW_BUCKET}/{FTW_VEC}US_{st}.parquet" for st in sts]

    HOME_BOX = bbox4326(HOME)
    return HOME_BOX, ftw_files, to5070


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## The joins, as plain SQL on a box

    Statements on `con`, each leaving a table for the next. Set the box (lon/lat
    W, S, E, N; the map notebook's opening view is the default), FTW's year, and
    press the button.
    """)
    return


@app.cell
def _(FTW_YEARS, HOME_BOX, mo):
    box = mo.ui.text(
        value=", ".join(f"{v:.4f}" for v in HOME_BOX),
        label="box W, S, E, N",
        debounce=True,
        full_width=True,
    )
    year = mo.ui.dropdown(
        options=[str(y) for y in FTW_YEARS], value=str(FTW_YEARS[-1]), label="FTW year"
    )
    go = mo.ui.run_button(label="run the SQL on this box")
    mo.hstack([box, year, go], widths=[3, 1, 1], align="end")
    return box, go, year


@app.cell
def _(HOME_BOX, box, con, ftw_files, go, mo, to5070, year):
    # the analysis box, at click time; `go` is only the trigger
    _ = go.value
    try:
        W, S, E, N = (float(v) for v in box.value.split(","))
    except ValueError:
        W, S, E, N = HOME_BOX
    x0, y0, x1, y1 = to5070(con, W, S, E, N)
    # the cells below are same-year joins on CDL's 10 m group: finest level
    # with <= ~1.5M pixel centres in the box
    FTW_YEAR = int(year.value)
    T, B = "cdl10_", 10
    K = next((k for k in (1, 2, 4, 8, 16) if (x1 - x0) * (y1 - y0) / (B * k) ** 2 <= 1.5e6), 32)
    PX_KM2 = (B * K / 1000) ** 2
    FILES = ftw_files(con, W, S, E, N)
    FILES_SQL = ", ".join(f"'{f}'" for f in FILES)
    mo.md(
        f"box **{W:.3f}, {S:.3f} → {E:.3f}, {N:.3f}** · CDL {FTW_YEAR} at "
        f"{B * K} m (`{T}{K}`, the 10 m group) · FTW {FTW_YEAR} from "
        f"{', '.join(f.rsplit('/', 1)[-1] for f in FILES) or 'no state file meets the box'}"
    )
    return E, FILES_SQL, FTW_YEAR, K, N, PX_KM2, S, T, W, x0, x1, y0, y1


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 1. The fields: `read_parquet` on the state file(s), pruned by the `bbox` struct
    """)
    return


@app.cell
def _(E, FILES_SQL, FTW_YEAR, N, S, W, con, mo):
    fields_view = mo.sql(
        f"""
        CREATE OR REPLACE TABLE fields_view AS
        SELECT id, "metrics:area" AS area_m2, geometry::GEOMETRY AS geometry
        FROM read_parquet([{FILES_SQL}])
        WHERE bbox.xmin > {W} AND bbox.xmax < {E}
          AND bbox.ymin > {S} AND bbox.ymax < {N}
          AND date_part('year', "determination:datetime" AT TIME ZONE 'UTC') = {FTW_YEAR}
        """,
        engine=con
    )
    return


@app.cell
def _(con, mo):
    fields_summary = mo.sql(
        """
        SELECT count(*) AS fields,
               round(sum(area_m2) / 4046.8564, 0) AS acres,
               round(quantile_cont(area_m2, 0.5) / 4046.8564, 1) AS median_acres,
               round(max(area_m2) / 4046.8564, 0) AS largest_acres
        FROM fields_view
        """,
        engine=con,
    )
    fields_summary
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 2. Pixel -> field, once: `ST_Contains` of the CDL pixel centres into the polygons
    """)
    return


@app.cell
def _(FTW_YEAR, K, T, con, mo, x0, x1, y0, y1):
    px2field = mo.sql(
        f"""
        CREATE OR REPLACE TABLE px2field AS
        WITH p AS (
            SELECT DISTINCT y, x,
                   ST_Transform(ST_Point(x, y), 'EPSG:5070', 'EPSG:4326',
                                always_xy := true) AS pt
            FROM {T}{K}
            WHERE year = {FTW_YEAR}
              AND x BETWEEN {x0} AND {x1} AND y BETWEEN {y0} AND {y1}
        )
        SELECT f.id, p.y, p.x
        FROM fields_view f JOIN p ON ST_Contains(f.geometry, p.pt)
        """,
        engine=con
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3. The crop of every field (same year as the polygons): majority class and purity
    """)
    return


@app.cell
def _(FTW_YEAR, K, T, con, mo, x0, x1, y0, y1):
    field_crop = mo.sql(
        f"""
        CREATE OR REPLACE TABLE field_crop AS
        WITH j AS (
            SELECT l.id, t.crop_type, count(*) AS n
            FROM {T}{K} t JOIN px2field l USING (y, x)
            WHERE t.year = {FTW_YEAR}
              AND t.x BETWEEN {x0} AND {x1} AND t.y BETWEEN {y0} AND {y1}
              AND t.crop_type NOT IN (0, 81)
            GROUP BY 1, 2
        ),
        m AS (
            SELECT id, crop_type, n,
                   sum(n) OVER (PARTITION BY id) AS tot,
                   row_number() OVER (PARTITION BY id ORDER BY n DESC, crop_type) AS rn
            FROM j
        )
        SELECT id, crop_type, n AS px, tot AS px_total, n / tot AS purity
        FROM m WHERE rn = 1
        """,
        engine=con
    )
    return


@app.cell
def _(con, mo):
    crop_by_field = mo.sql(
        """
        SELECT c.name AS crop, count(*) AS fields,
               round(sum(f.area_m2) / 4046.8564, 0) AS acres,
               round(quantile_cont(f.area_m2, 0.5) / 4046.8564, 1) AS median_field_acres,
               round(avg(fc.purity), 2) AS mean_purity
        FROM field_crop fc JOIN fields_view f USING (id) JOIN classes c ON c.code = fc.crop_type
        GROUP BY 1 ORDER BY fields DESC LIMIT 15
        """,
        engine=con,
    )
    crop_by_field
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4. Purity: the least pure fields with their top two crops (two fields FTW merged, or CDL noise inside one)
    """)
    return


@app.cell
def _(FTW_YEAR, K, T, con, mo, x0, x1, y0, y1):
    mixed_fields = mo.sql(
        f"""
        WITH j AS (
            SELECT l.id, t.crop_type, count(*) AS n
            FROM {T}{K} t JOIN px2field l USING (y, x)
            WHERE t.year = {FTW_YEAR}
              AND t.x BETWEEN {x0} AND {x1} AND t.y BETWEEN {y0} AND {y1}
              AND t.crop_type NOT IN (0, 81)
            GROUP BY 1, 2
        ),
        r AS (
            SELECT id, crop_type, n,
                   sum(n) OVER (PARTITION BY id) AS tot,
                   row_number() OVER (PARTITION BY id ORDER BY n DESC, crop_type) AS rn
            FROM j
        )
        SELECT a.id,
               round(f.area_m2 / 4046.8564, 1) AS acres,
               ca.name AS top_crop, round(a.n / a.tot, 2) AS share,
               cb.name AS second_crop, round(b.n / b.tot, 2) AS share_2
        FROM r a JOIN r b ON a.id = b.id AND b.rn = 2
        JOIN fields_view f ON f.id = a.id
        JOIN classes ca ON ca.code = a.crop_type
        JOIN classes cb ON cb.code = b.crop_type
        WHERE a.rn = 1 AND a.tot >= 40
        ORDER BY a.n / a.tot ASC LIMIT 20
        """,
        engine=con,
    )
    mixed_fields
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5. Disagreement: CDL crop / not-crop against FTW P(field) >= 0.5 from `ftw_4`

    Each CDL pixel centre is binned into its 40 m FTW cell by index arithmetic.
    Then the crops FTW most often misses, and the non-crop classes FTW calls fields.
    """)
    return


@app.cell
def _(E, FTW_RES, FTW_Y0, FTW_YEAR, K, N, S, T, W, con, mo, x0, x1, y0, y1):
    _res = FTW_RES * 4
    agreement = mo.sql(
        f"""
        CREATE OR REPLACE TABLE agreement AS
        WITH fp AS (
            SELECT floor((x + 180) / {_res})::BIGINT AS ix,
                   floor(({FTW_Y0} - y) / {_res})::BIGINT AS iy,
                   variables >= 0.5 AS is_field
            FROM ftw_4
            WHERE time = TIMESTAMP '{FTW_YEAR}-01-01' AND band = 'field'
              AND x BETWEEN {W} AND {E} AND y BETWEEN {S} AND {N}
        ),
        cp AS (
            SELECT t.crop_type, NOT c.noncrop AS is_crop, c.name,
                   ST_Transform(ST_Point(t.x, t.y), 'EPSG:5070', 'EPSG:4326',
                                always_xy := true) AS pt
            FROM {T}{K} t JOIN classes c ON c.code = t.crop_type
            WHERE t.year = {FTW_YEAR}
              AND t.x BETWEEN {x0} AND {x1} AND t.y BETWEEN {y0} AND {y1}
              AND t.crop_type NOT IN (0, 81)
        )
        SELECT cp.crop_type, cp.name, cp.is_crop, fp.is_field, count(*) AS px
        FROM cp JOIN fp
          ON floor((ST_X(cp.pt) + 180) / {_res})::BIGINT = fp.ix
         AND floor(({FTW_Y0} - ST_Y(cp.pt)) / {_res})::BIGINT = fp.iy
        GROUP BY 1, 2, 3, 4
        """,
        engine=con,
    )
    return


@app.cell
def _(PX_KM2, con, mo):
    two_by_two = mo.sql(
        f"""
        SELECT CASE WHEN is_crop THEN 'CDL crop' ELSE 'CDL not crop' END AS cdl,
               CASE WHEN is_field THEN 'FTW field' ELSE 'FTW not field' END AS ftw,
               round(sum(px) * {PX_KM2} * 247.105 / 1e3, 1) AS k_acres,
               round(100.0 * sum(px) / sum(sum(px)) OVER (), 1) AS pct
        FROM agreement GROUP BY 1, 2 ORDER BY 1, 2
        """,
        engine=con,
    )
    two_by_two
    return


@app.cell
def _(PX_KM2, con, mo):
    ftw_misses = mo.sql(
        f"""
        SELECT name AS cdl_crop,
               round(sum(px) * {PX_KM2} * 247.105 / 1e3, 1) AS k_acres,
               round(100.0 * sum(CASE WHEN is_field THEN px ELSE 0 END) / sum(px), 1)
                   AS pct_ftw_field
        FROM agreement WHERE is_crop
        GROUP BY 1 HAVING sum(px) >= 200
        ORDER BY pct_ftw_field ASC, k_acres DESC LIMIT 15
        """,
        engine=con,
    )
    ftw_false_fields = mo.sql(
        f"""
        SELECT name AS cdl_noncrop,
               round(sum(px) * {PX_KM2} * 247.105 / 1e3, 1) AS k_acres,
               round(100.0 * sum(CASE WHEN is_field THEN px ELSE 0 END) / sum(px), 1)
                   AS pct_ftw_field
        FROM agreement WHERE NOT is_crop
        GROUP BY 1 HAVING sum(px) >= 200
        ORDER BY pct_ftw_field DESC, k_acres DESC LIMIT 15
        """,
        engine=con,
    )
    mo.hstack([ftw_misses, ftw_false_fields], widths="equal", gap=2)
    return


if __name__ == "__main__":
    app.run()
