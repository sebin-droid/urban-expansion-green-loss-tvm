"""
Step 5a - Big spatial data processing with Apache Sedona on Spark (syllabus: MapReduce, Spark vs Sedona,
spatial partitioning, spatial indexing, distributed spatial joins).

Every land-cover pixel becomes a point record, so the raster analysis of Step 4 turns into the classic
big-spatial-data workload: millions of points joined to polygons, lines and a hexagon grid.

5.0 Export   Pixels and vectors are written to Parquet (geometry as WKB): 4.0 M Sentinel-2 points
             (10 m, 2025), 0.45 M Landsat points (30 m, one row per pixel, land cover for 2000/2010/
             2017/2025 + the 2000->2025 change code), 101 wards, 203 k OSM buildings, 29 k roads.
5.1 Benchmark  Point-in-polygon join (pixels -> wards, then count per ward) at growing input sizes:
             - Spark without spatial optimisation (Sedona join planning switched off, so Spark falls
               back to a nested-loop join: every point is tested against every ward)
             - Sedona broadcast index join (wards broadcast to every task, R-tree built on them)
             - Sedona partitioned join, KDB-tree partitioning + R-tree index
             - Sedona partitioned join, Quadtree partitioning + R-tree index
             - GeoPandas sjoin (single machine, STRtree) as the non-distributed reference
             The physical plan of every run is saved, which shows the MapReduce pattern behind the
             aggregation (partial count = map/combine, Exchange = shuffle, final count = reduce).
5.2 Partitioning  The 4 M points are spatially partitioned with a KDB-tree and a Quadtree and compared
             with Spark's default hash partitioning: points per partition (load balance) and how compact
             each partition is in space (locality, which is what makes the join prune work).
5.3 Analytics on the full data (all with Sedona SQL)
             a. Ward land-cover shares per epoch from the pixel join; checked against Step 4 (rasterio).
             b. Buildings per ward (OSM): count, footprint area, coverage %, density.
             c. Road length per ward (ST_Intersection with the ward polygons), main-road density.
             d. Distance from every 2000 green pixel to the nearest main road (kNN join), and the share
                that turned into built-up by 2025 in each distance band: does expansion follow roads?
             e. H3 hexagon grid (resolution 9, ~0.1 km2): green 2000/2025, built-up, green->built-up
                per hexagon, the hotspot layer for the dashboard.

Windows notes: Spark needs HADOOP_HOME to exist (an empty tools/hadoop folder is enough here because
nothing is written through Hadoop). The Sedona jars are put on the driver classpath instead of
spark.jars.packages, since copying packages needs winutils.exe. Python UDFs / RDD lambdas are not used:
all work runs inside the JVM (Spark SQL + Sedona SQL), results come back through Arrow.

Inputs:  data/processed/landcover_consistent/lc_landsat_<year>.tif, change_2000_2025.tif (Step 4)
         data/processed/landcover/lc_sentinel2_2025.tif (Step 3), base.gpkg (Step 1), ward_change.csv (Step 4)
Outputs: data/processed/bigdata/*.parquet                      (Spark inputs)
         data/processed/step5_benchmark.csv, step5_partitions.csv
         data/processed/step5_ward_metrics.csv                 (buildings, roads, Sedona land-cover shares)
         data/processed/step5_ward_check.csv                   (Sedona vs Step 4 shares)
         data/processed/step5_road_distance.csv
         data/processed/sedona.gpkg (layers h3_change, partitions)
         outputs/logs/step5_plans.txt                          (physical plans)
         outputs/maps/step5_benchmark.png, step5_partitions.png, step5_road_distance.png, step5_h3.png

Run:  .venv/Scripts/python scripts/step5_sedona.py [export|bench|partitions|analytics|maps]  (default: all)
"""
import os
import sys
import time
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
BIG = OUT / "bigdata"
GPKG = OUT / "base.gpkg"
SEDONA_GPKG = OUT / "sedona.gpkg"
MAPS = ROOT / "outputs" / "maps"
LOGS = ROOT / "outputs" / "logs"
UTM = "EPSG:32643"

# same codes as Steps 3-4
YEARS = [2000, 2010, 2017, 2025]
BUILT, TREE, GREEN = 1, 2, (2, 3)
GREEN_TO_BUILT = 3                       # change_2000_2025.tif code
MAIN_ROADS = ("highway", "major")        # motorway/trunk/primary/secondary (Step 1 road_class)

IVY = Path.home() / ".ivy2" / "jars"
SEDONA_JARS = ["org.apache.sedona_sedona-spark-shaded-3.5_2.12-1.8.0.jar",
               "org.datasyslab_geotools-wrapper-1.8.0-33.1.jar"]
SEDONA_PACKAGES = "org.apache.sedona:sedona-spark-shaded-3.5_2.12:1.8.0,org.datasyslab:geotools-wrapper:1.8.0-33.1"

BENCH_SIZES = [100_000, 500_000, 1_000_000, None]     # None = all Sentinel-2 points
REPEATS = 3                                           # best of REPEATS (first run also warms up)
N_PARTITIONS = 24
DIST_BANDS = [0, 100, 250, 500, 1000, 2000, np.inf]
H3_RES = 9
SEED = 42

BASE_CONF = {                               # reset before every benchmark method
    "sedona.join.optimizationmode": "all",
    "sedona.join.gridtype": "kdbtree",
    "sedona.join.numpartition": str(N_PARTITIONS),
    "sedona.global.index": "true",
    "sedona.global.indextype": "rtree",
    "sedona.join.autoBroadcastJoinThreshold": "-1",
    "spark.sql.autoBroadcastJoinThreshold": str(10 * 1024 * 1024),
}
METHODS = {   # name: (label, extra conf, join hint)
    "spark_nested_loop": ("Spark, no spatial optimisation (nested loop)",
                          {"sedona.join.optimizationmode": "none"}, ""),
    "sedona_broadcast": ("Sedona broadcast index join (R-tree)", {}, "/*+ BROADCAST(w) */"),
    "sedona_kdbtree": ("Sedona partitioned join, KDB-tree + R-tree",
                       {"spark.sql.autoBroadcastJoinThreshold": "-1"}, ""),
    "sedona_quadtree": ("Sedona partitioned join, Quadtree + R-tree",
                        {"spark.sql.autoBroadcastJoinThreshold": "-1", "sedona.join.gridtype": "quadtree"}, ""),
}
PLAN_NODES = ["BroadcastIndexJoin", "RangeJoin", "DistanceJoin", "BroadcastNestedLoopJoin",
              "CartesianProduct", "KNNJoin"]


# ---------------------------------------------------------------- 5.0 export to Parquet

def raster_points(path):
    """Pixel centres of all valid (non-zero) pixels."""
    with rasterio.open(path) as src:
        arr, t = src.read(1), src.transform
    rows, cols = np.nonzero(arr)
    x = t.c + (cols + 0.5) * t.a
    y = t.f + (rows + 0.5) * t.e
    return arr, rows, cols, x, y


def export_inputs(force=False):
    BIG.mkdir(parents=True, exist_ok=True)
    if not force and all((BIG / f).exists() for f in ("px10.parquet", "px30.parquet", "wards.parquet",
                                                     "buildings.parquet", "roads.parquet")):
        print("Parquet inputs already exported (data/processed/bigdata)")
        return
    lc = OUT / "landcover_consistent"
    maps = {}
    for y in YEARS:
        with rasterio.open(lc / f"lc_landsat_{y}.tif") as src:
            maps[y] = src.read(1)
    with rasterio.open(OUT / "change_2000_2025.tif") as src:
        change = src.read(1)
    a25, rows, cols, x, y = raster_points(lc / "lc_landsat_2025.tif")
    px30 = pd.DataFrame({"pixel_id": (rows * a25.shape[1] + cols).astype("int64"), "x": x, "y": y})
    for yr in YEARS:
        px30[f"lc_{yr}"] = maps[yr][rows, cols]
    px30["change"] = change[rows, cols]
    px30.to_parquet(BIG / "px30.parquet", index=False)

    a10, rows, cols, x, y = raster_points(OUT / "landcover" / "lc_sentinel2_2025.tif")
    pd.DataFrame({"x": x, "y": y, "lc": a10[rows, cols]}).to_parquet(BIG / "px10.parquet", index=False)

    layers = {"wards": ["ward_no", "ward_name", "area_km2"],
              "buildings": ["osm_id", "building", "area_m2"],
              "roads": ["osm_id", "highway", "road_class", "length_m"]}
    for name, cols_ in layers.items():
        g = gpd.read_file(GPKG, layer=name, columns=cols_).to_crs(UTM)
        df = pd.DataFrame(g[cols_])
        if "osm_id" in df:
            df["osm_id"] = df.osm_id.astype(str)
        df["wkb"] = g.geometry.to_wkb()
        df.to_parquet(BIG / f"{name}.parquet", index=False)
    print(f"Exported to {BIG.relative_to(ROOT)}: px10 {len(a10.nonzero()[0]):,} points, px30 {len(px30):,} "
          f"points, wards/buildings/roads")


# ---------------------------------------------------------------- Spark session

def spark_session():
    os.environ.setdefault("HADOOP_HOME", str(ROOT / "tools" / "hadoop"))
    (Path(os.environ["HADOOP_HOME"]) / "bin").mkdir(parents=True, exist_ok=True)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    from sedona.spark import SedonaContext

    missing = [j for j in SEDONA_JARS if not (IVY / j).exists()]
    if missing:
        sys.exit(f"Sedona jars not found in {IVY}: {missing}.\nResolve them once (they are downloaded from "
                 f"Maven Central into ~/.ivy2/jars) with:\n  .venv/Scripts/spark-submit --packages "
                 f"{SEDONA_PACKAGES} --version")
    builder = (SedonaContext.builder().master("local[*]").appName("tvm-step5-sedona")
               .config("spark.driver.extraClassPath", ";".join((IVY / j).as_posix() for j in SEDONA_JARS))
               .config("spark.driver.memory", "8g")
               .config("spark.sql.shuffle.partitions", str(N_PARTITIONS))
               .config("spark.sql.adaptive.enabled", "false")       # keep the plans readable and comparable
               .config("spark.sql.execution.arrow.pyspark.enabled", "true")
               .config("spark.ui.showConsoleProgress", "false"))
    spark = SedonaContext.create(builder.getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    print(f"Spark {spark.version}, {spark.sparkContext.defaultParallelism} local cores, Sedona 1.8.0")
    return spark


def load_views(spark):
    for name, cols in {"wards": "ward_no, ward_name, area_km2", "buildings": "osm_id, building, area_m2",
                       "roads": "osm_id, highway, road_class, length_m"}.items():
        (spark.read.parquet(str(BIG / f"{name}.parquet"))
         .selectExpr(*cols.split(", "), "ST_GeomFromWKB(wkb) AS geom").createOrReplaceTempView(name))
    px30 = spark.read.parquet(str(BIG / "px30.parquet")).selectExpr("*", "ST_Point(x, y) AS geom")
    px30.cache().createOrReplaceTempView("px30")
    px10 = spark.read.parquet(str(BIG / "px10.parquet")).selectExpr("*", "ST_Point(x, y) AS geom")
    px10.cache().createOrReplaceTempView("px10")
    print(f"Loaded px10 {px10.count():,} points, px30 {px30.count():,} points")
    return px10


def plan_of(df):
    return df._jdf.queryExecution().executedPlan().toString()


def join_node(plan):
    return next((n for n in PLAN_NODES if n in plan), "other")


def set_conf(spark, conf):
    for k, v in {**BASE_CONF, **conf}.items():
        spark.conf.set(k, v)


# ---------------------------------------------------------------- 5.1 benchmark

def geopandas_join(pts, wards):
    t = time.perf_counter()
    j = gpd.sjoin(pts, wards[["ward_no", "geometry"]], predicate="within", how="inner")
    j.groupby("ward_no").size()
    return time.perf_counter() - t


def benchmark(spark, px10):
    total = px10.count()
    wards_gdf = gpd.read_file(GPKG, layer="wards").to_crs(UTM)
    px10_pd = pd.read_parquet(BIG / "px10.parquet", columns=["x", "y"])
    rows, plans = [], []
    query = ("SELECT {hint} w.ward_no, count(*) AS n FROM wards w JOIN pts p "
             "ON ST_Contains(w.geom, p.geom) GROUP BY w.ward_no")
    for size in BENCH_SIZES:
        if size is None or size >= total:
            pts = px10
        else:
            pts = px10.sample(fraction=size / total, seed=SEED)
        pts = pts.select("geom").cache()
        n = pts.count()
        pts.createOrReplaceTempView("pts")
        print(f"  {n:>9,} points")
        for key, (label, conf, hint) in METHODS.items():
            set_conf(spark, conf)
            times = []
            for _ in range(REPEATS):
                # a fresh DataFrame each time: re-collecting the same one would reuse its shuffle files
                df = spark.sql(query.format(hint=hint))
                t = time.perf_counter()
                res = df.collect()
                times.append(time.perf_counter() - t)
            plan = plan_of(df)
            matched = sum(r["n"] for r in res)
            rows.append({"points": n, "method": key, "label": label, "join_operator": join_node(plan),
                         "seconds": round(min(times), 3), "matched": matched})
            print(f"    {label:<48} {join_node(plan):<24} {min(times):7.2f} s  ({matched:,} matched)")
            if size == BENCH_SIZES[-1]:
                plans.append(f"===== {label} ({n:,} points) =====\n{plan}\n")
        # single-machine reference on a same-size sample
        sample = px10_pd if n >= total else px10_pd.sample(n=n, random_state=SEED)
        g = gpd.GeoDataFrame(geometry=gpd.points_from_xy(sample.x, sample.y), crs=UTM)
        secs = min(geopandas_join(g, wards_gdf) for _ in range(REPEATS))
        rows.append({"points": n, "method": "geopandas", "label": "GeoPandas sjoin (single machine, STRtree)",
                     "join_operator": "STRtree", "seconds": round(secs, 3), "matched": None})
        print(f"    {'GeoPandas sjoin (single machine, STRtree)':<48} {'STRtree':<24} {secs:7.2f} s")
        pts.unpersist()
    set_conf(spark, {})
    bench = pd.DataFrame(rows)
    bench["points_per_second"] = (bench.points / bench.seconds).round(0)
    bench.to_csv(OUT / "step5_benchmark.csv", index=False)

    # the aggregation as MapReduce: partial count (map + combiner) -> Exchange (shuffle) -> final count (reduce)
    mr = spark.sql("SELECT /*+ BROADCAST(w) */ w.ward_no, p.lc, count(*) AS n FROM wards w JOIN px10 p "
                   "ON ST_Contains(w.geom, p.geom) GROUP BY w.ward_no, p.lc")
    plans.insert(0, "===== Ward x class pixel count, read as MapReduce =====\n"
                    "map     : scan points, find the ward (index join), emit key (ward_no, lc)\n"
                    "combine : HashAggregate partial_count inside each task\n"
                    "shuffle : Exchange hashpartitioning(ward_no, lc)\n"
                    "reduce  : HashAggregate count per key\n\n" + plan_of(mr) + "\n")
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "step5_plans.txt").write_text("\n".join(plans), encoding="utf-8")
    return bench


# ---------------------------------------------------------------- 5.2 spatial partitioning

def partitions(spark, px10):
    from sedona.spark import GridType, StructuredAdapter

    pts = px10.select("geom")
    n_total = pts.count()
    stats, cells = [], []

    def summarize(name, df, grids=None):
        df.createOrReplaceTempView("part")
        p = spark.sql("SELECT spark_partition_id() AS pid, count(*) AS n, ST_AsBinary(ST_Envelope_Aggr(geom)) AS wkb "
                      "FROM part GROUP BY spark_partition_id()").toPandas()
        g = gpd.GeoDataFrame(p.drop(columns="wkb"), geometry=gpd.GeoSeries.from_wkb(p.wkb), crs=UTM)
        extent = gpd.GeoSeries(g.unary_union.envelope, crs=UTM).area.iloc[0]
        stats.append({"scheme": name, "partitions": len(g), "points": int(g.n.sum()),
                      "min_points": int(g.n.min()), "max_points": int(g.n.max()), "mean_points": round(g.n.mean()),
                      "cv_points": round(g.n.std() / g.n.mean(), 3),
                      "mean_mbr_pct_of_extent": round(100 * g.area.mean() / extent, 2),
                      "mbr_overlap_ratio": round(g.area.sum() / extent, 2)})
        g["scheme"] = name
        cells.append(g)
        if grids is not None:
            print(f"  {name}: {len(grids)} partition cells")

    summarize("Hash (Spark default)", pts.repartition(N_PARTITIONS))
    for name, gt in (("KDB-tree", GridType.KDBTREE), ("Quadtree", GridType.QUADTREE)):
        srdd = StructuredAdapter.toSpatialRdd(pts, "geom")
        srdd.analyze()
        srdd.spatialPartitioning(gt, N_PARTITIONS)
        try:
            grids = srdd.getPartitioner().getGrids()
        except Exception:                                   # grids are only used for the log line
            grids = None
        summarize(name, StructuredAdapter.toSpatialPartitionedDf(srdd, spark), grids)

    s = pd.DataFrame(stats)
    s.to_csv(OUT / "step5_partitions.csv", index=False)
    print(s.to_string(index=False))
    gpd.GeoDataFrame(pd.concat(cells, ignore_index=True), crs=UTM).to_file(SEDONA_GPKG, layer="partitions",
                                                                           driver="GPKG")
    assert all(s.points == n_total), "partitioning lost or duplicated points"
    return s


# ---------------------------------------------------------------- 5.3 analytics

def ward_landcover(spark):
    spark.sql("SELECT /*+ BROADCAST(w) */ w.ward_no, p.* FROM wards w JOIN px30 p "
              "ON ST_Contains(w.geom, p.geom)").drop("geom").cache().createOrReplaceTempView("px30w")
    spark.sql("SELECT /*+ BROADCAST(w) */ w.ward_no, p.lc FROM wards w JOIN px10 p "
              "ON ST_Contains(w.geom, p.geom)").createOrReplaceTempView("px10w")
    stacked = ", ".join(f"'{y}', lc_{y}" for y in YEARS)
    q = f"""
        SELECT ward_no, epoch, count(*) AS n, sum(CASE WHEN lc = {BUILT} THEN 1 ELSE 0 END) AS built,
               sum(CASE WHEN lc IN {GREEN} THEN 1 ELSE 0 END) AS green, sum(CASE WHEN lc = {TREE} THEN 1 ELSE 0 END) AS tree
        FROM (SELECT ward_no, stack({len(YEARS)}, {stacked}) AS (epoch, lc) FROM px30w
              UNION ALL SELECT ward_no, '2025_s2' AS epoch, lc FROM px10w)
        WHERE lc > 0 GROUP BY ward_no, epoch"""
    c = spark.sql(q).toPandas()
    for k in ("built", "green", "tree"):
        c[k] = (100 * c[k] / c.n).round(2)
    wide = c.pivot(index="ward_no", columns="epoch", values=["built", "green", "tree"])
    wide.columns = [f"{k}_pct_{e}" for k, e in wide.columns]
    wide = wide.reset_index()

    step4 = pd.read_csv(OUT / "ward_change.csv")
    cols = [c_ for c_ in wide.columns if c_ != "ward_no"]
    m = wide.merge(step4[["ward_no"] + cols], on="ward_no", suffixes=("_sedona", "_step4"))
    check = pd.DataFrame({"column": cols,
                          "max_abs_diff_pp": [round((m[f"{c_}_sedona"] - m[f"{c_}_step4"]).abs().max(), 3) for c_ in cols],
                          "mean_abs_diff_pp": [round((m[f"{c_}_sedona"] - m[f"{c_}_step4"]).abs().mean(), 4) for c_ in cols]})
    check.to_csv(OUT / "step5_ward_check.csv", index=False)
    print(f"  land-cover shares vs Step 4: max |diff| = {check.max_abs_diff_pp.max():.3f} pp over "
          f"{len(cols)} columns x {len(m)} wards")
    return wide.rename(columns={c_: f"sedona_{c_}" for c_ in cols})


def ward_buildings(spark):
    spark.sql("SELECT osm_id, area_m2, ST_PointOnSurface(geom) AS pt FROM buildings").createOrReplaceTempView("bpts")
    b = spark.sql("""
        SELECT /*+ BROADCAST(w) */ w.ward_no, count(*) AS buildings, sum(b.area_m2) AS footprint_m2,
               percentile_approx(b.area_m2, 0.5) AS median_building_m2
        FROM wards w JOIN bpts b ON ST_Contains(w.geom, b.pt) GROUP BY w.ward_no""").toPandas()
    return b


def ward_roads(spark):
    r = spark.sql(f"""
        SELECT /*+ BROADCAST(w) */ w.ward_no,
               sum(ST_Length(ST_Intersection(w.geom, r.geom))) / 1000 AS road_km,
               sum(CASE WHEN r.road_class IN {MAIN_ROADS} THEN ST_Length(ST_Intersection(w.geom, r.geom)) ELSE 0 END)
                   / 1000 AS main_road_km
        FROM wards w JOIN roads r ON ST_Intersects(w.geom, r.geom) GROUP BY w.ward_no""").toPandas()
    return r


def road_distance(spark):
    """Nearest main road for every 30 m pixel that was green in 2000 (inside the corporation)."""
    spark.sql(f"SELECT osm_id, geom FROM roads WHERE road_class IN {MAIN_ROADS}").createOrReplaceTempView("main_roads")
    spark.sql(f"SELECT pixel_id, ward_no, change, ST_Point(x, y) AS geom FROM px30w "
              f"WHERE lc_2000 IN {GREEN} AND lc_2025 > 0").createOrReplaceTempView("green00")
    try:
        q = spark.sql("SELECT p.pixel_id, min(ST_Distance(p.geom, r.geom)) AS dist_m FROM green00 p "
                      "JOIN main_roads r ON ST_KNN(p.geom, r.geom, 1, false) GROUP BY p.pixel_id")
        node = join_node(plan_of(q))
        d = q.toPandas()
    except Exception as e:                       # fall back to a distance join, also Sedona-optimised
        print(f"  kNN join failed ({type(e).__name__}), using ST_DWithin distance join")
        q = spark.sql(f"SELECT p.pixel_id, min(ST_Distance(p.geom, r.geom)) AS dist_m FROM green00 p "
                      f"JOIN main_roads r ON ST_DWithin(p.geom, r.geom, {DIST_BANDS[-2]}) GROUP BY p.pixel_id")
        node = join_node(plan_of(q))
        d = q.toPandas()
    g = spark.table("green00").select("pixel_id", "change").toPandas().merge(d, on="pixel_id", how="left")
    g["dist_m"] = g.dist_m.fillna(np.inf)
    labels = [f"{int(a)}-{int(b)} m" if np.isfinite(b) else f">{int(a)} m" for a, b in zip(DIST_BANDS, DIST_BANDS[1:])]
    g["band"] = pd.cut(g.dist_m, DIST_BANDS, labels=labels, right=False)
    g["converted"] = g.change == GREEN_TO_BUILT
    t = g.groupby("band", observed=False).agg(green_2000_px=("pixel_id", "size"), converted_px=("converted", "sum"))
    t["green_2000_km2"] = (t.green_2000_px * 900 / 1e6).round(3)
    t["converted_km2"] = (t.converted_px * 900 / 1e6).round(3)
    t["conversion_rate_pct"] = (100 * t.converted_px / t.green_2000_px).round(2)
    t["share_of_all_conversion_pct"] = (100 * t.converted_px / t.converted_px.sum()).round(1)
    t["share_of_green_2000_pct"] = (100 * t.green_2000_px / t.green_2000_px.sum()).round(1)
    t = t[t.green_2000_px > 0].reset_index()
    t.to_csv(OUT / "step5_road_distance.csv", index=False)
    print(f"  nearest main road via {node}: {len(g):,} green-2000 pixels")
    print(t[["band", "green_2000_km2", "converted_km2", "conversion_rate_pct", "share_of_all_conversion_pct"]]
          .to_string(index=False))
    # pixel-level distance is reused by the Step 6 association rules
    g[["pixel_id", "dist_m"]].replace(np.inf, np.nan).to_parquet(BIG / "green2000_road_distance.parquet", index=False)
    return t


def h3_grid(spark):
    # Sedona's ST_Transform returns lon/lat for EPSG:4326 (checked here, flipped if a build differs)
    x = spark.sql("SELECT ST_X(ST_Transform(ST_Point(500000, 940000), 'EPSG:32643', 'EPSG:4326')) AS x").first().x
    wgs = "ST_Transform(ST_Point(x, y), 'EPSG:32643', 'EPSG:4326')"
    if not 70 < x < 80:
        wgs = f"ST_FlipCoordinates({wgs})"
    spark.sql(f"SELECT *, ST_H3CellIDs({wgs}, {H3_RES}, false)[0] AS h3 FROM px30w").createOrReplaceTempView("px30h")
    h = spark.sql(f"""
        SELECT h3, count(*) AS pixels, first(ward_no) AS ward_no,
               sum(CASE WHEN lc_2000 IN {GREEN} THEN 1 ELSE 0 END) AS green_2000,
               sum(CASE WHEN lc_2025 IN {GREEN} THEN 1 ELSE 0 END) AS green_2025,
               sum(CASE WHEN lc_2000 = {BUILT} THEN 1 ELSE 0 END) AS built_2000,
               sum(CASE WHEN lc_2025 = {BUILT} THEN 1 ELSE 0 END) AS built_2025,
               sum(CASE WHEN change = {GREEN_TO_BUILT} THEN 1 ELSE 0 END) AS green_to_built,
               ST_AsBinary(ST_H3ToGeom(array(h3))[0]) AS wkb
        FROM px30h GROUP BY h3""").toPandas()
    g = gpd.GeoDataFrame(h.drop(columns="wkb"), geometry=gpd.GeoSeries.from_wkb(h.wkb), crs="EPSG:4326")
    g["h3"] = g.h3.map(lambda v: format(int(v), "x"))
    for k in ("green_2000", "green_2025", "built_2000", "built_2025"):
        g[f"{k}_pct"] = (100 * g[k] / g.pixels).round(1)
    g["green_change_pp"] = (g.green_2025_pct - g.green_2000_pct).round(1)
    g["green_to_built_km2"] = (g.green_to_built * 900 / 1e6).round(4)
    g["green_to_built_pct_of_green2000"] = (100 * g.green_to_built / g.green_2000.clip(lower=1)).round(1)
    g = g[g.pixels >= 20]                       # drop slivers along the corporation edge
    g.to_crs(UTM).to_file(SEDONA_GPKG, layer="h3_change", driver="GPKG")
    print(f"  H3 res {H3_RES}: {len(g):,} hexagons, top hexagon lost "
          f"{g.green_to_built_km2.max():.3f} km2 of green to built-up")
    return g


def analytics(spark):
    print("5.3a ward land-cover shares (pixel-in-polygon join)")
    lc = ward_landcover(spark)
    print("5.3b buildings per ward")
    b = ward_buildings(spark)
    print("5.3c roads per ward")
    r = ward_roads(spark)
    wards = pd.read_parquet(BIG / "wards.parquet", columns=["ward_no", "ward_name", "area_km2"])
    m = wards.merge(b, on="ward_no", how="left").merge(r, on="ward_no", how="left").merge(lc, on="ward_no")
    m[["buildings", "footprint_m2", "road_km", "main_road_km"]] = m[["buildings", "footprint_m2", "road_km",
                                                                      "main_road_km"]].fillna(0)
    m["buildings_per_km2"] = (m.buildings / m.area_km2).round(0)
    m["building_coverage_pct"] = (100 * m.footprint_m2 / (m.area_km2 * 1e6)).round(2)
    m["road_density_km_per_km2"] = (m.road_km / m.area_km2).round(2)
    m["main_road_density_km_per_km2"] = (m.main_road_km / m.area_km2).round(2)
    m = m.round({"footprint_m2": 0, "road_km": 2, "main_road_km": 2, "median_building_m2": 1})
    m.to_csv(OUT / "step5_ward_metrics.csv", index=False)
    print(f"  {int(m.buildings.sum()):,} buildings and {m.road_km.sum():,.0f} km of road assigned to wards")
    print("5.3d distance to main roads (kNN join)")
    road_distance(spark)
    print("5.3e H3 hexagon grid")
    h3_grid(spark)


# ---------------------------------------------------------------- maps and charts

def chart_benchmark():
    b = pd.read_csv(OUT / "step5_benchmark.csv")
    colors = {"spark_nested_loop": "#9e9e9e", "geopandas": "#444444", "sedona_broadcast": "#2c7fb8",
              "sedona_kdbtree": "#d7301f", "sedona_quadtree": "#fd8d3c"}
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for key, grp in b.groupby("method", sort=False):
        ax.plot(grp.points / 1e6, grp.seconds, marker="o", markersize=5, linewidth=2, color=colors.get(key),
                label=grp.label.iloc[0])
    ax.set_xlabel("Points joined to 101 wards (millions)")
    ax.set_ylabel("Seconds (best of 3)")
    ax.grid(color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Point-in-polygon join: Spark vs Sedona vs GeoPandas (local, 12 threads)", fontsize=10)
    fig.tight_layout()
    fig.savefig(MAPS / "step5_benchmark.png", dpi=150)
    plt.close(fig)


def map_partitions():
    cells = gpd.read_file(SEDONA_GPKG, layer="partitions")
    wards = gpd.read_file(GPKG, layer="wards").to_crs(UTM)
    sample = pd.read_parquet(BIG / "px10.parquet", columns=["x", "y"]).sample(40_000, random_state=SEED)
    stats = pd.read_csv(OUT / "step5_partitions.csv").set_index("scheme")
    schemes = list(stats.index)
    fig, axes = plt.subplots(1, len(schemes), figsize=(5 * len(schemes), 6))
    for ax, s in zip(axes, schemes):
        ax.scatter(sample.x, sample.y, s=0.2, color="#bdbdbd", rasterized=True)
        wards.boundary.plot(ax=ax, color="#666666", linewidth=0.3)
        for geom in cells[cells.scheme == s].geometry:
            x0, y0, x1, y1 = geom.bounds
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#d7301f", linewidth=0.9))
        st = stats.loc[s]
        ax.set_title(f"{s}\n{int(st.partitions)} partitions, load CV {st.cv_points:.2f}, "
                     f"mean MBR {st.mean_mbr_pct_of_extent:.0f}% of extent", fontsize=9)
        ax.set_axis_off()
        ax.set_aspect("equal")
    fig.suptitle("Data extent of each Spark partition (4 M Sentinel-2 pixel points)", fontsize=11)
    fig.tight_layout()
    fig.savefig(MAPS / "step5_partitions.png", dpi=150)
    plt.close(fig)


def chart_road_distance():
    t = pd.read_csv(OUT / "step5_road_distance.csv")
    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.bar(t.band, t.conversion_rate_pct, color="#d7301f", width=0.6)
    for i, v in enumerate(t.conversion_rate_pct):
        ax1.annotate(f"{v:.1f}%", (i, v), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=9)
    ax1.set_ylabel("Green 2000 pixels built up by 2025 (%)")
    ax1.set_xlabel("Distance to nearest main road (trunk / primary / secondary)")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax1.set_axisbelow(True)
    ax1.set_title("Green -> built-up conversion rate by distance to main roads, 2000-2025", fontsize=10)
    fig.tight_layout()
    fig.savefig(MAPS / "step5_road_distance.png", dpi=150)
    plt.close(fig)


def map_h3():
    h = gpd.read_file(SEDONA_GPKG, layer="h3_change")
    corp = gpd.read_file(GPKG, layer="corporation").to_crs(UTM)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    h.plot(ax=axes[0], column="green_to_built_pct_of_green2000", cmap="Reds", vmin=0,
           vmax=h.green_to_built_pct_of_green2000.quantile(0.98), linewidth=0, legend=True,
           legend_kwds={"label": "% of 2000 green turned built-up", "shrink": 0.6})
    lim = max(abs(h.green_change_pp.quantile(0.02)), abs(h.green_change_pp.quantile(0.98)))
    h.plot(ax=axes[1], column="green_change_pp", cmap="RdYlGn", vmin=-lim, vmax=lim, linewidth=0, legend=True,
           legend_kwds={"label": "Green share change 2000-2025 (pp)", "shrink": 0.6})
    for ax, title in zip(axes, ["Green -> built-up hotspots", "Net green change"]):
        corp.boundary.plot(ax=ax, color="#333333", linewidth=0.6)
        ax.set_title(f"{title} (H3 resolution {H3_RES})", fontsize=10)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(MAPS / "step5_h3.png", dpi=150)
    plt.close(fig)


def maps():
    chart_benchmark()
    map_partitions()
    chart_road_distance()
    map_h3()
    print("Saved outputs/maps/step5_benchmark.png, step5_partitions.png, step5_road_distance.png, step5_h3.png")


def main():
    parts = sys.argv[1:] or ["export", "bench", "partitions", "analytics", "maps"]
    if "export" in parts:
        export_inputs()
    if {"bench", "partitions", "analytics"} & set(parts):
        spark = spark_session()
        px10 = load_views(spark)
        if "bench" in parts:
            print("5.1 benchmark: pixels -> wards")
            benchmark(spark, px10)
        if "partitions" in parts:
            print("5.2 spatial partitioning")
            partitions(spark, px10)
        if "analytics" in parts:
            analytics(spark)
        spark.stop()
    if "maps" in parts:
        maps()


if __name__ == "__main__":
    main()
