"""
Step 5b - PostGIS: load the project data into a spatial database and run the analysis in SQL.

5b.1 Load    Database tvm_green, schema tvm (rebuilt on every run): wards, corporation, buildings, roads,
             landuse (Step 1), ward_change (Step 4), h3_change (Step 5a, if present), 451 k Landsat pixel
             points (COPY) and the 2000->2025 change raster (raster2pgsql, 64x64 tiles).
5b.2 SQL     sql/step5_postgis.sql: GiST indexes, CLUSTER, ward assignment, a geohash LIST-partitioned
             pixel table, and the analysis views (ward green loss from pixels and from the raster,
             buildings/roads per ward, KNN nearest main road, conversion by OSM land use).
5b.3 Benchmark  EXPLAIN ANALYZE of the same queries with the GiST indexes usable and with index scans
             switched off (planner falls back to sequential scans / nested loops), plus partition pruning.
5b.4 Export  Views to CSV; cross-checks against Step 4 (raster) and Step 5a (Sedona) where available.

Connection: PGHOST (default localhost), PGPORT (5432), PGUSER (postgres), PGPASSWORD. If PGPASSWORD is
not set, the password is asked for interactively. The password is never written to disk or printed.
  PowerShell:  $env:PGPASSWORD = "..." ; .venv/Scripts/python scripts/step5_postgis.py

Outputs: data/processed/step5_postgis_ward_metrics.csv, step5_postgis_road_distance.csv,
         step5_postgis_landuse.csv, step5_postgis_raster_check.csv, step5_postgis_partitions.csv,
         step5_postgis_benchmark.csv
         outputs/logs/step5_postgis_plans.txt

Run:  .venv/Scripts/python scripts/step5_postgis.py [load|sql|bench|export]   (default: all)
"""
import getpass
import io
import json
import os
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)
import geopandas as gpd
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import URL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
BIG = OUT / "bigdata"
GPKG = OUT / "base.gpkg"
LOGS = ROOT / "outputs" / "logs"
SQL_FILE = ROOT / "sql" / "step5_postgis.sql"
UTM_EPSG = 32643
DB = os.environ.get("TVM_DB", "tvm_green")
SCHEMA = "tvm"
PG_BIN = Path(os.environ.get("PG_BIN", r"C:\Program Files\PostgreSQL\13\bin"))
KNN_SAMPLE_MOD = 100          # KNN without index is O(n*m): benchmark it on every 100th pixel


PGPASS = Path(os.environ.get("PGPASSFILE", Path(os.environ.get("APPDATA", "~")) / "postgresql" / "pgpass.conf"))


def conn_params():
    """Password from PGPASSWORD, else from pgpass.conf (read by libpq itself), else asked for."""
    if not os.environ.get("PGPASSWORD") and not PGPASS.expanduser().exists():
        if not sys.stdin.isatty():
            sys.exit(f"Set PGPASSWORD or create {PGPASS} (localhost:5432:*:postgres:<password>).")
        os.environ["PGPASSWORD"] = getpass.getpass("PostgreSQL password: ")
    return {"host": os.environ.get("PGHOST", "localhost"), "port": int(os.environ.get("PGPORT", 5432)),
            "user": os.environ.get("PGUSER", "postgres"), "password": os.environ.get("PGPASSWORD")}


def connect(dbname=DB, autocommit=False):
    c = psycopg2.connect(dbname=dbname, **conn_params())
    c.autocommit = autocommit
    return c


def engine():
    p = conn_params()
    return create_engine(URL.create("postgresql+psycopg2", username=p["user"], password=p["password"],
                                    host=p["host"], port=p["port"], database=DB))


# ---------------------------------------------------------------- 5b.1 load

def ensure_database():
    with closing(connect("postgres", autocommit=True)) as c, c.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{DB}"')
            print(f"Created database {DB}")
    with closing(connect(autocommit=True)) as c, c.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis")
        cur.execute("CREATE EXTENSION IF NOT EXISTS postgis_raster")
        cur.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
        cur.execute(f"CREATE SCHEMA {SCHEMA}")
        cur.execute("SELECT postgis_full_version()")
        print(cur.fetchone()[0].split(" GEOS")[0])


def load_vectors():
    eng = engine()
    layers = [(GPKG, "wards"), (GPKG, "corporation"), (GPKG, "buildings"), (GPKG, "roads"), (GPKG, "landuse"),
              (OUT / "change.gpkg", "ward_change"), (OUT / "sedona.gpkg", "h3_change")]
    for path, layer in layers:
        if not path.exists():
            print(f"  skip {layer} ({path.name} not found)")
            continue
        g = gpd.read_file(path, layer=layer).to_crs(epsg=UTM_EPSG).rename_geometry("geom")
        g.columns = [c.lower() for c in g.columns]
        t = time.perf_counter()
        g.to_postgis(layer, eng, schema=SCHEMA, if_exists="replace", index=False, chunksize=20_000)
        print(f"  {layer:<12} {len(g):>8,} rows  {time.perf_counter() - t:5.1f} s")


def load_pixels():
    px = pd.read_parquet(BIG / "px30.parquet")
    cols = ", ".join(f"{c} {'bigint' if c == 'pixel_id' else 'double precision' if c in ('x', 'y') else 'smallint'}"
                     for c in px.columns)
    buf = io.StringIO()
    px.to_csv(buf, index=False, header=False)
    buf.seek(0)
    t = time.perf_counter()
    with connect() as c, c.cursor() as cur:
        cur.execute(f"CREATE TABLE {SCHEMA}.pixels30 ({cols})")
        cur.copy_expert(f"COPY {SCHEMA}.pixels30 ({', '.join(px.columns)}) FROM STDIN WITH (FORMAT csv)", buf)
    print(f"  pixels30     {len(px):>8,} rows  {time.perf_counter() - t:5.1f} s  (COPY)")


def load_raster():
    exe = PG_BIN / "raster2pgsql.exe"
    if not exe.exists():
        print(f"  skip change_raster (raster2pgsql not found in {PG_BIN})")
        return
    t = time.perf_counter()
    sql = subprocess.run([str(exe), "-s", str(UTM_EPSG), "-t", "64x64", "-I", "-C", "-N", "0",
                          str(OUT / "change_2000_2025.tif"), f"{SCHEMA}.change_raster"],
                         capture_output=True, check=True).stdout.decode("utf-8")
    with closing(connect(autocommit=True)) as c, c.cursor() as cur:      # the dump has its own BEGIN/END
        cur.execute(sql)
        cur.execute(f"ANALYZE {SCHEMA}.change_raster")
        cur.execute(f"SELECT count(*) FROM {SCHEMA}.change_raster")
        n = cur.fetchone()[0]
    print(f"  change_raster {n:>7,} tiles {time.perf_counter() - t:5.1f} s  (raster2pgsql)")


def load():
    ensure_database()
    print("5b.1 loading")
    load_vectors()
    load_pixels()
    load_raster()


# ---------------------------------------------------------------- 5b.2 SQL

def run_sql():
    print(f"5b.2 running {SQL_FILE.relative_to(ROOT)}")
    t = time.perf_counter()
    with connect() as c, c.cursor() as cur:
        cur.execute(SQL_FILE.read_text(encoding="utf-8"))
    print(f"  done in {time.perf_counter() - t:.1f} s")


# ---------------------------------------------------------------- 5b.3 benchmark

def explain(cur, query, settings=()):
    cur.execute("RESET ALL")
    cur.execute(f"SET search_path = {SCHEMA}, public")
    for s in settings:
        cur.execute(s)
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + query)
    plan = cur.fetchone()[0][0]
    nodes = []

    def walk(n):
        nodes.append(n["Node Type"] + (f" on {n['Relation Name']}" if "Relation Name" in n else "")
                     + (f" using {n['Index Name']}" if "Index Name" in n else ""))
        for ch in n.get("Plans", []):
            walk(ch)
    walk(plan["Plan"])
    return plan["Execution Time"] / 1000, nodes, plan


NO_INDEX = ("SET enable_indexscan = off", "SET enable_bitmapscan = off", "SET enable_indexonlyscan = off")
BENCH = {
    "pixel_in_ward_join": (
        "Point-in-polygon join, 451 k pixels -> 101 wards, count per ward",
        "SELECT w.ward_no, count(*) FROM wards w JOIN pixels30 p ON ST_Contains(w.geom, p.geom) GROUP BY w.ward_no"),
    "buildings_in_ward_join": (
        "203 k building centroids -> wards",
        "SELECT w.ward_no, count(*) FROM wards w JOIN buildings b ON ST_Contains(w.geom, ST_PointOnSurface(b.geom)) "
        "GROUP BY w.ward_no"),
    "range_query_1km": (
        "Range query: green->built pixels within 1 km of Technopark",
        "SELECT count(*) FROM pixels30 WHERE change = 3 AND ST_DWithin(geom, "
        "ST_Transform(ST_SetSRID(ST_MakePoint(76.8790, 8.5580), 4326), 32643), 1000)"),
    "knn_nearest_main_road": (
        f"KNN: nearest main road for every {KNN_SAMPLE_MOD}th green-2000 pixel",
        "SELECT p.pixel_id, (SELECT ST_Distance(p.geom, r.geom) FROM roads r WHERE r.road_class IN ('highway','major') "
        "ORDER BY r.geom <-> p.geom LIMIT 1) FROM pixels30 p "
        f"WHERE p.ward_no IS NOT NULL AND p.lc_2000 IN (2,3) AND p.pixel_id % {KNN_SAMPLE_MOD} = 0"),
}


def benchmark():
    print("5b.3 benchmark (EXPLAIN ANALYZE, best of 2)")
    rows, log = [], []
    with connect() as c, c.cursor() as cur:
        for key, (label, q) in BENCH.items():
            for mode, settings in (("gist_index", ()), ("no_index", NO_INDEX)):
                runs = [explain(cur, q, settings) for _ in range(2)]
                secs, nodes, plan = min(runs, key=lambda r: r[0])
                rows.append({"query": key, "label": label, "mode": mode, "seconds": round(secs, 3),
                             "plan_nodes": " > ".join(dict.fromkeys(nodes))})
                log.append(f"===== {label} [{mode}] {secs:.3f} s =====\n{json.dumps(plan['Plan'], indent=1)[:6000]}\n")
                print(f"  {label:<62} {mode:<11} {secs:8.3f} s")
        # partition pruning: one geohash cell vs the whole partitioned table
        cur.execute("RESET ALL")
        cur.execute(f"SET search_path = {SCHEMA}, public")
        cur.execute("SELECT gh5, count(*) FROM pixels30 GROUP BY gh5 ORDER BY 2 DESC")
        parts = pd.DataFrame(cur.fetchall(), columns=["gh5", "pixels"])
        cell = parts.gh5.iloc[0]
        for mode, where in (("one_partition", f"WHERE gh5 = '{cell}' AND change = 3"),
                            ("all_partitions", "WHERE change = 3")):
            secs, nodes, plan = explain(cur, f"SELECT count(*) FROM pixels30_part {where}")
            scanned = sum(1 for n in nodes if "pixels30_gh_" in n)
            rows.append({"query": "partition_pruning", "label": "Count green->built pixels, partitioned table",
                         "mode": mode, "seconds": round(secs, 3), "plan_nodes": f"{scanned} of {len(parts)} partitions"})
            print(f"  {'Partition pruning: green->built count':<62} {mode:<14} {secs:6.3f} s  "
                  f"({scanned} of {len(parts)} partitions)")
            log.append(f"===== partition pruning [{mode}] =====\n{json.dumps(plan['Plan'], indent=1)[:4000]}\n")
    parts["pct"] = (100 * parts.pixels / parts.pixels.sum()).round(1)
    parts.to_csv(OUT / "step5_postgis_partitions.csv", index=False)
    b = pd.DataFrame(rows)
    b.to_csv(OUT / "step5_postgis_benchmark.csv", index=False)
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "step5_postgis_plans.txt").write_text("\n".join(log), encoding="utf-8")
    return b


# ---------------------------------------------------------------- 5b.4 export and checks

def export():
    print("5b.4 export")
    with connect() as c:
        q = lambda sql: pd.read_sql(sql, c)
        wm = q(f"SELECT * FROM {SCHEMA}.mv_ward_metrics ORDER BY ward_no")
        rd = q(f"SELECT * FROM {SCHEMA}.v_road_distance_bands")
        lu = q(f"SELECT * FROM {SCHEMA}.v_conversion_by_landuse")
        rc = q(f"SELECT ward_no, change_code, km2 FROM {SCHEMA}.v_ward_change_raster")
        tot = q(f"SELECT * FROM {SCHEMA}.v_corporation_totals")
    wm.to_csv(OUT / "step5_postgis_ward_metrics.csv", index=False)
    rd.to_csv(OUT / "step5_postgis_road_distance.csv", index=False)
    lu.to_csv(OUT / "step5_postgis_landuse.csv", index=False)

    raster = rc[rc.change_code == 3].rename(columns={"km2": "green_to_built_km2_raster"})[["ward_no",
                                                                                         "green_to_built_km2_raster"]]
    chk = wm[["ward_no", "ward_name", "green_to_built_km2"]].merge(raster, on="ward_no", how="left").fillna(0)
    chk["diff_km2"] = (chk.green_to_built_km2_raster - chk.green_to_built_km2).round(3)
    chk.to_csv(OUT / "step5_postgis_raster_check.csv", index=False)

    print("  corporation totals (km2):", tot.iloc[0].to_dict())
    s4 = pd.read_csv(OUT / "change_summary.csv").set_index("period").loc["2000-2025"]
    print(f"  Step 4 change_summary 2000-2025: green {s4.green_start_km2} -> {s4.green_end_km2} km2, "
          f"green->built {s4.green_to_built_km2} km2")
    print(f"  raster vs vector green->built per ward: max |diff| {chk.diff_km2.abs().max():.3f} km2, "
          f"totals {chk.green_to_built_km2_raster.sum():.2f} vs {chk.green_to_built_km2.sum():.2f} km2")
    sedona = OUT / "step5_ward_metrics.csv"
    if sedona.exists():
        s = pd.read_csv(sedona)[["ward_no", "buildings", "road_km"]].merge(
            wm[["ward_no", "buildings", "road_km"]], on="ward_no", suffixes=("_sedona", "_postgis"))
        print(f"  Sedona vs PostGIS per ward: buildings max |diff| "
              f"{(s.buildings_sedona - s.buildings_postgis).abs().max():.0f}, road km max |diff| "
              f"{(s.road_km_sedona - s.road_km_postgis).abs().max():.3f}")
    print("  green->built by distance to main road:")
    print(rd.to_string(index=False))
    print("  green->built by OSM land use (top 8):")
    print(lu.head(8).to_string(index=False))


def main():
    parts = sys.argv[1:] or ["load", "sql", "bench", "export"]
    conn_params()
    if "load" in parts or "sql" in parts:          # the SQL file expects freshly loaded tables
        load()
    if "sql" in parts:
        run_sql()
    if "bench" in parts:
        benchmark()
    if "export" in parts:
        export()


if __name__ == "__main__":
    main()
