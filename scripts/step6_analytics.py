"""
Step 6 - Spatial analytics and decision support (syllabus Module 3: clustering, association rules, DSS;
Module 4: case studies on environment, society and disasters).

6.0 Auxiliary layers (Earth Engine, downloaded once to data/processed/aux, resampled onto the 30 m grid)
      - Copernicus GLO-30 DEM -> elevation and slope
      - MERIT Hydro HAND (height above nearest drainage) -> low-lying land (HAND <= 5 m): the paddy /
        wetland / floodplain belt that the land-cover classes cannot separate at 30 m
      - Sentinel-1 VV flood extent, August 2018 Kerala floods: descending-orbit scenes of 9 and 21 Aug
        2018 against the Feb-Apr 2018 dry-season median (change detection, permanent water and slopes
        removed)
      - MODIS MOD11A2 daytime land surface temperature, dry season 2002/03 and 2024/25 (1 km, but complete:
        Landsat LST is missing over the NW ~20% of the study area)
6.1 Hotspots
      - Getis-Ord Gi* on the Step 5 H3 hexagons (green -> built-up km2 per hexagon, 999 permutations,
        6-nearest-neighbour weights); global Moran's I for hexagons and wards
      - DBSCAN on the green -> built-up pixels: compact clusters of conversion (eps 100 m, 12 pixels)
6.2 Association rules (FP-Growth on Spark MLlib)
      One transaction per green-2000 pixel in the corporation; items describe where the pixel is
      (road distance, distance to the city centre and to Technopark, elevation, slope, low-lying, flooded
      in 2018, built-up around it in 2000, ward density, OSM land use, tree / other vegetation) plus the
      outcome (-> built-up or stayed green). Rules with the outcome as consequent show which
      combinations of conditions go with conversion (lift = how many times the city-wide rate).
6.3 Case studies
      a. Green space per capita (same WorldPop release only: 2000 vs 2010 old release, 2017 vs 2025 R2025A)
      b. Urban heat: Landsat LST change of converted vs stable pixels; MODIS ward LST vs green cover
      c. Floods and low-lying land: conversion in HAND <= 5 m land and inside the 2018 flood extent
6.4 Decision support: ward green-loss priority index (7 normalised indicators, weighted sum) with a
      sensitivity check across 4 weighting schemes.

Inputs:  Steps 1-5 outputs (base.gpkg, landcover_consistent/, change_2000_2025.tif, composites/,
         ward_change.csv, sedona.gpkg h3_change, bigdata/green2000_road_distance.parquet)
Outputs: data/processed/aux/*.tif
         data/processed/analytics.gpkg (hex_hotspots, dbscan_clusters, ward_priority)
         data/processed/step6_*.csv
         outputs/maps/step6_hotspots.png, step6_rules.png, step6_heat_flood.png, step6_priority.png

Run:  .venv/Scripts/python scripts/step6_analytics.py [aux|hotspots|rules|cases|priority|maps]  (default: all)
"""
import os
import sys
import warnings
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy import ndimage, stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
AUX = OUT / "aux"
BIG = OUT / "bigdata"
GPKG = OUT / "base.gpkg"
AGPKG = OUT / "analytics.gpkg"
MAPS = ROOT / "outputs" / "maps"
UTM = "EPSG:32643"
EE_PROJECT = "geometric-wall-477105-i2"
SEED = 42

YEARS = [2000, 2010, 2017, 2025]
BUILT, TREE, OTHER_VEG, WATER, BARE = 1, 2, 3, 4, 5
GREEN = (TREE, OTHER_VEG)
GREEN_TO_BUILT = 3
PX_KM2 = 900 / 1e6

LANDMARKS = {"centre": (76.9484, 8.5002),        # Secretariat, Statue Junction
             "technopark": (76.8790, 8.5580)}     # Technopark Phase I (same point as Step 5b)
LOWLAND_HAND_M = 5
DBSCAN_EPS_M, DBSCAN_MIN = 100, 12
FP_MIN_SUPPORT, FP_MIN_CONF = 0.003, 0.10

# ---------------------------------------------------------------- grid helpers

GRID_REF = OUT / "landcover_consistent" / "lc_landsat_2025.tif"
with rasterio.open(GRID_REF) as _s:
    SHAPE, TRANSFORM = _s.shape, _s.transform


def read(path, band=1):
    with rasterio.open(path) as s:
        return s.read(band)


def read_band(path, name):
    with rasterio.open(path) as s:
        return s.read(list(s.descriptions).index(name) + 1).astype("float32")


def to_grid(path, resampling=Resampling.bilinear):
    """Reproject any raster onto the 30 m land-cover grid."""
    dst = np.full(SHAPE, np.nan, dtype="float32")
    with rasterio.open(path) as s:
        reproject(rasterio.band(s, 1), dst, dst_transform=TRANSFORM, dst_crs=UTM, resampling=resampling,
                  dst_nodata=np.nan)
    return dst


def xy_grid():
    cols, rows = np.meshgrid(np.arange(SHAPE[1]), np.arange(SHAPE[0]))
    return TRANSFORM.c + (cols + 0.5) * TRANSFORM.a, TRANSFORM.f + (rows + 0.5) * TRANSFORM.e


def ward_grid():
    w = gpd.read_file(GPKG, layer="wards").to_crs(UTM)
    return rasterize(zip(w.geometry, w.ward_no), out_shape=SHAPE, transform=TRANSFORM, fill=0, dtype="int32")


def landmark_utm(name):
    p = gpd.GeoSeries(gpd.points_from_xy([LANDMARKS[name][0]], [LANDMARKS[name][1]]), crs=4326).to_crs(UTM)
    return p.x[0], p.y[0]


# ---------------------------------------------------------------- 6.0 auxiliary layers from Earth Engine

def aux_layers():
    import ee
    import geedim  # noqa: F401  (registers the .gd accessor)

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    ee.Initialize(project=EE_PROJECT)
    AUX.mkdir(parents=True, exist_ok=True)
    study = gpd.read_file(GPKG, layer="study_area").to_crs(4326)
    region = ee.Geometry(study.geometry.simplify(0.0002).iloc[0].__geo_interface__)

    def download(img, name, scale, resampling="near", dtype="float32"):
        path = AUX / f"{name}.tif"
        if path.exists():
            print(f"  {name}.tif already downloaded")
            return
        print(f"  downloading {name}.tif ({scale} m)")
        (img.clip(region).gd.prepareForExport(crs=UTM, region=region, scale=scale, resampling=resampling,
                                              dtype=dtype).gd.toGeoTIFF(path))

    # DEM: the 2024 release if this account sees it, else the original GLO-30
    try:
        dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30_2024_1")
        dem_col.size().getInfo()
    except ee.EEException:
        dem_col = ee.ImageCollection("COPERNICUS/DEM/GLO30")
    dem_col = dem_col.filterBounds(region)
    dem = dem_col.select("DEM").mosaic().setDefaultProjection(dem_col.first().select("DEM").projection())
    download(dem.rename("dem"), "dem", 30, "bilinear")

    hand = ee.Image("MERIT/Hydro/v1_0_1").select("hnd")
    download(hand.rename("hand"), "hand", 90, "bilinear")

    # Sentinel-1 flood extent, August 2018 (descending orbit only, so pre and post share the geometry)
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(region)
          .filter(ee.Filter.eq("instrumentMode", "IW"))
          .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
          .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING")).select("VV"))
    proj = s1.first().projection()
    pre = s1.filterDate("2018-02-01", "2018-05-01").median().focalMedian(30, "circle", "meters")
    post = s1.filterDate("2018-08-08", "2018-08-25").min().focalMedian(30, "circle", "meters")
    perm_water = ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence").unmask(0).gte(50)
    slope = ee.Terrain.slope(dem)
    flood = (post.lt(-16).And(post.subtract(pre).lt(-3)).And(perm_water.Not()).And(slope.lt(5))
             .rename("flood").setDefaultProjection(proj))
    download(flood.toFloat(), "flood_2018", 10)

    # MODIS daytime LST, dry seasons (same Nov-Apr window as the composites)
    for tag, (a, b) in {"2003": ("2002-11-01", "2003-04-30"), "2025": ("2024-11-01", "2025-04-30")}.items():
        col = ee.ImageCollection("MODIS/061/MOD11A2").filterDate(a, b).select("LST_Day_1km")
        lst = (col.mean().multiply(0.02).subtract(273.15).rename("lst")
               .setDefaultProjection(col.first().projection()))
        download(lst, f"modis_lst_{tag}", 1000)


def load_aux():
    a = {"dem": to_grid(AUX / "dem.tif"), "hand": to_grid(AUX / "hand.tif"),
         "flood": to_grid(AUX / "flood_2018.tif", Resampling.average) >= 0.5,
         "lst03": to_grid(AUX / "modis_lst_2003.tif"), "lst25": to_grid(AUX / "modis_lst_2025.tif")}
    gy, gx = np.gradient(np.nan_to_num(a["dem"], nan=0.0), 30.0)
    a["slope"] = np.degrees(np.arctan(np.hypot(gx, gy)))
    return a


# ---------------------------------------------------------------- 6.1 hotspots

def hotspots():
    import esda
    from libpysal.weights import KNN, Queen
    from sklearn.cluster import DBSCAN

    hexes = gpd.read_file(OUT / "sedona.gpkg", layer="h3_change").to_crs(UTM)
    y = hexes.green_to_built_km2.to_numpy()
    w = KNN.from_dataframe(hexes, k=6)
    gi = esda.G_Local(y, w, transform="B", permutations=999, star=True, seed=SEED)
    hexes["gi_z"] = np.round(gi.Zs, 3)
    hexes["gi_p"] = gi.p_sim
    z, p = hexes.gi_z, hexes.gi_p
    hexes["gi_class"] = np.select(
        [(z > 0) & (p <= 0.01), (z > 0) & (p <= 0.05), (z > 0) & (p <= 0.10),
         (z < 0) & (p <= 0.01), (z < 0) & (p <= 0.05), (z < 0) & (p <= 0.10)],
        ["Hot 99%", "Hot 95%", "Hot 90%", "Cold 99%", "Cold 95%", "Cold 90%"], "Not significant")
    fdr_p = esda.fdr(gi.p_sim, 0.05)
    hexes["gi_hot_fdr"] = (z > 0) & (p <= fdr_p)
    moran_hex = esda.Moran(y, w, permutations=999)

    wards = gpd.read_file(OUT / "change.gpkg", layer="ward_change").to_crs(UTM)
    wq = Queen.from_dataframe(wards, use_index=False)
    moran_ward = esda.Moran(wards.green_change_pp_2000_2025.to_numpy(), wq, permutations=999)

    # DBSCAN on green -> built-up pixels inside the corporation
    change, wg = read(OUT / "change_2000_2025.tif"), ward_grid()
    rows, cols = np.nonzero((change == GREEN_TO_BUILT) & (wg > 0))
    xs = TRANSFORM.c + (cols + 0.5) * TRANSFORM.a
    ys = TRANSFORM.f + (rows + 0.5) * TRANSFORM.e
    lab = DBSCAN(eps=DBSCAN_EPS_M, min_samples=DBSCAN_MIN).fit_predict(np.column_stack([xs, ys]))
    pts = gpd.GeoDataFrame({"cluster": lab, "ward_no": wg[rows, cols]},
                           geometry=gpd.points_from_xy(xs, ys), crs=UTM)
    cl = pts[pts.cluster >= 0]
    names = wards.set_index("ward_no").ward_name
    agg = cl.groupby("cluster").agg(pixels=("ward_no", "size"),
                                    ward_no=("ward_no", lambda s: s.mode().iloc[0]),
                                    wards=("ward_no", "nunique"))
    hull = cl.dissolve("cluster").convex_hull
    clusters = gpd.GeoDataFrame(agg, geometry=hull.reindex(agg.index), crs=UTM).reset_index()
    clusters["converted_km2"] = (clusters.pixels * PX_KM2).round(3)
    clusters["hull_km2"] = (clusters.area / 1e6).round(3)
    clusters["ward_name"] = clusters.ward_no.map(names)
    clusters = clusters.sort_values("converted_km2", ascending=False).reset_index(drop=True)
    clusters["rank"] = np.arange(1, len(clusters) + 1)

    hexes.to_file(AGPKG, layer="hex_hotspots", driver="GPKG")
    clusters.to_file(AGPKG, layer="dbscan_clusters", driver="GPKG")
    summary = pd.DataFrame([
        {"metric": "Moran's I, hexagon green->built km2", "value": round(moran_hex.I, 3),
         "p_sim": moran_hex.p_sim},
        {"metric": "Moran's I, ward green change pp 2000-2025", "value": round(moran_ward.I, 3),
         "p_sim": moran_ward.p_sim},
        {"metric": "Gi* hot hexagons (p<=0.05)", "value": int((hexes.gi_class.isin(["Hot 99%", "Hot 95%"])).sum()),
         "p_sim": None},
        {"metric": "Gi* hot hexagons after FDR (q=0.05)", "value": int(hexes.gi_hot_fdr.sum()), "p_sim": None},
        {"metric": "Green->built km2 in hot hexagons (p<=0.05)",
         "value": round(hexes.loc[hexes.gi_class.isin(["Hot 99%", "Hot 95%"]), "green_to_built_km2"].sum(), 2),
         "p_sim": None},
        {"metric": "DBSCAN clusters", "value": len(clusters), "p_sim": None},
        {"metric": "Green->built pixels in DBSCAN clusters (%)", "value": round(100 * len(cl) / len(pts), 1),
         "p_sim": None},
    ])
    summary.to_csv(OUT / "step6_hotspot_summary.csv", index=False)
    clusters.drop(columns="geometry").head(25).to_csv(OUT / "step6_dbscan_top.csv", index=False)
    print(summary.to_string(index=False))
    print("Top DBSCAN clusters:")
    print(clusters[["rank", "converted_km2", "hull_km2", "ward_name", "wards"]].head(10).to_string(index=False))


# ---------------------------------------------------------------- 6.2 association rules

def band(v, edges, labels):
    return pd.cut(v, edges, labels=labels, right=False).astype(str)


def transactions():
    lc00, lc25 = read(OUT / "landcover_consistent" / "lc_landsat_2000.tif"), \
        read(OUT / "landcover_consistent" / "lc_landsat_2025.tif")
    change, wg = read(OUT / "change_2000_2025.tif"), ward_grid()
    a = load_aux()
    gx, gy = xy_grid()
    built_nb = ndimage.uniform_filter((lc00 == BUILT).astype("float32"), size=11)   # ~330 m window

    lu = gpd.read_file(GPKG, layer="landuse").to_crs(UTM)
    lu["cls"] = lu.landuse.fillna(lu["natural"]).map({
        "residential": "residential", "commercial": "commercial", "retail": "commercial",
        "industrial": "industrial", "construction": "construction", "farmland": "agriculture",
        "meadow": "agriculture", "orchard": "agriculture", "plantation": "agriculture", "farmyard": "agriculture",
        "forest": "forest", "wood": "forest", "wetland": "wetland", "water": "wetland", "grass": "grass",
        "military": "institutional", "education": "institutional", "religious": "institutional"}).fillna("other")
    codes = {c: i + 1 for i, c in enumerate(sorted(lu.cls.unique()))}
    lu = lu.assign(area=lu.area).sort_values("area", ascending=False)          # small polygons drawn last
    lu_grid = rasterize(zip(lu.geometry, lu.cls.map(codes)), out_shape=SHAPE, transform=TRANSFORM, fill=0,
                        dtype="int32")
    inv = {v: k for k, v in codes.items()}

    m = np.isin(lc00, GREEN) & (wg > 0)
    rows, cols = np.nonzero(m)
    df = pd.DataFrame({"pixel_id": rows * SHAPE[1] + cols, "ward_no": wg[rows, cols]})
    rd = pd.read_parquet(BIG / "green2000_road_distance.parquet").groupby("pixel_id", as_index=False).dist_m.min()
    df = df.merge(rd, on="pixel_id", how="left")                 # kNN ties gave some pixels two rows
    if df.dist_m.isna().any():                                     # edge pixels the Sedona ward join left out
        from scipy.spatial import cKDTree
        rx = TRANSFORM.c + (rd.pixel_id % SHAPE[1] + 0.5) * TRANSFORM.a
        ry = TRANSFORM.f + (rd.pixel_id // SHAPE[1] + 0.5) * TRANSFORM.e
        miss = df.dist_m.isna().to_numpy()
        _, j = cKDTree(np.column_stack([rx, ry])).query(np.column_stack([gx[rows[miss], cols[miss]],
                                                                          gy[rows[miss], cols[miss]]]))
        df.loc[miss, "dist_m"] = rd.dist_m.to_numpy()[j]
    cx, cy = landmark_utm("centre")
    tx, ty = landmark_utm("technopark")
    x, y = gx[rows, cols], gy[rows, cols]
    wc = pd.read_csv(OUT / "ward_change.csv")
    dens00 = (wc.pop_2000 / wc.area_km2)
    dens_cls = pd.qcut(dens00, 3, labels=["low", "mid", "high"]).astype(str)
    dens_map = dict(zip(wc.ward_no, dens_cls))

    items = pd.DataFrame({
        "road": "road=" + band(df.dist_m, [0, 100, 250, 500, np.inf], ["<100m", "100-250m", "250-500m", ">500m"]),
        "centre": "centre=" + band(np.hypot(x - cx, y - cy) / 1000, [0, 3, 6, 10, np.inf],
                                   ["<3km", "3-6km", "6-10km", ">10km"]),
        "tpark": "technopark=" + band(np.hypot(x - tx, y - ty) / 1000, [0, 2, 5, np.inf], ["<2km", "2-5km", ">5km"]),
        "elev": "elev=" + band(a["dem"][rows, cols], [-np.inf, 10, 30, 60, np.inf],
                               ["<10m", "10-30m", "30-60m", ">60m"]),
        "slope": "slope=" + band(a["slope"][rows, cols], [0, 2, 8, np.inf], ["flat", "gentle", "steep"]),
        "low": np.where(a["hand"][rows, cols] <= LOWLAND_HAND_M, "lowland", "upland"),
        "flood": np.where(a["flood"][rows, cols], "flooded_2018", "not_flooded_2018"),
        "nb": "built_around_2000=" + band(built_nb[rows, cols], [0, 0.10, 0.30, np.inf], ["low", "mid", "high"]),
        "dens": "ward_density_2000=" + df.ward_no.map(dens_map),
        "veg": np.where(lc00[rows, cols] == TREE, "veg2000=tree", "veg2000=other"),
        "osm": pd.Series(lu_grid[rows, cols]).map(lambda v: f"osm={inv[v]}" if v else None),
    })
    out = np.where(change[rows, cols] == GREEN_TO_BUILT, "-> built-up",
                   np.where(np.isin(lc25[rows, cols], GREEN), "stayed green", "-> other"))
    items["outcome"] = out
    return df.assign(outcome=out), items


def rules():
    from pyspark.ml.fpm import FPGrowth
    from pyspark.sql import SparkSession

    df, items = transactions()
    tx = pd.DataFrame({"pixel_id": df.pixel_id,
                       "items": [[v for v in r if isinstance(v, str)] for r in items.itertuples(index=False)]})
    tx = tx[items.outcome != "-> other"]                     # outcome is either conversion or staying green
    tx.to_parquet(BIG / "rules_transactions.parquet", index=False)
    base_rate = (items.outcome[items.outcome != "-> other"] == "-> built-up").mean()
    print(f"{len(tx):,} transactions (green-2000 pixels), city-wide conversion rate {100 * base_rate:.2f}%")

    os.environ.setdefault("HADOOP_HOME", str(ROOT / "tools" / "hadoop"))
    os.environ["PYSPARK_PYTHON"] = os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark = (SparkSession.builder.master("local[*]").appName("tvm-step6-fpgrowth")
             .config("spark.driver.memory", "6g").config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    sdf = spark.read.parquet(str(BIG / "rules_transactions.parquet"))
    model = FPGrowth(itemsCol="items", minSupport=FP_MIN_SUPPORT, minConfidence=FP_MIN_CONF).fit(sdf)
    n_sets = model.freqItemsets.count()
    ar = model.associationRules.toPandas()
    spark.stop()

    n = len(tx)
    ar = ar[ar.consequent.map(lambda c: c == ["-> built-up"] or c == ["stayed green"])]
    ar = ar[ar.antecedent.map(len) <= 3].copy()
    ar["outcome"] = ar.consequent.str[0]
    ar["rule"] = ar.antecedent.map(lambda a: " & ".join(sorted(a)))
    ar["pixels"] = (ar.support * n).round().astype(int)
    ar["km2"] = (ar.pixels * PX_KM2).round(2)
    ar["antecedent_km2"] = (ar.km2 / ar.confidence).round(2)
    ar = ar.round({"confidence": 3, "lift": 2, "support": 4})
    # drop rules that add a condition without raising lift by at least 5% over a sub-rule (redundant)
    ar = ar.sort_values(["outcome", "lift"], ascending=[True, False])
    keep = []
    for out, g in ar.groupby("outcome"):
        best = {}
        for r in g.sort_values(by="antecedent", key=lambda s: s.map(len)).itertuples():
            subs = [best[k] for k in best if set(k) < set(r.antecedent)]
            if not subs or r.lift > 1.05 * max(subs):
                keep.append(r.Index)
            best[tuple(r.antecedent)] = r.lift
    ar = ar.loc[keep]
    cols = ["outcome", "rule", "antecedent_km2", "km2", "support", "confidence", "lift"]
    conv = ar[ar.outcome == "-> built-up"].sort_values("lift", ascending=False)
    conv = conv[conv.pixels >= 200]
    stay = ar[(ar.outcome == "stayed green") & (ar.lift > 1)].sort_values("lift", ascending=False)
    conv[cols].to_csv(OUT / "step6_rules_conversion.csv", index=False)
    stay[cols].head(30).to_csv(OUT / "step6_rules_stayed_green.csv", index=False)

    # single-condition conversion rates, the plain-language version of the rules
    long = items[items.outcome != "-> other"].melt(id_vars="outcome", value_name="item").dropna()
    uni = (long.groupby("item").outcome.agg(pixels="size", converted=lambda s: (s == "-> built-up").sum())
           .reset_index())
    uni["green_2000_km2"] = (uni.pixels * PX_KM2).round(2)
    uni["conversion_rate_pct"] = (100 * uni.converted / uni.pixels).round(2)
    uni["lift"] = (uni.conversion_rate_pct / (100 * base_rate)).round(2)
    uni["attribute"] = uni["item"].str.split("=").str[0].replace(
        {"lowland": "terrain", "upland": "terrain", "flooded_2018": "flood", "not_flooded_2018": "flood"})
    uni = uni[~uni["item"].isin(["-> built-up", "stayed green"])].sort_values(["attribute", "item"])
    uni.drop(columns="converted").to_csv(OUT / "step6_conversion_by_condition.csv", index=False)
    pd.DataFrame([{"transactions": n, "frequent_itemsets": n_sets, "min_support": FP_MIN_SUPPORT,
                   "min_confidence": FP_MIN_CONF, "base_conversion_rate_pct": round(100 * base_rate, 2),
                   "conversion_rules": len(conv), "stayed_green_rules": len(stay)}]
                 ).to_csv(OUT / "step6_rules_summary.csv", index=False)
    print(f"FP-Growth: {n_sets:,} frequent itemsets, {len(conv)} conversion rules kept")
    print(conv[cols].head(15).to_string(index=False))


# ---------------------------------------------------------------- 6.3 case studies

def zonal_mean(arr, wg):
    ok = (wg > 0) & np.isfinite(arr)
    s = pd.Series(arr[ok]).groupby(wg[ok]).mean()
    return s


def cases():
    wg = ward_grid()
    corp = wg > 0
    a = load_aux()
    lc = {y: read(OUT / "landcover_consistent" / f"lc_landsat_{y}.tif") for y in YEARS}
    change = read(OUT / "change_2000_2025.tif")
    wc = pd.read_csv(OUT / "ward_change.csv")
    rows = []

    # a. green space per capita, compared only within one WorldPop release
    g = wc.assign(g00=wc.green_pct_2000 * wc.area_km2 * 1e4, g10=wc.green_pct_2010 * wc.area_km2 * 1e4,
                  g17=wc.green_pct_2017 * wc.area_km2 * 1e4, g25=wc.green_pct_2025 * wc.area_km2 * 1e4)
    for lbl, gc, pc in [("2000 (WorldPop 2000-2020)", "g00", "pop_2000"), ("2010 (WorldPop 2000-2020)", "g10", "pop_2010"),
                        ("2017 (WorldPop R2025A)", "g17", "pop_2017_r25"), ("2025 (WorldPop R2025A)", "g25", "pop_2025_r25")]:
        pcap = g[gc] / g[pc]
        rows.append({"case": "green per capita", "metric": f"Corporation green m2 per person, {lbl}",
                     "value": round(g[gc].sum() / g[pc].sum(), 1)})
        rows.append({"case": "green per capita", "metric": f"Wards below 50 m2 green per person, {lbl}",
                     "value": int((pcap < 50).sum())})

    # b. heat: Landsat LST change for converted vs stable pixels (where both years have LST)
    l00 = read_band(OUT / "composites" / "landsat_2000.tif", "LST_C")
    l25 = read_band(OUT / "composites" / "landsat_2025.tif", "LST_C")
    ok = corp & np.isfinite(l00) & np.isfinite(l25) & (l00 > 0) & (l25 > 0)
    heat = []
    groups = {"Green -> built-up": change == GREEN_TO_BUILT,
              "Stable green": np.isin(lc[2000], GREEN) & np.isin(lc[2025], GREEN),
              "Stable built-up": (lc[2000] == BUILT) & (lc[2025] == BUILT)}
    for k, msk in groups.items():
        s = ok & msk
        heat.append({"group": k, "pixels": int(s.sum()), "lst_2000_c": round(float(l00[s].mean()), 2),
                     "lst_2025_c": round(float(l25[s].mean()), 2),
                     "delta_c": round(float((l25[s] - l00[s]).mean()), 2)})
    heat = pd.DataFrame(heat)
    ref = heat.set_index("group").delta_c
    heat["extra_warming_vs_stable_green_c"] = (heat.delta_c - ref["Stable green"]).round(2)
    heat["lst_coverage_pct_of_corporation"] = round(100 * ok.sum() / corp.sum(), 1)
    heat.to_csv(OUT / "step6_heat_landsat.csv", index=False)
    cls25 = pd.DataFrame([{"class": n, "lst_2025_c": round(float(l25[corp & np.isfinite(l25) & (l25 > 0) & (lc[2025] == c)].mean()), 2)}
                          for c, n in [(BUILT, "Built-up"), (TREE, "Tree cover"), (OTHER_VEG, "Other vegetation"),
                                       (WATER, "Water")]])
    cls25.to_csv(OUT / "step6_heat_by_class_2025.csv", index=False)

    wl = pd.DataFrame({"ward_no": wc.ward_no})
    wl["modis_lst_2003"] = wl.ward_no.map(zonal_mean(a["lst03"], wg)).round(2)
    wl["modis_lst_2025"] = wl.ward_no.map(zonal_mean(a["lst25"], wg)).round(2)
    wl["modis_lst_change"] = (wl.modis_lst_2025 - wl.modis_lst_2003).round(2)
    wm = wc.merge(wl, on="ward_no")
    r1 = stats.spearmanr(wm.green_pct_2025, wm.modis_lst_2025)
    r2 = stats.spearmanr(wm.green_change_pp_2000_2025, wm.modis_lst_change)
    rows += [{"case": "heat", "metric": "Spearman rho, ward green % 2025 vs MODIS LST 2025", "value": round(r1.statistic, 3),
              "p": r1.pvalue},
             {"case": "heat", "metric": "Spearman rho, ward green change 2000-25 vs MODIS LST change 2003-25",
              "value": round(r2.statistic, 3), "p": r2.pvalue},
             {"case": "heat", "metric": "Corporation MODIS dry-season day LST 2003 (C)",
              "value": round(float(np.nanmean(a["lst03"][corp])), 2)},
             {"case": "heat", "metric": "Corporation MODIS dry-season day LST 2025 (C)",
              "value": round(float(np.nanmean(a["lst25"][corp])), 2)}]

    # c. floods and low-lying land
    low = (a["hand"] <= LOWLAND_HAND_M) & corp
    green00 = np.isin(lc[2000], GREEN) & corp
    conv = change == GREEN_TO_BUILT
    flood = a["flood"] & corp
    for lbl, msk in {"low-lying (HAND <= 5 m)": low, "upland (HAND > 5 m)": corp & ~low}.items():
        rows += [{"case": "flood", "metric": f"Green 2000 km2, {lbl}", "value": round((green00 & msk).sum() * PX_KM2, 2)},
                 {"case": "flood", "metric": f"Green -> built-up km2, {lbl}", "value": round((conv & msk).sum() * PX_KM2, 2)},
                 {"case": "flood", "metric": f"Conversion rate %, {lbl}",
                  "value": round(100 * (conv & msk).sum() / max((green00 & msk).sum(), 1), 2)}]
    g17_b25 = np.isin(lc[2017], GREEN) & (lc[2025] == BUILT)
    rows += [{"case": "flood", "metric": "Mapped 2018 flood extent in corporation km2 (S1 change detection)",
              "value": round(flood.sum() * PX_KM2, 2)},
             {"case": "flood", "metric": "Of which low-lying (HAND <= 5 m) %",
              "value": round(100 * (flood & low).sum() / max(flood.sum(), 1), 1)},
             {"case": "flood", "metric": "Green 2017 -> built-up 2025 inside the 2018 flood extent km2",
              "value": round((g17_b25 & flood).sum() * PX_KM2, 3)},
             {"case": "flood", "metric": "Built-up 2025 inside the 2018 flood extent km2",
              "value": round(((lc[2025] == BUILT) & flood).sum() * PX_KM2, 3)},
             {"case": "flood", "metric": "Corporation low-lying land km2", "value": round(low.sum() * PX_KM2, 2)}]
    pd.DataFrame(rows).to_csv(OUT / "step6_case_studies.csv", index=False)

    # ward-level indicators reused by the priority index
    wf = pd.DataFrame({"ward_no": wc.ward_no})
    cnt = lambda m: pd.Series(np.bincount(wg[m], minlength=wg.max() + 1))     # noqa: E731
    wf["lowland_green_2000_km2"] = wf.ward_no.map(cnt(green00 & low) * PX_KM2).round(3)
    wf["lowland_green_to_built_km2"] = wf.ward_no.map(cnt(conv & low) * PX_KM2).round(3)
    wf["flood_2018_km2"] = wf.ward_no.map(cnt(flood) * PX_KM2).round(3)
    wf = wf.merge(wl, on="ward_no")
    wf.to_csv(OUT / "step6_ward_env.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(heat.to_string(index=False))


# ---------------------------------------------------------------- 6.4 decision support: priority index

INDICATORS = {   # column: (label, direction)  direction +1 = higher value -> higher priority
    "green_loss_pp": ("Green loss 2000-2025 (pp)", +1),
    "recent_loss_pp": ("Green loss 2017-2025 (pp)", +1),
    "hot_share_pct": ("Area in Gi* hotspots (%)", +1),
    "lowland_loss_pct": ("Low-lying green converted (%)", +1),
    "lst_2025": ("Dry-season LST 2025 (C)", +1),
    "green_per_capita": ("Green per person 2025 (m2)", -1),
    "density": ("Population density 2025", +1),
}
WEIGHTS = {
    "equal": dict.fromkeys(INDICATORS, 1),
    "loss_pressure": {"green_loss_pp": 3, "recent_loss_pp": 3, "hot_share_pct": 2, "lowland_loss_pct": 1,
                      "lst_2025": 1, "green_per_capita": 1, "density": 1},
    "climate_risk": {"green_loss_pp": 1, "recent_loss_pp": 1, "hot_share_pct": 1, "lowland_loss_pct": 3,
                     "lst_2025": 3, "green_per_capita": 1, "density": 1},
    "people": {"green_loss_pp": 1, "recent_loss_pp": 1, "hot_share_pct": 1, "lowland_loss_pct": 1,
               "lst_2025": 1, "green_per_capita": 3, "density": 3},
}


def priority():
    wards = gpd.read_file(OUT / "change.gpkg", layer="ward_change").to_crs(UTM)
    env = pd.read_csv(OUT / "step6_ward_env.csv")
    hexes = gpd.read_file(AGPKG, layer="hex_hotspots")
    hot = hexes[hexes.gi_class.isin(["Hot 99%", "Hot 95%"])]
    hot_area = gpd.overlay(wards[["ward_no", "geometry"]], hot[["geometry"]], how="intersection")
    hot_area = hot_area.assign(a=hot_area.area).groupby("ward_no").a.sum()

    d = wards[["ward_no", "ward_name", "area_km2", "geometry"]].merge(env, on="ward_no")
    wc = pd.read_csv(OUT / "ward_change.csv")
    d = d.merge(wc[["ward_no", "green_change_pp_2000_2025", "green_change_pp_2017_2025", "green_pct_2025",
                    "pop_2025_r25", "density_2025_per_km2"]], on="ward_no")
    d["green_loss_pp"] = -d.green_change_pp_2000_2025
    d["recent_loss_pp"] = -d.green_change_pp_2017_2025
    d["hot_share_pct"] = (100 * d.ward_no.map(hot_area).fillna(0) / (d.area_km2 * 1e6)).clip(upper=100).round(1)
    d["lowland_loss_pct"] = (100 * d.lowland_green_to_built_km2 / d.lowland_green_2000_km2.where(
        d.lowland_green_2000_km2 > 0.05)).fillna(0).round(1)
    d["lst_2025"] = d.modis_lst_2025
    d["green_per_capita"] = (d.green_pct_2025 * d.area_km2 * 1e4 / d.pop_2025_r25).round(1)
    d["density"] = d.density_2025_per_km2

    norm = pd.DataFrame(index=d.index)
    for c, (_, sign) in INDICATORS.items():
        v = d[c].astype(float)
        v = v.clip(v.quantile(0.02), v.quantile(0.98))           # keep one outlier from flattening the rest
        s = (v - v.min()) / (v.max() - v.min())
        norm[c] = s if sign > 0 else 1 - s
        d[f"n_{c}"] = norm[c].round(3)
    for name, wts in WEIGHTS.items():
        w = pd.Series(wts)
        d[f"index_{name}"] = (100 * (norm[w.index] * w).sum(axis=1) / w.sum()).round(1)
        d[f"rank_{name}"] = d[f"index_{name}"].rank(ascending=False, method="min").astype(int)
    d["priority_index"] = d.index_equal
    d["priority_rank"] = d.rank_equal
    d["tier"] = pd.cut(d.priority_rank, [0, 20, 50, 200], labels=["Priority 1 (top 20)", "Priority 2 (21-50)",
                                                                 "Priority 3"]).astype(str)
    # sensitivity: how stable is the ranking under other weightings?
    ranks = d[[f"rank_{n}" for n in WEIGHTS]]
    d["rank_min"], d["rank_max"] = ranks.min(axis=1), ranks.max(axis=1)
    d["top20_in_all_schemes"] = (ranks <= 20).all(axis=1)
    sens = [{"scheme": n, "spearman_vs_equal": round(stats.spearmanr(d.rank_equal, d[f"rank_{n}"]).statistic, 3),
             "top20_overlap_with_equal": int(((d.rank_equal <= 20) & (d[f"rank_{n}"] <= 20)).sum())}
            for n in WEIGHTS]
    pd.DataFrame(sens).to_csv(OUT / "step6_priority_sensitivity.csv", index=False)

    keep = (["ward_no", "ward_name", "priority_rank", "priority_index", "tier", "rank_min", "rank_max",
             "top20_in_all_schemes"] + list(INDICATORS) + [f"n_{c}" for c in INDICATORS]
            + [f"index_{n}" for n in WEIGHTS] + [f"rank_{n}" for n in WEIGHTS])
    d = d.sort_values("priority_rank")
    d[keep + ["geometry"]].to_file(AGPKG, layer="ward_priority", driver="GPKG")
    d[keep].to_csv(OUT / "step6_ward_priority.csv", index=False)
    print(pd.DataFrame(sens).to_string(index=False))
    print(d[["priority_rank", "ward_name", "priority_index", "rank_min", "rank_max", "green_loss_pp",
             "hot_share_pct", "lowland_loss_pct", "lst_2025", "green_per_capita"]].head(20).to_string(index=False))


# ---------------------------------------------------------------- maps and charts

def outline(ax):
    gpd.read_file(GPKG, layer="corporation").to_crs(UTM).boundary.plot(ax=ax, color="#333333", linewidth=0.8)
    ax.set_axis_off()


def map_hotspots():
    hexes = gpd.read_file(AGPKG, layer="hex_hotspots")
    cl = gpd.read_file(AGPKG, layer="dbscan_clusters")
    colors = {"Hot 99%": "#b2182b", "Hot 95%": "#ef8a62", "Hot 90%": "#fddbc7", "Not significant": "#f0f0f0",
              "Cold 90%": "#d1e5f0", "Cold 95%": "#67a9cf", "Cold 99%": "#2166ac"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 9))
    ax = axes[0]
    for k, c in colors.items():
        s = hexes[hexes.gi_class == k]
        if len(s):
            s.plot(ax=ax, color=c, edgecolor="white", linewidth=0.2, label=f"{k} ({len(s)})")
    outline(ax)
    ax.legend(loc="lower left", fontsize=8, title="Getis-Ord Gi*")
    ax.set_title("Hotspots of green -> built-up conversion (H3 res 9)")
    ax = axes[1]
    wards = gpd.read_file(GPKG, layer="wards").to_crs(UTM)
    wards.boundary.plot(ax=ax, color="#cccccc", linewidth=0.4)
    cl.plot(ax=ax, column="converted_km2", cmap="Reds", edgecolor="#67000d", linewidth=0.6, legend=True,
            legend_kwds={"label": "Green -> built-up in cluster (km2)", "shrink": 0.6})
    for r in cl.head(8).itertuples():
        p = r.geometry.representative_point()
        ax.annotate(f"{r.rank}. {r.ward_name}", (p.x, p.y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    outline(ax)
    ax.set_title(f"DBSCAN clusters of conversion (eps {DBSCAN_EPS_M} m, min {DBSCAN_MIN} px): {len(cl)} clusters")
    fig.tight_layout()
    fig.savefig(MAPS / "step6_hotspots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_rules():
    r = pd.read_csv(OUT / "step6_rules_conversion.csv").head(15).iloc[::-1]
    base = pd.read_csv(OUT / "step6_rules_summary.csv").base_conversion_rate_pct[0]
    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(r.rule, r.lift, color="#d7301f")
    for b, c, k in zip(bars, r.confidence, r.km2):
        ax.text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2,
                f"{100 * c:.0f}% converted, {k:.2f} km2", va="center", fontsize=8)
    ax.axvline(1, color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Lift (times the city-wide conversion rate of {base:.1f}%)")
    ax.set_title("FP-Growth rules: conditions -> green converted to built-up, 2000-2025")
    ax.set_xlim(0, r.lift.max() * 1.45)
    ax.tick_params(axis="y", labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(MAPS / "step6_rules.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def chart_heat_flood():
    heat = pd.read_csv(OUT / "step6_heat_landsat.csv")
    cs = pd.read_csv(OUT / "step6_case_studies.csv").set_index("metric").value
    a = load_aux()
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), gridspec_kw={"width_ratios": [1, 1, 1.2]})
    ax = axes[0]
    ax.bar(heat.group, heat.delta_c, color=["#d7301f", "#1a7837", "#888888"])
    for i, v in enumerate(heat.delta_c):
        ax.text(i, v + 0.05 * np.sign(v), f"{v:+.2f} C", ha="center", va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="#333333", linewidth=0.6)
    ax.set_ylabel("Mean change in Landsat dry-season LST, 2000 -> 2025 (C)")
    ax.set_title(f"Heat: converted pixels warmed most\n(pixels with LST in both years, "
                 f"{heat.lst_coverage_pct_of_corporation[0]:.0f}% of the corporation)", fontsize=10)
    ax = axes[1]
    labels = ["low-lying (HAND <= 5 m)", "upland (HAND > 5 m)"]
    vals = [cs[f"Conversion rate %, {l}"] for l in labels]
    ax.bar(["Low-lying\n(HAND <= 5 m)", "Upland"], vals, color=["#2166ac", "#a6761d"])
    for i, (l, v) in enumerate(zip(labels, vals)):
        ax.text(i, v + 0.1, f"{v:.1f}%\n{cs[f'Green -> built-up km2, {l}']:.2f} km2", ha="center", va="bottom")
    ax.set_ylabel("Green 2000 converted to built-up by 2025 (%)")
    ax.set_title("Floods: conversion of low-lying green land", fontsize=10)
    ax.set_ylim(0, max(vals) * 1.35)
    for x in axes[:2]:
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
    ax = axes[2]
    wg = ward_grid()
    ext = (TRANSFORM.c, TRANSFORM.c + SHAPE[1] * TRANSFORM.a, TRANSFORM.f + SHAPE[0] * TRANSFORM.e, TRANSFORM.f)
    low = np.where((a["hand"] <= LOWLAND_HAND_M) & (wg > 0), 1.0, np.nan)
    ax.imshow(low, extent=ext, cmap="Blues", vmin=0, vmax=2, interpolation="nearest")
    fl = np.where(a["flood"] & (wg > 0), 1.0, np.nan)
    ax.imshow(fl, extent=ext, cmap=ListedColormap(["#ff0000"]), interpolation="nearest")
    conv = np.where(read(OUT / "change_2000_2025.tif") == GREEN_TO_BUILT, 1.0, np.nan)
    conv[wg == 0] = np.nan
    ax.imshow(conv, extent=ext, cmap="Greys", vmin=0, vmax=1.2, interpolation="nearest", alpha=0.8)
    outline(ax)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#9ecae1", label="Low-lying (HAND <= 5 m)"),
                       Patch(color="#ff0000", label="Flooded, Aug 2018 (Sentinel-1)"),
                       Patch(color="#333333", label="Green -> built-up 2000-2025")], loc="lower left", fontsize=8)
    ax.set_title("Low-lying land, 2018 flood extent and conversion", fontsize=10)
    fig.tight_layout()
    fig.savefig(MAPS / "step6_heat_flood.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def map_priority():
    d = gpd.read_file(AGPKG, layer="ward_priority")
    fig, axes = plt.subplots(1, 2, figsize=(15, 9), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    d.plot(ax=ax, column="priority_index", cmap="YlOrRd", edgecolor="white", linewidth=0.4, legend=True,
           legend_kwds={"label": "Green-loss priority index (0-100)", "shrink": 0.6})
    d[d.top20_in_all_schemes].boundary.plot(ax=ax, color="#000000", linewidth=1.2)
    for r in d.head(10).itertuples():
        p = r.geometry.representative_point()
        ax.annotate(str(r.priority_rank), (p.x, p.y), fontsize=8, ha="center", fontweight="bold")
    outline(ax)
    ax.set_title("Ward priority index (equal weights)\nbold outline = top 20 under all 4 weightings")
    ax = axes[1]
    top = d.head(15).iloc[::-1]
    ncols = [f"n_{c}" for c in INDICATORS]
    left = np.zeros(len(top))
    cmap = plt.get_cmap("tab10")
    for i, c in enumerate(ncols):
        v = top[c].to_numpy() * 100 / len(ncols)
        ax.barh(top.ward_name, v, left=left, color=cmap(i), label=INDICATORS[c[2:]][0])
        left += v
    for y, (v, lo, hi) in enumerate(zip(top.priority_index, top.rank_min, top.rank_max)):
        ax.text(v + 0.5, y, f"{v:.0f}  (rank {lo}-{hi})", va="center", fontsize=8)
    ax.set_xlim(0, top.priority_index.max() * 1.3)
    ax.set_xlabel("Index contribution by indicator")
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("Top 15 wards: what drives the score")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(MAPS / "step6_priority.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def maps():
    map_hotspots()
    chart_rules()
    chart_heat_flood()
    map_priority()
    print("Saved outputs/maps/step6_hotspots.png, step6_rules.png, step6_heat_flood.png, step6_priority.png")


def main():
    parts = sys.argv[1:] or ["aux", "hotspots", "rules", "cases", "priority", "maps"]
    steps = {"aux": aux_layers, "hotspots": hotspots, "rules": rules, "cases": cases, "priority": priority,
             "maps": maps}
    for p in parts:
        print(f"===== 6: {p} =====")
        steps[p]()


if __name__ == "__main__":
    main()
