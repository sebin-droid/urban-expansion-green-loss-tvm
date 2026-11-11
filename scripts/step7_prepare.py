"""
Step 7a - Prepare the dashboard data (dashboard/data). The app itself is dashboard/app.py.

Everything the dashboard shows is pre-computed here into small web-ready files, so the app starts in
seconds and needs no Spark or Earth Engine:
  wards.geojson          101 wards (WGS84) with every ward indicator from Steps 4-6
  hexes.geojson          H3 hexagons with Gi* classes (Step 6) and green / built-up shares (Step 5)
  clusters.geojson       DBSCAN clusters of green -> built-up conversion (Step 6)
  roads_main.geojson     motorway / trunk / primary / secondary roads (Step 1)
  corporation.geojson    corporation outline
  img/*.png + overlays.json   land cover 2000/2010/2017/2025 (Landsat) and 2025 (Sentinel-2), change
                         2000-2025, low-lying land and the 2018 flood extent as transparent PNG overlays,
                         warped to Web Mercator so they line up with the base map
  px30.parquet           the 451 k Landsat pixel points, for the local (non-PostGIS) spatial query
  tables/*.csv           trend, transitions, rules, case studies, priority, benchmarks

Run:  .venv/Scripts/python scripts/step7_prepare.py
"""
import json
import os
import shutil
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.features import rasterize
from rasterio.warp import Resampling, calculate_default_transform, reproject, transform_bounds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
DASH = ROOT / "dashboard" / "data"
IMG = DASH / "img"
TAB = DASH / "tables"
GPKG = OUT / "base.gpkg"
UTM = "EPSG:32643"
YEARS = [2000, 2010, 2017, 2025]

LC_COLORS = {1: "#d7301f", 2: "#1a7837", 3: "#a6d96a", 4: "#2c7fb8", 5: "#e8d8a8"}
LC_NAMES = {1: "Built-up", 2: "Tree cover", 3: "Other vegetation", 4: "Water", 5: "Bare/sand"}
CHANGE_COLORS = {1: "#9e9e9e", 2: "#c7e9c0", 3: "#d7301f", 4: "#fd8d3c", 5: "#8c6d31", 6: "#1a7837", 7: "#c6dbef"}
CHANGE_NAMES = {1: "Built-up (stable)", 2: "Green (stable)", 3: "Green -> built-up", 4: "Other -> built-up",
                5: "Green -> water/bare", 6: "Gained green", 7: "Water/bare (stable)"}


def rgba(hex_, a=255):
    return tuple(int(hex_[i:i + 2], 16) for i in (1, 3, 5)) + (a,)


def corp_mask_like(path):
    corp = gpd.read_file(GPKG, layer="corporation").to_crs(UTM)
    with rasterio.open(path) as s:
        return rasterize(corp.geometry, out_shape=s.shape, transform=s.transform, fill=0, dtype="uint8") == 1


def to_png(arr, src_transform, name, colors, alpha=235, resampling=Resampling.nearest, upscale=1):
    """Warp a class raster (UTM) to Web Mercator and write a transparent PNG; return its lat/lon bounds."""
    h, w = arr.shape
    left, top = src_transform.c, src_transform.f
    right, bottom = left + w * src_transform.a, top + h * src_transform.e
    dt, dw, dh = calculate_default_transform(UTM, "EPSG:3857", w, h, left, bottom, right, top)
    dw, dh = dw * upscale, dh * upscale
    dt = dt * dt.scale(1 / upscale, 1 / upscale)
    dst = np.zeros((dh, dw), dtype="uint8")
    reproject(arr.astype("uint8"), dst, src_transform=src_transform, src_crs=UTM, dst_transform=dt,
              dst_crs="EPSG:3857", resampling=resampling, src_nodata=0, dst_nodata=0)
    lut = np.zeros((256, 4), dtype="uint8")
    for k, c in colors.items():
        lut[k] = rgba(c, alpha)
    Image.fromarray(lut[dst], "RGBA").save(IMG / f"{name}.png", optimize=True)
    b = transform_bounds("EPSG:3857", "EPSG:4326", dt.c, dt.f + dh * dt.e, dt.c + dw * dt.a, dt.f)
    return [[b[1], b[0]], [b[3], b[2]]]                      # [[south, west], [north, east]] for Leaflet


def overlays():
    meta = {}
    lc_dir = OUT / "landcover_consistent"
    for y in YEARS:
        p = lc_dir / f"lc_landsat_{y}.tif"
        with rasterio.open(p) as s:
            a = s.read(1) * corp_mask_like(p)
            meta[f"lc_{y}"] = {"bounds": to_png(a, s.transform, f"lc_{y}", LC_COLORS, upscale=2),
                               "label": f"Land cover {y} (Landsat 30 m)"}
    p = OUT / "landcover" / "lc_sentinel2_2025.tif"
    with rasterio.open(p) as s:
        a = s.read(1) * corp_mask_like(p)
        meta["lc_2025_s2"] = {"bounds": to_png(a, s.transform, "lc_2025_s2", LC_COLORS),
                              "label": "Land cover 2025 (Sentinel-2 10 m)"}
    p = OUT / "change_2000_2025.tif"
    with rasterio.open(p) as s:
        a = s.read(1) * corp_mask_like(p)
        t = s.transform
        meta["change"] = {"bounds": to_png(a, t, "change", CHANGE_COLORS, upscale=2),
                          "label": "Change 2000-2025"}
        only = np.where(a == 3, 3, 0)
        meta["green_to_built"] = {"bounds": to_png(only, t, "green_to_built", {3: "#d7301f"}, upscale=2),
                                  "label": "Green -> built-up 2000-2025"}
        corp = a > 0
    aux = OUT / "aux"
    if (aux / "hand.tif").exists():
        hand = np.full(corp.shape, np.nan, dtype="float32")
        with rasterio.open(aux / "hand.tif") as s:
            reproject(rasterio.band(s, 1), hand, dst_transform=t, dst_crs=UTM, resampling=Resampling.bilinear,
                      dst_nodata=np.nan)
        low = np.where((hand <= 5) & corp, 1, 0)
        meta["lowland"] = {"bounds": to_png(low, t, "lowland", {1: "#6baed6"}, alpha=170, upscale=2),
                           "label": "Low-lying land (HAND <= 5 m)"}
        fl = np.zeros(corp.shape, dtype="float32")
        with rasterio.open(aux / "flood_2018.tif") as s:
            reproject(rasterio.band(s, 1), fl, dst_transform=t, dst_crs=UTM, resampling=Resampling.average)
        meta["flood"] = {"bounds": to_png(np.where((fl >= 0.5) & corp, 1, 0), t, "flood", {1: "#ff00ff"},
                                          upscale=2),
                         "label": "Flooded, Aug 2018 (Sentinel-1)"}
    meta["_legend"] = {"landcover": {LC_NAMES[k]: v for k, v in LC_COLORS.items()},
                       "change": {CHANGE_NAMES[k]: v for k, v in CHANGE_COLORS.items()}}
    (DASH / "overlays.json").write_text(json.dumps(meta, indent=1))
    return corp, t


def corporation_trend(corp, t):
    rows = []
    for y in YEARS:
        with rasterio.open(OUT / "landcover_consistent" / f"lc_landsat_{y}.tif") as s:
            a = s.read(1)[corp]
        for k, n in LC_NAMES.items():
            rows.append({"year": y, "class": n, "km2": round((a == k).sum() * 900 / 1e6, 2)})
    d = pd.DataFrame(rows)
    d["pct"] = (100 * d.km2 / d.groupby("year").km2.transform("sum")).round(2)
    d.to_csv(TAB / "trend.csv", index=False)


def wards():
    w = gpd.read_file(OUT / "change.gpkg", layer="ward_change")
    keep_wc = [c for c in w.columns if c != "geometry"]
    m5 = pd.read_csv(OUT / "step5_ward_metrics.csv")
    m5 = m5[["ward_no", "buildings", "footprint_m2", "buildings_per_km2", "building_coverage_pct", "road_km",
             "main_road_km", "road_density_km_per_km2", "main_road_density_km_per_km2"]]
    env = pd.read_csv(OUT / "step6_ward_env.csv")
    pr = pd.read_csv(OUT / "step6_ward_priority.csv").drop(columns=["ward_name"])
    pg = OUT / "step5_postgis_ward_metrics.csv"
    g2b = (pd.read_csv(pg)[["ward_no", "green_to_built_km2"]] if pg.exists() else None)
    d = w[keep_wc + ["geometry"]].merge(m5, on="ward_no", how="left").merge(env, on="ward_no", how="left") \
        .merge(pr, on="ward_no", how="left")
    if g2b is not None:
        d = d.merge(g2b, on="ward_no", how="left")
    d["green_per_capita_2025"] = (d.green_pct_2025 * d.area_km2 * 1e4 / d.pop_2025_r25).round(1)
    d = d.to_crs(4326)
    d["geometry"] = d.geometry.simplify(0.00005)
    d.to_file(DASH / "wards.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)
    print(f"wards.geojson: {len(d)} wards, {len(d.columns)} columns")


def vectors():
    hexes = gpd.read_file(OUT / "analytics.gpkg", layer="hex_hotspots").to_crs(4326)
    hexes[["h3", "ward_no", "pixels", "green_2000_pct", "green_2025_pct", "built_2000_pct", "built_2025_pct",
           "green_change_pp", "green_to_built_km2", "green_to_built_pct_of_green2000", "gi_z", "gi_p", "gi_class",
           "gi_hot_fdr", "geometry"]].to_file(DASH / "hexes.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)
    cl = gpd.read_file(OUT / "analytics.gpkg", layer="dbscan_clusters").to_crs(4326)
    cl.to_file(DASH / "clusters.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)
    roads = gpd.read_file(GPKG, layer="roads", columns=["name", "highway", "road_class"])
    roads = roads[roads.road_class.isin(["highway", "major"])].to_crs(UTM)
    roads["geometry"] = roads.geometry.simplify(5)
    roads.to_crs(4326).to_file(DASH / "roads_main.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)
    corp = gpd.read_file(GPKG, layer="corporation").to_crs(4326)
    corp[["geometry"]].to_file(DASH / "corporation.geojson", driver="GeoJSON", COORDINATE_PRECISION=6)


def tables():
    copy = {"transitions.csv": "transitions.csv", "change_summary.csv": "change_summary.csv",
            "step6_rules_conversion.csv": "rules_conversion.csv",
            "step6_rules_stayed_green.csv": "rules_stayed_green.csv",
            "step6_rules_summary.csv": "rules_summary.csv",
            "step6_conversion_by_condition.csv": "conversion_by_condition.csv",
            "step6_case_studies.csv": "case_studies.csv", "step6_heat_landsat.csv": "heat_landsat.csv",
            "step6_heat_by_class_2025.csv": "heat_by_class.csv", "step6_hotspot_summary.csv": "hotspot_summary.csv",
            "step6_priority_sensitivity.csv": "priority_sensitivity.csv",
            "step5_benchmark.csv": "bench_sedona.csv", "step5_partitions.csv": "partitions_sedona.csv",
            "step5_postgis_benchmark.csv": "bench_postgis.csv", "step5_road_distance.csv": "road_distance.csv",
            "classification_accuracy.csv": "classification_accuracy.csv"}
    for src, dst in copy.items():
        if (OUT / src).exists():
            shutil.copy(OUT / src, TAB / dst)
    shutil.copy(OUT / "bigdata" / "px30.parquet", DASH / "px30.parquet")


def main():
    for d in (DASH, IMG, TAB):
        d.mkdir(parents=True, exist_ok=True)
    corp, t = overlays()
    corporation_trend(corp, t)
    wards()
    vectors()
    tables()
    size = sum(f.stat().st_size for f in DASH.rglob("*") if f.is_file()) / 1e6
    print(f"dashboard/data ready ({size:.1f} MB)")


if __name__ == "__main__":
    main()
