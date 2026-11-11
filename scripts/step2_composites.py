"""
Step 2 - Cloud-free satellite composites, spectral indices and ward population (Google Earth Engine).

For each study year a dry-season (Nov-Apr) median composite is built from all clear pixels,
then NDVI, NDBI, MNDWI and land surface temperature are added and the result is downloaded
as a GeoTIFF clipped to the study area (corporation + 2 km), in UTM 43N.

  Landsat (30 m):     2000, 2010  -> Landsat 5 TM + Landsat 7 ETM+ (two dry seasons, too few scenes in one)
                      2017        -> Landsat 8 OLI
                      2025        -> Landsat 8 + 9 OLI
  Sentinel-2 (10 m):  2019, 2025  -> Sentinel-2 L2A, Cloud Score+ masking
                                     (2017 has only 3 L2A scenes over the city, so 2019 is the first usable year)
  Population:         WorldPop 100 m, summed per ward for 2000, 2010, 2020 (and 2025 from the R2025A release)

Inputs:  data/processed/base.gpkg (from Step 1)
Outputs: data/processed/composites/landsat_<year>.tif, sentinel2_<year>.tif
         data/processed/composites_summary.csv
         data/processed/ward_population.csv
         outputs/maps/step2_preview.png

Run:  .venv/Scripts/python scripts/step2_composites.py [landsat|sentinel|population|preview]  (default: all)
"""
import os
import sys
from pathlib import Path

# PostgreSQL/PostGIS sets PROJ_LIB and GDAL_DATA system-wide to its own older copies, which break
# rasterio; dropping them lets rasterio and pyproj use the data files bundled with their wheels.
os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import ee
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio

import geedim  # noqa: F401  (registers the .gd accessor on ee.Image)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "processed"
COMP = OUT / "composites"
MAPS = ROOT / "outputs" / "maps"
GPKG = OUT / "base.gpkg"

EE_PROJECT = "geometric-wall-477105-i2"
UTM = "EPSG:32643"
DRY_MONTHS = (11, 4)                                   # Nov-Apr; calendarRange wraps across the year end
S2_CLEAR_THRESHOLD = 0.60                              # Cloud Score+ cs_cdf; higher = stricter

LANDSAT_EPOCHS = {
    2000: ("1999-11-01", "2001-04-30", ["LT05", "LE07"]),
    2010: ("2009-11-01", "2011-04-30", ["LT05", "LE07"]),
    2017: ("2016-11-01", "2017-04-30", ["LC08"]),
    2025: ("2024-11-01", "2025-04-30", ["LC08", "LC09"]),
}
SENTINEL_EPOCHS = {
    2019: ("2018-11-01", "2019-04-30"),
    2025: ("2024-11-01", "2025-04-30"),
}

# Landsat Collection 2 band names -> common names
LANDSAT_BANDS = {
    "LT05": (["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"], "ST_B6"),
    "LE07": (["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"], "ST_B6"),
    "LC08": (["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], "ST_B10"),
    "LC09": (["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"], "ST_B10"),
}
COMMON = ["blue", "green", "red", "nir", "swir1", "swir2"]


def study_geometry():
    """Study area (corporation + 2 km) as an ee.Geometry, simplified to keep requests small."""
    study = gpd.read_file(GPKG, layer="study_area").to_crs("EPSG:4326")
    return ee.Geometry(study.geometry.simplify(0.0002).iloc[0].__geo_interface__)


def dry_season():
    return ee.Filter.calendarRange(*DRY_MONTHS, "month")


def add_indices(img):
    ndvi = img.normalizedDifference(["nir", "red"]).rename("NDVI")      # vegetation
    ndbi = img.normalizedDifference(["swir1", "nir"]).rename("NDBI")    # built-up
    mndwi = img.normalizedDifference(["green", "swir1"]).rename("MNDWI")  # open water
    return img.addBands([ndvi, ndbi, mndwi])


# ---------------------------------------------------------------- Landsat

def prep_landsat(sensor):
    optical, thermal = LANDSAT_BANDS[sensor]

    def prep(img):
        qa = img.select("QA_PIXEL")
        # bits: 1 dilated cloud, 2 cirrus, 3 cloud, 4 cloud shadow
        clear = (qa.bitwiseAnd(1 << 1).eq(0)
                 .And(qa.bitwiseAnd(1 << 2).eq(0))
                 .And(qa.bitwiseAnd(1 << 3).eq(0))
                 .And(qa.bitwiseAnd(1 << 4).eq(0))
                 .And(img.select("QA_RADSAT").eq(0)))
        sr = img.select(optical, COMMON).multiply(0.0000275).add(-0.2)       # surface reflectance
        lst = img.select(thermal).multiply(0.00341802).add(149.0).subtract(273.15).rename("LST_C")
        return sr.addBands(lst).updateMask(clear).copyProperties(img, ["system:time_start"])

    return prep


def landsat_composite(year, region):
    start, end, sensors = LANDSAT_EPOCHS[year]
    merged = None
    for s in sensors:
        col = (ee.ImageCollection(f"LANDSAT/{s}/C02/T1_L2")
               .filterBounds(region).filterDate(start, end).filter(dry_season())
               .map(prep_landsat(s)))
        merged = col if merged is None else merged.merge(col)
    n_scenes = merged.size().getInfo()
    clear_obs = merged.select("red").count().rename("clear_obs").toFloat()
    img = add_indices(merged.median()).addBands(clear_obs).clip(region)
    return img, n_scenes, ", ".join(sensors)


# ---------------------------------------------------------------- Sentinel-2

def sentinel_composite(year, region):
    start, end = SENTINEL_EPOCHS[year]
    cs = ee.ImageCollection("GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED")
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterBounds(region).filterDate(start, end).filter(dry_season())
           .linkCollection(cs, ["cs_cdf"]))
    n_scenes = col.size().getInfo()

    def prep(img):
        # bilinear here, on each scene, so the 20 m SWIR bands are smoothed onto the 10 m grid
        sr = img.select(["B2", "B3", "B4", "B8", "B11", "B12"], COMMON).divide(10000).resample("bilinear")
        return sr.updateMask(img.select("cs_cdf").gte(S2_CLEAR_THRESHOLD))

    col = col.map(prep)
    clear_obs = col.select("red").count().rename("clear_obs").toFloat()
    img = add_indices(col.median()).addBands(clear_obs).clip(region)
    return img, n_scenes, "S2A/S2B L2A"


# ---------------------------------------------------------------- export

def download(img, path, region, scale):
    if path.exists():
        print(f"  {path.name} already exists, skipping download")
        return
    # A composite has no native grid (EE falls back to a 1-degree default), so pin it to the
    # target UTM grid and export with nearest-neighbour; resampling it would interpolate that coarse grid.
    (img.toFloat()
        .setDefaultProjection(crs=UTM, scale=scale)
        .gd.prepareForExport(crs=UTM, region=region, scale=scale, resampling="near", dtype="float32")
        .gd.toGeoTIFF(path))


def raster_stats(path, n_scenes, sensors, year):
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        obs = src.read(names.index("clear_obs") + 1, masked=True)
        ndvi = src.read(names.index("NDVI") + 1, masked=True)
        # share of study-area pixels that got at least one clear observation
        valid = np.count_nonzero(~np.ma.getmaskarray(ndvi)) / np.count_nonzero(~np.ma.getmaskarray(obs))
        return {"file": path.name, "year": year, "sensors": sensors, "scenes": n_scenes,
                "width": src.width, "height": src.height, "res_m": src.res[0],
                "median_clear_obs": float(np.ma.median(obs)),
                "pct_pixels_with_data": round(100 * float(valid), 1),
                "mean_ndvi": round(float(ndvi.mean()), 3)}


def run_composites(kind):
    COMP.mkdir(parents=True, exist_ok=True)
    region = study_geometry()
    rows = []
    epochs = LANDSAT_EPOCHS if kind == "landsat" else SENTINEL_EPOCHS
    for year in epochs:
        if kind == "landsat":
            img, n, sensors = landsat_composite(year, region)
            path, scale = COMP / f"landsat_{year}.tif", 30
        else:
            img, n, sensors = sentinel_composite(year, region)
            path, scale = COMP / f"sentinel2_{year}.tif", 10
        print(f"{kind} {year}: {n} scenes ({sensors}) -> downloading {path.name}")
        download(img, path, region, scale)
        rows.append(raster_stats(path, n, sensors, year))

    summary = OUT / "composites_summary.csv"
    new = pd.DataFrame(rows)
    if summary.exists():
        old = pd.read_csv(summary)
        new = pd.concat([old[~old.file.isin(new.file)], new]).sort_values("file")
    new.to_csv(summary, index=False)
    print(new.to_string(index=False))


# ---------------------------------------------------------------- population

def run_population():
    wards = gpd.read_file(GPKG, layer="wards").to_crs("EPSG:4326")
    wards["geometry"] = wards.geometry.simplify(0.00005)
    fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry(g.__geo_interface__), {"ward_no": int(n)})
        for n, g in zip(wards.ward_no, wards.geometry)
    ])

    # The two WorldPop releases use different models, so never compare a year from one with the other:
    #   original release (2000-2020)  -> pop_2000, pop_2010, pop_2020
    #   R2025A release (2015-2030)    -> pop_2017_r25, pop_2025_r25
    releases = {
        "WorldPop/GP/100m/pop": {2000: "pop_2000", 2010: "pop_2010", 2020: "pop_2020"},
        "WorldPop/GP/100m/R2025A/v1/pop": {2017: "pop_2017_r25", 2025: "pop_2025_r25"},
    }
    bands = []
    for asset, years in releases.items():
        col = ee.ImageCollection(asset).filter(ee.Filter.eq("country", "IND"))
        for year, name in years.items():
            bands.append(col.filter(ee.Filter.eq("year", year)).first().select(0).rename(name))
    names = [n for years in releases.values() for n in years.values()]

    res = ee.Image.cat(bands).reduceRegions(collection=fc, reducer=ee.Reducer.sum(), scale=100).getInfo()

    df = pd.DataFrame([f["properties"] for f in res["features"]])
    w = gpd.read_file(GPKG, layer="wards")[["ward_no", "ward_name", "area_km2"]]
    df = w.merge(df, on="ward_no")
    df[names] = df[names].round().astype(int)
    df["density_2025_per_km2"] = (df["pop_2025_r25"] / df["area_km2"]).round()
    df["growth_2000_2020_pct"] = (100 * (df["pop_2020"] / df["pop_2000"] - 1)).round(1)
    df["growth_2017_2025_pct"] = (100 * (df["pop_2025_r25"] / df["pop_2017_r25"] - 1)).round(1)
    df.to_csv(OUT / "ward_population.csv", index=False)
    print("Corporation population  " + "  ".join(f"{n}: {df[n].sum():,}" for n in names))
    print("Saved data/processed/ward_population.csv")


# ---------------------------------------------------------------- preview

def stretch(rgb):
    lo, hi = np.nanpercentile(rgb, [2, 98])
    return np.clip((rgb - lo) / (hi - lo), 0, 1)


def run_preview():
    files = [COMP / "landsat_2000.tif", COMP / "landsat_2025.tif", COMP / "sentinel2_2025.tif"]
    files = [f for f in files if f.exists()]
    corp = gpd.read_file(GPKG, layer="corporation")
    fig, axes = plt.subplots(2, len(files), figsize=(5 * len(files), 11))
    axes = np.atleast_2d(axes).reshape(2, len(files))
    for i, f in enumerate(files):
        with rasterio.open(f) as src:
            names = list(src.descriptions)
            band = lambda n: src.read(names.index(n) + 1, masked=True).astype(float).filled(np.nan)
            rgb = np.dstack([band("red"), band("green"), band("blue")])
            ndvi = band("NDVI")
            ext = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        axes[0, i].imshow(stretch(rgb), extent=ext)
        axes[1, i].imshow(ndvi, extent=ext, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
        for ax in axes[:, i]:
            corp.boundary.plot(ax=ax, color="white" if ax is axes[0, i] else "black", linewidth=0.8)
            ax.set_axis_off()
        axes[0, i].set_title(f"{f.stem} - true colour")
        axes[1, i].set_title(f"{f.stem} - NDVI")
    fig.tight_layout()
    fig.savefig(MAPS / "step2_preview.png", dpi=150)
    print("Saved outputs/maps/step2_preview.png")


def main():
    ee.Initialize(project=EE_PROJECT)
    tasks = sys.argv[1:] or ["landsat", "sentinel", "population", "preview"]
    if "landsat" in tasks:
        run_composites("landsat")
    if "sentinel" in tasks:
        run_composites("sentinel")
    if "population" in tasks:
        run_population()
    if "preview" in tasks:
        run_preview()


if __name__ == "__main__":
    main()
