"""
Step 3 - Random Forest land-cover classification of every composite from Step 2.

Classes:  1 Built-up   2 Tree cover   3 Other vegetation (grass, crops, paddy, scrub, wetland)
          4 Water      5 Bare / sand
          Green space = classes 2 + 3.

Training labels are generated automatically, in two levels.

Sentinel-2 (10 m, 2019 and 2025) - from reference products, filtered hard so only clean pixels survive:
  1. ESA WorldCover 2021 label that AGREES with Dynamic World (dry-season mode) for the epoch year.
  2. GHSL built-up surface for the epoch must back it up: built-up >= 20 % built, other classes <= 10 %.
  3. At least 7 of the 9 pixels in the 3x3 window carry the same label (no mixed edge pixels).
  4. The pixel's spectra must fit the class (NDVI / MNDWI rules).

Landsat (30 m, 2000/2010/2017/2025) - from the Sentinel-2 2025 map, so both resolutions share one
built-up definition (TVM's built-up is scattered under coconut canopy; trained on reference products
alone, a 30 m model only calls the dense core built-up):
  1. Each 30 m cell takes the 10 m class that covers at least half of it.
  2. Built-up must be backed by GHSL (>= 20 %) in that epoch.
  3. Years before 2025: the cell must be spectrally stable versus Landsat 2025 (change-vector
     magnitude below the class's 75th percentile), i.e. labels migrate back only where little changed.

Land surface temperature is not used as a feature: it is missing over the north-west of the study area.

One Random Forest is trained per composite. Accuracy is measured on a spatially separate test set
(1 km blocks held out), then the model is refit on all samples and the whole raster is predicted and
cleaned with a 3x3 majority filter.

Inputs:  data/processed/composites/*.tif (Step 2), data/processed/base.gpkg (Step 1)
Outputs: data/processed/reference/ref_30m.tif, ref_10m.tif          (reference stacks from GEE)
         data/processed/landcover/lc_<composite>.tif                 (uint8, 0 = nodata)
         data/processed/training_samples.csv                         (x, y, class per composite - reused in Step 5)
         data/processed/classification_accuracy.csv, confusion_matrices.csv, feature_importance.csv
         data/processed/landcover_area.csv                           (km2 per class, corporation and study area)
         data/processed/landcover/validation_points.gpkg            (for optional manual checking)
         outputs/maps/step3_landcover.png

Run:  .venv/Scripts/python scripts/step3_classification.py [reference|classify|maps|validate]  (default: first three)
"""
import os
import sys
from pathlib import Path

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from step2_composites import (COMP, EE_PROJECT, GPKG, LANDSAT_EPOCHS, MAPS, OUT, SENTINEL_EPOCHS, UTM,
                              dry_season, study_geometry)

REF = OUT / "reference"
LC = OUT / "landcover"
SEED = 42

CLASSES = {1: "Built-up", 2: "Tree cover", 3: "Other vegetation", 4: "Water", 5: "Bare/sand"}
COLORS = ["#d7301f", "#1a7837", "#a6d96a", "#2c7fb8", "#e8d8a8"]

# ESA WorldCover v200 -> our classes
WC_MAP = {10: 2, 20: 3, 30: 3, 40: 3, 50: 1, 60: 5, 80: 4, 90: 3, 95: 2, 100: 3}
# Dynamic World label (stored +1 so 0 can be nodata): water, trees, grass, flooded veg, crops, shrub, built, bare, snow
DW_MAP = {1: 4, 2: 2, 3: 3, 4: 3, 5: 3, 6: 3, 7: 1, 8: 5}

# (composite, year, Dynamic World year used for label agreement, GHSL epoch)
# Sentinel-2 runs first: its 2025 map is the training reference for the Landsat years.
EPOCHS = [
    ("sentinel2", 2019, 2019, 2020),
    ("sentinel2", 2025, 2025, 2025),
    ("landsat", 2000, 2025, 2000),
    ("landsat", 2010, 2025, 2010),
    ("landsat", 2017, 2017, 2015),
    ("landsat", 2025, 2025, 2025),
]
DW_YEARS = [2017, 2019, 2025]
GHSL_YEARS = [2000, 2010, 2015, 2020, 2025]
REF_BANDS = ["worldcover"] + [f"dw_{y}" for y in DW_YEARS] + [f"ghsl_{y}" for y in GHSL_YEARS]

REFLECTANCE = ["blue", "green", "red", "nir", "swir1", "swir2"]
SAMPLES_PER_CLASS = 2000
MIN_SAME_NEIGHBOURS = 7             # of the 9 pixels in the 3x3 window
S2_MIN_SHARE = 0.5                  # Landsat label = Sentinel-2 class covering at least this share of the 30 m cell
STABLE_PERCENTILE = 75             # 2000/2010: keep pixels whose change vs 2025 is below this class percentile
BLOCK_M = 1000                      # spatial block size for the train/test split
TEST_SHARE = 0.3


# ---------------------------------------------------------------- reference stacks (GEE)

def reference_image(region):
    import ee
    wc = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map").rename("worldcover")
    bands = [wc]
    seasons = {**{y: v[:2] for y, v in LANDSAT_EPOCHS.items()}, **SENTINEL_EPOCHS}
    for y in DW_YEARS:
        start, end = seasons[y]
        dw =(ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(region)
              .filterDate(start, end).filter(dry_season()).select("label").mode())
        bands.append(dw.add(1).unmask(0).rename(f"dw_{y}"))
    for y in GHSL_YEARS:
        built = ee.Image(f"JRC/GHSL/P2023A/GHS_BUILT_S/{y}").select("built_surface")
        bands.append(built.divide(100).unmask(0).rename(f"ghsl_{y}"))     # m2 per 100 m cell -> % built
    return ee.Image.cat(bands).toUint8().clip(region)


def run_reference():
    import ee
    ee.Initialize(project=EE_PROJECT)
    REF.mkdir(parents=True, exist_ok=True)
    region = study_geometry()
    img = reference_image(region)
    for scale in (30, 10):
        path = REF / f"ref_{scale}m.tif"
        if path.exists():
            print(f"  {path.name} already exists, skipping download")
            continue
        print(f"downloading {path.name}")
        (img.setDefaultProjection(crs=UTM, scale=scale)
            .gd.prepareForExport(crs=UTM, region=region, scale=scale, resampling="near", dtype="uint8")
            .gd.toGeoTIFF(path))


def read_reference(scale, like):
    """Reference stack as {band: array}, checked to sit on exactly the composite's grid."""
    with rasterio.open(REF / f"ref_{scale}m.tif") as src:
        if src.transform != like.transform or src.shape != like.shape:
            raise ValueError(f"ref_{scale}m.tif grid {src.transform} {src.shape} does not match "
                             f"composite grid {like.transform} {like.shape}")
        return {name: src.read(i + 1) for i, name in enumerate(src.descriptions)}


# ---------------------------------------------------------------- labels

def remap(arr, table):
    lut = np.zeros(256, dtype=np.uint8)
    for k, v in table.items():
        lut[k] = v
    return lut[arr]


def read_composite(path):
    """Feature stack, feature names, valid-pixel mask and raster profile of a composite.

    LST is left out of the features: Collection-2 surface temperature is missing over ~20 % of the
    study area (the north-west) in every Landsat year, while reflectance there is complete.
    """
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        data = src.read().astype("float32")
        profile = src.profile
    valid = np.all(np.isfinite(data[[names.index(b) for b in REFLECTANCE]]), axis=0)
    features = [n for n in names if n not in ("clear_obs", "LST_C")]
    stack = np.nan_to_num(data[[names.index(n) for n in features]], nan=0, posinf=0, neginf=0)
    return stack, features, valid, profile


def spectral_ok(cls, f):
    """Loose per-class spectral rules on the composite itself (f: band name -> array)."""
    ndvi, mndwi = f["NDVI"], f["MNDWI"]
    rules = {
        1: (ndvi < 0.35) & (mndwi < 0.0),
        2: ndvi > 0.55,
        3: ndvi > 0.30,
        4: (mndwi > 0.0) & (ndvi < 0.2),
        5: (ndvi < 0.25) & (mndwi < 0.2),
    }
    return rules[cls]


def zscore(stack, valid):
    out = np.empty_like(stack)
    for i, b in enumerate(stack):
        v = b[valid]
        out[i] = (b - v.mean()) / v.std()
    return out


def change_magnitude(stack, features, valid):
    """Change-vector magnitude between this Landsat composite and Landsat 2025 (z-scored per image)."""
    ref, ref_feat, ref_valid, _ = read_composite(COMP / "landsat_2025.tif")
    keep = REFLECTANCE + ["NDVI"]
    a =stack[[features.index(k) for k in keep]]
    b = ref[[ref_feat.index(k) for k in keep]]
    both = valid & ref_valid
    mag = np.sqrt(((zscore(a, both) - zscore(b, both)) ** 2).sum(axis=0))
    mag[~both] = np.inf
    return mag


def make_labels(kind, year, dw_year, ghsl_year, stack, features, valid, ref):
    wc = remap(ref["worldcover"], WC_MAP)
    dw = remap(ref[f"dw_{dw_year}"], DW_MAP)
    label = np.where((wc == dw) & (wc > 0), wc, 0).astype(np.uint8)

    # 3x3 homogeneity: most of the neighbourhood must carry the centre pixel's label
    same = np.zeros(label.shape, np.float32)
    for c in CLASSES:
        same = np.where(label == c, ndimage.uniform_filter((label == c).astype(np.float32), 3) * 9, same)
    label[same < MIN_SAME_NEIGHBOURS - 0.01] = 0

    ghsl = ref[f"ghsl_{ghsl_year}"]
    f = {n: stack[i] for i, n in enumerate(features)}
    keep = np.zeros(label.shape, bool)
    for c in CLASSES:
        ghsl_ok = ghsl >= 20 if c == 1 else ghsl <= 10
        keep |= (label == c) & ghsl_ok & spectral_ok(c, f)
    label[~(keep & valid)] = 0

    return label


def s2_labels(year, ghsl_year, stack, features, valid, ref, profile):
    """Landsat labels from the Sentinel-2 2025 map: each 30 m cell takes the 10 m class covering at least
    half of it. Reference-product labels alone keep only the dense core as built-up (mixed homestead
    cells end up as trees), so the 30 m maps would use a different built-up definition than the 10 m ones."""
    with rasterio.open(LC / "lc_sentinel2_2025.tif") as src:
        s2, s2_transform, s2_crs = src.read(1), src.transform, src.crs
    share = np.zeros((len(CLASSES),) + valid.shape, np.float32)
    for i, c in enumerate(CLASSES):
        reproject((s2 == c).astype(np.float32), share[i], src_transform=s2_transform, src_crs=s2_crs,
                  dst_transform=profile["transform"], dst_crs=profile["crs"], resampling=Resampling.average)
    label = (np.argmax(share, axis=0) + 1).astype(np.uint8)
    label[(share.max(axis=0) < S2_MIN_SHARE) | ~valid] = 0
    label[(label == 1) & (ref[f"ghsl_{ghsl_year}"] < 20)] = 0      # GHSL must back built-up in that epoch

    if year < 2025:
        # migrate back in time only through pixels whose spectra barely changed since 2025
        mag = change_magnitude(stack, features, valid)
        for c in CLASSES:
            m = label == c
            if m.any():
                label[m & (mag > np.percentile(mag[m], STABLE_PERCENTILE))] = 0
    return label


def sample(label, res, rng):
    """Up to SAMPLES_PER_CLASS random labelled pixels per class, with a 1 km spatial block id each."""
    rows, cols, ys = [], [], []
    for c in CLASSES:
        r, k = np.nonzero(label == c)
        if len(r) < 50:
            print(f"    warning: only {len(r)} clean pixels for {CLASSES[c]}")
        idx = rng.choice(len(r), size=min(SAMPLES_PER_CLASS, len(r)), replace=False)
        rows.append(r[idx]); cols.append(k[idx]); ys.append(np.full(len(idx), c, np.uint8))
    rows, cols, y = map(np.concatenate, (rows, cols, ys))
    step = BLOCK_M // int(res)
    block = (rows // step) * 10_000 + cols // step
    return rows, cols, y, block


# ---------------------------------------------------------------- classification

def majority_filter(lc, valid):
    counts = np.stack([ndimage.uniform_filter((lc == c).astype("float32"), 3) for c in CLASSES])
    counts += 0.01 * np.stack([(lc == c) for c in CLASSES])      # ties keep the original class
    out = (np.argmax(counts, axis=0) + 1).astype(np.uint8)
    out[~valid] = 0
    return out


def classify_one(kind, year, dw_year, ghsl_year, corp, rng):
    name = f"{kind}_{year}"
    stack, features, valid, profile = read_composite(COMP / f"{name}.tif")
    res = profile["transform"].a
    with rasterio.open(COMP / f"{name}.tif") as like:
        ref = read_reference(int(res), like)
    corp_mask = geometry_mask(corp.geometry, out_shape=valid.shape, transform=profile["transform"], invert=True)

    if kind == "sentinel2":
        label = make_labels(kind, year, dw_year, ghsl_year, stack, features, valid, ref)
    else:
        label = s2_labels(year, ghsl_year, stack, features, valid, ref, profile)
    rows, cols, y, block = sample(label, res, rng)
    X = stack[:, rows, cols].T

    blocks = np.unique(block)
    test_blocks = rng.choice(blocks, size=int(TEST_SHARE * len(blocks)), replace=False)
    test = np.isin(block, test_blocks)

    rf = RandomForestClassifier(n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=SEED)
    rf.fit(X[~test], y[~test])
    pred = rf.predict(X[test])
    labels = list(CLASSES)
    f1 = f1_score(y[test], pred, labels=labels, average=None, zero_division=0)
    acc = {"composite": name, "year": year, "n_train": int((~test).sum()), "n_test": int(test.sum()),
           "overall_accuracy": round(accuracy_score(y[test], pred), 4),
           "kappa": round(cohen_kappa_score(y[test], pred), 4),
           **{f"f1_{CLASSES[c]}": round(v, 3) for c, v in zip(labels, f1)}}
    # independent sanity check on the built-up trend, inside the corporation
    acc[f"dw_{dw_year}_built_pct"] = round(100 * float(np.mean(ref[f"dw_{dw_year}"][corp_mask] == 7)), 2)
    acc[f"ghsl_{ghsl_year}_built_surface_pct"] = round(float(np.mean(ref[f"ghsl_{ghsl_year}"][corp_mask])), 2)
    cm = pd.DataFrame(confusion_matrix(y[test], pred, labels=labels),
                      index=[CLASSES[c] for c in labels], columns=[CLASSES[c] for c in labels])
    cm.index.name = "reference \\ predicted"
    print(f"  {name}: OA {acc['overall_accuracy']:.3f}  kappa {acc['kappa']:.3f}  "
          f"(train {acc['n_train']}, test {acc['n_test']})")

    # refit on every sample for the final map
    rf.fit(X, y)
    flat = stack.reshape(len(features), -1).T
    lc = np.zeros(flat.shape[0], np.uint8)
    idx = np.flatnonzero(valid.ravel())
    for s in range(0, len(idx), 500_000):
        chunk = idx[s:s + 500_000]
        lc[chunk] = rf.predict(flat[chunk])
    lc = majority_filter(lc.reshape(valid.shape), valid)

    LC.mkdir(parents=True, exist_ok=True)
    profile.update(count=1, dtype="uint8", nodata=0, compress="deflate")
    with rasterio.open(LC / f"lc_{name}.tif", "w", **profile) as dst:
        dst.write(lc, 1)
        dst.write_colormap(1, {c: tuple(int(COLORS[c - 1][i:i + 2], 16) for i in (1, 3, 5)) + (255,)
                               for c in CLASSES})
        dst.set_band_description(1, "landcover")

    xs, ys = rasterio.transform.xy(profile["transform"], rows, cols)
    samples = pd.DataFrame({"composite": name, "year": year, "x": xs, "y": ys, "class": y,
                            "split": np.where(test, "test", "train")})
    importance = pd.DataFrame({"composite": name, "feature": features,
                               "importance": rf.feature_importances_.round(4)})
    return acc, cm, samples, importance, lc, profile, corp_mask


def area_table(name, year, lc, profile, corp_mask):
    px_km2 = profile["transform"].a ** 2 / 1e6
    rows = []
    for c, cname in CLASSES.items():
        rows.append({"composite": name, "year": year, "class": c, "class_name": cname,
                     "study_area_km2": round(np.count_nonzero(lc == c) * px_km2, 3),
                     "corporation_km2": round(np.count_nonzero((lc == c) & corp_mask) * px_km2, 3)})
    df = pd.DataFrame(rows)
    df["corporation_pct"] = (100 * df.corporation_km2 / df.corporation_km2.sum()).round(2)
    return df


def run_classify():
    rng = np.random.default_rng(SEED)
    corp = gpd.read_file(GPKG, layer="corporation").to_crs(UTM)
    accs, cms, samples, imps, areas = [], [], [], [], []
    for kind, year, dw_year, ghsl_year in EPOCHS:
        acc, cm, s, imp, lc, profile, corp_mask = classify_one(kind, year, dw_year, ghsl_year, corp, rng)
        accs.append(acc); samples.append(s); imps.append(imp)
        cms.append(cm.assign(composite=acc["composite"]).reset_index())
        areas.append(area_table(acc["composite"], year, lc, profile, corp_mask))

    pd.DataFrame(accs).to_csv(OUT / "classification_accuracy.csv", index=False)
    pd.concat(cms).to_csv(OUT / "confusion_matrices.csv", index=False)
    pd.concat(samples).to_csv(OUT / "training_samples.csv", index=False)
    pd.concat(imps).to_csv(OUT / "feature_importance.csv", index=False)
    area = pd.concat(areas)
    area.to_csv(OUT / "landcover_area.csv", index=False)

    print("\nCorporation land cover (% of area):")
    print(area.pivot(index="composite", columns="class_name", values="corporation_pct")
          [list(CLASSES.values())].to_string())
    make_validation_points(rng)


# ---------------------------------------------------------------- independent validation (optional, manual)

def make_validation_points(rng, per_class=50):
    """Stratified random points on the two 2025 maps; fill `ref_class` by eye (Google Earth / QGIS)
    and run `validate` for an accuracy that does not depend on the reference products."""
    path = LC / "validation_points.gpkg"
    if path.exists():
        return
    frames = []
    for name in ("landsat_2025", "sentinel2_2025"):
        with rasterio.open(LC / f"lc_{name}.tif") as src:
            lc = src.read(1)
            for c in CLASSES:
                r, k = np.nonzero(lc == c)
                idx = rng.choice(len(r), size=min(per_class, len(r)), replace=False)
                xs, ys = rasterio.transform.xy(src.transform, r[idx], k[idx])
                frames.append(gpd.GeoDataFrame({"composite": name, "map_class": c, "ref_class": 0,
                                                "geometry": gpd.points_from_xy(xs, ys)}, crs=UTM))
    pts = pd.concat(frames, ignore_index=True)
    pts = pts.sample(frac=1, random_state=SEED).reset_index(drop=True)    # shuffle so the class isn't obvious
    pts.insert(0, "point_id", range(1, len(pts) + 1))
    pts.to_file(path, layer="points", driver="GPKG")
    print(f"Saved {len(pts)} validation points to {path.relative_to(OUT.parents[1])}")


def run_validate():
    pts = gpd.read_file(LC / "validation_points.gpkg")
    done = pts[pts.ref_class > 0]
    if done.empty:
        print("No points labelled yet - fill ref_class (1-5) in validation_points.gpkg first.")
        return
    for name, g in done.groupby("composite"):
        print(f"{name}: {len(g)} points  OA {accuracy_score(g.ref_class, g.map_class):.3f}  "
              f"kappa {cohen_kappa_score(g.ref_class, g.map_class):.3f}")


# ---------------------------------------------------------------- maps

def run_maps():
    names = [f"{k}_{y}" for k, y, _, _ in sorted(EPOCHS)]          # Landsat row first, then Sentinel-2
    corp = gpd.read_file(GPKG, layer="corporation").to_crs(UTM)
    area = pd.read_csv(OUT / "landcover_area.csv")
    acc = pd.read_csv(OUT / "classification_accuracy.csv").set_index("composite")
    cmap = ListedColormap(["none"] + COLORS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 13))
    for ax, name in zip(axes.ravel(), names):
        with rasterio.open(LC / f"lc_{name}.tif") as src:
            lc = src.read(1)
            ext = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
        ax.imshow(np.ma.masked_equal(lc, 0), cmap=cmap, vmin=0, vmax=5, extent=ext, interpolation="nearest")
        corp.boundary.plot(ax=ax, color="black", linewidth=0.8)
        green = area[(area.composite == name) & area["class"].isin([2, 3])].corporation_pct.sum()
        built = area[(area.composite == name) & (area["class"] == 1)].corporation_pct.sum()
        ax.set_title(f"{name}\nbuilt {built:.1f}%  green {green:.1f}%  (OA {acc.loc[name, 'overall_accuracy']:.2f})")
        ax.set_axis_off()
    fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(COLORS, CLASSES.values())],
               loc="lower center", ncol=5, frameon=False, fontsize=11)
    fig.suptitle("Random Forest land cover, Thiruvananthapuram (shares within corporation boundary)", fontsize=14)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(MAPS / "step3_landcover.png", dpi=150)
    print("Saved outputs/maps/step3_landcover.png")


def main():
    tasks = sys.argv[1:] or ["reference", "classify", "maps"]
    if "reference" in tasks:
        run_reference()
    if "classify" in tasks:
        run_classify()
    if "maps" in tasks:
        run_maps()
    if "validate" in tasks:
        run_validate()


if __name__ == "__main__":
    main()
