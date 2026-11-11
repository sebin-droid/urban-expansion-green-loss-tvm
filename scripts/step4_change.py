"""
Step 4 - Change detection: green space loss and urban expansion, 2000-2025.

4.1 Backdating. The Step 3 Landsat maps were classified year by year, so class flips between years
    are mostly classification noise (built-up stayed ~21-23 % from 2000 to 2025). A consistent series
    is built by starting from the 2025 map and stepping back one epoch at a time: a pixel keeps the
    later epoch's class unless its spectra changed (change-vector magnitude > mean + k * std), and
    only changed pixels take that year's own classification. Built-up cannot revert: a pixel that is
    not built-up later cannot be built-up earlier.
    k is calibrated: every k in K_GRID is run and the one whose built-up trend best matches GHSL
    (independent, Copernicus/JRC) is kept. The sweep is saved as a sensitivity table.
4.2 From-to transitions (km2) inside the corporation for 2000-2010, 2010-2017, 2017-2025, 2000-2025,
    and a 2000->2025 change raster.
4.3 Ward statistics for all 101 wards: built / green / tree share per epoch (30 m Landsat series),
    the 2025 shares at 10 m from Sentinel-2, change in percentage points, and green space per person.
    WorldPop has two releases that must not be compared with each other: green per person for 2000
    uses the 2000-2020 release, 2017 and 2025 use R2025A, so only 2017->2025 is a valid per-person change.
4.4 Maps: 2000->2025 change map, ward choropleths, corporation trend chart.

Inputs:  data/processed/landcover/lc_*.tif, composites/landsat_<year>.tif, reference/ref_30m.tif (Steps 2-3)
         data/processed/base.gpkg, ward_population.csv (Steps 1-2)
Outputs: data/processed/landcover_consistent/lc_landsat_<year>.tif, change_2000_2025.tif
         data/processed/backdating_k_sensitivity.csv, backdating_log.csv, built_trend_check.csv
         data/processed/transitions.csv, change_summary.csv
         data/processed/ward_change.csv, change.gpkg (layer ward_change)
         outputs/maps/step4_change_2000_2025.png, step4_ward_maps.png, step4_trend.png

Run:  .venv/Scripts/python scripts/step4_change.py
"""
import os
from functools import lru_cache

os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from rasterio.features import geometry_mask, rasterize
from scipy import ndimage

from step3_classification import (CLASSES, COMP, GPKG, LC, MAPS, OUT, REF, REFLECTANCE, UTM, read_composite,
                                  zscore)

YEARS = [2025, 2017, 2010, 2000]       # newest first: each year is backdated from the one after it
K_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]   # "changed" = change magnitude > mean + k * std
MIN_PATCH = 3                          # changed patches smaller than this (pixels) are treated as noise
BUILT, TREE = 1, 2
GREEN = [2, 3]
GHSL_EPOCH = {2000: 2000, 2010: 2010, 2017: 2015, 2025: 2025}
PERIODS = [(2000, 2010), (2010, 2017), (2017, 2025), (2000, 2025)]
PX_KM2 = 30 * 30 / 1e6
CONSISTENT = OUT / "landcover_consistent"

CHANGE_CLASSES = {1: "Built-up (stable)", 2: "Green (stable)", 3: "Green -> built-up", 4: "Other -> built-up",
                  5: "Green -> water/bare", 6: "Gained green", 7: "Water/bare (stable)"}
# stable classes muted, change classes saturated
CHANGE_COLORS = ["#9e9e9e", "#c7e9c0", "#d7301f", "#fd8d3c", "#8c6d31", "#1a7837", "#c6dbef"]


# ---------------------------------------------------------------- 4.1 backdating

@lru_cache(maxsize=None)
def change_magnitude(year, later):
    """Change-vector magnitude between two Landsat composites (both z-scored per image)."""
    a, feat, va, _ = read_composite(COMP / f"landsat_{year}.tif")
    b, _, vb, _ = read_composite(COMP / f"landsat_{later}.tif")
    keep = [feat.index(k) for k in REFLECTANCE + ["NDVI"]]
    both = va & vb
    mag = np.sqrt(((zscore(a[keep], both) - zscore(b[keep], both)) ** 2).sum(axis=0))
    return mag, both, va


def change_mask(year, later, k):
    mag, both, valid = change_magnitude(year, later)
    m = mag[both]
    changed = both & (mag > m.mean() + k * m.std())
    # drop specks: keep only connected changed patches of MIN_PATCH pixels or more
    patches, _ = ndimage.label(changed)
    sizes = np.bincount(patches.ravel())
    changed &= sizes[patches] >= MIN_PATCH
    return changed, valid


def read_own(year):
    with rasterio.open(LC / f"lc_landsat_{year}.tif") as src:
        return src.read(1)


def backdate(k):
    maps = {2025: read_own(2025)}
    log = []
    for later, year in zip(YEARS, YEARS[1:]):
        own = read_own(year)
        changed, valid = change_mask(year, later, k)
        out = maps[later].copy()                       # unchanged pixels inherit the later class
        take = changed & (own > 0)
        out[take] = own[take]                          # changed pixels take this year's own classification
        revert = (out == BUILT) & (maps[later] != BUILT) & (maps[later] > 0)
        out[revert] = maps[later][revert]              # built-up cannot turn back into something else
        maps[year] = out
        log.append({"year": year, "backdated_from": later, "change_k": k,
                    "changed_pct": round(100 * changed[valid].mean(), 2),
                    "built_reverted_px": int(revert.sum())})
    return maps, pd.DataFrame(log)


def check_trend(maps, corp_mask, ghsl):
    """Built-up share in the corporation: independent maps vs consistent maps vs GHSL."""
    rows = []
    for year in sorted(maps):
        rows.append({"year": year,
                     "built_pct_independent": round(100 * np.mean(read_own(year)[corp_mask] == BUILT), 2),
                     "built_pct_consistent": round(100 * np.mean(maps[year][corp_mask] == BUILT), 2),
                     "green_pct_consistent": round(100 * np.isin(maps[year][corp_mask], GREEN).mean(), 2),
                     "ghsl_built_surface_pct": round(float(ghsl[GHSL_EPOCH[year]][corp_mask].mean()), 2)})
    df = pd.DataFrame(rows)
    # the shapes of the two trends should match, even though the absolute levels are defined differently
    df["built_vs_2025"] = (df.built_pct_consistent / df.built_pct_consistent.iloc[-1]).round(2)
    df["ghsl_vs_2025"] = (df.ghsl_built_surface_pct / df.ghsl_built_surface_pct.iloc[-1]).round(2)
    return df


def calibrate(profile, colormap, corp_mask):
    with rasterio.open(REF / "ref_30m.tif") as src:
        ghsl = {y: src.read(list(src.descriptions).index(f"ghsl_{y}") + 1) for y in set(GHSL_EPOCH.values())}
    rows, best = [], None
    for k in K_GRID:
        maps, log = backdate(k)
        trend = check_trend(maps, corp_mask, ghsl)
        past = trend[trend.year < 2025]
        err = float(((past.built_vs_2025 - past.ghsl_vs_2025) ** 2).sum())
        rows.append({"change_k": k, "sq_error_vs_ghsl": round(err, 4),
                     **{f"built_pct_{y}": v for y, v in zip(trend.year, trend.built_pct_consistent)},
                     **{f"green_pct_{y}": v for y, v in zip(trend.year, trend.green_pct_consistent)}})
        if best is None or err < best[0]:
            best = (err, k, maps, log, trend)
    sens = pd.DataFrame(rows)
    sens.to_csv(OUT / "backdating_k_sensitivity.csv", index=False)
    print("k sensitivity (corporation, % of area):")
    print(sens.to_string(index=False))

    _, k, maps, log, trend = best
    print(f"\nChosen k = {k} (built-up trend closest to GHSL)\n")
    print(log.to_string(index=False), "\n")
    print(trend.to_string(index=False), "\n")
    log.to_csv(OUT / "backdating_log.csv", index=False)
    trend.to_csv(OUT / "built_trend_check.csv", index=False)
    CONSISTENT.mkdir(exist_ok=True)
    for year, lc in maps.items():
        write_raster(CONSISTENT / f"lc_landsat_{year}.tif", lc, profile, colormap)
    return maps


def write_raster(path, arr, profile, colormap):
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
        dst.write_colormap(1, colormap)


# ---------------------------------------------------------------- 4.2 transitions

def transitions(maps, corp_mask):
    rows, summary = [], []
    for a, b in PERIODS:
        x, y = maps[a][corp_mask], maps[b][corp_mask]
        ok = (x > 0) & (y > 0)
        counts = np.bincount(x[ok].astype(int) * 6 + y[ok], minlength=36).reshape(6, 6)
        for f in CLASSES:
            for t in CLASSES:
                rows.append({"period": f"{a}-{b}", "from_class": f, "from_name": CLASSES[f],
                             "to_class": t, "to_name": CLASSES[t], "km2": round(counts[f, t] * PX_KM2, 3)})
        gx, gy = np.isin(x, GREEN) & ok, np.isin(y, GREEN) & ok
        summary.append({"period": f"{a}-{b}", "years": b - a,
                        "green_start_km2": round(gx.sum() * PX_KM2, 2),
                        "green_end_km2": round(gy.sum() * PX_KM2, 2),
                        "green_lost_km2": round((gx & ~gy).sum() * PX_KM2, 2),
                        "green_to_built_km2": round((gx & (y == BUILT)).sum() * PX_KM2, 2),
                        "green_gained_km2": round((~gx & gy).sum() * PX_KM2, 2),
                        "built_gain_km2": round(((x != BUILT) & (y == BUILT) & ok).sum() * PX_KM2, 2)})
    s = pd.DataFrame(summary)
    s["net_green_change_km2"] = (s.green_end_km2 - s.green_start_km2).round(2)
    s["net_green_change_pct"] = (100 * s.net_green_change_km2 / s.green_start_km2).round(2)
    s["green_lost_km2_per_year"] = (s.green_lost_km2 / s.years).round(3)
    pd.DataFrame(rows).to_csv(OUT / "transitions.csv", index=False)
    s.to_csv(OUT / "change_summary.csv", index=False)
    print("Change inside the corporation (km2):")
    print(s.to_string(index=False), "\n")


def change_raster(x, y):
    gx, gy = np.isin(x, GREEN), np.isin(y, GREEN)
    bx, by = x == BUILT, y == BUILT
    vx, vy = x > 0, y > 0
    out = np.zeros(x.shape, np.uint8)
    out[bx & by] = 1
    out[gx & gy] = 2
    out[gx & by] = 3
    out[vx & ~gx & ~bx & by] = 4
    out[gx & vy & ~gy & ~by] = 5
    out[vx & ~gx & gy] = 6
    out[vx & ~gx & ~bx & vy & ~gy & ~by] = 7
    return out


# ---------------------------------------------------------------- 4.3 ward statistics

def ward_stats(maps, profile):
    wards = gpd.read_file(GPKG, layer="wards").to_crs(UTM)
    df = wards[["ward_no", "ward_name", "geometry"]].copy()
    ids = df.ward_no.to_numpy()
    n = int(ids.max()) + 1

    def add_shares(lc, transform, suffix):
        zones = rasterize(zip(wards.geometry, wards.ward_no), out_shape=lc.shape, transform=transform,
                          fill=0, dtype="int32").ravel()
        total = np.bincount(zones, weights=(lc > 0).ravel(), minlength=n)
        for name, cls in (("built", [BUILT]), ("green", GREEN), ("tree", [TREE])):
            hit = np.bincount(zones, weights=np.isin(lc, cls).ravel(), minlength=n)
            df[f"{name}_pct_{suffix}"] = (100 * hit / np.maximum(total, 1))[ids].round(2)

    for year in sorted(maps):
        add_shares(maps[year], profile["transform"], str(year))
    with rasterio.open(LC / "lc_sentinel2_2025.tif") as src:          # 10 m detail for today
        add_shares(src.read(1), src.transform, "2025_s2")

    pop = pd.read_csv(OUT / "ward_population.csv").drop(columns="ward_name")
    df = df.merge(pop, on="ward_no")
    for a, b in PERIODS:
        df[f"green_change_pp_{a}_{b}"] = (df[f"green_pct_{b}"] - df[f"green_pct_{a}"]).round(2)
        df[f"built_change_pp_{a}_{b}"] = (df[f"built_pct_{b}"] - df[f"built_pct_{a}"]).round(2)
    green_m2 = lambda year: df[f"green_pct_{year}"] / 100 * df.area_km2 * 1e6
    df["green_m2_per_person_2000"] = (green_m2(2000) / df.pop_2000).round(1)       # WorldPop 2000-2020 release
    df["green_m2_per_person_2017"] = (green_m2(2017) / df.pop_2017_r25).round(1)   # WorldPop R2025A
    df["green_m2_per_person_2025"] = (green_m2(2025) / df.pop_2025_r25).round(1)   # WorldPop R2025A
    df["green_loss_rank_2000_2025"] = df.green_change_pp_2000_2025.rank(method="min").astype(int)

    gpd.GeoDataFrame(df, crs=UTM).to_file(OUT / "change.gpkg", layer="ward_change", driver="GPKG")
    df.drop(columns="geometry").to_csv(OUT / "ward_change.csv", index=False)
    cols = ["ward_no", "ward_name", "green_pct_2000", "green_pct_2025", "green_change_pp_2000_2025",
            "built_change_pp_2000_2025", "green_pct_2025_s2", "green_m2_per_person_2025"]
    print("Wards with the largest green loss 2000-2025:")
    print(df.sort_values("green_change_pp_2000_2025")[cols].head(10).to_string(index=False), "\n")
    return gpd.GeoDataFrame(df, crs=UTM)


# ---------------------------------------------------------------- 4.4 maps and charts

def map_change(change, profile, wards, corp):
    b = rasterio.transform.array_bounds(*change.shape, profile["transform"])
    fig, ax = plt.subplots(figsize=(9, 11))
    ax.imshow(np.ma.masked_equal(change, 0), cmap=ListedColormap(CHANGE_COLORS), vmin=0.5, vmax=7.5,
              extent=[b[0], b[2], b[1], b[3]], interpolation="nearest")
    wards.boundary.plot(ax=ax, color="#555555", linewidth=0.25)
    corp.boundary.plot(ax=ax, color="black", linewidth=0.9)
    xmin, ymin, xmax, ymax = corp.total_bounds
    ax.set_xlim(xmin - 500, xmax + 500)
    ax.set_ylim(ymin - 500, ymax + 500)
    ax.set_axis_off()
    ax.legend(handles=[Patch(color=c, label=n) for c, n in zip(CHANGE_COLORS, CHANGE_CLASSES.values())],
              loc="lower left", frameon=False, fontsize=9)
    ax.set_title("Land-cover change 2000 -> 2025 (Landsat, backdated series)", fontsize=13)
    fig.tight_layout()
    fig.savefig(MAPS / "step4_change_2000_2025.png", dpi=150)
    plt.close(fig)


def map_wards(wards, corp):
    lim = np.abs(wards.green_change_pp_2000_2025).max()
    panels = [
        ("green_pct_2025_s2", "Green space 2025 (% of ward, Sentinel-2 10 m)", "Greens", None),
        ("green_change_pp_2000_2025", "Green change 2000-2025 (percentage points)", "BrBG",
         TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)),
        ("built_change_pp_2000_2025", "Built-up gain 2000-2025 (percentage points)", "Reds", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    for ax, (col, title, cmap, norm) in zip(axes, panels):
        # an explicit norm=None makes geopandas draw a 0-1 colorbar, so only pass it when set
        scale = {"norm": norm} if norm else {"vmin": 0, "vmax": wards[col].max()}
        wards.plot(column=col, ax=ax, cmap=cmap, edgecolor="white", linewidth=0.3, legend=True,
                   legend_kwds={"shrink": 0.6}, **scale)
        corp.boundary.plot(ax=ax, color="black", linewidth=0.6)
        ax.set_title(title, fontsize=11)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(MAPS / "step4_ward_maps.png", dpi=150)
    plt.close(fig)


def chart_trend():
    t = pd.read_csv(OUT / "built_trend_check.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for col, label, color in (("green_pct_consistent", "Green space", "#1a7837"),
                              ("built_pct_consistent", "Built-up", "#d7301f")):
        ax.plot(t.year, t[col], color=color, linewidth=2, marker="o", markersize=5, label=label)
        for i in (0, len(t) - 1):
            ax.annotate(f"{t[col].iloc[i]:.1f}%", (t.year.iloc[i], t[col].iloc[i]), textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=9, color="#333333")
    ax.set_ylim(0, 100)
    ax.set_xticks(t.year)
    ax.set_ylabel("% of corporation area")
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="center right")
    ax.set_title("Thiruvananthapuram corporation, Landsat backdated series", fontsize=11)
    fig.tight_layout()
    fig.savefig(MAPS / "step4_trend.png", dpi=150)
    plt.close(fig)


def main():
    with rasterio.open(LC / "lc_landsat_2025.tif") as src:
        profile, colormap = src.profile, src.colormap(1)
    corp = gpd.read_file(GPKG, layer="corporation").to_crs(UTM)
    corp_mask = geometry_mask(corp.geometry, out_shape=(profile["height"], profile["width"]),
                              transform=profile["transform"], invert=True)

    maps = calibrate(profile, colormap, corp_mask)
    transitions(maps, corp_mask)
    change = change_raster(maps[2000], maps[2025])
    write_raster(OUT / "change_2000_2025.tif", change, profile,
                 {c: tuple(int(CHANGE_COLORS[c - 1][i:i + 2], 16) for i in (1, 3, 5)) + (255,)
                  for c in CHANGE_CLASSES})
    wards = ward_stats(maps, profile)

    map_change(change, profile, wards, corp)
    map_wards(wards, corp)
    chart_trend()
    print("Saved outputs/maps/step4_change_2000_2025.png, step4_ward_maps.png, step4_trend.png")


if __name__ == "__main__":
    main()
