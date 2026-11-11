"""
Step 1 - Study area and base data for Thiruvananthapuram Corporation.

Inputs (data/raw):
  ksmart_wards/Kerala_ward_boundaries.geojson   K-SMART ward map (2024 delimitation), via github.com/madpk/kerala-ward-map
  osm/southern-zone-latest.osm.pbf              Geofabrik OSM extract for South India

Outputs:
  data/processed/base.gpkg   layers: wards, corporation, study_area, roads, buildings, landuse
  data/processed/corporation_boundary.geojson   (WGS84, for uploading to Google Earth Engine)
  outputs/maps/step1_base_map.png

Run:  .venv/Scripts/python scripts/step1_base_data.py
"""
import os
from pathlib import Path

# PostgreSQL/PostGIS sets PROJ_LIB and GDAL_DATA system-wide to its own older copies; ignore them
os.environ.pop("PROJ_LIB", None)
os.environ.pop("GDAL_DATA", None)

import geopandas as gpd
import matplotlib.pyplot as plt
import pyogrio

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
MAPS = ROOT / "outputs" / "maps"
GPKG = OUT / "base.gpkg"

CORP_CODE = "C010100"   # Thiruvananthapuram Corporation in K-SMART
UTM = "EPSG:32643"      # WGS 84 / UTM zone 43N - metres, used for all area and distance work
BUFFER_M = 2000         # study area = corporation + 2 km, so edge growth is not cut off

ROAD_CLASSES = {
    "motorway": "highway", "trunk": "highway", "primary": "major",
    "secondary": "major", "tertiary": "minor", "unclassified": "minor",
    "residential": "local", "living_street": "local", "service": "local",
}


def load_wards():
    """Read Kerala wards, keep only the 101 Corporation wards and tidy the columns."""
    w = gpd.read_file(RAW / "ksmart_wards" / "Kerala_ward_boundaries.geojson")
    w = w[w["lb_code"] == CORP_CODE]
    # Drop surveyor names and phone numbers (personal data) and duplicate fields
    w = w.rename(columns={"WardNumber": "ward_no", "Ward Eng": "ward_name", "WardMal": "ward_name_ml"})
    w = w[["ward_no", "ward_name", "ward_name_ml", "geometry"]].copy()
    w["ward_no"] = w["ward_no"].astype(int)
    w["ward_name"] = w["ward_name"].str.strip()
    w = w.sort_values("ward_no").reset_index(drop=True)

    w = w.to_crs(UTM)
    w["geometry"] = w.geometry.make_valid()
    w["area_km2"] = (w.area / 1e6).round(3)
    assert len(w) == 101 and w["ward_no"].is_unique, "expected 101 unique wards"
    return w


def build_boundaries(wards):
    corp = wards.dissolve()[["geometry"]]
    # Close the hairline gaps between neighbouring ward polygons so the outline has no slivers
    corp["geometry"] = corp.buffer(10).buffer(-10)
    corp["name"] = "Thiruvananthapuram Corporation"
    corp["area_km2"] = round(corp.area.iloc[0] / 1e6, 2)

    study = corp.copy()
    study["geometry"] = corp.buffer(BUFFER_M)
    study["name"] = f"Study area (corporation + {BUFFER_M // 1000} km)"
    study["area_km2"] = round(study.area.iloc[0] / 1e6, 2)
    return corp, study


def read_osm(layer, where, study, columns):
    """Read one layer of the OSM PBF, limited to the study area bbox, then clip to it."""
    pbf = RAW / "osm" / "southern-zone-latest.osm.pbf"
    bbox = tuple(study.to_crs("EPSG:4326").total_bounds)
    gdf = pyogrio.read_dataframe(pbf, layer=layer, where=where, bbox=bbox, columns=columns)
    gdf = gdf.set_crs("EPSG:4326", allow_override=True).to_crs(UTM)
    return gpd.clip(gdf, study)


def extract_osm(study):
    roads = read_osm("lines", "highway IS NOT NULL", study, ["osm_id", "name", "highway"])
    roads = roads[roads["highway"].isin(ROAD_CLASSES)].copy()
    roads["road_class"] = roads["highway"].map(ROAD_CLASSES)
    roads["length_m"] = roads.length.round(1)

    buildings = read_osm("multipolygons", "building IS NOT NULL", study,
                         ["osm_id", "osm_way_id", "name", "building"])
    buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    buildings["area_m2"] = buildings.area.round(1)

    # Land use, water and wetland polygons - useful hints when picking training points in Step 3
    landuse = read_osm("multipolygons", "landuse IS NOT NULL OR \"natural\" IS NOT NULL", study,
                       ["osm_id", "osm_way_id", "name", "landuse", "natural"])
    landuse = landuse[landuse.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    return roads, buildings, landuse


def plot_map(wards, corp, study, roads, buildings):
    fig, ax = plt.subplots(figsize=(9, 11))
    study.boundary.plot(ax=ax, color="grey", linestyle="--", linewidth=1)
    wards.plot(ax=ax, facecolor="#eef3ee", edgecolor="#9db8a3", linewidth=0.5)
    buildings.plot(ax=ax, color="#b5532c", linewidth=0)
    roads[roads.road_class.isin(["highway", "major"])].plot(ax=ax, color="#333333", linewidth=1)
    corp.boundary.plot(ax=ax, color="#1f4e3d", linewidth=2)
    for _, r in wards.iterrows():
        p = r.geometry.representative_point()
        ax.annotate(str(r.ward_no), (p.x, p.y), fontsize=5, ha="center", color="#1f4e3d")
    ax.set_title("Thiruvananthapuram Corporation - 101 wards, major roads, OSM buildings")
    ax.set_axis_off()
    fig.savefig(MAPS / "step1_base_map.png", dpi=200, bbox_inches="tight")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    MAPS.mkdir(parents=True, exist_ok=True)

    wards = load_wards()
    corp, study = build_boundaries(wards)
    print(f"Wards: {len(wards)}  |  corporation {corp.area_km2.iloc[0]} km2  |  study area {study.area_km2.iloc[0]} km2")

    roads, buildings, landuse = extract_osm(study)
    print(f"Roads: {len(roads)} segments, {roads.length_m.sum() / 1000:.0f} km  |  "
          f"buildings: {len(buildings)}  |  land-use polygons: {len(landuse)}")

    for name, gdf in [("wards", wards), ("corporation", corp), ("study_area", study),
                      ("roads", roads), ("buildings", buildings), ("landuse", landuse)]:
        gdf.to_file(GPKG, layer=name, driver="GPKG")
    corp.to_crs("EPSG:4326").to_file(OUT / "corporation_boundary.geojson", driver="GeoJSON")
    study.to_crs("EPSG:4326").to_file(OUT / "study_area.geojson", driver="GeoJSON")

    plot_map(wards, corp, study, roads, buildings)
    print(f"Saved {GPKG.relative_to(ROOT)} and outputs/maps/step1_base_map.png")


if __name__ == "__main__":
    main()
