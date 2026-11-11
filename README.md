# Urban Expansion and Green Space Loss in Thiruvananthapuram (2000–2025)

MSc (Semester 3) Spatial Big Data course project, Digital University Kerala, November 2026.

This project measures how much green cover the 101 wards of Thiruvananthapuram Municipal Corporation lost to urban
expansion between 2000 and 2025. It also shows where the loss happened and which conditions made it more likely.
The heavy spatial processing is done with Apache Sedona on Spark and with PostgreSQL/PostGIS.

## Main results

- Built-up land grew from 15.6% to 22.9% of the corporation. Green cover fell from 77.7% to 74.2% (net −7.4 km²).
- 12.2 km² of green land became built-up, mostly in the southern and central wards, near roads and existing buildings.
- The yearly loss slowed from 0.71 km² (2000–2010) to 0.52 km² (2017–2025).
- Green land that became built-up warmed about 1.6 °C more than land that stayed green.
- Top priority wards: Sreevaraham, Perunthanni, Kamaleswaram, Pettah and Kalippankulam.

The full write-up is in [`final report/final report.pdf`](final%20report/final%20report.pdf). The slides are in
[`report/`](report/).

## Repository layout

| Folder | Contents |
|---|---|
| `scripts/` | The pipeline, one script per step (see below) |
| `sql/` | PostGIS queries used in step 5 |
| `dashboard/` | Streamlit dashboard (`app.py`) and the small data files it needs |
| `outputs/maps/` | Maps and charts made by each step |
| `final report/` | Project report (PDF / DOCX) |
| `report/` | Presentation slides |

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_base_data.py` | Ward boundaries (K-SMART 2024), geometry repair, corporation outline and study area, OSM roads, land use and buildings, ward population |
| 2 | `step2_composites.py` | Dry-season Landsat (2000, 2010, 2017, 2025) and Sentinel-2 (2019, 2025) composites in Google Earth Engine |
| 3 | `step3_classification.py` | Random Forest land-cover classification (built-up, tree, other vegetation, water, bare) |
| 4 | `step4_change.py` | Consistent change series (backdating from 2025), ward change statistics |
| 5 | `step5_sedona.py`, `step5_postgis.py` | Spatial joins, partitioning and index benchmarks in Apache Sedona and PostGIS, road distance, H3 hexagons |
| 6 | `step6_analytics.py` | Gi* / DBSCAN hotspots, FP-Growth association rules, heat / flood / green-per-person case studies, ward priority index |
| 7 | `step7_prepare.py` | Builds the files in `dashboard/data/` |

## Running it

Python 3.12 is recommended.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Dashboard only.** The data it needs is already in the repository:

```powershell
.venv\Scripts\streamlit run dashboard/app.py
```

**Full pipeline.** The `data/` folder is not in the repository, because it is about 1 GB and some files are larger
than GitHub allows. To rebuild it:

- Download the K-SMART ward boundaries (`Kerala_ward_boundaries.geojson`, from the
  [kerala-ward-map](https://github.com/madpk/kerala-ward-map) repository) into `data/raw/ksmart_wards/`.
- Download the Geofabrik `southern-zone-latest.osm.pbf` extract into `data/raw/osm/`.
- Steps 2, 3 and 6 need a Google Earth Engine account. Change `EE_PROJECT` in those scripts to your own Cloud project.
- Step 5 needs Java 17 for Spark/Sedona, and a local PostgreSQL with PostGIS. The database password is read from
  `PGPASSWORD` or `pgpass.conf` and is never stored in the code.

Then run the scripts in `scripts/` in order, from step 1 to step 7.

## Data sources

Landsat 5/7/8/9 and Sentinel-2 (Google Earth Engine), ESA WorldCover, Dynamic World, GHSL, WorldPop, MODIS LST,
MERIT Hydro, Sentinel-1, OpenStreetMap (© OpenStreetMap contributors, ODbL) and the K-SMART 2024 ward delimitation.
The original ward file contains surveyor names and phone numbers. These fields are removed in step 1 and are not
included anywhere in this repository.

## Author

Sebin George
