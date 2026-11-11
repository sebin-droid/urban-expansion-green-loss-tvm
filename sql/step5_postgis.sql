-- =====================================================================================================
-- Step 5b - PostGIS spatial database: Thiruvananthapuram urban expansion and green space loss
--
-- Run by scripts/step5_postgis.py after it has loaded these tables into schema tvm (EPSG:32643):
--   wards, corporation, buildings, roads, landuse   OSM + K-SMART layers from Step 1 (geom column)
--   ward_change                                     Step 4 ward statistics with geometry
--   pixels30                                        451 k Landsat pixel centres: lc_2000 .. lc_2025, change
--   change_raster                                   2000->2025 change raster (raster2pgsql, 64x64 tiles)
-- Codes: land cover 1 built-up, 2 tree, 3 other vegetation, 4 water, 5 bare; green = 2,3
--        change 1 built stable, 2 green stable, 3 green->built, 4 other->built, 5 green->water/bare,
--               6 gained green, 7 water/bare stable
-- One 30 m pixel = 0.0009 km2.
-- =====================================================================================================
SET search_path = tvm, public;

-- -----------------------------------------------------------------------------------------------------
-- 1. Geometry for the pixel table (loaded as plain x, y with COPY, which is much faster than WKB inserts)
-- -----------------------------------------------------------------------------------------------------
ALTER TABLE pixels30 ADD COLUMN geom geometry(Point, 32643);
UPDATE pixels30 SET geom = ST_SetSRID(ST_MakePoint(x, y), 32643);
ALTER TABLE pixels30 ADD PRIMARY KEY (pixel_id);

-- -----------------------------------------------------------------------------------------------------
-- 2. Indexes. GiST = R-tree over bounding boxes; it is what turns a spatial join from
--    "every row against every row" into "only rows whose boxes overlap".
-- -----------------------------------------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_pixels30_geom     ON pixels30    USING gist (geom);
CREATE INDEX IF NOT EXISTS idx_wards_geom        ON wards       USING gist (geom);
CREATE INDEX IF NOT EXISTS idx_corporation_geom  ON corporation USING gist (geom);
CREATE INDEX IF NOT EXISTS idx_buildings_geom    ON buildings   USING gist (geom);
CREATE INDEX IF NOT EXISTS idx_roads_geom        ON roads       USING gist (geom);
CREATE INDEX IF NOT EXISTS idx_landuse_geom      ON landuse     USING gist (geom);
-- partial index: only the ~1,100 main-road segments, used by the nearest-road (KNN) search
CREATE INDEX IF NOT EXISTS idx_roads_main_geom   ON roads       USING gist (geom)
    WHERE road_class IN ('highway', 'major');
CREATE INDEX IF NOT EXISTS idx_pixels30_change   ON pixels30 (change);
CREATE UNIQUE INDEX IF NOT EXISTS idx_wards_no   ON wards (ward_no);
ANALYZE;

-- -----------------------------------------------------------------------------------------------------
-- 3. Ward of every pixel: one indexed point-in-polygon join, stored so later queries need no geometry
-- -----------------------------------------------------------------------------------------------------
ALTER TABLE pixels30 ADD COLUMN ward_no integer;
UPDATE pixels30 p SET ward_no = w.ward_no FROM wards w WHERE ST_Contains(w.geom, p.geom);
CREATE INDEX idx_pixels30_ward ON pixels30 (ward_no);
ANALYZE pixels30;

-- -----------------------------------------------------------------------------------------------------
-- 4. Spatial partitioning inside the database: a LIST-partitioned copy of the pixels, one partition
--    per geohash-5 cell (~4.9 km x 4.9 km). Same idea as Sedona's grid partitioning: a query that
--    names its cell(s) only reads those partitions (partition pruning).
-- -----------------------------------------------------------------------------------------------------
ALTER TABLE pixels30 ADD COLUMN gh5 text;
UPDATE pixels30 SET gh5 = ST_GeoHash(ST_Transform(geom, 4326), 5);

-- store the pixel rows in R-tree order, so neighbouring points sit on the same disk pages
CLUSTER pixels30 USING idx_pixels30_geom;
ANALYZE pixels30;

CREATE TABLE pixels30_part (LIKE pixels30) PARTITION BY LIST (gh5);
DO $$
DECLARE cell text;
BEGIN
    FOR cell IN SELECT DISTINCT gh5 FROM pixels30 ORDER BY 1 LOOP
        EXECUTE format('CREATE TABLE pixels30_gh_%s PARTITION OF pixels30_part FOR VALUES IN (%L)', cell, cell);
    END LOOP;
END $$;
INSERT INTO pixels30_part SELECT * FROM pixels30;
CREATE INDEX idx_pixels30_part_geom ON pixels30_part USING gist (geom);   -- created on every partition
ANALYZE pixels30_part;

-- -----------------------------------------------------------------------------------------------------
-- 5. Analysis views
-- -----------------------------------------------------------------------------------------------------

-- 5a. Green space and built-up per ward from the pixel table (vector route)
CREATE OR REPLACE VIEW v_ward_green_loss AS
SELECT w.ward_no, w.ward_name, w.area_km2,
       round(count(p.*) FILTER (WHERE p.lc_2000 IN (2, 3)) * 0.0009, 3) AS green_2000_km2,
       round(count(p.*) FILTER (WHERE p.lc_2025 IN (2, 3)) * 0.0009, 3) AS green_2025_km2,
       round(count(p.*) FILTER (WHERE p.lc_2000 = 1)       * 0.0009, 3) AS built_2000_km2,
       round(count(p.*) FILTER (WHERE p.lc_2025 = 1)       * 0.0009, 3) AS built_2025_km2,
       round(count(p.*) FILTER (WHERE p.change = 3)        * 0.0009, 3) AS green_to_built_km2,
       round(100.0 * count(p.*) FILTER (WHERE p.lc_2025 IN (2, 3)) / nullif(count(p.*), 0)
           - 100.0 * count(p.*) FILTER (WHERE p.lc_2000 IN (2, 3)) / nullif(count(p.*), 0), 2) AS green_change_pp
FROM wards w LEFT JOIN pixels30 p ON p.ward_no = w.ward_no
GROUP BY w.ward_no, w.ward_name, w.area_km2;

-- 5b. The same quantity from the raster (raster route): count change codes inside each ward
CREATE OR REPLACE VIEW v_ward_change_raster AS
SELECT w.ward_no, (vc).value::int AS change_code, round(sum((vc).count) * 0.0009, 3) AS km2
FROM wards w
JOIN change_raster r ON ST_Intersects(r.rast, w.geom)
CROSS JOIN LATERAL ST_ValueCount(ST_Clip(r.rast, w.geom), 1, true) AS vc
GROUP BY w.ward_no, (vc).value;

-- 5c. Buildings and roads per ward, combined with the green figures
CREATE MATERIALIZED VIEW mv_ward_metrics AS
WITH b AS (
    SELECT w.ward_no, count(*) AS buildings, sum(b.area_m2) AS footprint_m2
    FROM wards w JOIN buildings b ON ST_Contains(w.geom, ST_PointOnSurface(b.geom))
    GROUP BY w.ward_no),
r AS (
    SELECT w.ward_no,
           sum(ST_Length(ST_Intersection(w.geom, r.geom))) / 1000 AS road_km,
           coalesce(sum(ST_Length(ST_Intersection(w.geom, r.geom)))
                    FILTER (WHERE r.road_class IN ('highway', 'major')), 0) / 1000 AS main_road_km
    FROM wards w JOIN roads r ON ST_Intersects(w.geom, r.geom)
    GROUP BY w.ward_no)
SELECT g.*,
       coalesce(b.buildings, 0)                                         AS buildings,
       round(coalesce(b.footprint_m2, 0)::numeric, 0)                   AS footprint_m2,
       round((coalesce(b.buildings, 0) / g.area_km2)::numeric, 0)       AS buildings_per_km2,
       round((100 * coalesce(b.footprint_m2, 0) / (g.area_km2 * 1e6))::numeric, 2) AS building_coverage_pct,
       round(coalesce(r.road_km, 0)::numeric, 2)                        AS road_km,
       round(coalesce(r.main_road_km, 0)::numeric, 2)                   AS main_road_km,
       round((coalesce(r.road_km, 0) / g.area_km2)::numeric, 2)         AS road_density_km_per_km2
FROM v_ward_green_loss g
LEFT JOIN b USING (ward_no)
LEFT JOIN r USING (ward_no);
CREATE UNIQUE INDEX ON mv_ward_metrics (ward_no);

-- 5d. Nearest main road for every pixel that was green in 2000 - index-assisted KNN (<-> operator)
CREATE MATERIALIZED VIEW mv_green2000_road_dist AS
SELECT p.pixel_id, p.ward_no, p.change, nr.dist_m
FROM pixels30 p
CROSS JOIN LATERAL (
    SELECT ST_Distance(p.geom, r.geom) AS dist_m
    FROM roads r
    WHERE r.road_class IN ('highway', 'major')
    ORDER BY r.geom <-> p.geom
    LIMIT 1) nr
WHERE p.ward_no IS NOT NULL AND p.lc_2000 IN (2, 3);

CREATE OR REPLACE VIEW v_road_distance_bands AS
WITH banded AS (
    SELECT change,
           CASE WHEN dist_m < 100  THEN 1 WHEN dist_m < 250  THEN 2 WHEN dist_m < 500 THEN 3
                WHEN dist_m < 1000 THEN 4 WHEN dist_m < 2000 THEN 5 ELSE 6 END AS band_no
    FROM mv_green2000_road_dist)
SELECT band_no,
       (ARRAY['0-100 m', '100-250 m', '250-500 m', '500-1000 m', '1000-2000 m', '>2000 m'])[band_no] AS band,
       round(count(*) * 0.0009, 3)                                   AS green_2000_km2,
       round(count(*) FILTER (WHERE change = 3) * 0.0009, 3)         AS converted_km2,
       round(100.0 * count(*) FILTER (WHERE change = 3) / count(*), 2) AS conversion_rate_pct
FROM banded GROUP BY band_no ORDER BY band_no;

-- 5e. Where did green turn into built-up? OSM land-use polygon (today's mapping) of each converted pixel
CREATE OR REPLACE VIEW v_conversion_by_landuse AS
SELECT coalesce(l.landuse, l."natural", '(no OSM land use)') AS osm_landuse,
       round(count(*) FILTER (WHERE p.lc_2000 IN (2, 3)) * 0.0009, 3) AS green_2000_km2,
       round(count(*) FILTER (WHERE p.change = 3) * 0.0009, 3)        AS green_to_built_km2,
       round(100.0 * count(*) FILTER (WHERE p.change = 3)
             / nullif(count(*) FILTER (WHERE p.lc_2000 IN (2, 3)), 0), 2) AS conversion_rate_pct
FROM pixels30 p
LEFT JOIN LATERAL (
    SELECT l.landuse, l."natural" FROM landuse l
    WHERE ST_Contains(l.geom, p.geom)
    ORDER BY ST_Area(l.geom) LIMIT 1) l ON true          -- smallest polygon = most specific tag
WHERE p.ward_no IS NOT NULL
GROUP BY 1
ORDER BY green_to_built_km2 DESC;

-- 5f. Corporation totals per period from the Step 4 table (sanity check against change_summary.csv)
CREATE OR REPLACE VIEW v_corporation_totals AS
SELECT round(sum(green_2000_km2), 2) AS green_2000_km2,
       round(sum(green_2025_km2), 2) AS green_2025_km2,
       round(sum(green_to_built_km2), 2) AS green_to_built_km2,
       round(sum(built_2000_km2), 2) AS built_2000_km2,
       round(sum(built_2025_km2), 2) AS built_2025_km2
FROM v_ward_green_loss;
