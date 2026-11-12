import os
import warnings
import ee
import geopandas as gpd
from shapely.geometry import shape
from shapely.validation import make_valid

warnings.filterwarnings("ignore")
ee.Initialize(project="ee-goriola-obafemi-ogoniland")

gaul2 = ee.FeatureCollection("FAO/GAUL/2025/level2")
aoi = gaul2.filter(ee.Filter.And(
    ee.Filter.eq("GAUL1_NAME", "Bayelsa"),
    ee.Filter.eq("GAUL2_NAME", "Yenegoa"),
))
lga = gpd.GeoDataFrame(geometry=[shape(aoi.geometry().getInfo())], crs=4326)
UTM = 32632
lga_area = lga.to_crs(UTM).area.sum() / 1e6
print(f"Yenagoa LGA area: {lga_area:.0f} km2\n")

base = os.path.expanduser("~/nigeria-flood-access/data/flood")


def clipped_area(path):
    gdf = gpd.read_file(path, bbox=tuple(lga.total_bounds))
    if len(gdf) == 0:
        return 0.0
    gdf["geometry"] = gdf.geometry.apply(make_valid)
    gdf = gdf.to_crs(4326)
    part = gpd.clip(gdf, lga)
    if len(part) == 0:
        return 0.0
    g = part.to_crs(UTM).geometry
    u = g.union_all() if hasattr(g, "union_all") else g.unary_union
    return u.area / 1e6


windows = ["20220901_20220930", "20221008_20221012", "20221013_20221017",
           "20221019_20221023", "20221001_20221025"]
kinds = ["MaximumFloodWaterExtent", "CloudObstruction", "PermanentWater"]

print("All figures are clipped to the Yenagoa LGA boundary")
print(f"{'window':<20}{'flood km2':<11}{'flood %':<9}{'cloud km2':<11}{'cloud %':<9}{'perm water km2'}")
for win in windows:
    vals = {}
    for kind in kinds:
        p = f"{base}/FL20221019NGA_SHP/VIIRS_{win}_{kind}_Nigeria.shp"
        vals[kind] = clipped_area(p) if os.path.exists(p) else float("nan")
    f, c, pw = vals["MaximumFloodWaterExtent"], vals["CloudObstruction"], vals["PermanentWater"]
    print(f"{win:<20}{f:<11.1f}{f / lga_area * 100:<9.1f}{c:<11.1f}{c / lga_area * 100:<9.1f}{pw:.1f}")

p = f"{base}/yenagoa_clipped/yenagoa_flood_final.shp"
g = gpd.read_file(p)
print("\nLayer used for the road analysis (yenagoa_flood_final.shp):")
print("  features:", len(g), " area km2:", round(clipped_area(p), 1))
print("  Water_Clas values:", sorted(set(g["Water_Clas"].astype(str))))
print("  Sensor_Dat values:", sorted(set(g["Sensor_Dat"].astype(str))))
