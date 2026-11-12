import numpy as np
import rasterio
from rasterio import features
from shapely.geometry import shape
import ee

ee.Initialize(project="ee-goriola-obafemi-ogoniland")
gaul2 = ee.FeatureCollection("FAO/GAUL/2025/level2")
aoi = gaul2.filter(ee.Filter.And(
    ee.Filter.eq("GAUL1_NAME", "Bayelsa"),
    ee.Filter.eq("GAUL2_NAME", "Yenegoa"),
))
geom = shape(aoi.geometry().getInfo())

with rasterio.open("data/s1_yenagoa_2022-10-25.tif") as src:
    T, H, W, crs = src.transform, src.height, src.width, src.crs
    arr = src.read()
print(f"Raster {W} x {H}, CRS {crs}")
inside = features.rasterize([(geom, 1)], out_shape=(H, W), transform=T,
                            fill=0, dtype="uint8").astype(bool)
np.save("data/inside_lga.npy", inside)

allzero = (arr == 0).all(axis=0)
print(f"Raster area at 25 m: {H * W * 625 / 1e6:,.0f} km2")
print(f"Pixels inside the LGA boundary: {inside.sum():,}  (about {inside.sum() * 625 / 1e6:,.0f} km2, the LGA is 706 km2)")
print(f"Pixels outside the LGA: {(~inside).sum():,}")
print(f"  outside pixels where all four bands are exactly 0: {allzero[~inside].mean() * 100:.1f}%")
print(f"  inside pixels where all four bands are exactly 0:  {allzero[inside].mean() * 100:.1f}%")
