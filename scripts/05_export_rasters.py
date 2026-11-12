import os
import ee
import geemap
import rasterio

ee.Initialize(project="ee-goriola-obafemi-ogoniland")

gaul2 = ee.FeatureCollection("FAO/GAUL/2025/level2")
aoi = gaul2.filter(ee.Filter.And(
    ee.Filter.eq("GAUL1_NAME", "Bayelsa"),
    ee.Filter.eq("GAUL2_NAME", "Yenegoa"),
))
geom = aoi.geometry()
aoi_area = geom.area(100)


def add_cover(img):
    inter = img.geometry().intersection(geom, 100).area(100)
    return img.set("cover", inter.divide(aoi_area).multiply(100))


s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
      .filterBounds(geom)
      .filter(ee.Filter.eq("instrumentMode", "IW"))
      .filter(ee.Filter.eq("orbitProperties_pass", "DESCENDING"))
      .filter(ee.Filter.eq("relativeOrbitNumber_start", 22))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
      .map(add_cover)
      .filter(ee.Filter.gt("cover", 90)))


def smooth(img):
    return img.select(["VV", "VH"]).focalMedian(30, "circle", "meters")


def one_date(day):
    d0 = ee.Date(day)
    return smooth(s1.filterDate(d0, d0.advance(1, "day")).first())


before_dates = ["2022-08-02", "2022-08-14", "2022-08-26", "2022-09-07", "2022-09-19"]
before = ee.ImageCollection([one_date(d) for d in before_dates]).median()

os.makedirs("data", exist_ok=True)
NODATA = -32768

for d in ["2022-10-13", "2022-10-25"]:
    fl = one_date(d)
    stack = ee.Image.cat([
        before.select("VV").rename("VV_before"),
        before.select("VH").rename("VH_before"),
        fl.select("VV").rename("VV_flood"),
        fl.select("VH").rename("VH_flood"),
    ])
    out = stack.multiply(100).round().toInt16().unmask(NODATA).clip(geom)
    path = f"data/s1_yenagoa_{d}.tif"
    print("Exporting", path, "...")
    geemap.ee_export_image(out, filename=path, scale=25, region=geom, file_per_band=False)
    with rasterio.open(path) as src:
        arr = src.read(1)
        valid = (arr != NODATA).mean() * 100
        print(f"  size {src.width} x {src.height}, bands {src.count}, dtype {src.dtypes[0]}, CRS {src.crs}")
        print(f"  band 1 non empty pixels: {valid:.1f}% of the bounding box")
print("Values are dB x 100 (for example -1850 means -18.5 dB). Nodata is -32768.")
