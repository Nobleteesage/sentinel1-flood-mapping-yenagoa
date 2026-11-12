import os
import datetime
import requests
import ee

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
    return img.select("VV").focalMedian(30, "circle", "meters")


def one_date(day):
    d0 = ee.Date(day)
    return smooth(s1.filterDate(d0, d0.advance(1, "day")).first())


before_dates = ["2022-08-02", "2022-08-14", "2022-08-26", "2022-09-07", "2022-09-19"]
before_coll = ee.ImageCollection([one_date(d) for d in before_dates])
before = before_coll.median()

flood_dates = ["2022-10-01", "2022-10-13", "2022-10-25", "2022-11-06"]

layers = {"before_median_aug_to_sep": before}
for d in flood_dates:
    layers["flood_" + d] = one_date(d)

os.makedirs("preview", exist_ok=True)
print("Mean VV (dB) over Yenagoa, and share of pixels below -18 dB (rough open water)")
for name, img in layers.items():
    img = img.rename("VV")
    stats = img.reduceRegion(ee.Reducer.mean(), geom, 100, maxPixels=1e9).getInfo()
    low = img.lt(-18).reduceRegion(ee.Reducer.mean(), geom, 100, maxPixels=1e9).getInfo()
    print(f"{name:<28} mean VV {stats['VV']:.2f}   below -18 dB: {low['VV'] * 100:.1f}%")
    url = img.clip(geom).getThumbURL({
        "region": geom.bounds(), "dimensions": 1100,
        "min": -25, "max": 0, "format": "png",
    })
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(f"preview/{name}.png", "wb") as f:
        f.write(r.content)
print("Saved images in preview/")
