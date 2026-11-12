import os
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
    return img.select(["VV", "VH"]).focalMedian(30, "circle", "meters")


def one_date(day):
    d0 = ee.Date(day)
    return smooth(s1.filterDate(d0, d0.advance(1, "day")).first())


before_dates = ["2022-08-02", "2022-08-14", "2022-08-26", "2022-09-07", "2022-09-19"]
before = ee.ImageCollection([one_date(d) for d in before_dates]).median()

flood_dates = ["2022-10-01", "2022-10-13", "2022-10-25", "2022-11-06"]

os.makedirs("preview", exist_ok=True)


def share(mask_img, name):
    v = mask_img.rename("m").reduceRegion(ee.Reducer.mean(), geom, 50, maxPixels=1e9).getInfo()
    return v["m"] * 100


def save(img, name, vis):
    vis = dict(vis)
    vis.update({"region": geom.bounds(), "dimensions": 1100, "format": "png"})
    url = img.clip(geom).getThumbURL(vis)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with open(f"preview/{name}.png", "wb") as f:
        f.write(r.content)


print(f"{'date':<12}{'VV down >3dB':<15}{'VV up >3dB':<13}{'VH down >3dB':<15}{'VH up >3dB':<13}{'new dark VV (<-18 now, not before)'}")
for d in flood_dates:
    fl = one_date(d)
    ch = fl.subtract(before)
    vv_down = share(ch.select("VV").lt(-3), "a")
    vv_up = share(ch.select("VV").gt(3), "b")
    vh_down = share(ch.select("VH").lt(-3), "c")
    vh_up = share(ch.select("VH").gt(3), "d")
    new_dark = share(fl.select("VV").lt(-18).And(before.select("VV").gte(-18)), "e")
    print(f"{d:<12}{vv_down:<15.1f}{vv_up:<13.1f}{vh_down:<15.1f}{vh_up:<13.1f}{new_dark:.1f}")
    pal = ["0000ff", "ffffff", "ff0000"]
    save(ch.select("VV"), f"change_VV_{d}", {"min": -6, "max": 6, "palette": pal})
    save(ch.select("VH"), f"change_VH_{d}", {"min": -6, "max": 6, "palette": pal})
print("Saved change images in preview/ (blue = darker than before, red = brighter than before)")
