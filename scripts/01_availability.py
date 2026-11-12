import datetime
import ee

ee.Initialize(project="ee-goriola-obafemi-ogoniland")

gaul2 = ee.FeatureCollection("FAO/GAUL/2025/level2")
aoi = gaul2.filter(ee.Filter.And(
    ee.Filter.eq("GAUL1_NAME", "Bayelsa"),
    ee.Filter.eq("GAUL2_NAME", "Yenegoa"),
))
print("AOI features found:", aoi.size().getInfo())
geom = aoi.geometry()
aoi_area = geom.area(100)


def add_cover(img):
    inter = img.geometry().intersection(geom, 100).area(100)
    return img.set("cover", inter.divide(aoi_area).multiply(100))


s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
      .filterBounds(geom)
      .filter(ee.Filter.eq("instrumentMode", "IW"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
      .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
      .filterDate("2022-07-01", "2022-12-31")
      .sort("system:time_start")
      .map(add_cover))

times = s1.aggregate_array("system:time_start").getInfo()
passes = s1.aggregate_array("orbitProperties_pass").getInfo()
orbits = s1.aggregate_array("relativeOrbitNumber_start").getInfo()
sats = s1.aggregate_array("platform_number").getInfo()
covers = s1.aggregate_array("cover").getInfo()

print("Scenes found:", len(times))
print(f"{'date':<12}{'pass':<12}{'rel orbit':<11}{'satellite':<11}{'covers AOI %'}")
for t, p, o, s, c in zip(times, passes, orbits, sats, covers):
    d = datetime.datetime.fromtimestamp(t / 1000, datetime.timezone.utc).strftime("%Y-%m-%d")
    print(f"{d:<12}{p:<12}{o:<11}{s:<11}{c:.1f}")
