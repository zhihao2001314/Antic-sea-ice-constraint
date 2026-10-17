# %% 1. 导入依赖

from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from shapely.geometry import shape


# %% 2. 参数与路径设置

root = Path(r"F:\sea\Antic sea ice constraint")
source = root / "output/s02sia/s02_arctic_region/p14_mld_fixed_regions_raw/mld_coast_regions_20261012.geojson"
stem = root / "figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/MLD_coast_sampling_regions_20261012"
land_path = Path(r"C:\Users\Lenovo\.local\share\cartopy\shapefiles\natural_earth\physical\ne_110m_land.shp")
formats = ["png", "pdf", "svg"]
assert not any(stem.with_suffix("." + suffix).exists() for suffix in formats)
plt.rcParams.update({"font.family": "Arial", "font.size": 14,
                     "pdf.fonttype": 42, "svg.fonttype": "none"})
projection = ccrs.LambertConformal(central_longitude=-25, central_latitude=67.5,
                                  standard_parallels=(50, 80))


# %% 3. 读取海岸裁剪后的区域

features = json.loads(source.read_text(encoding="utf-8"))["features"]
geometry = {item["properties"]["region"]: shape(item["geometry"]) for item in features}
land = cfeature.ShapelyFeature(list(Reader(land_path).geometries()),
                              ccrs.PlateCarree(), facecolor="#E6E5DF",
                              edgecolor="#707A7D", linewidth=0.7)
regions = [("Labrador-Irminger", "#B05D14", "--", "Labrador–Irminger sector", 1, -42, 56),
           ("Nordic Seas", "#006C99", "-", "Nordic–Iceland sector", 2, 1, 72)]


# %% 4. 绘制固定区域

fig = plt.figure(figsize=(10, 8.5), facecolor="white")
ax = fig.add_axes([0.08, 0.18, 0.84, 0.73], projection=projection)
ax.set_extent([-65, 22, 50, 85], crs=ccrs.PlateCarree())
ax.set_facecolor("#F8FBFC")
grid = ax.gridlines(draw_labels=True, xlocs=[-60, -40, -20, 0, 20],
                    ylocs=[50, 60, 70, 80], linewidth=0.6,
                    color="#ACB8BE", alpha=0.6, linestyle=":")
grid.top_labels = False
grid.right_labels = False
grid.x_inline = False
grid.y_inline = False
grid.xlabel_style = {"size": 12, "color": "#56616A"}
grid.ylabel_style = {"size": 12, "color": "#56616A"}
ax.add_feature(land, zorder=1)
handles = []
for region, color, style, label, number, label_lon, label_lat in regions:
    ax.add_geometries([geometry[region]], ccrs.PlateCarree(), facecolor=color,
                      edgecolor="none", alpha=0.16, zorder=2)
    ax.add_geometries([geometry[region]], ccrs.PlateCarree(), facecolor="none",
                      edgecolor=color, linewidth=2, linestyle=style, zorder=3)
    handles.append(Patch(facecolor="white", edgecolor=color, linewidth=2,
                         linestyle=style, label=f"{number}  {label}"))
    ax.text(label_lon, label_lat, str(number), transform=ccrs.PlateCarree(),
            ha="center", va="center", color=color, fontsize=20, fontweight="bold",
            bbox={"boxstyle": "circle,pad=0.25", "fc": "white", "ec": color, "lw": 1.5}, zorder=4)
ax.text(-42, 74, "Greenland", color="#626D72", fontsize=14,
        ha="center", transform=ccrs.PlateCarree(), zorder=5)
fig.text(0.5, 0.955, "MLD sampling regions", ha="center", fontsize=23,
         fontweight="bold", color="#25343C")
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.065),
           ncol=1, frameon=False, fontsize=15, handlelength=2.5)
fig.text(0.5, 0.025, "Fixed offshore boundaries; continental boundaries follow the coastline.",
         ha="center", fontsize=11, color="#56616A")


# %% 5. 保存图片

for suffix in formats:
    fig.savefig(stem.with_suffix("." + suffix), dpi=400, facecolor="white")
plt.close(fig)
print(stem)
