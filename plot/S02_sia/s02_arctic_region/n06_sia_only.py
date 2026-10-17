# -*- coding: utf-8 -*-
"""
绘制未经 piControl 订正的北极分区 SIA pattern，并把最新重算的 thetao / tas 区域值标到 CA / NA / NP。

【是否需要运行】
需要。仅绘图脚本；先保证 SIA plot-ready 数据和新的温度 region summary 已存在。

【输入文件】
output/s02sia/s02_arctic_region/p04_pattern_unadjusted/n01_arctic_sia_pattern/
    arctic_sia_pattern_cells.csv
    arctic_sia_pattern_summary.csv
    arctic_sia_pattern_metadata.json
output/s02sia/s02_arctic_region/p04_pattern_unadjusted/n02_temperature_pattern/
    arctic_temperature_region_summary.csv
output/s02sia/s02_arctic_region/p02_pattern_region/n02/
    observed_sic_1988_2020_climatology_edge.npz

【输出文件】
figure/s02_sia/s02_arctic_region/p04_pattern_unadjusted/n01_arctic_sia_pattern/
    arctic_sia_partition_pattern.png
    arctic_sia_partition_pattern.pdf
    arctic_sia_partition_pattern.svg

【什么时候需要重新运行】
SIA summary、温度 summary 或绘图样式变化后。

说明
1. 本脚本只读 output，不读取原始 NetCDF。
2. SIA / thetao / tas 均使用当前未订正结果。
3. CA=70–90°N；NA=45–70°N, 60°W–30°E；NP=45–70°N, 120°E–150°W。
4. CA / NP 科学量使用 axes 坐标固定在空白区域；NA 保持地理坐标。
5. 经纬度标注采用清晰圆周布局：180°W、120°W、60°W、0°、60°E、120°E，以及60°N、70°N、80°N。
6. 同名图直接覆盖，不弹出 Figure 窗口。
"""

# %% 1. 导入依赖
from pathlib import Path
import json
import string
import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.util import add_cyclic_point
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

# %% 2. 路径
ROOT = Path(r"F:\sea\Antic sea ice constraint")
DATA_DIR = ROOT / "output" / "s02sia" / "s02_arctic_region" / "p04_pattern_unadjusted" / "n01_arctic_sia_pattern"
TEMP_DIR = ROOT / "output" / "s02sia" / "s02_arctic_region" / "p04_pattern_unadjusted" / "n02_temperature_pattern"
FIG_DIR = ROOT / "figure" / "s02_sia" / "s02_arctic_region" / "p04_pattern_unadjusted" / "n01_arctic_sia_pattern"
CELL_FILE = DATA_DIR / "arctic_sia_pattern_cells.csv"
SUMMARY_FILE = DATA_DIR / "arctic_sia_pattern_summary.csv"
META_FILE = DATA_DIR / "arctic_sia_pattern_metadata.json"
TEMPERATURE_SUMMARY_FILE = TEMP_DIR / "arctic_temperature_region_summary.csv"
OBS_FILE = ROOT / "output" / "s02sia" / "s02_arctic_region" / "p02_pattern_region" / "n02" / "observed_sic_1988_2020_climatology_edge.npz"
PNG_FILE = FIG_DIR / "arctic_sia_partition_pattern.png"
PDF_FILE = FIG_DIR / "arctic_sia_partition_pattern.pdf"
SVG_FILE = FIG_DIR / "arctic_sia_partition_pattern.svg"
for path in [CELL_FILE, SUMMARY_FILE, META_FILE, TEMPERATURE_SUMMARY_FILE]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少输入文件：{path}")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% 3. 读取 SIA plot-ready 数据
cells = pd.read_csv(CELL_FILE)
summary = pd.read_csv(SUMMARY_FILE)
metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
MODELS = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4", "MIROC-ES2L", "UKESM1-0-LL", "CAS-ESM2-0", "NorESM2-LM"]
models_successful = metadata.get("models_successful", [])
missing_models = [model for model in MODELS if model not in models_successful]
if missing_models:
    raise RuntimeError(f"正式9模式pattern不完整，缺少：{missing_models}")
required_columns = {"model", "latitude_deg_north", "longitude_deg_east", "display_delta_sia_thousand_km2", "edge_raw_co2_sic_fraction", "edge_raw_cdr_sic_fraction"}
missing_columns = required_columns - set(cells.columns)
if missing_columns:
    raise KeyError(f"pattern cells缺少字段：{sorted(missing_columns)}")
required_summary = {"model", "region", "raw_delta_sia_million_km2"}
if not required_summary.issubset(summary.columns):
    raise KeyError(f"SIA summary缺少字段：{sorted(required_summary - set(summary.columns))}")
latitude = np.sort(cells["latitude_deg_north"].unique())
longitude = np.sort(cells["longitude_deg_east"].unique())
if latitude.size != 45 or longitude.size != 360:
    raise ValueError(f"共同pattern网格应为45×360，实际={latitude.size}×{longitude.size}")
sia_lookup = summary.set_index(["model", "region"])["raw_delta_sia_million_km2"]
color_limit = float(metadata["sia_color_limit_thousand_km2"])

# %% 4. 读取新的 thetao / tas 区域值
temp_summary = pd.read_csv(TEMPERATURE_SUMMARY_FILE)
required_temp_columns = {"model", "region", "raw_delta_thetao_degC", "raw_delta_tas_degC"}
if not required_temp_columns.issubset(temp_summary.columns):
    raise KeyError(f"温度summary缺少字段：{sorted(required_temp_columns - set(temp_summary.columns))}")
temp_summary = temp_summary[["model", "region", "raw_delta_thetao_degC", "raw_delta_tas_degC"]].copy()
temp_summary["region"] = temp_summary["region"].replace({"NA": "North Atlantic", "north_atlantic": "North Atlantic", "CA": "Central Arctic", "central_arctic": "Central Arctic", "NP": "North Pacific", "north_pacific": "North Pacific"})
temp_summary = temp_summary[temp_summary["region"].isin(["North Atlantic", "Central Arctic", "North Pacific"])].drop_duplicates(["model", "region"], keep="last")
thetao_lookup = temp_summary.set_index(["model", "region"])["raw_delta_thetao_degC"]
tas_lookup = temp_summary.set_index(["model", "region"])["raw_delta_tas_degC"]
for model in MODELS:
    for region in ["North Atlantic", "Central Arctic", "North Pacific"]:
        if (model, region) not in thetao_lookup.index:
            raise KeyError(f"thetao summary缺少：model={model}, region={region}")
        if (model, region) not in tas_lookup.index:
            raise KeyError(f"tas summary缺少：model={model}, region={region}")

# %% 5. 观测边缘
obs_available = OBS_FILE.is_file()
if obs_available:
    obs = np.load(OBS_FILE)
    obs_x = np.asarray(obs["projected_x_m"], dtype=float)
    obs_y = np.asarray(obs["projected_y_m"], dtype=float)
    obs_sic = np.asarray(obs["sic_smoothed_for_edge_fraction"], dtype=float)
    print(f"[观测边缘] {OBS_FILE}")
else:
    print(f"[提示] 未找到观测边缘：{OBS_FILE}")

# %% 6. 绘图样式
plt.rcParams.update({"font.family": "Arial", "font.size": 8.5, "axes.titlesize": 9.5, "axes.linewidth": 0.8, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
DATA_CRS = ccrs.PlateCarree()
MAP_CRS = ccrs.NorthPolarStereo(central_longitude=0.0)
OBS_CRS = ccrs.epsg(3411)
CA_COLOR = "#2166AC"
NA_COLOR = "#D95F02"
NP_COLOR = "#1B9E77"
CO2_EDGE_COLOR = "#00A6D6"
CDR_EDGE_COLOR = "#9C4DCC"
OBS_EDGE_COLOR = "#000000"
LAND_COLOR = "#D8D8D8"
COAST_COLOR = "#8A8A8A"
GRID_COLOR = "#AFAFAF"
COORD_COLOR = "#555555"
cmap = plt.get_cmap("RdBu_r").copy()
cmap.set_bad("#F7F7F7")
norm = TwoSlopeNorm(vmin=-color_limit, vcenter=0.0, vmax=color_limit)
theta = np.linspace(0.0, 2.0 * np.pi, 361)
circle = mpath.Path(np.column_stack([0.5 + 0.5 * np.sin(theta), 0.5 + 0.5 * np.cos(theta)]))

# %% 7. 分区边界与经纬度标注
ca_lon = np.linspace(-180.0, 180.0, 721)
na_lon = np.linspace(-60.0, 30.0, 181)
na_side_lat = np.linspace(45.0, 70.0, 101)
np_lon_east = np.linspace(120.0, 180.0, 121)
np_lon_west = np.linspace(-180.0, -150.0, 61)
np_side_lat = np.linspace(45.0, 70.0, 101)
COORD_FONTSIZE = 6.0
COORD_STROKE = 1.6
lon_label_specs = [(180.0, 46.4, "180°W", 0), (-120.0, 46.8, "120°W", -30), (-60.0, 46.8, "60°W", 30), (0.0, 46.2, "0°", 0), (60.0, 46.8, "60°E", -30), (120.0, 46.8, "120°E", 30)]
lat_label_specs = [(67.0, 60.0, "60°N", -26), (55.0, 70.0, "70°N", -18), (18.0, 80.0, "80°N", -5)]

# %% 8. Figure
fig = plt.figure(figsize=(12.0, 13.2))
axes = [fig.add_subplot(3, 3, index + 1, projection=MAP_CRS) for index in range(9)]
fig.suptitle("Arctic sea-ice-area change (CDR years 131–140 − 1pctCO$_2$ years 1–10)", x=0.5, y=0.985, fontsize=13.0, fontweight="bold")
mesh = None

# %% 9. 九模式
for index, model in enumerate(MODELS):
    ax = axes[index]
    model_cells = cells[cells["model"] == model].sort_values(["latitude_deg_north", "longitude_deg_east"]).reset_index(drop=True)
    if len(model_cells) != latitude.size * longitude.size:
        raise ValueError(f"{model} SIA pattern格点不完整：{len(model_cells)}")
    display = model_cells["display_delta_sia_thousand_km2"].to_numpy(dtype=float).reshape(latitude.size, longitude.size)
    co2_edge = model_cells["edge_raw_co2_sic_fraction"].to_numpy(dtype=float).reshape(latitude.size, longitude.size)
    cdr_edge = model_cells["edge_raw_cdr_sic_fraction"].to_numpy(dtype=float).reshape(latitude.size, longitude.size)
    display_cyclic, longitude_cyclic = add_cyclic_point(display, coord=longitude)
    co2_edge_cyclic, _ = add_cyclic_point(co2_edge, coord=longitude)
    cdr_edge_cyclic, _ = add_cyclic_point(cdr_edge, coord=longitude)
    lon_grid, lat_grid = np.meshgrid(longitude_cyclic, latitude)
    ax.set_extent([-180.0, 180.0, 45.0, 90.0], crs=DATA_CRS)
    ax.set_boundary(circle, transform=ax.transAxes)

    # %% 9.1 SIA底图
    mesh = ax.pcolormesh(lon_grid, lat_grid, np.ma.masked_invalid(display_cyclic), transform=DATA_CRS, cmap=cmap, norm=norm, shading="auto", rasterized=True, zorder=1)

    # %% 9.2 经纬网
    gridliner = ax.gridlines(crs=DATA_CRS, draw_labels=False, linewidth=0.40, color=GRID_COLOR, alpha=0.42, linestyle=":", zorder=2)
    gridliner.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gridliner.ylocator = mticker.FixedLocator([60, 70, 80])

    # %% 9.3 海冰边缘
    if np.isfinite(co2_edge_cyclic).any() and np.nanmin(co2_edge_cyclic) <= 0.15 <= np.nanmax(co2_edge_cyclic):
        ax.contour(lon_grid, lat_grid, np.ma.masked_invalid(co2_edge_cyclic), levels=[0.15], colors=[CO2_EDGE_COLOR], linewidths=1.15, linestyles="-", transform=DATA_CRS, zorder=3.5)
    if np.isfinite(cdr_edge_cyclic).any() and np.nanmin(cdr_edge_cyclic) <= 0.15 <= np.nanmax(cdr_edge_cyclic):
        ax.contour(lon_grid, lat_grid, np.ma.masked_invalid(cdr_edge_cyclic), levels=[0.15], colors=[CDR_EDGE_COLOR], linewidths=1.20, linestyles="--", transform=DATA_CRS, zorder=3.6)
    if obs_available:
        ax.contour(obs_x, obs_y, np.ma.masked_invalid(obs_sic), levels=[0.15], colors=[OBS_EDGE_COLOR], linewidths=0.95, linestyles=":", transform=OBS_CRS, zorder=3.7)

    # %% 9.4 陆地与海岸线
    ax.add_feature(cfeature.LAND, facecolor=LAND_COLOR, edgecolor="none", zorder=4.0)
    ax.coastlines(resolution="110m", color=COAST_COLOR, linewidth=0.45, zorder=4.1)

    # %% 9.5 分区边界
    ax.plot(ca_lon, np.full(ca_lon.size, 70.0), color=CA_COLOR, linewidth=1.9, transform=DATA_CRS, zorder=5)
    ax.plot(na_lon, np.full(na_lon.size, 45.0), color=NA_COLOR, linewidth=1.7, transform=DATA_CRS, zorder=5)
    ax.plot(np.full(na_side_lat.size, -60.0), na_side_lat, color=NA_COLOR, linewidth=1.7, transform=DATA_CRS, zorder=5)
    ax.plot(np.full(na_side_lat.size, 30.0), na_side_lat, color=NA_COLOR, linewidth=1.7, transform=DATA_CRS, zorder=5)
    ax.plot(np_lon_east, np.full(np_lon_east.size, 45.0), color=NP_COLOR, linewidth=1.7, transform=DATA_CRS, zorder=5)
    ax.plot(np_lon_west, np.full(np_lon_west.size, 45.0), color=NP_COLOR, linewidth=1.7, transform=DATA_CRS, zorder=5)
    ax.plot(np.full(np_side_lat.size, 120.0), np_side_lat, color=NP_COLOR, linewidth=1.7, transform=DATA_CRS, zorder=5)
    ax.plot(np.full(np_side_lat.size, -150.0), np_side_lat, color=NP_COLOR, linewidth=1.7, transform=DATA_CRS, zorder=5)

    # %% 9.6 清晰经纬度文字
    for lon_value, lat_value, label, rotation in lon_label_specs:
        text = ax.text(lon_value, lat_value, label, transform=DATA_CRS, ha="center", va="center", fontsize=COORD_FONTSIZE, color=COORD_COLOR, rotation=rotation, rotation_mode="anchor", zorder=6)
        text.set_path_effects([path_effects.withStroke(linewidth=COORD_STROKE, foreground="white")])
    for lon_value, lat_value, label, rotation in lat_label_specs:
        text = ax.text(lon_value, lat_value, label, transform=DATA_CRS, ha="center", va="center", fontsize=COORD_FONTSIZE, color=COORD_COLOR, rotation=rotation, rotation_mode="anchor", zorder=6)
        text.set_path_effects([path_effects.withStroke(linewidth=COORD_STROKE, foreground="white")])

    # %% 9.7 区域科学量
    sia_ca = float(sia_lookup.loc[(model, "Central Arctic")])
    sia_na = float(sia_lookup.loc[(model, "North Atlantic")])
    sia_np = float(sia_lookup.loc[(model, "North Pacific")])
    sia_arctic = float(sia_lookup.loc[(model, "Arctic total (45-90N)")])
    thetao_ca = float(thetao_lookup.loc[(model, "Central Arctic")])
    thetao_na = float(thetao_lookup.loc[(model, "North Atlantic")])
    thetao_np = float(thetao_lookup.loc[(model, "North Pacific")])
    tas_ca = float(tas_lookup.loc[(model, "Central Arctic")])
    tas_na = float(tas_lookup.loc[(model, "North Atlantic")])
    tas_np = float(tas_lookup.loc[(model, "North Pacific")])

    # %% 9.8 panel标题
    panel_letter = string.ascii_lowercase[index]
    ax.set_title(f"({panel_letter})  {model}", loc="left", pad=7, fontsize=9.5, fontweight="bold")

    # %% 9.9 区域联合标注
    ca_label = f"CA\nΔSIA {sia_ca:+.2f}\n" + rf"$\Delta\theta_o$ {thetao_ca:+.2f}°C" + "\n" + rf"$\Delta T_{{as}}$ {tas_ca:+.2f}°C"
    na_label = f"NA\nΔSIA {sia_na:+.2f}\n" + rf"$\Delta\theta_o$ {thetao_na:+.2f}°C" + "\n" + rf"$\Delta T_{{as}}$ {tas_na:+.2f}°C"
    np_label = f"NP\nΔSIA {sia_np:+.2f}\n" + rf"$\Delta\theta_o$ {thetao_np:+.2f}°C" + "\n" + rf"$\Delta T_{{as}}$ {tas_np:+.2f}°C"
    text_ca = ax.text(0.50, 0.59, ca_label, transform=ax.transAxes, ha="center", va="center", fontsize=5.8, color=CA_COLOR, fontweight="bold", linespacing=0.92, zorder=7)
    text_na = ax.text(-15.0, 53.0, na_label, transform=DATA_CRS, ha="center", va="center", fontsize=5.9, color=NA_COLOR, fontweight="bold", linespacing=0.95, zorder=7)
    text_np = ax.text(0.61, 0.89, np_label, transform=ax.transAxes, ha="center", va="center", fontsize=5.9, color=NP_COLOR, fontweight="bold", linespacing=0.95, zorder=7)
    for text in [text_ca, text_na, text_np]:
        text.set_path_effects([path_effects.withStroke(linewidth=2.1, foreground="white")])

    # %% 9.10 Arctic total
    ax.text(0.5, -0.055, f"Arctic total (45°–90°N) = {sia_arctic:+.2f} × 10$^6$ km$^2$", transform=ax.transAxes, ha="center", va="top", fontsize=7.0, color="#222222")
    ax.spines["geo"].set_linewidth(0.75)
    ax.spines["geo"].set_edgecolor("#333333")

# %% 10. 公共图例
legend_handles = [
    Line2D([0], [0], color=CA_COLOR, lw=2.2, label="Central Arctic (CA; 70–90°N)"),
    Line2D([0], [0], color=NA_COLOR, lw=2.2, label="North Atlantic (NA; 60°W–30°E, south of 70°N)"),
    Line2D([0], [0], color=NP_COLOR, lw=2.2, label="North Pacific (NP; 120°E–150°W, south of 70°N)"),
]
if obs_available:
    legend_handles.append(Line2D([0], [0], color=OBS_EDGE_COLOR, lw=1.05, ls=":", label="Observed climatology (SIC = 15%)"))
legend_handles.extend([
    Line2D([0], [0], color=CO2_EDGE_COLOR, lw=1.35, ls="-", label="Raw 1pctCO$_2$ initial (SIC = 15%)"),
    Line2D([0], [0], color=CDR_EDGE_COLOR, lw=1.40, ls="--", label="Raw CDR final (SIC = 15%)"),
])
fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, 0.072), ncol=3, frameon=False, fontsize=7.8, handlelength=2.8, handletextpad=0.6, columnspacing=1.4)

# %% 11. 色标
cax = fig.add_axes([0.17, 0.027, 0.66, 0.017])
cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal", extend="both")
cbar.set_label(r"Grid-cell $\Delta$SIA ($10^3$ km$^2$ per 1° bin)", fontsize=8.3)
cbar.ax.tick_params(labelsize=7.5, length=3)

# %% 12. 布局
fig.subplots_adjust(left=0.035, right=0.965, top=0.935, bottom=0.125, wspace=0.08, hspace=0.19)

# %% 13. 覆盖保存
fig.savefig(PNG_FILE, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
fig.savefig(PDF_FILE, bbox_inches="tight", pad_inches=0.06, facecolor="white")
fig.savefig(SVG_FILE, bbox_inches="tight", pad_inches=0.06, facecolor="white")
plt.close(fig)
print(f"[覆盖保存] {PNG_FILE}")
print(f"[覆盖保存] {PDF_FILE}")
print(f"[覆盖保存] {SVG_FILE}")
print(f"[使用的新温度summary] {TEMPERATURE_SUMMARY_FILE}")