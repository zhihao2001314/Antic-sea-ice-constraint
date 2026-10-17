# -*- coding: utf-8 -*-
"""
绘制8模式North Atlantic March MLD三个阶段变化，并叠加initial/later海冰边缘。

【是否需要运行】
需要。先运行：
src/s02_sia/s02_arctic_region/n10_mldcalculate.py

【输入文件】
output/s02sia/s02_arctic_region/p06_mld/n02_na_deep_mixing_three_phase/
    <model>_na_march_mld_three_phase_native.npz

data/s02_sia/s02_arctic_region/
    arctic_sia_pattern_values_1deg.csv

【输出文件】
figure/s02_sia/s02_arctic_region/p06_mld/n02_na_deep_mixing_three_phase/
    na_march_mld_ramp_up_8models.png/pdf/svg
    na_march_mld_ramp_down_8models.png/pdf/svg
    na_march_mld_later_early_8models.png/pdf/svg

【什么时候需要重新运行】
MLD结果、海冰pattern结果或绘图样式变化后。

绘图定义
1. 彩色填色为March MLD变化。
2. 黑色实线为Reference March MLD。
3. 紫色虚线为initial sea-ice edge：1pctCO2 years1-10，SIC=15%。
4. 绿色虚线为later sea-ice edge：1pctCO2-cdr years131-140，SIC=15%。
5. North Atlantic显示50-85°N、65°W-20°E。
6. 海冰直接读取已有1° plot-ready结果，不重新计算。
7. 海冰稀疏格点表先恢复到完整1°网格，缺失格点按原海冰工作流填0。
"""

# %% 1. 依赖
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %% 2. 路径
ROOT = Path(r"F:\sea\Antic sea ice constraint")
DATA_DIR = ROOT/"output"/"s02sia"/"s02_arctic_region"/"p06_mld"/"n02_na_deep_mixing_three_phase"
ICE_FILE = ROOT/"data"/"s02_sia"/"s02_arctic_region"/"arctic_sia_pattern_values_1deg.csv"
FIG_DIR = ROOT/"figure"/"s02_sia"/"s02_arctic_region"/"p06_mld"/"n02_mld_sia"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% 3. 参数
MODELS = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4", "MIROC-ES2L", "UKESM1-0-LL", "NorESM2-LM"]

LON_MIN, LON_MAX = -65.0, 20.0
LAT_MIN, LAT_MAX = 50.0, 85.0
DISPLAY_RES = 0.25

CONTOUR_LEVELS = [200, 500, 1000, 1500, 2000]
CONTOUR_LABEL_LEVELS = [500, 1000, 2000]

ICE_EDGE_LEVEL = 0.15
ICE_LATITUDE = np.arange(45.5, 90.0, 1.0)
ICE_LONGITUDE = np.arange(-179.5, 180.0, 1.0)

INITIAL_ICE_COLOR = "#6A3D9A"
LATER_ICE_COLOR = "#2E8B57"
ICE_EDGE_WIDTH = 1.35

MAIN_PROJ = ccrs.LambertConformal(central_longitude=-25.0, central_latitude=67.5, standard_parallels=(50.0, 80.0))
DATA_CRS = ccrs.PlateCarree()
OCEAN_NEUTRAL = "#f5f3ef"

# %% 4. 样式
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10.0,
    "axes.titlesize": 12.5,
    "axes.linewidth": 0.75,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

MLD_CMAP = LinearSegmentedColormap.from_list(
    "mld_change",
    ["#145d9e", "#347fb5", "#70a6ca", "#b6d0df", OCEAN_NEUTRAL, "#ecc8b4", "#df936e", "#ce5144", "#b60026"],
    N=256,
)
MLD_CMAP.set_bad(OCEAN_NEUTRAL)

LAND50 = cfeature.NaturalEarthFeature("physical", "land", "50m", facecolor="#dedede", edgecolor="none")
COAST50 = cfeature.COASTLINE.with_scale("50m")

# %% 5. 输入检查
native_files = {model: DATA_DIR/f"{model}_na_march_mld_three_phase_native.npz" for model in MODELS}
missing_native_files = [str(native_files[model]) for model in MODELS if not native_files[model].is_file()]

if missing_native_files:
    raise FileNotFoundError(f"缺少原生网格结果：{missing_native_files}")

if not ICE_FILE.is_file():
    raise FileNotFoundError(f"缺少海冰plot-ready文件：{ICE_FILE}")

# %% 6. MLD统一显示网格
lon_plot = np.arange(LON_MIN, LON_MAX + DISPLAY_RES*0.5, DISPLAY_RES)
lat_plot = np.arange(LAT_MIN, LAT_MAX + DISPLAY_RES*0.5, DISPLAY_RES)
lon2_plot, lat2_plot = np.meshgrid(lon_plot, lat_plot)
target_points = np.column_stack([lon2_plot.ravel(), lat2_plot.ravel()])

# %% 7. MLD显示场
plot_fields = {}
delta_for_common_scale = []

field_names = [
    "up_initial_march_mld_m",
    "up_final_march_mld_m",
    "down_initial_march_mld_m",
    "down_final_march_mld_m",
    "delta_up_march_mld_m",
    "delta_down_march_mld_m",
    "delta_later_early_march_mld_m",
]

for model in MODELS:
    with np.load(native_files[model]) as native:
        lon_native = native["longitude_deg_east"].astype(float)
        lat_native = native["latitude_deg_north"].astype(float)
        common_valid_mask = native["common_valid_mask"].astype(bool)
        native_fields = {name: native[name].astype(float) for name in field_names}

    valid = common_valid_mask & np.isfinite(lon_native) & np.isfinite(lat_native)

    for name in field_names:
        valid &= np.isfinite(native_fields[name])

    if int(valid.sum()) < 4:
        raise ValueError(f"{model}: 有效MLD格点不足，n={int(valid.sum())}")

    valid_lat_min = float(np.nanmin(lat_native[valid]))
    valid_lat_max = float(np.nanmax(lat_native[valid]))
    print(f"[{model}] native valid latitude = {valid_lat_min:.2f}–{valid_lat_max:.2f}°N")

    if valid_lat_max < 84.0:
        raise ValueError(f"{model}: MLD最高有效纬度只有{valid_lat_max:.2f}°N，请确认已运行50–85°N版n10")

    source_points = np.column_stack([lon_native[valid], lat_native[valid]])
    plot_fields[model] = {}

    for name in field_names:
        plot_fields[model][name] = griddata(source_points, native_fields[name][valid], target_points, method="linear").reshape(lat2_plot.shape)

    for name in ["delta_up_march_mld_m", "delta_down_march_mld_m", "delta_later_early_march_mld_m"]:
        finite_values = plot_fields[model][name][np.isfinite(plot_fields[model][name])]
        delta_for_common_scale.append(np.abs(finite_values))

# %% 8. 读取已有海冰plot-ready数据
ice_columns = [
    "model",
    "latitude_deg_north",
    "longitude_deg_east",
    "edge_raw_co2_sic_fraction",
    "edge_raw_cdr_sic_fraction",
]

ice_data = pd.read_csv(ICE_FILE, usecols=ice_columns)
ice_data["longitude_deg_east"] = (ice_data["longitude_deg_east"] + 180.0)%360.0 - 180.0

full_ice_index = pd.MultiIndex.from_product(
    [ICE_LATITUDE, ICE_LONGITUDE],
    names=["latitude_deg_north", "longitude_deg_east"],
)

ice_lon2_full, ice_lat2_full = np.meshgrid(ICE_LONGITUDE, ICE_LATITUDE)
ice_region_mask = (ICE_LATITUDE >= LAT_MIN) & (ICE_LATITUDE <= LAT_MAX)

ice_fields = {}

for model in MODELS:
    model_ice = ice_data[ice_data["model"] == model].copy()

    if model_ice.empty:
        raise ValueError(f"{model}: 海冰plot-ready数据为空")

    if model_ice.duplicated(["latitude_deg_north", "longitude_deg_east"]).any():
        raise ValueError(f"{model}: 海冰plot-ready存在重复格点")

    model_ice = model_ice.set_index(["latitude_deg_north", "longitude_deg_east"]).reindex(full_ice_index)

    missing_initial = int(model_ice["edge_raw_co2_sic_fraction"].isna().sum())
    missing_later = int(model_ice["edge_raw_cdr_sic_fraction"].isna().sum())

    model_ice["edge_raw_co2_sic_fraction"] = model_ice["edge_raw_co2_sic_fraction"].fillna(0.0)
    model_ice["edge_raw_cdr_sic_fraction"] = model_ice["edge_raw_cdr_sic_fraction"].fillna(0.0)

    initial_full = model_ice["edge_raw_co2_sic_fraction"].to_numpy(float).reshape(ICE_LATITUDE.size, ICE_LONGITUDE.size)
    later_full = model_ice["edge_raw_cdr_sic_fraction"].to_numpy(float).reshape(ICE_LATITUDE.size, ICE_LONGITUDE.size)

    if np.nanmin(initial_full) < 0.0 or np.nanmax(initial_full) > 1.01:
        raise ValueError(f"{model}: initial SIC超出0–1")

    if np.nanmin(later_full) < 0.0 or np.nanmax(later_full) > 1.01:
        raise ValueError(f"{model}: later SIC超出0–1")

    ice_fields[model] = {
        "lon": ice_lon2_full[ice_region_mask],
        "lat": ice_lat2_full[ice_region_mask],
        "initial": initial_full[ice_region_mask],
        "later": later_full[ice_region_mask],
    }

    print(f"[{model}] sea ice source rows={len(ice_data[ice_data['model'] == model])} | restored full 1° grid | filled initial/later={missing_initial}/{missing_later}")

# %% 9. 三张图共同色标
all_absolute_delta = np.concatenate(delta_for_common_scale)
color_limit = float(np.ceil(np.nanpercentile(all_absolute_delta, 98.0)/100.0)*100.0)
color_limit = max(color_limit, 500.0)
color_levels = np.linspace(-color_limit, color_limit, 41)
color_ticks = np.linspace(-color_limit, color_limit, 5)

phase_settings = [
    {
        "name": "ramp_up",
        "title": "North Atlantic March mixed-layer depth change: ramp-up",
        "reference": "up_initial_march_mld_m",
        "delta": "delta_up_march_mld_m",
    },
    {
        "name": "ramp_down",
        "title": "North Atlantic March mixed-layer depth change: ramp-down",
        "reference": "down_initial_march_mld_m",
        "delta": "delta_down_march_mld_m",
    },
    {
        "name": "later_early",
        "title": "North Atlantic March mixed-layer depth change: later − early",
        "reference": "up_initial_march_mld_m",
        "delta": "delta_later_early_march_mld_m",
    },
]

# %% 10. 绘制三张8模式拼图
for phase in phase_settings:
    fig, axes = plt.subplots(2, 4, figsize=(19.2, 10.8), subplot_kw={"projection": MAIN_PROJ}, facecolor="white")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.895, bottom=0.145, wspace=0.035, hspace=0.13)
    fig.suptitle(phase["title"], fontsize=24.0, fontweight="bold", y=0.965)

    filled_contours = None

    for panel_index, model in enumerate(MODELS):
        row_index = panel_index//4
        column_index = panel_index%4
        ax = axes[row_index, column_index]

        reference_field = plot_fields[model][phase["reference"]]
        delta_field = plot_fields[model][phase["delta"]]

        ice_lon2 = ice_fields[model]["lon"]
        ice_lat2 = ice_fields[model]["lat"]
        initial_sic = ice_fields[model]["initial"]
        later_sic = ice_fields[model]["later"]

        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=DATA_CRS)
        ax.set_facecolor(OCEAN_NEUTRAL)

        gridliner = ax.gridlines(crs=DATA_CRS, draw_labels=True, x_inline=False, y_inline=False, linewidth=0.45, color="#9b9b9b", linestyle=":", alpha=0.65, zorder=0)
        gridliner.xlocator = FixedLocator([-60, -40, -20, 0, 20])
        gridliner.ylocator = FixedLocator([50, 55, 60, 65, 70, 75, 80, 85])
        gridliner.top_labels = False
        gridliner.right_labels = False
        gridliner.left_labels = column_index == 0
        gridliner.bottom_labels = row_index == 1
        gridliner.xlabel_style = {"size": 8.5, "color": "#4b4b4b"}
        gridliner.ylabel_style = {"size": 8.5, "color": "#4b4b4b"}

        filled_contours = ax.contourf(lon2_plot, lat2_plot, delta_field, levels=color_levels, cmap=MLD_CMAP, extend="both", transform=DATA_CRS, antialiased=False, zorder=1)

        finite_reference = reference_field[np.isfinite(reference_field)]
        available_contour_levels = [level for level in CONTOUR_LEVELS if finite_reference.size > 0 and float(np.nanmin(finite_reference)) <= level <= float(np.nanmax(finite_reference))]

        if available_contour_levels:
            line_contours = ax.contour(lon2_plot, lat2_plot, reference_field, levels=available_contour_levels, colors="#202020", linewidths=0.72, transform=DATA_CRS, zorder=3)
            available_label_levels = [level for level in CONTOUR_LABEL_LEVELS if level in available_contour_levels]

            if available_label_levels:
                ax.clabel(line_contours, levels=available_label_levels, fmt="%d", fontsize=6.8, inline=True, inline_spacing=3, colors="#202020")

        initial_finite = initial_sic[np.isfinite(initial_sic)]
        later_finite = later_sic[np.isfinite(later_sic)]

        if initial_finite.size > 0 and np.nanmin(initial_finite) <= ICE_EDGE_LEVEL <= np.nanmax(initial_finite):
            ax.contour(ice_lon2, ice_lat2, initial_sic, levels=[ICE_EDGE_LEVEL], colors=[INITIAL_ICE_COLOR], linewidths=ICE_EDGE_WIDTH, linestyles="--", transform=DATA_CRS, zorder=7)

        if later_finite.size > 0 and np.nanmin(later_finite) <= ICE_EDGE_LEVEL <= np.nanmax(later_finite):
            ax.contour(ice_lon2, ice_lat2, later_sic, levels=[ICE_EDGE_LEVEL], colors=[LATER_ICE_COLOR], linewidths=ICE_EDGE_WIDTH, linestyles="--", transform=DATA_CRS, zorder=8)

        ax.add_feature(LAND50, zorder=9)
        ax.add_feature(COAST50, linewidth=0.48, edgecolor="#555555", zorder=10)

        ax.spines["geo"].set_linewidth(0.70)
        ax.spines["geo"].set_edgecolor("#5f5f5f")

        panel_letter = chr(ord("a") + panel_index)
        ax.set_title(f"({panel_letter})  {model}", loc="left", fontsize=12.5, fontweight="bold", pad=5)

    # 公共色标
    colorbar_axis = fig.add_axes([0.255, 0.074, 0.49, 0.024])
    colorbar = fig.colorbar(filled_contours, cax=colorbar_axis, orientation="horizontal", ticks=color_ticks)
    colorbar.set_label("March mixed-layer depth change (m)", fontsize=11.5, labelpad=4)
    colorbar.ax.tick_params(labelsize=9.0, length=3.0, width=0.65, pad=2)
    colorbar.outline.set_linewidth(0.65)

    # 公共图例
    legend_handles = [
        Line2D([0], [0], color="#202020", linewidth=0.9, linestyle="-"),
        Line2D([0], [0], color=INITIAL_ICE_COLOR, linewidth=ICE_EDGE_WIDTH, linestyle="--"),
        Line2D([0], [0], color=LATER_ICE_COLOR, linewidth=ICE_EDGE_WIDTH, linestyle="--"),
    ]

    legend_labels = [
        "Reference March MLD",
        "Initial sea-ice edge",
        "Later sea-ice edge",
    ]

    fig.legend(legend_handles, legend_labels, loc="lower left", bbox_to_anchor=(0.045, 0.063), ncol=1, frameon=False, fontsize=9.7, handlelength=3.0, handletextpad=0.8, labelspacing=0.55)

    png_file = FIG_DIR/f"na_march_mld_{phase['name']}_8models.png"
    pdf_file = FIG_DIR/f"na_march_mld_{phase['name']}_8models.pdf"
    svg_file = FIG_DIR/f"na_march_mld_{phase['name']}_8models.svg"

    fig.savefig(png_file, dpi=500, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_file, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg_file, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)

    print(f"[完成] {phase['name']}: {png_file}")

# %% 11. 输出
print("\n" + "="*110)
print("NORTH ATLANTIC MARCH MLD + SEA-ICE EDGES COMPLETE")
print("="*110)
print(f"[display latitude] {LAT_MIN:.0f}–{LAT_MAX:.0f}°N")
print(f"[sea-ice edge] initial/later = {ICE_EDGE_LEVEL*100:.0f}% SIC")
print(f"[common color limit] ±{color_limit:.0f} m")
print(f"[figure dir] {FIG_DIR}")
print("="*110)