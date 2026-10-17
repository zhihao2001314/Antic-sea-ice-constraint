# -*- coding: utf-8 -*-
"""
绘制8模式North Atlantic March MLD的三个阶段变化拼图，空间范围扩展至85°N。

【是否需要运行】
需要。先运行：
src/s02_sia/s02_arctic_region/n10_mldcalculate.py

【输入文件】
output/s02sia/s02_arctic_region/p06_mld/n02_na_deep_mixing_three_phase/
    <model>_na_march_mld_three_phase_native.npz

【输出文件】
figure/s02_sia/s02_arctic_region/p06_mld/n02_na_deep_mixing_three_phase/
    na_march_mld_ramp_up_8models.png/pdf/svg
    na_march_mld_ramp_down_8models.png/pdf/svg
    na_march_mld_later_early_8models.png/pdf/svg

【什么时候需要重新运行】
1. plot-ready March MLD数据变化；
2. 绘图空间范围、样式、色标或等值线变化。

绘图定义
1. 三张图分别表示ramp-up、ramp-down和later-early变化。
2. 彩色填色为March MLD变化，黑线为每个差值所对应的参考阶段March MLD。
3. North Atlantic显示范围为50–85°N、65°W–20°E。
4. 统一空间范围、统一色标和统一等值线，保证三张图可直接比较。
5. 直接读取各模式原生网格结果；插值仅用于地图显示，不参与科学统计。
6. 不对原始数据范围之外进行外推。
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
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# %% 2. 路径
ROOT = Path(r"F:\sea\Antic sea ice constraint")
DATA_DIR = ROOT/"output"/"s02sia"/"s02_arctic_region"/"p06_mld"/"n02_na_deep_mixing_three_phase"
FIG_DIR = ROOT/"figure"/"s02_sia"/"s02_arctic_region"/"p06_mld"/"n02_namld_three_phase"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% 3. 参数
MODELS = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4", "MIROC-ES2L", "UKESM1-0-LL", "NorESM2-LM"]

LON_MIN, LON_MAX = -65.0, 20.0
LAT_MIN, LAT_MAX = 50.0, 85.0
DISPLAY_RES = 0.25

CONTOUR_LEVELS = [200, 500, 1000, 1500, 2000]
CONTOUR_LABEL_LEVELS = [500, 1000, 2000]

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

# %% 5. 检查输入文件
native_files = {model: DATA_DIR/f"{model}_na_march_mld_three_phase_native.npz" for model in MODELS}
missing_native_files = [str(native_files[model]) for model in MODELS if not native_files[model].is_file()]

if missing_native_files:
    raise FileNotFoundError(f"缺少原生网格结果：{missing_native_files}")

# %% 6. 准备50–85°N统一绘图网格
lon_plot = np.arange(LON_MIN, LON_MAX + DISPLAY_RES*0.5, DISPLAY_RES)
lat_plot = np.arange(LAT_MIN, LAT_MAX + DISPLAY_RES*0.5, DISPLAY_RES)
lon2_plot, lat2_plot = np.meshgrid(lon_plot, lat_plot)
target_points = np.column_stack([lon2_plot.ravel(), lat2_plot.ravel()])

# %% 7. 将8模式原生网格结果插值到显示网格
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
        valid = valid & np.isfinite(native_fields[name])

    if int(valid.sum()) < 4:
        raise ValueError(f"{model}: 有效格点不足，n={int(valid.sum())}")

    valid_lat_min = float(np.nanmin(lat_native[valid]))
    valid_lat_max = float(np.nanmax(lat_native[valid]))
    print(f"[{model}] native valid latitude = {valid_lat_min:.2f}–{valid_lat_max:.2f}°N")

    if valid_lat_max < 84.0:
        print(f"[警告] {model}: 当前NPZ最高有效纬度只有{valid_lat_max:.2f}°N，75–85°N可能缺数据；这种情况需要修改并重跑n10_mldcalculate.py")

    source_points = np.column_stack([lon_native[valid], lat_native[valid]])
    plot_fields[model] = {}

    for name in field_names:
        display_field = griddata(source_points, native_fields[name][valid], target_points, method="linear").reshape(lat2_plot.shape)
        plot_fields[model][name] = display_field

    for name in ["delta_up_march_mld_m", "delta_down_march_mld_m", "delta_later_early_march_mld_m"]:
        finite_values = plot_fields[model][name][np.isfinite(plot_fields[model][name])]
        delta_for_common_scale.append(np.abs(finite_values))

# %% 8. 设置三张图共同色标
all_absolute_delta = np.concatenate(delta_for_common_scale)
color_limit = float(np.ceil(np.nanpercentile(all_absolute_delta, 98)/100.0)*100.0)
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

# %% 9. 绘制三张8模式拼图
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

        ax.set_extent([LON_MIN, LON_MAX, LAT_MIN, LAT_MAX], crs=DATA_CRS)
        ax.set_facecolor(OCEAN_NEUTRAL)

        gridliner = ax.gridlines(
            crs=DATA_CRS,
            draw_labels=True,
            x_inline=False,
            y_inline=False,
            linewidth=0.45,
            color="#9b9b9b",
            linestyle=":",
            alpha=0.65,
            zorder=0,
        )

        gridliner.xlocator = FixedLocator([-60, -40, -20, 0, 20])
        gridliner.ylocator = FixedLocator([50, 55, 60, 65, 70, 75, 80, 85])
        gridliner.top_labels = False
        gridliner.right_labels = False
        gridliner.left_labels = column_index == 0
        gridliner.bottom_labels = row_index == 1
        gridliner.xlabel_style = {"size": 8.5, "color": "#4b4b4b"}
        gridliner.ylabel_style = {"size": 8.5, "color": "#4b4b4b"}

        filled_contours = ax.contourf(
            lon2_plot,
            lat2_plot,
            delta_field,
            levels=color_levels,
            cmap=MLD_CMAP,
            extend="both",
            transform=DATA_CRS,
            antialiased=False,
            zorder=1,
        )

        finite_reference = reference_field[np.isfinite(reference_field)]
        available_contour_levels = [level for level in CONTOUR_LEVELS if finite_reference.size > 0 and float(np.nanmin(finite_reference)) <= level <= float(np.nanmax(finite_reference))]

        if available_contour_levels:
            line_contours = ax.contour(
                lon2_plot,
                lat2_plot,
                reference_field,
                levels=available_contour_levels,
                colors="#202020",
                linewidths=0.72,
                transform=DATA_CRS,
                zorder=3,
            )

            available_label_levels = [level for level in CONTOUR_LABEL_LEVELS if level in available_contour_levels]

            if available_label_levels:
                ax.clabel(
                    line_contours,
                    levels=available_label_levels,
                    fmt="%d",
                    fontsize=6.8,
                    inline=True,
                    inline_spacing=3,
                    colors="#202020",
                )

        ax.add_feature(LAND50, zorder=5)
        ax.add_feature(COAST50, linewidth=0.48, edgecolor="#555555", zorder=6)

        ax.spines["geo"].set_linewidth(0.70)
        ax.spines["geo"].set_edgecolor("#5f5f5f")

        panel_letter = chr(ord("a")+panel_index)
        ax.set_title(f"({panel_letter})  {model}", loc="left", fontsize=12.5, fontweight="bold", pad=5)

    # 共用色标
    colorbar_axis = fig.add_axes([0.255, 0.074, 0.49, 0.024])
    colorbar = fig.colorbar(filled_contours, cax=colorbar_axis, orientation="horizontal", ticks=color_ticks)
    colorbar.set_label("March mixed-layer depth change (m)", fontsize=11.5, labelpad=4)
    colorbar.ax.tick_params(labelsize=9.0, length=3.0, width=0.65, pad=2)
    colorbar.outline.set_linewidth(0.65)

    # Reference MLD图例
    contour_legend = Line2D([0], [0], color="#202020", linewidth=0.9)

    fig.legend(
        [contour_legend],
        ["Reference March MLD"],
        loc="lower left",
        bbox_to_anchor=(0.045, 0.065),
        frameon=False,
        fontsize=10.0,
        handlelength=2.8,
    )

    png_file = FIG_DIR/f"na_march_mld_{phase['name']}_8models.png"
    pdf_file = FIG_DIR/f"na_march_mld_{phase['name']}_8models.pdf"
    svg_file = FIG_DIR/f"na_march_mld_{phase['name']}_8models.svg"

    fig.savefig(png_file, dpi=500, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(pdf_file, facecolor="white", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(svg_file, facecolor="white", bbox_inches="tight", pad_inches=0.05)

    plt.close(fig)

    print(f"[完成] {phase['name']}: {png_file}")

# %% 10. 输出
print("\n"+"="*110)
print("NORTH ATLANTIC MARCH MLD THREE-PHASE MULTIMODEL FIGURES COMPLETE")
print("="*110)
print(f"[display latitude] {LAT_MIN:.0f}–{LAT_MAX:.0f}°N")
print(f"[common color limit] ±{color_limit:.0f} m")
print(f"[figure dir] {FIG_DIR}")
print("="*110)