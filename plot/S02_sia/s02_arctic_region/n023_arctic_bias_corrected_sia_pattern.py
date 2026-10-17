# -*- coding: utf-8 -*-
"""
绘制北极年平均bias-corrected SIA later−early pattern。

本脚本只读取output中的计算结果，不读取原始模式数据，
不执行bias correction，也不进行piControl漂移订正。
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
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from cartopy.util import add_cyclic_point
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


# %% 2. 参数与路径设置

project_root = Path(r"F:\sea\Antic sea ice constraint")
data_directory = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p16_bias_corrected_pattern"
    / "n023_arctic_sia_pattern"
)
temperature_directory = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p04_pattern_unadjusted"
    / "n02_temperature_pattern"
)
figure_directory = (
    project_root
    / "figure"
    / "s02_sia"
    / "s02_arctic_region"
    / "p16_bias_corrected_pattern"
    / "n023_arctic_sia_pattern"
)
figure_directory.mkdir(parents=True, exist_ok=True)

cell_file = data_directory / "arctic_bias_corrected_sia_pattern_cells.csv"
summary_file = data_directory / "arctic_bias_corrected_sia_pattern_summary.csv"
edge_file = data_directory / "arctic_bias_corrected_sia_pattern_edges.npz"
metadata_file = data_directory / "arctic_bias_corrected_sia_pattern_metadata.json"
temperature_file = temperature_directory / "arctic_temperature_region_summary.csv"
observation_edge_file = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p02_pattern_region"
    / "n02"
    / "observed_sic_1988_2020_climatology_edge.npz"
)

png_file = figure_directory / "arctic_bias_corrected_sia_partition_pattern.png"
pdf_file = figure_directory / "arctic_bias_corrected_sia_partition_pattern.pdf"
svg_file = figure_directory / "arctic_bias_corrected_sia_partition_pattern.svg"

model_order = [
    "ACCESS-ESM1-5",
    "CanESM5",
    "CESM2",
    "CNRM-ESM2-1",
    "GFDL-ESM4",
    "MIROC-ES2L",
    "UKESM1-0-LL",
    "CAS-ESM2-0",
    "NorESM2-LM",
]

for required_file in [
    cell_file,
    summary_file,
    edge_file,
    metadata_file,
    temperature_file,
]:
    if not required_file.is_file():
        raise FileNotFoundError(f"缺少计算结果：{required_file}")


# %% 3. 读取bias-corrected SIA pattern

cells = pd.read_csv(cell_file)
summary = pd.read_csv(summary_file)
metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
edges = np.load(edge_file)

required_cell_columns = {
    "model",
    "latitude_deg_north",
    "longitude_deg_east",
    "bias_corrected_display_delta_sia_thousand_km2",
}
required_summary_columns = {
    "model",
    "region",
    "bias_corrected_delta_sia_million_km2",
}
if not required_cell_columns.issubset(cells.columns):
    raise KeyError(
        f"pattern cells缺少字段：{sorted(required_cell_columns - set(cells.columns))}"
    )
if not required_summary_columns.issubset(summary.columns):
    raise KeyError(
        f"pattern summary缺少字段：{sorted(required_summary_columns - set(summary.columns))}"
    )

latitude = np.sort(cells["latitude_deg_north"].unique())
longitude = np.sort(cells["longitude_deg_east"].unique())
if latitude.size != 45 or longitude.size != 360:
    raise ValueError(
        f"共同pattern网格应为45×360，实际={latitude.size}×{longitude.size}"
    )

edge_models = [str(value) for value in edges["models"]]
if edge_models != model_order:
    raise ValueError(f"edge模式顺序不一致：{edge_models}")
edge_latitude = np.asarray(edges["latitude_deg_north"], dtype=float)
edge_longitude = np.asarray(edges["longitude_deg_east"], dtype=float)
edge_early = np.asarray(edges["early_bias_corrected_sic_fraction"], dtype=float)
edge_later = np.asarray(edges["later_bias_corrected_sic_fraction"], dtype=float)
expected_edge_shape = (len(model_order), latitude.size, longitude.size)
if edge_early.shape != expected_edge_shape:
    raise ValueError(f"early edge数组维度错误：{edge_early.shape}")
if edge_later.shape != expected_edge_shape:
    raise ValueError(f"later edge数组维度错误：{edge_later.shape}")
if not np.array_equal(edge_latitude, latitude):
    raise ValueError("edge纬度与pattern纬度不一致")
if not np.array_equal(edge_longitude, longitude):
    raise ValueError("edge经度与pattern经度不一致")

sia_lookup = summary.set_index(["model", "region"])[
    "bias_corrected_delta_sia_million_km2"
]
color_limit = float(metadata["color_limit_thousand_km2"])


# %% 4. 读取既有温度区域结果

temperature_summary = pd.read_csv(temperature_file)
required_temperature_columns = {
    "model",
    "region",
    "raw_delta_thetao_degC",
    "raw_delta_tas_degC",
}
if not required_temperature_columns.issubset(temperature_summary.columns):
    raise KeyError(
        "温度summary缺少字段："
        f"{sorted(required_temperature_columns - set(temperature_summary.columns))}"
    )

temperature_summary = temperature_summary[
    ["model", "region", "raw_delta_thetao_degC", "raw_delta_tas_degC"]
].copy()
temperature_summary["region"] = temperature_summary["region"].replace(
    {
        "NA": "North Atlantic",
        "north_atlantic": "North Atlantic",
        "CA": "Central Arctic",
        "central_arctic": "Central Arctic",
        "NP": "North Pacific",
        "north_pacific": "North Pacific",
    }
)
temperature_summary = temperature_summary.loc[
    temperature_summary["region"].isin(
        ["Central Arctic", "North Atlantic", "North Pacific"]
    )
].drop_duplicates(["model", "region"], keep="last")
thetao_lookup = temperature_summary.set_index(["model", "region"])[
    "raw_delta_thetao_degC"
]
tas_lookup = temperature_summary.set_index(["model", "region"])[
    "raw_delta_tas_degC"
]

for model in model_order:
    for region_name in ["Central Arctic", "North Atlantic", "North Pacific"]:
        if (model, region_name) not in thetao_lookup.index:
            raise KeyError(f"thetao summary缺少：{model}, {region_name}")
        if (model, region_name) not in tas_lookup.index:
            raise KeyError(f"tas summary缺少：{model}, {region_name}")


# %% 5. 读取观测海冰边缘

observation_edge_available = observation_edge_file.is_file()
if observation_edge_available:
    observation_edge = np.load(observation_edge_file)
    observation_x = np.asarray(observation_edge["projected_x_m"], dtype=float)
    observation_y = np.asarray(observation_edge["projected_y_m"], dtype=float)
    observation_sic = np.asarray(
        observation_edge["sic_smoothed_for_edge_fraction"],
        dtype=float,
    )


# %% 6. 绘图样式和分区边界

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8.5,
        "axes.titlesize": 9.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

data_crs = ccrs.PlateCarree()
map_crs = ccrs.NorthPolarStereo(central_longitude=0.0)
nsidc_crs = ccrs.epsg(3411)

central_arctic_color = "#2166AC"
north_atlantic_color = "#D95F02"
north_pacific_color = "#1B9E77"
early_edge_color = "#00A6D6"
later_edge_color = "#9C4DCC"
observation_edge_color = "#000000"
land_color = "#D8D8D8"
coast_color = "#8A8A8A"
grid_color = "#AFAFAF"
coordinate_color = "#555555"

color_map = plt.get_cmap("RdBu_r").copy()
color_map.set_bad("#F7F7F7")
color_norm = TwoSlopeNorm(
    vmin=-color_limit,
    vcenter=0.0,
    vmax=color_limit,
)

theta = np.linspace(0.0, 2.0 * np.pi, 361)
circle = mpath.Path(
    np.column_stack(
        [0.5 + 0.5 * np.sin(theta), 0.5 + 0.5 * np.cos(theta)]
    )
)

central_arctic_longitude = np.linspace(-180.0, 180.0, 721)
north_atlantic_longitude = np.linspace(-60.0, 30.0, 181)
north_atlantic_side_latitude = np.linspace(45.0, 70.0, 101)
north_pacific_east_longitude = np.linspace(120.0, 180.0, 121)
north_pacific_west_longitude = np.linspace(-180.0, -150.0, 61)
north_pacific_side_latitude = np.linspace(45.0, 70.0, 101)

longitude_label_specs = [
    (180.0, 46.4, "180°W", 0),
    (-120.0, 46.8, "120°W", -30),
    (-60.0, 46.8, "60°W", 30),
    (0.0, 46.2, "0°", 0),
    (60.0, 46.8, "60°E", -30),
    (120.0, 46.8, "120°E", 30),
]
latitude_label_specs = [
    (67.0, 60.0, "60°N", -26),
    (55.0, 70.0, "70°N", -18),
    (18.0, 80.0, "80°N", -5),
]


# %% 7. 创建九模式极地图

figure = plt.figure(figsize=(12.0, 13.2))
axes = [
    figure.add_subplot(3, 3, index + 1, projection=map_crs)
    for index in range(9)
]
figure.suptitle(
    "Arctic bias-corrected annual-mean sea-ice-area change "
    "(CDR years 131–140 − 1pctCO$_2$ years 1–10)",
    x=0.5,
    y=0.985,
    fontsize=13.0,
    fontweight="bold",
)

color_mesh = None
for model_index, model in enumerate(model_order):
    axis = axes[model_index]
    model_cells = cells.loc[cells["model"] == model].sort_values(
        ["latitude_deg_north", "longitude_deg_east"]
    )
    if len(model_cells) != latitude.size * longitude.size:
        raise ValueError(f"{model} pattern格点不完整：{len(model_cells)}")

    pattern = model_cells[
        "bias_corrected_display_delta_sia_thousand_km2"
    ].to_numpy(float).reshape(latitude.size, longitude.size)
    pattern_cyclic, longitude_cyclic = add_cyclic_point(
        pattern,
        coord=longitude,
    )
    longitude_grid, latitude_grid = np.meshgrid(longitude_cyclic, latitude)

    axis.set_extent([-180.0, 180.0, 45.0, 90.0], crs=data_crs)
    axis.set_boundary(circle, transform=axis.transAxes)
    color_mesh = axis.pcolormesh(
        longitude_grid,
        latitude_grid,
        np.ma.masked_invalid(pattern_cyclic),
        transform=data_crs,
        cmap=color_map,
        norm=color_norm,
        shading="auto",
        rasterized=True,
        zorder=1,
    )

    gridliner = axis.gridlines(
        crs=data_crs,
        draw_labels=False,
        linewidth=0.40,
        color=grid_color,
        alpha=0.42,
        linestyle=":",
        zorder=2,
    )
    gridliner.xlocator = mticker.FixedLocator([-180, -120, -60, 0, 60, 120])
    gridliner.ylocator = mticker.FixedLocator([60, 70, 80])

    early_field = edge_early[model_index]
    later_field = edge_later[model_index]
    early_field_cyclic, _ = add_cyclic_point(
        early_field,
        coord=longitude,
    )
    later_field_cyclic, _ = add_cyclic_point(
        later_field,
        coord=longitude,
    )
    if np.nanmin(early_field_cyclic) <= 0.15 <= np.nanmax(early_field_cyclic):
        axis.contour(
            longitude_grid,
            latitude_grid,
            np.ma.masked_invalid(early_field_cyclic),
            levels=[0.15],
            colors=[early_edge_color],
            linewidths=1.15,
            linestyles="-",
            transform=data_crs,
            zorder=3.5,
        )
    if np.nanmin(later_field_cyclic) <= 0.15 <= np.nanmax(later_field_cyclic):
        axis.contour(
            longitude_grid,
            latitude_grid,
            np.ma.masked_invalid(later_field_cyclic),
            levels=[0.15],
            colors=[later_edge_color],
            linewidths=1.20,
            linestyles="--",
            transform=data_crs,
            zorder=3.6,
        )
    if observation_edge_available:
        axis.contour(
            observation_x,
            observation_y,
            np.ma.masked_invalid(observation_sic),
            levels=[0.15],
            colors=[observation_edge_color],
            linewidths=0.95,
            linestyles=":",
            transform=nsidc_crs,
            zorder=3.7,
        )

    axis.add_feature(cfeature.LAND, facecolor=land_color, edgecolor="none", zorder=4)
    axis.coastlines(
        resolution="110m",
        color=coast_color,
        linewidth=0.45,
        zorder=4.1,
    )

    axis.plot(
        central_arctic_longitude,
        np.full(central_arctic_longitude.size, 70.0),
        color=central_arctic_color,
        linewidth=1.9,
        transform=data_crs,
        zorder=5,
    )
    axis.plot(
        north_atlantic_longitude,
        np.full(north_atlantic_longitude.size, 45.0),
        color=north_atlantic_color,
        linewidth=1.7,
        transform=data_crs,
        zorder=5,
    )
    axis.plot(
        np.full(north_atlantic_side_latitude.size, -60.0),
        north_atlantic_side_latitude,
        color=north_atlantic_color,
        linewidth=1.7,
        transform=data_crs,
        zorder=5,
    )
    axis.plot(
        np.full(north_atlantic_side_latitude.size, 30.0),
        north_atlantic_side_latitude,
        color=north_atlantic_color,
        linewidth=1.7,
        transform=data_crs,
        zorder=5,
    )
    axis.plot(
        north_pacific_east_longitude,
        np.full(north_pacific_east_longitude.size, 45.0),
        color=north_pacific_color,
        linewidth=1.7,
        transform=data_crs,
        zorder=5,
    )
    axis.plot(
        north_pacific_west_longitude,
        np.full(north_pacific_west_longitude.size, 45.0),
        color=north_pacific_color,
        linewidth=1.7,
        transform=data_crs,
        zorder=5,
    )
    axis.plot(
        np.full(north_pacific_side_latitude.size, 120.0),
        north_pacific_side_latitude,
        color=north_pacific_color,
        linewidth=1.7,
        transform=data_crs,
        zorder=5,
    )
    axis.plot(
        np.full(north_pacific_side_latitude.size, -150.0),
        north_pacific_side_latitude,
        color=north_pacific_color,
        linewidth=1.7,
        transform=data_crs,
        zorder=5,
    )

    for longitude_value, latitude_value, label, rotation in longitude_label_specs:
        coordinate_text = axis.text(
            longitude_value,
            latitude_value,
            label,
            transform=data_crs,
            ha="center",
            va="center",
            fontsize=6.0,
            color=coordinate_color,
            rotation=rotation,
            rotation_mode="anchor",
            zorder=6,
        )
        coordinate_text.set_path_effects(
            [path_effects.withStroke(linewidth=1.6, foreground="white")]
        )
    for longitude_value, latitude_value, label, rotation in latitude_label_specs:
        coordinate_text = axis.text(
            longitude_value,
            latitude_value,
            label,
            transform=data_crs,
            ha="center",
            va="center",
            fontsize=6.0,
            color=coordinate_color,
            rotation=rotation,
            rotation_mode="anchor",
            zorder=6,
        )
        coordinate_text.set_path_effects(
            [path_effects.withStroke(linewidth=1.6, foreground="white")]
        )

    sia_central_arctic = float(sia_lookup.loc[(model, "Central Arctic")])
    sia_north_atlantic = float(sia_lookup.loc[(model, "North Atlantic")])
    sia_north_pacific = float(sia_lookup.loc[(model, "North Pacific")])
    sia_total = float(sia_lookup.loc[(model, "Total (CA+NA+NP)")])
    thetao_central_arctic = float(thetao_lookup.loc[(model, "Central Arctic")])
    thetao_north_atlantic = float(thetao_lookup.loc[(model, "North Atlantic")])
    thetao_north_pacific = float(thetao_lookup.loc[(model, "North Pacific")])
    tas_central_arctic = float(tas_lookup.loc[(model, "Central Arctic")])
    tas_north_atlantic = float(tas_lookup.loc[(model, "North Atlantic")])
    tas_north_pacific = float(tas_lookup.loc[(model, "North Pacific")])

    panel_letter = string.ascii_lowercase[model_index]
    axis.set_title(
        f"({panel_letter})  {model}",
        loc="left",
        pad=7,
        fontsize=9.5,
        fontweight="bold",
    )

    central_arctic_label = (
        f"CA\nΔSIA {sia_central_arctic:+.2f}\n"
        + rf"$\Delta\theta_o$ {thetao_central_arctic:+.2f}°C"
        + "\n"
        + rf"$\Delta T_{{as}}$ {tas_central_arctic:+.2f}°C"
    )
    north_atlantic_label = (
        f"NA\nΔSIA {sia_north_atlantic:+.2f}\n"
        + rf"$\Delta\theta_o$ {thetao_north_atlantic:+.2f}°C"
        + "\n"
        + rf"$\Delta T_{{as}}$ {tas_north_atlantic:+.2f}°C"
    )
    north_pacific_label = (
        f"NP\nΔSIA {sia_north_pacific:+.2f}\n"
        + rf"$\Delta\theta_o$ {thetao_north_pacific:+.2f}°C"
        + "\n"
        + rf"$\Delta T_{{as}}$ {tas_north_pacific:+.2f}°C"
    )
    central_arctic_text = axis.text(
        0.50,
        0.59,
        central_arctic_label,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=5.8,
        color=central_arctic_color,
        fontweight="bold",
        linespacing=0.92,
        zorder=7,
    )
    north_atlantic_text = axis.text(
        -15.0,
        53.0,
        north_atlantic_label,
        transform=data_crs,
        ha="center",
        va="center",
        fontsize=5.9,
        color=north_atlantic_color,
        fontweight="bold",
        linespacing=0.95,
        zorder=7,
    )
    north_pacific_text = axis.text(
        0.61,
        0.89,
        north_pacific_label,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=5.9,
        color=north_pacific_color,
        fontweight="bold",
        linespacing=0.95,
        zorder=7,
    )
    for scientific_text in [
        central_arctic_text,
        north_atlantic_text,
        north_pacific_text,
    ]:
        scientific_text.set_path_effects(
            [path_effects.withStroke(linewidth=2.1, foreground="white")]
        )

    axis.text(
        0.5,
        -0.055,
        f"Total (CA+NA+NP) = {sia_total:+.2f} × 10$^6$ km$^2$",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.0,
        color="#222222",
    )
    axis.spines["geo"].set_linewidth(0.75)
    axis.spines["geo"].set_edgecolor("#333333")


# %% 8. 图例、色标和保存

legend_handles = [
    Line2D(
        [0],
        [0],
        color=central_arctic_color,
        linewidth=2.2,
        label="Central Arctic (CA; 70–90°N)",
    ),
    Line2D(
        [0],
        [0],
        color=north_atlantic_color,
        linewidth=2.2,
        label="North Atlantic (NA; 60°W–30°E, south of 70°N)",
    ),
    Line2D(
        [0],
        [0],
        color=north_pacific_color,
        linewidth=2.2,
        label="North Pacific (NP; 120°E–150°W, south of 70°N)",
    ),
]
if observation_edge_available:
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color=observation_edge_color,
            linewidth=1.05,
            linestyle=":",
            label="Observed climatology (SIC = 15%)",
        )
    )
legend_handles.extend(
    [
        Line2D(
            [0],
            [0],
            color=early_edge_color,
            linewidth=1.35,
            linestyle="-",
            label="Bias-corrected 1pctCO$_2$ early (SIC = 15%)",
        ),
        Line2D(
            [0],
            [0],
            color=later_edge_color,
            linewidth=1.40,
            linestyle="--",
            label="Bias-corrected CDR later (SIC = 15%)",
        ),
    ]
)
figure.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.072),
    ncol=3,
    frameon=False,
    fontsize=7.8,
    handlelength=2.8,
    handletextpad=0.6,
    columnspacing=1.4,
)

color_axis = figure.add_axes([0.17, 0.027, 0.66, 0.017])
color_bar = figure.colorbar(
    color_mesh,
    cax=color_axis,
    orientation="horizontal",
    extend="both",
)
color_bar.set_label(
    r"Bias-corrected $\Delta$SIA ($10^3$ km$^2$ per 1° bin)",
    fontsize=8.3,
)
color_bar.ax.tick_params(labelsize=7.5, length=3)

figure.subplots_adjust(
    left=0.035,
    right=0.965,
    top=0.935,
    bottom=0.125,
    wspace=0.08,
    hspace=0.19,
)

figure.savefig(
    png_file,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.06,
    facecolor="white",
)
figure.savefig(
    pdf_file,
    bbox_inches="tight",
    pad_inches=0.06,
    facecolor="white",
)
figure.savefig(
    svg_file,
    bbox_inches="tight",
    pad_inches=0.06,
    facecolor="white",
)
plt.close(figure)

print(f"[PNG] {png_file}")
print(f"[PDF] {pdf_file}")
print(f"[SVG] {svg_file}")
