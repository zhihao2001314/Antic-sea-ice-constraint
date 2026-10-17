# %% 1. 导入依赖

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


# %% 2. 参数与路径设置

project_root = Path(r"F:\sea\Antic sea ice constraint")
output_path = (
    project_root
    / "output"
    / "s02_arctic_region"
    / "p01_regional_sia"
    / "n01"
    / "arctic_regional_sia_annual.csv"
)
figure_directory = project_root / "figure" / "s02_arctic_region"
figure_directory.mkdir(parents=True, exist_ok=True)

figure_png = figure_directory / "arctic_regional_sia_nine_models.png"
figure_pdf = figure_directory / "arctic_regional_sia_nine_models.pdf"

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

arctic_total_name = "Arctic total 45-90N"
central_arctic_name = "Central Arctic"
north_atlantic_name = "North Atlantic"
north_pacific_name = "North Pacific"

arctic_total_color = "#111111"
central_arctic_color = "#A64CE6"
north_atlantic_color = "#F28E1C"
north_pacific_color = "#0072B2"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.5,
        "axes.linewidth": 0.9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }
)


# %% 3. 读取绘图数据

annual_dataframe = pd.read_csv(output_path)

plot_dataframe = annual_dataframe.loc[
    annual_dataframe["region"].isin(
        [
            arctic_total_name,
            central_arctic_name,
            north_atlantic_name,
            north_pacific_name,
        ]
    )
].copy()

smooth_column = "picontrol_corrected_sia_11yr_mean_million_km2"
annual_column = "picontrol_corrected_sia_million_km2"


# %% 4. 绘制九模式区域海冰时间序列

figure, axes = plt.subplots(
    3,
    3,
    figsize=(12.0, 9.2),
    sharex=True,
    sharey=True,
)
north_atlantic_axes = []
north_pacific_axes = []
panel_letters = list("abcdefghi")

for model_index, model_name in enumerate(model_order):
    row_index = model_index // 3
    column_index = model_index % 3
    axis = axes[row_index, column_index]
    north_atlantic_axis = axis.twinx()
    north_pacific_axis = axis.twinx()
    north_pacific_axis.spines["right"].set_position(("axes", 1.18))
    north_atlantic_axes.append(north_atlantic_axis)
    north_pacific_axes.append(north_pacific_axis)

    model_dataframe = plot_dataframe.loc[
        plot_dataframe["model"] == model_name
    ]
    arctic_total_dataframe = model_dataframe.loc[
        model_dataframe["region"] == arctic_total_name
    ]
    central_arctic_dataframe = model_dataframe.loc[
        model_dataframe["region"] == central_arctic_name
    ]
    north_atlantic_dataframe = model_dataframe.loc[
        model_dataframe["region"] == north_atlantic_name
    ]
    north_pacific_dataframe = model_dataframe.loc[
        model_dataframe["region"] == north_pacific_name
    ]

    axis.plot(
        arctic_total_dataframe["nominal_year"],
        arctic_total_dataframe[annual_column],
        color=arctic_total_color,
        linewidth=0.7,
        alpha=0.18,
    )
    axis.plot(
        central_arctic_dataframe["nominal_year"],
        central_arctic_dataframe[annual_column],
        color=central_arctic_color,
        linewidth=0.7,
        alpha=0.20,
    )
    north_atlantic_axis.plot(
        north_atlantic_dataframe["nominal_year"],
        north_atlantic_dataframe[annual_column],
        color=north_atlantic_color,
        linewidth=0.7,
        alpha=0.20,
    )
    north_pacific_axis.plot(
        north_pacific_dataframe["nominal_year"],
        north_pacific_dataframe[annual_column],
        color=north_pacific_color,
        linewidth=0.7,
        alpha=0.20,
    )

    axis.plot(
        arctic_total_dataframe["nominal_year"],
        arctic_total_dataframe[smooth_column],
        color=arctic_total_color,
        linewidth=2.0,
        zorder=4,
    )
    axis.plot(
        central_arctic_dataframe["nominal_year"],
        central_arctic_dataframe[smooth_column],
        color=central_arctic_color,
        linewidth=1.8,
        zorder=4,
    )
    north_atlantic_axis.plot(
        north_atlantic_dataframe["nominal_year"],
        north_atlantic_dataframe[smooth_column],
        color=north_atlantic_color,
        linewidth=1.8,
        zorder=4,
    )
    north_pacific_axis.plot(
        north_pacific_dataframe["nominal_year"],
        north_pacific_dataframe[smooth_column],
        color=north_pacific_color,
        linewidth=1.8,
        zorder=4,
    )

    axis.axvline(2140, color="#9A9A9A", linewidth=0.8, zorder=0)
    axis.set_xlim(2000, 2280)
    axis.set_ylim(0.0, 13.5)
    axis.set_xticks([2000, 2140, 2280])
    axis.set_yticks([0, 4, 8, 12])
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)

    north_atlantic_axis.set_ylim(0.0, 6.0)
    north_atlantic_axis.set_yticks([0.0, 2.0, 4.0, 6.0])
    north_atlantic_axis.tick_params(axis="y", colors=north_atlantic_color)
    north_atlantic_axis.spines["right"].set_color(north_atlantic_color)
    north_atlantic_axis.spines["top"].set_visible(False)

    north_pacific_axis.set_ylim(-0.15, 1.05)
    north_pacific_axis.set_yticks([0.0, 0.4, 0.8])
    north_pacific_axis.tick_params(axis="y", colors=north_pacific_color)
    north_pacific_axis.spines["right"].set_color(north_pacific_color)
    north_pacific_axis.spines["top"].set_visible(False)

    if column_index != 2:
        north_atlantic_axis.set_yticklabels([])
        north_atlantic_axis.spines["right"].set_visible(False)
        north_pacific_axis.set_yticklabels([])
        north_pacific_axis.spines["right"].set_visible(False)

    if column_index == 0:
        axis.set_ylabel(r"Arctic total / CA SIA ($10^6$ km$^2$)")

    if column_index == 2:
        north_atlantic_axis.set_ylabel(
            r"NA SIA ($10^6$ km$^2$)",
            color=north_atlantic_color,
            labelpad=7,
        )
        north_pacific_axis.set_ylabel(
            r"NP SIA ($10^6$ km$^2$)",
            color=north_pacific_color,
            labelpad=7,
        )

    if row_index == 2:
        axis.set_xlabel("Nominal year")

    axis.set_title(
        f"({panel_letters[model_index]}) {model_name}",
        loc="left",
        fontweight="bold",
        pad=7,
    )
    axis.spines["top"].set_visible(False)

legend_handles = [
    Line2D(
        [0],
        [0],
        color=arctic_total_color,
        linewidth=2.2,
        label="Arctic total (45–90°N)",
    ),
    Line2D(
        [0],
        [0],
        color=central_arctic_color,
        linewidth=2.0,
        label="Central Arctic (CA)",
    ),
    Line2D(
        [0],
        [0],
        color=north_atlantic_color,
        linewidth=2.0,
        label="North Atlantic (NA; inner right axis)",
    ),
    Line2D(
        [0],
        [0],
        color=north_pacific_color,
        linewidth=2.0,
        label="North Pacific (NP; outer right axis)",
    ),
]

figure.suptitle(
    r"Arctic regional SIA through 1pctCO$_2$ and CDR",
    fontsize=15.0,
    fontweight="bold",
    y=0.988,
)
figure.text(
    0.5,
    0.955,
    "piControl-corrected annual values (thin) and 11-year running means (thick)",
    ha="center",
    va="top",
    color="#555555",
    fontsize=9.5,
)
figure.legend(
    handles=legend_handles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.012),
    ncol=4,
    frameon=False,
)
figure.subplots_adjust(
    left=0.075,
    right=0.855,
    bottom=0.095,
    top=0.905,
    wspace=0.19,
    hspace=0.27,
)


# %% 5. 保存图片

figure.savefig(figure_png, dpi=400)
figure.savefig(figure_pdf)
plt.close(figure)

print(figure_png)
