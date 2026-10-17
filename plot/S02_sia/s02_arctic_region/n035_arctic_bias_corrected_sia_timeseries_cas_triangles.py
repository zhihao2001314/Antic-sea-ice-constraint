# -*- coding: utf-8 -*-
"""
绘制与n023 bias-corrected annual-mean SIA pattern严格一致的280年序列，
并在CAS-ESM2-0曲线上添加三角形标记。

本脚本只读取output中的年平均序列和一致性QC，
不读取原始NetCDF，不重新执行偏差订正、区域积分、平滑或MME计算。
"""


# %% 1. 导入依赖

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


# %% 2. 参数与路径设置

project_root = Path(r"F:\sea\Antic sea ice constraint")
input_directory = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p16_bias_corrected_pattern"
    / "n023_arctic_sia_pattern"
)
figure_directory = (
    project_root
    / "figure"
    / "s02_sia"
    / "s02_arctic_region"
    / "p17_cas_focus"
    / "n035_bias_corrected_timeseries_cas_triangles"
)
figure_directory.mkdir(parents=True, exist_ok=True)

timeseries_file = input_directory / "arctic_bias_corrected_annual_sia_280yr.csv"
consistency_qc_file = (
    input_directory
    / "arctic_bias_corrected_annual_sia_pattern_consistency_qc.csv"
)
# 与n023计算脚本一致：1e-6 million km² = 1 km²。
pattern_timeseries_tolerance_million_km2 = 1.0e-6

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
model_colors = {
    "ACCESS-ESM1-5": "#2389B9",
    "CanESM5": "#D96B17",
    "CESM2": "#1AA67D",
    "CNRM-ESM2-1": "#CC79A7",
    "GFDL-ESM4": "#E9A50B",
    "MIROC-ES2L": "#58B4E5",
    "UKESM1-0-LL": "#9D6A5B",
    "CAS-ESM2-0": "#646464",
    "NorESM2-LM": "#7450A5",
}
cas_model = "CAS-ESM2-0"
cas_marker = "^"
cas_mark_every = 20
cas_marker_size = 7.0
cas_marker_color = "#D62728"

region_order = [
    "Central Arctic",
    "North Atlantic",
    "North Pacific",
    "Total (CA+NA+NP)",
]
region_titles = {
    "Central Arctic": "Central Arctic (CA; 70–90°N)",
    "North Atlantic": "North Atlantic (NA; 45–70°N, 60°W–30°E)",
    "North Pacific": "North Pacific (NP; 45–70°N, 120°E–150°W)",
    "Total (CA+NA+NP)": "CA + NA + NP",
}
region_file_stems = {
    "Central Arctic": "central_arctic",
    "North Atlantic": "north_atlantic",
    "North Pacific": "north_pacific",
    "Total (CA+NA+NP)": "total_ca_na_np",
}


# %% 3. 读取并检查计算结果

for required_file in [timeseries_file, consistency_qc_file]:
    if not required_file.is_file():
        raise FileNotFoundError(f"缺少计算结果：{required_file}")

timeseries = pd.read_csv(timeseries_file)
consistency_qc = pd.read_csv(consistency_qc_file)
required_columns = {
    "model",
    "region",
    "simulation_year",
    "phase",
    "co2_ppm",
    "bias_corrected_annual_mean_sia_million_km2",
    "bias_corrected_annual_mean_sia_11yr_mean_million_km2",
    "intermodel_sd_11yr_mean_million_km2",
    "series_type",
}
if not required_columns.issubset(timeseries.columns):
    raise KeyError(
        f"年平均序列缺少字段：{sorted(required_columns - set(timeseries.columns))}"
    )

available_models = set(
    timeseries.loc[
        timeseries["series_type"] == "individual model",
        "model",
    ]
)
if available_models != set(model_order):
    raise ValueError(
        f"模式集合不一致：缺少={sorted(set(model_order) - available_models)}，"
        f"额外={sorted(available_models - set(model_order))}"
    )

maximum_consistency_error = float(
    consistency_qc["timeseries_minus_pattern_delta_million_km2"].abs().max()
)
if maximum_consistency_error > pattern_timeseries_tolerance_million_km2:
    raise ValueError(
        "年平均序列与pattern端点不一致："
        f"{maximum_consistency_error:.3e} million km²；"
        "允许误差="
        f"{pattern_timeseries_tolerance_million_km2:.1e} million km²"
    )


# %% 4. 绘图样式和公共图例

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 10.5,
        "axes.linewidth": 1.0,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)

legend_handles = [
    Line2D(
        [0],
        [0],
        color=model_colors[model],
        linewidth=1.8,
        marker=cas_marker if model == cas_model else None,
        markersize=cas_marker_size if model == cas_model else 0,
        markerfacecolor=cas_marker_color if model == cas_model else model_colors[model],
        markeredgecolor="white",
        markeredgewidth=0.55,
        label=model,
    )
    for model in model_order
]
legend_handles.extend(
    [
        Line2D(
            [0],
            [0],
            color="#111111",
            linewidth=3.0,
            label="9-model MME",
        ),
        Patch(
            facecolor="#AFAFAF",
            alpha=0.24,
            label="Intermodel ±1 SD",
        ),
        Line2D(
            [0],
            [0],
            color="#A51C5B",
            linewidth=2.6,
            linestyle=(0, (6, 4)),
            label="Atmospheric CO$_2$",
        ),
    ]
)


# %% 5. 分别绘制三个区域和三区总和

for region_name in region_order:
    figure, axis = plt.subplots(figsize=(10.8, 7.5))
    figure.subplots_adjust(left=0.105, right=0.88, bottom=0.22, top=0.84)

    region_table = timeseries.loc[timeseries["region"] == region_name]
    mme_table = region_table.loc[
        region_table["model"] == "9-model MME"
    ].sort_values("simulation_year")

    lower = (
        mme_table["bias_corrected_annual_mean_sia_11yr_mean_million_km2"]
        - mme_table["intermodel_sd_11yr_mean_million_km2"]
    ).clip(lower=0.0)
    upper = (
        mme_table["bias_corrected_annual_mean_sia_11yr_mean_million_km2"]
        + mme_table["intermodel_sd_11yr_mean_million_km2"]
    )
    axis.fill_between(
        mme_table["simulation_year"],
        lower,
        upper,
        color="#AFAFAF",
        alpha=0.24,
        linewidth=0,
        zorder=1,
    )

    for model in model_order:
        model_table = region_table.loc[
            region_table["model"] == model
        ].sort_values("simulation_year")
        axis.plot(
            model_table["simulation_year"],
            model_table[
                "bias_corrected_annual_mean_sia_11yr_mean_million_km2"
            ],
            color=model_colors[model],
            linewidth=1.05,
            alpha=0.92,
            marker=cas_marker if model == cas_model else None,
            markevery=cas_mark_every if model == cas_model else None,
            markersize=cas_marker_size if model == cas_model else 0,
            markerfacecolor=cas_marker_color if model == cas_model else model_colors[model],
            markeredgecolor="white",
            markeredgewidth=0.55,
            zorder=4 if model == cas_model else 2,
        )
        endpoint_row = consistency_qc.loc[
            (consistency_qc["model"] == model)
            & (consistency_qc["region"] == region_name)
        ].iloc[0]
        axis.scatter(
            [5.5, 275.5],
            [
                endpoint_row["timeseries_early_sia_million_km2"],
                endpoint_row["timeseries_later_sia_million_km2"],
            ],
            s=24,
            color=model_colors[model],
            edgecolor="white",
            linewidth=0.55,
            zorder=6,
        )

    axis.plot(
        mme_table["simulation_year"],
        mme_table["bias_corrected_annual_mean_sia_11yr_mean_million_km2"],
        color="#111111",
        linewidth=2.6,
        zorder=5,
    )
    axis.axvline(140, color="#777777", linewidth=1.0, linestyle="--")
    axis.axvspan(1, 10, color="#2166AC", alpha=0.045, linewidth=0)
    axis.axvspan(271, 280, color="#B2182B", alpha=0.045, linewidth=0)
    axis.set_xlim(1, 280)
    axis.set_xticks([1, 40, 80, 120, 140, 180, 220, 260, 280])
    axis.set_ylim(bottom=0.0)
    axis.grid(True, color="#D8D8D8", linewidth=0.70, alpha=0.60)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.set_xlabel("Simulation year", fontsize=12.0)
    axis.set_ylabel(
        "Bias-corrected annual-mean sea-ice area ($10^6$ km$^2$)",
        fontsize=12.0,
    )

    co2_axis = axis.twinx()
    co2_axis.plot(
        mme_table["simulation_year"],
        mme_table["co2_ppm"],
        color="#A51C5B",
        linewidth=2.6,
        linestyle=(0, (6, 4)),
        zorder=1,
    )
    co2_axis.set_ylim(250, 1200)
    co2_axis.set_yticks([400, 600, 800, 1000])
    co2_axis.set_ylabel(
        "Atmospheric CO$_2$ (ppm)",
        color="#7A1747",
        fontsize=12.0,
        labelpad=12,
    )
    co2_axis.tick_params(axis="y", colors="#7A1747")
    co2_axis.spines["top"].set_visible(False)
    co2_axis.spines["right"].set_color("#7A1747")
    co2_axis.set_zorder(1)
    axis.set_zorder(2)
    axis.patch.set_visible(False)

    axis.text(
        0.25,
        0.96,
        "ramp-up",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=12.0,
        color="#555555",
    )
    axis.text(
        0.75,
        0.96,
        "ramp-down",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=12.0,
        color="#555555",
    )

    figure.suptitle(
        "Arctic annual-mean sea-ice area — " + region_titles[region_name],
        x=0.105,
        ha="left",
        fontsize=16.0,
        fontweight="bold",
        y=0.96,
    )
    axis.set_title(
        "Bias corrected; 11-year running mean shown | "
        "dots are the 10-year endpoint means used by the pattern",
        fontsize=9.8,
        color="#555555",
        pad=13,
    )
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.035),
        ncol=4,
        frameon=False,
        fontsize=9.2,
        columnspacing=1.35,
        handlelength=2.7,
        handletextpad=0.55,
    )

    figure_stem = (
        figure_directory
        / (
            "arctic_annual_mean_"
            + region_file_stems[region_name]
            + "_bias_corrected_280yr_sia_cas_triangles"
        )
    )
    for suffix in ["png", "pdf", "svg"]:
        figure.savefig(
            figure_stem.with_suffix(f".{suffix}"),
            dpi=400 if suffix == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)

    print(f"[PNG] {figure_stem.with_suffix('.png')}")
    print(f"[PDF] {figure_stem.with_suffix('.pdf')}")
    print(f"[SVG] {figure_stem.with_suffix('.svg')}")

