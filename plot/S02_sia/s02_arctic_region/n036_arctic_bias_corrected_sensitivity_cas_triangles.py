# -*- coding: utf-8 -*-
"""
绘制北大西洋年平均订正海冰序列的六种模式剔除敏感性结果，
并在各面板的CAS-ESM2-0曲线上添加三角形标记。

本脚本只读取output中的n023逐模式序列和n025敏感性统计，
不读取原始数据，不执行bias correction、区域积分、平滑或MME计算。
"""


# %% 1. 导入依赖

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


# %% 2. 参数与路径设置

project_root = Path(r"F:\sea\Antic sea ice constraint")
base_input_directory = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p16_bias_corrected_pattern"
    / "n023_arctic_sia_pattern"
)
sensitivity_input_directory = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p16_bias_corrected_pattern"
    / "n025_annual_sia_mme_sensitivity"
)
figure_directory = (
    project_root
    / "figure"
    / "s02_sia"
    / "s02_arctic_region"
    / "p17_cas_focus"
    / "n036_bias_corrected_sensitivity_cas_triangles"
)
figure_directory.mkdir(parents=True, exist_ok=True)

base_timeseries_file = (
    base_input_directory / "arctic_bias_corrected_annual_sia_280yr.csv"
)
sensitivity_timeseries_file = (
    sensitivity_input_directory
    / "north_atlantic_mme_exclusion_sensitivity_280yr.csv"
)
sensitivity_endpoints_file = (
    sensitivity_input_directory
    / "north_atlantic_mme_exclusion_sensitivity_endpoints.csv"
)

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

case_order = [
    "without_cesm2",
    "without_cnrm_esm2_1",
    "without_canesm5",
    "without_noresm2_lm",
    "without_cesm2_and_canesm5",
    "without_cesm2_and_noresm2_lm",
]
case_titles = {
    "without_cesm2": "Without CESM2",
    "without_cnrm_esm2_1": "Without CNRM-ESM2-1",
    "without_canesm5": "Without CanESM5",
    "without_noresm2_lm": "Without NorESM2-LM",
    "without_cesm2_and_canesm5": "Without CESM2 and CanESM5",
    "without_cesm2_and_noresm2_lm": "Without CESM2 and NorESM2-LM",
}
case_file_stems = {
    "without_cesm2": "without_cesm2",
    "without_cnrm_esm2_1": "without_cnrm_esm2_1",
    "without_canesm5": "without_canesm5",
    "without_noresm2_lm": "without_noresm2_lm",
    "without_cesm2_and_canesm5": "without_cesm2_and_canesm5",
    "without_cesm2_and_noresm2_lm": "without_cesm2_and_noresm2_lm",
}


# %% 3. 读取并检查计算结果

for required_file in [
    base_timeseries_file,
    sensitivity_timeseries_file,
    sensitivity_endpoints_file,
]:
    if not required_file.is_file():
        raise FileNotFoundError(f"缺少计算结果：{required_file}")

base_timeseries = pd.read_csv(base_timeseries_file)
sensitivity_timeseries = pd.read_csv(sensitivity_timeseries_file)
sensitivity_endpoints = pd.read_csv(sensitivity_endpoints_file)

base_timeseries = base_timeseries.loc[
    (base_timeseries["region"] == "North Atlantic")
    & (base_timeseries["series_type"] == "individual model")
].copy()

available_cases = set(sensitivity_timeseries["case"].unique())
if available_cases != set(case_order):
    raise ValueError(
        f"敏感性方案不一致：缺少={sorted(set(case_order) - available_cases)}，"
        f"额外={sorted(available_cases - set(case_order))}"
    )


# %% 4. 绘图样式

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


# %% 5. 分别绘制六种模式剔除方案

for case_name in case_order:
    case_statistics = sensitivity_timeseries.loc[
        sensitivity_timeseries["case"] == case_name
    ].sort_values("simulation_year")
    case_endpoints = sensitivity_endpoints.loc[
        sensitivity_endpoints["case"] == case_name
    ].copy()
    retained_models = [
        model
        for model in model_order
        if model in set(case_endpoints["model"].unique())
    ]
    retained_model_count = int(
        case_statistics["retained_model_count"].iloc[0]
    )

    figure, axis = plt.subplots(figsize=(10.8, 7.5))
    figure.subplots_adjust(left=0.105, right=0.88, bottom=0.22, top=0.84)

    lower = (
        case_statistics["ensemble_mean_11yr_sia_million_km2"]
        - case_statistics["intermodel_sd_11yr_sia_million_km2"]
    ).clip(lower=0.0)
    upper = (
        case_statistics["ensemble_mean_11yr_sia_million_km2"]
        + case_statistics["intermodel_sd_11yr_sia_million_km2"]
    )
    axis.fill_between(
        case_statistics["simulation_year"],
        lower,
        upper,
        color="#AFAFAF",
        alpha=0.24,
        linewidth=0,
        zorder=1,
    )

    for model in retained_models:
        model_table = base_timeseries.loc[
            base_timeseries["model"] == model
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

        endpoint_row = case_endpoints.loc[
            case_endpoints["model"] == model
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
        case_statistics["simulation_year"],
        case_statistics["ensemble_mean_11yr_sia_million_km2"],
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
        case_statistics["simulation_year"],
        case_statistics["co2_ppm"],
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
        "Arctic annual-mean sea-ice area — North Atlantic",
        x=0.105,
        ha="left",
        fontsize=16.0,
        fontweight="bold",
        y=0.96,
    )
    axis.set_title(
        case_titles[case_name]
        + "; 11-year running mean | dots: years 1–10 and 271–280 means",
        fontsize=9.8,
        color="#555555",
        pad=13,
    )

    legend_handles = []
    for model in retained_models:
        legend_handles.append(
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
        )
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                color="#111111",
                linewidth=3.0,
                label=f"{retained_model_count}-model MME",
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
            "arctic_annual_mean_north_atlantic_bias_corrected_"
            + case_file_stems[case_name]
            + "_cas_triangles"
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

