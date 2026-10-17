# -*- coding: utf-8 -*-
"""
绘制9模式40°N、65°N和40°N−65°N的piControl-relative AMOC anomaly拼图，
每个模式panel同时显示对应的CO2浓度路径。

【是否需要运行】
需要。
如果已经运行：
src/amoc/n12_amoc_deepwater_regions_300m.py
且年度CSV没有变化，则不需要重新运行src，只运行本绘图脚本。

【输入文件】
output/amoc_s01_n01/n12_deepwater_regions_300m/
    amoc_deepwater_regions_300m_annual.csv
    amoc_deepwater_regions_300m_metadata.json

【输出文件】
figure/amoc_s01_n01/n12_deepwater_regions_300m/
    amoc_deepwater_regions_300m_9models_anomaly.png
    amoc_deepwater_regions_300m_9models_anomaly.pdf
    amoc_deepwater_regions_300m_9models_anomaly.svg

【什么时候需要重新运行】
AMOC anomaly结果或绘图样式变化后。

绘图定义
--------
1. 一张图包含9个模式，采用3×3布局。
2. 前8个模式顺序与North Atlantic March MLD图保持一致，CAS-ESM2-0放最后。
3. 每个panel包含：
   40°N total AMOC；
   65°N high-latitude overturning diagnostic；
   40°N−65°N Labrador–Irminger-related proxy；
   atmospheric CO2 pathway。
4. 三条AMOC序列均相对于各自piControl climatology。
5. CO2由284.7 ppm开始，1pctCO2增加140年，然后CDR对称下降140年。
6. AMOC采用11年中心滑动平均，仅用于显示。
7. simulation year 140为ramp-up与ramp-down转换。
8. 9个模式使用统一对称AMOC纵轴范围，便于跨模式比较。
9. CO2使用每个panel的右侧独立纵轴，但只在最右列显示刻度标签。
10. 底部使用一个公共图例。
"""

# %% 1. 依赖

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


# %% 2. 路径

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT/"output"/"amoc_s01_n01"/"n12_deepwater_regions_300m"
OUT = ROOT/"figure"/"amoc_s01_n01"/"n12_deepwater_regions_300m"

ANNUAL_FILE = INPUT/"amoc_deepwater_regions_300m_annual.csv"
META_FILE = INPUT/"amoc_deepwater_regions_300m_metadata.json"

PNG_FILE = OUT/"amoc_deepwater_regions_300m_9models_anomaly.png"
PDF_FILE = OUT/"amoc_deepwater_regions_300m_9models_anomaly.pdf"
SVG_FILE = OUT/"amoc_deepwater_regions_300m_9models_anomaly.svg"

if not ANNUAL_FILE.is_file():
    raise FileNotFoundError(ANNUAL_FILE)
if not META_FILE.is_file():
    raise FileNotFoundError(META_FILE)

OUT.mkdir(parents=True, exist_ok=True)


# %% 3. 参数

MODELS = [
    "ACCESS-ESM1-5",
    "CanESM5",
    "CESM2",
    "CNRM-ESM2-1",
    "GFDL-ESM4",
    "MIROC-ES2L",
    "UKESM1-0-LL",
    "NorESM2-LM",
    "CAS-ESM2-0",
]

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)"]

SMOOTH = 11
END_YEAR = 280
RAMP_END = 140
CO2_BASELINE_PPM = 284.7
RAMP_YEARS = 140

COLOR_40 = "#202020"
COLOR_65 = "#0072B2"
COLOR_LI = "#D55E00"
COLOR_CO2 = "#8A8A8A"

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10.0,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 9.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# %% 4. 读取数据

annual = pd.read_csv(ANNUAL_FILE)
metadata = json.loads(META_FILE.read_text(encoding="utf-8"))

required = {
    "model",
    "simulation_year",
    "amoc_40N_anomaly_sv",
    "amoc_65N_anomaly_sv",
    "labrador_irminger_proxy_anomaly_sv",
}

missing = required-set(annual.columns)

if missing:
    raise KeyError(f"AMOC年度文件缺少字段：{sorted(missing)}")

successful = metadata.get("models_successful", [])
missing_models = [model for model in MODELS if model not in successful]

if missing_models:
    raise RuntimeError(f"9模式AMOC结果不完整：{missing_models}")


# %% 5. CO2路径

simulation_year = np.arange(1, END_YEAR+1, dtype=int)
index = simulation_year-1
co2_exponent = np.where(index < RAMP_YEARS, index+1, np.maximum(2*RAMP_YEARS-1-index, 0))
co2_ppm = CO2_BASELINE_PPM*1.01**co2_exponent

co2_min = float(np.nanmin(co2_ppm))
co2_max = float(np.nanmax(co2_ppm))
co2_span = co2_max-co2_min


# %% 6. 准备9模式显示数据

plot_data = {}
all_display_values = []

for model in MODELS:
    data = annual.loc[annual["model"] == model].sort_values("simulation_year").copy()
    data = data.loc[data["simulation_year"].between(1, END_YEAR)].copy()

    if len(data) != END_YEAR:
        raise ValueError(f"{model}应有{END_YEAR}年，实际={len(data)}")

    year = data["simulation_year"].to_numpy(int)

    if not np.array_equal(year, simulation_year):
        raise ValueError(f"{model} simulation year不是连续1–{END_YEAR}")

    amoc40 = data["amoc_40N_anomaly_sv"].rolling(SMOOTH, center=True, min_periods=(SMOOTH+1)//2).mean().to_numpy(float)
    amoc65 = data["amoc_65N_anomaly_sv"].rolling(SMOOTH, center=True, min_periods=(SMOOTH+1)//2).mean().to_numpy(float)
    li = data["labrador_irminger_proxy_anomaly_sv"].rolling(SMOOTH, center=True, min_periods=(SMOOTH+1)//2).mean().to_numpy(float)

    plot_data[model] = {"year": year, "amoc40": amoc40, "amoc65": amoc65, "li": li}

    all_display_values.extend(amoc40[np.isfinite(amoc40)])
    all_display_values.extend(amoc65[np.isfinite(amoc65)])
    all_display_values.extend(li[np.isfinite(li)])


# %% 7. 统一AMOC纵轴

all_display_values = np.asarray(all_display_values, float)

if all_display_values.size == 0:
    raise RuntimeError("9模式AMOC显示序列全部无有效值")

common_limit = max(abs(float(np.nanmin(all_display_values))), abs(float(np.nanmax(all_display_values))))
common_limit = max(common_limit, 1.0)*1.08

if common_limit <= 5.0:
    common_limit = np.ceil(common_limit)
elif common_limit <= 10.0:
    common_limit = np.ceil(common_limit/2.0)*2.0
else:
    common_limit = np.ceil(common_limit/5.0)*5.0

print(f"[统一AMOC纵轴] {-common_limit:.1f} ~ {common_limit:.1f} Sv")


# %% 8. Figure

fig, axes = plt.subplots(3, 3, figsize=(16.8, 12.0), sharex=True, sharey=True, facecolor="white")
fig.subplots_adjust(left=0.065, right=0.945, top=0.905, bottom=0.135, wspace=0.085, hspace=0.175)

fig.suptitle("North Atlantic overturning response across nine models", fontsize=22.0, fontweight="bold", y=0.965)


# %% 9. 逐模式panel

for panel_index, model in enumerate(MODELS):
    row = panel_index//3
    column = panel_index%3
    ax = axes[row, column]
    ax_co2 = ax.twinx()

    year = plot_data[model]["year"]
    amoc40 = plot_data[model]["amoc40"]
    amoc65 = plot_data[model]["amoc65"]
    li = plot_data[model]["li"]

    # CO2放最底层
    ax_co2.plot(simulation_year, co2_ppm, color=COLOR_CO2, lw=1.25, ls="--", alpha=0.55, zorder=1)
    ax_co2.set_ylim(co2_min-0.08*co2_span, co2_max+0.12*co2_span)
    ax_co2.spines["top"].set_visible(False)
    ax_co2.spines["left"].set_visible(False)
    ax_co2.spines["right"].set_color("#BBBBBB")
    ax_co2.spines["right"].set_linewidth(0.55)

    if column == 2:
        ax_co2.set_yticks([CO2_BASELINE_PPM, 700.0, co2_max])
        ax_co2.set_yticklabels([f"{CO2_BASELINE_PPM:.0f}", "700", f"{co2_max:.0f}"])
        ax_co2.tick_params(axis="y", colors="#888888", labelsize=7.3, length=2.3, width=0.55)
        ax_co2.set_ylabel("CO$_2$ (ppm)", rotation=270, labelpad=14, fontsize=8.2, color="#777777")
    else:
        ax_co2.set_yticks([])
        ax_co2.spines["right"].set_visible(False)

    # AMOC
    ax.plot(year, amoc40, color=COLOR_40, lw=2.15, zorder=6)
    ax.plot(year, amoc65, color=COLOR_65, lw=1.90, zorder=5)
    ax.plot(year, li, color=COLOR_LI, lw=2.00, zorder=5)

    ax.axhline(0.0, color="#555555", lw=0.85, ls="--", alpha=0.75, zorder=2)
    ax.axvline(RAMP_END, color="#777777", lw=0.90, ls=":", alpha=0.85, zorder=3)

    ax.set_xlim(1, END_YEAR)
    ax.set_ylim(-common_limit, common_limit)
    ax.set_xticks([1, 70, 140, 210, 280])

    ax.set_title(model, fontsize=12.0, fontweight="bold", pad=5.5)
    ax.text(0.015, 0.965, PANEL_LABELS[panel_index], transform=ax.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold")

    ax.grid(axis="y", color="#B8B8B8", lw=0.40, ls=":", alpha=0.50)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3.0, width=0.7)

    if row == 2:
        ax.set_xlabel("Simulation year", fontsize=10.5)

    if column == 0:
        ax.set_ylabel("AMOC anomaly (Sv)", fontsize=10.5)

    ax.text(0.245, 0.045, "ramp-up", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.8, color="#777777")
    ax.text(0.755, 0.045, "ramp-down", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.8, color="#777777")


# %% 10. 公共图例

legend_handles = [
    Line2D([0], [0], color=COLOR_40, lw=2.4),
    Line2D([0], [0], color=COLOR_65, lw=2.1),
    Line2D([0], [0], color=COLOR_LI, lw=2.2),
    Line2D([0], [0], color=COLOR_CO2, lw=1.4, ls="--", alpha=0.70),
]

legend_labels = [
    "40°N total AMOC",
    "65°N high-latitude diagnostic",
    "Labrador–Irminger-related proxy (40°N − 65°N)",
    "Atmospheric CO$_2$",
]

fig.legend(handles=legend_handles, labels=legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.055), ncol=4, frameon=False, fontsize=9.2, handlelength=3.1, handletextpad=0.7, columnspacing=1.8)

fig.text(0.5, 0.018, "AMOC: 11-year centered running mean; anomalies relative to each model's piControl climatology", ha="center", va="bottom", fontsize=8.5, color="#555555")


# %% 11. 保存

fig.savefig(PNG_FILE, dpi=600, bbox_inches="tight", pad_inches=0.06, facecolor="white")
fig.savefig(PDF_FILE, bbox_inches="tight", pad_inches=0.06, facecolor="white")
fig.savefig(SVG_FILE, bbox_inches="tight", pad_inches=0.06, facecolor="white")

plt.close(fig)


# %% 12. 输出

print("\n"+"="*108)
print("9-MODEL AMOC DEEP-WATER REGION FIGURE COMPLETE")
print("="*108)
print(f"models = {MODELS}")
print(f"display smoothing = {SMOOTH} years")
print(f"common AMOC range = {-common_limit:.1f} ~ {common_limit:.1f} Sv")
print(f"CO2 range = {co2_min:.1f}–{co2_max:.1f} ppm")
print(f"PNG = {PNG_FILE}")
print(f"PDF = {PDF_FILE}")
print(f"SVG = {SVG_FILE}")
print("="*108)