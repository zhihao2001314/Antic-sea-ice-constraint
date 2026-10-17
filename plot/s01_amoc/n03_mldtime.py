# -*- coding: utf-8 -*-
"""
绘制8模式Nordic Seas和Labrador-Irminger Seas March MLD时间序列。

【是否需要运行】
需要。先运行：
src/s02_sia/s02_arctic_region/n16_na_mld_region_timeseries.py

【输入文件】
output/s02sia/s02_arctic_region/p06_mld/n04_na_region_timeseries/
    na_march_mld_region_timeseries.csv
    na_march_mld_region_metadata.json

【输出文件】
figure/s02_sia/s02_arctic_region/p06_mld/n04_na_region_timeseries/
    na_march_mld_region_timeseries_8models.png
    na_march_mld_region_timeseries_8models.pdf
    na_march_mld_region_timeseries_8models.svg

【什么时候需要重新运行】
plot-ready数据或绘图样式变化后。

图意
1. 蓝线：Nordic Seas March MLD anomaly。
2. 橙线：Labrador-Irminger Seas March MLD anomaly。
3. 正值表示相对初始状态混合层加深，负值表示混合层变浅。
4. 使用src已经计算好的11年中心滑动平均，不在plot中重新计算。
5. 灰色虚线表示atmospheric CO2。
"""

# %% 1. 依赖
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# %% 2. 路径
ROOT = Path(r"F:\sea\Antic sea ice constraint")
DATA_DIR = ROOT / "output" / "s02sia" / "s02_arctic_region" / "p06_mld" / "n04_na_region_timeseries"
FIG_DIR = ROOT / "figure" / "s02_sia" / "s02_arctic_region" / "p06_mld" / "n04_na_region_timeseries"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SERIES_FILE = DATA_DIR / "na_march_mld_region_timeseries.csv"
META_FILE = DATA_DIR / "na_march_mld_region_metadata.json"

PNG_FILE = FIG_DIR / "na_march_mld_region_timeseries_8models.png"
PDF_FILE = FIG_DIR / "na_march_mld_region_timeseries_8models.pdf"
SVG_FILE = FIG_DIR / "na_march_mld_region_timeseries_8models.svg"

if not SERIES_FILE.is_file():
    raise FileNotFoundError(f"缺少：{SERIES_FILE}")

if not META_FILE.is_file():
    raise FileNotFoundError(f"缺少：{META_FILE}")

# %% 3. 读取plot-ready数据
series = pd.read_csv(SERIES_FILE)

with open(META_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)

MODELS = metadata["models"]
MLD_YMIN, MLD_YMAX = metadata["mld_anomaly_ylim_m"]
MLD_YTICKS = metadata["mld_anomaly_yticks_m"]
CO2_YMIN, CO2_YMAX = metadata["co2_ylim_ppm"]
CO2_TICKS = metadata["co2_ticks_ppm"]

required_columns = {"model", "simulation_year", "co2_ppm", "nordic_mld_anomaly_11yr_m", "li_mld_anomaly_11yr_m"}
missing_columns = required_columns - set(series.columns)

if missing_columns:
    raise KeyError(f"plot-ready CSV缺少字段：{sorted(missing_columns)}")

missing_models = [model for model in MODELS if model not in set(series["model"])]

if missing_models:
    raise ValueError(f"plot-ready CSV缺少模式：{missing_models}")

# %% 4. 样式
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "axes.linewidth": 0.75,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

NORDIC_COLOR = "#1976b9"
LI_COLOR = "#e66b00"
CO2_COLOR = "#b5b5b5"
ZERO_COLOR = "#777777"
GRID_COLOR = "#d7d7d7"

# %% 5. 建立2×4画布
fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.5), sharex=True, sharey=True, facecolor="white")
fig.subplots_adjust(left=0.060, right=0.950, top=0.885, bottom=0.155, wspace=0.085, hspace=0.205)

fig.suptitle("North Atlantic deep-mixing response across eight models", fontsize=21.0, fontweight="bold", y=0.965)

# %% 6. 绘制各模式
for panel_index, model in enumerate(MODELS):
    row_index = panel_index // 4
    column_index = panel_index % 4
    ax = axes[row_index, column_index]

    model_data = series[series["model"] == model].sort_values("simulation_year")

    year = model_data["simulation_year"].to_numpy(dtype=float)
    nordic = model_data["nordic_mld_anomaly_11yr_m"].to_numpy(dtype=float)
    li = model_data["li_mld_anomaly_11yr_m"].to_numpy(dtype=float)
    co2 = model_data["co2_ppm"].to_numpy(dtype=float)

    if year.size != 280:
        raise ValueError(f"{model} plot-ready时间长度不是280年：{year.size}")

    ax.axhline(0.0, color=ZERO_COLOR, linewidth=0.8, linestyle="--", zorder=1)
    ax.axvline(140.0, color="#888888", linewidth=0.8, linestyle=":", zorder=1)

    ax.plot(year, nordic, color=NORDIC_COLOR, linewidth=2.15, zorder=4)
    ax.plot(year, li, color=LI_COLOR, linewidth=2.15, zorder=4)

    ax.set_xlim(1, 280)
    ax.set_ylim(MLD_YMIN, MLD_YMAX)
    ax.set_xticks([1, 70, 140, 210, 280])
    ax.set_yticks(MLD_YTICKS)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, linestyle=":", alpha=0.65, zorder=0)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    panel_letter = chr(ord("a") + panel_index)
    ax.set_title(model, fontsize=11.7, fontweight="bold", pad=7)
    ax.text(0.015, 0.965, f"({panel_letter})", transform=ax.transAxes, ha="left", va="top", fontsize=9.7, fontweight="bold")

    ax.text(0.25, 0.055, "ramp-up", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.8, color="#777777")
    ax.text(0.75, 0.055, "ramp-down", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.8, color="#777777")

    ax_co2 = ax.twinx()
    ax_co2.plot(year, co2, color=CO2_COLOR, linewidth=1.2, linestyle="--", zorder=2)
    ax_co2.set_ylim(CO2_YMIN, CO2_YMAX)
    ax_co2.set_yticks(CO2_TICKS)
    ax_co2.spines["top"].set_visible(False)
    ax_co2.spines["left"].set_visible(False)

    if column_index != 3:
        ax_co2.set_yticklabels([])
        ax_co2.spines["right"].set_visible(False)
        ax_co2.tick_params(axis="y", right=False)
    else:
        ax_co2.spines["right"].set_color("#cccccc")
        ax_co2.tick_params(axis="y", colors="#999999", labelsize=7.8, length=0, pad=4)
        ax_co2.set_ylabel(r"CO$_2$ (ppm)", fontsize=8.7, color="#888888", rotation=270, labelpad=15)

# %% 7. 公共坐标轴
axes[0, 0].set_ylabel("March MLD anomaly (m)")
axes[1, 0].set_ylabel("March MLD anomaly (m)")

for column_index in range(4):
    axes[1, column_index].set_xlabel("Simulation year")

# %% 8. 公共图例
legend_handles = [
    Line2D([0], [0], color=NORDIC_COLOR, linewidth=2.5),
    Line2D([0], [0], color=LI_COLOR, linewidth=2.5),
    Line2D([0], [0], color=CO2_COLOR, linewidth=1.4, linestyle="--"),
]

legend_labels = ["Nordic Seas", "Labrador-Irminger Seas", "Atmospheric CO$_2$"]

fig.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.50, 0.072), ncol=3, frameon=False, fontsize=10.0, handlelength=3.1, handletextpad=0.7, columnspacing=2.4)

fig.text(0.50, 0.027, "11-year centered running mean; anomalies relative to each model-region's 1pctCO2 years 1-10; positive values indicate deeper mixing", ha="center", va="center", fontsize=8.7, color="#666666")

# %% 9. 保存
fig.savefig(PNG_FILE, dpi=500, facecolor="white", bbox_inches="tight", pad_inches=0.06)
fig.savefig(PDF_FILE, facecolor="white", bbox_inches="tight", pad_inches=0.06)
fig.savefig(SVG_FILE, facecolor="white", bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

print("=" * 110)
print("8-MODEL NORTH ATLANTIC MLD TIME-SERIES FIGURE COMPLETE")
print("=" * 110)
print(f"[PNG] {PNG_FILE}")
print(f"[PDF] {PDF_FILE}")
print(f"[SVG] {SVG_FILE}")
print("=" * 110)
