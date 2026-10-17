# -*- coding: utf-8 -*-
"""
绘制8模式两个固定North Atlantic深混合区域的最终March MLD时间序列。

【是否需要运行】
需要。先运行：
src/s02_sia/s02_arctic_region/n037_coast_region_mld_anomaly.py

【输入文件】
output/s02sia/s02_arctic_region/p14_mld_fixed_regions_raw/
    coast_region_20261012_anomaly_plot_data.csv
    na_march_mld_coast_regions_20261012_regions_co2.csv
    coast_region_20261012_anomaly_metadata.json

【输出文件】
figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/
    coast_region_20261012_mean_timeseries.png
    coast_region_20261012_mean_timeseries.pdf
    coast_region_20261012_mean_timeseries.svg

【什么时候需要重新运行】
src正式结果或绘图样式变化后。

绘图定义
1. 2×4，共8模式。
2. 蓝线 = Nordic Seas固定polygon。
3. 橙线 = Labrador-Irminger固定polygon。
4. 使用crossing-only regional mean March MLD。
5. MLD为相对每个模式各区域初始10年平均的异常。
6. 使用src预先计算的11-year centered running mean。
7. 灰色虚线 = atmospheric CO2。
8. 不画all MLD、不画P90、不画maximum、不画MMM。
9. 不在plot中重新计算平滑、统计量或绘图范围。
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
DATA_DIR = ROOT/"output"/"s02sia"/"s02_arctic_region"/"p14_mld_fixed_regions_raw"
ANNUAL_FILE = DATA_DIR/"coast_region_20261012_anomaly_plot_data.csv"
CO2_FILE = DATA_DIR/"na_march_mld_coast_regions_20261012_regions_co2.csv"
META_FILE = DATA_DIR/"coast_region_20261012_anomaly_metadata.json"

FIG_DIR = ROOT/"figure"/"s02_sia"/"s02_arctic_region"/"p14_mld_fixed_regions_raw"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PNG_FILE = FIG_DIR/"coast_region_20261012_anomaly_timeseries.png"
PDF_FILE = FIG_DIR/"coast_region_20261012_anomaly_timeseries.pdf"
SVG_FILE = FIG_DIR/"coast_region_20261012_anomaly_timeseries.svg"

for path in [ANNUAL_FILE, CO2_FILE, META_FILE]:
    if not path.is_file():
        raise FileNotFoundError(path)

assert not any(path.exists() for path in [PNG_FILE, PDF_FILE, SVG_FILE])

# %% 3. 读取正式plot-ready数据
data = pd.read_csv(ANNUAL_FILE)
co2 = pd.read_csv(CO2_FILE)
metadata = json.loads(META_FILE.read_text(encoding="utf-8"))

MODELS = metadata["models"]
REGIONS = ["Labrador-Irminger", "Nordic Seas"]
Y_MIN = float(metadata["plot_y_min_m"])
Y_MAX = float(metadata["plot_y_max_m"])
CO2_Y_MIN = float(metadata["co2_plot_y_min_ppm"])
CO2_Y_MAX = float(metadata["co2_plot_y_max_ppm"])
CO2_TICKS = np.asarray(metadata["co2_plot_ticks_ppm"], dtype=float)

required = {
    "model",
    "region",
    "simulation_year",
    "march_mld_anomaly_m_11yr",
}

missing = required-set(data.columns)

if missing:
    raise KeyError(f"MLD文件缺少字段：{sorted(missing)}")

if not {"simulation_year", "co2_ppm"}.issubset(co2.columns):
    raise KeyError("CO2文件缺少simulation_year或co2_ppm")

if len(co2) != 280 or not np.array_equal(co2["simulation_year"].to_numpy(int), np.arange(1, 281)):
    raise ValueError("CO2序列不是完整simulation years 1-280")

for model in MODELS:
    for region in REGIONS:
        sub = data[(data["model"] == model) & (data["region"] == region)].sort_values("simulation_year")

        if len(sub) != 280:
            raise ValueError(f"{model} {region}: year count={len(sub)}，期望280")

        if not np.array_equal(sub["simulation_year"].to_numpy(int), np.arange(1, 281)):
            raise ValueError(f"{model} {region}: simulation_year不连续")

# %% 4. 出版级绘图样式
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9.0,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

NORDIC_COLOR = "#1675B7"
LI_COLOR = "#E66B00"
CO2_COLOR = "#A8A8A8"
GRID_COLOR = "#E4E4E4"
TRANSITION_COLOR = "#9FC5E3"

co2_x = co2["simulation_year"].to_numpy(float)
co2_y = co2["co2_ppm"].to_numpy(float)

# %% 5. 创建2×4画布
fig, axes = plt.subplots(2, 4, figsize=(14.8, 7.35), sharex=True, sharey=True, facecolor="white")
fig.subplots_adjust(left=0.062, right=0.955, top=0.890, bottom=0.145, wspace=0.12, hspace=0.24)

fig.suptitle("March MLD anomalies in coast-bounded fixed regions", fontsize=20, fontweight="bold", y=0.967)
fig.text(0.5, 0.018, "Anomalies relative to each model/region’s initial 10-year mean; 11-year centered running mean", ha="center", va="center", fontsize=10, color="#666666")

# %% 6. 绘制8模式
for panel_index, model in enumerate(MODELS):
    ax = axes.flat[panel_index]
    ax2 = ax.twinx()

    nordic = data[(data["model"] == model) & (data["region"] == "Nordic Seas")].sort_values("simulation_year")
    li = data[(data["model"] == model) & (data["region"] == "Labrador-Irminger")].sort_values("simulation_year")

    x = nordic["simulation_year"].to_numpy(float)
    nordic_mld = nordic["march_mld_anomaly_m_11yr"].to_numpy(float)
    li_mld = li["march_mld_anomaly_m_11yr"].to_numpy(float)

    ax2.plot(co2_x, co2_y, color=CO2_COLOR, linewidth=1.15, linestyle="--", zorder=1)
    ax2.set_ylim(CO2_Y_MIN, CO2_Y_MAX)
    ax2.set_yticks(CO2_TICKS)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    if panel_index%4 == 3:
        ax2.tick_params(axis="y", colors="#969696", labelsize=8, length=0, pad=3)
        ax2.spines["right"].set_color("#CCCCCC")
        ax2.set_ylabel(r"CO$_2$ (ppm)", color="#969696", rotation=270, labelpad=16)
    else:
        ax2.set_yticklabels([])
        ax2.tick_params(axis="y", length=0)
        ax2.spines["right"].set_visible(False)

    ax.axvline(140.5, color=TRANSITION_COLOR, linewidth=0.8, linestyle=":", zorder=1)
    ax.plot(x, nordic_mld, color=NORDIC_COLOR, linewidth=1.75, zorder=4)
    ax.plot(x, li_mld, color=LI_COLOR, linewidth=1.75, zorder=4)

    ax.set_xlim(1, 280)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.axhline(0, color="#505050", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xticks([1, 70, 140, 210, 280])
    ax.grid(True, linewidth=0.4, linestyle=":", color=GRID_COLOR, alpha=0.75)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    panel_letter = chr(ord("a")+panel_index)
    ax.set_title(model, fontweight="bold", pad=7)
    ax.text(0.015, 0.965, f"({panel_letter})", transform=ax.transAxes, ha="left", va="top", fontsize=10, fontweight="bold")

    ax.text(0.25, 0.91, "ramp-up", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#808080")
    ax.text(0.76, 0.91, "ramp-down", transform=ax.transAxes, ha="center", va="center", fontsize=8, color="#808080")

    if panel_index%4 == 0:
        ax.set_ylabel("March MLD anomaly (m)")

    if panel_index >= 4:
        ax.set_xlabel("Simulation year")

# %% 7. 公共图例
legend_handles = [
    Line2D([0], [0], color=NORDIC_COLOR, linewidth=2.0),
    Line2D([0], [0], color=LI_COLOR, linewidth=2.0),
    Line2D([0], [0], color=CO2_COLOR, linewidth=1.4, linestyle="--"),
]

legend_labels = [
    "Nordic–Iceland sector",
    "Labrador–Irminger sector",
    r"Atmospheric CO$_2$",
]

fig.legend(legend_handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 0.045), ncol=3, frameon=False, handlelength=3.0, columnspacing=2.2)

# %% 8. 方法说明

# %% 9. 保存
fig.savefig(PNG_FILE, dpi=500, facecolor="white", bbox_inches="tight", pad_inches=0.06)
fig.savefig(PDF_FILE, facecolor="white", bbox_inches="tight", pad_inches=0.06)
fig.savefig(SVG_FILE, facecolor="white", bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

# %% 10. 完成
print("\n"+"="*112)
print("FINAL FIXED-REGION NORTH ATLANTIC MARCH MLD FIGURE COMPLETE")
print("="*112)
print(f"[PNG] {PNG_FILE}")
print(f"[PDF] {PDF_FILE}")
print(f"[SVG] {SVG_FILE}")
print("="*112)
