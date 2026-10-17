# -*- coding: utf-8 -*-
"""
绘制40°N、65°N及40°N−65°N的forced原始AMOC强度补充图。

【是否需要运行】
需要。先运行：
src/amoc/n13_amoc_deepwater_regions_raw_strength.py

【输入文件】
output/amoc_s01_n01/n13_deepwater_regions_raw_strength/
    amoc_deepwater_regions_raw_strength_plot_data.csv
    amoc_deepwater_regions_raw_strength_metadata.json

【输出文件】
figure/amoc_s01_n01/n13_deepwater_regions_raw_strength/
    north_atlantic_overturning_raw_strength_9models.png
    north_atlantic_overturning_raw_strength_9models.pdf
    north_atlantic_overturning_raw_strength_9models.svg

【什么时候需要重新运行】
plot-ready结果或绘图样式变化后。

职责
只读取src输出并绘图，不重新计算AMOC、proxy、平滑、CO2或纵轴范围。
"""

# %% 1. 依赖
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

# %% 2. 路径
ROOT = Path(r"F:\sea\Antic sea ice constraint")

DATA_DIR = ROOT/"output"/"amoc_s01_n01"/"n13_deepwater_regions_raw_strength"
FIG_DIR = ROOT/"figure"/"amoc_s01_n01"/"n13_deepwater_regions_raw_strength"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = DATA_DIR/"amoc_deepwater_regions_raw_strength_plot_data.csv"
META_FILE = DATA_DIR/"amoc_deepwater_regions_raw_strength_metadata.json"

PNG_FILE = FIG_DIR/"north_atlantic_overturning_raw_strength_9models.png"
PDF_FILE = FIG_DIR/"north_atlantic_overturning_raw_strength_9models.pdf"
SVG_FILE = FIG_DIR/"north_atlantic_overturning_raw_strength_9models.svg"

if not DATA_FILE.is_file():
    raise FileNotFoundError(f"缺少：{DATA_FILE}")

if not META_FILE.is_file():
    raise FileNotFoundError(f"缺少：{META_FILE}")

# %% 3. 读取plot-ready数据
data = pd.read_csv(DATA_FILE)

with open(META_FILE, "r", encoding="utf-8") as file:
    metadata = json.load(file)

MODELS = metadata["models"]
Y_MIN, Y_MAX = metadata["y_limits_sv"]
Y_TICKS = metadata["y_ticks_sv"]
CO2_MIN, CO2_MAX = metadata["co2_limits_ppm"]
CO2_TICKS = metadata["co2_ticks_ppm"]

required = {
    "model", "simulation_year", "co2_ppm",
    "amoc_40N_raw_11yr_sv",
    "amoc_65N_raw_11yr_sv",
    "labrador_irminger_proxy_raw_11yr_sv",
}

missing = required-set(data.columns)

if missing:
    raise KeyError(f"plot-ready文件缺少字段：{sorted(missing)}")

# %% 4. 样式
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9.4,
    "axes.titlesize": 11.5,
    "axes.labelsize": 10.5,
    "axes.linewidth": 0.75,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

BLACK = "#222222"
BLUE = "#1677b8"
ORANGE = "#e66b00"
CO2_COLOR = "#b5b5b5"
GRID_COLOR = "#dddddd"

# %% 5. 画布
fig, axes = plt.subplots(3, 3, figsize=(17.8, 11.0), sharex=True, sharey=True)
fig.subplots_adjust(left=0.060, right=0.950, top=0.905, bottom=0.125, wspace=0.085, hspace=0.175)

fig.suptitle("North Atlantic overturning strength across nine models", fontsize=21.0, fontweight="bold", y=0.975)

# %% 6. 九模式
for panel_index, model in enumerate(MODELS):
    row = panel_index//3
    column = panel_index%3
    ax = axes[row, column]

    model_data = data[data["model"] == model].sort_values("simulation_year")

    year = model_data["simulation_year"].to_numpy()
    amoc40 = model_data["amoc_40N_raw_11yr_sv"].to_numpy()
    amoc65 = model_data["amoc_65N_raw_11yr_sv"].to_numpy()
    li = model_data["labrador_irminger_proxy_raw_11yr_sv"].to_numpy()
    co2 = model_data["co2_ppm"].to_numpy()

    if len(year) != 280:
        raise RuntimeError(f"{model} plot-ready长度={len(year)}，应为280")

    ax.axvline(140.0, color="#888888", linewidth=0.8, linestyle=":", zorder=1)

    ax.plot(year, amoc40, color=BLACK, linewidth=2.1, zorder=4)
    ax.plot(year, amoc65, color=BLUE, linewidth=2.0, zorder=4)
    ax.plot(year, li, color=ORANGE, linewidth=2.0, zorder=4)

    ax.set_xlim(1, 280)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks([1, 70, 140, 210, 280])
    ax.set_yticks(Y_TICKS)
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.45, linestyle=":", alpha=0.55)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    letter = chr(ord("a")+panel_index)
    ax.set_title(model, fontweight="bold", pad=7)
    ax.text(0.015, 0.965, f"({letter})", transform=ax.transAxes, ha="left", va="top", fontsize=9.5, fontweight="bold")

    ax.text(0.25, 0.055, "ramp-up", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.8, color="#777777")
    ax.text(0.75, 0.055, "ramp-down", transform=ax.transAxes, ha="center", va="bottom", fontsize=7.8, color="#777777")

    ax_co2 = ax.twinx()
    ax_co2.plot(year, co2, color=CO2_COLOR, linewidth=1.2, linestyle="--", zorder=2)
    ax_co2.set_ylim(CO2_MIN, CO2_MAX)
    ax_co2.set_yticks(CO2_TICKS)
    ax_co2.spines["top"].set_visible(False)
    ax_co2.spines["left"].set_visible(False)

    if column != 2:
        ax_co2.set_yticklabels([])
        ax_co2.spines["right"].set_visible(False)
        ax_co2.tick_params(right=False)
    else:
        ax_co2.spines["right"].set_color("#cccccc")
        ax_co2.tick_params(axis="y", colors="#999999", labelsize=7.8, length=0, pad=4)
        ax_co2.set_ylabel(r"CO$_2$ (ppm)", fontsize=8.7, color="#888888", rotation=270, labelpad=15)

# %% 7. 坐标轴
axes[0, 0].set_ylabel("Overturning strength (Sv)")
axes[1, 0].set_ylabel("Overturning strength (Sv)")
axes[2, 0].set_ylabel("Overturning strength (Sv)")

for column in range(3):
    axes[2, column].set_xlabel("Simulation year")

# %% 8. 公共图例
handles = [
    Line2D([0], [0], color=BLACK, linewidth=2.4),
    Line2D([0], [0], color=BLUE, linewidth=2.4),
    Line2D([0], [0], color=ORANGE, linewidth=2.4),
    Line2D([0], [0], color=CO2_COLOR, linewidth=1.4, linestyle="--"),
]

labels = [
    "40°N total AMOC",
    "65°N high-latitude diagnostic",
    "Labrador-Irminger-related proxy (40°N − 65°N)",
    "Atmospheric CO$_2$",
]

fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.50, 0.058), ncol=4, frameon=False, fontsize=9.4, handlelength=3.1, columnspacing=1.8)

fig.text(0.50, 0.020, "11-year centered running mean; raw forced overturning strength; no piControl subtraction", ha="center", va="center", fontsize=8.7, color="#666666")

# %% 9. 保存
fig.savefig(PNG_FILE, dpi=500, facecolor="white", bbox_inches="tight", pad_inches=0.06)
fig.savefig(PDF_FILE, facecolor="white", bbox_inches="tight", pad_inches=0.06)
fig.savefig(SVG_FILE, facecolor="white", bbox_inches="tight", pad_inches=0.06)
plt.close(fig)

print("="*108)
print("AMOC RAW-STRENGTH SUPPLEMENT FIGURE COMPLETE")
print("="*108)
print(f"[PNG] {PNG_FILE}")
print(f"[PDF] {PDF_FILE}")
print(f"[SVG] {SVG_FILE}")
print("="*108)