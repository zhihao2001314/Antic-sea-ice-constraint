# -*- coding: utf-8 -*-
"""
绘制 North Atlantic SIA + 40°N AMOC individual-model time series。

【是否需要运行】
需要。先保证对应src已经生成plot-ready数据。

【输入文件】
output/s02sia/s02_arctic_region/n03-siaandamoc_n01_newNA_raw/
    amoc_sia_co2_plot_data.csv
    amoc_sia_co2_metadata.json

【输出文件】
figure/s02_sia/s02_arctic_region/n03-siaandamoc_n01_newNA_raw/
    north_atlantic_sia_amoc40N_models.png
    north_atlantic_sia_amoc40N_models.pdf
    north_atlantic_sia_amoc40N_models.svg

【什么时候需要重新运行】
plot-ready数据或绘图样式变化后。

说明
1. 本脚本只绘图，不读取任何原始NetCDF。
2. SIA = North Atlantic 45–70°N, 60°W–30°E。
3. SIA anomaly相对各模式1pctCO2实验年1–10。
4. SIA不做piControl漂移订正。
5. AMOC沿用已有40°N plot-ready序列。
6. (a)(b)放在坐标轴外，不进入数据区。
7. 同名图直接覆盖，不弹出Figure窗口。
"""

# %% 1. 导入依赖
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

# %% 2. 路径
ROOT = Path(r"F:\sea\Antic sea ice constraint")
DATA_DIR = ROOT / "output" / "s02sia" / "s02_arctic_region" / "n03-siaandamoc_n01_newNA_raw"
FIG_DIR = ROOT / "figure" / "s02_sia" / "s02_arctic_region" / "n03-siaandamoc_n01_newNA_raw"

DATA_FILE = DATA_DIR / "amoc_sia_co2_plot_data.csv"
META_FILE = DATA_DIR / "amoc_sia_co2_metadata.json"

PNG_FILE = FIG_DIR / "north_atlantic_sia_amoc40N_models.png"
PDF_FILE = FIG_DIR / "north_atlantic_sia_amoc40N_models.pdf"
SVG_FILE = FIG_DIR / "north_atlantic_sia_amoc40N_models.svg"

for path in [DATA_FILE, META_FILE]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少plot-ready文件：{path}")

FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% 3. 读取plot-ready数据
data = pd.read_csv(DATA_FILE).sort_values("simulation_year").reset_index(drop=True)
metadata = json.loads(META_FILE.read_text(encoding="utf-8"))
models = metadata.get("common_models", [])

EXPECTED_MODELS = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4", "MIROC-ES2L", "UKESM1-0-LL", "CAS-ESM2-0", "NorESM2-LM"]

if not models:
    raise RuntimeError("metadata中没有common_models")

missing_models = [model for model in EXPECTED_MODELS if model not in models]

if missing_models:
    raise RuntimeError(f"正式9模式序列不完整，缺少：{missing_models}")

models = EXPECTED_MODELS
required = {"simulation_year", "stage", "co2_ppm"}

for model in models:
    required.update({f"sia_{model}", f"amoc_{model}"})

missing = required - set(data.columns)

if missing:
    raise KeyError(f"plot-ready文件缺少字段：{sorted(missing)}")

if len(data) != 280:
    raise ValueError(f"联合序列应为280年，实际={len(data)}")

x = data["simulation_year"].to_numpy(dtype=float)
co2 = data["co2_ppm"].to_numpy(dtype=float)
stage = data["stage"].astype(str).to_numpy()
cdr_index = np.flatnonzero(stage == "1pctCO2-cdr")

if cdr_index.size == 0 or int(cdr_index[0]) != 140:
    raise ValueError("1pctCO2-cdr阶段不是从联合序列第141年开始")

transition_x = 140.5

# %% 4. 数据QA
sia_all = np.concatenate([data[f"sia_{model}"].to_numpy(dtype=float) for model in models])
amoc_all = np.concatenate([data[f"amoc_{model}"].to_numpy(dtype=float) for model in models])

if not np.all(np.isfinite(sia_all)):
    raise ValueError("SIA plot-ready数据含NaN/Inf")

if not np.all(np.isfinite(amoc_all)):
    raise ValueError("AMOC plot-ready数据含NaN/Inf")

if not np.all(np.isfinite(co2)):
    raise ValueError("CO2 plot-ready数据含NaN/Inf")

sia_min, sia_max = float(np.min(sia_all)), float(np.max(sia_all))
amoc_min, amoc_max = float(np.min(amoc_all)), float(np.max(amoc_all))
co2_min, co2_max = float(np.min(co2)), float(np.max(co2))

sia_pad = max(0.06, 0.07 * (sia_max - sia_min))
amoc_pad = max(0.35, 0.06 * (amoc_max - amoc_min))
co2_pad = max(20.0, 0.05 * (co2_max - co2_min))

# %% 5. 绘图样式
TEXT_COLOR = "#222222"
GRID_COLOR = "#D9D9D9"
CO2_COLOR = "#76539A"

MODEL_COLORS = {
    "ACCESS-ESM1-5": "#D55E00",
    "CanESM5": "#0072B2",
    "CESM2": "#CC79A7",
    "CNRM-ESM2-1": "#E69F00",
    "GFDL-ESM4": "#56B4E9",
    "MIROC-ES2L": "#B89C00",
    "UKESM1-0-LL": "#8064A2",
    "CAS-ESM2-0": "#009E73",
    "NorESM2-LM": "#333333",
}

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9.0,
    "axes.labelsize": 10.0,
    "xtick.labelsize": 8.6,
    "ytick.labelsize": 8.6,
    "legend.fontsize": 8.0,
    "axes.linewidth": 0.85,
    "xtick.major.width": 0.75,
    "ytick.major.width": 0.75,
    "xtick.major.size": 3.8,
    "ytick.major.size": 3.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

# %% 6. 创建Figure
fig, (ax_co2, ax_sia, ax_amoc) = plt.subplots(3, 1, figsize=(7.9, 8.1), sharex=True, gridspec_kw={"height_ratios": [0.72, 3.0, 3.0]})

fig.suptitle("Individual-model North Atlantic sea-ice and AMOC evolution", x=0.53, y=0.986, fontsize=11.7, fontweight="bold", color=TEXT_COLOR)
fig.text(0.53, 0.961, "North Atlantic: 45–70°N, 60°W–30°E", ha="center", va="top", fontsize=8.4, color="0.38")

# %% 7. CO2 trajectory
ax_co2.plot(x, co2, color=CO2_COLOR, lw=1.9, solid_capstyle="round")
ax_co2.set_ylim(co2_min - co2_pad, co2_max + co2_pad)
ax_co2.set_ylabel(r"CO$_2$" "\n(ppm)", color=CO2_COLOR, labelpad=6)
ax_co2.tick_params(axis="y", colors=CO2_COLOR)
ax_co2.yaxis.set_major_locator(MaxNLocator(3))
ax_co2.grid(axis="y", color=GRID_COLOR, lw=0.5, alpha=0.55)
ax_co2.spines["top"].set_visible(False)
ax_co2.spines["right"].set_visible(False)
ax_co2.axvline(transition_x, color="0.42", lw=0.8, ls="--", alpha=0.65)

ax_co2.text(0.20, 1.10, r"1pctCO$_2$", transform=ax_co2.transAxes, ha="center", va="bottom", fontsize=8.8, color=TEXT_COLOR)
ax_co2.text(0.80, 1.10, r"1pctCO$_2$-cdr", transform=ax_co2.transAxes, ha="center", va="bottom", fontsize=8.8, color=TEXT_COLOR)

# %% 8. SIA和AMOC公共样式
for ax in [ax_sia, ax_amoc]:
    ax.axvline(transition_x, color="0.42", lw=0.8, ls="--", alpha=0.65, zorder=1)
    ax.axhline(0.0, color="0.4", lw=0.7, alpha=0.45, zorder=1)
    ax.grid(axis="y", color=GRID_COLOR, lw=0.55, alpha=0.65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0)

# %% 9. 九模式曲线
for model in models:
    color = MODEL_COLORS[model]
    ax_sia.plot(x, data[f"sia_{model}"].to_numpy(dtype=float), color=color, lw=1.50, alpha=0.96, solid_capstyle="round")
    ax_amoc.plot(x, data[f"amoc_{model}"].to_numpy(dtype=float), color=color, lw=1.50, alpha=0.96, solid_capstyle="round")

ax_sia.set_ylim(sia_min - sia_pad, sia_max + sia_pad)
ax_amoc.set_ylim(amoc_min - amoc_pad, amoc_max + amoc_pad)

ax_sia.set_ylabel(r"North Atlantic SIA anomaly (10$^6$ km$^2$)", labelpad=7)
ax_amoc.set_ylabel("40°N AMOC anomaly (Sv)", labelpad=7)
ax_amoc.set_xlabel("Simulation year", labelpad=6)

ax_amoc.set_xlim(1, 280)
ax_amoc.set_xticks([1, 40, 80, 120, 140, 180, 220, 260, 280])

# %% 10. 图例
handles = [Line2D([0], [0], color=MODEL_COLORS[model], lw=2.0, label=model) for model in models]

fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.025), ncol=3, frameon=False, handlelength=2.2, handletextpad=0.55, columnspacing=1.5, labelspacing=0.5)
fig.text(0.965, 0.026, f"n = {len(models)} models", ha="right", va="bottom", fontsize=7.8, color="0.45")

# %% 11. 布局与panel labels
fig.subplots_adjust(left=0.165, right=0.975, bottom=0.160, top=0.900, hspace=0.145)

sia_pos = ax_sia.get_position()
amoc_pos = ax_amoc.get_position()

fig.text(sia_pos.x0, sia_pos.y1 + 0.008, "(a)", ha="left", va="bottom", fontsize=10.8, fontweight="bold", color=TEXT_COLOR)
fig.text(amoc_pos.x0, amoc_pos.y1 + 0.008, "(b)", ha="left", va="bottom", fontsize=10.8, fontweight="bold", color=TEXT_COLOR)

# %% 12. 覆盖保存，不弹窗
fig.savefig(PNG_FILE, dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
fig.savefig(PDF_FILE, bbox_inches="tight", pad_inches=0.05, facecolor="white")
fig.savefig(SVG_FILE, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.close(fig)

print(f"[覆盖保存] {PNG_FILE}")
print(f"[覆盖保存] {PDF_FILE}")
print(f"[覆盖保存] {SVG_FILE}")