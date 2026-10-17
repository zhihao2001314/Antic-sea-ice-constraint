# -*- coding: utf-8 -*-
"""
绘制North Atlantic两个深水形成区的层结垂直剖面3×3图：
每张图为8个模式 + 1个IAPv4观测，共9个panel。

【是否需要运行】
需要。
前提是以下结果已经存在：
1. 8模式初始March stratification结果；
2. IAPv4观测March stratification结果。

【输入文件】
output/s02sia/s02_arctic_region/p07_stratification/n01_initial_profiles/
    na_initial_march_stratification_profiles.csv

output/s02sia/s02_arctic_region/p07_stratification/n02_iapv4_observation/
    iapv4_march_2005_2024_stratification_profiles.csv
    iapv4_march_2005_2024_stratification_metadata.json

【输出文件】
figure/s02_sia/s02_arctic_region/p07_stratification/n03_profiles_3x3/
    nordic_seas_stratification_profiles_3x3.png/pdf/svg
    labrador_irminger_seas_stratification_profiles_3x3.png/pdf/svg

【什么时候需要重新运行】
模式剖面、观测剖面或绘图样式变化后。

职责
只读取plot-ready结果绘图。
不重新计算N²、区域平均、MMM、观测气候态或坐标范围。
"""

# %% 1. 依赖
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

# %% 2. 路径
ROOT = Path(r"F:\sea\Antic sea ice constraint")

MODEL_FILE = ROOT/"output"/"s02sia"/"s02_arctic_region"/"p07_stratification"/"n01_initial_profiles"/"na_initial_march_stratification_profiles.csv"
OBS_DIR = ROOT/"output"/"s02sia"/"s02_arctic_region"/"p07_stratification"/"n02_iapv4_observation"
OBS_FILE = OBS_DIR/"iapv4_march_2005_2024_stratification_profiles.csv"
META_FILE = OBS_DIR/"iapv4_march_2005_2024_stratification_metadata.json"

OUT = ROOT/"figure"/"s02_sia"/"s02_arctic_region"/"p07_stratification"/"n03_profiles_3x3"
OUT.mkdir(parents=True, exist_ok=True)

# %% 3. 输入检查
for path in [MODEL_FILE, OBS_FILE, META_FILE]:
    if not path.is_file():
        raise FileNotFoundError(path)

# %% 4. 读取数据
profiles = pd.read_csv(MODEL_FILE)
obs_profiles = pd.read_csv(OBS_FILE)

with open(META_FILE, "r", encoding="utf-8") as f:
    metadata = json.load(f)

MODELS = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4", "MIROC-ES2L", "UKESM1-0-LL", "NorESM2-LM"]
REGIONS = ["Nordic Seas", "Labrador-Irminger Seas"]

X_MIN, X_MAX = metadata["plot_x_limits_n2_1e5_s2"]
Y_MIN, Y_MAX = metadata["plot_y_limits_depth_m"]

required_model = {"model", "region", "depth_m", "initial_march_n2_1e5_s2"}
required_obs = {"region", "depth_m", "iapv4_march_mean_n2_1e5_s2"}

missing_model = required_model-set(profiles.columns)
missing_obs = required_obs-set(obs_profiles.columns)

if missing_model:
    raise KeyError(f"模式结果缺少字段：{sorted(missing_model)}")
if missing_obs:
    raise KeyError(f"观测结果缺少字段：{sorted(missing_obs)}")

for model in MODELS:
    for region in REGIONS:
        if profiles[(profiles["model"] == model) & (profiles["region"] == region)].empty:
            raise ValueError(f"缺少模式剖面：{model} | {region}")

for region in REGIONS:
    if obs_profiles[obs_profiles["region"] == region].empty:
        raise ValueError(f"缺少观测剖面：{region}")

# %% 5. 样式
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9.4,
    "axes.titlesize": 10.8,
    "axes.labelsize": 10.0,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 8.4,
    "ytick.labelsize": 8.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})

MODEL_COLORS = {
    "ACCESS-ESM1-5": "#4C78A8",
    "CanESM5": "#F58518",
    "CESM2": "#54A24B",
    "CNRM-ESM2-1": "#E45756",
    "GFDL-ESM4": "#72B7B2",
    "MIROC-ES2L": "#B279A2",
    "UKESM1-0-LL": "#FF9DA6",
    "NorESM2-LM": "#9D755D",
}

OBS_COLOR = "#111111"
GRID_COLOR = "#DDDDDD"

REGION_LABELS = {
    "Nordic Seas": "65–80°N, 30°W–20°E",
    "Labrador-Irminger Seas": "50–65°N, 65°W–20°W",
}

REGION_TITLES = {
    "Nordic Seas": "Initial March stratification profiles in the Nordic Seas",
    "Labrador-Irminger Seas": "Initial March stratification profiles in the Labrador–Irminger Seas",
}

REGION_FILES = {
    "Nordic Seas": "nordic_seas_stratification_profiles_3x3",
    "Labrador-Irminger Seas": "labrador_irminger_seas_stratification_profiles_3x3",
}

PANEL_ORDER = MODELS + ["IAPv4 observation"]

# %% 6. 逐海区绘图
for region in REGIONS:
    fig, axes = plt.subplots(3, 3, figsize=(13.6, 11.2), sharex=True, sharey=True, facecolor="white")
    axes = axes.ravel()

    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.11, wspace=0.16, hspace=0.23)
    fig.suptitle(REGION_TITLES[region], fontsize=16.2, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925, REGION_LABELS[region], ha="center", va="center", fontsize=9.5, color="#666666")

    for i, name in enumerate(PANEL_ORDER):
        ax = axes[i]
        panel_label = chr(ord("a")+i)

        if name == "IAPv4 observation":
            data = obs_profiles[obs_profiles["region"] == region].sort_values("depth_m")
            ax.plot(data["iapv4_march_mean_n2_1e5_s2"], data["depth_m"], color=OBS_COLOR, lw=2.6, zorder=5)
            ax.set_title(f"({panel_label})  IAPv4 observation", loc="left", fontweight="bold", pad=6)
        else:
            data = profiles[(profiles["model"] == name) & (profiles["region"] == region)].sort_values("depth_m")
            ax.plot(data["initial_march_n2_1e5_s2"], data["depth_m"], color=MODEL_COLORS[name], lw=2.1, zorder=5)
            ax.set_title(f"({panel_label})  {name}", loc="left", fontweight="bold", pad=6)

        ax.axvline(0.0, color="#999999", lw=0.6, ls=":", zorder=1)
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MAX, Y_MIN)
        ax.set_yticks([0, 200, 400, 600, 800, 1000])
        ax.grid(True, color=GRID_COLOR, lw=0.45, ls=":", alpha=0.8, zorder=0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i % 3 == 0:
            ax.set_ylabel("Depth (m)")
        if i >= 6:
            ax.set_xlabel(r"$N^2$ ($10^{-5}$ s$^{-2}$)")

    fig.text(0.5, 0.04, "Models: 1pctCO2 years 1–10 March | Observation: IAPv4 2005–2024 March", ha="center", va="center", fontsize=8.5, color="#777777")

    stem = REGION_FILES[region]
    png_file = OUT/f"{stem}.png"
    pdf_file = OUT/f"{stem}.pdf"
    svg_file = OUT/f"{stem}.svg"

    fig.savefig(png_file, dpi=500, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(pdf_file, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    fig.savefig(svg_file, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)

    print(f"[完成] {region}: {png_file}")

# %% 7. 完成
print("\n" + "="*110)
print("STRATIFICATION PROFILE 3x3 FIGURES COMPLETE")
print("="*110)
print(f"figure directory = {OUT}")
print("="*110)