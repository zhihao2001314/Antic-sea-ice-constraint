# -*- coding: utf-8 -*-
"""
AMOC 30°S 时间序列绘图

【是否需要运行】需要
【输入文件】output/amoc_s01_n01/30S/amoc_figure1_9models_plot_data.csv
【输出文件】figure/amoc_s01_n01/30S/AMOC_anomaly_30S_9models.png/.pdf/.svg
【什么时候需要重新运行】30°S plot-ready数据更新或绘图样式修改后

说明：本脚本只负责绘图，不进行AMOC提取、piControl订正、平滑、MMM或标准差计算。
"""


# %% 1. 导入依赖

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd


# %% 2. 参数与路径

ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT/"output"/"amoc_s01_n01"/"l_30s"/"amoc_figure1_9models_plot_data.csv"
FIG_DIR = ROOT/"figure"/"amoc_s01_n01"/"l_30s"
PNG_FILE = FIG_DIR/"AMOC_anomaly_30S_9models.png"
PDF_FILE = FIG_DIR/"AMOC_anomaly_30S_9models.pdf"
SVG_FILE = FIG_DIR/"AMOC_anomaly_30S_9models.svg"

MODELS = [
    "ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4",
    "MIROC-ES2L", "NorESM2-LM", "UKESM1-0-LL", "CAS-ESM2-0",
]

COLORS = {
    "ACCESS-ESM1-5":"#CC79A7",
    "CanESM5":"#009E73",
    "CESM2":"#E69F00",
    "CNRM-ESM2-1":"#0072B2",
    "GFDL-ESM4":"#8C564B",
    "MIROC-ES2L":"#56B4E9",
    "NorESM2-LM":"#D55E00",
    "UKESM1-0-LL":"#A6761D",
    "CAS-ESM2-0":"#6A5ACD",
}

plt.rcParams.update({
    "font.family":"Arial",
    "font.size":13,
    "axes.linewidth":1.8,
    "xtick.major.width":1.6,
    "ytick.major.width":1.6,
    "xtick.minor.width":1.1,
    "ytick.minor.width":1.1,
    "pdf.fonttype":42,
    "ps.fonttype":42,
})


# %% 3. 读取plot-ready数据

if not INPUT_FILE.is_file():
    raise FileNotFoundError(f"缺少30°S plot-ready数据：{INPUT_FILE}")

data = pd.read_csv(INPUT_FILE)

required = ["calendar_year","co2_ppm","mmm","lower","upper",*MODELS]
missing = [column for column in required if column not in data.columns]

if missing:
    raise ValueError(f"plot-ready数据缺少字段：{missing}")

if data.empty:
    raise ValueError("plot-ready数据为空")

year = data["calendar_year"].to_numpy()
co2 = data["co2_ppm"].to_numpy()


# %% 4. 归档旧图

existing = [path for path in (PNG_FILE,PDF_FILE,SVG_FILE) if path.exists()]

if existing:
    archive = ROOT/"archive"/f"{datetime.now():%Y%m%d%H%M}_amoc_30S_figure"

    for path in existing:
        target = archive/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        path.replace(target)

FIG_DIR.mkdir(parents=True,exist_ok=True)


# %% 5. 绘图

fig, ax = plt.subplots(figsize=(13.8,7.8))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.fill_between(
    year, data["lower"], data["upper"],
    facecolor="0.82", edgecolor="none", alpha=0.78, zorder=1,
)

for model in MODELS:
    ax.plot(year,data[model],color=COLORS[model],linewidth=1.9,alpha=0.93,zorder=3)

ax.plot(year,data["mmm"],color="black",linewidth=4.0,zorder=5)
ax.axhline(0,color="black",linewidth=1.6,linestyle=(0,(5,3)),zorder=2)

ax.set_title("AMOC Anomaly",fontsize=22,fontweight="bold",pad=20)
ax.set_xlabel("Calendar Year",fontsize=17,labelpad=12)
ax.set_ylabel("AMOC Anomaly (Sv) at 30°S",fontsize=18,fontweight="bold",labelpad=13)

ax.set_xlim(year[0],year[-1])

first_tick = int((year[0]+49)//50*50)
last_tick = int(year[-1]//50*50)

if first_tick<=last_tick:
    ax.set_xticks(range(first_tick,last_tick+1,50))

ax.minorticks_on()
ax.tick_params(axis="both",which="major",direction="in",length=9,width=1.8,labelsize=14)
ax.tick_params(axis="both",which="minor",direction="in",length=4.5,width=1.2)

ax.spines["left"].set_linewidth(2.0)
ax.spines["bottom"].set_linewidth(2.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(False)


# %% 6. CO2右轴

ax_co2 = ax.twinx()

ax_co2.plot(year,co2,color="#B000C8",linewidth=3.0,linestyle=(0,(7,4)),zorder=4)

ax_co2.set_ylim(100,1250)
ax_co2.set_yticks([200,400,600,800,1000,1200])
ax_co2.set_ylabel("Atmospheric CO$_2$ (ppm)",fontsize=17,fontweight="bold",labelpad=16)

ax_co2.minorticks_on()
ax_co2.tick_params(axis="y",which="major",direction="in",length=9,width=1.8,labelsize=14)
ax_co2.tick_params(axis="y",which="minor",direction="in",length=4.5,width=1.2)

ax_co2.spines["right"].set_linewidth(2.0)
ax_co2.spines["top"].set_visible(False)
ax_co2.spines["left"].set_visible(False)


# %% 7. 图例

legend_handles = [
    Line2D([],[],color=COLORS[model],linewidth=2.5,label=model)
    for model in MODELS
]

legend_handles += [
    Patch(facecolor="0.82",edgecolor="none",alpha=0.78,label=r"$\pm$1 Std Dev"),
    Line2D([],[],color="black",linewidth=4.0,label="9-model MMM"),
    Line2D([],[],color="#B000C8",linewidth=3.0,linestyle=(0,(7,4)),label="Atmospheric CO$_2$"),
]

legend = ax.legend(
    handles=legend_handles,
    loc="center left",
    bbox_to_anchor=(1.14,0.50),
    frameon=True,
    fancybox=False,
    framealpha=1.0,
    edgecolor="black",
    fontsize=10.8,
    handlelength=2.8,
    borderpad=0.7,
    labelspacing=0.55,
)

legend.get_frame().set_linewidth(1.2)

fig.subplots_adjust(left=0.095,right=0.72,bottom=0.14,top=0.88)


# %% 8. 保存

fig.savefig(PNG_FILE,dpi=600,bbox_inches="tight",pad_inches=0.08)
fig.savefig(PDF_FILE,bbox_inches="tight",pad_inches=0.08)
fig.savefig(SVG_FILE,bbox_inches="tight",pad_inches=0.08)

print(f"[完成] PNG：{PNG_FILE}")
print(f"[完成] PDF：{PDF_FILE}")
print(f"[完成] SVG：{SVG_FILE}")

plt.show()
plt.close(fig)