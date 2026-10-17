# %% 1. 导入依赖

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# %% 2. 参数与路径设置

root = Path(r"F:\sea\Antic sea ice constraint")
input_path = root / "output/amoc_s01_n01/l_40/amoc_initial_reference_plot_data.csv"
stem = root / "figure/amoc_s01_n01/l_40/AMOC_change_40N_reference_style"
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4",
          "MIROC-ES2L", "NorESM2-LM", "UKESM1-0-LL", "CAS-ESM2-0"]
colors = ["#CC79A7", "#009E73", "#E69F00", "#0072B2", "#8C564B",
          "#56B4E9", "#D55E00", "#9C7D16", "#8064A2"]
plt.rcParams.update({"font.family": "Arial", "font.size": 13, "pdf.fonttype": 42,
                     "axes.linewidth": 1.8, "xtick.major.width": 1.6, "ytick.major.width": 1.6})


# %% 3. 读取绘图数据

table = pd.read_csv(input_path)


# %% 4. 绘图

# 源序列的日历年已在原始年表记录：第1年对应1850年。
# 这里仅变换显示坐标，不重新计算科学量。
year = table.simulation_year + 1849
fig, ax = plt.subplots(figsize=(13.8, 7.8))
ax.fill_between(year, table.lower, table.upper, color="0.82",
                alpha=.78, edgecolor="none", zorder=1)
for model, color in zip(models, colors):
    ax.plot(year, table[model], color=color, linewidth=1.9, alpha=.93, zorder=3)
ax.plot(year, table.mmm, color="black", linewidth=4, zorder=5)
ax.axhline(0, color="black", linestyle=(0, (5, 3)), linewidth=1.6, zorder=2)
co2_axis = ax.twinx()
co2_axis.plot(year, table.co2_ppm, color="#B000C8", linestyle=(0, (7, 4)),
              linewidth=3, zorder=4)
co2_axis.set_ylabel(r"Atmospheric CO$_2$ (ppm)", fontsize=17, fontweight="bold", labelpad=16)
co2_axis.set_ylim(100, 1250)
co2_axis.set_yticks([200, 400, 600, 800, 1000, 1200])
ax.set_xlim(year.iloc[0], year.iloc[-1])
ax.set_xticks(range(1850, 2151, 50))
ax.set_xlabel("Calendar Year", fontsize=17, labelpad=12)
ax.set_ylabel("AMOC Anomaly (Sv) at 40°N", fontsize=18, fontweight="bold", labelpad=13)
ax.set_title("AMOC Anomaly", fontsize=22, fontweight="bold", pad=20)
ax.minorticks_on()
co2_axis.minorticks_on()
ax.tick_params(axis="both", which="major", direction="in", length=9, width=1.8, labelsize=14)
ax.tick_params(axis="both", which="minor", direction="in", length=4.5, width=1.2)
co2_axis.tick_params(axis="y", which="major", direction="in", length=9, width=1.8, labelsize=14)
co2_axis.tick_params(axis="y", which="minor", direction="in", length=4.5, width=1.2)
ax.spines["left"].set_linewidth(2)
ax.spines["bottom"].set_linewidth(2)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
co2_axis.spines["right"].set_linewidth(2)
co2_axis.spines["top"].set_visible(False)
co2_axis.spines["left"].set_visible(False)
handles = [Line2D([], [], color=color, linewidth=2.5, label=model)
           for model, color in zip(models, colors)]
handles += [Patch(facecolor="0.82", edgecolor="none", alpha=.78, label=r"$\pm$1 Std Dev"),
            Line2D([], [], color="black", linewidth=4, label="9-model MMM"),
            Line2D([], [], color="#B000C8", linewidth=3, linestyle=(0, (7, 4)), label=r"Atmospheric CO$_2$")]
legend = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.14, .5),
                   frameon=True, fancybox=False, framealpha=1, edgecolor="black",
                   fontsize=11, handlelength=2.8, borderpad=.7, labelspacing=.55)
legend.get_frame().set_linewidth(1.2)
fig.subplots_adjust(left=.095, right=.72, bottom=.14, top=.88)


# %% 5. 保存新增图件

# 使用新文件名保留旧图；图中只写变量、参照状态和试验阶段。
for suffix in ["png", "pdf", "svg"]:
    fig.savefig(stem.with_suffix("." + suffix), dpi=600, bbox_inches="tight", pad_inches=.08, facecolor="white")
plt.close(fig)
print(stem)
