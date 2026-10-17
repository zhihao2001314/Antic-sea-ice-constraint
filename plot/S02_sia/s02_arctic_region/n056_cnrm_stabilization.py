# %% 1. 导入依赖

from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# %% 2. 参数与路径设置

root = Path(r"F:\sea\Antic sea ice constraint")
directory = root / "output/s02sia/s02_arctic_region/p14_mld_fixed_regions_raw"
destination = root / "figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw"
prefix = "cnrm_coast_stabilization_20261013"
regions = ["Labrador-Irminger", "Nordic Seas"]
region_labels = ["Labrador–Irminger", "Nordic–Iceland"]
region_colors = ["#D55E00", "#0072B2"]
phases = ["ramp_up", "ramp_down", "stabilization"]
phase_labels = ["Ramp-up", "Ramp-down", "Stabilization"]
phase_colors = ["#D55E00", "#0072B2", "#009E73"]
plt.rcParams.update({"font.family": "Arial", "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 16, "axes.linewidth": 0.9, "pdf.fonttype": 42, "svg.fonttype": "none"})
assert not any((destination / (prefix + suffix + "." + extension)).exists() for suffix in ["_timeseries", "_hysteresis"] for extension in ["png", "pdf", "svg"])


# %% 3. 仅读取完整阶段计算结果

annual = pd.read_csv(directory / (prefix + "_annual.csv"))
summary = pd.read_csv(directory / (prefix + "_summary.csv"))
ocean = annual.loc[annual.region == regions[0]]


# %% 4. 时间序列：强迫、翻转、深混合与深混合面积

fig, axes = plt.subplots(4, 1, figsize=(13, 12), sharex=True, gridspec_kw={"height_ratios": [0.8, 1.2, 1.2, 1.0]})
fig.subplots_adjust(left=0.10, right=0.98, bottom=0.10, top=0.86, hspace=0.42)
for ax in axes:
    ax.axvspan(280.5, 340, color="#009E73", alpha=0.075, linewidth=0)
    ax.axvline(140.5, color="#9AA2AA", linestyle="--", linewidth=0.8)
    ax.axvline(280.5, color="#9AA2AA", linestyle="--", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(1, 340)
axes[0].plot(ocean.simulation_year, ocean.co2_ppm, color="#666666", linewidth=2)
axes[0].set_ylabel("CO$_2$ (ppm)")
axes[0].set_title("(a) Prescribed CO$_2$ trajectory", loc="left", pad=10)
for year, label, color in [(70, "Ramp-up", phase_colors[0]), (210, "Ramp-down", phase_colors[1]), (310, "Stabilization", phase_colors[2])]:
    axes[0].text(year, 1250, label, color=color, ha="center", fontsize=12)
axes[0].set_ylim(200, 1400)
for column, label, color, style in [("amoc_40n_sv", "AMOC at 40°N", "#222222", "-"), ("overturning_65n_sv", "Overturning at 65°N", "#0072B2", "-"), ("li_transport_proxy_sv", "40°N minus 65°N proxy", "#D55E00", "--")]:
    axes[1].plot(ocean.simulation_year, ocean[column], color=color, linewidth=0.6, alpha=0.20)
    axes[1].plot(ocean.simulation_year, ocean[column + "_11yr"], color=color, linewidth=2, linestyle=style, label=label)
axes[1].set_ylabel("Strength (Sv)")
axes[1].set_ylim(0, 29)
axes[1].set_title("(b) Atlantic overturning diagnostics", loc="left", pad=10)
axes[1].legend(loc="upper right", frameon=False, fontsize=10, ncol=3)
for index, region in enumerate(regions):
    subset = annual.loc[annual.region == region]
    for ax, column in [(axes[2], "mld_mean_m"), (axes[3], "deep_area_1000m_percent")]:
        ax.plot(subset.simulation_year, subset[column], color=region_colors[index], linewidth=0.6, alpha=0.20)
        ax.plot(subset.simulation_year, subset[column + "_11yr"], color=region_colors[index], linewidth=2, label=region_labels[index])
axes[2].set_title("(c) March mixed-layer depth", loc="left", pad=10)
axes[2].set_ylabel("Regional mean (m)")
axes[2].legend(loc="upper right", frameon=False, ncol=2, fontsize=10)
axes[3].set_title("(d) Resolved deep-mixing area: MLD ≥ 1000 m", loc="left", pad=10)
axes[3].set_ylabel("Area fraction (%)")
axes[3].set_xlabel("Simulation year")
axes[3].set_ylim(bottom=0)
fig.suptitle("CNRM-ESM2-1: recovery under stabilized CO$_2$", fontsize=25, weight="bold", y=0.965)
fig.text(0.54, 0.915, "Ramp-up → ramp-down → constant forcing", ha="center", fontsize=15, color="#505A63")
fig.text(0.54, 0.035, "Thin: annual values; thick: 11-year means. Shading: stabilization.", ha="center", fontsize=11, color="#64717B")
for extension in ["png", "pdf", "svg"]:
    fig.savefig(destination / (prefix + "_timeseries." + extension), dpi=400, facecolor="white")
plt.close(fig)


# %% 5. CO2响应路径，箭头沿真实时间方向

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.18, hspace=0.38, wspace=0.22)
panels = [(regions[0], "mld_mean_m", "(a) Labrador–Irminger deep mixing", "March mean MLD (m)"), (regions[1], "mld_mean_m", "(b) Nordic–Iceland deep mixing", "March mean MLD (m)"), (regions[0], "amoc_40n_sv", "(c) AMOC at 40°N", "Strength (Sv)"), (regions[0], "overturning_65n_sv", "(d) Overturning at 65°N", "Strength (Sv)")]
for ax, panel in zip(axes.ravel(), panels):
    region, column, title, ylabel = panel
    for index, phase in enumerate(phases):
        subset = annual.loc[(annual.region == region) & (annual.phase == phase)].sort_values("simulation_year")
        ax.plot(subset.co2_ppm, subset[column + "_11yr"], color=phase_colors[index], linewidth=2, linestyle=":" if phase == "stabilization" else "-")
        # 箭头仅链接同一阶段的实际时间索引，不绘制假想平衡分支。
        for first, last in ([(35, 43), (90, 98)] if phase != "stabilization" else [(20, 28)]):
            start = subset.iloc[first]
            end = subset.iloc[last]
            ax.annotate("", xy=(end.co2_ppm, end[column + "_11yr"]), xytext=(start.co2_ppm, start[column + "_11yr"]), arrowprops={"arrowstyle": "-|>", "color": phase_colors[index], "lw": 1.5, "mutation_scale": 15})
        if phase == "stabilization":
            last = subset.loc[subset[column + "_11yr"].notna()].iloc[-1]
            ax.scatter(last.co2_ppm, last[column + "_11yr"], marker="*", s=110, color=phase_colors[index], edgecolor="#333333", linewidth=0.5, zorder=5)
    ax.set_title(title, loc="left", pad=12)
    ax.set_xlabel("Atmospheric CO$_2$ (ppm)")
    ax.set_ylabel(ylabel)
    ax.set_xlim(240, 1200)
    ax.set_ylim(bottom=0)
    ax.spines[["top", "right"]].set_visible(False)
handles = [Line2D([], [], color=phase_colors[index], linewidth=2, linestyle=":" if phase == "stabilization" else "-", label=phase_labels[index]) for index, phase in enumerate(phases)]
fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.54, 0.075), frameon=False, ncol=3, fontsize=13)
fig.suptitle("CNRM-ESM2-1: CO$_2$–response pathways", fontsize=25, weight="bold", y=0.965)
fig.text(0.54, 0.90, "Arrows follow time; stabilization proceeds at constant CO$_2$", ha="center", fontsize=14, color="#505A63")
fig.text(0.54, 0.035, "11-year means; star: final available stabilized state.", ha="center", fontsize=11, color="#64717B")
for extension in ["png", "pdf", "svg"]:
    fig.savefig(destination / (prefix + "_hysteresis." + extension), dpi=400, facecolor="white")
plt.close(fig)
print(destination / (prefix + "_timeseries.png"))
print(destination / (prefix + "_hysteresis.png"))
