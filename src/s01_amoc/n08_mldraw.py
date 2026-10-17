# -*- coding: utf-8 -*-
"""
AMOC深水形成区原始强度补充诊断：40°N、65°N及40°N−65°N。

【是否需要运行】
需要。先完成：
src/amoc/n12_amoc_deepwater_regions_300m.py

完成后运行：
plot/amoc/n13_amoc_deepwater_regions_raw_strength.py

【输入文件】
output/amoc_s01_n01/n12_deepwater_regions_300m/
    amoc_deepwater_regions_300m_annual.csv
    amoc_deepwater_regions_300m_pi_reference.csv
    amoc_deepwater_regions_300m_metadata.json

【输出文件】
output/amoc_s01_n01/n13_deepwater_regions_raw_strength/
    amoc_deepwater_regions_raw_strength_plot_data.csv
    amoc_deepwater_regions_raw_strength_metadata.json

【什么时候需要重新运行】
n12年度结果或平滑窗口变化后。

科学定义
1. 直接使用n12已提取的forced raw AMOC strength。
2. 40°N作为总体AMOC诊断。
3. 65°N作为GIN Seas高纬AMOC诊断。
4. 40°N−65°N作为Labrador-Irminger Sea相关proxy，不是严格输运收支。
5. 不减piControl climatology，因此不是anomaly。
6. 不增加额外piControl drift correction。
7. 11年中心滑动平均仅用于展示。
8. 不重新读取原始NetCDF，不改变n12 AMOC定义。
"""

# %% 1. 依赖
from pathlib import Path
import json
import numpy as np
import pandas as pd

# %% 2. 路径与参数
ROOT = Path(r"F:\sea\Antic sea ice constraint")

N12_DIR = ROOT/"output"/"amoc_s01_n01"/"n12_deepwater_regions_300m"
OUT = ROOT/"output"/"amoc_s01_n01"/"n13_deepwater_regions_raw_strength"
OUT.mkdir(parents=True, exist_ok=True)

ANNUAL_SOURCE = N12_DIR/"amoc_deepwater_regions_300m_annual.csv"
PI_SOURCE = N12_DIR/"amoc_deepwater_regions_300m_pi_reference.csv"
N12_META_SOURCE = N12_DIR/"amoc_deepwater_regions_300m_metadata.json"

PLOT_FILE = OUT/"amoc_deepwater_regions_raw_strength_plot_data.csv"
META_FILE = OUT/"amoc_deepwater_regions_raw_strength_metadata.json"

MODELS = [
    "ACCESS-ESM1-5", "CanESM5", "CESM2",
    "CNRM-ESM2-1", "GFDL-ESM4", "MIROC-ES2L",
    "UKESM1-0-LL", "NorESM2-LM", "CAS-ESM2-0",
]

SMOOTH_YEARS = 11
RAMP_YEARS = 140
CO2_BASELINE_PPM = 284.7

# %% 3. 输入检查
for path in [ANNUAL_SOURCE, PI_SOURCE, N12_META_SOURCE]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少：{path}")

annual = pd.read_csv(ANNUAL_SOURCE)
pi_reference = pd.read_csv(PI_SOURCE)

with open(N12_META_SOURCE, "r", encoding="utf-8") as file:
    n12_metadata = json.load(file)

required_annual = {
    "model", "simulation_year",
    "amoc_40N_raw_sv", "amoc_65N_raw_sv",
    "labrador_irminger_proxy_raw_sv",
}

required_pi = {
    "model", "pi_40N_mean_sv", "pi_65N_mean_sv",
    "pi_40N_minus_65N_mean_sv",
}

missing_annual = required_annual-set(annual.columns)
missing_pi = required_pi-set(pi_reference.columns)

if missing_annual:
    raise KeyError(f"n12 annual缺少字段：{sorted(missing_annual)}")

if missing_pi:
    raise KeyError(f"n12 pi reference缺少字段：{sorted(missing_pi)}")

successful = list(n12_metadata["models_successful"])

if set(successful) != set(MODELS):
    raise RuntimeError(f"n12成功模式与预期9模式不一致：{successful}")

if annual.duplicated(["model", "simulation_year"]).any():
    raise RuntimeError("n12 annual存在重复model/simulation_year")

if pi_reference["model"].duplicated().any():
    raise RuntimeError("n12 pi reference存在重复模式")

# %% 4. CO2轨迹
simulation_year = np.arange(1, 281)
index = simulation_year-1
co2_exponent = np.where(index < RAMP_YEARS, index+1, np.maximum(2*RAMP_YEARS-1-index, 0))
co2_ppm = CO2_BASELINE_PPM*1.01**co2_exponent

# %% 5. 原始强度与11年平滑
plot_rows = []
summary_rows = []

for model in MODELS:
    model_data = annual[annual["model"] == model].sort_values("simulation_year").copy()

    if len(model_data) != 280:
        raise RuntimeError(f"{model} n12年度数据长度={len(model_data)}，应为280")

    years = model_data["simulation_year"].to_numpy(int)

    if not np.array_equal(years, simulation_year):
        raise RuntimeError(f"{model} simulation_year不是连续1-280")

    raw40 = model_data["amoc_40N_raw_sv"].to_numpy(float)
    raw65 = model_data["amoc_65N_raw_sv"].to_numpy(float)
    raw_li = model_data["labrador_irminger_proxy_raw_sv"].to_numpy(float)

    if not np.all(np.isfinite(raw40)) or not np.all(np.isfinite(raw65)) or not np.all(np.isfinite(raw_li)):
        raise RuntimeError(f"{model} raw AMOC存在NaN/Inf")

    if not np.allclose(raw_li, raw40-raw65, atol=1e-8):
        raise RuntimeError(f"{model} raw 40N−65N内部一致性失败")

    smooth40 = pd.Series(raw40).rolling(SMOOTH_YEARS, center=True, min_periods=1).mean().to_numpy()
    smooth65 = pd.Series(raw65).rolling(SMOOTH_YEARS, center=True, min_periods=1).mean().to_numpy()
    smooth_li = pd.Series(raw_li).rolling(SMOOTH_YEARS, center=True, min_periods=1).mean().to_numpy()

    pi = pi_reference[pi_reference["model"] == model]

    if len(pi) != 1:
        raise RuntimeError(f"{model} piControl reference记录数={len(pi)}")

    pi40 = float(pi["pi_40N_mean_sv"].iloc[0])
    pi65 = float(pi["pi_65N_mean_sv"].iloc[0])
    pi_li = float(pi["pi_40N_minus_65N_mean_sv"].iloc[0])

    for i in range(280):
        plot_rows.append({
            "model": model,
            "simulation_year": int(simulation_year[i]),
            "experiment": "1pctCO2" if simulation_year[i] <= RAMP_YEARS else "1pctCO2-cdr",
            "co2_ppm": float(co2_ppm[i]),
            "amoc_40N_raw_sv": float(raw40[i]),
            "amoc_65N_raw_sv": float(raw65[i]),
            "labrador_irminger_proxy_raw_sv": float(raw_li[i]),
            "amoc_40N_raw_11yr_sv": float(smooth40[i]),
            "amoc_65N_raw_11yr_sv": float(smooth65[i]),
            "labrador_irminger_proxy_raw_11yr_sv": float(smooth_li[i]),
            "pi_40N_mean_sv": pi40,
            "pi_65N_mean_sv": pi65,
            "pi_40N_minus_65N_mean_sv": pi_li,
        })

    summary_rows.append({
        "model": model,
        "pi_40N_mean_sv": pi40,
        "pi_65N_mean_sv": pi65,
        "pi_40N_minus_65N_mean_sv": pi_li,
        "initial10_40N_sv": float(np.mean(raw40[:10])),
        "initial10_65N_sv": float(np.mean(raw65[:10])),
        "initial10_li_sv": float(np.mean(raw_li[:10])),
        "minimum_40N_sv": float(np.min(smooth40)),
        "minimum_65N_sv": float(np.min(smooth65)),
        "minimum_li_sv": float(np.min(smooth_li)),
        "final10_40N_sv": float(np.mean(raw40[-10:])),
        "final10_65N_sv": float(np.mean(raw65[-10:])),
        "final10_li_sv": float(np.mean(raw_li[-10:])),
    })

    print(
        f"{model:<15} | "
        f"40N {np.mean(raw40[:10]):6.2f}->{np.mean(raw40[-10:]):6.2f} Sv | "
        f"65N {np.mean(raw65[:10]):6.2f}->{np.mean(raw65[-10:]):6.2f} Sv | "
        f"LI {np.mean(raw_li[:10]):6.2f}->{np.mean(raw_li[-10:]):6.2f} Sv"
    )

plot_data = pd.DataFrame(plot_rows)
summary = pd.DataFrame(summary_rows)

# %% 6. 公共纵轴
smooth_values = plot_data[
    ["amoc_40N_raw_11yr_sv", "amoc_65N_raw_11yr_sv", "labrador_irminger_proxy_raw_11yr_sv"]
].to_numpy(float)

data_min = float(np.nanmin(smooth_values))
data_max = float(np.nanmax(smooth_values))

y_min = float(np.floor((data_min-1.0)/5.0)*5.0)
y_max = float(np.ceil((data_max+1.0)/5.0)*5.0)
y_min = min(0.0, y_min)

if y_max <= y_min:
    raise RuntimeError(f"纵轴范围异常：{y_min}, {y_max}")

y_ticks = np.arange(y_min, y_max+0.1, 5.0).tolist()

# %% 7. 保存
plot_data.to_csv(PLOT_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")
summary.to_csv(OUT/"amoc_deepwater_regions_raw_strength_summary.csv", index=False, encoding="utf-8-sig", float_format="%.8f")

metadata = {
    "source": str(ANNUAL_SOURCE),
    "models": MODELS,
    "model_count": len(MODELS),
    "basin": n12_metadata["basin"],
    "depth_definition": n12_metadata["depth_definition"],
    "forced_sequence": n12_metadata["forced_sequence"],
    "value_definition": "raw forced overturning strength; no piControl climatology subtraction",
    "drift_correction": "none; inherited from n12",
    "smoothing": f"{SMOOTH_YEARS}-year centered running mean",
    "diagnostics": {
        "40N": "total AMOC diagnostic",
        "65N": "GIN Seas high-latitude diagnostic",
        "40N_minus_65N": "Labrador-Irminger Sea-related proxy; not a strict transport budget",
    },
    "y_limits_sv": [y_min, y_max],
    "y_ticks_sv": y_ticks,
    "co2_limits_ppm": [250.0, 1200.0],
    "co2_ticks_ppm": [285.0, 700.0, 1147.0],
}

META_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

# %% 8. 完成
print("\n"+"="*108)
print("AMOC RAW-STRENGTH SUPPLEMENT COMPLETE")
print("="*108)
print(summary.round(2).to_string(index=False))
print(f"\nplot-ready = {PLOT_FILE}")
print(f"metadata   = {META_FILE}")
print(f"y range    = {y_min:.0f} to {y_max:.0f} Sv")
print("="*108)