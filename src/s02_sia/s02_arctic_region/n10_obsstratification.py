# -*- coding: utf-8 -*-
"""
计算IAPv4观测的Nordic Seas和Labrador-Irminger Seas March stratification垂向剖面。

【是否需要运行】
需要。

【输入文件】
F:/sea/IAP-1-2000-stratification-1960-2025/
    IAPv4_1_2000m_N2_year_YYYY_month_MM.nc

已有模式结果，仅用于共同深度和最终绘图范围：
output/s02sia/s02_arctic_region/p07_stratification/n01_initial_profiles/
    na_initial_march_stratification_profiles.csv

【输出文件】
output/s02sia/s02_arctic_region/p07_stratification/n02_iapv4_observation/
    iapv4_march_2005_2024_stratification_profiles.csv
    iapv4_march_2005_2024_stratification_metadata.json

【什么时候需要重新运行】
IAPv4数据、观测时间区间、海区范围、模式共同深度或绘图范围定义变化后。

科学定义
1. 只计算IAPv4观测，不重新计算、不修改、不覆盖原来的8模式和MMM结果。
2. 观测采用2005-2024年March，共20个March。
3. Nordic Seas = 65-80°N, 30°W-20°E。
4. Labrador-Irminger Seas = 50-65°N, 65°W-20°W。
5. IAPv4为规则1°网格，区域平均使用cos(latitude)面积权重。
6. IAPv4 N²变量为N2，原始单位10^{-4} s^{-2}。
7. IAPv4 N²垂向坐标为depth_mid，单位m，standard_name=depth_std。
8. 每个March先计算区域平均N²剖面，再对20个March取平均。
9. 观测平均剖面线性插值到n18模式共同N²深度5-995 m，不做垂向外推。
10. 最终绘图横轴范围在本src中根据模式、MMM和观测共同确定并写入metadata。
"""

# %% 1. 导入依赖
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import xarray as xr

# %% 2. 参数与路径设置
ROOT = Path(r"F:\sea\Antic sea ice constraint")
OBS_DIR = Path(r"F:\sea\IAP-1-2000-stratification-1960-2025")

MODEL_PROFILE_FILE = ROOT/"output"/"s02sia"/"s02_arctic_region"/"p07_stratification"/"n01_initial_profiles"/"na_initial_march_stratification_profiles.csv"

OUT = ROOT/"output"/"s02sia"/"s02_arctic_region"/"p07_stratification"/"n02_iapv4_observation"
OUT.mkdir(parents=True, exist_ok=True)

PROFILE_OUT = OUT/"iapv4_march_2005_2024_stratification_profiles.csv"
META_OUT = OUT/"iapv4_march_2005_2024_stratification_metadata.json"

MODELS = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4", "MIROC-ES2L", "UKESM1-0-LL", "NorESM2-LM"]
REGIONS = ["Nordic Seas", "Labrador-Irminger Seas"]

OBS_YEAR_START, OBS_YEAR_END = 2005, 2024
OBS_MONTH = 3
OBS_YEAR_COUNT = OBS_YEAR_END-OBS_YEAR_START+1

NORDIC_LAT_MIN, NORDIC_LAT_MAX = 65.0, 80.0
NORDIC_LON_MIN, NORDIC_LON_MAX = -30.0, 20.0
LI_LAT_MIN, LI_LAT_MAX = 50.0, 65.0
LI_LON_MIN, LI_LON_MAX = -65.0, -20.0

N2_SCALE_TO_S2 = 1e-4
PLOT_Y_MIN, PLOT_Y_MAX = 0.0, 1000.0

# %% 3. 输入检查与模式共同深度
if not OBS_DIR.is_dir():
    raise FileNotFoundError(f"IAPv4目录不存在：{OBS_DIR}")

if not MODEL_PROFILE_FILE.is_file():
    raise FileNotFoundError(f"缺少原n18模式结果：{MODEL_PROFILE_FILE}")

model_profiles = pd.read_csv(MODEL_PROFILE_FILE)
required_model_columns = {"model", "region", "depth_m", "initial_march_n2_1e5_s2"}
missing_model_columns = required_model_columns-set(model_profiles.columns)

if missing_model_columns:
    raise KeyError(f"n18结果缺少字段：{sorted(missing_model_columns)}")

target_depth = np.sort(model_profiles.loc[model_profiles["model"] == "MMM", "depth_m"].dropna().unique().astype(float))

if target_depth.size == 0:
    raise ValueError("n18结果没有MMM共同深度")

if np.any(np.diff(target_depth) <= 0.0):
    raise ValueError("n18共同深度不是严格递增")

print(f"[target depth] {target_depth[0]:.1f}-{target_depth[-1]:.1f} m | n={target_depth.size}")

# %% 4. 建立2005-2024年March文件清单
obs_files = []

for path in sorted(OBS_DIR.glob("*.nc")):
    match = re.search(r"year_(\d{4})_month_(\d{2})", path.name, flags=re.IGNORECASE)

    if match is None:
        continue

    year = int(match.group(1))
    month = int(match.group(2))

    if OBS_YEAR_START <= year <= OBS_YEAR_END and month == OBS_MONTH:
        obs_files.append((year, path))

obs_files = sorted(obs_files, key=lambda item: item[0])
obs_years = np.array([item[0] for item in obs_files], dtype=int)
expected_years = np.arange(OBS_YEAR_START, OBS_YEAR_END+1)

if not np.array_equal(obs_years, expected_years):
    raise FileNotFoundError(f"IAPv4 March文件不完整：找到={obs_years.tolist()}，期望={expected_years.tolist()}")

print("\n"+"="*110)
print("IAPv4 MARCH STRATIFICATION OBSERVATION")
print("="*110)
print(f"[period] {OBS_YEAR_START}-{OBS_YEAR_END} March | files={len(obs_files)}")
print(f"[first] {obs_files[0][1]}")
print(f"[last ] {obs_files[-1][1]}")

# %% 5. 严格检查IAPv4产品结构
with xr.open_dataset(obs_files[0][1]) as first_ds:
    if "N2" not in first_ds.data_vars:
        raise KeyError("IAPv4缺少N2变量")

    if "depth_mid" not in first_ds.coords or "lat" not in first_ds.coords or "lon" not in first_ds.coords:
        raise KeyError("IAPv4缺少depth_mid、lat或lon坐标")

    if first_ds["N2"].dims != ("depth_mid", "lat", "lon"):
        raise ValueError(f"IAPv4 N2 dims异常：{first_ds['N2'].dims}")

    if str(first_ds["N2"].attrs.get("units", "")).strip() != "10^{-4} s^{-2}":
        raise ValueError(f"IAPv4 N2单位异常：{first_ds['N2'].attrs.get('units')}")

    if str(first_ds["depth_mid"].attrs.get("units", "")).strip() != "m":
        raise ValueError(f"IAPv4 depth_mid单位异常：{first_ds['depth_mid'].attrs.get('units')}")

    if str(first_ds["depth_mid"].attrs.get("standard_name", "")).strip() != "depth_std":
        raise ValueError(f"IAPv4 depth_mid standard_name异常：{first_ds['depth_mid'].attrs.get('standard_name')}")

    native_depth_reference = np.asarray(first_ds["depth_mid"].values, dtype=float)
    lat = np.asarray(first_ds["lat"].values, dtype=float)
    lon = (np.asarray(first_ds["lon"].values, dtype=float)+180.0)%360.0-180.0

print(f"[N2] dims=('depth_mid', 'lat', 'lon') | units=10^-4 s^-2")
print(f"[depth_mid] {native_depth_reference[0]:.1f}-{native_depth_reference[-1]:.1f} m | n={native_depth_reference.size}")

# %% 6. 建立两个海区和面积权重
lon2, lat2 = np.meshgrid(lon, lat)
weights2 = np.cos(np.deg2rad(lat2))

nordic_mask = (lat2 >= NORDIC_LAT_MIN) & (lat2 < NORDIC_LAT_MAX) & (lon2 >= NORDIC_LON_MIN) & (lon2 < NORDIC_LON_MAX)
li_mask = (lat2 >= LI_LAT_MIN) & (lat2 < LI_LAT_MAX) & (lon2 >= LI_LON_MIN) & (lon2 < LI_LON_MAX)

if not np.any(nordic_mask):
    raise ValueError("Nordic Seas没有IAPv4格点")

if not np.any(li_mask):
    raise ValueError("Labrador-Irminger Seas没有IAPv4格点")

print(f"[Nordic Seas] cells={int(nordic_mask.sum())}")
print(f"[Labrador-Irminger Seas] cells={int(li_mask.sum())}")

yearly_nordic = np.full((OBS_YEAR_COUNT, native_depth_reference.size), np.nan)
yearly_li = np.full((OBS_YEAR_COUNT, native_depth_reference.size), np.nan)

# %% 7. 逐年March计算区域平均N²剖面
for file_index, (year, path) in enumerate(obs_files):
    with xr.open_dataset(path) as ds:
        if ds["N2"].dims != ("depth_mid", "lat", "lon"):
            raise ValueError(f"{path.name}: N2 dims异常={ds['N2'].dims}")

        depth_m = np.asarray(ds["depth_mid"].values, dtype=float)

        if depth_m.shape != native_depth_reference.shape or not np.allclose(depth_m, native_depth_reference, atol=1e-8, rtol=0.0):
            raise ValueError(f"{path.name}: depth_mid与首文件不同")

        n2 = np.asarray(ds["N2"].values, dtype=float)*N2_SCALE_TO_S2

    if n2.shape != (native_depth_reference.size, lat.size, lon.size):
        raise ValueError(f"{path.name}: N2 shape异常={n2.shape}")

    nordic_values = n2[:, nordic_mask]
    nordic_weights = weights2[nordic_mask]
    nordic_valid = np.isfinite(nordic_values) & np.isfinite(nordic_weights[None, :]) & (nordic_weights[None, :] > 0.0)
    nordic_weight_sum = np.nansum(np.where(nordic_valid, nordic_weights[None, :], np.nan), axis=1)
    nordic_weighted_sum = np.nansum(np.where(nordic_valid, nordic_values*nordic_weights[None, :], np.nan), axis=1)
    nordic_good = nordic_weight_sum > 0.0
    yearly_nordic[file_index, nordic_good] = nordic_weighted_sum[nordic_good]/nordic_weight_sum[nordic_good]

    li_values = n2[:, li_mask]
    li_weights = weights2[li_mask]
    li_valid = np.isfinite(li_values) & np.isfinite(li_weights[None, :]) & (li_weights[None, :] > 0.0)
    li_weight_sum = np.nansum(np.where(li_valid, li_weights[None, :], np.nan), axis=1)
    li_weighted_sum = np.nansum(np.where(li_valid, li_values*li_weights[None, :], np.nan), axis=1)
    li_good = li_weight_sum > 0.0
    yearly_li[file_index, li_good] = li_weighted_sum[li_good]/li_weight_sum[li_good]

    print(f"[{file_index+1:02d}/{OBS_YEAR_COUNT}] {year}-03")

# %% 8. 计算2005-2024 March观测气候态
nordic_valid_years = np.sum(np.isfinite(yearly_nordic), axis=0)
li_valid_years = np.sum(np.isfinite(yearly_li), axis=0)

nordic_mean_native = np.full(native_depth_reference.size, np.nan)
nordic_sd_native = np.full(native_depth_reference.size, np.nan)
li_mean_native = np.full(native_depth_reference.size, np.nan)
li_sd_native = np.full(native_depth_reference.size, np.nan)

nordic_complete = nordic_valid_years == OBS_YEAR_COUNT
li_complete = li_valid_years == OBS_YEAR_COUNT

nordic_mean_native[nordic_complete] = np.mean(yearly_nordic[:, nordic_complete], axis=0)
nordic_sd_native[nordic_complete] = np.std(yearly_nordic[:, nordic_complete], axis=0, ddof=1)
li_mean_native[li_complete] = np.mean(yearly_li[:, li_complete], axis=0)
li_sd_native[li_complete] = np.std(yearly_li[:, li_complete], axis=0, ddof=1)

# %% 9. 插值到n18模式共同N²深度
nordic_valid = np.isfinite(native_depth_reference) & np.isfinite(nordic_mean_native)
li_valid = np.isfinite(native_depth_reference) & np.isfinite(li_mean_native)

if np.sum(nordic_valid) < 2 or np.sum(li_valid) < 2:
    raise ValueError("IAPv4有效观测深度不足")

nordic_mean = np.interp(target_depth, native_depth_reference[nordic_valid], nordic_mean_native[nordic_valid], left=np.nan, right=np.nan)
nordic_sd = np.interp(target_depth, native_depth_reference[nordic_valid], nordic_sd_native[nordic_valid], left=np.nan, right=np.nan)
li_mean = np.interp(target_depth, native_depth_reference[li_valid], li_mean_native[li_valid], left=np.nan, right=np.nan)
li_sd = np.interp(target_depth, native_depth_reference[li_valid], li_sd_native[li_valid], left=np.nan, right=np.nan)

# %% 10. 整理观测plot-ready结果
output_rows = []

for depth_index, depth_m in enumerate(target_depth):
    output_rows.append({
        "region": "Nordic Seas",
        "depth_m": float(depth_m),
        "iapv4_march_mean_n2_s2": float(nordic_mean[depth_index]),
        "iapv4_march_mean_n2_1e5_s2": float(nordic_mean[depth_index]*1e5),
        "iapv4_march_interannual_sd_n2_s2": float(nordic_sd[depth_index]),
        "iapv4_march_interannual_sd_n2_1e5_s2": float(nordic_sd[depth_index]*1e5),
    })

    output_rows.append({
        "region": "Labrador-Irminger Seas",
        "depth_m": float(depth_m),
        "iapv4_march_mean_n2_s2": float(li_mean[depth_index]),
        "iapv4_march_mean_n2_1e5_s2": float(li_mean[depth_index]*1e5),
        "iapv4_march_interannual_sd_n2_s2": float(li_sd[depth_index]),
        "iapv4_march_interannual_sd_n2_1e5_s2": float(li_sd[depth_index]*1e5),
    })

obs_profiles = pd.DataFrame(output_rows)

# %% 11. 计算最终绘图范围
model_plot_values = model_profiles["initial_march_n2_1e5_s2"].to_numpy(float)
obs_plot_values = obs_profiles["iapv4_march_mean_n2_1e5_s2"].to_numpy(float)
plot_values = np.concatenate([model_plot_values, obs_plot_values])
plot_values = plot_values[np.isfinite(plot_values)]

if plot_values.size == 0:
    raise ValueError("模式和观测均没有有效N²绘图数据")

plot_x_min_data = float(np.nanmin(plot_values))
plot_x_max_data = float(np.nanmax(plot_values))
plot_x_span = plot_x_max_data-plot_x_min_data
plot_x_pad = max(plot_x_span*0.06, 0.1)
plot_x_min = float(np.floor((plot_x_min_data-plot_x_pad)*10.0)/10.0)
plot_x_max = float(np.ceil((plot_x_max_data+plot_x_pad)*10.0)/10.0)

if plot_x_min > 0.0:
    plot_x_min = 0.0

# %% 12. 保存观测结果和metadata
obs_profiles.to_csv(PROFILE_OUT, index=False, encoding="utf-8-sig", float_format="%.10g")

metadata = {
    "models": MODELS,
    "regions": REGIONS,
    "product": "IAPv4 1-2000 m monthly stratification",
    "observation_period": "2005-2024 March",
    "year_count": OBS_YEAR_COUNT,
    "n2_variable": "N2",
    "n2_native_units": "10^{-4} s^{-2}",
    "n2_scale_to_s2": N2_SCALE_TO_S2,
    "native_depth_coordinate": "depth_mid",
    "native_depth_units": "m",
    "native_depth_standard_name": "depth_std",
    "regional_weighting": "cos(latitude)",
    "target_depth_source": str(MODEL_PROFILE_FILE),
    "plot_x_limits_n2_1e5_s2": [plot_x_min, plot_x_max],
    "plot_y_limits_depth_m": [PLOT_Y_MIN, PLOT_Y_MAX],
    "nordic_seas": "65<=lat<80N, -30<=lon<20E",
    "labrador_irminger_seas": "50<=lat<65N, -65<=lon<-20E",
    "note": "Observation-only output. Original n18 CMIP6 model and MMM results are unchanged.",
}

META_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

# %% 13. 完成
print("\n"+"="*110)
print("IAPv4 OBSERVATION PROFILE COMPLETE")
print("="*110)
print(f"[Nordic max N²] {np.nanmax(nordic_mean)*1e5:.3f} ×10^-5 s^-2")
print(f"[LI max N²] {np.nanmax(li_mean)*1e5:.3f} ×10^-5 s^-2")
print(f"[plot x] {plot_x_min:.2f} to {plot_x_max:.2f} ×10^-5 s^-2")
print(f"[profile] {PROFILE_OUT}")
print(f"[metadata] {META_OUT}")
print("[原8模式和MMM结果] 未修改、未覆盖")
print("="*110)