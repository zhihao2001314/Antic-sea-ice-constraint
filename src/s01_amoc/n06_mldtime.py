# -*- coding: utf-8 -*-
"""
计算8模式Nordic Seas和Labrador-Irminger Seas March MLD时间序列。

【是否需要运行】
需要。

【输入文件】
F:/sea/cmipthetaoad
F:/sea/cmipthetao
F:/sea/cmip6_so
F:/sea/cmip6_so/NorESM-LM-thkcello
F:/sea/areacello

【输出文件】
output/s02sia/s02_arctic_region/p06_mld/n04_na_region_timeseries/
    na_march_mld_region_timeseries.csv
    na_march_mld_region_summary.csv
    na_march_mld_region_qa.csv
    na_march_mld_region_metadata.json

【什么时候需要重新运行】
原始thetao/so/thkcello/areacello、成员、MLD定义、区域定义、时间窗口或平滑窗口变化后。

科学定义
1. 8模式与现有North Atlantic March MLD图一致。
2. thetao + so按TEOS-10计算potential density anomaly sigma0。
3. MLD采用10 m参考深度、Δsigma0=0.03 kg m-3密度阈值。
4. 只使用每年的March。
5. 1pctCO2前140个实验年映射到simulation year 1-140。
6. 1pctCO2-cdr前140个实验年映射到simulation year 141-280。
7. thetao和so按实验年索引对应，不要求绝对calendar year相同。
8. Nordic Seas诊断区：65-80°N, 30°W-20°E。
9. Labrador-Irminger Seas诊断区：50-65°N, 65°W-20°W。
10. 区域平均使用原生网格areacello面积加权，不插值。
11. anomaly相对各模式、各区域1pctCO2 years 1-10 March MLD平均。
12. 11年中心滑动平均在src中完成，plot不重新计算。
13. 不做piControl漂移订正。
14. 计算前先裁剪目标区域，再按10年批量加载，避免逐年重复磁盘读取。
"""

# %% 1. 导入依赖
from pathlib import Path
import json
import re
import gsw
import numpy as np
import pandas as pd
import xarray as xr

# %% 2. 参数与路径
ROOT = Path(r"F:\sea\Antic sea ice constraint")
SEA = Path(r"F:\sea")

THETAO_ROOTS = {"co2": SEA / "cmipthetaoad", "cdr": SEA / "cmipthetao"}
SO_ROOT = SEA / "cmip6_so"
AREA_ROOT = SEA / "areacello"
NOR_THK_ROOT = SO_ROOT / "NorESM-LM-thkcello"

OUT_DIR = ROOT / "output" / "s02sia" / "s02_arctic_region" / "p06_mld" / "n04_na_region_timeseries"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SERIES_FILE = OUT_DIR / "na_march_mld_region_timeseries.csv"
SUMMARY_FILE = OUT_DIR / "na_march_mld_region_summary.csv"
QA_FILE = OUT_DIR / "na_march_mld_region_qa.csv"
META_FILE = OUT_DIR / "na_march_mld_region_metadata.json"

MODELS = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4", "MIROC-ES2L", "UKESM1-0-LL", "NorESM2-LM"]

THETAO_FOLDERS = {
    "ACCESS-ESM1-5": "ACCESS",
    "CanESM5": "CANESM5",
    "CESM2": "CESM2",
    "CNRM-ESM2-1": "CNRM-ESM2",
    "GFDL-ESM4": "GFDL",
    "MIROC-ES2L": "MIROC-ES2L",
    "UKESM1-0-LL": "UKESM1-0-LL",
    "NorESM2-LM": "NORESM2-LM2",
}

AREA_FOLDERS = {
    "ACCESS-ESM1-5": "ACCESSESM5",
    "CanESM5": "CANESM5",
    "CESM2": "CESM2",
    "CNRM-ESM2-1": "CNRM-ESM2",
    "GFDL-ESM4": "GFDL",
    "MIROC-ES2L": "MIROC-ES2L",
    "UKESM1-0-LL": "UKESM1-0-LL",
    "NorESM2-LM": "NORESM2-LM2",
}

MEMBERS = {
    "ACCESS-ESM1-5": "r1i1p1f1",
    "CanESM5": "r1i1p2f1",
    "CESM2": "r1i1p1f1",
    "CNRM-ESM2-1": "r1i1p1f2",
    "GFDL-ESM4": "r1i1p1f1",
    "MIROC-ES2L": "r1i1p1f2",
    "UKESM1-0-LL": "r1i1p1f2",
    "NorESM2-LM": "r1i1p1f1",
}

EXPERIMENTS = {"co2": "1pctCO2", "cdr": "1pctCO2-cdr"}

YEARS_PER_PHASE = 140
BASELINE_YEARS = 10
SMOOTH_WINDOW = 11
YEAR_BATCH = 10
REFERENCE_DEPTH_M = 10.0
DENSITY_THRESHOLD_KG_M3 = 0.03
CO2_BASELINE_PPM = 284.7
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

NORDIC_LAT_MIN, NORDIC_LAT_MAX = 65.0, 80.0
NORDIC_LON_MIN, NORDIC_LON_MAX = -30.0, 20.0
LI_LAT_MIN, LI_LAT_MAX = 50.0, 65.0
LI_LON_MIN, LI_LON_MAX = -65.0, -20.0

# %% 3. 检查输入目录与建立so文件索引
for path in [THETAO_ROOTS["co2"], THETAO_ROOTS["cdr"], SO_ROOT, AREA_ROOT]:
    if not path.is_dir():
        raise FileNotFoundError(f"目录不存在：{path}")

if not NOR_THK_ROOT.is_dir():
    raise FileNotFoundError(f"NorESM2-LM thkcello目录不存在：{NOR_THK_ROOT}")

print("[扫描] 建立so文件索引...")
SO_FILES_ALL = sorted([path for path in SO_ROOT.rglob("*.nc") if path.name.lower().startswith("so_")])
print(f"[扫描] so NetCDF={len(SO_FILES_ALL)}")

# %% 4. CO2轨迹
simulation_year_all = np.arange(1, 281, dtype=int)
co2_ppm_all = np.empty(280, dtype=float)
co2_ppm_all[:140] = CO2_BASELINE_PPM * 1.01 ** np.arange(1, 141)
co2_ppm_all[140:] = co2_ppm_all[139] * 0.99 ** np.arange(1, 141)

# %% 5. 逐模式计算
series_rows = []
qa_rows = []

for model in MODELS:
    member = MEMBERS[model]

    print("\n" + "=" * 120)
    print(f"[{model}] Nordic Seas / Labrador-Irminger Seas March MLD")
    print("=" * 120)

    # 读取areacello
    area_dir = AREA_ROOT / AREA_FOLDERS[model]
    if not area_dir.is_dir():
        raise FileNotFoundError(f"{model} areacello目录不存在：{area_dir}")

    area_files = sorted([path for path in area_dir.rglob("*.nc") if path.name.lower().startswith("areacello_")])
    if not area_files:
        raise FileNotFoundError(f"{model} 未找到areacello：{area_dir}")

    area_ds = xr.open_dataset(area_files[0], decode_times=False)

    if "areacello" not in area_ds:
        area_ds.close()
        raise KeyError(f"{model} areacello文件缺少areacello变量")

    area_da = area_ds["areacello"].squeeze(drop=True)
    if area_da.ndim != 2:
        area_ds.close()
        raise ValueError(f"{model} areacello必须为二维，实际dims={area_da.dims}")

    area_values_native = np.asarray(area_da.values, dtype=float)

    if "latitude" in area_ds.variables and "longitude" in area_ds.variables:
        area_lat_da = area_ds["latitude"]
        area_lon_da = area_ds["longitude"]
    elif "lat" in area_ds.variables and "lon" in area_ds.variables:
        area_lat_da = area_ds["lat"]
        area_lon_da = area_ds["lon"]
    else:
        area_ds.close()
        raise KeyError(f"{model} areacello缺少latitude/longitude或lat/lon")

    if area_lat_da.ndim == 1 and area_lon_da.ndim == 1:
        area_lat_grid, area_lon_grid = xr.broadcast(area_lat_da, area_lon_da)
        area_lat_native = np.asarray(area_lat_grid.values, dtype=float)
        area_lon_native = np.asarray(area_lon_grid.values, dtype=float)
    elif area_lat_da.ndim == 2 and area_lon_da.ndim == 2:
        area_lat_native = np.asarray(area_lat_da.values, dtype=float)
        area_lon_native = np.asarray(area_lon_da.values, dtype=float)
    else:
        area_ds.close()
        raise ValueError(f"{model} areacello经纬度维数异常")

    area_ds.close()
    area_lon_native = (area_lon_native + 180.0) % 360.0 - 180.0

    # 两个实验
    for stage in ["co2", "cdr"]:
        experiment = EXPERIMENTS[stage]
        theta_dir = THETAO_ROOTS[stage] / THETAO_FOLDERS[model]

        if not theta_dir.is_dir():
            raise FileNotFoundError(f"{model} {experiment} thetao目录不存在：{theta_dir}")

        theta_files = sorted([path for path in theta_dir.rglob("*.nc") if path.name.lower().startswith("thetao_") and f"_{experiment.lower()}_" in path.name.lower() and f"_{member.lower()}_" in path.name.lower()])
        so_files = [path for path in SO_FILES_ALL if f"_{model.lower()}_" in path.name.lower() and f"_{experiment.lower()}_" in path.name.lower() and f"_{member.lower()}_" in path.name.lower()]

        if not theta_files:
            raise FileNotFoundError(f"{model} {experiment} 未找到thetao，member={member}")

        if not so_files:
            raise FileNotFoundError(f"{model} {experiment} 未找到so，member={member}")

        print(f"[{experiment}] thetao files={len(theta_files)} | so files={len(so_files)} | member={member}")

        theta_ds = xr.open_mfdataset(theta_files, combine="by_coords", decode_times=TIME_CODER, chunks={"time": 12}, data_vars="minimal", coords="minimal", compat="override", parallel=False)
        so_ds = xr.open_mfdataset(so_files, combine="by_coords", decode_times=TIME_CODER, chunks={"time": 12}, data_vars="minimal", coords="minimal", compat="override", parallel=False)

        if "thetao" not in theta_ds or "so" not in so_ds:
            theta_ds.close()
            so_ds.close()
            raise KeyError(f"{model} {experiment} 缺少thetao或so变量")

        theta = theta_ds["thetao"]
        salinity = so_ds["so"]

        # 经纬度
        if "latitude" in theta_ds.variables and "longitude" in theta_ds.variables:
            lat_da = theta_ds["latitude"]
            lon_da = theta_ds["longitude"]
        elif "lat" in theta_ds.variables and "lon" in theta_ds.variables:
            lat_da = theta_ds["lat"]
            lon_da = theta_ds["lon"]
        else:
            theta_ds.close()
            so_ds.close()
            raise KeyError(f"{model} {experiment} thetao缺少经纬度")

        if "latitude" in so_ds.variables and "longitude" in so_ds.variables:
            so_lat_da = so_ds["latitude"]
            so_lon_da = so_ds["longitude"]
        elif "lat" in so_ds.variables and "lon" in so_ds.variables:
            so_lat_da = so_ds["lat"]
            so_lon_da = so_ds["lon"]
        else:
            theta_ds.close()
            so_ds.close()
            raise KeyError(f"{model} {experiment} so缺少经纬度")

        if lat_da.ndim == 1 and lon_da.ndim == 1:
            lat_grid_da, lon_grid_da = xr.broadcast(lat_da, lon_da)
            horizontal_dims = list(lat_grid_da.dims)
            lat_native = np.asarray(lat_grid_da.values, dtype=float)
            lon_native = np.asarray(lon_grid_da.values, dtype=float)
        elif lat_da.ndim == 2 and lon_da.ndim == 2:
            horizontal_dims = list(lat_da.dims)
            lat_native = np.asarray(lat_da.values, dtype=float)
            lon_native = np.asarray(lon_da.values, dtype=float)
        else:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} thetao经纬度维数异常")

        if so_lat_da.ndim == 1 and so_lon_da.ndim == 1:
            so_lat_grid_da, so_lon_grid_da = xr.broadcast(so_lat_da, so_lon_da)
            so_horizontal_dims = list(so_lat_grid_da.dims)
            so_lat_native = np.asarray(so_lat_grid_da.values, dtype=float)
            so_lon_native = np.asarray(so_lon_grid_da.values, dtype=float)
        elif so_lat_da.ndim == 2 and so_lon_da.ndim == 2:
            so_horizontal_dims = list(so_lat_da.dims)
            so_lat_native = np.asarray(so_lat_da.values, dtype=float)
            so_lon_native = np.asarray(so_lon_da.values, dtype=float)
        else:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} so经纬度维数异常")

        lon_native = (lon_native + 180.0) % 360.0 - 180.0
        so_lon_native = (so_lon_native + 180.0) % 360.0 - 180.0

        if lat_native.shape != so_lat_native.shape:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} thetao/so水平shape不同")

        grid_valid = np.isfinite(lat_native) & np.isfinite(lon_native) & np.isfinite(so_lat_native) & np.isfinite(so_lon_native)
        lat_grid_error = float(np.nanmax(np.abs(lat_native[grid_valid] - so_lat_native[grid_valid])))
        lon_grid_error = float(np.nanmax(np.abs((lon_native[grid_valid] - so_lon_native[grid_valid] + 180.0) % 360.0 - 180.0)))

        if lat_grid_error > 1.0e-6 or lon_grid_error > 1.0e-6:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} thetao/so网格不一致：Δlat={lat_grid_error:.3e}, Δlon={lon_grid_error:.3e}")

        # 垂向维度
        theta_vertical_dims = [dim for dim in theta.dims if dim != "time" and dim not in horizontal_dims]
        so_vertical_dims = [dim for dim in salinity.dims if dim != "time" and dim not in so_horizontal_dims]

        if len(theta_vertical_dims) != 1 or len(so_vertical_dims) != 1:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} 无法唯一确定垂向维度")

        z_dim = theta_vertical_dims[0]
        so_z_dim = so_vertical_dims[0]

        if theta.sizes[z_dim] != salinity.sizes[so_z_dim]:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} thetao/so垂向层数不同")

        # 只取March和前140个实验年
        theta_march = theta.where(theta_ds["time"].dt.month == 3, drop=True)
        so_march = salinity.where(so_ds["time"].dt.month == 3, drop=True)

        if theta_march.sizes["time"] < YEARS_PER_PHASE or so_march.sizes["time"] < YEARS_PER_PHASE:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} March不足140年：thetao={theta_march.sizes['time']}, so={so_march.sizes['time']}")

        theta_march = theta_march.isel(time=slice(0, YEARS_PER_PHASE))
        so_march = so_march.isel(time=slice(0, YEARS_PER_PHASE))

        theta_years = np.asarray(theta_march["time"].dt.year.values, dtype=int)
        so_years = np.asarray(so_march["time"].dt.year.values, dtype=int)

        if not np.all(np.diff(theta_years) == 1):
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} thetao March年份不连续")

        if not np.all(np.diff(so_years) == 1):
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} so March年份不连续")

        # areacello与thetao网格验证
        if area_values_native.shape == lat_native.shape:
            area_native = area_values_native
            area_lat_test = area_lat_native
            area_lon_test = area_lon_native
        elif area_values_native.T.shape == lat_native.shape:
            area_native = area_values_native.T
            area_lat_test = area_lat_native.T
            area_lon_test = area_lon_native.T
        else:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} areacello shape={area_values_native.shape}无法对应thetao shape={lat_native.shape}")

        area_valid = np.isfinite(area_lat_test) & np.isfinite(area_lon_test) & np.isfinite(lat_native) & np.isfinite(lon_native)
        area_lat_error = float(np.nanmax(np.abs(area_lat_test[area_valid] - lat_native[area_valid])))
        area_lon_error = float(np.nanmax(np.abs((area_lon_test[area_valid] - lon_native[area_valid] + 180.0) % 360.0 - 180.0)))

        if area_lat_error > 1.0e-5 or area_lon_error > 1.0e-5:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} areacello网格不匹配：Δlat={area_lat_error:.3e}, Δlon={area_lon_error:.3e}")

        # 原生网格区域mask
        nordic_mask = np.isfinite(area_native) & (area_native > 0.0) & np.isfinite(lat_native) & np.isfinite(lon_native) & (lat_native >= NORDIC_LAT_MIN) & (lat_native <= NORDIC_LAT_MAX) & (lon_native >= NORDIC_LON_MIN) & (lon_native <= NORDIC_LON_MAX)
        li_mask = np.isfinite(area_native) & (area_native > 0.0) & np.isfinite(lat_native) & np.isfinite(lon_native) & (lat_native >= LI_LAT_MIN) & (lat_native < LI_LAT_MAX) & (lon_native >= LI_LON_MIN) & (lon_native <= LI_LON_MAX)
        region_mask = nordic_mask | li_mask

        if int(nordic_mask.sum()) < 5:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} Nordic Seas原生格点过少")

        if int(li_mask.sum()) < 5:
            theta_ds.close()
            so_ds.close()
            raise ValueError(f"{model} {experiment} Labrador-Irminger Seas原生格点过少")

        # 先裁剪最小二维矩形，显著减少磁盘读取量
        row_index, column_index = np.where(region_mask)
        row_start, row_stop = int(row_index.min()), int(row_index.max()) + 1
        column_start, column_stop = int(column_index.min()), int(column_index.max()) + 1

        h0, h1 = horizontal_dims
        s0, s1 = so_horizontal_dims

        theta_box = theta_march.isel({h0: slice(row_start, row_stop), h1: slice(column_start, column_stop)})
        so_box = so_march.isel({s0: slice(row_start, row_stop), s1: slice(column_start, column_stop)})

        lat_box = lat_native[row_start:row_stop, column_start:column_stop]
        lon_box = lon_native[row_start:row_stop, column_start:column_stop]
        area_box = area_native[row_start:row_stop, column_start:column_stop]
        nordic_box = nordic_mask[row_start:row_stop, column_start:column_stop]
        li_box = li_mask[row_start:row_stop, column_start:column_stop]
        region_box = region_mask[row_start:row_stop, column_start:column_stop]

        cell_indices = np.flatnonzero(region_box.ravel())
        lat_cells = lat_box.ravel()[cell_indices]
        lon_cells = lon_box.ravel()[cell_indices]
        area_cells = area_box.ravel()[cell_indices]
        nordic_cells = nordic_box.ravel()[cell_indices]
        li_cells = li_box.ravel()[cell_indices]

        theta_region = theta_box.transpose("time", z_dim, h0, h1).stack(cell=(h0, h1)).isel(cell=cell_indices).astype("float32")
        so_region = so_box.transpose("time", so_z_dim, s0, s1).stack(cell=(s0, s1)).isel(cell=cell_indices).astype("float32")

        print(f"[区域裁剪] full={lat_native.shape} -> box={lat_box.shape} -> target ocean cells={cell_indices.size}")
        print(f"[区域格点] Nordic={int(nordic_cells.sum())} | Labrador-Irminger={int(li_cells.sum())}")

        # 垂向坐标准备
        theta_units = str(theta.attrs.get("units", "")).strip().lower()
        so_units = str(salinity.attrs.get("units", "")).strip().lower()
        z_standard_name = str(theta_ds[z_dim].attrs.get("standard_name", "")).strip().lower()
        z_units = str(theta_ds[z_dim].attrs.get("units", "")).strip().lower()

        vertical_method = ""
        geometric_depth_m = None
        geometric_order = None
        eta_region = None
        sigma_values = None
        bathymetry_cells = None
        depth_c = None
        nsigma = None
        zlev_values = None
        thk_ds = None
        thk_region = None

        if model == "NorESM2-LM":
            thk_files = sorted([path for path in NOR_THK_ROOT.rglob("*.nc") if path.name.lower().startswith("thkcello_") and f"_{experiment.lower()}_" in path.name.lower() and f"_{member.lower()}_" in path.name.lower()])

            if not thk_files:
                theta_ds.close()
                so_ds.close()
                raise FileNotFoundError(f"{model} {experiment} 未找到thkcello")

            thk_ds = xr.open_mfdataset(thk_files, combine="by_coords", decode_times=TIME_CODER, chunks={"time": 12}, data_vars="minimal", coords="minimal", compat="override", parallel=False)
            thk = thk_ds["thkcello"]

            if "latitude" in thk_ds.variables and "longitude" in thk_ds.variables:
                thk_lat_da = thk_ds["latitude"]
                thk_lon_da = thk_ds["longitude"]
            elif "lat" in thk_ds.variables and "lon" in thk_ds.variables:
                thk_lat_da = thk_ds["lat"]
                thk_lon_da = thk_ds["lon"]
            else:
                theta_ds.close()
                so_ds.close()
                thk_ds.close()
                raise KeyError(f"{model} {experiment} thkcello缺少经纬度")

            if thk_lat_da.ndim == 1 and thk_lon_da.ndim == 1:
                thk_lat_grid_da, thk_lon_grid_da = xr.broadcast(thk_lat_da, thk_lon_da)
                thk_horizontal_dims = list(thk_lat_grid_da.dims)
                thk_lat_native = np.asarray(thk_lat_grid_da.values, dtype=float)
                thk_lon_native = np.asarray(thk_lon_grid_da.values, dtype=float)
            elif thk_lat_da.ndim == 2 and thk_lon_da.ndim == 2:
                thk_horizontal_dims = list(thk_lat_da.dims)
                thk_lat_native = np.asarray(thk_lat_da.values, dtype=float)
                thk_lon_native = np.asarray(thk_lon_da.values, dtype=float)
            else:
                theta_ds.close()
                so_ds.close()
                thk_ds.close()
                raise ValueError(f"{model} {experiment} thkcello经纬度维数异常")

            thk_lon_native = (thk_lon_native + 180.0) % 360.0 - 180.0

            if thk_lat_native.shape != lat_native.shape:
                theta_ds.close()
                so_ds.close()
                thk_ds.close()
                raise ValueError(f"{model} {experiment} thkcello水平shape与thetao不同")

            thk_grid_valid = np.isfinite(thk_lat_native) & np.isfinite(thk_lon_native) & np.isfinite(lat_native) & np.isfinite(lon_native)
            thk_lat_error = float(np.nanmax(np.abs(thk_lat_native[thk_grid_valid] - lat_native[thk_grid_valid])))
            thk_lon_error = float(np.nanmax(np.abs((thk_lon_native[thk_grid_valid] - lon_native[thk_grid_valid] + 180.0) % 360.0 - 180.0)))

            if thk_lat_error > 1.0e-6 or thk_lon_error > 1.0e-6:
                theta_ds.close()
                so_ds.close()
                thk_ds.close()
                raise ValueError(f"{model} {experiment} thkcello与thetao网格不一致")

            thk_vertical_dims = [dim for dim in thk.dims if dim != "time" and dim not in thk_horizontal_dims]

            if len(thk_vertical_dims) != 1:
                theta_ds.close()
                so_ds.close()
                thk_ds.close()
                raise ValueError(f"{model} {experiment} 无法唯一确定thkcello垂向维")

            thk_z_dim = thk_vertical_dims[0]

            if thk.sizes[thk_z_dim] != theta.sizes[z_dim]:
                theta_ds.close()
                so_ds.close()
                thk_ds.close()
                raise ValueError(f"{model} {experiment} thkcello与thetao垂向层数不同")

            thk_march = thk.where(thk_ds["time"].dt.month == 3, drop=True)

            if thk_march.sizes["time"] < YEARS_PER_PHASE:
                theta_ds.close()
                so_ds.close()
                thk_ds.close()
                raise ValueError(f"{model} {experiment} thkcello March不足140年")

            thk_march = thk_march.isel(time=slice(0, YEARS_PER_PHASE))
            t0, t1 = thk_horizontal_dims
            thk_box = thk_march.isel({t0: slice(row_start, row_stop), t1: slice(column_start, column_stop)})
            thk_region = thk_box.transpose("time", thk_z_dim, t0, t1).stack(cell=(t0, t1)).isel(cell=cell_indices).astype("float32")
            vertical_method = "thkcello_layer_center"

        elif z_standard_name == "ocean_sigma_z":
            formula_terms = str(theta_ds[z_dim].attrs.get("formula_terms", ""))
            formula_pairs = dict(re.findall(r"([A-Za-z0-9_]+):\s*([A-Za-z0-9_]+)", formula_terms))
            required_terms = {"sigma", "eta", "depth", "depth_c", "nsigma", "zlev"}

            if not required_terms.issubset(formula_pairs):
                theta_ds.close()
                so_ds.close()
                raise ValueError(f"{model} {experiment} ocean_sigma_z formula_terms不完整：{formula_terms}")

            sigma_name = formula_pairs["sigma"]
            eta_name = formula_pairs["eta"]
            depth_name = formula_pairs["depth"]
            depth_c_name = formula_pairs["depth_c"]
            nsigma_name = formula_pairs["nsigma"]
            zlev_name = formula_pairs["zlev"]

            for variable_name in [sigma_name, eta_name, depth_name, depth_c_name, nsigma_name, zlev_name]:
                if variable_name not in theta_ds.variables:
                    theta_ds.close()
                    so_ds.close()
                    raise KeyError(f"{model} {experiment} ocean_sigma_z缺少{variable_name}")

            sigma_values = np.asarray(theta_ds[sigma_name].values, dtype=float).reshape(-1)
            zlev_values = np.asarray(theta_ds[zlev_name].values, dtype=float).reshape(-1)
            depth_c = float(np.asarray(theta_ds[depth_c_name].values).squeeze())
            nsigma = int(np.asarray(theta_ds[nsigma_name].values).squeeze())

            eta_da = theta_ds[eta_name]
            depth_da = theta_ds[depth_name]
            eta_march = eta_da.where(theta_ds["time"].dt.month == 3, drop=True).isel(time=slice(0, YEARS_PER_PHASE))
            eta_box = eta_march.isel({h0: slice(row_start, row_stop), h1: slice(column_start, column_stop)})
            eta_region = eta_box.transpose("time", h0, h1).stack(cell=(h0, h1)).isel(cell=cell_indices).astype("float32")

            depth_box = depth_da.isel({h0: slice(row_start, row_stop), h1: slice(column_start, column_stop)})
            bathymetry_cells = np.asarray(depth_box.transpose(h0, h1).stack(cell=(h0, h1)).isel(cell=cell_indices).values, dtype=float)

            if sigma_values.size != theta.sizes[z_dim] or zlev_values.size != theta.sizes[z_dim]:
                theta_ds.close()
                so_ds.close()
                raise ValueError(f"{model} {experiment} ocean_sigma_z垂向数组长度错误")

            vertical_method = "CF_ocean_sigma_z"

        else:
            z_values = np.asarray(theta_ds[z_dim].values, dtype=float).reshape(-1)

            if z_units in ["m", "meter", "meters", "metre", "metres"]:
                geometric_depth_m = np.abs(z_values)
            elif z_units in ["cm", "centimeter", "centimeters", "centimetre", "centimetres"]:
                geometric_depth_m = np.abs(z_values) / 100.0
            else:
                theta_ds.close()
                so_ds.close()
                raise ValueError(f"{model} {experiment} 垂向单位不明确：{z_units}")

            geometric_order = np.argsort(geometric_depth_m)
            geometric_depth_m = geometric_depth_m[geometric_order]
            vertical_method = "geometric_depth"

        print(f"[垂向] {vertical_method}")
        print(f"[计算] {YEARS_PER_PHASE}个March，{YEAR_BATCH}年/批")

        stage_no_cross_nordic = []
        stage_no_cross_li = []

        # 批量读取，每10年一次从磁盘加载
        for batch_start in range(0, YEARS_PER_PHASE, YEAR_BATCH):
            batch_stop = min(batch_start + YEAR_BATCH, YEARS_PER_PHASE)

            print(f"  [读取] experiment years {batch_start + 1}-{batch_stop} / {YEARS_PER_PHASE}")

            theta_batch = theta_region.isel(time=slice(batch_start, batch_stop)).load()
            so_batch = so_region.isel(time=slice(batch_start, batch_stop)).load()

            theta_batch_values = np.asarray(theta_batch.values, dtype=float)
            so_batch_values = np.asarray(so_batch.values, dtype=float)

            if theta_units in ["k", "kelvin"]:
                theta_batch_values -= 273.15

            finite_so = so_batch_values[np.isfinite(so_batch_values)]

            if finite_so.size == 0 or float(np.nanmedian(finite_so)) < 1.0 or float(np.nanmedian(finite_so)) > 50.0:
                theta_ds.close()
                so_ds.close()
                if thk_ds is not None:
                    thk_ds.close()
                raise ValueError(f"{model} {experiment} so数值不符合Practical Salinity量级，units={so_units}")

            if model == "NorESM2-LM":
                thk_batch_values = np.asarray(thk_region.isel(time=slice(batch_start, batch_stop)).load().values, dtype=float)

            elif vertical_method == "CF_ocean_sigma_z":
                eta_batch_values = np.asarray(eta_region.isel(time=slice(batch_start, batch_stop)).load().values, dtype=float)

            for local_index in range(batch_stop - batch_start):
                year_index = batch_start + local_index
                theta_values = theta_batch_values[local_index]
                so_values = so_batch_values[local_index]

                if model == "NorESM2-LM":
                    thickness_values = thk_batch_values[local_index]
                    depth_values = np.cumsum(thickness_values, axis=0) - 0.5 * thickness_values
                    sort_key = np.where(np.isfinite(depth_values), depth_values, np.inf)
                    depth_order = np.argsort(sort_key, axis=0)
                    depth_values = np.take_along_axis(depth_values, depth_order, axis=0)
                    theta_values = np.take_along_axis(theta_values, depth_order, axis=0)
                    so_values = np.take_along_axis(so_values, depth_order, axis=0)

                elif vertical_method == "CF_ocean_sigma_z":
                    eta_values = eta_batch_values[local_index]
                    sigma_depth_limit = np.minimum(depth_c, bathymetry_cells)
                    z_values = np.full(theta_values.shape, np.nan, dtype=float)
                    z_values[:nsigma] = eta_values[None, :] + sigma_values[:nsigma, None] * (sigma_depth_limit[None, :] + eta_values[None, :])
                    z_values[nsigma:] = zlev_values[nsigma:, None]
                    depth_values = np.abs(z_values)
                    sort_key = np.where(np.isfinite(depth_values), depth_values, np.inf)
                    depth_order = np.argsort(sort_key, axis=0)
                    depth_values = np.take_along_axis(depth_values, depth_order, axis=0)
                    theta_values = np.take_along_axis(theta_values, depth_order, axis=0)
                    so_values = np.take_along_axis(so_values, depth_order, axis=0)

                else:
                    theta_values = theta_values[geometric_order]
                    so_values = so_values[geometric_order]
                    depth_values = np.broadcast_to(geometric_depth_m[:, None], theta_values.shape)

                # TEOS-10
                pressure = gsw.p_from_z(-depth_values, lat_cells[None, :])
                absolute_salinity = gsw.SA_from_SP(so_values, pressure, lon_cells[None, :], lat_cells[None, :])
                conservative_temperature = gsw.CT_from_pt(absolute_salinity, theta_values)
                sigma0 = gsw.sigma0(absolute_salinity, conservative_temperature)

                # 10 m参考密度
                profile_valid = np.isfinite(depth_values) & np.isfinite(sigma0)
                reference_distance = np.where(profile_valid, np.abs(depth_values - REFERENCE_DEPTH_M), np.inf)
                reference_index = np.argmin(reference_distance, axis=0)
                cell_number = np.arange(reference_index.size)

                reference_distance_min = reference_distance[reference_index, cell_number]
                reference_valid = np.isfinite(reference_distance_min)
                reference_sigma = sigma0[reference_index, cell_number]
                reference_depth = depth_values[reference_index, cell_number]
                target_sigma = reference_sigma + DENSITY_THRESHOLD_KG_M3

                # 向下寻找首次达到密度阈值的深度
                crossing = profile_valid & reference_valid[None, :] & (depth_values >= reference_depth[None, :]) & (sigma0 >= target_sigma[None, :])
                has_crossing = np.any(crossing, axis=0)
                crossing_index = np.argmax(crossing, axis=0)

                mld = np.full(reference_index.size, np.nan, dtype=float)
                crossing_cells = np.flatnonzero(has_crossing)

                if crossing_cells.size > 0:
                    k2 = crossing_index[crossing_cells]
                    k1 = np.maximum(k2 - 1, 0)
                    d1 = depth_values[k1, crossing_cells]
                    d2 = depth_values[k2, crossing_cells]
                    s1 = sigma0[k1, crossing_cells]
                    s2 = sigma0[k2, crossing_cells]
                    target = target_sigma[crossing_cells]
                    denominator = s2 - s1

                    interpolation_valid = (k2 > 0) & np.isfinite(d1) & np.isfinite(d2) & np.isfinite(s1) & np.isfinite(s2) & np.isfinite(denominator) & (np.abs(denominator) > 1.0e-12)

                    mld_cross = d2.copy()
                    fraction = np.zeros(crossing_cells.size, dtype=float)
                    fraction[interpolation_valid] = (target[interpolation_valid] - s1[interpolation_valid]) / denominator[interpolation_valid]
                    fraction = np.clip(fraction, 0.0, 1.0)
                    mld_cross[interpolation_valid] = d1[interpolation_valid] + fraction[interpolation_valid] * (d2[interpolation_valid] - d1[interpolation_valid])
                    mld[crossing_cells] = mld_cross

                # 区域面积加权平均
                nordic_valid = nordic_cells & np.isfinite(mld) & np.isfinite(area_cells) & (area_cells > 0.0)
                li_valid = li_cells & np.isfinite(mld) & np.isfinite(area_cells) & (area_cells > 0.0)

                if int(nordic_valid.sum()) < 3:
                    theta_ds.close()
                    so_ds.close()
                    if thk_ds is not None:
                        thk_ds.close()
                    raise ValueError(f"{model} {experiment} year={year_index + 1} Nordic Seas有效MLD格点少于3")

                if int(li_valid.sum()) < 3:
                    theta_ds.close()
                    so_ds.close()
                    if thk_ds is not None:
                        thk_ds.close()
                    raise ValueError(f"{model} {experiment} year={year_index + 1} Labrador-Irminger有效MLD格点少于3")

                nordic_mean = float(np.sum(mld[nordic_valid] * area_cells[nordic_valid]) / np.sum(area_cells[nordic_valid]))
                li_mean = float(np.sum(mld[li_valid] * area_cells[li_valid]) / np.sum(area_cells[li_valid]))

                nordic_reference_count = int(np.sum(nordic_cells & reference_valid))
                li_reference_count = int(np.sum(li_cells & reference_valid))

                nordic_no_cross = float(np.sum(nordic_cells & reference_valid & ~has_crossing) / nordic_reference_count)
                li_no_cross = float(np.sum(li_cells & reference_valid & ~has_crossing) / li_reference_count)

                stage_no_cross_nordic.append(nordic_no_cross)
                stage_no_cross_li.append(li_no_cross)

                simulation_year = year_index + 1 if stage == "co2" else YEARS_PER_PHASE + year_index + 1

                series_rows.append({
                    "model": model,
                    "simulation_year": simulation_year,
                    "phase": "ramp-up" if stage == "co2" else "ramp-down",
                    "phase_year": year_index + 1,
                    "co2_ppm": float(co2_ppm_all[simulation_year - 1]),
                    "nordic_march_mld_m": nordic_mean,
                    "li_march_mld_m": li_mean,
                    "nordic_valid_cells": int(nordic_valid.sum()),
                    "li_valid_cells": int(li_valid.sum()),
                    "nordic_no_cross_fraction": nordic_no_cross,
                    "li_no_cross_fraction": li_no_cross,
                })

            del theta_batch, so_batch, theta_batch_values, so_batch_values

            if model == "NorESM2-LM":
                del thk_batch_values

            if vertical_method == "CF_ocean_sigma_z":
                del eta_batch_values

            print(f"  [完成] experiment years {batch_start + 1}-{batch_stop} / {YEARS_PER_PHASE}")

        # 每阶段QA
        qa_rows.append({
            "model": model,
            "experiment": experiment,
            "member": member,
            "thetao_files": len(theta_files),
            "so_files": len(so_files),
            "thetao_first_march_year": int(theta_years[0]),
            "thetao_last_march_year": int(theta_years[-1]),
            "so_first_march_year": int(so_years[0]),
            "so_last_march_year": int(so_years[-1]),
            "absolute_calendar_same": bool(np.array_equal(theta_years, so_years)),
            "experiment_index_alignment": "first_140_March_by_each_variable",
            "vertical_method": vertical_method,
            "thetao_units": theta_units,
            "so_units": so_units,
            "region_box_rows": row_stop - row_start,
            "region_box_columns": column_stop - column_start,
            "nordic_native_cells": int(nordic_cells.sum()),
            "li_native_cells": int(li_cells.sum()),
            "nordic_mean_no_cross_fraction": float(np.mean(stage_no_cross_nordic)),
            "li_mean_no_cross_fraction": float(np.mean(stage_no_cross_li)),
            "thetao_so_lat_max_error_deg": lat_grid_error,
            "thetao_so_lon_max_error_deg": lon_grid_error,
            "areacello_lat_max_error_deg": area_lat_error,
            "areacello_lon_max_error_deg": area_lon_error,
        })

        print(f"[{experiment}] theta years={theta_years[0]}~{theta_years[-1]} | so years={so_years[0]}~{so_years[-1]}")
        print(f"[{experiment}] absolute calendar same={np.array_equal(theta_years, so_years)} | experiment-index=OK")
        print(f"[{experiment}] Nordic no-cross={np.mean(stage_no_cross_nordic):.3f} | LI no-cross={np.mean(stage_no_cross_li):.3f}")

        theta_ds.close()
        so_ds.close()

        if thk_ds is not None:
            thk_ds.close()

# %% 6. 计算异常与11年中心平滑
series = pd.DataFrame(series_rows).sort_values(["model", "simulation_year"]).reset_index(drop=True)

series["nordic_mld_anomaly_m"] = np.nan
series["li_mld_anomaly_m"] = np.nan
series["nordic_mld_anomaly_11yr_m"] = np.nan
series["li_mld_anomaly_11yr_m"] = np.nan

summary_rows = []

for model in MODELS:
    model_index = series.index[series["model"] == model]
    model_data = series.loc[model_index].sort_values("simulation_year")

    if model_data.shape[0] != 280:
        raise ValueError(f"{model} 应为280年，实际={model_data.shape[0]}")

    nordic_absolute = model_data["nordic_march_mld_m"].to_numpy(dtype=float)
    li_absolute = model_data["li_march_mld_m"].to_numpy(dtype=float)

    nordic_baseline = float(np.mean(nordic_absolute[:BASELINE_YEARS]))
    li_baseline = float(np.mean(li_absolute[:BASELINE_YEARS]))

    nordic_anomaly = nordic_absolute - nordic_baseline
    li_anomaly = li_absolute - li_baseline

    nordic_smooth = pd.Series(nordic_anomaly).rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW).mean().to_numpy()
    li_smooth = pd.Series(li_anomaly).rolling(SMOOTH_WINDOW, center=True, min_periods=SMOOTH_WINDOW).mean().to_numpy()

    sorted_index = model_data.index
    series.loc[sorted_index, "nordic_mld_anomaly_m"] = nordic_anomaly
    series.loc[sorted_index, "li_mld_anomaly_m"] = li_anomaly
    series.loc[sorted_index, "nordic_mld_anomaly_11yr_m"] = nordic_smooth
    series.loc[sorted_index, "li_mld_anomaly_11yr_m"] = li_smooth

    summary_rows.append({
        "model": model,
        "nordic_baseline_years1_10_m": nordic_baseline,
        "li_baseline_years1_10_m": li_baseline,
        "nordic_rampup_final10_anomaly_m": float(np.mean(nordic_anomaly[130:140])),
        "li_rampup_final10_anomaly_m": float(np.mean(li_anomaly[130:140])),
        "nordic_rampdown_final10_anomaly_m": float(np.mean(nordic_anomaly[270:280])),
        "li_rampdown_final10_anomaly_m": float(np.mean(li_anomaly[270:280])),
        "nordic_min_anomaly_m": float(np.min(nordic_anomaly)),
        "li_min_anomaly_m": float(np.min(li_anomaly)),
        "nordic_max_anomaly_m": float(np.max(nordic_anomaly)),
        "li_max_anomaly_m": float(np.max(li_anomaly)),
    })

summary = pd.DataFrame(summary_rows)
qa = pd.DataFrame(qa_rows)

# %% 7. 计算统一绘图范围
smooth_values = series[["nordic_mld_anomaly_11yr_m", "li_mld_anomaly_11yr_m"]].to_numpy(dtype=float)
finite_smooth = smooth_values[np.isfinite(smooth_values)]

if finite_smooth.size == 0:
    raise RuntimeError("没有有效11年平滑MLD数据")

mld_limit = float(np.ceil(np.nanpercentile(np.abs(finite_smooth), 99.0) / 100.0) * 100.0)
mld_limit = max(mld_limit, 200.0)
mld_ticks = np.linspace(-mld_limit, mld_limit, 5).tolist()

# %% 8. 保存plot-ready结果
series.to_csv(SERIES_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")
summary.to_csv(SUMMARY_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")
qa.to_csv(QA_FILE, index=False, encoding="utf-8-sig", float_format="%.8g")

metadata = {
    "models": MODELS,
    "excluded_model": "CAS-ESM2-0",
    "diagnostic": "March mixed-layer depth in Nordic Seas and Labrador-Irminger Seas",
    "mld_source": "thetao + so, TEOS-10 sigma0",
    "reference_depth_m": REFERENCE_DEPTH_M,
    "density_threshold_kg_m3": DENSITY_THRESHOLD_KG_M3,
    "month": "March",
    "years_per_phase": YEARS_PER_PHASE,
    "ramp_up_simulation_years": [1, 140],
    "ramp_down_simulation_years": [141, 280],
    "thetao_so_alignment": "experiment index, not absolute calendar year",
    "baseline": "each model-region mean March MLD over 1pctCO2 simulation years 1-10",
    "smooth": "11-year centered running mean; no edge filling",
    "piControl_adjustment": "none",
    "interpolation": "none for regional averaging; vertical density-threshold crossing linearly interpolated between adjacent model levels",
    "regional_weighting": "native-grid areacello",
    "io_strategy": f"native-grid regional crop followed by {YEAR_BATCH}-year batch loading",
    "nordic_seas_box": {"lat_min": NORDIC_LAT_MIN, "lat_max": NORDIC_LAT_MAX, "lon_min": NORDIC_LON_MIN, "lon_max": NORDIC_LON_MAX},
    "labrador_irminger_box": {"lat_min": LI_LAT_MIN, "lat_max": LI_LAT_MAX, "lon_min": LI_LON_MIN, "lon_max": LI_LON_MAX},
    "mld_anomaly_ylim_m": [-mld_limit, mld_limit],
    "mld_anomaly_yticks_m": mld_ticks,
    "co2_ylim_ppm": [250.0, 1200.0],
    "co2_ticks_ppm": [285.0, 700.0, 1147.0],
}

with open(META_FILE, "w", encoding="utf-8") as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)

# %% 9. 完成
print("\n" + "=" * 120)
print("8-MODEL NORDIC / LABRADOR-IRMINGER MARCH MLD TIME SERIES COMPLETE")
print("=" * 120)
print(summary.round(2).to_string(index=False))
print(f"\n[series]   {SERIES_FILE}")
print(f"[summary]  {SUMMARY_FILE}")
print(f"[QA]       {QA_FILE}")
print(f"[metadata] {META_FILE}")
print(f"[y-limit]  ±{mld_limit:.0f} m")
print("=" * 120)