# -*- coding: utf-8 -*-
"""
Recalculate North Atlantic raw SIA and combine with existing plot-ready 40°N AMOC.

【是否需要运行】
需要。新NA区域、原始SIC、areacello、成员或CAS老师整理文件变化后重新运行。

【输入文件】
F:/sea/cmipseaicead/<model>/**/*.nc
F:/sea/cmipseaice/<model>/**/*.nc
F:/sea/areacello/**/*.nc

CAS CDR专用：
F:/sea/cmipseaice/CAS/siconc.nc

现有AMOC/CO2 plot-ready结果：
output/n03-siaandamoc_n01/amoc_sia_co2_plot_data.csv

可选pattern QA：
output/s02sia/s02_arctic_region/p04_pattern_unadjusted/n06_sia_only/arctic_sia_unadjusted_cells.csv
或：
output/s02sia/s02_arctic_region/p04_pattern_unadjusted/n05/unadjusted_three_pattern_cells.csv

【输出文件】
output/n03-siaandamoc_n01_newNA_raw/north_atlantic_sia_raw_annual_timeseries.csv
output/n03-siaandamoc_n01_newNA_raw/north_atlantic_sia_processing_qa.csv
output/n03-siaandamoc_n01_newNA_raw/north_atlantic_sia_failures.csv
output/n03-siaandamoc_n01_newNA_raw/north_atlantic_sia_pattern_comparison.csv
output/n03-siaandamoc_n01_newNA_raw/amoc_sia_co2_plot_data.csv
output/n03-siaandamoc_n01_newNA_raw/amoc_sia_co2_metadata.json

【什么时候需要重新运行】
原始SIC、areacello、新NA定义、端点窗口、CAS老师整理文件或模式成员变化后。

科学口径
1. NA = 45–70°N, 60°W–30°E；45 <= lat < 70，-60 <= lon < 30。
2. SIA = Σ(SIC fraction × areacello)，在原生海洋网格积分，不插值SIC。
3. SIA不使用piControl，不做漂移订正。
4. SIA anomaly基准 = 各模式raw 1pctCO2 initial 10 yr平均。
5. 普通模式使用前140个完整连续实验年。
6. CAS CDR固定文件F:/sea/cmipseaice/CAS/siconc.nc：
   aice，%，196×360，1920个月=160年；
   前1680个月=CDR实验年1–140；
   time[1560:1680]=CDR实验年131–140；
   time[1680:1920]后20年不进入本研究140年ramp-down。
7. CAS CDR按连续12个记录组成一个实验年，不使用解码后的340–500 calendar year分组。
8. 月SIA为12个月等权年平均。
9. 11年中心滑动平均只用于画图；endpoint QA使用未平滑年度SIA。
10. areacello先在0–90°N finite-SIC海洋网格完成球面最近邻匹配，再截取NA。
11. pattern只在SIA独立计算完成后做QA，不参与SIA计算。
12. AMOC和CO2直接复制既有plot-ready结果，不重新计算。
"""

# %% 1. 导入依赖
from datetime import datetime
from pathlib import Path
import json
import os
import re
import shutil
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore", message=".*multiple fill values.*")

# %% 2. Python环境、路径与参数
VIPER_PYTHON = Path(r"D:\Download\Anaconda\envs\viper\python.exe")

if os.name == "nt" and VIPER_PYTHON.exists():
    current_python = os.path.normcase(os.path.abspath(sys.executable))
    target_python = os.path.normcase(os.path.abspath(str(VIPER_PYTHON)))
    if current_python != target_python:
        result = subprocess.run([str(VIPER_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=str(Path(__file__).resolve().parent), check=False)
        raise SystemExit(result.returncode)

ROOT = Path(r"F:\sea\Antic sea ice constraint")
SEA = Path(r"F:\sea")

SIC_ROOTS = {"up": SEA / "cmipseaicead", "cdr": SEA / "cmipseaice"}
AREACELLO_ROOT = SEA / "areacello"
CAS_CDR_FILE = SEA / "cmipseaice" / "CAS" / "siconc.nc"

OLD_PLOT_DIR = ROOT / "output" / "n03-siaandamoc_n01"
OLD_PLOT_DATA_FILE = OLD_PLOT_DIR / "amoc_sia_co2_plot_data.csv"

PATTERN_CELL_CANDIDATES = [
    ROOT / "output" / "s02sia" / "s02_arctic_region" / "p04_pattern_unadjusted" / "n06_sia_only" / "arctic_sia_unadjusted_cells.csv",
    ROOT / "output" / "s02sia" / "s02_arctic_region" / "p04_pattern_unadjusted" / "n05" / "unadjusted_three_pattern_cells.csv",
]

OUT = ROOT / "output" / "n03-siaandamoc_n01_newNA_raw"
SIA_ANNUAL_FILE = OUT / "north_atlantic_sia_raw_annual_timeseries.csv"
SIA_QA_FILE = OUT / "north_atlantic_sia_processing_qa.csv"
SIA_FAIL_FILE = OUT / "north_atlantic_sia_failures.csv"
COMPARISON_FILE = OUT / "north_atlantic_sia_pattern_comparison.csv"
PLOT_DATA_FILE = OUT / "amoc_sia_co2_plot_data.csv"
METADATA_FILE = OUT / "amoc_sia_co2_metadata.json"

MODELS = [
    "ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4",
    "MIROC-ES2L", "UKESM1-0-LL", "CAS-ESM2-0", "NorESM2-LM",
]

FOLDER_CANDIDATES = {
    "ACCESS-ESM1-5": ("ACCESSESM5", "ACCESS", "ACCESS-ESM1-5"),
    "CanESM5": ("CanESM5", "CANESM5"),
    "CESM2": ("CESM2",),
    "CNRM-ESM2-1": ("CNRM-ESM2", "CNRM-ESM2-1"),
    "GFDL-ESM4": ("GFDL", "GFDL-ESM4"),
    "MIROC-ES2L": ("MIROC-ES2L",),
    "UKESM1-0-LL": ("UKESM1-0-LL",),
    "CAS-ESM2-0": ("CAS", "CAS-ESM2-0", "CAS-ESM2"),
    "NorESM2-LM": ("NORESM2-LM2", "NORESM2-LM", "NorESM2-LM"),
}

MEMBERS = {
    "ACCESS-ESM1-5": {"up": "r1i1p1f1", "cdr": "r1i1p1f1"},
    "CanESM5": {"up": "r1i1p2f1", "cdr": "r1i1p2f1"},
    "CESM2": {"up": "r1i1p1f1", "cdr": "r1i1p1f1"},
    "CNRM-ESM2-1": {"up": "r1i1p1f2", "cdr": "r1i1p1f2"},
    "GFDL-ESM4": {"up": "r1i1p1f1", "cdr": "r1i1p1f1"},
    "MIROC-ES2L": {"up": "r1i1p1f2", "cdr": "r1i1p1f2"},
    "UKESM1-0-LL": {"up": "r1i1p1f2", "cdr": "r1i1p1f2"},
    "CAS-ESM2-0": {"up": "r1i1p1f1", "cdr": None},
    "NorESM2-LM": {"up": "r1i1p1f1", "cdr": "r1i1p1f1"},
}

SIC_NAMES = ("siconc", "siconca", "sic", "aice")
LAT_NAMES = {"lat", "latitude", "nav_lat", "geolat", "tlat", "ulat"}
LON_NAMES = {"lon", "longitude", "nav_lon", "geolon", "tlon", "ulon"}

NA_LAT_MIN, NA_LAT_MAX = 45.0, 70.0
NA_LON_MIN, NA_LON_MAX = -60.0, 30.0
MATCH_LAT_MIN, MATCH_LAT_MAX = 0.0, 90.0

RAMP_YEARS = 140
BASELINE_YEARS = 10
SMOOTH_YEARS = 11

CAS_CDR_TOTAL_MONTHS = 1920
CAS_CDR_TOTAL_YEARS = 160
CAS_CDR_USED_MONTHS = RAMP_YEARS * 12
CAS_CDR_FINAL10_START = (RAMP_YEARS - BASELINE_YEARS) * 12
CAS_CDR_FINAL10_END = RAMP_YEARS * 12

EARTH_RADIUS_KM = 6371.0
MAX_AREA_MATCH_DISTANCE_KM = 300.0
MAX_AREA_MATCH_P95_DISTANCE_KM = 180.0
MIN_AREA_MATCH_COVERAGE = 0.95
MAX_DUPLICATE_AREA_SOURCE_FRACTION = 0.03
MAX_STAGE_AREA_RELATIVE_DIFFERENCE = 0.02
MIN_SIA_TIME_COVERAGE = 0.95

TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

for path in [*SIC_ROOTS.values(), AREACELLO_ROOT]:
    if not path.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{path}")

if not CAS_CDR_FILE.is_file():
    raise FileNotFoundError(f"CAS老师整理CDR文件不存在：{CAS_CDR_FILE}")

if not OLD_PLOT_DATA_FILE.is_file():
    raise FileNotFoundError(f"缺少现有AMOC/CO2 plot-ready文件：{OLD_PLOT_DATA_FILE}")

OUT.mkdir(parents=True, exist_ok=True)

formal_outputs = [SIA_ANNUAL_FILE, SIA_QA_FILE, SIA_FAIL_FILE, COMPARISON_FILE, PLOT_DATA_FILE, METADATA_FILE]
existing_outputs = [path for path in formal_outputs if path.exists()]

if existing_outputs:
    archive_dir = OUT / "archive" / datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir.mkdir(parents=True, exist_ok=True)
    for path in existing_outputs:
        shutil.move(str(path), str(archive_dir / path.name))
    print(f"[归档旧结果] {archive_dir}")

all_area_files = [path for path in sorted(AREACELLO_ROOT.rglob("*.nc")) if "areacello" in path.name.lower()]

# %% 3. 独立重算新NA SIA
sia_annual_rows = []
sia_qa_rows = []
sia_failure_rows = []

sia_anomaly_series = {}
sia_smooth_series = {}
sia_endpoint_delta = {}

for model in MODELS:
    print(f"\n{'=' * 108}")
    print(f"{model} | raw NA SIA: 45–70°N, 60°W–30°E")
    print("=" * 108)

    try:
        stage_series = {}
        stage_years = {}
        stage_qa = {}

        for stage in ("up", "cdr"):
            cas_teacher_cdr = model == "CAS-ESM2-0" and stage == "cdr"
            stage_root = SIC_ROOTS[stage]

            folder_map = {}

            for name in FOLDER_CANDIDATES[model]:
                path = stage_root / name
                if path.is_dir():
                    key = os.path.normcase(os.path.realpath(str(path)))
                    folder_map.setdefault(key, path)

            folders = list(folder_map.values())

            if len(folders) == 0:
                raise FileNotFoundError(f"{model} {stage} 未找到模式目录：{FOLDER_CANDIDATES[model]}")

            if len(folders) > 1:
                raise RuntimeError(f"{model} {stage} 命中多个真实不同目录：{folders}")

            folder = folders[0]
            member = MEMBERS[model][stage]

            if cas_teacher_cdr:
                sic_files = [CAS_CDR_FILE]
                sic_name = "aice"
                print(f"[CAS CDR专用文件] {CAS_CDR_FILE}")
            else:
                nc_files = sorted(folder.rglob("*.nc"))
                sic_files = [
                    path for path in nc_files
                    if "siconc" in path.name.lower()
                    and member.lower() in path.as_posix().lower()
                ]

                if not sic_files:
                    raise FileNotFoundError(f"{model} {stage} 未找到成员 {member} 的siconc文件")

                sic_name = "siconc"

            ds = xr.open_mfdataset(
                sic_files, combine="by_coords", decode_times=TIME_CODER, parallel=False,
                data_vars="minimal", coords="minimal", compat="override",
                combine_attrs="override", mask_and_scale=True,
            )

            lower_map = {str(name).lower(): str(name) for name in ds.data_vars}

            if sic_name not in ds.data_vars and sic_name.lower() in lower_map:
                sic_name = lower_map[sic_name.lower()]

            if sic_name not in ds.data_vars:
                candidates = []

                for name, variable in ds.data_vars.items():
                    standard_name = str(variable.attrs.get("standard_name", "")).lower()
                    long_name = str(variable.attrs.get("long_name", "")).lower()

                    if str(name).lower() in SIC_NAMES:
                        candidates.append(str(name))
                    elif standard_name == "sea_ice_area_fraction":
                        candidates.append(str(name))
                    elif "sea ice" in long_name and ("concentration" in long_name or "ice area" in long_name):
                        candidates.append(str(name))

                candidates = sorted(set(candidates))

                if len(candidates) != 1:
                    ds.close()
                    raise KeyError(f"{model} {stage} 无法唯一确认SIC变量：{candidates}")

                sic_name = candidates[0]

            sic = ds[sic_name].squeeze(drop=True)

            if "time" not in sic.dims:
                ds.close()
                raise ValueError(f"{model} {stage} SIC缺少time维")

            spatial_dims = tuple(dim for dim in sic.dims if dim != "time")

            if len(spatial_dims) != 2:
                ds.close()
                raise ValueError(f"{model} {stage} SIC应为time+二维水平网格，实际={sic.dims}")

            lat_names = [
                name for name in ds.variables
                if str(ds[name].attrs.get("standard_name", "")).lower() == "latitude"
                and "time" not in ds[name].dims
                and set(ds[name].dims).issubset(set(spatial_dims))
            ]

            lon_names = [
                name for name in ds.variables
                if str(ds[name].attrs.get("standard_name", "")).lower() == "longitude"
                and "time" not in ds[name].dims
                and set(ds[name].dims).issubset(set(spatial_dims))
            ]

            if len(lat_names) != 1:
                lat_names = [
                    name for name in ds.variables
                    if str(name).lower() in LAT_NAMES
                    and "time" not in ds[name].dims
                    and set(ds[name].dims).issubset(set(spatial_dims))
                ]

            if len(lon_names) != 1:
                lon_names = [
                    name for name in ds.variables
                    if str(name).lower() in LON_NAMES
                    and "time" not in ds[name].dims
                    and set(ds[name].dims).issubset(set(spatial_dims))
                ]

            if len(lat_names) != 1 or len(lon_names) != 1:
                ds.close()
                raise ValueError(f"{model} {stage} 无法唯一确认经纬度：lat={lat_names}, lon={lon_names}")

            sic_lat, sic_lon = xr.broadcast(ds[lat_names[0]], ds[lon_names[0]])
            sic_lat = sic_lat.transpose(*spatial_dims)
            sic_lon = sic_lon.transpose(*spatial_dims)

            lat = np.asarray(sic_lat.values, dtype=float)
            lon_raw = np.asarray(sic_lon.values, dtype=float)

            valid_coord = (
                np.isfinite(lat)
                & np.isfinite(lon_raw)
                & (lat >= -90.0)
                & (lat <= 90.0)
                & (lon_raw >= -360.0)
                & (lon_raw <= 720.0)
            )

            lon = np.full(lon_raw.shape, np.nan, dtype=float)
            lon[valid_coord] = (lon_raw[valid_coord] + 180.0) % 360.0 - 180.0

            units = str(sic.attrs.get("units", "")).strip().lower().replace(" ", "")

            if units in {"%", "percent", "percentage"}:
                sic_fraction = sic.astype("float64") / 100.0
                unit_mode = "percent_to_fraction"
            elif units in {"1", "fraction", "0-1", "0–1", "unitless", "dimensionless"}:
                sic_fraction = sic.astype("float64")
                unit_mode = "explicit_fraction_metadata"
            else:
                ds.close()
                raise ValueError(f"{model} {stage} SIC units={sic.attrs.get('units', '')!r}；禁止按幅度猜单位")

            sic_fraction = sic_fraction.where((sic_fraction >= 0.0) & (sic_fraction <= 1.0))

            # %% 3a. 时间轴
            if cas_teacher_cdr:
                time_count = int(sic.sizes["time"])

                if sic_name.lower() != "aice":
                    ds.close()
                    raise ValueError(f"CAS老师整理CDR预期变量aice，实际={sic_name}")

                if units not in {"%", "percent", "percentage"}:
                    ds.close()
                    raise ValueError(f"CAS老师整理CDR预期单位%，实际={sic.attrs.get('units', '')!r}")

                if time_count != CAS_CDR_TOTAL_MONTHS:
                    ds.close()
                    raise ValueError(
                        f"CAS老师整理CDR预期{CAS_CDR_TOTAL_MONTHS}个月，实际={time_count}；"
                        "文件结构发生变化，停止而不猜测"
                    )

                selected_index = np.arange(0, CAS_CDR_USED_MONTHS, dtype=int)
                reference_index = np.arange(CAS_CDR_FINAL10_START, CAS_CDR_FINAL10_END, dtype=int)
                target_years = np.arange(1, RAMP_YEARS + 1, dtype=int)
                steps_per_year = 12
                annualization = "teacher_processed_12_record_blocks"

                print("[CAS CDR时间结构]")
                print(f"  raw steps       = {time_count}")
                print(f"  frequency       = monthly")
                print(f"  total years     = {CAS_CDR_TOTAL_YEARS}")
                print(f"  used months     = 0:{CAS_CDR_USED_MONTHS}")
                print(f"  used CDR years  = 1–{RAMP_YEARS}")
                print(f"  final10 months  = {CAS_CDR_FINAL10_START}:{CAS_CDR_FINAL10_END}")
                print(f"  excluded months = {CAS_CDR_USED_MONTHS}:{time_count}")
                print(f"  excluded years  = {(time_count - CAS_CDR_USED_MONTHS) // 12}")
                print("  calendar grouping = disabled")

            else:
                years = np.asarray(sic.time.dt.year.values, dtype=int)
                months = np.asarray(sic.time.dt.month.values, dtype=int)
                unique_years = np.unique(years)

                if unique_years.size < RAMP_YEARS:
                    ds.close()
                    raise ValueError(f"{model} {stage} 只有{unique_years.size}个年份，少于{RAMP_YEARS}")

                target_years = unique_years[:RAMP_YEARS]

                if np.any(np.diff(target_years) != 1):
                    ds.close()
                    raise ValueError(f"{model} {stage} 前{RAMP_YEARS}个实验年份不连续")

                year_counts = np.array([np.count_nonzero(years == year) for year in target_years], dtype=int)

                if np.all(year_counts == 12):
                    steps_per_year = 12

                    for year in target_years:
                        year_months = np.sort(months[years == year])

                        if not np.array_equal(year_months, np.arange(1, 13)):
                            ds.close()
                            raise ValueError(f"{model} {stage} {year}年不是完整1–12月")

                    annualization = "decoded_calendar_12_month_mean"

                elif np.all(year_counts == 1):
                    steps_per_year = 1
                    annualization = "decoded_calendar_annual_input"

                else:
                    ds.close()
                    raise ValueError(
                        f"{model} {stage} 前140年每年step不是统一12或1："
                        f"unique={np.unique(year_counts).tolist()}"
                    )

                selected_index = np.flatnonzero(np.isin(years, target_years))
                reference_years = target_years[:BASELINE_YEARS] if stage == "up" else target_years[-BASELINE_YEARS:]
                reference_index = np.flatnonzero(np.isin(years, reference_years))

            # %% 3b. 0–90°N finite-SIC目标网格
            sic_reference = sic_fraction.isel(time=reference_index).mean("time", skipna=True).transpose(*spatial_dims).compute()
            sic_reference_values = np.asarray(sic_reference.values, dtype=float)

            match_mask = (
                valid_coord
                & np.isfinite(sic_reference_values)
                & (lat >= MATCH_LAT_MIN)
                & (lat <= MATCH_LAT_MAX)
            )

            na_geo_mask = (
                valid_coord
                & (lat >= NA_LAT_MIN)
                & (lat < NA_LAT_MAX)
                & (lon >= NA_LON_MIN)
                & (lon < NA_LON_MAX)
            )

            match_count = int(np.count_nonzero(match_mask))

            if match_count < 100:
                ds.close()
                raise ValueError(f"{model} {stage} 0–90°N finite-SIC目标格点只有{match_count}")

            # %% 3c. areacello候选
            aliases = {re.sub(r"[^a-z0-9]", "", text.lower()) for text in (model, *FOLDER_CANDIDATES[model])}
            area_file_map = {}

            for path in all_area_files:
                normalized = re.sub(r"[^a-z0-9]", "", path.as_posix().lower())

                if any(alias and alias in normalized for alias in aliases):
                    key = os.path.normcase(os.path.realpath(str(path)))
                    area_file_map.setdefault(key, path)

            area_candidates = list(area_file_map.values())

            if not area_candidates:
                ds.close()
                raise FileNotFoundError(f"{model} 未找到明确匹配的areacello")

            target_flat = np.flatnonzero(match_mask.ravel())
            target_lat = lat.ravel()[target_flat]
            target_lon = lon.ravel()[target_flat]

            target_lat_rad = np.deg2rad(target_lat)
            target_lon_rad = np.deg2rad(target_lon)

            target_xyz = np.column_stack((
                np.cos(target_lat_rad) * np.cos(target_lon_rad),
                np.cos(target_lat_rad) * np.sin(target_lon_rad),
                np.sin(target_lat_rad),
            ))

            passed = []
            attempted = []

            for area_path in area_candidates:
                ads = xr.open_dataset(area_path, decode_times=False, mask_and_scale=True)

                if "areacello" not in ads.data_vars:
                    ads.close()
                    continue

                area = ads["areacello"].squeeze(drop=True)
                area_units = str(area.attrs.get("units", "")).strip().lower().replace(" ", "")

                if area_units not in {"m2", "m^2", "m**2", "m²"}:
                    ads.close()
                    continue

                alat_names = [
                    name for name in ads.variables
                    if str(ads[name].attrs.get("standard_name", "")).lower() == "latitude"
                    and set(ads[name].dims).issubset(set(area.dims))
                ]

                alon_names = [
                    name for name in ads.variables
                    if str(ads[name].attrs.get("standard_name", "")).lower() == "longitude"
                    and set(ads[name].dims).issubset(set(area.dims))
                ]

                if len(alat_names) != 1:
                    alat_names = [
                        name for name in ads.variables
                        if str(name).lower() in LAT_NAMES
                        and set(ads[name].dims).issubset(set(area.dims))
                    ]

                if len(alon_names) != 1:
                    alon_names = [
                        name for name in ads.variables
                        if str(name).lower() in LON_NAMES
                        and set(ads[name].dims).issubset(set(area.dims))
                    ]

                if len(alat_names) != 1 or len(alon_names) != 1:
                    ads.close()
                    continue

                alat, alon = xr.broadcast(ads[alat_names[0]], ads[alon_names[0]])

                if set(alat.dims) != set(area.dims) or set(alon.dims) != set(area.dims):
                    ads.close()
                    continue

                alat = alat.transpose(*area.dims)
                alon = alon.transpose(*area.dims)

                area_values = np.asarray(area.values, dtype=float)
                area_lat = np.asarray(alat.values, dtype=float)
                area_lon_raw = np.asarray(alon.values, dtype=float)

                valid_area_coord = (
                    np.isfinite(area_lat)
                    & np.isfinite(area_lon_raw)
                    & (area_lat >= -90.0)
                    & (area_lat <= 90.0)
                    & (area_lon_raw >= -360.0)
                    & (area_lon_raw <= 720.0)
                )

                area_lon = np.full(area_lon_raw.shape, np.nan, dtype=float)
                area_lon[valid_area_coord] = (area_lon_raw[valid_area_coord] + 180.0) % 360.0 - 180.0

                source_valid = (
                    np.isfinite(area_values)
                    & (area_values > 0.0)
                    & valid_area_coord
                    & (area_lat >= MATCH_LAT_MIN - 5.0)
                    & (area_lat <= MATCH_LAT_MAX)
                )

                if int(np.count_nonzero(source_valid)) < 100:
                    ads.close()
                    continue

                source_area = area_values[source_valid]
                source_lat = area_lat[source_valid]
                source_lon = area_lon[source_valid]

                source_lat_rad = np.deg2rad(source_lat)
                source_lon_rad = np.deg2rad(source_lon)

                source_xyz = np.column_stack((
                    np.cos(source_lat_rad) * np.cos(source_lon_rad),
                    np.cos(source_lat_rad) * np.sin(source_lon_rad),
                    np.sin(source_lat_rad),
                ))

                tree = cKDTree(source_xyz)
                chord, source_index = tree.query(target_xyz, k=1)

                distance = 2.0 * EARTH_RADIUS_KM * np.arcsin(
                    np.clip(np.asarray(chord, dtype=float) / 2.0, 0.0, 1.0)
                )

                accepted = distance <= MAX_AREA_MATCH_DISTANCE_KM
                accepted_count = int(np.count_nonzero(accepted))
                coverage = accepted_count / match_count

                accepted_distance = distance[accepted]
                p95 = float(np.nanpercentile(accepted_distance, 95.0)) if accepted_distance.size else np.nan
                maximum = float(np.nanmax(accepted_distance)) if accepted_distance.size else np.nan

                accepted_source = np.asarray(source_index[accepted], dtype=int)
                duplicate = 1.0 - np.unique(accepted_source).size / accepted_count if accepted_count else 1.0

                mapped = np.full(lat.shape, np.nan, dtype=float)

                if accepted_count:
                    multiplicity = np.bincount(accepted_source, minlength=source_area.size)
                    mapped.ravel()[target_flat[accepted]] = source_area[accepted_source] / multiplicity[accepted_source]

                record = {
                    "path": area_path,
                    "mapped": mapped,
                    "coverage": coverage,
                    "p95": p95,
                    "maximum": maximum,
                    "duplicate": duplicate,
                    "accepted": accepted_count,
                }

                attempted.append(record)

                if (
                    coverage >= MIN_AREA_MATCH_COVERAGE
                    and np.isfinite(p95)
                    and p95 <= MAX_AREA_MATCH_P95_DISTANCE_KM
                    and duplicate <= MAX_DUPLICATE_AREA_SOURCE_FRACTION
                ):
                    passed.append(record)

                ads.close()

            if passed:
                best = passed[0]
                best_score = (best["p95"], best["duplicate"], -best["coverage"])

                for candidate in passed[1:]:
                    candidate_score = (candidate["p95"], candidate["duplicate"], -candidate["coverage"])

                    if candidate_score < best_score:
                        best = candidate
                        best_score = candidate_score

            elif attempted:
                best = attempted[0]
                best_p95 = best["p95"] if np.isfinite(best["p95"]) else np.inf
                best_score = (1.0 - best["coverage"], best["duplicate"], best_p95)

                for candidate in attempted[1:]:
                    candidate_p95 = candidate["p95"] if np.isfinite(candidate["p95"]) else np.inf
                    candidate_score = (1.0 - candidate["coverage"], candidate["duplicate"], candidate_p95)

                    if candidate_score < best_score:
                        best = candidate
                        best_score = candidate_score

                ds.close()

                raise ValueError(
                    f"{model} {stage} 最佳areacello未通过QA："
                    f"coverage={best['coverage']:.2%}, "
                    f"p95={best['p95']:.2f} km, "
                    f"duplicate={best['duplicate']:.2%}, "
                    f"file={best['path'].name}"
                )

            else:
                ds.close()
                raise ValueError(f"{model} {stage} 没有可用areacello候选")

            # %% 3d. 截取新NA并计算SIA
            area_da = xr.DataArray(best["mapped"], dims=spatial_dims)
            na_valid_mask = na_geo_mask & np.isfinite(best["mapped"])
            na_da = xr.DataArray(na_valid_mask, dims=spatial_dims)

            na_ocean_cells = int(np.count_nonzero(na_valid_mask))
            na_area_million_km2 = float(np.nansum(best["mapped"][na_valid_mask])) / 1.0e12

            if na_ocean_cells < 20:
                ds.close()
                raise ValueError(f"{model} {stage} 匹配后NA有效海洋格点只有{na_ocean_cells}")

            selected_sic = sic_fraction.isel(time=selected_index).where(na_da)
            selected_sia = (selected_sic * area_da).sum(dim=spatial_dims, skipna=True, min_count=1)

            valid_area_ts = area_da.where(selected_sic.notnull()).sum(dim=spatial_dims, skipna=True)
            total_na_area = area_da.where(na_da).sum(dim=spatial_dims, skipna=True)
            sia_coverage = (valid_area_ts / total_na_area).compute()
            sia_coverage_values = np.asarray(sia_coverage.values, dtype=float)

            if not np.all(np.isfinite(sia_coverage_values)):
                ds.close()
                raise ValueError(f"{model} {stage} NA SIA时间覆盖率含NaN/Inf")

            min_time_coverage = float(np.min(sia_coverage_values))
            median_time_coverage = float(np.median(sia_coverage_values))

            if min_time_coverage < MIN_SIA_TIME_COVERAGE:
                ds.close()
                raise ValueError(
                    f"{model} {stage} NA SIA最低时间覆盖率={min_time_coverage:.2%}，"
                    f"低于{MIN_SIA_TIME_COVERAGE:.0%}"
                )

            # %% 3e. 年平均
            if cas_teacher_cdr:
                monthly_values = np.asarray(selected_sia.compute().values, dtype=float) / 1.0e12

                if monthly_values.size != CAS_CDR_USED_MONTHS:
                    ds.close()
                    raise RuntimeError(f"CAS CDR用于分析的月数不是{CAS_CDR_USED_MONTHS}：{monthly_values.size}")

                annual_values = monthly_values.reshape(RAMP_YEARS, 12).mean(axis=1)
                annual_years = np.arange(1, RAMP_YEARS + 1, dtype=int)

            elif steps_per_year == 12:
                annual_sia = selected_sia.groupby("time.year").mean("time", skipna=True)
                annual_years = np.asarray(annual_sia["year"].values, dtype=int)
                annual_values = np.asarray(annual_sia.compute().values, dtype=float) / 1.0e12

            else:
                annual_sia = selected_sia.groupby("time.year").mean("time", skipna=True)
                annual_years = np.asarray(annual_sia["year"].values, dtype=int)
                annual_values = np.asarray(annual_sia.compute().values, dtype=float) / 1.0e12

            if annual_values.size != RAMP_YEARS:
                ds.close()
                raise ValueError(f"{model} {stage} 年度NA SIA不是{RAMP_YEARS}年：{annual_values.size}")

            if not np.array_equal(annual_years, target_years):
                ds.close()
                raise ValueError(f"{model} {stage} 年度SIA年份/实验年索引与目标140年不一致")

            if not np.all(np.isfinite(annual_values)):
                ds.close()
                raise ValueError(f"{model} {stage} 年度NA SIA含NaN/Inf")

            stage_series[stage] = annual_values
            stage_years[stage] = annual_years

            stage_qa[stage] = {
                "member": member if member is not None else "teacher_processed_CAS_CDR",
                "sic_file": ";".join(str(path) for path in sic_files),
                "sic_variable": sic_name,
                "sic_unit_mode": unit_mode,
                "annualization": annualization,
                "source_year_start": int(annual_years[0]),
                "source_year_end": int(annual_years[-1]),
                "lat_name": lat_names[0],
                "lon_name": lon_names[0],
                "match_target_cells_0to90N": match_count,
                "area_file": str(best["path"]),
                "area_accepted_cells_0to90N": best["accepted"],
                "area_coverage_0to90N": best["coverage"],
                "area_p95_km_0to90N": best["p95"],
                "area_max_km_0to90N": best["maximum"],
                "area_duplicate_fraction_0to90N": best["duplicate"],
                "na_ocean_cells": na_ocean_cells,
                "na_area_million_km2": na_area_million_km2,
                "minimum_time_coverage": min_time_coverage,
                "median_time_coverage": median_time_coverage,
                "teacher_processed_cdr": cas_teacher_cdr,
            }

            ds.close()

            print(
                f"[{stage}] target={match_count} | area coverage={best['coverage']:.2%} | "
                f"p95={best['p95']:.2f} km | duplicate={best['duplicate']:.2%} | "
                f"NA cells={na_ocean_cells} | time coverage min={min_time_coverage:.2%}"
            )

        # %% 3f. up/CDR面积一致性 + raw endpoint
        up_area = stage_qa["up"]["na_area_million_km2"]
        cdr_area = stage_qa["cdr"]["na_area_million_km2"]
        area_mean = 0.5 * (abs(up_area) + abs(cdr_area))
        area_difference = abs(cdr_area - up_area) / max(area_mean, 1.0e-12)

        if area_difference > MAX_STAGE_AREA_RELATIVE_DIFFERENCE:
            raise ValueError(
                f"{model} up/CDR NA有效海洋面积差异={area_difference:.2%}，"
                f"超过{MAX_STAGE_AREA_RELATIVE_DIFFERENCE:.0%}"
            )

        up_raw = stage_series["up"]
        cdr_raw = stage_series["cdr"]

        co2_initial10 = float(np.mean(up_raw[:BASELINE_YEARS]))
        co2_final10 = float(np.mean(up_raw[-BASELINE_YEARS:]))
        cdr_initial10 = float(np.mean(cdr_raw[:BASELINE_YEARS]))
        cdr_final10 = float(np.mean(cdr_raw[-BASELINE_YEARS:]))
        endpoint_delta = cdr_final10 - co2_initial10

        raw_anomaly = np.concatenate([up_raw, cdr_raw]) - co2_initial10
        smooth_anomaly = pd.Series(raw_anomaly).rolling(SMOOTH_YEARS, center=True, min_periods=1).mean().to_numpy(dtype=float)

        sia_anomaly_series[model] = raw_anomaly
        sia_smooth_series[model] = smooth_anomaly
        sia_endpoint_delta[model] = endpoint_delta

        for stage, values, source_years, offset in [
            ("up", up_raw, stage_years["up"], 0),
            ("cdr", cdr_raw, stage_years["cdr"], RAMP_YEARS),
        ]:
            for index in range(RAMP_YEARS):
                combined = offset + index

                sia_annual_rows.append({
                    "model": model,
                    "experiment": stage,
                    "analysis_year": index + 1,
                    "simulation_year": combined + 1,
                    "source_year": int(source_years[index]),
                    "sia_raw_million_km2": float(values[index]),
                    "sia_anomaly_vs_co2_initial10_million_km2": float(raw_anomaly[combined]),
                    "sia_11yr_smoothed_anomaly_million_km2": float(smooth_anomaly[combined]),
                })

        for stage in ("up", "cdr"):
            q = stage_qa[stage]

            sia_qa_rows.append({
                "model": model,
                "experiment": stage,
                "selected_member": q["member"],
                "sic_file": q["sic_file"],
                "sic_variable": q["sic_variable"],
                "sic_unit_mode": q["sic_unit_mode"],
                "annualization": q["annualization"],
                "teacher_processed_cdr": q["teacher_processed_cdr"],
                "source_year_start": q["source_year_start"],
                "source_year_end": q["source_year_end"],
                "sic_latitude_name": q["lat_name"],
                "sic_longitude_name": q["lon_name"],
                "match_target_cells_0to90N": q["match_target_cells_0to90N"],
                "areacello_file": q["area_file"],
                "area_match_accepted_cells_0to90N": q["area_accepted_cells_0to90N"],
                "area_match_coverage_0to90N": q["area_coverage_0to90N"],
                "area_match_p95_km_0to90N": q["area_p95_km_0to90N"],
                "area_match_max_km_0to90N": q["area_max_km_0to90N"],
                "area_duplicate_source_fraction_0to90N": q["area_duplicate_fraction_0to90N"],
                "na_ocean_cells": q["na_ocean_cells"],
                "matched_na_area_million_km2": q["na_area_million_km2"],
                "na_time_coverage_min": q["minimum_time_coverage"],
                "na_time_coverage_median": q["median_time_coverage"],
                "up_cdr_na_area_relative_difference": area_difference,
                "co2_initial10_mean_million_km2": co2_initial10,
                "co2_final10_mean_million_km2": co2_final10,
                "cdr_initial10_mean_million_km2": cdr_initial10,
                "cdr_final10_mean_million_km2": cdr_final10,
                "cdr_final10_minus_co2_initial10_million_km2": endpoint_delta,
                "picontrol_used": False,
            })

        print(
            f"[SIA OK] CO2 initial10={co2_initial10:.4f} | "
            f"CDR final10={cdr_final10:.4f} | "
            f"Δ={endpoint_delta:+.4f} million km²"
        )

        print(
            f"[AREA QA] up={up_area:.4f} | cdr={cdr_area:.4f} million km² | "
            f"relative diff={area_difference:.3%}"
        )

    except Exception as error:
        sia_failure_rows.append({"model": model, "failure_reason": f"{type(error).__name__}: {error}"})
        print(f"[FAILED] {model}: {type(error).__name__}: {error}")

if len(sia_smooth_series) < 2:
    pd.DataFrame(sia_annual_rows).to_csv(SIA_ANNUAL_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")
    pd.DataFrame(sia_qa_rows).to_csv(SIA_QA_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")
    pd.DataFrame(sia_failure_rows, columns=["model", "failure_reason"]).to_csv(SIA_FAIL_FILE, index=False, encoding="utf-8-sig")
    raise RuntimeError(f"新NA SIA成功模式少于2个：{list(sia_smooth_series)}")

# %% 4. 与当前pattern raw SIA做事后独立QA
pattern_cell_file = None

for path in PATTERN_CELL_CANDIDATES:
    if path.is_file():
        pattern_cell_file = path
        break

comparison_rows = []

if pattern_cell_file is not None:
    pattern = pd.read_csv(pattern_cell_file)

    required_pattern = {
        "model",
        "latitude_deg_north",
        "longitude_deg_east",
        "raw_delta_sia_thousand_km2",
    }

    if not required_pattern.issubset(pattern.columns):
        print(f"[PATTERN QA跳过] {pattern_cell_file.name} 缺少字段：{sorted(required_pattern - set(pattern.columns))}")

    else:
        pattern_lon = (pattern["longitude_deg_east"].to_numpy(dtype=float) + 180.0) % 360.0 - 180.0
        pattern_lat = pattern["latitude_deg_north"].to_numpy(dtype=float)

        pattern_na_mask = (
            np.isfinite(pattern_lat)
            & np.isfinite(pattern_lon)
            & (pattern_lat >= NA_LAT_MIN)
            & (pattern_lat < NA_LAT_MAX)
            & (pattern_lon >= NA_LON_MIN)
            & (pattern_lon < NA_LON_MAX)
        )

        for model, independent_value in sia_endpoint_delta.items():
            model_mask = pattern["model"].astype(str).to_numpy() == model
            selected = pattern.loc[model_mask & pattern_na_mask, "raw_delta_sia_thousand_km2"].to_numpy(dtype=float)
            pattern_value = float(np.nansum(selected)) / 1000.0 if selected.size else np.nan
            difference = independent_value - pattern_value if np.isfinite(pattern_value) else np.nan
            same_2dp = bool(round(independent_value, 2) == round(pattern_value, 2)) if np.isfinite(pattern_value) else False

            comparison_rows.append({
                "model": model,
                "independent_cdr_year131_140_minus_co2_year1_10_million_km2": independent_value,
                "pattern_cells_na_delta_million_km2": pattern_value,
                "independent_minus_pattern_million_km2": difference,
                "absolute_difference_million_km2": abs(difference) if np.isfinite(difference) else np.nan,
                "same_when_rounded_to_2dp": same_2dp,
            })

            print(
                f"[PATTERN QA] {model:<15} | independent={independent_value:+.4f} | "
                f"pattern={pattern_value:+.4f} | diff={difference:+.4f} | 2dp={same_2dp}"
            )

        print(f"[PATTERN QA source] {pattern_cell_file}")

else:
    print("[PATTERN QA跳过] 没找到当前pattern cells文件")

pd.DataFrame(
    comparison_rows,
    columns=[
        "model",
        "independent_cdr_year131_140_minus_co2_year1_10_million_km2",
        "pattern_cells_na_delta_million_km2",
        "independent_minus_pattern_million_km2",
        "absolute_difference_million_km2",
        "same_when_rounded_to_2dp",
    ],
).to_csv(COMPARISON_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")

# %% 5. 原样读取旧图中的AMOC和CO2，不重新计算
old_plot = pd.read_csv(OLD_PLOT_DATA_FILE).sort_values("simulation_year").reset_index(drop=True)

required_old = {"simulation_year", "calendar_year", "stage", "co2_ppm"}

if required_old - set(old_plot.columns):
    raise KeyError(f"旧plot-ready文件缺少基础列：{sorted(required_old - set(old_plot.columns))}")

if len(old_plot) != 2 * RAMP_YEARS:
    raise ValueError(f"旧AMOC plot-ready序列应为280年，实际={len(old_plot)}")

if not np.array_equal(old_plot["simulation_year"].to_numpy(dtype=int), np.arange(1, 281)):
    raise ValueError("旧AMOC plot-ready simulation_year不是严格1–280")

common_models = []

for model in MODELS:
    amoc_column = f"amoc_{model}"

    if model not in sia_smooth_series:
        continue

    if amoc_column not in old_plot.columns:
        print(f"[AMOC skip] {model}: 旧plot-ready无{amoc_column}")
        continue

    amoc_values = old_plot[amoc_column].to_numpy(dtype=float)

    if not np.all(np.isfinite(amoc_values)):
        print(f"[AMOC skip] {model}: 旧plot-ready AMOC含NaN/Inf")
        continue

    common_models.append(model)
    print(f"[AMOC reuse] {model}: 原样复制旧plot-ready 40°N AMOC")

if len(common_models) < 2:
    raise RuntimeError(f"新SIA与旧AMOC共同模式少于2个：{common_models}")

print(f"\n[共同模式] {len(common_models)}：" + ", ".join(common_models))

# %% 6. 构造新的plot-ready文件
plot_data = old_plot[["simulation_year", "calendar_year", "stage", "co2_ppm"]].copy()

for model in common_models:
    plot_data[f"sia_{model}"] = sia_smooth_series[model]
    plot_data[f"sia_raw_anomaly_{model}"] = sia_anomaly_series[model]
    plot_data[f"amoc_{model}"] = old_plot[f"amoc_{model}"].to_numpy(dtype=float)

# %% 7. 保存
pd.DataFrame(sia_annual_rows).to_csv(SIA_ANNUAL_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")
pd.DataFrame(sia_qa_rows).to_csv(SIA_QA_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")
pd.DataFrame(sia_failure_rows, columns=["model", "failure_reason"]).to_csv(SIA_FAIL_FILE, index=False, encoding="utf-8-sig")
plot_data.to_csv(PLOT_DATA_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")

metadata = {
    "north_atlantic_region": {
        "latitude": "45 <= lat < 70 degN",
        "longitude": "-60 <= lon < 30 degE",
    },
    "sia_definition": "sum(raw SIC fraction * areacello) on native ocean grid",
    "sia_picontrol_used": False,
    "sia_baseline": "raw 1pctCO2 experiment years 1-10 mean",
    "sia_endpoint": "raw CDR experiment years 131-140 mean minus raw 1pctCO2 experiment years 1-10 mean",
    "sia_smoothing": "11-year centered running mean, plotting only",
    "areacello_match_domain": "0-90N finite endpoint SIC ocean grid, then subset to NA",
    "pattern_used_in_sia_calculation": False,
    "pattern_qa_cells": str(pattern_cell_file) if pattern_cell_file is not None else None,
    "amoc_recalculated": False,
    "amoc_source": str(OLD_PLOT_DATA_FILE),
    "amoc_rule": "existing amoc_* plot-ready columns copied unchanged",
    "common_models": common_models,
    "cas_cdr": {
        "file": str(CAS_CDR_FILE),
        "variable": "aice",
        "units": "%",
        "grid": "196x360",
        "raw_months": CAS_CDR_TOTAL_MONTHS,
        "raw_years": CAS_CDR_TOTAL_YEARS,
        "analysis_months": "0:1680",
        "analysis_years": "1-140",
        "final10_months": "1560:1680",
        "final10_years": "131-140",
        "excluded_months": "1680:1920",
        "excluded_years": 20,
        "calendar_year_groupby": False,
        "annualization": "consecutive 12-record blocks",
    },
    "nor_rule": "r1i1p1f1 1pctCO2 -> r1i1p1f1 CDR only; no p4->p1 splice",
}

METADATA_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

print("\n" + "=" * 108)
print("NEW NORTH ATLANTIC RAW SIA RECALCULATION COMPLETE")
print(f"SIA annual : {SIA_ANNUAL_FILE}")
print(f"SIA QA     : {SIA_QA_FILE}")
print(f"SIA failed : {SIA_FAIL_FILE}")
print(f"Pattern QA : {COMPARISON_FILE}")
print(f"Plot data  : {PLOT_DATA_FILE}")
print(f"Metadata   : {METADATA_FILE}")
print("=" * 108)