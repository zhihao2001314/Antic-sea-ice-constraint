# -*- coding: utf-8 -*-
"""
AMOC深水形成区诊断：40°N、65°N及40°N−65°N的piControl-relative anomaly。

【是否需要运行】
需要。完成后运行：
plot/amoc/n12_amoc_deepwater_regions_300m.py

【输入文件】
input/cmip6amocad 或 F:/sea/cmip6amocad
input/cmip6amoc 或 F:/sea/cmip6amoc
input/cmip6amoc-picontrol 或 F:/sea/cmip6amoc-picontrol

【输出文件】
output/amoc_s01_n01/n12_deepwater_regions_300m/
    amoc_deepwater_regions_300m_annual.csv
    amoc_deepwater_regions_300m_pi_reference.csv
    amoc_deepwater_regions_300m_qa.csv
    amoc_deepwater_regions_300m_failures.csv
    amoc_deepwater_regions_300m_metadata.json

【什么时候需要重新运行】
AMOC原始数据、member、目标纬度、深度阈值、盆地选择或piControl参考定义变化后。

科学定义
--------
1. Atlantic-Arctic overturning。
2. 固定40°N与65°N最近物理纬度。
3. 先做年度平均streamfunction剖面，再在depth >= 300 m取正向极大值。
4. 40°N作为总体AMOC诊断。
5. 65°N作为GIN Seas高纬AMOC诊断。
6. 40°N−65°N作为Labrador-Irminger Sea相关贡献proxy。
7. 1pctCO2 years 1–140 + CDR years 1–140组成280年forced序列。
8. 每个诊断直接减去对应piControl climatology，不进行线性drift外推。
9. piControl climatology使用最长连续共同有效块的全部年份。
10. 保存raw和piControl-relative anomaly。
11. 不补年、不插值、不外推。
12. NorESM2-LM使用本地实际分支：pi=p1、1pctCO2=p4、CDR=p1，仅作诊断。
"""

# %% 1. 依赖

from operator import itemgetter
from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)


# %% 2. 路径与参数

ROOT = Path(__file__).resolve().parents[2]

PI_ROOT = ROOT/"input"/"cmip6amoc-picontrol"
UP_ROOT = ROOT/"input"/"cmip6amocad"
CDR_ROOT = ROOT/"input"/"cmip6amoc"

if not PI_ROOT.is_dir():
    PI_ROOT = Path(r"F:\sea\cmip6amoc-picontrol")
if not UP_ROOT.is_dir():
    UP_ROOT = Path(r"F:\sea\cmip6amocad")
if not CDR_ROOT.is_dir():
    CDR_ROOT = Path(r"F:\sea\cmip6amoc")

OUT = ROOT/"output"/"amoc_s01_n01"/"n12_deepwater_regions_300m"
CNRM_CACHE = ROOT/"output"/"amoc_s01_n01"/"cnrm_grid_cache"/"CNRM-ESM2-1_j294_latitude_mapping_verified_v4.npz"

ANNUAL_FILE = OUT/"amoc_deepwater_regions_300m_annual.csv"
PI_FILE = OUT/"amoc_deepwater_regions_300m_pi_reference.csv"
QA_FILE = OUT/"amoc_deepwater_regions_300m_qa.csv"
FAIL_FILE = OUT/"amoc_deepwater_regions_300m_failures.csv"
META_FILE = OUT/"amoc_deepwater_regions_300m_metadata.json"

MODELS = [
    "ACCESS-ESM1-5", "CanESM5", "CAS-ESM2-0", "CESM2",
    "CNRM-ESM2-1", "GFDL-ESM4", "MIROC-ES2L",
    "NorESM2-LM", "UKESM1-0-LL",
]

DIRS = {
    "ACCESS-ESM1-5": "ACCESSESM5",
    "CanESM5": "CanESM5",
    "CAS-ESM2-0": "CAS",
    "CESM2": "CESM2",
    "CNRM-ESM2-1": "CNRM-ESM2",
    "GFDL-ESM4": "GFDL",
    "MIROC-ES2L": "MIROC-ES2L",
    "NorESM2-LM": "NORESM2-LM2",
    "UKESM1-0-LL": "UKESM1-0-LL",
}

MEMBERS = {
    "ACCESS-ESM1-5": {"pi": "r1i1p1f1", "up": "r1i1p1f1", "cdr": "r1i1p1f1"},
    "CanESM5": {"pi": "r1i1p2f1", "up": "r1i1p2f1", "cdr": "r1i1p2f1"},
    "CAS-ESM2-0": {"pi": "r1i1p1f1", "up": "r1i1p1f1", "cdr": "teacher_product"},
    "CESM2": {"pi": "r1i1p1f1", "up": "r1i1p1f1", "cdr": "r1i1p1f1"},
    "CNRM-ESM2-1": {"pi": "r1i1p1f2", "up": "r1i1p1f2", "cdr": "r1i1p1f2"},
    "GFDL-ESM4": {"pi": "r1i1p1f1", "up": "r1i1p1f1", "cdr": "r1i1p1f1"},
    "MIROC-ES2L": {"pi": "r1i1p1f2", "up": "r1i1p1f2", "cdr": "r1i1p1f2"},
    "NorESM2-LM": {"pi": "r1i1p1f1", "up": "r1i1p1f1", "cdr": "r1i1p1f1"},
    "UKESM1-0-LL": {"pi": "r1i1p1f2", "up": "r1i1p1f2", "cdr": "r1i1p1f2"},
}

VARIABLES = {
    "ACCESS-ESM1-5": {"pi": "msftmz", "up": "msftmz", "cdr": "msftmz"},
    "CanESM5": {"pi": "msftmz", "up": "msftmz", "cdr": "msftmz"},
    "CAS-ESM2-0": {"pi": "msftmz", "up": "msftmz", "cdr": "amoc"},
    "CESM2": {"pi": "msftmz", "up": "msftmz", "cdr": "msftmz"},
    "CNRM-ESM2-1": {"pi": "msftyz", "up": "msftyz", "cdr": "msftyz"},
    "GFDL-ESM4": {"pi": "msftyz", "up": "msftyz", "cdr": "msftyz"},
    "MIROC-ES2L": {"pi": "msftmz", "up": "msftmz", "cdr": "msftmz"},
    "NorESM2-LM": {"pi": "msftmz", "up": "msftmz", "cdr": "msftmz"},
    "UKESM1-0-LL": {"pi": "msftyz", "up": "msftyz", "cdr": "msftyz"},
}

ROOTS = {"pi": PI_ROOT, "up": UP_ROOT, "cdr": CDR_ROOT}

TARGET_LATS = [40.0, 65.0]
MAX_LAT_DISTANCE = 5.0
DEPTH_MIN = 300.0
FORCED_YEARS = 140
MIN_PI_YEARS = 100
HARD_QA_ABS_MAX_SV = 1000.0

TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)
MEMBER_RE = re.compile(r"r\d+i\d+p\d+f\d+", re.I)

AUX_GRID_ROOTS = [
    ROOT/"input"/"areacello", ROOT/"input"/"cmipseaicead",
    ROOT/"input"/"cmipseaice", ROOT/"input"/"cmipseaice-picontrol",
    Path(r"F:\sea\areacello"), Path(r"F:\sea\cmipseaicead"),
    Path(r"F:\sea\cmipseaice"), Path(r"F:\sea\cmipseaice-picontrol"),
]

OUT.mkdir(parents=True, exist_ok=True)
CNRM_CACHE.parent.mkdir(parents=True, exist_ok=True)


# %% 3. 提取AMOC

annual_rows = []
pi_rows = []
qa_rows = []
failure_rows = []
successful = []

for model in MODELS:
    print("\n"+"="*108)
    print(f"{model} | 40°N / 65°N AMOC | depth >= {DEPTH_MIN:.0f} m")
    print("="*108)

    try:
        stage_results = {}
        basin_from_pi = None
        cnrm_orientation = "max"

        for exp in ("pi", "up", "cdr"):
            folder = ROOTS[exp]/DIRS[model]
            member = MEMBERS[model][exp]
            variable_name = VARIABLES[model][exp]
            cas_teacher = model == "CAS-ESM2-0" and exp == "cdr" and variable_name == "amoc"

            search_folders = [folder]
            if model == "ACCESS-ESM1-5" and exp == "pi":
                search_folders.append(Path(r"F:\sea\cmip6amoc-picontrol")/DIRS[model])

            candidate_files = []
            for search_folder in search_folders:
                if search_folder.is_dir():
                    candidate_files.extend(search_folder.rglob("*.nc"))

            candidate_files = sorted(set(candidate_files))
            if not candidate_files:
                raise FileNotFoundError(f"{model} {exp}没有NetCDF：{folder}")

            files = []

            for path in candidate_files:
                try:
                    with xr.open_dataset(path, decode_times=False, mask_and_scale=False, cache=False) as head:
                        if variable_name not in head.data_vars:
                            continue

                        if cas_teacher:
                            files.append(path)
                            continue

                        match = MEMBER_RE.search(path.as_posix())
                        path_member = match.group(0) if match else None
                        file_member = path_member or str(head.attrs.get("variant_label", "") or head.attrs.get("member_id", "")).strip()

                        if file_member == member:
                            files.append(path)

                except (OSError, ValueError):
                    continue

            if not files:
                if cas_teacher:
                    raise FileNotFoundError(f"CAS老师处理后的CDR目录未找到amoc：{folder}")
                raise FileNotFoundError(f"{model} {exp}未找到 {variable_name} / {member}")

            print(f"[{exp}] {variable_name} | {'teacher product' if cas_teacher else member} | files={len(files)}")

            ds = xr.open_mfdataset(
                [str(path) for path in files], combine="by_coords", decode_times=TIME_CODER,
                data_vars="minimal", coords="minimal", compat="override",
                parallel=False, chunks={"time": 120},
            )

            try:
                psi = ds[variable_name].squeeze().astype("float64")
                psi = psi.where(np.isfinite(psi) & (np.abs(psi) < 1e20))

                if int(psi.count().compute().item()) == 0:
                    raise ValueError(f"{model} {exp} {variable_name}全部无效")
                if "time" not in psi.dims:
                    raise ValueError(f"{model} {exp}缺少time维：{psi.dims}")

                variant_label = str(ds.attrs.get("variant_label", "")).strip().lower()
                if cas_teacher:
                    variant_label = variant_label or "teacher_product"
                elif variant_label and variant_label != member.lower():
                    raise ValueError(f"{model} {exp} metadata member={variant_label}，选择={member}")

                # %% 3.1 Atlantic-Arctic basin

                basin_dim = next(
                    (
                        dim for dim in psi.dims
                        if "basin" in dim.lower() or "region" in dim.lower() or "sector" in dim.lower()
                        or dim.lower() in {"transport_reg", "transport_region"}
                    ),
                    None,
                )

                basin_index = None
                basin_text = "pre-derived Atlantic AMOC"

                if cas_teacher:
                    if basin_dim is not None:
                        raise ValueError("CAS teacher CDR amoc意外存在basin维")
                    basin_text = "teacher Atlantic-Arctic AMOC"

                elif basin_dim is not None:
                    nbasin = int(psi.sizes[basin_dim])

                    if exp != "pi" and basin_from_pi is not None:
                        if basin_from_pi >= nbasin:
                            raise ValueError(f"{model} {exp} basin维长度变化")
                        basin_index = basin_from_pi
                        basin_text = f"reuse piControl basin[{basin_index}]"

                    elif model == "CESM2" and basin_dim.lower() in {"transport_reg", "transport_region"} and nbasin == 2:
                        basin_index, basin_text = 1, "CESM2 POP transport_reg[1] Atlantic-Arctic"

                    else:
                        candidates, seen = [], set()

                        for name in ("transport_regions", "transport_region", "sector", "basin", "region", basin_dim):
                            if name in ds and ds[name].ndim in (1, 2) and ds[name].dims[0] == basin_dim:
                                candidates.append((name, ds[name]))
                                seen.add(name)

                        for name, coord in ds.variables.items():
                            if name not in seen and coord.ndim in (1, 2) and coord.dims[0] == basin_dim:
                                candidates.append((name, coord))
                                seen.add(name)

                        for name, coord in candidates:
                            requested = str(coord.attrs.get("requested", "")).lower()
                            match = re.search(r"(?:atlantic_arctic_ocean|atlantic_ocean)\s*=\s*(-?\d+)", requested)

                            if match:
                                value = int(match.group(1))
                                flat = np.asarray(coord.values).reshape(-1)

                                if flat.size == nbasin and np.issubdtype(flat.dtype, np.number):
                                    pos = np.where(np.isclose(flat.astype(float), value))[0]
                                    basin_index = int(pos[0]) if pos.size else None
                                elif 0 <= value < nbasin:
                                    basin_index = value

                                if basin_index is not None:
                                    basin_text = f"{name}:requested Atlantic-Arctic"
                                    break

                            names = re.findall(
                                r"(?:atlantic_arctic_ocean|atlantic_ocean|indian_pacific_ocean|global_ocean)",
                                requested,
                            )
                            hits = [i for i, text in enumerate(names[:nbasin]) if text in {"atlantic_arctic_ocean", "atlantic_ocean"}]

                            if hits:
                                basin_index = hits[0]
                                basin_text = f"{name}:requested Atlantic-Arctic"
                                break

                            values = np.asarray(coord.values)
                            labels = []

                            if values.dtype.kind in {"U", "S", "O"}:
                                if values.ndim == 1 and values.size == nbasin:
                                    labels = [
                                        value.decode("utf-8", "ignore").strip().lower()
                                        if isinstance(value, (bytes, np.bytes_)) else str(value).strip().lower()
                                        for value in values
                                    ]
                                elif values.ndim == 2 and values.shape[0] == nbasin:
                                    labels = [
                                        "".join(
                                            value.decode("utf-8", "ignore")
                                            if isinstance(value, (bytes, np.bytes_)) else str(value)
                                            for value in row
                                        ).strip().lower()
                                        for row in values
                                    ]

                            hits = [i for i, text in enumerate(labels) if "atlantic" in text or text.startswith("atl") or "atl_" in text]

                            if hits:
                                basin_index = hits[0]
                                basin_text = f"{name}:{labels[hits[0]]}"
                                break

                        if basin_index is None and model == "CESM2" and nbasin >= 3:
                            basin_index, basin_text = 0, "CESM2 verified basin[0] Atlantic-Arctic"

                        if basin_index is None and nbasin == 1:
                            basin_index, basin_text = 0, "single basin"

                        if basin_index is None:
                            raise ValueError(f"{model} {exp}无法确认Atlantic-Arctic basin")

                    psi = psi.isel({basin_dim: basin_index})

                    if exp == "pi":
                        basin_from_pi = basin_index

                else:
                    raise ValueError(f"{model} {exp}没有可验证Atlantic-Arctic basin维")

                # %% 3.2 MOC component

                comp_dim = next(
                    (
                        dim for dim in psi.dims
                        if re.sub(r"[^A-Z0-9]", "", dim.upper())
                        in {"MOCCOMP", "MOCCOMPONENT", "MOCCOMPONENTS"}
                    ),
                    None,
                )

                component_text = "none"

                if comp_dim:
                    if model == "CESM2":
                        valid_component = psi.notnull().any(comp_dim)
                        component_text = f"sum({comp_dim}, n={psi.sizes[comp_dim]})"
                        psi = psi.sum(comp_dim, skipna=True).where(valid_component)
                    elif psi.sizes[comp_dim] == 1:
                        component_text = f"{comp_dim}[0]"
                        psi = psi.isel({comp_dim: 0})
                    else:
                        raise ValueError(f"{model} {exp}未验证MOC component：{comp_dim}={psi.sizes[comp_dim]}")

                # %% 3.3 深度与纬度

                variables_all = dict(ds.variables)
                variables_all.update({name: psi.coords[name] for name in psi.coords})

                depth_candidates = []

                for name, coord in variables_all.items():
                    coord = coord.squeeze()

                    if coord.ndim != 1 or coord.dims[0] not in psi.dims:
                        continue

                    score = 10 if str(coord.attrs.get("standard_name", "")).lower() in {"depth", "ocean_depth", "depth_below_geoid"} else 0
                    score += 6 if str(coord.attrs.get("axis", "")).upper() == "Z" else 0
                    score += 5 if name.lower() in {"lev", "olevel", "depth", "depth_coord", "moc_z", "z_t", "z"} else 0

                    if score > 0:
                        depth_candidates.append((score, name, coord))

                if not depth_candidates:
                    raise ValueError(f"{model} {exp}无法识别深度")

                depth_candidates.sort(key=itemgetter(0), reverse=True)
                _, depth_name, depth_coord = depth_candidates[0]
                depth_dim = depth_coord.dims[0]

                lat_candidates = []

                for name, coord in variables_all.items():
                    coord = coord.squeeze()

                    if coord.ndim != 1 or coord.dims[0] not in psi.dims:
                        continue

                    score = 10 if str(coord.attrs.get("standard_name", "")).lower() == "latitude" else 0
                    score += 6 if str(coord.attrs.get("axis", "")).upper() == "Y" else 0
                    score += 5 if name.lower() in {"lat", "latitude", "nav_lat", "geolat", "rlat", "lat_aux_grid", "y"} else 0

                    coord_units = str(coord.attrs.get("units", "")).lower()
                    score += 4 if "degree_north" in coord_units or "degrees_north" in coord_units else 0

                    if score > 0:
                        values = np.asarray(coord.values, float).reshape(-1)
                        finite = values[np.isfinite(values)]

                        if finite.size >= 0.8*values.size and finite.min() >= -90.01 and finite.max() <= 90.01:
                            lat_candidates.append((score, name, coord))

                latitude = None
                lat_source = ""

                if lat_candidates:
                    lat_candidates.sort(key=itemgetter(0), reverse=True)
                    _, lat_name, latitude = lat_candidates[0]
                    lat_source = f"{lat_name} [AMOC file]"

                # %% 3.4 CNRM j-mean

                if latitude is None and model == "CNRM-ESM2-1":
                    j_dim = next((dim for dim in psi.dims if dim.lower() in {"j-mean", "j_mean", "jmean"}), None)

                    if j_dim is None or psi.sizes[j_dim] != 294:
                        raise ValueError(f"CNRM {exp}要求294行j-mean，实际={psi.dims}")

                    row_lat = None

                    if CNRM_CACHE.is_file():
                        cached = np.asarray(np.load(CNRM_CACHE, allow_pickle=False)["row_latitude"], float).reshape(-1)

                        if cached.size == 294 and np.isfinite(cached).sum() >= 20:
                            j40 = int(np.nanargmin(np.abs(cached-40.0)))
                            actual40 = float(cached[j40])

                            if j40 == 207 and abs(actual40-39.970472) <= 0.25:
                                row_lat = cached
                                lat_source = f"verified cache: {CNRM_CACHE}"

                    if row_lat is None:
                        grid_files = []

                        for grid_root in AUX_GRID_ROOTS:
                            if not grid_root.is_dir():
                                continue

                            for alias in ("CNRM-ESM2", "CNRM-ESM2-1"):
                                folder_grid = grid_root/alias

                                if folder_grid.is_dir():
                                    grid_files.extend(folder_grid.rglob("*.nc"))

                        grid_files = sorted(set(grid_files))

                        for grid_file in grid_files:
                            try:
                                grid = xr.open_dataset(grid_file, decode_times=False, cache=False)
                            except (OSError, ValueError):
                                continue

                            try:
                                lat2d = next(
                                    (
                                        grid[name].squeeze()
                                        for name in ("nav_lat", "latitude", "lat", "geolat")
                                        if name in grid and grid[name].squeeze().ndim == 2
                                    ),
                                    None,
                                )
                                lon2d = next(
                                    (
                                        grid[name].squeeze()
                                        for name in ("nav_lon", "longitude", "lon", "geolon")
                                        if name in grid and grid[name].squeeze().ndim == 2
                                    ),
                                    None,
                                )

                                if lat2d is None or lon2d is None or set(lat2d.dims) != set(lon2d.dims):
                                    continue

                                distance = [abs(int(lat2d.sizes[dim])-294) for dim in lat2d.dims]
                                grid_j = lat2d.dims[int(np.argmin(distance))]
                                source_n = int(lat2d.sizes[grid_j])

                                if abs(source_n-294) > 6:
                                    continue

                                grid_i = next(dim for dim in lat2d.dims if dim != grid_j)
                                latv = np.asarray(lat2d.transpose(grid_j, grid_i).values, float)
                                lonv = np.asarray(lon2d.transpose(grid_j, grid_i).values, float)
                                lon180 = (lonv+180.0)%360.0-180.0

                                mask = (
                                    np.isfinite(latv) & np.isfinite(lon180)
                                    & (lon180 >= -100.0) & (lon180 <= 20.0)
                                    & (latv >= -20.0) & (latv <= 80.0)
                                )

                                source = np.full(source_n, np.nan)

                                for row in range(source_n):
                                    row_values = latv[row, mask[row]]

                                    if row_values.size >= 3:
                                        source[row] = float(np.nanmedian(row_values))

                                finite = np.isfinite(source)

                                if finite.sum() < 20:
                                    continue

                                if source_n == 294:
                                    trial = source
                                else:
                                    sx = np.linspace(0.0, 1.0, source_n)
                                    tx = np.linspace(0.0, 1.0, 294)
                                    trial = np.interp(tx, sx[finite], source[finite], left=np.nan, right=np.nan)

                                j40 = int(np.nanargmin(np.abs(trial-40.0)))
                                actual40 = float(trial[j40])

                                if j40 == 207 and abs(actual40-39.970472) <= 0.25:
                                    row_lat = trial
                                    lat_source = f"Atlantic row median: {grid_file}"
                                    np.savez_compressed(CNRM_CACHE, row_latitude=row_lat)
                                    break

                            finally:
                                grid.close()

                    if row_lat is None:
                        raise ValueError("CNRM j-mean纬度映射失败")

                    jcoord = np.asarray(psi[j_dim].values) if j_dim in psi.coords else np.arange(294)
                    latitude = xr.DataArray(row_lat, dims=(j_dim,), coords={j_dim: jcoord})

                if latitude is None:
                    raise ValueError(f"{model} {exp}无法可靠识别物理纬度")

                lat_dim = latitude.dims[0]
                lat_values = np.asarray(latitude.values, float).reshape(-1)

                # %% 3.5 depth >= 300 m

                depth_units = str(depth_coord.attrs.get("units", "")).lower().replace(" ", "")
                depth_values = np.abs(np.asarray(depth_coord.values, float))

                if depth_units in {"m", "meter", "meters", "metre", "metres"}:
                    depth_m = depth_values
                elif depth_units in {"cm", "centimeter", "centimeters", "centimetre", "centimetres"}:
                    depth_m = depth_values/100.0
                elif depth_units in {"km", "kilometer", "kilometers", "kilometre", "kilometres"}:
                    depth_m = depth_values*1000.0
                else:
                    raise ValueError(f"{model} {exp}无法确认depth单位={depth_coord.attrs.get('units', '')!r}")

                depth_index = np.flatnonzero(np.isfinite(depth_m) & (depth_m >= DEPTH_MIN))

                if not depth_index.size:
                    raise ValueError(f"{model} {exp} {DEPTH_MIN:.0f}m以下无有效深度")

                psi = psi.isel({depth_dim: depth_index})

                # %% 3.6 单位

                units = str(ds[variable_name].attrs.get("units", "")).strip()
                unit_key = units.lower().replace(" ", "").replace("**", "^")

                if "sv" in unit_key or "sverdrup" in unit_key:
                    factor, conversion = 1.0, "already Sv"
                elif "kg" in unit_key and any(value in unit_key for value in ("s-1", "s^-1", "/s")):
                    factor, conversion = 1e-9, "kg s-1 / 1e9"
                elif ("m3" in unit_key or "m^3" in unit_key) and any(value in unit_key for value in ("s-1", "s^-1", "/s")):
                    factor, conversion = 1e-6, "m3 s-1 / 1e6"
                else:
                    raise ValueError(f"{model} {exp}无法确认AMOC单位={units!r}")

                psi = psi*factor

                max_abs_profile = float(np.abs(psi).max(skipna=True).compute())

                if np.isfinite(max_abs_profile) and max_abs_profile > HARD_QA_ABS_MAX_SV:
                    raise ValueError(f"{model} {exp} max|Psi|={max_abs_profile:.3e} Sv")

                # %% 3.7 年平均

                years = np.asarray(psi.time.dt.year.values, int)
                unique_years, counts = np.unique(years, return_counts=True)
                steps = int(pd.Series(counts).mode().iloc[0])

                if steps not in {1, 12}:
                    raise ValueError(f"{model} {exp}每年典型时间步={steps}")

                complete = unique_years[counts == steps]

                if exp == "pi":
                    keep_years = complete
                else:
                    if complete.size < FORCED_YEARS:
                        raise ValueError(f"{model} {exp}只有{complete.size}个完整年份")

                    keep_years = complete[:FORCED_YEARS]

                    if np.any(np.diff(keep_years) != 1):
                        raise ValueError(f"{model} {exp}前{FORCED_YEARS}年不连续")

                psi = psi.isel(time=np.flatnonzero(np.isin(years, keep_years)))

                if steps == 12:
                    days = psi.time.dt.days_in_month.astype("float64")
                    numerator = (psi*days).groupby("time.year").sum("time", skipna=True)
                    denominator = days.where(psi.notnull()).groupby("time.year").sum("time", skipna=True)
                    annual_profile = (numerator/denominator.where(denominator > 0)).rename({"year": "source_year"})
                else:
                    annual_profile = psi.groupby("time.year").mean("time", skipna=True).rename({"year": "source_year"})

                # %% 3.8 40N / 65N

                lat_results = {}

                for target_lat in TARGET_LATS:
                    lat_index = int(np.nanargmin(np.abs(lat_values-target_lat)))
                    actual_lat = float(lat_values[lat_index])

                    if abs(actual_lat-target_lat) > MAX_LAT_DISTANCE:
                        raise ValueError(f"{model} {exp}最近纬度={actual_lat:.3f}°N，距离{target_lat:.0f}°N过远")

                    profile = annual_profile.isel({lat_dim: lat_index})
                    extra_dims = [dim for dim in profile.dims if dim not in {"source_year", depth_dim}]

                    if extra_dims:
                        raise ValueError(f"{model} {exp}固定盆地和纬度后仍有维度={extra_dims}")

                    valid_profile = profile.notnull().any(depth_dim)
                    positive = profile.max(depth_dim, skipna=True).where(valid_profile)
                    negative = (-profile.min(depth_dim, skipna=True)).where(valid_profile)

                    if model == "CNRM-ESM2-1":
                        if exp == "pi" and target_lat == 40.0:
                            pos_mean = float(positive.mean(skipna=True).compute())
                            neg_mean = float(negative.mean(skipna=True).compute())

                            if 2.0 <= pos_mean <= 50.0:
                                cnrm_orientation = "max"
                            elif pos_mean < 1.0 and 2.0 <= neg_mean <= 80.0:
                                cnrm_orientation = "negative_min"
                            else:
                                raise ValueError(f"CNRM sign QA失败：max={pos_mean:.2f}, -min={neg_mean:.2f} Sv")

                        amoc = negative if cnrm_orientation == "negative_min" else positive
                        orientation = cnrm_orientation

                    else:
                        amoc = positive
                        orientation = "max"

                    source_year = np.asarray(amoc.source_year.values, int)
                    values = np.asarray(amoc.compute().values, float).reshape(-1)

                    if exp != "pi":
                        source_year = source_year[:FORCED_YEARS]
                        values = values[:FORCED_YEARS]

                        if values.size != FORCED_YEARS:
                            raise ValueError(f"{model} {exp} {target_lat:.0f}N长度={values.size}")
                        if not np.all(np.isfinite(values)):
                            raise ValueError(f"{model} {exp} {target_lat:.0f}N存在NaN/Inf")

                    lat_results[target_lat] = {
                        "source_year": source_year,
                        "values": values,
                        "actual_lat": actual_lat,
                        "orientation": orientation,
                    }

                stage_results[exp] = {
                    "lat": lat_results,
                    "member": member,
                    "variant_label": variant_label,
                    "parent_variant": str(ds.attrs.get("parent_variant_label", "")).strip(),
                    "basin_index": basin_index,
                    "basin_label": basin_text,
                    "component": component_text,
                    "latitude_source": lat_source,
                    "depth_min": float(depth_m[depth_index].min()),
                    "depth_max": float(depth_m[depth_index].max()),
                    "unit_conversion": conversion,
                    "max_abs_profile": max_abs_profile,
                }

                print(
                    f"  basin={basin_text} | 40N={lat_results[40.0]['actual_lat']:.3f}°N | "
                    f"65N={lat_results[65.0]['actual_lat']:.3f}°N | "
                    f"depth={depth_m[depth_index].min():.0f}-{depth_m[depth_index].max():.0f}m"
                )

            finally:
                ds.close()

        # %% 4. 三阶段纬度一致性

        for target_lat in TARGET_LATS:
            lats = np.asarray([stage_results[exp]["lat"][target_lat]["actual_lat"] for exp in ("pi", "up", "cdr")], float)

            if np.ptp(lats) > 1.0:
                raise ValueError(f"{model} {target_lat:.0f}N三阶段实际纬度不一致={lats.tolist()}")

        # %% 5. piControl最长连续共同有效块

        pi40_year = stage_results["pi"]["lat"][40.0]["source_year"]
        pi65_year = stage_results["pi"]["lat"][65.0]["source_year"]
        pi40 = stage_results["pi"]["lat"][40.0]["values"]
        pi65 = stage_results["pi"]["lat"][65.0]["values"]

        if not np.array_equal(pi40_year, pi65_year):
            raise ValueError(f"{model} piControl 40N与65N年份不一致")

        valid = np.isfinite(pi40) & np.isfinite(pi65)
        valid_index = np.flatnonzero(valid)
        valid_years = pi40_year[valid_index]

        if valid_years.size < MIN_PI_YEARS:
            raise ValueError(f"{model} piControl共同有效年份只有{valid_years.size}")

        breaks = np.where(np.diff(valid_years) != 1)[0]+1
        blocks = np.split(valid_index, breaks)
        eligible = [block for block in blocks if block.size >= MIN_PI_YEARS]

        if not eligible:
            raise ValueError(f"{model} piControl无连续{MIN_PI_YEARS}年")

        block = eligible[int(np.argmax([item.size for item in eligible]))]
        ref_years = pi40_year[block]
        pi40_ref_series = pi40[block]
        pi65_ref_series = pi65[block]
        li_ref_series = pi40_ref_series-pi65_ref_series

        pi40_mean = float(np.mean(pi40_ref_series))
        pi65_mean = float(np.mean(pi65_ref_series))
        li_mean = float(np.mean(li_ref_series))

        # %% 6. forced raw与pi-relative anomaly

        up40 = stage_results["up"]["lat"][40.0]["values"]
        up65 = stage_results["up"]["lat"][65.0]["values"]
        cdr40 = stage_results["cdr"]["lat"][40.0]["values"]
        cdr65 = stage_results["cdr"]["lat"][65.0]["values"]

        raw40 = np.r_[up40, cdr40]
        raw65 = np.r_[up65, cdr65]
        raw_li = raw40-raw65

        anomaly40 = raw40-pi40_mean
        anomaly65 = raw65-pi65_mean
        anomaly_li = raw_li-li_mean

        if not np.allclose(anomaly_li, anomaly40-anomaly65, atol=1e-10):
            raise RuntimeError(f"{model} 40N−65N anomaly内部一致性失败")

        simulation_year = np.arange(1, 281)
        experiment = np.where(simulation_year <= 140, "1pctCO2", "1pctCO2-cdr")
        phase_year = np.where(simulation_year <= 140, simulation_year, simulation_year-140)

        for i in range(280):
            annual_rows.append({
                "model": model,
                "simulation_year": int(simulation_year[i]),
                "experiment": experiment[i],
                "phase_year": int(phase_year[i]),
                "amoc_40N_raw_sv": float(raw40[i]),
                "amoc_65N_raw_sv": float(raw65[i]),
                "labrador_irminger_proxy_raw_sv": float(raw_li[i]),
                "amoc_40N_anomaly_sv": float(anomaly40[i]),
                "amoc_65N_anomaly_sv": float(anomaly65[i]),
                "labrador_irminger_proxy_anomaly_sv": float(anomaly_li[i]),
            })

        pi_rows.append({
            "model": model,
            "pi_reference_start_year": int(ref_years[0]),
            "pi_reference_end_year": int(ref_years[-1]),
            "pi_reference_year_count": int(ref_years.size),
            "pi_40N_mean_sv": pi40_mean,
            "pi_65N_mean_sv": pi65_mean,
            "pi_40N_minus_65N_mean_sv": li_mean,
            "pi_40N_sd_sv": float(np.std(pi40_ref_series, ddof=1)),
            "pi_65N_sd_sv": float(np.std(pi65_ref_series, ddof=1)),
            "pi_40N_minus_65N_sd_sv": float(np.std(li_ref_series, ddof=1)),
        })

        for exp in ("pi", "up", "cdr"):
            info = stage_results[exp]

            qa_rows.append({
                "model": model,
                "experiment": exp,
                "member": info["member"],
                "variant_label": info["variant_label"],
                "parent_variant_label": info["parent_variant"],
                "basin_index": info["basin_index"],
                "basin_label": info["basin_label"],
                "component": info["component"],
                "latitude_source": info["latitude_source"],
                "actual_40N_degN": info["lat"][40.0]["actual_lat"],
                "actual_65N_degN": info["lat"][65.0]["actual_lat"],
                "depth_min_m": info["depth_min"],
                "depth_max_m": info["depth_max"],
                "orientation_40N": info["lat"][40.0]["orientation"],
                "orientation_65N": info["lat"][65.0]["orientation"],
                "unit_conversion": info["unit_conversion"],
                "mixed_forced_branch": model == "NorESM2-LM" and exp in {"up", "cdr"},
            })

        successful.append(model)

        print(
            f"[OK] pi={ref_years[0]}-{ref_years[-1]} ({ref_years.size} yr) | "
            f"PI40={pi40_mean:.2f} | PI65={pi65_mean:.2f} | PI(40-65)={li_mean:+.2f} Sv"
        )
        print(
            f"     CDR131-140 anomalies: 40N={np.mean(anomaly40[270:280]):+.2f} | "
            f"65N={np.mean(anomaly65[270:280]):+.2f} | "
            f"LI={np.mean(anomaly_li[270:280]):+.2f} Sv"
        )

    except Exception as error:
        failure_rows.append({"model": model, "error": f"{type(error).__name__}: {error}"})
        print(f"[FAILED] {model}: {type(error).__name__}: {error}")


# %% 7. 保存

annual = pd.DataFrame(annual_rows)
pi_reference = pd.DataFrame(pi_rows)
qa = pd.DataFrame(qa_rows)
failures = pd.DataFrame(failure_rows, columns=["model", "error"])

annual.to_csv(ANNUAL_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")
pi_reference.to_csv(PI_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")
qa.to_csv(QA_FILE, index=False, encoding="utf-8-sig", float_format="%.8f")
failures.to_csv(FAIL_FILE, index=False, encoding="utf-8-sig")

metadata = {
    "models_requested": MODELS,
    "models_successful": successful,
    "basin": "Atlantic-Arctic",
    "depth_definition": "depth >= 300 m",
    "forced_sequence": "1pctCO2 years 1-140 + 1pctCO2-cdr years 1-140",
    "pi_reference": "mean of the longest continuous common-valid piControl block; minimum 100 yr",
    "drift_correction": "none",
    "diagnostics": {
        "40N": "total AMOC diagnostic",
        "65N": "GIN Seas high-latitude diagnostic",
        "40N_minus_65N": "Labrador-Irminger Sea-related proxy; not a strict transport budget",
    },
    "anomaly_definition": {
        "40N": "forced AMOC40 - piControl mean AMOC40",
        "65N": "forced AMOC65 - piControl mean AMOC65",
        "40N_minus_65N": "(forced40-forced65) - piControl mean(pi40-pi65)",
    },
    "time_smoothing": "none in calculation output",
}

META_FILE.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


# %% 8. 输出

print("\n"+"="*108)
print("AMOC PI-CONTROL RELATIVE ANOMALY COMPLETE")
print("="*108)
print(f"successful = {len(successful)}/{len(MODELS)}")
print(f"models = {successful}")
print(f"annual = {ANNUAL_FILE}")
print(f"pi reference = {PI_FILE}")
print(f"QA = {QA_FILE}")
print(f"failures = {FAIL_FILE}")
print("="*108)

if not successful:
    raise RuntimeError("没有模式成功，请检查failures.csv")