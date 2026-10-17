# -*- coding: utf-8 -*-
"""
计算8模式Nordic Seas和Labrador-Irminger Seas初始March 0-1000 m stratification垂向剖面。

【是否需要运行】
需要。

【输入文件】
F:/sea/cmipthetaoad
F:/sea/cmip6_so/1pctCO2
F:/sea/areacello
F:/sea/cmip6_so/NorESM-LM-thkcello

【输出文件】
output/s02sia/s02_arctic_region/p07_stratification/n01_initial_profiles/
    na_initial_march_stratification_profiles.csv
    na_initial_march_stratification_summary.csv
    na_initial_march_stratification_metadata.json

【什么时候需要重新运行】
thetao/so/thkcello/areacello、模式成员、区域、initial时间窗、月份、标准深度网格或N²算法变化后。

科学定义
1. 8模式：ACCESS-ESM1-5、CanESM5、CESM2、CNRM-ESM2-1、GFDL-ESM4、MIROC-ES2L、UKESM1-0-LL、NorESM2-LM。
2. initial = 1pctCO2实验years 1-10的March。
3. Nordic Seas = 65-80°N, 30°W-20°E。
4. Labrador-Irminger Seas = 50-65°N, 65°W-20°W。
5. thetao按0 dbar reference potential temperature处理，so按Practical Salinity SP处理。
6. 先将每个原生水柱的thetao和so线性插值到0-1000 m、10 m间隔的统一几何深度。
7. 只在原生有效深度范围内插值，不进行垂向外推。
8. 在统一深度上计算p、SA、CT，再用gsw.Nsquared计算N²。
9. 这样所有模式均在相同10 m垂向尺度上计算N²，避免原生极薄层放大垂向梯度。
10. 每个March先计算N²，再使用原生网格areacello做区域面积加权，最后平均10个March。
11. MIROC-ES2L使用CF ocean_sigma_z实际深度。
12. NorESM2-LM使用thkcello逐层累计重建layer-center depth。
13. 不做piControl订正、不做时间插值、不补缺失月份、不做时间平滑。
"""

# %% 1. 依赖
from pathlib import Path
import json
import re
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import gsw

warnings.filterwarnings("ignore", message=".*multiple fill values.*")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# %% 2. 路径与参数
ROOT = Path(r"F:\sea\Antic sea ice constraint")
SEA = Path(r"F:\sea")

THETAO_ROOT = SEA/"cmipthetaoad"
SO_ROOT = SEA/"cmip6_so"/"1pctCO2"
AREA_ROOT = SEA/"areacello"
NOR_THK_ROOT = SEA/"cmip6_so"/"NorESM-LM-thkcello"

OUT = ROOT/"output"/"s02sia"/"s02_arctic_region"/"p07_stratification"/"n01_initial_profiles"
OUT.mkdir(parents=True, exist_ok=True)

PROFILE_OUT = OUT/"na_initial_march_stratification_profiles.csv"
SUMMARY_OUT = OUT/"na_initial_march_stratification_summary.csv"
META_OUT = OUT/"na_initial_march_stratification_metadata.json"

for path in [THETAO_ROOT, SO_ROOT, AREA_ROOT, NOR_THK_ROOT]:
    if not path.is_dir():
        raise FileNotFoundError(path)

MODELS = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4", "MIROC-ES2L", "UKESM1-0-LL", "NorESM2-LM"]

ALIASES = {
    "ACCESS-ESM1-5": ["access-esm1-5", "accessesm5", "access"],
    "CanESM5": ["canesm5", "canesm"],
    "CESM2": ["cesm2"],
    "CNRM-ESM2-1": ["cnrm-esm2-1", "cnrm-esm2"],
    "GFDL-ESM4": ["gfdl-esm4", "gfdl"],
    "MIROC-ES2L": ["miroc-es2l", "miroc"],
    "UKESM1-0-LL": ["ukesm1-0-ll", "ukesm"],
    "NorESM2-LM": ["noresm2-lm", "noresm2-lm2", "noresm"],
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

EXPECTED_START_YEARS = {
    "ACCESS-ESM1-5": 101,
    "CanESM5": 1850,
    "CESM2": 1,
    "CNRM-ESM2-1": 1850,
    "GFDL-ESM4": 6,
    "MIROC-ES2L": 1850,
    "UKESM1-0-LL": 1850,
    "NorESM2-LM": 1,
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

NORDIC_LAT_MIN, NORDIC_LAT_MAX = 65.0, 80.0
NORDIC_LON_MIN, NORDIC_LON_MAX = -30.0, 20.0
LI_LAT_MIN, LI_LAT_MAX = 50.0, 65.0
LI_LON_MIN, LI_LON_MAX = -65.0, -20.0

WINDOW_YEARS = 10
STANDARD_DEPTH_M = np.arange(0.0, 1000.1, 10.0)
N2_DEPTH_M = 0.5*(STANDARD_DEPTH_M[:-1] + STANDARD_DEPTH_M[1:])
THK_MIN_M = 1e-6
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

# %% 3. 建立文件清单
THETAO_FILES_ALL = sorted(THETAO_ROOT.rglob("*.nc"))
SO_FILES_ALL = sorted(SO_ROOT.rglob("*.nc"))

thk_inventory_rows = []

for path in sorted(NOR_THK_ROOT.rglob("*.nc")):
    with xr.open_dataset(path, decode_times=False) as ds:
        if "thkcello" not in ds.variables:
            continue
        if str(ds.attrs.get("source_id", "")).strip() != "NorESM2-LM":
            continue

        thk_inventory_rows.append({
            "path": str(path),
            "experiment_id": str(ds.attrs.get("experiment_id", "")).strip(),
            "variant_label": str(ds.attrs.get("variant_label", "")).strip(),
            "grid_label": str(ds.attrs.get("grid_label", "")).strip(),
        })

thk_inventory = pd.DataFrame(thk_inventory_rows)

if thk_inventory.empty:
    raise FileNotFoundError(f"{NOR_THK_ROOT}中没有NorESM2-LM thkcello")

# %% 4. 逐模式计算
profile_rows = []
summary_rows = []
metadata_models = {}

for model in MODELS:
    member = MEMBERS[model]
    start_year = EXPECTED_START_YEARS[model]

    print("\n"+"="*120)
    print(f"[{model}] INITIAL MARCH STRATIFICATION")
    print("="*120)

    thetao_files = []
    so_files = []

    for path in THETAO_FILES_ALL:
        text = str(path).lower()
        name = path.name.lower()

        if (name.startswith("thetao_") or "_thetao_" in name) and member.lower() in text and any(alias in text for alias in ALIASES[model]) and "yearmean" not in name and "r360x180" not in name:
            thetao_files.append(path)

    for path in SO_FILES_ALL:
        text = str(path).lower()
        name = path.name.lower()

        if (name.startswith("so_") or "_so_" in name) and member.lower() in text and any(alias in text for alias in ALIASES[model]) and "yearmean" not in name and "r360x180" not in name:
            so_files.append(path)

    thetao_files = sorted(set(thetao_files))
    so_files = sorted(set(so_files))

    if not thetao_files:
        raise FileNotFoundError(f"{model}: 未找到thetao {member}")

    if not so_files:
        raise FileNotFoundError(f"{model}: 未找到so {member}")

    print(f"[1pctCO2] thetao files={len(thetao_files)} | so files={len(so_files)} | member={member}")

    tds = xr.open_mfdataset(thetao_files, combine="by_coords", decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override", join="exact")
    sds = xr.open_mfdataset(so_files, combine="by_coords", decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override", join="exact")

    if "thetao" not in tds or "so" not in sds:
        raise KeyError(f"{model}: thetao/so变量缺失")

    tda = tds["thetao"].squeeze(drop=True)
    sda = sds["so"].squeeze(drop=True)

    if "time" not in tda.dims or "time" not in sda.dims:
        raise ValueError(f"{model}: thetao/so缺少time维")

    if set(tda.dims) != set(sda.dims):
        raise ValueError(f"{model}: thetao/so dims不一致，T={tda.dims}, S={sda.dims}")

    target_years = start_year + np.arange(WINDOW_YEARS)
    target_year_month = target_years*100 + 3

    t_year_month = np.asarray(tda["time"].dt.year.values, dtype=int)*100 + np.asarray(tda["time"].dt.month.values, dtype=int)
    s_year_month = np.asarray(sda["time"].dt.year.values, dtype=int)*100 + np.asarray(sda["time"].dt.month.values, dtype=int)

    t_indices = []
    s_indices = []

    for year_month in target_year_month:
        t_match = np.flatnonzero(t_year_month == year_month)
        s_match = np.flatnonzero(s_year_month == year_month)

        if t_match.size != 1 or s_match.size != 1:
            raise ValueError(f"{model}: {year_month} thetao/so count异常，T={t_match.size}, S={s_match.size}")

        t_indices.append(int(t_match[0]))
        s_indices.append(int(s_match[0]))

    t_indices = np.asarray(t_indices, dtype=int)
    s_indices = np.asarray(s_indices, dtype=int)

    print(f"[initial] March={target_years[0]}-{target_years[-1]} | thetao/so aligned")

    # 识别垂向维
    z_candidates = []

    for dim in tda.dims:
        if dim == "time" or dim not in tds.variables:
            continue

        var = tds[dim]
        standard_name = str(var.attrs.get("standard_name", "")).lower()
        axis = str(var.attrs.get("axis", "")).upper()
        positive = str(var.attrs.get("positive", "")).lower()

        if axis == "Z" or positive in {"up", "down"} or "depth" in standard_name or "sigma" in standard_name or dim.lower() in {"lev", "olevel", "depth", "rho"}:
            z_candidates.append(dim)

    if len(z_candidates) != 1:
        raise ValueError(f"{model}: 无法唯一识别垂向维，candidates={z_candidates}, dims={tda.dims}")

    zdim = z_candidates[0]
    hdims = [dim for dim in tda.dims if dim not in {"time", zdim}]

    if len(hdims) != 2:
        raise ValueError(f"{model}: 水平维不是2维，dims={tda.dims}")

    # 经纬度
    lat_name = None
    lon_name = None

    for name in tds.variables:
        standard_name = str(tds[name].attrs.get("standard_name", "")).lower()

        if standard_name == "latitude" and lat_name is None:
            lat_name = name

        if standard_name == "longitude" and lon_name is None:
            lon_name = name

    if lat_name is None:
        for name in ["lat", "latitude", "nav_lat"]:
            if name in tds.variables:
                lat_name = name
                break

    if lon_name is None:
        for name in ["lon", "longitude", "nav_lon"]:
            if name in tds.variables:
                lon_name = name
                break

    if lat_name is None or lon_name is None:
        raise KeyError(f"{model}: 无法识别latitude/longitude")

    lat_da = tds[lat_name]
    lon_da = tds[lon_name]

    if "time" in lat_da.dims:
        lat_da = lat_da.isel(time=0)

    if "time" in lon_da.dims:
        lon_da = lon_da.isel(time=0)

    lat_da, lon_da = xr.broadcast(lat_da, lon_da)
    lat_da = lat_da.transpose(*hdims)
    lon_da = lon_da.transpose(*hdims)

    lat2_full = np.asarray(lat_da.values, dtype=float)
    lon2_full = (np.asarray(lon_da.values, dtype=float) + 180.0)%360.0 - 180.0

    nordic_full = np.isfinite(lat2_full) & np.isfinite(lon2_full) & (lat2_full >= NORDIC_LAT_MIN) & (lat2_full < NORDIC_LAT_MAX) & (lon2_full >= NORDIC_LON_MIN) & (lon2_full < NORDIC_LON_MAX)
    li_full = np.isfinite(lat2_full) & np.isfinite(lon2_full) & (lat2_full >= LI_LAT_MIN) & (lat2_full < LI_LAT_MAX) & (lon2_full >= LI_LON_MIN) & (lon2_full < LI_LON_MAX)
    domain_full = nordic_full | li_full

    yy, xx = np.where(domain_full)

    if yy.size == 0:
        raise ValueError(f"{model}: 两个目标海区均无网格")

    yslice = slice(int(yy.min()), int(yy.max()) + 1)
    xslice = slice(int(xx.min()), int(xx.max()) + 1)

    lat2 = lat2_full[yslice, xslice]
    lon2 = lon2_full[yslice, xslice]
    nordic = nordic_full[yslice, xslice]
    li = li_full[yslice, xslice]
    domain = domain_full[yslice, xslice]

    tsel = tda.isel(time=t_indices, **{hdims[0]: yslice, hdims[1]: xslice}).transpose("time", zdim, *hdims).load()
    ssel = sda.isel(time=s_indices, **{hdims[0]: yslice, hdims[1]: xslice}).transpose("time", zdim, *hdims).load()

    tvals = np.asarray(tsel.values, dtype=float)
    svals = np.asarray(ssel.values, dtype=float)

    if tvals.shape != svals.shape:
        raise ValueError(f"{model}: thetao/so shape不一致，T={tvals.shape}, S={svals.shape}")

    finite_t = tvals[np.isfinite(tvals)]
    finite_s = svals[np.isfinite(svals)]

    if finite_t.size == 0 or finite_s.size == 0:
        raise ValueError(f"{model}: thetao/so无有效值")

    if np.nanmin(finite_t) < -10.0 or np.nanmax(finite_t) > 60.0:
        raise ValueError(f"{model}: thetao范围异常")

    if np.nanmin(finite_s) < 0.0 or np.nanmax(finite_s) > 50.0:
        raise ValueError(f"{model}: so范围异常")

    nz = tvals.shape[1]

    # 原生物理深度
    vertical_method = "geometric_depth"

    if model == "MIROC-ES2L":
        zcoord = tds[zdim]
        standard_name = str(zcoord.attrs.get("standard_name", "")).strip()
        formula_terms = str(zcoord.attrs.get("formula_terms", "")).strip()

        if standard_name not in {"ocean_sigma_z_coordinate", "ocean_sigma_z"}:
            raise ValueError(f"MIROC-ES2L: standard_name={standard_name!r}")

        terms = dict(re.findall(r"([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)", formula_terms))
        required_terms = ["sigma", "eta", "depth", "depth_c", "nsigma", "zlev"]
        missing_terms = [key for key in required_terms if key not in terms]

        if missing_terms:
            raise KeyError(f"MIROC-ES2L: formula_terms缺少{missing_terms}")

        sigma = np.asarray(tds[terms["sigma"]].values, dtype=float).reshape(-1)
        zlev = np.asarray(tds[terms["zlev"]].values, dtype=float).reshape(-1)

        depth_c_da = tds[terms["depth_c"]]
        depth_c = float(np.asarray(depth_c_da.values).squeeze())
        depth_c_units = str(depth_c_da.attrs.get("units", "m")).lower()

        if depth_c_units in {"cm", "centimeter", "centimeters"}:
            depth_c /= 100.0
        elif depth_c_units not in {"m", "meter", "metre", "meters", "metres", ""}:
            raise ValueError(f"MIROC-ES2L: depth_c units={depth_c_units!r}")

        nsigma = int(np.asarray(tds[terms["nsigma"]].values).squeeze())

        bathy_da = tds[terms["depth"]]

        if "time" in bathy_da.dims:
            bathy_da = bathy_da.isel(time=0)

        bathy_da = bathy_da.transpose(*hdims).isel(**{hdims[0]: yslice, hdims[1]: xslice})
        bathy = np.asarray(bathy_da.values, dtype=float)
        bathy_units = str(bathy_da.attrs.get("units", "m")).lower()

        if bathy_units in {"cm", "centimeter", "centimeters"}:
            bathy /= 100.0
        elif bathy_units not in {"m", "meter", "metre", "meters", "metres", ""}:
            raise ValueError(f"MIROC-ES2L: depth units={bathy_units!r}")

        eta_da = tds[terms["eta"]]

        if "time" in eta_da.dims:
            eta_sel = eta_da.isel(time=t_indices, **{hdims[0]: yslice, hdims[1]: xslice}).transpose("time", *hdims).load()
            eta = np.asarray(eta_sel.values, dtype=float)
        else:
            eta_static = eta_da.transpose(*hdims).isel(**{hdims[0]: yslice, hdims[1]: xslice})
            eta = np.broadcast_to(np.asarray(eta_static.values, dtype=float)[None, :, :], (WINDOW_YEARS,) + bathy.shape).copy()

        eta_units = str(eta_da.attrs.get("units", "m")).lower()

        if eta_units in {"cm", "centimeter", "centimeters"}:
            eta /= 100.0
        elif eta_units not in {"m", "meter", "metre", "meters", "metres", ""}:
            raise ValueError(f"MIROC-ES2L: eta units={eta_units!r}")

        zlev_units = str(tds[terms["zlev"]].attrs.get("units", "m")).lower()

        if zlev_units in {"cm", "centimeter", "centimeters"}:
            zlev /= 100.0
        elif zlev_units not in {"m", "meter", "metre", "meters", "metres", ""}:
            raise ValueError(f"MIROC-ES2L: zlev units={zlev_units!r}")

        if nsigma < 0 or nsigma > nz:
            raise ValueError(f"MIROC-ES2L: nsigma={nsigma}, nz={nz}")

        depth4 = np.full(tvals.shape, np.nan, dtype=float)
        min_depth = np.minimum(depth_c, bathy)

        for k in range(nz):
            if k < nsigma:
                if k >= sigma.size or not np.isfinite(sigma[k]):
                    raise ValueError(f"MIROC-ES2L: sigma level {k}无效")

                z_height = eta + sigma[k]*(min_depth[None, :, :] + eta)
            else:
                if zlev.size == nz:
                    z_const = zlev[k]
                elif zlev.size == nz - nsigma:
                    z_const = zlev[k - nsigma]
                else:
                    raise ValueError(f"MIROC-ES2L: zlev size={zlev.size}, nz={nz}, nsigma={nsigma}")

                z_height = np.full_like(eta, z_const, dtype=float)

            depth4[:, k] = -z_height

        local_bottom = np.broadcast_to(bathy[None, None, :, :], depth4.shape)
        depth4 = np.where(np.isfinite(depth4) & (depth4 >= 0.0) & (depth4 <= local_bottom + 1e-6), depth4, np.nan)
        vertical_method = f"CF_sigma_over_z:{standard_name}"

    elif model == "NorESM2-LM":
        sub_thk = thk_inventory[(thk_inventory["variant_label"] == member) & (thk_inventory["experiment_id"] == "1pctCO2")].copy()

        if sub_thk.empty:
            raise FileNotFoundError(f"NorESM2-LM: 未找到1pctCO2 {member} thkcello")

        thk_files = [Path(path) for path in sub_thk["path"].tolist()]
        print(f"[Nor thkcello] files={len(thk_files)} | member={member}")

        hds = xr.open_mfdataset(thk_files, combine="by_coords", decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override", join="exact")
        hda = hds["thkcello"].squeeze(drop=True)

        hz_candidates = [dim for dim in hda.dims if dim != "time" and dim not in hdims]

        if len(hz_candidates) != 1:
            raise ValueError(f"NorESM2-LM: thkcello垂向维异常，dims={hda.dims}")

        hzdim = hz_candidates[0]

        if hda.sizes[hzdim] != nz:
            raise ValueError(f"NorESM2-LM: thkcello层数={hda.sizes[hzdim]}, thetao层数={nz}")

        h_year_month = np.asarray(hda["time"].dt.year.values, dtype=int)*100 + np.asarray(hda["time"].dt.month.values, dtype=int)
        h_indices = []

        for year_month in target_year_month:
            h_match = np.flatnonzero(h_year_month == year_month)

            if h_match.size != 1:
                raise ValueError(f"NorESM2-LM: thkcello {year_month} count={h_match.size}")

            h_indices.append(int(h_match[0]))

        hsel = hda.isel(time=np.asarray(h_indices, dtype=int), **{hdims[0]: yslice, hdims[1]: xslice}).transpose("time", hzdim, *hdims).load()
        thk = np.asarray(hsel.values, dtype=float)
        thk_units = str(hda.attrs.get("units", "m")).lower()

        if thk_units in {"cm", "centimeter", "centimeters"}:
            thk /= 100.0
        elif thk_units not in {"m", "meter", "metre", "meters", "metres", ""}:
            raise ValueError(f"NorESM2-LM: thkcello units={thk_units!r}")

        if thk.shape != tvals.shape:
            raise ValueError(f"NorESM2-LM: thkcello/T shape不一致")

        thk_valid = np.isfinite(thk) & (thk > THK_MIN_M)
        thk_work = np.where(thk_valid, thk, 0.0)
        depth4 = np.cumsum(thk_work, axis=1) - 0.5*thk_work
        depth4 = np.where(thk_valid, depth4, np.nan)

        hds.close()
        vertical_method = f"thkcello_layer_center:external files={len(thk_files)}"

    else:
        z_da = tds[zdim]
        zvals = np.asarray(z_da.values, dtype=float).reshape(-1)

        if zvals.size != nz:
            raise ValueError(f"{model}: vertical coord size={zvals.size}, nz={nz}")

        z_units = str(z_da.attrs.get("units", "")).strip().lower()

        if z_units in {"cm", "centimeter", "centimeters"}:
            zvals /= 100.0
        elif z_units not in {"m", "meter", "metre", "meters", "metres"}:
            raise ValueError(f"{model}: vertical units={z_units!r}")

        positive = str(z_da.attrs.get("positive", "down")).strip().lower()

        if positive == "up":
            zvals = -zvals
        elif positive != "down":
            raise ValueError(f"{model}: vertical positive={positive!r}")

        if np.any(~np.isfinite(zvals)) or np.any(zvals < 0.0) or np.any(np.diff(zvals) <= 0.0):
            raise ValueError(f"{model}: 几何深度不是严格递增非负深度")

        depth4 = np.broadcast_to(zvals[None, :, None, None], tvals.shape)

    print(f"[vertical] {vertical_method}")

    # 严格匹配areacello
    area_dir = AREA_ROOT/AREA_FOLDERS[model]

    if not area_dir.is_dir():
        raise FileNotFoundError(f"{model}: areacello目录不存在：{area_dir}")

    area_files = sorted(area_dir.rglob("*.nc"))
    matched_area = None
    matched_area_file = None
    matched_area_method = None

    for area_file in area_files:
        ads = xr.open_dataset(area_file, decode_times=False)

        if "areacello" not in ads.variables:
            ads.close()
            continue

        ada = ads["areacello"].squeeze(drop=True)

        if ada.ndim != 2:
            ads.close()
            continue

        area_lat_name = None
        area_lon_name = None

        for name in ads.variables:
            standard_name = str(ads[name].attrs.get("standard_name", "")).lower()

            if standard_name == "latitude" and area_lat_name is None:
                area_lat_name = name

            if standard_name == "longitude" and area_lon_name is None:
                area_lon_name = name

        if area_lat_name is None:
            for name in ["lat", "latitude", "nav_lat"]:
                if name in ads.variables:
                    area_lat_name = name
                    break

        if area_lon_name is None:
            for name in ["lon", "longitude", "nav_lon"]:
                if name in ads.variables:
                    area_lon_name = name
                    break

        if area_lat_name is None or area_lon_name is None:
            ads.close()
            continue

        area_lat_da = ads[area_lat_name]
        area_lon_da = ads[area_lon_name]
        area_lat_da, area_lon_da = xr.broadcast(area_lat_da, area_lon_da)

        try:
            area_lat_da = area_lat_da.transpose(*ada.dims)
            area_lon_da = area_lon_da.transpose(*ada.dims)
        except ValueError:
            ads.close()
            continue

        area_values = np.asarray(ada.values, dtype=float)
        area_lat = np.asarray(area_lat_da.values, dtype=float)
        area_lon = (np.asarray(area_lon_da.values, dtype=float) + 180.0)%360.0 - 180.0

        candidates = []

        if area_values.shape == lat2_full.shape:
            candidates.append((area_values, area_lat, area_lon, "native"))

        if area_values.T.shape == lat2_full.shape:
            candidates.append((area_values.T, area_lat.T, area_lon.T, "transpose"))

        for area_candidate, area_lat_candidate, area_lon_candidate, method in candidates:
            coord_valid = np.isfinite(area_lat_candidate) & np.isfinite(area_lon_candidate) & np.isfinite(lat2_full) & np.isfinite(lon2_full)

            if not np.any(coord_valid):
                continue

            lat_error = float(np.nanmax(np.abs(area_lat_candidate[coord_valid] - lat2_full[coord_valid])))
            lon_error = float(np.nanmax(np.abs((area_lon_candidate[coord_valid] - lon2_full[coord_valid] + 180.0)%360.0 - 180.0)))

            if lat_error <= 1e-5 and lon_error <= 1e-5:
                matched_area = area_candidate
                matched_area_file = area_file
                matched_area_method = method
                break

        ads.close()

        if matched_area is not None:
            break

    if matched_area is None:
        raise ValueError(f"{model}: 没有areacello与thetao原生经纬度严格匹配")

    area2 = matched_area[yslice, xslice]

    print(f"[area] {matched_area_method} | {matched_area_file.name}")

    # 展平目标海区
    mask_flat = domain.ravel()
    lat_flat = lat2.ravel()[mask_flat]
    lon_flat = lon2.ravel()[mask_flat]
    area_flat = area2.ravel()[mask_flat]
    nordic_flat = nordic.ravel()[mask_flat]
    li_flat = li.ravel()[mask_flat]

    region_masks = {
        "Nordic Seas": nordic_flat,
        "Labrador-Irminger Seas": li_flat,
    }

    n_cells = int(mask_flat.sum())
    n_depth = N2_DEPTH_M.size

    pressure_standard = gsw.p_from_z(-STANDARD_DEPTH_M[:, None], lat_flat[None, :])
    lat_standard = np.broadcast_to(lat_flat[None, :], pressure_standard.shape)
    lon_standard = np.broadcast_to(lon_flat[None, :], pressure_standard.shape)

    yearly_profiles = {
        "Nordic Seas": np.full((WINDOW_YEARS, n_depth), np.nan),
        "Labrador-Irminger Seas": np.full((WINDOW_YEARS, n_depth), np.nan),
    }

    yearly_coverage = {
        "Nordic Seas": np.full((WINDOW_YEARS, n_depth), np.nan),
        "Labrador-Irminger Seas": np.full((WINDOW_YEARS, n_depth), np.nan),
    }

    # 逐年：先T/S插值到统一深度，再算N²
    for iy in range(WINDOW_YEARS):
        pt_native = tvals[iy].reshape(nz, -1)[:, mask_flat]
        sp_native = svals[iy].reshape(nz, -1)[:, mask_flat]
        depth_native = depth4[iy].reshape(nz, -1)[:, mask_flat]

        pt_standard = np.full((STANDARD_DEPTH_M.size, n_cells), np.nan)
        sp_standard = np.full((STANDARD_DEPTH_M.size, n_cells), np.nan)

        for icell in range(n_cells):
            valid = np.isfinite(pt_native[:, icell]) & np.isfinite(sp_native[:, icell]) & np.isfinite(depth_native[:, icell]) & (depth_native[:, icell] >= 0.0) & (sp_native[:, icell] >= 0.0)

            if np.sum(valid) < 2:
                continue

            z = depth_native[valid, icell]
            pt = pt_native[valid, icell]
            sp = sp_native[valid, icell]

            order = np.argsort(z)
            z = z[order]
            pt = pt[order]
            sp = sp[order]

            unique = np.concatenate(([True], np.diff(z) > 1e-8))
            z = z[unique]
            pt = pt[unique]
            sp = sp[unique]

            if z.size < 2:
                continue

            pt_standard[:, icell] = np.interp(STANDARD_DEPTH_M, z, pt, left=np.nan, right=np.nan)
            sp_standard[:, icell] = np.interp(STANDARD_DEPTH_M, z, sp, left=np.nan, right=np.nan)

        valid_standard = np.isfinite(pt_standard) & np.isfinite(sp_standard) & np.isfinite(pressure_standard) & (sp_standard >= 0.0)

        SA = gsw.SA_from_SP(np.where(valid_standard, sp_standard, np.nan), pressure_standard, lon_standard, lat_standard)
        CT = gsw.CT_from_pt(SA, np.where(valid_standard, pt_standard, np.nan))
        N2, _ = gsw.Nsquared(SA, CT, pressure_standard, lat=lat_standard, axis=0)
        N2 = np.asarray(N2, dtype=float)

        for region, region_mask in region_masks.items():
            weights = area_flat[region_mask]
            values = N2[:, region_mask]

            region_total_area = float(np.nansum(np.where(np.isfinite(weights) & (weights > 0.0), weights, np.nan)))

            if region_total_area <= 0.0:
                raise ValueError(f"{model}: {region}没有有效面积")

            valid_values = np.isfinite(values) & np.isfinite(weights[None, :]) & (weights[None, :] > 0.0)
            valid_area = np.nansum(np.where(valid_values, weights[None, :], np.nan), axis=1)
            weighted_sum = np.nansum(np.where(valid_values, values*weights[None, :], np.nan), axis=1)

            good = valid_area > 0.0
            yearly_profiles[region][iy, good] = weighted_sum[good]/valid_area[good]
            yearly_coverage[region][iy, good] = valid_area[good]/region_total_area

        year_max = np.nanmax(np.abs(N2))
        print(f"[March {iy + 1:02d}/10] year={target_years[iy]} | max |N²|={year_max:.3e} s^-2")

    # 10年平均
    for region in ["Nordic Seas", "Labrador-Irminger Seas"]:
        valid_year_count = np.sum(np.isfinite(yearly_profiles[region]), axis=0)
        profile = np.full(n_depth, np.nan)
        profile_sd = np.full(n_depth, np.nan)
        coverage = np.nanmean(yearly_coverage[region], axis=0)

        complete = valid_year_count == WINDOW_YEARS
        profile[complete] = np.mean(yearly_profiles[region][:, complete], axis=0)
        profile_sd[complete] = np.std(yearly_profiles[region][:, complete], axis=0, ddof=1)

        for idepth, depth_m in enumerate(N2_DEPTH_M):
            profile_rows.append({
                "model": model,
                "region": region,
                "depth_m": float(depth_m),
                "initial_march_n2_s2": float(profile[idepth]) if np.isfinite(profile[idepth]) else np.nan,
                "initial_march_n2_1e5_s2": float(profile[idepth]*1e5) if np.isfinite(profile[idepth]) else np.nan,
                "interannual_sd_n2_s2": float(profile_sd[idepth]) if np.isfinite(profile_sd[idepth]) else np.nan,
                "mean_valid_area_fraction": float(coverage[idepth]) if np.isfinite(coverage[idepth]) else np.nan,
                "valid_march_count": int(valid_year_count[idepth]),
            })

        upper_mask = (N2_DEPTH_M <= 200.0) & np.isfinite(profile)
        full_mask = (N2_DEPTH_M <= 1000.0) & np.isfinite(profile)

        mean_200 = float(np.mean(profile[upper_mask])) if np.any(upper_mask) else np.nan
        mean_1000 = float(np.mean(profile[full_mask])) if np.any(full_mask) else np.nan
        max_abs = float(np.nanmax(np.abs(profile)))

        summary_rows.append({
            "model": model,
            "region": region,
            "initial_year_first": int(target_years[0]),
            "initial_year_last": int(target_years[-1]),
            "mean_n2_0_200m_s2": mean_200,
            "mean_n2_0_200m_1e5_s2": mean_200*1e5,
            "mean_n2_0_1000m_s2": mean_1000,
            "mean_n2_0_1000m_1e5_s2": mean_1000*1e5,
            "max_abs_profile_n2_s2": max_abs,
            "minimum_mean_valid_area_fraction": float(np.nanmin(coverage)),
            "vertical_method": vertical_method,
        })

        print(f"[{region}] mean N² 0-200 m={mean_200*1e5:.3f} ×10^-5 s^-2 | 0-1000 m={mean_1000*1e5:.3f} ×10^-5 s^-2 | max |profile|={max_abs:.3e}")

    metadata_models[model] = {
        "member": member,
        "initial_first_year": int(target_years[0]),
        "initial_last_year": int(target_years[-1]),
        "vertical_method": vertical_method,
        "thetao_files": len(thetao_files),
        "so_files": len(so_files),
        "areacello_file": str(matched_area_file),
        "areacello_matching": matched_area_method,
    }

    tds.close()
    sds.close()

# %% 5. MMM
profiles = pd.DataFrame(profile_rows)
summary = pd.DataFrame(summary_rows)
mmm_rows = []

for region in ["Nordic Seas", "Labrador-Irminger Seas"]:
    for depth_m in N2_DEPTH_M:
        values = profiles[(profiles["region"] == region) & np.isclose(profiles["depth_m"], depth_m)]["initial_march_n2_s2"].to_numpy(float)
        values = values[np.isfinite(values)]

        mmm_rows.append({
            "model": "MMM",
            "region": region,
            "depth_m": float(depth_m),
            "initial_march_n2_s2": float(np.mean(values)) if values.size else np.nan,
            "initial_march_n2_1e5_s2": float(np.mean(values)*1e5) if values.size else np.nan,
            "interannual_sd_n2_s2": np.nan,
            "mean_valid_area_fraction": np.nan,
            "valid_march_count": int(values.size),
        })

profiles = pd.concat([profiles, pd.DataFrame(mmm_rows)], ignore_index=True)

# %% 6. plot-ready统一横轴
finite_x = profiles.loc[profiles["model"] != "MMM", "initial_march_n2_1e5_s2"].to_numpy(float)
finite_x = finite_x[np.isfinite(finite_x)]

if finite_x.size == 0:
    raise RuntimeError("没有有效N²结果")

x_min_data = float(np.nanmin(finite_x))
x_max_data = float(np.nanmax(finite_x))
x_span = x_max_data - x_min_data
x_pad = max(x_span*0.06, 0.1)

x_min = float(np.floor((x_min_data - x_pad)*10.0)/10.0)
x_max = float(np.ceil((x_max_data + x_pad)*10.0)/10.0)

if x_min > 0.0:
    x_min = 0.0

if x_max <= x_min:
    raise RuntimeError(f"N²横轴范围异常：{x_min}, {x_max}")

# %% 7. 保存
profiles.to_csv(PROFILE_OUT, index=False, encoding="utf-8-sig", float_format="%.10g")
summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig", float_format="%.10g")

metadata = {
    "models": MODELS,
    "month": "March",
    "initial_definition": "1pctCO2 experiment years 1-10",
    "stratification_metric": "TEOS-10 Brunt-Vaisala frequency squared N2",
    "method": "thetao and so interpolated to common 0-1000 m geometric-depth grid before gsw.Nsquared",
    "standard_depth_levels_m": {
        "minimum": 0.0,
        "maximum": 1000.0,
        "interval": 10.0,
    },
    "n2_output_depth_m": {
        "minimum": float(N2_DEPTH_M[0]),
        "maximum": float(N2_DEPTH_M[-1]),
        "interval": 10.0,
    },
    "vertical_extrapolation": "none",
    "regional_weighting": "native-grid areacello",
    "nordic_seas": {
        "latitude": "65<=lat<80N",
        "longitude": "-30<=lon<20E",
    },
    "labrador_irminger_seas": {
        "latitude": "50<=lat<65N",
        "longitude": "-65<=lon<-20E",
    },
    "x_limits_n2_1e5_s2": [x_min, x_max],
    "y_limits_depth_m": [0.0, 1000.0],
    "models_metadata": metadata_models,
}

META_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

# %% 8. 完成
print("\n"+"="*120)
print("INITIAL NORTH ATLANTIC STRATIFICATION COMPLETE")
print("="*120)
print(summary.round(5).to_string(index=False))
print(f"\n[profiles] {PROFILE_OUT}")
print(f"[summary]  {SUMMARY_OUT}")
print(f"[metadata] {META_OUT}")
print(f"[plot x]   {x_min:.2f} to {x_max:.2f} ×10^-5 s^-2")
print("="*120)