# -*- coding: utf-8 -*-
"""
计算8模式两个固定North Atlantic深混合区域的逐年March MLD原值，并严格检查no-cross/bottom-limited影响。

【是否需要运行】
需要。会覆盖上一版p14_mld_fixed_regions_raw结果。

【输入文件】
F:/sea/cmipthetaoad
F:/sea/cmipthetao
F:/sea/cmip6_so/1pctCO2
F:/sea/cmip6_so/1pctCO2-cdr
NorESM2-LM additionally uses:
F:/sea/cmip6_so/NorESM-LM-thkcello

【输出文件】
output/s02sia/s02_arctic_region/p14_mld_fixed_regions_raw/
    na_march_mld_fixed_regions_annual.csv
    na_march_mld_fixed_regions_qa.csv
    na_march_mld_fixed_regions_co2.csv
    na_march_mld_fixed_regions_vertices.csv
    na_march_mld_fixed_regions_metadata.json

【什么时候需要重新运行】
原始thetao/so/thkcello、成员、MLD定义、固定polygon或QC定义变化后。

科学定义
1. 8模式：ACCESS-ESM1-5、CanESM5、CESM2、CNRM-ESM2-1、GFDL-ESM4、
   MIROC-ES2L、UKESM1-0-LL、NorESM2-LM；CAS-ESM2-0排除。
2. 1pctCO2 years1-140 + 1pctCO2-cdr years1-140，共280年。
3. 每年只取March。
4. MLD = 相对10 m reference sigma0增加0.03 kg m-3的首次向下crossing，
   并在相邻层之间线性插值。
5. 两区域为根据GFDL-ESM4 March MLD pattern人工确认的固定polygon。
6. 不使用MLD>300 m或500 m筛选区域。
7. 不做anomaly，不做piControl订正，不计算later-early，不计算MMM。
8. all MLD：若阈值直到最深有效层仍未cross，则记录为deepest valid depth，
   并显式标记为no_cross/bottom_limited。
9. crossing-only MLD：严格排除所有no_cross格点，只保留真实0.03 kg m-3 crossing。
10. 区域平均和百分位使用cos(latitude)权重。
11. resolved_mld_ge_500m_fraction和resolved_mld_ge_1000m_fraction：
    分子只包括真实crossing且MLD达到阈值的面积权重，分母为全部有效MLD面积权重。
12. 同时保存all与crossing-only两套11-year centered running mean。
13. Atmospheric CO2 = 285 ppm * 1.01^exponent；
    ramp-up exponent=1..140，ramp-down exponent=139..0。
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
from matplotlib.path import Path as MplPath

warnings.filterwarnings("ignore", message=".*multiple fill values.*")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# %% 2. 路径与参数
ROOT = Path(r"F:\sea\Antic sea ice constraint")
SEA = Path(r"F:\sea")

THETAO_ROOTS = {"1pctCO2": SEA/"cmipthetaoad", "1pctCO2-cdr": SEA/"cmipthetao"}
SO_ROOTS = {"1pctCO2": SEA/"cmip6_so"/"1pctCO2", "1pctCO2-cdr": SEA/"cmip6_so"/"1pctCO2-cdr"}

OUT = ROOT/"output"/"s02sia"/"s02_arctic_region"/"p14_mld_fixed_regions_raw"
OUT.mkdir(parents=True, exist_ok=True)

ANNUAL_OUT = OUT/"na_march_mld_fixed_regions_annual.csv"
QA_OUT = OUT/"na_march_mld_fixed_regions_qa.csv"
CO2_OUT = OUT/"na_march_mld_fixed_regions_co2.csv"
VERTEX_OUT = OUT/"na_march_mld_fixed_regions_vertices.csv"
META_OUT = OUT/"na_march_mld_fixed_regions_metadata.json"

for path in list(THETAO_ROOTS.values())+list(SO_ROOTS.values()):
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
    "ACCESS-ESM1-5": {"1pctCO2": "r1i1p1f1", "1pctCO2-cdr": "r1i1p1f1"},
    "CanESM5": {"1pctCO2": "r1i1p2f1", "1pctCO2-cdr": "r1i1p2f1"},
    "CESM2": {"1pctCO2": "r1i1p1f1", "1pctCO2-cdr": "r1i1p1f1"},
    "CNRM-ESM2-1": {"1pctCO2": "r1i1p1f2", "1pctCO2-cdr": "r1i1p1f2"},
    "GFDL-ESM4": {"1pctCO2": "r1i1p1f1", "1pctCO2-cdr": "r1i1p1f1"},
    "MIROC-ES2L": {"1pctCO2": "r1i1p1f2", "1pctCO2-cdr": "r1i1p1f2"},
    "UKESM1-0-LL": {"1pctCO2": "r1i1p1f2", "1pctCO2-cdr": "r1i1p1f2"},
    "NorESM2-LM": {"1pctCO2": "r1i1p1f1", "1pctCO2-cdr": "r1i1p1f1"},
}

EXPECTED_START_YEARS = {
    "ACCESS-ESM1-5": {"1pctCO2": 101, "1pctCO2-cdr": 241},
    "CanESM5": {"1pctCO2": 1850, "1pctCO2-cdr": 1991},
    "CESM2": {"1pctCO2": 1, "1pctCO2-cdr": 1},
    "CNRM-ESM2-1": {"1pctCO2": 1850, "1pctCO2-cdr": 1990},
    "GFDL-ESM4": {"1pctCO2": 6, "1pctCO2-cdr": 141},
    "MIROC-ES2L": {"1pctCO2": 1850, "1pctCO2-cdr": 1989},
    "UKESM1-0-LL": {"1pctCO2": 1850, "1pctCO2-cdr": 1990},
    "NorESM2-LM": {"1pctCO2": 1, "1pctCO2-cdr": 141},
}

REF_DEPTH_M = 10.0
DENSITY_THRESHOLD = 0.03
THK_MIN_M = 1e-6
SMOOTH_YEARS = 11
RAMP_YEARS = 140
TOTAL_YEARS = 280
CO2_BASELINE_PPM = 285.0
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)

# %% 3. 固定绿色polygon
LI_VERTICES = [
    (-60.0, 50.0),
    (-59.0, 54.0),
    (-55.0, 57.0),
    (-50.0, 59.0),
    (-45.0, 57.0),
    (-40.0, 56.0),
    (-35.0, 57.0),
    (-30.0, 59.0),
    (-26.0, 61.0),
    (-23.0, 59.0),
    (-22.0, 55.0),
    (-19.0, 53.0),
    (-20.0, 50.0),
    (-28.0, 49.0),
    (-38.0, 49.0),
    (-48.0, 49.0),
    (-56.0, 50.0),
    (-60.0, 50.0),
]

NORDIC_VERTICES = [
    (-20.0, 56.0),
    (-18.0, 60.0),
    (-17.0, 66.0),
    (-16.0, 72.0),
    (-14.0, 75.0),
    (-10.0, 76.0),
    (-5.0, 74.0),
    (2.0, 73.0),
    (8.0, 70.0),
    (10.0, 65.0),
    (9.0, 60.0),
    (6.0, 57.0),
    (1.0, 56.0),
    (-6.0, 56.0),
    (-13.0, 58.0),
    (-20.0, 56.0),
]

REGIONS = ["Labrador-Irminger", "Nordic Seas"]
LI_PATH = MplPath(np.asarray(LI_VERTICES, dtype=float))
NORDIC_PATH = MplPath(np.asarray(NORDIC_VERTICES, dtype=float))

LAT_MIN, LAT_MAX = 49.0, 77.0
LON_MIN, LON_MAX = -61.0, 11.0

# %% 4. 保存polygon顶点
vertex_rows = []

for order, vertex in enumerate(LI_VERTICES):
    vertex_rows.append({"region": "Labrador-Irminger", "vertex_order": order, "longitude_deg_east": vertex[0], "latitude_deg_north": vertex[1]})

for order, vertex in enumerate(NORDIC_VERTICES):
    vertex_rows.append({"region": "Nordic Seas", "vertex_order": order, "longitude_deg_east": vertex[0], "latitude_deg_north": vertex[1]})

pd.DataFrame(vertex_rows).to_csv(VERTEX_OUT, index=False, encoding="utf-8-sig")

# %% 5. CO2序列
simulation_year = np.arange(1, TOTAL_YEARS+1, dtype=int)
index_0based = simulation_year-1
co2_exponent = np.where(index_0based < RAMP_YEARS, index_0based+1, np.maximum(2*RAMP_YEARS-1-index_0based, 0))
co2_ppm = CO2_BASELINE_PPM*1.01**co2_exponent

co2 = pd.DataFrame({
    "simulation_year": simulation_year,
    "stage": np.where(simulation_year <= RAMP_YEARS, "ramp-up", "ramp-down"),
    "co2_exponent": co2_exponent,
    "co2_ppm": co2_ppm,
})

co2.to_csv(CO2_OUT, index=False, encoding="utf-8-sig", float_format="%.8f")

# %% 6. NorESM2-LM thkcello inventory
NOR_THK_ROOT = SEA/"cmip6_so"/"NorESM-LM-thkcello"

if not NOR_THK_ROOT.is_dir():
    raise FileNotFoundError(NOR_THK_ROOT)

thk_inventory = []

for path in sorted(NOR_THK_ROOT.rglob("*.nc")):
    with xr.open_dataset(path, decode_times=False) as ds:
        if "thkcello" not in ds.variables:
            continue
        if str(ds.attrs.get("source_id", "")).strip() != "NorESM2-LM":
            continue
        thk_inventory.append({
            "path": path,
            "experiment_id": str(ds.attrs.get("experiment_id", "")).strip(),
            "variant_label": str(ds.attrs.get("variant_label", "")).strip(),
            "grid_label": str(ds.attrs.get("grid_label", "")).strip(),
        })

thk_inventory = pd.DataFrame(thk_inventory)

if thk_inventory.empty:
    raise FileNotFoundError(f"{NOR_THK_ROOT}中没有NorESM2-LM thkcello")

# %% 7. 建立thetao和so文件目录
thetao_catalog = {experiment: sorted(root.rglob("*.nc")) for experiment, root in THETAO_ROOTS.items()}
so_catalog = {experiment: sorted(root.rglob("*.nc")) for experiment, root in SO_ROOTS.items()}

# %% 8. 逐模式逐实验计算
annual_rows = []
model_metadata = {}

for model in MODELS:
    print("\n"+"="*138)
    print(f"[{model}] FIXED-REGION ABSOLUTE MARCH MLD + STRICT NO-CROSS QC")
    print("="*138)

    model_metadata[model] = {}

    for experiment in ["1pctCO2", "1pctCO2-cdr"]:
        member = MEMBERS[model][experiment]
        thetao_files = []
        so_files = []

        for path in thetao_catalog[experiment]:
            text = str(path).lower()
            name = path.name.lower()
            if (name.startswith("thetao_") or "_thetao_" in name) and member.lower() in text and any(alias in text for alias in ALIASES[model]) and "yearmean" not in name and "r360x180" not in name:
                thetao_files.append(path)

        for path in so_catalog[experiment]:
            text = str(path).lower()
            name = path.name.lower()
            if (name.startswith("so_") or "_so_" in name) and member.lower() in text and any(alias in text for alias in ALIASES[model]) and "yearmean" not in name and "r360x180" not in name:
                so_files.append(path)

        thetao_files = sorted(set(thetao_files))
        so_files = sorted(set(so_files))

        if not thetao_files:
            raise FileNotFoundError(f"{model} {experiment}: 未找到thetao {member}")
        if not so_files:
            raise FileNotFoundError(f"{model} {experiment}: 未找到so {member}")

        print(f"[{experiment}] thetao={len(thetao_files)} files | so={len(so_files)} files | member={member}")

        tds = xr.open_mfdataset(thetao_files, combine="by_coords", decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override", join="exact")
        sds = xr.open_mfdataset(so_files, combine="by_coords", decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override", join="exact")

        if "thetao" not in tds or "so" not in sds:
            raise KeyError(f"{model} {experiment}: thetao/so变量缺失")

        tda = tds["thetao"].squeeze(drop=True)
        sda = sds["so"].squeeze(drop=True)

        if "time" not in tda.dims or "time" not in sda.dims:
            raise ValueError(f"{model} {experiment}: thetao/so缺少time维")

        if set(tda.dims) != set(sda.dims):
            raise ValueError(f"{model} {experiment}: thetao/so dims不一致，T={tda.dims}, S={sda.dims}")

        t_year = np.asarray(tda["time"].dt.year.values, dtype=int)
        t_month = np.asarray(tda["time"].dt.month.values, dtype=int)
        s_year = np.asarray(sda["time"].dt.year.values, dtype=int)
        s_month = np.asarray(sda["time"].dt.month.values, dtype=int)
        t_ym = t_year*100+t_month
        s_ym = s_year*100+s_month

        start_year = EXPECTED_START_YEARS[model][experiment]
        target_years = start_year+np.arange(RAMP_YEARS, dtype=int)
        target_ym = target_years*100+3
        t_indices = []
        s_indices = []

        for year_month in target_ym:
            t_match = np.flatnonzero(t_ym == year_month)
            s_match = np.flatnonzero(s_ym == year_month)

            if t_match.size != 1 or s_match.size != 1:
                raise ValueError(f"{model} {experiment}: {year_month} thetao count={t_match.size}, so count={s_match.size}")

            t_indices.append(int(t_match[0]))
            s_indices.append(int(s_match[0]))

        t_indices = np.asarray(t_indices, dtype=int)
        s_indices = np.asarray(s_indices, dtype=int)

        # 识别垂向维
        z_candidates = []

        for dim in tda.dims:
            if dim == "time":
                continue

            if dim in tds.variables:
                var = tds[dim]
                std = str(var.attrs.get("standard_name", "")).lower()
                axis = str(var.attrs.get("axis", "")).upper()
                positive = str(var.attrs.get("positive", "")).lower()

                if axis == "Z" or positive in {"up", "down"} or "depth" in std or "sigma" in std or dim.lower() in {"lev", "olevel", "depth", "rho"}:
                    z_candidates.append(dim)

        if len(z_candidates) != 1:
            raise ValueError(f"{model} {experiment}: 垂向维无法唯一识别，candidates={z_candidates}")

        zdim = z_candidates[0]
        hdims = [dim for dim in tda.dims if dim not in {"time", zdim}]

        if len(hdims) != 2:
            raise ValueError(f"{model} {experiment}: 水平维不是2维，dims={tda.dims}")

        # 经纬度
        lat_name = None
        lon_name = None

        for name in tds.variables:
            std = str(tds[name].attrs.get("standard_name", "")).lower()

            if std == "latitude" and lat_name is None:
                lat_name = name
            if std == "longitude" and lon_name is None:
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
            raise KeyError(f"{model} {experiment}: 无法识别latitude/longitude")

        lat_da = tds[lat_name]
        lon_da = tds[lon_name]

        if "time" in lat_da.dims:
            lat_da = lat_da.isel(time=0)
        if "time" in lon_da.dims:
            lon_da = lon_da.isel(time=0)

        lat_da, lon_da = xr.broadcast(lat_da, lon_da)
        lat_da = lat_da.transpose(*hdims)
        lon_da = lon_da.transpose(*hdims)

        lat_full = np.asarray(lat_da.values, dtype=float)
        lon_full = (np.asarray(lon_da.values, dtype=float)+180.0)%360.0-180.0

        crop_full = np.isfinite(lat_full) & np.isfinite(lon_full) & (lat_full >= LAT_MIN) & (lat_full <= LAT_MAX) & (lon_full >= LON_MIN) & (lon_full <= LON_MAX)
        yy, xx = np.where(crop_full)

        if yy.size == 0:
            raise ValueError(f"{model} {experiment}: 固定区域范围内没有网格")

        yslice = slice(int(yy.min()), int(yy.max())+1)
        xslice = slice(int(xx.min()), int(xx.max())+1)

        lat2 = lat_full[yslice, xslice]
        lon2 = lon_full[yslice, xslice]
        crop = crop_full[yslice, xslice]

        points = np.column_stack([lon2.ravel(), lat2.ravel()])
        li_mask = LI_PATH.contains_points(points, radius=1e-9).reshape(lat2.shape) & crop
        nordic_mask = NORDIC_PATH.contains_points(points, radius=1e-9).reshape(lat2.shape) & crop

        if not np.any(li_mask):
            raise ValueError(f"{model} {experiment}: Labrador-Irminger polygon没有网格")
        if not np.any(nordic_mask):
            raise ValueError(f"{model} {experiment}: Nordic Seas polygon没有网格")

        union_mask = li_mask | nordic_mask
        union_flat = union_mask.ravel()
        lat_union = lat2.ravel()[union_flat]
        lon_union = lon2.ravel()[union_flat]
        weight_union = np.cos(np.deg2rad(lat_union))
        li_selector = li_mask.ravel()[union_flat]
        nordic_selector = nordic_mask.ravel()[union_flat]

        nz = int(tda.sizes[zdim])
        vertical_method = "geometric_depth"
        zvals = None
        bathy = None
        sigma = None
        zlev = None
        depth_c = None
        nsigma = None
        eta_da = None
        eta_ym = None
        hds = None
        hda = None
        h_ym = None
        hzdim = None

        # MIROC sigma-over-z
        if model == "MIROC-ES2L":
            zcoord = tds[zdim]
            std = str(zcoord.attrs.get("standard_name", "")).strip()
            formula_terms = str(zcoord.attrs.get("formula_terms", "")).strip()
            accepted = {"ocean_sigma_z_coordinate", "ocean_sigma_z"}

            if std not in accepted:
                raise ValueError(f"MIROC {experiment}: standard_name={std!r}")

            terms = dict(re.findall(r"([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)", formula_terms))
            required_terms = ["sigma", "eta", "depth", "depth_c", "nsigma", "zlev"]
            missing_terms = [key for key in required_terms if key not in terms]

            if missing_terms:
                raise KeyError(f"MIROC {experiment}: formula_terms缺少{missing_terms}")

            sigma = np.asarray(tds[terms["sigma"]].values, dtype=float).reshape(-1)
            zlev = np.asarray(tds[terms["zlev"]].values, dtype=float).reshape(-1)

            depth_c_da = tds[terms["depth_c"]]
            depth_c = float(np.asarray(depth_c_da.values).squeeze())
            depth_c_units = str(depth_c_da.attrs.get("units", "m")).lower()

            if depth_c_units in {"cm", "centimeter", "centimeters"}:
                depth_c /= 100.0
            elif depth_c_units not in {"m", "meter", "metre", "meters", "metres", ""}:
                raise ValueError(f"MIROC {experiment}: depth_c units={depth_c_units!r}")

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
                raise ValueError(f"MIROC {experiment}: bathymetry units={bathy_units!r}")

            eta_da = tds[terms["eta"]]

            if "time" in eta_da.dims:
                eta_year = np.asarray(eta_da["time"].dt.year.values, dtype=int)
                eta_month = np.asarray(eta_da["time"].dt.month.values, dtype=int)
                eta_ym = eta_year*100+eta_month

            eta_units = str(eta_da.attrs.get("units", "m")).lower()

            if eta_units not in {"m", "meter", "metre", "meters", "metres", "", "cm", "centimeter", "centimeters"}:
                raise ValueError(f"MIROC {experiment}: eta units={eta_units!r}")

            zlev_units = str(tds[terms["zlev"]].attrs.get("units", "m")).lower()

            if zlev_units in {"cm", "centimeter", "centimeters"}:
                zlev /= 100.0
            elif zlev_units not in {"m", "meter", "metre", "meters", "metres", ""}:
                raise ValueError(f"MIROC {experiment}: zlev units={zlev_units!r}")

            if nsigma < 0 or nsigma > nz:
                raise ValueError(f"MIROC {experiment}: nsigma={nsigma}, nz={nz}")

            vertical_method = f"CF_sigma_over_z:{std}"

        # NorESM2-LM thkcello
        elif model == "NorESM2-LM":
            if "thkcello" in tds.variables:
                hds = tds
            elif "thkcello" in sds.variables:
                hds = sds
            else:
                sub_thk = thk_inventory[(thk_inventory["experiment_id"] == experiment) & (thk_inventory["variant_label"] == member)].copy()

                if sub_thk.empty:
                    raise FileNotFoundError(f"NorESM2-LM {experiment}: 没有匹配thkcello")

                thk_files = [Path(path) for path in sub_thk["path"].tolist()]
                hds = xr.open_mfdataset(thk_files, combine="by_coords", decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override", join="exact")
                print(f"[Nor thkcello] {experiment} | files={len(thk_files)}")

            hda = hds["thkcello"].squeeze(drop=True)
            hz_candidates = [dim for dim in hda.dims if dim != "time" and dim not in hdims]

            if len(hz_candidates) != 1:
                raise ValueError(f"NorESM2-LM {experiment}: thkcello垂向维无法识别，dims={hda.dims}")

            hzdim = hz_candidates[0]

            if hda.sizes[hzdim] != nz:
                raise ValueError(f"NorESM2-LM {experiment}: thkcello层数={hda.sizes[hzdim]}, thetao层数={nz}")

            if "time" in hda.dims:
                h_year = np.asarray(hda["time"].dt.year.values, dtype=int)
                h_month = np.asarray(hda["time"].dt.month.values, dtype=int)
                h_ym = h_year*100+h_month

            vertical_method = "thkcello_layer_center"

        # 其他模式几何深度
        else:
            z_da = tds[zdim]
            zvals = np.asarray(z_da.values, dtype=float).reshape(-1)

            if zvals.size != nz:
                raise ValueError(f"{model} {experiment}: depth size={zvals.size}, nz={nz}")

            z_units = str(z_da.attrs.get("units", "")).strip().lower()

            if z_units in {"cm", "centimeter", "centimeters"}:
                zvals /= 100.0
            elif z_units not in {"m", "meter", "metre", "meters", "metres"}:
                raise ValueError(f"{model} {experiment}: depth units={z_units!r}")

            positive = str(z_da.attrs.get("positive", "down")).strip().lower()

            if positive == "up":
                zvals = -zvals
            elif positive != "down":
                raise ValueError(f"{model} {experiment}: vertical positive={positive!r}")

            if np.any(~np.isfinite(zvals)) or np.any(zvals < 0.0) or np.any(np.diff(zvals) <= 0.0):
                raise ValueError(f"{model} {experiment}: 几何深度无效")

        model_metadata[model][experiment] = {
            "member": member,
            "calendar_start_year": int(start_year),
            "vertical_method": vertical_method,
            "li_native_cells": int(li_selector.sum()),
            "nordic_native_cells": int(nordic_selector.sum()),
        }

        # 逐年March诊断
        for offset in range(RAMP_YEARS):
            year_month = int(target_ym[offset])
            t_index = int(t_indices[offset])
            s_index = int(s_indices[offset])

            tsel = tda.isel(time=t_index, **{hdims[0]: yslice, hdims[1]: xslice}).transpose(zdim, *hdims).load()
            ssel = sda.isel(time=s_index, **{hdims[0]: yslice, hdims[1]: xslice}).transpose(zdim, *hdims).load()

            tvals = np.asarray(tsel.values, dtype=float)
            svals = np.asarray(ssel.values, dtype=float)

            if tvals.shape != svals.shape:
                raise ValueError(f"{model} {experiment} {year_month}: T/S shape不一致")

            # 当前March实际物理深度
            if model == "MIROC-ES2L":
                if "time" in eta_da.dims:
                    eta_match = np.flatnonzero(eta_ym == year_month)

                    if eta_match.size != 1:
                        raise ValueError(f"MIROC {experiment}: eta {year_month} count={eta_match.size}")

                    eta_sel = eta_da.isel(time=int(eta_match[0]), **{hdims[0]: yslice, hdims[1]: xslice}).transpose(*hdims).load()
                    eta = np.asarray(eta_sel.values, dtype=float)
                else:
                    eta_sel = eta_da.isel(**{hdims[0]: yslice, hdims[1]: xslice}).transpose(*hdims).load()
                    eta = np.asarray(eta_sel.values, dtype=float)

                eta_units = str(eta_da.attrs.get("units", "m")).lower()

                if eta_units in {"cm", "centimeter", "centimeters"}:
                    eta /= 100.0

                depth3 = np.full(tvals.shape, np.nan, dtype=float)
                min_depth = np.minimum(depth_c, bathy)

                for k in range(nz):
                    if k < nsigma:
                        if k >= sigma.size or not np.isfinite(sigma[k]):
                            raise ValueError(f"MIROC {experiment}: sigma level {k}无效")

                        z_height = eta+sigma[k]*(min_depth+eta)
                    else:
                        if zlev.size == nz:
                            z_const = zlev[k]
                        elif zlev.size == nz-nsigma:
                            z_const = zlev[k-nsigma]
                        else:
                            raise ValueError(f"MIROC {experiment}: zlev size={zlev.size}, nz={nz}, nsigma={nsigma}")

                        z_height = np.full_like(eta, z_const, dtype=float)

                    depth3[k] = -z_height

                local_bottom = np.broadcast_to(bathy[None, :, :], depth3.shape)
                depth3 = np.where(np.isfinite(depth3) & (depth3 >= 0.0) & (depth3 <= local_bottom+1e-6), depth3, np.nan)

            elif model == "NorESM2-LM":
                if "time" in hda.dims:
                    h_match = np.flatnonzero(h_ym == year_month)

                    if h_match.size != 1:
                        raise ValueError(f"NorESM2-LM {experiment}: thkcello {year_month} count={h_match.size}")

                    hsel = hda.isel(time=int(h_match[0]), **{hdims[0]: yslice, hdims[1]: xslice}).transpose(hzdim, *hdims).load()
                    thk = np.asarray(hsel.values, dtype=float)
                else:
                    hsel = hda.isel(**{hdims[0]: yslice, hdims[1]: xslice}).transpose(hzdim, *hdims).load()
                    thk = np.asarray(hsel.values, dtype=float)

                thk_units = str(hda.attrs.get("units", "m")).strip().lower()

                if thk_units in {"cm", "centimeter", "centimeters"}:
                    thk /= 100.0
                elif thk_units not in {"m", "meter", "metre", "meters", "metres", ""}:
                    raise ValueError(f"NorESM2-LM {experiment}: thkcello units={thk_units!r}")

                thk_valid = np.isfinite(thk) & (thk > THK_MIN_M)
                thk_work = np.where(thk_valid, thk, 0.0)
                depth3 = np.cumsum(thk_work, axis=0)-0.5*thk_work
                depth3 = np.where(thk_valid, depth3, np.nan)

            else:
                depth3 = np.broadcast_to(zvals[:, None, None], tvals.shape)

            # 两个polygon内格点
            pt = tvals.reshape(nz, -1)[:, union_flat]
            SP = svals.reshape(nz, -1)[:, union_flat]
            depth = depth3.reshape(nz, -1)[:, union_flat]

            finite_s = SP[np.isfinite(SP)]
            finite_t = pt[np.isfinite(pt)]

            if finite_s.size == 0 or finite_t.size == 0:
                raise ValueError(f"{model} {experiment} {year_month}: polygon内thetao/so无有效值")
            if np.nanmax(finite_s) > 50.0 or np.nanmin(finite_s) < 0.0:
                raise ValueError(f"{model} {experiment} {year_month}: so范围异常")
            if np.nanmax(finite_t) > 60.0 or np.nanmin(finite_t) < -10.0:
                raise ValueError(f"{model} {experiment} {year_month}: thetao范围异常")

            lat_b = lat_union[None, :]
            lon_b = lon_union[None, :]
            p = gsw.p_from_z(-depth, lat_b)

            valid_ts = np.isfinite(pt) & np.isfinite(SP) & np.isfinite(depth) & np.isfinite(p) & (SP >= 0.0)
            SA = gsw.SA_from_SP(np.where(valid_ts, SP, np.nan), p, lon_b, lat_b)
            CT = gsw.CT_from_pt(SA, np.where(valid_ts, pt, np.nan))
            sigma0 = gsw.sigma0(SA, CT)
            sigma0 = np.where(valid_ts & np.isfinite(SA) & np.isfinite(CT) & np.isfinite(sigma0), sigma0, np.nan)

            valid = np.isfinite(sigma0) & np.isfinite(depth) & (depth >= 0.0)
            lo_mask = valid & (depth <= REF_DEPTH_M)
            hi_mask = valid & (depth >= REF_DEPTH_M)
            has_ref = np.any(lo_mask, axis=0) & np.any(hi_mask, axis=0)

            ilo = np.argmax(np.where(lo_mask, depth, -np.inf), axis=0)
            ihi = np.argmin(np.where(hi_mask, depth, np.inf), axis=0)
            cols = np.arange(depth.shape[1])

            zlo = depth[ilo, cols]
            zhi = depth[ihi, cols]
            slo = sigma0[ilo, cols]
            shi = sigma0[ihi, cols]

            sigma_ref = np.full(depth.shape[1], np.nan, dtype=float)
            same = has_ref & np.isfinite(zlo) & np.isfinite(zhi) & (np.abs(zhi-zlo) <= 1e-10)
            interp = has_ref & ~same & np.isfinite(zlo) & np.isfinite(zhi) & np.isfinite(slo) & np.isfinite(shi) & (zhi > zlo)

            sigma_ref[same] = slo[same]
            sigma_ref[interp] = slo[interp]+(shi[interp]-slo[interp])*(REF_DEPTH_M-zlo[interp])/(zhi[interp]-zlo[interp])

            dsigma = sigma0-sigma_ref[None, :]
            cross_mask = valid & has_ref[None, :] & (depth >= REF_DEPTH_M) & np.isfinite(dsigma) & (dsigma >= DENSITY_THRESHOLD)
            has_cross = np.any(cross_mask, axis=0)
            icross = np.argmax(cross_mask, axis=0)

            z1 = depth[icross, cols]
            d1 = dsigma[icross, cols]

            prev_mask = valid & has_ref[None, :] & (depth >= REF_DEPTH_M) & (depth < z1[None, :]) & np.isfinite(dsigma) & (dsigma < DENSITY_THRESHOLD)
            has_prev = np.any(prev_mask, axis=0)
            iprev = np.argmax(np.where(prev_mask, depth, -np.inf), axis=0)

            z0 = np.full(depth.shape[1], REF_DEPTH_M, dtype=float)
            d0 = np.zeros(depth.shape[1], dtype=float)
            z0[has_prev] = depth[iprev[has_prev], cols[has_prev]]
            d0[has_prev] = dsigma[iprev[has_prev], cols[has_prev]]

            mld_crossing = np.full(depth.shape[1], np.nan, dtype=float)
            good_cross = has_cross & has_ref & np.isfinite(z0) & np.isfinite(z1) & np.isfinite(d0) & np.isfinite(d1) & (z1 >= z0) & (d1 > d0)
            mld_crossing[good_cross] = z0[good_cross]+(DENSITY_THRESHOLD-d0[good_cross])*(z1[good_cross]-z0[good_cross])/(d1[good_cross]-d0[good_cross])

            deepest = np.max(np.where(valid & (depth >= REF_DEPTH_M), depth, -np.inf), axis=0)
            no_cross = has_ref & ~has_cross & np.isfinite(deepest) & (deepest > -np.inf)

            mld_all = mld_crossing.copy()
            mld_all[no_cross] = deepest[no_cross]

            unresolved = ~np.isfinite(mld_all)
            simulation_year_value = offset+1 if experiment == "1pctCO2" else RAMP_YEARS+offset+1
            experiment_year = offset+1
            calendar_year = int(target_years[offset])

            # 区域QC
            for region_name, selector in [("Labrador-Irminger", li_selector), ("Nordic Seas", nordic_selector)]:
                region_all = mld_all[selector]
                region_crossing = mld_crossing[selector]
                region_has_cross = good_cross[selector]
                region_no_cross = no_cross[selector]
                region_unresolved = unresolved[selector]
                region_weight = weight_union[selector]

                valid_all = np.isfinite(region_all) & np.isfinite(region_weight) & (region_weight > 0.0)
                valid_crossing = np.isfinite(region_crossing) & region_has_cross & np.isfinite(region_weight) & (region_weight > 0.0)

                if not np.any(valid_all):
                    raise ValueError(f"{model} {experiment} year={experiment_year}: {region_name}没有有效MLD")

                all_mld = region_all[valid_all]
                all_weight = region_weight[valid_all]
                all_weight_sum = float(np.sum(all_weight))

                mean_all = float(np.sum(all_mld*all_weight)/all_weight_sum)
                max_all = float(np.max(all_mld))

                all_order = np.argsort(all_mld)
                all_sorted_mld = all_mld[all_order]
                all_sorted_weight = all_weight[all_order]
                all_cumulative = np.cumsum(all_sorted_weight)
                all_p90_index = int(np.searchsorted(all_cumulative, 0.90*all_cumulative[-1], side="left"))
                all_p90_index = min(all_p90_index, len(all_sorted_mld)-1)
                p90_all = float(all_sorted_mld[all_p90_index])

                if np.any(valid_crossing):
                    crossing_mld = region_crossing[valid_crossing]
                    crossing_weight = region_weight[valid_crossing]
                    crossing_weight_sum = float(np.sum(crossing_weight))
                    mean_crossing = float(np.sum(crossing_mld*crossing_weight)/crossing_weight_sum)
                    max_crossing = float(np.max(crossing_mld))

                    crossing_order = np.argsort(crossing_mld)
                    crossing_sorted_mld = crossing_mld[crossing_order]
                    crossing_sorted_weight = crossing_weight[crossing_order]
                    crossing_cumulative = np.cumsum(crossing_sorted_weight)
                    crossing_p90_index = int(np.searchsorted(crossing_cumulative, 0.90*crossing_cumulative[-1], side="left"))
                    crossing_p90_index = min(crossing_p90_index, len(crossing_sorted_mld)-1)
                    p90_crossing = float(crossing_sorted_mld[crossing_p90_index])
                else:
                    mean_crossing = np.nan
                    max_crossing = np.nan
                    p90_crossing = np.nan
                    crossing_weight_sum = 0.0

                no_cross_valid = region_no_cross & valid_all
                unresolved_valid = region_unresolved
                no_cross_count_fraction = float(np.sum(no_cross_valid)/np.sum(valid_all))
                no_cross_weight_fraction = float(np.sum(region_weight[no_cross_valid])/all_weight_sum)
                crossing_count_fraction = float(np.sum(valid_crossing)/np.sum(valid_all))
                crossing_weight_fraction = float(crossing_weight_sum/all_weight_sum)

                ge500_resolved = valid_crossing & (region_crossing >= 500.0)
                ge1000_resolved = valid_crossing & (region_crossing >= 1000.0)
                resolved_ge500_fraction = float(np.sum(region_weight[ge500_resolved])/all_weight_sum)
                resolved_ge1000_fraction = float(np.sum(region_weight[ge1000_resolved])/all_weight_sum)

                unresolved_fraction = float(np.sum(unresolved_valid)/len(region_all))
                mean_difference = float(mean_all-mean_crossing) if np.isfinite(mean_crossing) else np.nan

                if not np.isclose(no_cross_count_fraction+crossing_count_fraction, 1.0, atol=1e-10):
                    raise RuntimeError(f"{model} {experiment} year={experiment_year} {region_name}: crossing/no-cross count fractions不闭合")

                if not np.isclose(no_cross_weight_fraction+crossing_weight_fraction, 1.0, atol=1e-10):
                    raise RuntimeError(f"{model} {experiment} year={experiment_year} {region_name}: crossing/no-cross weight fractions不闭合")

                if np.isfinite(p90_crossing) and np.isfinite(max_crossing) and p90_crossing > max_crossing+1e-8:
                    raise RuntimeError(f"{model} {experiment} year={experiment_year} {region_name}: crossing P90 > max")

                if p90_all > max_all+1e-8:
                    raise RuntimeError(f"{model} {experiment} year={experiment_year} {region_name}: all P90 > max")

                annual_rows.append({
                    "model": model,
                    "region": region_name,
                    "simulation_year": int(simulation_year_value),
                    "experiment": experiment,
                    "experiment_year": int(experiment_year),
                    "calendar_year": calendar_year,
                    "march_mld_mean_all_m": mean_all,
                    "march_mld_mean_crossing_only_m": mean_crossing,
                    "march_mld_all_minus_crossing_mean_m": mean_difference,
                    "march_mld_p90_all_m": p90_all,
                    "march_mld_p90_crossing_only_m": p90_crossing,
                    "march_mld_max_all_m": max_all,
                    "march_mld_max_crossing_only_m": max_crossing,
                    "valid_cells": int(np.sum(valid_all)),
                    "crossing_cells": int(np.sum(valid_crossing)),
                    "no_cross_cells": int(np.sum(no_cross_valid)),
                    "unresolved_cells": int(np.sum(unresolved_valid)),
                    "region_native_cells": int(selector.sum()),
                    "valid_fraction": float(np.sum(valid_all)/selector.sum()),
                    "unresolved_fraction": unresolved_fraction,
                    "crossing_fraction": crossing_count_fraction,
                    "no_cross_fraction": no_cross_count_fraction,
                    "crossing_weight_fraction": crossing_weight_fraction,
                    "no_cross_weight_fraction": no_cross_weight_fraction,
                    "resolved_mld_ge_500m_fraction": resolved_ge500_fraction,
                    "resolved_mld_ge_1000m_fraction": resolved_ge1000_fraction,
                })

            if experiment_year == 1 or experiment_year%20 == 0 or experiment_year == RAMP_YEARS:
                print(f"[{experiment}] year={experiment_year:3d}/140 | calendar={calendar_year} | simulation={simulation_year_value}")

        if hds is not None and hds is not tds and hds is not sds:
            hds.close()

        tds.close()
        sds.close()

# %% 9. 整理逐年结果
annual = pd.DataFrame(annual_rows)
annual["model"] = pd.Categorical(annual["model"], categories=MODELS, ordered=True)
annual["region"] = pd.Categorical(annual["region"], categories=REGIONS, ordered=True)
annual = annual.sort_values(["model", "region", "simulation_year"]).reset_index(drop=True)
annual["model"] = annual["model"].astype(str)
annual["region"] = annual["region"].astype(str)

expected_rows = len(MODELS)*len(REGIONS)*TOTAL_YEARS

if len(annual) != expected_rows:
    raise RuntimeError(f"输出行数={len(annual)}，期望={expected_rows}")

# %% 10. 加入CO2并计算两套11年平滑
co2_map = co2.set_index("simulation_year")["co2_ppm"]
annual["co2_ppm"] = annual["simulation_year"].map(co2_map)
annual["march_mld_mean_all_11yr_m"] = np.nan
annual["march_mld_mean_crossing_only_11yr_m"] = np.nan
annual["no_cross_weight_fraction_11yr"] = np.nan
annual["resolved_mld_ge_500m_fraction_11yr"] = np.nan
annual["resolved_mld_ge_1000m_fraction_11yr"] = np.nan

for model in MODELS:
    for region in REGIONS:
        mask = (annual["model"] == model) & (annual["region"] == region)
        sub = annual.loc[mask].sort_values("simulation_year")

        if len(sub) != TOTAL_YEARS or not np.array_equal(sub["simulation_year"].to_numpy(int), np.arange(1, TOTAL_YEARS+1)):
            raise RuntimeError(f"{model} {region}: simulation_year不完整")

        smooth_all = pd.Series(sub["march_mld_mean_all_m"].to_numpy(float)).rolling(SMOOTH_YEARS, center=True, min_periods=SMOOTH_YEARS).mean().to_numpy()
        smooth_crossing = pd.Series(sub["march_mld_mean_crossing_only_m"].to_numpy(float)).rolling(SMOOTH_YEARS, center=True, min_periods=SMOOTH_YEARS).mean().to_numpy()
        smooth_no_cross = pd.Series(sub["no_cross_weight_fraction"].to_numpy(float)).rolling(SMOOTH_YEARS, center=True, min_periods=SMOOTH_YEARS).mean().to_numpy()
        smooth_ge500 = pd.Series(sub["resolved_mld_ge_500m_fraction"].to_numpy(float)).rolling(SMOOTH_YEARS, center=True, min_periods=SMOOTH_YEARS).mean().to_numpy()
        smooth_ge1000 = pd.Series(sub["resolved_mld_ge_1000m_fraction"].to_numpy(float)).rolling(SMOOTH_YEARS, center=True, min_periods=SMOOTH_YEARS).mean().to_numpy()

        annual.loc[sub.index, "march_mld_mean_all_11yr_m"] = smooth_all
        annual.loc[sub.index, "march_mld_mean_crossing_only_11yr_m"] = smooth_crossing
        annual.loc[sub.index, "no_cross_weight_fraction_11yr"] = smooth_no_cross
        annual.loc[sub.index, "resolved_mld_ge_500m_fraction_11yr"] = smooth_ge500
        annual.loc[sub.index, "resolved_mld_ge_1000m_fraction_11yr"] = smooth_ge1000

# %% 11. 模式×区域QC汇总
qa_rows = []

for model in MODELS:
    for region in REGIONS:
        sub = annual[(annual["model"] == model) & (annual["region"] == region)].sort_values("simulation_year").copy()

        all_mean_idx = sub["march_mld_mean_all_m"].idxmax()
        crossing_mean_idx = sub["march_mld_mean_crossing_only_m"].idxmax()
        all_p90_idx = sub["march_mld_p90_all_m"].idxmax()
        crossing_p90_idx = sub["march_mld_p90_crossing_only_m"].idxmax()
        all_max_idx = sub["march_mld_max_all_m"].idxmax()
        crossing_max_idx = sub["march_mld_max_crossing_only_m"].idxmax()
        no_cross_idx = sub["no_cross_weight_fraction"].idxmax()
        ge500_idx = sub["resolved_mld_ge_500m_fraction"].idxmax()
        ge1000_idx = sub["resolved_mld_ge_1000m_fraction"].idxmax()
        difference_idx = sub["march_mld_all_minus_crossing_mean_m"].abs().idxmax()

        qa_rows.append({
            "model": model,
            "region": region,
            "mean_280yr_all_m": float(sub["march_mld_mean_all_m"].mean()),
            "mean_280yr_crossing_only_m": float(sub["march_mld_mean_crossing_only_m"].mean()),
            "max_annual_mean_all_m": float(sub.loc[all_mean_idx, "march_mld_mean_all_m"]),
            "max_annual_mean_all_year": int(sub.loc[all_mean_idx, "simulation_year"]),
            "max_annual_mean_crossing_only_m": float(sub.loc[crossing_mean_idx, "march_mld_mean_crossing_only_m"]),
            "max_annual_mean_crossing_only_year": int(sub.loc[crossing_mean_idx, "simulation_year"]),
            "max_p90_all_m": float(sub.loc[all_p90_idx, "march_mld_p90_all_m"]),
            "max_p90_all_year": int(sub.loc[all_p90_idx, "simulation_year"]),
            "max_p90_crossing_only_m": float(sub.loc[crossing_p90_idx, "march_mld_p90_crossing_only_m"]),
            "max_p90_crossing_only_year": int(sub.loc[crossing_p90_idx, "simulation_year"]),
            "max_gridcell_all_m": float(sub.loc[all_max_idx, "march_mld_max_all_m"]),
            "max_gridcell_all_year": int(sub.loc[all_max_idx, "simulation_year"]),
            "max_gridcell_crossing_only_m": float(sub.loc[crossing_max_idx, "march_mld_max_crossing_only_m"]),
            "max_gridcell_crossing_only_year": int(sub.loc[crossing_max_idx, "simulation_year"]),
            "max_no_cross_weight_fraction": float(sub.loc[no_cross_idx, "no_cross_weight_fraction"]),
            "max_no_cross_weight_fraction_year": int(sub.loc[no_cross_idx, "simulation_year"]),
            "max_resolved_ge_500m_fraction": float(sub.loc[ge500_idx, "resolved_mld_ge_500m_fraction"]),
            "max_resolved_ge_500m_fraction_year": int(sub.loc[ge500_idx, "simulation_year"]),
            "max_resolved_ge_1000m_fraction": float(sub.loc[ge1000_idx, "resolved_mld_ge_1000m_fraction"]),
            "max_resolved_ge_1000m_fraction_year": int(sub.loc[ge1000_idx, "simulation_year"]),
            "max_abs_all_minus_crossing_mean_m": float(abs(sub.loc[difference_idx, "march_mld_all_minus_crossing_mean_m"])),
            "max_abs_all_minus_crossing_mean_year": int(sub.loc[difference_idx, "simulation_year"]),
            "minimum_valid_fraction": float(sub["valid_fraction"].min()),
            "maximum_unresolved_fraction": float(sub["unresolved_fraction"].max()),
        })

qa = pd.DataFrame(qa_rows)

# %% 12. 全局严格QC
fraction_columns = [
    "valid_fraction",
    "unresolved_fraction",
    "crossing_fraction",
    "no_cross_fraction",
    "crossing_weight_fraction",
    "no_cross_weight_fraction",
    "resolved_mld_ge_500m_fraction",
    "resolved_mld_ge_1000m_fraction",
]

for column in fraction_columns:
    values = annual[column].to_numpy(float)
    if np.any(~np.isfinite(values)):
        raise RuntimeError(f"{column}含非有限值")
    if np.any(values < -1e-10) or np.any(values > 1.0+1e-10):
        raise RuntimeError(f"{column}超出[0,1]")

closure_count = annual["crossing_fraction"].to_numpy(float)+annual["no_cross_fraction"].to_numpy(float)
closure_weight = annual["crossing_weight_fraction"].to_numpy(float)+annual["no_cross_weight_fraction"].to_numpy(float)

if not np.allclose(closure_count, 1.0, atol=1e-10):
    raise RuntimeError("全局crossing/no-cross count fraction不闭合")

if not np.allclose(closure_weight, 1.0, atol=1e-10):
    raise RuntimeError("全局crossing/no-cross weight fraction不闭合")

if np.any(annual["resolved_mld_ge_1000m_fraction"].to_numpy(float) > annual["resolved_mld_ge_500m_fraction"].to_numpy(float)+1e-10):
    raise RuntimeError("存在resolved >=1000 m fraction大于resolved >=500 m fraction")

# %% 13. plot-ready范围
all_smooth = annual["march_mld_mean_all_11yr_m"].to_numpy(float)
crossing_smooth = annual["march_mld_mean_crossing_only_11yr_m"].to_numpy(float)
plot_values = np.concatenate([all_smooth[np.isfinite(all_smooth)], crossing_smooth[np.isfinite(crossing_smooth)]])

if plot_values.size == 0:
    raise RuntimeError("MLD 11-year序列没有有效值")

plot_y_max = float(np.ceil(np.nanmax(plot_values)*1.10/100.0)*100.0)
plot_y_max = max(plot_y_max, 500.0)

# %% 14. 保存
annual.to_csv(ANNUAL_OUT, index=False, encoding="utf-8-sig", float_format="%.8f")
qa.to_csv(QA_OUT, index=False, encoding="utf-8-sig", float_format="%.8f")

metadata = {
    "analysis": "Absolute March MLD with strict crossing versus bottom-limited QC in two fixed North Atlantic polygons",
    "models": MODELS,
    "regions": REGIONS,
    "region_basis": "Fixed polygons approximating the confirmed green outlines on the GFDL-ESM4 March MLD pattern",
    "labrador_irminger_vertices_lon_lat": LI_VERTICES,
    "nordic_seas_vertices_lon_lat": NORDIC_VERTICES,
    "forced_sequence": "1pctCO2 years1-140 followed by 1pctCO2-cdr years1-140",
    "mld_reference_depth_m": REF_DEPTH_M,
    "density_threshold_kg_m3": DENSITY_THRESHOLD,
    "density_method": "TEOS-10 sigma0",
    "all_mld_definition": "true threshold crossing depth; if no crossing before deepest valid model level, deepest valid depth retained and flagged as no_cross",
    "crossing_only_definition": "only cells with a resolved 0.03 kg m-3 density-threshold crossing",
    "regional_weighting": "cos(latitude) on native-grid cells inside fixed polygons",
    "resolved_500_fraction": "weight of true-crossing cells with MLD>=500 m divided by total valid MLD weight",
    "resolved_1000_fraction": "weight of true-crossing cells with MLD>=1000 m divided by total valid MLD weight",
    "mld_threshold_region_selection": "none",
    "anomaly": False,
    "piControl_correction": False,
    "later_early_calculation": False,
    "mmm": False,
    "smoothing": "11-year centered running mean; min_periods=11",
    "co2_baseline_ppm": CO2_BASELINE_PPM,
    "co2_formula": "285 * 1.01**exponent; exponent=1..140 during ramp-up and 139..0 during ramp-down",
    "plot_y_min_m": 0.0,
    "plot_y_max_m": plot_y_max,
    "co2_plot_y_min_ppm": 250.0,
    "co2_plot_y_max_ppm": 1200.0,
    "co2_plot_ticks_ppm": [285.0, 700.0, float(np.round(co2["co2_ppm"].max()))],
    "model_details": model_metadata,
}

META_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

# %% 15. 终端QC报告
report_columns = [
    "model",
    "region",
    "mean_280yr_all_m",
    "mean_280yr_crossing_only_m",
    "max_annual_mean_all_m",
    "max_annual_mean_crossing_only_m",
    "max_p90_all_m",
    "max_p90_crossing_only_m",
    "max_gridcell_all_m",
    "max_gridcell_crossing_only_m",
    "max_no_cross_weight_fraction",
    "max_resolved_ge_500m_fraction",
    "max_resolved_ge_1000m_fraction",
    "max_abs_all_minus_crossing_mean_m",
]

print("\n"+"="*172)
print("FIXED-REGION ABSOLUTE MARCH MLD STRICT QC COMPLETE")
print("="*172)
print(qa[report_columns].round(3).to_string(index=False))
print("-"*172)
print("重点检查：")
print("1. max_gridcell_all_m 与 max_gridcell_crossing_only_m：判断3000-4000 m是否主要来自no-cross/bottom-limited。")
print("2. mean_280yr_all_m 与 mean_280yr_crossing_only_m：判断bottom-limited格点对区域平均影响。")
print("3. max_no_cross_weight_fraction：判断某些年份有多少区域处于bottom-limited状态。")
print("4. max_resolved_ge_500m_fraction / ge_1000m_fraction：只统计真正发生density crossing的深混合面积。")
print("-"*172)
print(f"[annual]   {ANNUAL_OUT}")
print(f"[QA]       {QA_OUT}")
print(f"[CO2]      {CO2_OUT}")
print(f"[vertices] {VERTEX_OUT}")
print(f"[metadata] {META_OUT}")
print("="*172)