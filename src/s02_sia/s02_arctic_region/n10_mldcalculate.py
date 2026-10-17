# -*- coding: utf-8 -*-
"""
使用CMIP6月平均thetao+so诊断8模式North Atlantic March MLD的三个阶段变化。

【是否需要运行】
需要。本版使用TEOS-10密度：SP->SA、potential temperature->CT、sigma0(SA,CT)。

【输入文件】
F:\\sea\\cmipthetaoad
F:\\sea\\cmipthetao
F:\\sea\\cmip6_so\\1pctCO2
F:\\sea\\cmip6_so\\1pctCO2-cdr
NorESM2-LM使用本机已经存在的thkcello文件；代码自动根据文件metadata识别，不猜目录名。

【输出文件】
output/s02sia/s02_arctic_region/p06_mld/n02_na_deep_mixing_three_phase/
    na_march_mld_three_phase_cells.csv
    na_march_mld_three_phase_summary.csv
    na_march_mld_three_phase_metadata.json
    <model>_na_march_mld_three_phase_native.npz

【什么时候需要重新运行】
原始thetao/so/thkcello、成员、MLD定义、TEOS-10密度处理或时间窗变化后。

科学定义
1. 8模式：ACCESS-ESM1-5、CanESM5、CESM2、CNRM-ESM2-1、GFDL-ESM4、MIROC-ES2L、UKESM1-0-LL、NorESM2-LM。
2. CAS-ESM2-0排除。
3. 每个实验分别诊断years 1-10和years 131-140的逐年March MLD，再作10年平均。
4. ramp-up变化 = 1pctCO2 years131-140 minus years1-10。
5. ramp-down变化 = CDR years131-140 minus years1-10；对应总实验years271-280 minus 141-150。
6. later-early变化 = CDR years131-140 minus 1pctCO2 years1-10。
6. MLD = 相对10 m reference sigma0增加0.03 kg m-3的首次向下crossing，并在相邻层之间线性插值。
7. CMIP6 so按Practical Salinity SP处理。
8. CMIP6 thetao按0 dbar reference potential temperature处理。
9. p=gsw.p_from_z(-z,lat)。
10. SA=gsw.SA_from_SP(SP,p,lon,lat)。
11. CT=gsw.CT_from_pt(SA,thetao)。
12. sigma0=gsw.sigma0(SA,CT)。
13. MIROC-ES2L使用CF sigma-over-z实际深度并施加local-bottom mask。
14. NorESM2-LM使用thkcello逐层累计重建layer-center depth，thkcello<=1e-6 m作为collapsed layer排除。
15. thetao和so目标March必须具有完全相同的真实year-month标签，不按文件位置容忍错配。
16. North Atlantic绘图区聚焦50-85N、65W-20E，覆盖Labrador-Irminger Sea和Nordic Seas。
17. 不做时间插值、不补缺失月份、不做piControl订正。
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
THETAO_ROOTS = {"1pctCO2": SEA / "cmipthetaoad", "1pctCO2-cdr": SEA / "cmipthetao"}
SO_ROOTS = {"1pctCO2": SEA / "cmip6_so" / "1pctCO2", "1pctCO2-cdr": SEA / "cmip6_so" / "1pctCO2-cdr"}
OUT = ROOT / "output" / "s02sia" / "s02_arctic_region" / "p06_mld" / "n02_na_deep_mixing_three_phase"
OUT.mkdir(parents=True, exist_ok=True)
CELL_OUT = OUT / "na_march_mld_three_phase_cells.csv"
SUMMARY_OUT = OUT / "na_march_mld_three_phase_summary.csv"
META_OUT = OUT / "na_march_mld_three_phase_metadata.json"
for path in list(THETAO_ROOTS.values()) + list(SO_ROOTS.values()):
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
LAT_MIN, LAT_MAX = 50.0, 85.0
LON_MIN, LON_MAX = -65.0, 20.0
REF_DEPTH_M = 10.0
DENSITY_THRESHOLD = 0.03
THK_MIN_M = 1e-6
GRID_RES_DEG = 1.0
WINDOW_YEARS = 10
SELECTED_MARCHES = 2 * WINDOW_YEARS
TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)
# %% 3. 建立NorESM2-LM层厚目录
# 只扫描已知的thkcello目录，不再递归扫描整个F:\sea。
NOR_THK_ROOT = SEA / "cmip6_so" / "NorESM-LM-thkcello"
if not NOR_THK_ROOT.is_dir():
    raise FileNotFoundError(NOR_THK_ROOT)

thk_inventory = []
for path in sorted(NOR_THK_ROOT.rglob("*.nc")):
    with xr.open_dataset(path, decode_times=False) as ds:
        if "thkcello" not in ds.variables:
            continue
        source_id = str(ds.attrs.get("source_id", "")).strip()
        experiment_id = str(ds.attrs.get("experiment_id", "")).strip()
        variant_label = str(ds.attrs.get("variant_label", "")).strip()
        grid_label = str(ds.attrs.get("grid_label", "")).strip()
        if source_id != "NorESM2-LM":
            continue
        thk_inventory.append({
            "path": path,
            "source_id": source_id,
            "experiment_id": experiment_id,
            "variant_label": variant_label,
            "grid_label": grid_label,
            "ntime": int(ds["thkcello"].sizes.get("time", 0)),
        })

thk_inventory_df = pd.DataFrame(thk_inventory)
if thk_inventory_df.empty:
    raise FileNotFoundError(f"{NOR_THK_ROOT}中没有识别到NorESM2-LM thkcello")


# %% 4. 一次建立文件目录
# 原脚本会对每个模式重复扫描整个输入目录；这里每个实验只扫描一次。
thetao_file_catalog = {
    experiment: sorted(root.rglob("*.nc"))
    for experiment, root in THETAO_ROOTS.items()
}
so_file_catalog = {
    experiment: sorted(root.rglob("*.nc"))
    for experiment, root in SO_ROOTS.items()
}


# %% 5. 逐模式计算
all_cells = []
summary_rows = []
metadata_models = {}
for model in MODELS:
    print("\n" + "=" * 128)
    print(f"[{model}] TEOS-10 March MLD")
    print("=" * 128)
    stage = {}
    for experiment in ["1pctCO2", "1pctCO2-cdr"]:
        member = MEMBERS[model][experiment]
        thetao_files = []
        so_files = []
        for path in thetao_file_catalog[experiment]:
            text = str(path).lower()
            name = path.name.lower()
            if (name.startswith("thetao_") or "_thetao_" in name) and member.lower() in text and any(alias in text for alias in ALIASES[model]) and "yearmean" not in name and "r360x180" not in name:
                thetao_files.append(path)
        for path in so_file_catalog[experiment]:
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
        print(f"[{experiment}] thetao files={len(thetao_files)} | so files={len(so_files)} | member={member}")
        tds = xr.open_mfdataset(thetao_files, combine="by_coords", decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override", join="exact")
        sds = xr.open_mfdataset(so_files, combine="by_coords", decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override", join="exact")
        if "thetao" not in tds or "so" not in sds:
            raise KeyError(f"{model} {experiment}: thetao/so变量缺失")
        tda = tds["thetao"].squeeze(drop=True)
        sda = sds["so"].squeeze(drop=True)
        if "time" not in tda.dims or "time" not in sda.dims:
            raise ValueError(f"{model} {experiment}: thetao/so缺少time维")
        if int(tda["time"].dt.month.values[0]) != 1 or int(sda["time"].dt.month.values[0]) != 1:
            raise ValueError(f"{model} {experiment}: thetao/so不是January起始")
        t_all_years = np.asarray(tda["time"].dt.year.values, dtype=int)
        t_all_months = np.asarray(tda["time"].dt.month.values, dtype=int)
        s_all_years = np.asarray(sda["time"].dt.year.values, dtype=int)
        s_all_months = np.asarray(sda["time"].dt.month.values, dtype=int)
        # 使用各模式文件所采用的实验起始日历年。
        # GFDL从模式年6开始，因此其日历年6-15对应本实验相对years1-10。
        experiment_start_year = EXPECTED_START_YEARS[model][experiment]
        # 每个实验只读取前10年和后10年的March，共20个时次。
        # 一次完成读取与MLD诊断，避免为三个差值重复打开同一批大文件。
        initial_offsets = np.arange(0, 10, dtype=int)
        final_offsets = np.arange(130, 140, dtype=int)
        target_offsets = np.concatenate([initial_offsets, final_offsets])
        target_years = experiment_start_year + target_offsets
        target_year_month = target_years * 100 + 3
        t_all_year_month = t_all_years * 100 + t_all_months
        s_all_year_month = s_all_years * 100 + s_all_months
        t_target_indices = []
        s_target_indices = []
        for year_month in target_year_month:
            t_match = np.flatnonzero(t_all_year_month == year_month)
            s_match = np.flatnonzero(s_all_year_month == year_month)
            if t_match.size != 1 or s_match.size != 1:
                raise ValueError(
                    f"{model} {experiment}: 目标年月{year_month}必须在thetao和so中各出现一次，"
                    f"thetao count={t_match.size}，so count={s_match.size}"
                )
            t_target_indices.append(int(t_match[0]))
            s_target_indices.append(int(s_match[0]))
        t_target_indices = np.asarray(t_target_indices, dtype=int)
        s_target_indices = np.asarray(s_target_indices, dtype=int)
        print(
            f"[{experiment}] experiment start={experiment_start_year} | "
            f"initial March={target_years[0]}-{target_years[9]} | "
            f"final March={target_years[10]}-{target_years[-1]} | "
            "thetao/so year-month aligned"
        )
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
            raise ValueError(f"{model} {experiment}: 无法唯一识别thetao垂向维，candidates={z_candidates}, dims={tda.dims}")
        zdim = z_candidates[0]
        hdims = [dim for dim in tda.dims if dim not in {"time", zdim}]
        if len(hdims) != 2:
            raise ValueError(f"{model} {experiment}: thetao水平维不是2维，dims={tda.dims}")
        if set(sda.dims) != set(tda.dims):
            raise ValueError(f"{model} {experiment}: thetao/so dims不一致，T={tda.dims} S={sda.dims}")
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
        lat2_full = np.asarray(lat_da.values, dtype=float)
        lon2_full = ((np.asarray(lon_da.values, dtype=float) + 180.0) % 360.0) - 180.0
        domain_full = np.isfinite(lat2_full) & np.isfinite(lon2_full) & (lat2_full >= LAT_MIN) & (lat2_full < LAT_MAX) & (lon2_full >= LON_MIN) & (lon2_full < LON_MAX)
        yy, xx = np.where(domain_full)
        if yy.size == 0:
            raise ValueError(f"{model} {experiment}: North Atlantic没有网格")
        yslice = slice(int(yy.min()), int(yy.max()) + 1)
        xslice = slice(int(xx.min()), int(xx.max()) + 1)
        lat2 = lat2_full[yslice, xslice]
        lon2 = lon2_full[yslice, xslice]
        domain = domain_full[yslice, xslice]
        tsel = tda.isel(time=t_target_indices, **{hdims[0]: yslice, hdims[1]: xslice}).transpose("time", zdim, *hdims).load()
        ssel = sda.isel(time=s_target_indices, **{hdims[0]: yslice, hdims[1]: xslice}).transpose("time", zdim, *hdims).load()
        tvals = np.asarray(tsel.values, dtype=float)
        svals = np.asarray(ssel.values, dtype=float)
        if tvals.shape != svals.shape:
            raise ValueError(f"{model} {experiment}: thetao/so shape不一致，T={tvals.shape} S={svals.shape}")
        finite_s = svals[np.isfinite(svals)]
        finite_t = tvals[np.isfinite(tvals)]
        if finite_s.size == 0 or finite_t.size == 0:
            raise ValueError(f"{model} {experiment}: thetao/so没有有效值")
        if np.nanmax(finite_s) > 50.0 or np.nanmin(finite_s) < 0.0:
            raise ValueError(f"{model} {experiment}: so范围异常，min={np.nanmin(finite_s):.3f}, max={np.nanmax(finite_s):.3f}")
        if np.nanmax(finite_t) > 60.0 or np.nanmin(finite_t) < -10.0:
            raise ValueError(f"{model} {experiment}: thetao范围异常，min={np.nanmin(finite_t):.3f}, max={np.nanmax(finite_t):.3f}")
        nz = tvals.shape[1]
        vertical_method = "geometric_depth"
        if model == "MIROC-ES2L":
            zcoord = tds[zdim]
            std = str(zcoord.attrs.get("standard_name", "")).strip()
            formula_terms = str(zcoord.attrs.get("formula_terms", "")).strip()
            accepted_sigma_z_names = {"ocean_sigma_z_coordinate", "ocean_sigma_z"}
            if std not in accepted_sigma_z_names:
                raise ValueError(f"MIROC {experiment}: 无法确认sigma-over-z垂向坐标，standard_name={std!r}")
            terms = dict(re.findall(r"([A-Za-z0-9_]+)\s*:\s*([A-Za-z0-9_]+)", formula_terms))
            required_terms = ["sigma", "eta", "depth", "depth_c", "nsigma", "zlev"]
            missing_terms = [key for key in required_terms if key not in terms]
            if missing_terms:
                raise KeyError(f"MIROC {experiment}: sigma-z formula_terms缺少{missing_terms}；standard_name={std!r}；formula_terms={formula_terms!r}")
            print(f"[MIROC vertical] standard_name={std} | formula_terms=PASS")
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
                raise ValueError(f"MIROC {experiment}: depth units={bathy_units!r}")
            eta_da = tds[terms["eta"]]
            if "time" in eta_da.dims:
                eta_sel = eta_da.isel(time=t_target_indices, **{hdims[0]: yslice, hdims[1]: xslice}).transpose("time", *hdims).load()
                eta = np.asarray(eta_sel.values, dtype=float)
            else:
                eta_static = eta_da.transpose(*hdims).isel(**{hdims[0]: yslice, hdims[1]: xslice})
                eta = np.broadcast_to(
                    np.asarray(eta_static.values, dtype=float)[None, :, :],
                    (SELECTED_MARCHES,) + bathy.shape,
                ).copy()
            eta_units = str(eta_da.attrs.get("units", "m")).lower()
            if eta_units in {"cm", "centimeter", "centimeters"}:
                eta /= 100.0
            elif eta_units not in {"m", "meter", "metre", "meters", "metres", ""}:
                raise ValueError(f"MIROC {experiment}: eta units={eta_units!r}")
            zlev_units = str(tds[terms["zlev"]].attrs.get("units", "m")).lower()
            if zlev_units in {"cm", "centimeter", "centimeters"}:
                zlev /= 100.0
            elif zlev_units not in {"m", "meter", "metre", "meters", "metres", ""}:
                raise ValueError(f"MIROC {experiment}: zlev units={zlev_units!r}")
            if nsigma < 0 or nsigma > nz:
                raise ValueError(f"MIROC {experiment}: nsigma={nsigma}, nz={nz}")
            depth4 = np.full(tvals.shape, np.nan, dtype=float)
            min_depth = np.minimum(depth_c, bathy)
            for k in range(nz):
                if k < nsigma:
                    if k >= sigma.size or not np.isfinite(sigma[k]):
                        raise ValueError(f"MIROC {experiment}: sigma level {k}无效")
                    z_height = eta + sigma[k] * (min_depth[None, :, :] + eta)
                else:
                    if zlev.size == nz:
                        z_const = zlev[k]
                    elif zlev.size == nz - nsigma:
                        z_const = zlev[k - nsigma]
                    else:
                        raise ValueError(f"MIROC {experiment}: zlev size={zlev.size}, nz={nz}, nsigma={nsigma}")
                    z_height = np.full_like(eta, z_const, dtype=float)
                depth4[:, k, :, :] = -z_height
            local_bottom = np.broadcast_to(bathy[None, None, :, :], depth4.shape)
            depth4 = np.where(np.isfinite(depth4) & (depth4 >= 0.0) & (depth4 <= local_bottom + 1e-6), depth4, np.nan)
            vertical_method = f"CF_sigma_over_z:{std}"
        elif model == "NorESM2-LM":
            thk_source = ""
            hds = None
            if "thkcello" in tds.variables:
                hds = tds
                thk_source = "thetao dataset"
            elif "thkcello" in sds.variables:
                hds = sds
                thk_source = "so dataset"
            else:
                sub_thk = thk_inventory_df[
                    (thk_inventory_df["variant_label"].astype(str) == member)
                    & (thk_inventory_df["experiment_id"].astype(str) == experiment)
                ].copy()
                if sub_thk.empty:
                    available = thk_inventory_df[["experiment_id", "variant_label", "grid_label", "path"]].copy()
                    raise FileNotFoundError(
                        f"NorESM2-LM {experiment}: 已扫描F:\\sea但没有找到metadata同时满足 "
                        f"source_id=NorESM2-LM, experiment_id={experiment}, variant_label={member} 的thkcello。\n"
                        f"扫描到的NorESM2-LM thkcello为：\n{available.to_string(index=False)}"
                    )
                thk_files = [Path(path) for path in sub_thk["path"].tolist()]
                print(f"[Nor thkcello] {experiment} | files={len(thk_files)} | member={member}")
                print(f"[Nor thkcello first] {thk_files[0]}")
                print(f"[Nor thkcello last ] {thk_files[-1]}")
                hds = xr.open_mfdataset(thk_files, combine="by_coords", decode_times=TIME_CODER, data_vars="minimal", coords="minimal", compat="override", join="exact")
                thk_source = f"external files={len(thk_files)}"
            if "thkcello" not in hds.variables:
                raise KeyError(f"NorESM2-LM {experiment}: 选中的数据源没有thkcello变量")
            hda = hds["thkcello"].squeeze(drop=True)
            hz_candidates = [dim for dim in hda.dims if dim != "time" and dim not in hdims]
            if len(hz_candidates) != 1:
                raise ValueError(f"NorESM2-LM {experiment}: thkcello垂向维无法唯一识别，dims={hda.dims}, hdims={hdims}")
            hzdim = hz_candidates[0]
            if hda.sizes[hzdim] != nz:
                raise ValueError(f"NorESM2-LM {experiment}: thkcello垂向层数={hda.sizes[hzdim]}，thetao层数={nz}")
            if not all(dim in hda.dims for dim in hdims):
                raise ValueError(f"NorESM2-LM {experiment}: thkcello水平维与thetao不一致，thk={hda.dims}, thetao={tda.dims}")
            if "time" in hda.dims:
                h_all_years = np.asarray(hda["time"].dt.year.values, dtype=int)
                h_all_months = np.asarray(hda["time"].dt.month.values, dtype=int)
                h_all_year_month = h_all_years * 100 + h_all_months
                h_target_indices = []
                for year_month in target_year_month:
                    h_match = np.flatnonzero(h_all_year_month == year_month)
                    if h_match.size != 1:
                        raise ValueError(
                            f"NorESM2-LM {experiment}: 目标年月{year_month}必须在thkcello中出现一次，"
                            f"count={h_match.size}"
                        )
                    h_target_indices.append(int(h_match[0]))
                h_target_indices = np.asarray(h_target_indices, dtype=int)
                hsel = hda.isel(time=h_target_indices, **{hdims[0]: yslice, hdims[1]: xslice}).transpose("time", hzdim, *hdims).load()
                thk = np.asarray(hsel.values, dtype=float)
            else:
                hsel = hda.isel(**{hdims[0]: yslice, hdims[1]: xslice}).transpose(hzdim, *hdims).load()
                thk_static = np.asarray(hsel.values, dtype=float)
                thk = np.broadcast_to(thk_static[None, :, :, :], tvals.shape).copy()
            thk_units = str(hda.attrs.get("units", "m")).strip().lower()
            if thk_units in {"cm", "centimeter", "centimeters"}:
                thk /= 100.0
            elif thk_units not in {"m", "meter", "metre", "meters", "metres", ""}:
                raise ValueError(f"NorESM2-LM {experiment}: thkcello units={thk_units!r}")
            if thk.shape != tvals.shape:
                raise ValueError(f"NorESM2-LM {experiment}: thkcello/T shape不一致，thk={thk.shape}, T={tvals.shape}")
            thk_valid = np.isfinite(thk) & (thk > THK_MIN_M)
            thk_work = np.where(thk_valid, thk, 0.0)
            depth4 = np.cumsum(thk_work, axis=1) - 0.5 * thk_work
            depth4 = np.where(thk_valid, depth4, np.nan)
            if hds is not tds and hds is not sds:
                hds.close()
            vertical_method = f"thkcello_layer_center:{thk_source}"
            print(f"[Nor vertical] {experiment} | {vertical_method} | shape={depth4.shape}")
        else:
            z_da = tds[zdim]
            zvals = np.asarray(z_da.values, dtype=float).reshape(-1)
            if zvals.size != nz:
                raise ValueError(f"{model} {experiment}: vertical coord size={zvals.size}, nz={nz}")
            z_units = str(z_da.attrs.get("units", "")).strip().lower()
            if z_units in {"cm", "centimeter", "centimeters"}:
                zvals /= 100.0
            elif z_units not in {"m", "meter", "metre", "meters", "metres"}:
                raise ValueError(f"{model} {experiment}: 垂向单位无法确认={z_units!r}")
            positive = str(z_da.attrs.get("positive", "down")).strip().lower()
            if positive == "up":
                zvals = -zvals
            elif positive != "down":
                raise ValueError(f"{model} {experiment}: vertical positive={positive!r}")
            if np.any(~np.isfinite(zvals)) or np.any(zvals < 0.0) or np.any(np.diff(zvals) <= 0.0):
                raise ValueError(f"{model} {experiment}: 几何深度不是严格递增的非负深度")
            depth4 = np.broadcast_to(zvals[None, :, None, None], tvals.shape)
        mask_flat = domain.ravel()
        lat_flat = lat2.ravel()[mask_flat]
        lon_flat = lon2.ravel()[mask_flat]
        mld_years = np.full((SELECTED_MARCHES, int(mask_flat.sum())), np.nan, dtype=float)
        no_cross_years = np.zeros((SELECTED_MARCHES, int(mask_flat.sum())), dtype=bool)
        for iy in range(SELECTED_MARCHES):
            pt = tvals[iy].reshape(nz, -1)[:, mask_flat]
            SP = svals[iy].reshape(nz, -1)[:, mask_flat]
            depth = depth4[iy].reshape(nz, -1)[:, mask_flat]
            if np.any(np.isfinite(depth) & (depth < 0.0)):
                raise ValueError(f"{model} {experiment} year#{iy + 1}: 出现负depth")
            lat_b = lat_flat[None, :]
            lon_b = lon_flat[None, :]
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
            lo_pick = np.where(lo_mask, depth, -np.inf)
            hi_pick = np.where(hi_mask, depth, np.inf)
            ilo = np.argmax(lo_pick, axis=0)
            ihi = np.argmin(hi_pick, axis=0)
            cols = np.arange(depth.shape[1])
            zlo = depth[ilo, cols]
            zhi = depth[ihi, cols]
            slo = sigma0[ilo, cols]
            shi = sigma0[ihi, cols]
            sigma_ref = np.full(depth.shape[1], np.nan, dtype=float)
            same = has_ref & np.isfinite(zlo) & np.isfinite(zhi) & (np.abs(zhi - zlo) <= 1e-10)
            sigma_ref[same] = slo[same]
            interp = has_ref & ~same & np.isfinite(zlo) & np.isfinite(zhi) & np.isfinite(slo) & np.isfinite(shi) & (zhi > zlo)
            sigma_ref[interp] = slo[interp] + (shi[interp] - slo[interp]) * (REF_DEPTH_M - zlo[interp]) / (zhi[interp] - zlo[interp])
            dsigma = sigma0 - sigma_ref[None, :]
            cross_mask = valid & has_ref[None, :] & (depth >= REF_DEPTH_M) & np.isfinite(dsigma) & (dsigma >= DENSITY_THRESHOLD)
            has_cross = np.any(cross_mask, axis=0)
            icross = np.argmax(cross_mask, axis=0)
            z1 = depth[icross, cols]
            d1 = dsigma[icross, cols]
            prev_mask = valid & has_ref[None, :] & (depth >= REF_DEPTH_M) & (depth < z1[None, :]) & np.isfinite(dsigma) & (dsigma < DENSITY_THRESHOLD)
            has_prev = np.any(prev_mask, axis=0)
            prev_pick = np.where(prev_mask, depth, -np.inf)
            iprev = np.argmax(prev_pick, axis=0)
            z0 = np.full(depth.shape[1], REF_DEPTH_M, dtype=float)
            d0 = np.zeros(depth.shape[1], dtype=float)
            z0[has_prev] = depth[iprev[has_prev], cols[has_prev]]
            d0[has_prev] = dsigma[iprev[has_prev], cols[has_prev]]
            mld = np.full(depth.shape[1], np.nan, dtype=float)
            good_cross = has_cross & has_ref & np.isfinite(z0) & np.isfinite(z1) & np.isfinite(d0) & np.isfinite(d1) & (z1 >= z0) & (d1 > d0)
            mld[good_cross] = z0[good_cross] + (DENSITY_THRESHOLD - d0[good_cross]) * (z1[good_cross] - z0[good_cross]) / (d1[good_cross] - d0[good_cross])
            deepest = np.max(np.where(valid & (depth >= REF_DEPTH_M), depth, -np.inf), axis=0)
            no_cross = has_ref & ~has_cross & np.isfinite(deepest) & (deepest > -np.inf)
            mld[no_cross] = deepest[no_cross]
            no_cross_years[iy] = no_cross
            mld_years[iy] = mld
        # 先逐年诊断MLD，再分别对前10年和后10年取平均。
        # 不能先平均thetao/so再诊断，否则密度阈值的非线性会改变MLD结果。
        initial_years = mld_years[:WINDOW_YEARS]
        final_years = mld_years[WINDOW_YEARS:]
        initial_valid_years = np.sum(np.isfinite(initial_years), axis=0)
        final_valid_years = np.sum(np.isfinite(final_years), axis=0)
        initial_mean = np.full(int(mask_flat.sum()), np.nan, dtype=float)
        final_mean = np.full(int(mask_flat.sum()), np.nan, dtype=float)
        initial_good = initial_valid_years == WINDOW_YEARS
        final_good = final_valid_years == WINDOW_YEARS
        initial_mean[initial_good] = np.mean(initial_years[:, initial_good], axis=0)
        final_mean[final_good] = np.mean(final_years[:, final_good], axis=0)

        initial_field = np.full(domain.shape, np.nan, dtype=float)
        final_field = np.full(domain.shape, np.nan, dtype=float)
        initial_flat = initial_field.ravel()
        final_flat = final_field.ravel()
        initial_flat[mask_flat] = initial_mean
        final_flat[mask_flat] = final_mean
        initial_field = initial_flat.reshape(domain.shape)
        final_field = final_flat.reshape(domain.shape)
        stage[experiment] = {
            "lat": lat2,
            "lon": lon2,
            "domain": domain,
            "initial_mld": initial_field,
            "final_mld": final_field,
            "initial_target_march_first": int(target_year_month[0]),
            "initial_target_march_last": int(target_year_month[WINDOW_YEARS - 1]),
            "final_target_march_first": int(target_year_month[WINDOW_YEARS]),
            "final_target_march_last": int(target_year_month[-1]),
            "initial_no_cross_fraction": float(
                no_cross_years[:WINDOW_YEARS].sum() / no_cross_years[:WINDOW_YEARS].size
            ),
            "final_no_cross_fraction": float(
                no_cross_years[WINDOW_YEARS:].sum() / no_cross_years[WINDOW_YEARS:].size
            ),
            "vertical_method": vertical_method,
        }
        print(
            f"[{experiment}] March years={SELECTED_MARCHES} | "
            f"initial mean={np.nanmean(initial_field):.2f} m | "
            f"final mean={np.nanmean(final_field):.2f} m | "
            f"valid cells initial/final={np.sum(np.isfinite(initial_field))}/{np.sum(np.isfinite(final_field))} | "
            f"z={vertical_method}"
        )
        tds.close()
        sds.close()
    up = stage["1pctCO2"]
    cdr = stage["1pctCO2-cdr"]
    if up["lat"].shape != cdr["lat"].shape:
        raise ValueError(f"{model}: up/CDR裁剪网格shape不一致，up={up['lat'].shape}, cdr={cdr['lat'].shape}")
    max_dlat = float(np.nanmax(np.abs(up["lat"] - cdr["lat"])))
    dlon = np.abs(up["lon"] - cdr["lon"])
    max_dlon = float(np.nanmax(np.minimum(dlon, 360.0 - dlon)))
    if max_dlat > 1e-8 or max_dlon > 1e-8:
        raise ValueError(f"{model}: up/CDR网格坐标不一致，maxΔlat={max_dlat}, maxΔlon={max_dlon}")
    common = (
        up["domain"]
        & cdr["domain"]
        & np.isfinite(up["initial_mld"])
        & np.isfinite(up["final_mld"])
        & np.isfinite(cdr["initial_mld"])
        & np.isfinite(cdr["final_mld"])
    )
    if int(common.sum()) == 0:
        raise ValueError(f"{model}: 四个阶段没有共同有效MLD格点")

    up_initial_native = np.where(common, up["initial_mld"], np.nan)
    up_final_native = np.where(common, up["final_mld"], np.nan)
    down_initial_native = np.where(common, cdr["initial_mld"], np.nan)
    down_final_native = np.where(common, cdr["final_mld"], np.nan)
    delta_up_native = up_final_native - up_initial_native
    delta_down_native = down_final_native - down_initial_native
    delta_later_early_native = down_final_native - up_initial_native

    np.savez_compressed(
        OUT / f"{model}_na_march_mld_three_phase_native.npz",
        latitude_deg_north=up["lat"],
        longitude_deg_east=up["lon"],
        up_initial_march_mld_m=up_initial_native,
        up_final_march_mld_m=up_final_native,
        down_initial_march_mld_m=down_initial_native,
        down_final_march_mld_m=down_final_native,
        delta_up_march_mld_m=delta_up_native,
        delta_down_march_mld_m=delta_down_native,
        delta_later_early_march_mld_m=delta_later_early_native,
        common_valid_mask=common,
    )
    native_df = pd.DataFrame({
        "lat": up["lat"][common],
        "lon": up["lon"][common],
        "up_initial": up_initial_native[common],
        "up_final": up_final_native[common],
        "down_initial": down_initial_native[common],
        "down_final": down_final_native[common],
    })
    native_df["lat_bin"] = np.floor((native_df["lat"] - LAT_MIN) / GRID_RES_DEG).astype(int)
    native_df["lon_bin"] = np.floor((native_df["lon"] - LON_MIN) / GRID_RES_DEG).astype(int)
    native_df = native_df[
        (native_df["lat_bin"] >= 0)
        & (native_df["lat_bin"] < int((LAT_MAX - LAT_MIN) / GRID_RES_DEG))
        & (native_df["lon_bin"] >= 0)
        & (native_df["lon_bin"] < int((LON_MAX - LON_MIN) / GRID_RES_DEG))
    ].copy()
    native_df["w"] = np.cos(np.deg2rad(native_df["lat"]))
    native_df["up_initial_w"] = native_df["up_initial"] * native_df["w"]
    native_df["up_final_w"] = native_df["up_final"] * native_df["w"]
    native_df["down_initial_w"] = native_df["down_initial"] * native_df["w"]
    native_df["down_final_w"] = native_df["down_final"] * native_df["w"]
    grouped = native_df.groupby(["lat_bin", "lon_bin"], as_index=False).agg(
        w_sum=("w", "sum"),
        up_initial_w_sum=("up_initial_w", "sum"),
        up_final_w_sum=("up_final_w", "sum"),
        down_initial_w_sum=("down_initial_w", "sum"),
        down_final_w_sum=("down_final_w", "sum"),
        native_count=("up_initial", "size"),
    )
    grouped["latitude_deg_north"] = LAT_MIN + (grouped["lat_bin"] + 0.5) * GRID_RES_DEG
    grouped["longitude_deg_east"] = LON_MIN + (grouped["lon_bin"] + 0.5) * GRID_RES_DEG
    grouped["up_initial_march_mld_m"] = grouped["up_initial_w_sum"] / grouped["w_sum"]
    grouped["up_final_march_mld_m"] = grouped["up_final_w_sum"] / grouped["w_sum"]
    grouped["down_initial_march_mld_m"] = grouped["down_initial_w_sum"] / grouped["w_sum"]
    grouped["down_final_march_mld_m"] = grouped["down_final_w_sum"] / grouped["w_sum"]
    grouped["delta_up_march_mld_m"] = grouped["up_final_march_mld_m"] - grouped["up_initial_march_mld_m"]
    grouped["delta_down_march_mld_m"] = grouped["down_final_march_mld_m"] - grouped["down_initial_march_mld_m"]
    grouped["delta_later_early_march_mld_m"] = grouped["down_final_march_mld_m"] - grouped["up_initial_march_mld_m"]
    model_cells = grouped[
        [
            "latitude_deg_north",
            "longitude_deg_east",
            "up_initial_march_mld_m",
            "up_final_march_mld_m",
            "down_initial_march_mld_m",
            "down_final_march_mld_m",
            "delta_up_march_mld_m",
            "delta_down_march_mld_m",
            "delta_later_early_march_mld_m",
            "native_count",
        ]
    ].copy()
    model_cells.insert(0, "model", model)
    all_cells.append(model_cells)
    weights = np.cos(np.deg2rad(model_cells["latitude_deg_north"].to_numpy(dtype=float)))
    up_initial_values = model_cells["up_initial_march_mld_m"].to_numpy(dtype=float)
    up_final_values = model_cells["up_final_march_mld_m"].to_numpy(dtype=float)
    down_initial_values = model_cells["down_initial_march_mld_m"].to_numpy(dtype=float)
    down_final_values = model_cells["down_final_march_mld_m"].to_numpy(dtype=float)
    delta_up_values = model_cells["delta_up_march_mld_m"].to_numpy(dtype=float)
    delta_down_values = model_cells["delta_down_march_mld_m"].to_numpy(dtype=float)
    delta_later_early_values = model_cells["delta_later_early_march_mld_m"].to_numpy(dtype=float)
    weight_sum = np.sum(weights)
    summary_rows.append({
        "model": model,
        "up_initial_mean_m": float(np.sum(up_initial_values * weights) / weight_sum),
        "up_final_mean_m": float(np.sum(up_final_values * weights) / weight_sum),
        "down_initial_mean_m": float(np.sum(down_initial_values * weights) / weight_sum),
        "down_final_mean_m": float(np.sum(down_final_values * weights) / weight_sum),
        "delta_up_mean_m": float(np.sum(delta_up_values * weights) / weight_sum),
        "delta_down_mean_m": float(np.sum(delta_down_values * weights) / weight_sum),
        "delta_later_early_mean_m": float(np.sum(delta_later_early_values * weights) / weight_sum),
        "delta_up_min_m": float(np.nanmin(delta_up_values)),
        "delta_up_max_m": float(np.nanmax(delta_up_values)),
        "delta_down_min_m": float(np.nanmin(delta_down_values)),
        "delta_down_max_m": float(np.nanmax(delta_down_values)),
        "delta_later_early_min_m": float(np.nanmin(delta_later_early_values)),
        "delta_later_early_max_m": float(np.nanmax(delta_later_early_values)),
        "valid_cells": int(len(model_cells)),
        "up_initial_no_cross_fraction": up["initial_no_cross_fraction"],
        "up_final_no_cross_fraction": up["final_no_cross_fraction"],
        "down_initial_no_cross_fraction": cdr["initial_no_cross_fraction"],
        "down_final_no_cross_fraction": cdr["final_no_cross_fraction"],
    })
    metadata_models[model] = {
        "member_up": MEMBERS[model]["1pctCO2"],
        "member_cdr": MEMBERS[model]["1pctCO2-cdr"],
        "up_vertical_method": up["vertical_method"],
        "cdr_vertical_method": cdr["vertical_method"],
        "up_initial_target_march_first": up["initial_target_march_first"],
        "up_initial_target_march_last": up["initial_target_march_last"],
        "up_final_target_march_first": up["final_target_march_first"],
        "up_final_target_march_last": up["final_target_march_last"],
        "down_initial_target_march_first": cdr["initial_target_march_first"],
        "down_initial_target_march_last": cdr["initial_target_march_last"],
        "down_final_target_march_first": cdr["final_target_march_first"],
        "down_final_target_march_last": cdr["final_target_march_last"],
        "up_initial_no_cross_fraction": up["initial_no_cross_fraction"],
        "up_final_no_cross_fraction": up["final_no_cross_fraction"],
        "down_initial_no_cross_fraction": cdr["initial_no_cross_fraction"],
        "down_final_no_cross_fraction": cdr["final_no_cross_fraction"],
    }
    print(
        f"[RESULT] {model} | Δup={summary_rows[-1]['delta_up_mean_m']:+.2f} m | "
        f"Δdown={summary_rows[-1]['delta_down_mean_m']:+.2f} m | "
        f"Δlater-early={summary_rows[-1]['delta_later_early_mean_m']:+.2f} m | "
        f"cells={len(model_cells)}"
    )
# %% 6. 保存正式结果
cells = pd.concat(all_cells, ignore_index=True)
summary = pd.DataFrame(summary_rows)
cells["model"] = pd.Categorical(cells["model"], categories=MODELS, ordered=True)
cells = cells.sort_values(["model", "latitude_deg_north", "longitude_deg_east"]).reset_index(drop=True)
cells["model"] = cells["model"].astype(str)
summary["model"] = pd.Categorical(summary["model"], categories=MODELS, ordered=True)
summary = summary.sort_values("model").reset_index(drop=True)
summary["model"] = summary["model"].astype(str)
cells.to_csv(CELL_OUT, index=False, encoding="utf-8-sig")
summary.to_csv(SUMMARY_OUT, index=False, encoding="utf-8-sig")
metadata = {
    "analysis": "8-model North Atlantic March MLD diagnosed from monthly CMIP6 thetao and so",
    "density_method": "TEOS-10",
    "salinity_interpretation": "CMIP6 so treated as Practical Salinity SP",
    "temperature_interpretation": "CMIP6 thetao treated as potential temperature referenced to 0 dbar",
    "teos10_chain": [
        "p = gsw.p_from_z(-depth, latitude)",
        "SA = gsw.SA_from_SP(SP, p, longitude, latitude)",
        "CT = gsw.CT_from_pt(SA, thetao)",
        "sigma0 = gsw.sigma0(SA, CT)",
    ],
    "mld_reference_depth_m": REF_DEPTH_M,
    "density_threshold_kg_m3": DENSITY_THRESHOLD,
    "windows": {
        "up_initial": "1pctCO2 years1-10 March",
        "up_final": "1pctCO2 years131-140 March",
        "down_initial": "1pctCO2-cdr years1-10 March; combined simulation years141-150",
        "down_final": "1pctCO2-cdr years131-140 March; combined simulation years271-280",
    },
    "changes": {
        "ramp_up": "up_final minus up_initial",
        "ramp_down": "down_final minus down_initial",
        "later_early": "down_final minus up_initial",
    },
    "region": "50<=lat<85N, -65<=lon<20E",
    "regional_focus": "Labrador-Irminger Sea and Nordic Seas",
    "plot_ready_grid": "1-degree geographic bins after native-grid MLD diagnosis",
    "piControl_correction": False,
    "models": MODELS,
    "cas_excluded": True,
    "special_vertical_coordinates": {
        "MIROC-ES2L": "CF sigma-over-z; accepts ocean_sigma_z and ocean_sigma_z_coordinate; formula_terms required",
        "NorESM2-LM": "thkcello discovered from actual files/metadata; layer-center depths reconstructed cumulatively; thickness<=1e-6 m excluded",
    },
    "model_details": metadata_models,
}
META_OUT.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
# %% 7. 终端报告
print("\n" + "=" * 128)
print("8-MODEL TEOS-10 NORTH ATLANTIC MARCH MLD THREE-PHASE COMPLETE")
print("=" * 128)
print(summary.round(3).to_string(index=False))
print("-" * 128)
print(f"[cells]   {CELL_OUT}")
print(f"[summary] {SUMMARY_OUT}")
print(f"[meta]    {META_OUT}")
print("下一步运行plot/S02_sia/s02_arctic_region/n08_mld8modelpattern.py生成三张8模式拼图。")
print("=" * 128)
