# %% 1. 导入依赖

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import gsw


# %% 2. 参数与路径设置

root = Path(r"F:\sea\Antic sea ice constraint")
directory = root / "output/s02sia/s02_arctic_region/p14_mld_fixed_regions_raw"
prefix = "cesm2_coast_stabilization_20261013"
assert not any((directory / (prefix + suffix)).exists() for suffix in ["_annual.csv", "_summary.csv", "_metadata.json"])
temperature_directory = Path(r"F:\sea\cmipthetao\CESM2")
salinity_directory = Path(r"F:\sea\cmip6_so\1pctCO2-cdr\CESM2")
overturning_directories = {"ramp_up": Path(r"F:\sea\cmip6amocad\CESM2"), "cdr": Path(r"F:\sea\cmip6amoc\CESM2")}
regions = ["Labrador-Irminger", "Nordic Seas"]
threshold = 0.03
reference_depth = 10.0
warnings.filterwarnings("ignore", message=".*multiple fill values.*")
coder = xr.coders.CFDatetimeCoder(use_cftime=True)


# %% 3. 读取固定掩膜和已有MLD，核对原生维度

geometry = np.load(directory / "CESM2_1pctCO2-cdr_coast_regions_20261012_geometry.npz")
latitude = geometry["latitude"]
longitude = geometry["longitude"]
area = geometry["cell_area_m2"]
li_mask = geometry["li_mask"].astype(bool)
nordic_mask = geometry["nordic_mask"].astype(bool)
union = li_mask | nordic_mask
latitude_union = latitude[union]
longitude_union = longitude[union]
weight_union = area[union]
selectors = {"Labrador-Irminger": li_mask[union], "Nordic Seas": nordic_mask[union]}
existing = pd.read_csv(directory / "na_march_mld_coast_regions_20261012_regions_annual.csv")
existing = existing.loc[existing.model == "CESM2"]
mld_rows = []
file_metadata = []
temperature_files = sorted(temperature_directory.glob("thetao_Omon_CESM2_1pctCO2-cdr_r1i1p1f1_gn_*.nc"))


# %% 4. 重算CDR末期及恒定强迫阶段三月MLD

for temperature_path in temperature_files:
    salinity_path = salinity_directory / temperature_path.name.replace("thetao_", "so_", 1)
    with xr.open_dataset(temperature_path, decode_times=coder) as temperature, xr.open_dataset(salinity_path, decode_times=coder) as salinity:
        assert temperature.thetao.dims == ("time", "lev", "nlat", "nlon")
        assert salinity.so.dims == temperature.thetao.dims
        assert temperature.lev.attrs["units"] == "centimeters"
        np.testing.assert_array_equal(temperature.time.values, salinity.time.values)
        np.testing.assert_allclose(temperature.lev.values, salinity.lev.values)
        full_latitude = temperature.lat.values
        start_row = int(np.argmin(np.abs(full_latitude[:, 0] - latitude[0, 0])))
        y_slice = slice(start_row, start_row + latitude.shape[0])
        np.testing.assert_allclose(full_latitude[y_slice], latitude, atol=1e-5)
        native_longitude = (temperature.lon.values[y_slice] + 180) % 360 - 180
        np.testing.assert_allclose(native_longitude, longitude, atol=1e-5)
        np.testing.assert_allclose(salinity.lat.values[y_slice], latitude, atol=1e-5)
        salinity_longitude = (salinity.lon.values[y_slice] + 180) % 360 - 180
        np.testing.assert_allclose(salinity_longitude, longitude, atol=1e-5)
        depth_levels = temperature.lev.values.astype(float) / 100
        years = temperature.time.dt.year.values
        months = temperature.time.dt.month.values
        selected = np.flatnonzero((months == 3) & (years >= 131))
        file_metadata.append({"thetao": str(temperature_path), "so": str(salinity_path), "first_year": int(years.min()), "last_year": int(years.max()), "dimensions": ["time", "lev", "nlat", "nlon"], "depth_unit": "centimeters", "crop_rows": [y_slice.start, y_slice.stop]})
        for time_index in selected:
            year = int(years[time_index])
            pt = temperature.thetao.isel(time=int(time_index), nlat=y_slice).values.astype(float)[:, union]
            sp = salinity.so.isel(time=int(time_index), nlat=y_slice).values.astype(float)[:, union]
            depth = np.broadcast_to(depth_levels[:, None], pt.shape)
            pressure = gsw.p_from_z(-depth, latitude_union[None, :])
            valid_ts = np.isfinite(pt) & np.isfinite(sp) & (sp >= 0)
            absolute_salinity = gsw.SA_from_SP(np.where(valid_ts, sp, np.nan), pressure, longitude_union[None, :], latitude_union[None, :])
            conservative_temperature = gsw.CT_from_pt(absolute_salinity, np.where(valid_ts, pt, np.nan))
            density = gsw.sigma0(absolute_salinity, conservative_temperature)
            valid = np.isfinite(density) & np.isfinite(depth)
            low = valid & (depth <= reference_depth)
            high = valid & (depth >= reference_depth)
            has_reference = low.any(axis=0) & high.any(axis=0)
            low_index = np.argmax(np.where(low, depth, -np.inf), axis=0)
            high_index = np.argmin(np.where(high, depth, np.inf), axis=0)
            columns = np.arange(pt.shape[1])
            low_depth = depth[low_index, columns]
            high_depth = depth[high_index, columns]
            low_density = density[low_index, columns]
            high_density = density[high_index, columns]
            reference_density = np.full(pt.shape[1], np.nan)
            same = has_reference & (np.abs(high_depth - low_depth) <= 1e-10)
            interpolate = has_reference & ~same & (high_depth > low_depth)
            reference_density[same] = low_density[same]
            reference_density[interpolate] = low_density[interpolate] + (high_density[interpolate] - low_density[interpolate]) * (reference_depth - low_depth[interpolate]) / (high_depth[interpolate] - low_depth[interpolate])
            difference = density - reference_density[None, :]
            crossing = valid & has_reference[None, :] & (depth >= reference_depth) & (difference >= threshold)
            has_crossing = crossing.any(axis=0)
            crossing_index = np.argmax(crossing, axis=0)
            z1 = depth[crossing_index, columns]
            d1 = difference[crossing_index, columns]
            previous = valid & has_reference[None, :] & (depth >= reference_depth) & (depth < z1[None, :]) & (difference < threshold)
            has_previous = previous.any(axis=0)
            previous_index = np.argmax(np.where(previous, depth, -np.inf), axis=0)
            z0 = np.full(pt.shape[1], reference_depth)
            d0 = np.zeros(pt.shape[1])
            z0[has_previous] = depth[previous_index[has_previous], columns[has_previous]]
            d0[has_previous] = difference[previous_index[has_previous], columns[has_previous]]
            good = has_crossing & has_reference & np.isfinite(z1) & np.isfinite(d1) & (z1 >= z0) & (d1 > d0)
            mld = np.full(pt.shape[1], np.nan)
            mld[good] = z0[good] + (threshold - d0[good]) * (z1[good] - z0[good]) / (d1[good] - d0[good])
            deepest = np.max(np.where(valid & (depth >= reference_depth), depth, -np.inf), axis=0)
            no_cross = has_reference & ~has_crossing & np.isfinite(deepest)
            valid_all = good | no_cross
            for region in regions:
                selected_valid = selectors[region] & good
                selected_all = selectors[region] & valid_all
                regional_weights = weight_union[selected_valid]
                regional_mld = mld[selected_valid]
                assert len(regional_mld) > 0
                mean_mld = float(np.average(regional_mld, weights=regional_weights))
                sorted_indices = np.argsort(regional_mld)
                cumulative = np.cumsum(regional_weights[sorted_indices])
                p90 = float(regional_mld[sorted_indices[np.searchsorted(cumulative, 0.9 * cumulative[-1])]])
                all_weight = weight_union[selected_all].sum()
                mld_rows.append({"simulation_year": 140 + year, "region": region, "mld_mean_m": mean_mld, "mld_p90_m": p90, "deep_area_1000m_percent": 100 * weight_union[selectors[region] & good & (mld >= 1000)].sum() / all_weight, "no_cross_area_percent": 100 * weight_union[selectors[region] & no_cross].sum() / all_weight})
            print("March MLD", year, "done", flush=True)
extended_mld = pd.DataFrame(mld_rows)
overlap_errors = []
for row in extended_mld.loc[extended_mld.simulation_year <= 280].itertuples():
    old = existing.loc[(existing.region == row.region) & (existing.simulation_year == row.simulation_year)].iloc[0]
    overlap_errors.append(abs(row.mld_mean_m - old.march_mld_mean_crossing_only_m))
assert max(overlap_errors) < 1e-5


# %% 5. 从完整原始翻转剖面重算40N与65N

overturning_rows = []
latitude_records = []
for phase, folder in overturning_directories.items():
    experiment = "1pctCO2" if phase == "ramp_up" else "1pctCO2-cdr"
    profiles = {40: [], 65: []}
    days = []
    dates = []
    for path in sorted(folder.glob(f"msftmz_Omon_CESM2_{experiment}_r1i1p1f1_gn_*.nc")):
        with xr.open_dataset(path, decode_times=coder) as dataset:
            assert dataset.msftmz.dims == ("time", "basin", "lev", "lat")
            assert dataset.msftmz.attrs["units"] == "kg s-1"
            assert dataset.lev.attrs["units"] == "centimeters"
            years = dataset.time.dt.year.values
            keep = np.flatnonzero(years <= (140 if phase == "ramp_up" else 200))
            if len(keep) == 0:
                continue
            dates.extend(zip(years[keep].tolist(), dataset.time.dt.month.values[keep].tolist()))
            days.extend(dataset.time.dt.days_in_month.values[keep].tolist())
            for target in [40, 65]:
                latitude_index = int(np.argmin(np.abs(dataset.lat.values - target)))
                latitude_records.append({"target": target, "actual": float(dataset.lat.values[latitude_index]), "index": latitude_index})
                field = dataset.msftmz.isel(time=keep, basin=0, lat=latitude_index, lev=dataset.lev.values / 100 >= 500).values.astype(float)
                profiles[target].append(field)
    n_years = 140 if phase == "ramp_up" else 200
    assert dates == [(year, month) for year in range(1, n_years + 1) for month in range(1, 13)]
    weights = np.array(days).reshape(n_years, 12, 1)
    strengths = {}
    for target in [40, 65]:
        monthly = np.concatenate(profiles[target]).reshape(n_years, 12, -1)
        denominator = np.where(np.isfinite(monthly), weights, 0).sum(axis=1)
        annual_profile = np.divide(np.nansum(monthly * weights, axis=1), denominator, out=np.full((n_years, monthly.shape[2]), np.nan), where=denominator > 0)
        strengths[target] = np.nanmax(annual_profile, axis=1) / 1e9
    for index in range(n_years):
        simulation_year = index + 1 + (0 if phase == "ramp_up" else 140)
        overturning_rows.append({"simulation_year": simulation_year, "amoc_40n_sv": strengths[40][index], "overturning_65n_sv": strengths[65][index], "li_transport_proxy_sv": strengths[40][index] - strengths[65][index]})
overturning = pd.DataFrame(overturning_rows)
old_amoc = pd.read_csv(root / "output/amoc_s01_n01/l_40/CMIP6_AMOC_9models_annual_series.csv")
old_amoc = old_amoc.loc[(old_amoc.Model == "CESM2") & old_amoc["Simulation year"].between(1, 280)].sort_values("Simulation year")
amoc_error = float(np.max(np.abs(overturning.iloc[:280].amoc_40n_sv.values - old_amoc["Forced AMOC strength (Sv)"].values)))
assert amoc_error < 1e-5


# %% 6. 拼接完整阶段、预设窗口与绘图平滑

all_rows = []
for region in regions:
    regional = overturning.copy()
    regional["region"] = region
    for column in ["mld_mean_m", "mld_p90_m", "deep_area_1000m_percent", "no_cross_area_percent"]:
        regional[column] = np.nan
    for row in existing.loc[existing.region == region].itertuples():
        selected = regional.simulation_year == row.simulation_year
        regional.loc[selected, "mld_mean_m"] = row.march_mld_mean_crossing_only_m
        regional.loc[selected, "mld_p90_m"] = row.march_mld_p90_crossing_only_m
        regional.loc[selected, "deep_area_1000m_percent"] = 100 * row.resolved_mld_ge_1000m_fraction
        regional.loc[selected, "no_cross_area_percent"] = 100 * row.no_cross_weight_fraction
    for row in extended_mld.loc[(extended_mld.region == region) & (extended_mld.simulation_year > 280)].itertuples():
        selected = regional.simulation_year == row.simulation_year
        for column in ["mld_mean_m", "mld_p90_m", "deep_area_1000m_percent", "no_cross_area_percent"]:
            regional.loc[selected, column] = getattr(row, column)
    regional["phase"] = np.select([regional.simulation_year <= 140, regional.simulation_year <= 280], ["ramp_up", "ramp_down"], default="stabilization")
    exponent = np.where(regional.simulation_year <= 140, regional.simulation_year, np.maximum(280 - regional.simulation_year, 0))
    regional["co2_ppm"] = 284.7 * 1.01 ** exponent
    for column in ["mld_mean_m", "deep_area_1000m_percent", "amoc_40n_sv", "overturning_65n_sv", "li_transport_proxy_sv"]:
        regional[column + "_11yr"] = np.nan
        # 平滑连续真实时间序列，避免在阶段边界人为重启窗口造成跳变。
        smoothed = regional[column].rolling(11, center=True, min_periods=1).mean()
        smoothed.loc[regional[column].isna()] = np.nan
        regional[column + "_11yr"] = smoothed.values
    all_rows.append(regional)
annual = pd.concat(all_rows, ignore_index=True)
summary_rows = []
for region in regions:
    for window, first, last in [("initial", 1, 10), ("ramp_end", 131, 140), ("cdr_end", 271, 280), ("stabilization_first10", 281, 290), ("stabilization_last10", 330, 339)]:
        subset = annual.loc[(annual.region == region) & annual.simulation_year.between(first, last)]
        row = {"region": region, "window": window, "first_year": first, "last_year": last}
        for column in ["mld_mean_m", "mld_p90_m", "deep_area_1000m_percent", "no_cross_area_percent", "amoc_40n_sv", "overturning_65n_sv", "li_transport_proxy_sv"]:
            assert subset[column].notna().sum() == 10
            row[column] = float(subset[column].mean())
        summary_rows.append(row)
summary = pd.DataFrame(summary_rows)


# %% 7. 保存结果与QC

annual.to_csv(directory / (prefix + "_annual.csv"), index=False)
summary.to_csv(directory / (prefix + "_summary.csv"), index=False)
metadata = {"mld_definition": "same n035 sigma0 +0.03kg/m3 relative10m; actual crossings only; native-area weights", "mld_dimensions": ["time", "lev", "nlat", "nlon"], "overturning_dimensions": ["time", "basin", "lev", "lat"], "overturning_definition": "Atlantic-Arctic basin0; day-weighted annual profile then maximum below500m; kg/s divided1e9", "regional_caveat": "40N/65N are latitude diagnostics, not independent overturning of the drawn polygons.40N minus65N is a proxy, not closed mass budget.", "smoothing": "11year centered mean along continuous series across phases; shorter edge windows; missing original observations retainedmissing. Endpoint stats unsmoothed.", "phase_years": {"ramp_up": [1, 140], "ramp_down": [141, 280], "stabilization": [281, 340]}, "co2": "Idealized prescribed trajectory inherited from project:284.7*1.01**year forup, reverse exponent fordown,284.7constant after280; not measured annualCO2", "coverage": "MLD available1-339, AMOC1-340; no thetaoCDRyear200 locally, MLD year340 keptmissing", "interpretation": "Finite-rate hysteresis and59-60year constant-forcing recovery do not alone establish equilibrium bifurcation/tipping or permanent irreversibility", "old_mld_overlap_max_error_m": max(overlap_errors), "old_amoc_overlap_max_error_sv": amoc_error, "latitude_records": latitude_records, "input_files": file_metadata}
(directory / (prefix + "_metadata.json")).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(summary.to_string(index=False))
print("MLD overlap error:", max(overlap_errors), "AMOC overlap error:", amoc_error)
