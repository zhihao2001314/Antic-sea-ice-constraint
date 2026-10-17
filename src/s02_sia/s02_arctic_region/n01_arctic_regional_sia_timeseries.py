# %% 1. 导入依赖

from pathlib import Path
import re

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree


# %% 2. 参数与路径设置

project_root = Path(r"F:\sea\Antic sea ice constraint")
data_root = Path(r"F:\sea")

stage_directories = {
    "1pctCO2": data_root / "cmipseaicead",
    "1pctCO2-cdr": data_root / "cmipseaice",
    "piControl": data_root / "cmipseaice-picontrol",
}
area_root = data_root / "areacello"

output_directory = (
    project_root
    / "output"
    / "s02_arctic_region"
    / "p01_regional_sia"
    / "n01"
)
output_directory.mkdir(parents=True, exist_ok=True)

annual_output_path = output_directory / "arctic_regional_sia_annual.csv"
multimodel_output_path = output_directory / "arctic_regional_multimodel_timeseries.csv"
endpoint_output_path = output_directory / "arctic_regional_endpoint_summary.csv"
model_output_path = output_directory / "arctic_regional_model_summary.csv"
region_output_path = output_directory / "arctic_region_definitions.csv"
quality_output_path = output_directory / "arctic_regional_sia_qc.csv"

ramp_up_years = 140
ramp_down_years = 140
endpoint_window_years = 10
nominal_start_year = 2001
pi_reference_initial_window_centre = 4.5
million_km2_in_m2 = 1.0e12
earth_radius_km = 6371.0
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)

model_information = {
    "ACCESS-ESM1-5": {
        "folder": "ACCESSESM5",
        "forced_member": "r1i1p1f1",
        "picontrol_member": "r1i1p1f1",
    },
    "CanESM5": {
        "folder": "CANESM5",
        "forced_member": "r1i1p2f1",
        "picontrol_member": "r1i1p2f1",
    },
    "CESM2": {
        "folder": "CESM2",
        "forced_member": "r1i1p1f1",
        "picontrol_member": "r1i1p1f1",
    },
    "CNRM-ESM2-1": {
        "folder": "CNRM-ESM2",
        "forced_member": "r1i1p1f2",
        "picontrol_member": "r1i1p1f2",
    },
    "GFDL-ESM4": {
        "folder": "GFDL",
        "forced_member": "r1i1p1f1",
        "picontrol_member": "r1i1p1f1",
    },
    "MIROC-ES2L": {
        "folder": "MIROC-ES2L",
        "forced_member": "r1i1p1f2",
        "picontrol_member": "r1i1p1f2",
    },
    "UKESM1-0-LL": {
        "folder": "UKESM1-0-LL",
        "forced_member": "r1i1p1f2",
        "picontrol_member": "r1i1p1f2",
    },
    "CAS-ESM2-0": {
        "folder": "CAS",
        "forced_member": "r1i1p1f1",
        "picontrol_member": "r1i1p1f1",
    },
    "NorESM2-LM": {
        "folder": "NORESM2-LM2",
        "forced_member": "r1i1p1f1",
        "picontrol_member": "r1i1p1f1",
    },
}

region_names = [
    "North Atlantic",
    "Central Arctic",
    "North Pacific",
    "Arctic total 45-90N",
    "Northern Hemisphere total 0-90N",
]

region_abbreviations = {
    "North Atlantic": "NA",
    "Central Arctic": "CA",
    "North Pacific": "NP",
    "Arctic total 45-90N": "ARC",
    "Northern Hemisphere total 0-90N": "NH",
}


# %% 3. 保存区域定义

region_definition_rows = [
    {
        "region": "North Atlantic",
        "abbreviation": "NA",
        "latitude_and_longitude_definition": (
            "45<=lat<80 and -120<=lon<90"
        ),
        "role": "North Atlantic sector",
    },
    {
        "region": "Central Arctic",
        "abbreviation": "CA",
        "latitude_and_longitude_definition": (
            "(80<=lat<=90 and -120<=lon<90) or "
            "(65<=lat<=90 and (lon>=90 or lon<-120))"
        ),
        "role": "Central Arctic sector",
    },
    {
        "region": "North Pacific",
        "abbreviation": "NP",
        "latitude_and_longitude_definition": (
            "45<=lat<65 and (lon>=90 or lon<-120)"
        ),
        "role": "North Pacific sector",
    },
    {
        "region": "Arctic total 45-90N",
        "abbreviation": "ARC",
        "latitude_and_longitude_definition": "union of NA, CA and NP",
        "role": "regional Arctic total used for shares",
    },
    {
        "region": "Northern Hemisphere total 0-90N",
        "abbreviation": "NH",
        "latitude_and_longitude_definition": "0<=lat<=90",
        "role": "compatibility total for the former 0-90N map",
    },
]
pd.DataFrame(region_definition_rows).to_csv(
    region_output_path,
    index=False,
    encoding="utf-8-sig",
)


# %% 4. 逐模式读取三阶段海冰并计算区域年平均SIA

annual_rows = []
endpoint_rows = []
model_rows = []
quality_rows = []

for model_name, model_configuration in model_information.items():
    print(f"\n处理模式：{model_name}")

    area_folder = area_root / model_configuration["folder"]
    area_files = sorted(area_folder.rglob("*.nc"))
    if len(area_files) != 1:
        raise ValueError(
            f"{model_name} 的 areacello 文件数应为1，实际为{len(area_files)}。"
        )
    area_path = area_files[0]

    stage_annual_values = {}
    stage_elapsed_years = {}
    stage_file_counts = {}
    stage_area_match_p95_km = {}
    stage_area_match_max_km = {}

    for stage_name, stage_root in stage_directories.items():
        stage_folder = stage_root / model_configuration["folder"]
        stage_files = sorted(stage_folder.rglob("*.nc"))

        if stage_name == "piControl":
            selected_member = model_configuration["picontrol_member"]
        else:
            selected_member = model_configuration["forced_member"]

        labelled_files = [
            path
            for path in stage_files
            if re.search(r"r\d+i\d+p\d+f\d+", path.as_posix(), re.IGNORECASE)
        ]
        if labelled_files:
            stage_files = [
                path
                for path in labelled_files
                if selected_member.lower() in path.as_posix().lower()
            ]

        # CAS-ESM2-0 的 CDR 文件名为 siconc.nc，不带成员标签。
        if model_name == "CAS-ESM2-0" and stage_name == "1pctCO2-cdr":
            stage_files = sorted(stage_folder.rglob("*.nc"))

        if not stage_files:
            raise FileNotFoundError(
                f"{model_name} {stage_name} 没有成员 {selected_member} 的文件。"
            )

        stage_file_counts[stage_name] = len(stage_files)
        print(f"  {stage_name}: {len(stage_files)} files")

        dataset = xr.open_mfdataset(
            [str(path) for path in stage_files],
            combine="by_coords",
            decode_times=time_coder,
            parallel=False,
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )

        ice_name = ""
        for candidate_name in ("siconc", "siconca", "sic", "aice"):
            if candidate_name in dataset.data_vars:
                ice_name = candidate_name
                break
        if not ice_name:
            raise KeyError(f"{model_name} {stage_name} 未找到海冰浓度变量。")

        ice = dataset[ice_name].squeeze(drop=True)
        time_dimension = "time"
        spatial_dimensions = [
            dimension for dimension in ice.dims if dimension != time_dimension
        ]
        if len(spatial_dimensions) != 2:
            raise ValueError(
                f"{model_name} {stage_name} 海冰维度应为time+y+x，实际为{ice.dims}。"
            )
        ice = ice.transpose(time_dimension, *spatial_dimensions)
        target_shape = (
            ice.sizes[spatial_dimensions[0]],
            ice.sizes[spatial_dimensions[1]],
        )

        latitude_candidates = []
        longitude_candidates = []
        for variable_name in dataset.variables:
            coordinate = dataset[variable_name]
            if coordinate.ndim not in {1, 2}:
                continue
            standard_name = str(coordinate.attrs.get("standard_name", "")).lower()
            units = str(coordinate.attrs.get("units", "")).lower()
            axis = str(coordinate.attrs.get("axis", "")).upper()

            latitude_score = 0
            if standard_name == "latitude":
                latitude_score += 12
            if axis == "Y":
                latitude_score += 5
            if variable_name.lower() in {
                "lat", "latitude", "nav_lat", "geolat", "tlat", "ulat"
            }:
                latitude_score += 6
            if "degree_north" in units:
                latitude_score += 4
            if latitude_score:
                latitude_candidates.append((latitude_score, variable_name))

            longitude_score = 0
            if standard_name == "longitude":
                longitude_score += 12
            if axis == "X":
                longitude_score += 5
            if variable_name.lower() in {
                "lon", "longitude", "nav_lon", "geolon", "tlon", "ulon"
            }:
                longitude_score += 6
            if "degree_east" in units:
                longitude_score += 4
            if longitude_score:
                longitude_candidates.append((longitude_score, variable_name))

        latitude_candidates.sort(reverse=True)
        longitude_candidates.sort(reverse=True)
        latitude = np.asarray(
            dataset[latitude_candidates[0][1]].squeeze(drop=True).values,
            dtype=float,
        )
        longitude = np.asarray(
            dataset[longitude_candidates[0][1]].squeeze(drop=True).values,
            dtype=float,
        )

        if latitude.ndim == 1:
            if latitude.size == target_shape[0]:
                latitude = np.repeat(latitude[:, None], target_shape[1], axis=1)
            else:
                latitude = np.repeat(latitude[None, :], target_shape[0], axis=0)
        if longitude.ndim == 1:
            if longitude.size == target_shape[1]:
                longitude = np.repeat(longitude[None, :], target_shape[0], axis=0)
            else:
                longitude = np.repeat(longitude[:, None], target_shape[1], axis=1)
        if latitude.T.shape == target_shape and latitude.shape != target_shape:
            latitude = latitude.T
        if longitude.T.shape == target_shape and longitude.shape != target_shape:
            longitude = longitude.T

        latitude[(latitude < -90.0001) | (latitude > 90.0001)] = np.nan
        longitude[np.abs(longitude) > 1.0e6] = np.nan
        finite_longitude = np.isfinite(longitude)
        longitude[finite_longitude] = (
            (longitude[finite_longitude] + 180.0) % 360.0
        ) - 180.0

        area_dataset = xr.open_dataset(
            area_path,
            decode_times=False,
            mask_and_scale=True,
        )
        area_name = "areacello"
        area = area_dataset[area_name].squeeze(drop=True)
        area_values = np.asarray(area.values, dtype=float)
        area_units = str(area.attrs.get("units", "")).lower()
        if "km" in area_units:
            area_values = area_values * 1.0e6

        area_latitude_candidates = []
        area_longitude_candidates = []
        for variable_name in area_dataset.variables:
            coordinate = area_dataset[variable_name]
            if coordinate.ndim not in {1, 2}:
                continue
            standard_name = str(coordinate.attrs.get("standard_name", "")).lower()
            units = str(coordinate.attrs.get("units", "")).lower()
            axis = str(coordinate.attrs.get("axis", "")).upper()

            latitude_score = 0
            if standard_name == "latitude":
                latitude_score += 12
            if axis == "Y":
                latitude_score += 5
            if variable_name.lower() in {
                "lat", "latitude", "nav_lat", "geolat", "tlat", "ulat"
            }:
                latitude_score += 6
            if "degree_north" in units:
                latitude_score += 4
            if latitude_score:
                area_latitude_candidates.append((latitude_score, variable_name))

            longitude_score = 0
            if standard_name == "longitude":
                longitude_score += 12
            if axis == "X":
                longitude_score += 5
            if variable_name.lower() in {
                "lon", "longitude", "nav_lon", "geolon", "tlon", "ulon"
            }:
                longitude_score += 6
            if "degree_east" in units:
                longitude_score += 4
            if longitude_score:
                area_longitude_candidates.append((longitude_score, variable_name))

        area_latitude_candidates.sort(reverse=True)
        area_longitude_candidates.sort(reverse=True)
        area_latitude = np.asarray(
            area_dataset[area_latitude_candidates[0][1]].squeeze(drop=True).values,
            dtype=float,
        )
        area_longitude = np.asarray(
            area_dataset[area_longitude_candidates[0][1]].squeeze(drop=True).values,
            dtype=float,
        )
        area_shape = area_values.shape

        if area_latitude.ndim == 1:
            if area_latitude.size == area_shape[0]:
                area_latitude = np.repeat(
                    area_latitude[:, None], area_shape[1], axis=1
                )
            else:
                area_latitude = np.repeat(
                    area_latitude[None, :], area_shape[0], axis=0
                )
        if area_longitude.ndim == 1:
            if area_longitude.size == area_shape[1]:
                area_longitude = np.repeat(
                    area_longitude[None, :], area_shape[0], axis=0
                )
            else:
                area_longitude = np.repeat(
                    area_longitude[:, None], area_shape[1], axis=1
                )
        if area_latitude.T.shape == area_shape and area_latitude.shape != area_shape:
            area_latitude = area_latitude.T
        if area_longitude.T.shape == area_shape and area_longitude.shape != area_shape:
            area_longitude = area_longitude.T

        area_latitude[(area_latitude < -90.0001) | (area_latitude > 90.0001)] = np.nan
        area_longitude[np.abs(area_longitude) > 1.0e6] = np.nan
        finite_area_longitude = np.isfinite(area_longitude)
        area_longitude[finite_area_longitude] = (
            (area_longitude[finite_area_longitude] + 180.0) % 360.0
        ) - 180.0

        source_valid = (
            np.isfinite(area_values)
            & (area_values > 0.0)
            & np.isfinite(area_latitude)
            & np.isfinite(area_longitude)
            & (area_latitude >= -5.0)
            & (area_latitude <= 90.0)
        )
        sic_grid_valid = np.asarray(
            np.isfinite(ice.isel({time_dimension: 0}).load().values),
            dtype=bool,
        )
        target_valid = (
            np.isfinite(latitude)
            & np.isfinite(longitude)
            & sic_grid_valid
            & (latitude >= 0.0)
            & (latitude <= 90.0)
        )

        source_latitude_radians = np.deg2rad(area_latitude[source_valid])
        source_longitude_radians = np.deg2rad(area_longitude[source_valid])
        source_xyz = np.column_stack(
            [
                np.cos(source_latitude_radians) * np.cos(source_longitude_radians),
                np.cos(source_latitude_radians) * np.sin(source_longitude_radians),
                np.sin(source_latitude_radians),
            ]
        )
        target_latitude_radians = np.deg2rad(latitude[target_valid])
        target_longitude_radians = np.deg2rad(longitude[target_valid])
        target_xyz = np.column_stack(
            [
                np.cos(target_latitude_radians) * np.cos(target_longitude_radians),
                np.cos(target_latitude_radians) * np.sin(target_longitude_radians),
                np.sin(target_latitude_radians),
            ]
        )

        nearest_chord, nearest_source_index = cKDTree(source_xyz).query(
            target_xyz,
            k=1,
        )
        nearest_distance_km = (
            2.0
            * earth_radius_km
            * np.arcsin(np.clip(nearest_chord * 0.5, 0.0, 1.0))
        )
        if float(np.nanmax(nearest_distance_km)) > 300.0:
            raise ValueError(
                f"{model_name} {stage_name} areacello最大匹配距离超过300 km。"
            )

        source_flat_indices = np.flatnonzero(source_valid)
        matched_source_flat_indices = source_flat_indices[nearest_source_index]
        matched_area = area_values.ravel()[matched_source_flat_indices]
        matched_area_latitude = area_latitude.ravel()[matched_source_flat_indices]
        matched_area_longitude = area_longitude.ravel()[matched_source_flat_indices]

        unique_source_indices, inverse_source_indices, source_multiplicity = np.unique(
            matched_source_flat_indices,
            return_inverse=True,
            return_counts=True,
        )
        del unique_source_indices
        matched_area = matched_area / source_multiplicity[inverse_source_indices]

        area_on_sic = np.full(target_shape, np.nan, dtype=float)
        area_latitude_on_sic = np.full(target_shape, np.nan, dtype=float)
        area_longitude_on_sic = np.full(target_shape, np.nan, dtype=float)
        area_on_sic[target_valid] = matched_area
        area_latitude_on_sic[target_valid] = matched_area_latitude
        area_longitude_on_sic[target_valid] = matched_area_longitude

        stage_area_match_p95_km[stage_name] = float(
            np.nanpercentile(nearest_distance_km, 95.0)
        )
        stage_area_match_max_km[stage_name] = float(
            np.nanmax(nearest_distance_km)
        )

        atlantic_longitude = (
            (area_longitude_on_sic >= -120.0)
            & (area_longitude_on_sic < 90.0)
        )
        pacific_longitude = ~atlantic_longitude
        north_atlantic_mask = (
            np.isfinite(area_on_sic)
            & atlantic_longitude
            & (area_latitude_on_sic >= 45.0)
            & (area_latitude_on_sic < 80.0)
        )
        central_arctic_mask = (
            np.isfinite(area_on_sic)
            & (
                (
                    atlantic_longitude
                    & (area_latitude_on_sic >= 80.0)
                    & (area_latitude_on_sic <= 90.0)
                )
                | (
                    pacific_longitude
                    & (area_latitude_on_sic >= 65.0)
                    & (area_latitude_on_sic <= 90.0)
                )
            )
        )
        north_pacific_mask = (
            np.isfinite(area_on_sic)
            & pacific_longitude
            & (area_latitude_on_sic >= 45.0)
            & (area_latitude_on_sic < 65.0)
        )
        arctic_total_mask = (
            north_atlantic_mask | central_arctic_mask | north_pacific_mask
        )
        northern_hemisphere_mask = (
            np.isfinite(area_on_sic)
            & (area_latitude_on_sic >= 0.0)
            & (area_latitude_on_sic <= 90.0)
        )

        region_masks = np.stack(
            [
                north_atlantic_mask,
                central_arctic_mask,
                north_pacific_mask,
                arctic_total_mask,
                northern_hemisphere_mask,
            ],
            axis=0,
        )
        region_area_weights = np.where(
            region_masks,
            area_on_sic[None, :, :],
            0.0,
        )
        area_weight_dataarray = xr.DataArray(
            region_area_weights,
            dims=("region", *spatial_dimensions),
            coords={"region": region_names},
        )

        ice_units = str(ice.attrs.get("units", "")).lower()
        sic_scale = 0.01 if ("%" in ice_units or "percent" in ice_units) else 1.0
        if sic_scale == 1.0 and ice_units.strip() not in {"1", "fraction", "0-1", "0–1"}:
            sample_values = np.asarray(
                ice.isel({time_dimension: [0, ice.sizes[time_dimension] // 2, -1]})
                .load()
                .values,
                dtype=float,
            )
            sample_values = sample_values[
                np.isfinite(sample_values)
                & (sample_values >= -5.0)
                & (sample_values <= 150.0)
            ]
            if float(np.nanpercentile(sample_values, 99.0)) > 1.5:
                sic_scale = 0.01

        ice_fraction = ice.astype(float) * sic_scale
        ice_fraction = ice_fraction.where(
            np.isfinite(ice_fraction)
            & (ice_fraction >= -0.05)
            & (ice_fraction <= 1.5)
        ).clip(min=0.0, max=1.0)

        monthly_region_sia = (
            (ice_fraction * area_weight_dataarray)
            .sum(dim=spatial_dimensions, skipna=True)
            .transpose(time_dimension, "region")
            / million_km2_in_m2
        ).compute()

        calendar_years = np.asarray(
            dataset[time_dimension].dt.year.values,
            dtype=int,
        )
        complete_calendar_years = []
        annual_value_rows = []
        for calendar_year in np.unique(calendar_years):
            year_indices = np.flatnonzero(calendar_years == calendar_year)
            if year_indices.size == 12:
                complete_calendar_years.append(int(calendar_year))
                annual_value_rows.append(
                    np.asarray(
                        monthly_region_sia.isel({time_dimension: year_indices})
                        .mean(dim=time_dimension)
                        .values,
                        dtype=float,
                    )
                )

        annual_values = np.vstack(annual_value_rows)
        complete_calendar_years = np.asarray(complete_calendar_years, dtype=int)
        stage_annual_values[stage_name] = annual_values
        stage_elapsed_years[stage_name] = (
            complete_calendar_years - complete_calendar_years[0]
        ).astype(float)

        quality_rows.append(
            {
                "model": model_name,
                "stage": stage_name,
                "member": selected_member,
                "file_count": len(stage_files),
                "complete_year_count": annual_values.shape[0],
                "first_calendar_year_in_file": int(complete_calendar_years[0]),
                "last_calendar_year_in_file": int(complete_calendar_years[-1]),
                "sic_shape": str(target_shape),
                "areacello_file": str(area_path),
                "area_match_p95_km": stage_area_match_p95_km[stage_name],
                "area_match_max_km": stage_area_match_max_km[stage_name],
                "regional_sum_max_error_million_km2": float(
                    np.nanmax(
                        np.abs(
                            annual_values[:, 0]
                            + annual_values[:, 1]
                            + annual_values[:, 2]
                            - annual_values[:, 3]
                        )
                    )
                ),
            }
        )

        area_dataset.close()
        dataset.close()

    if stage_annual_values["1pctCO2"].shape[0] < endpoint_window_years:
        raise ValueError(f"{model_name} 1pctCO2不足{endpoint_window_years}个完整年。")
    if stage_annual_values["1pctCO2-cdr"].shape[0] < ramp_down_years:
        raise ValueError(f"{model_name} CDR不足{ramp_down_years}个完整年。")
    if stage_annual_values["piControl"].shape[0] < 20:
        raise ValueError(f"{model_name} piControl不足20个完整年。")

    raw_combined = np.full(
        (ramp_up_years + ramp_down_years, len(region_names)),
        np.nan,
        dtype=float,
    )
    available_ramp_up_years = min(
        ramp_up_years,
        stage_annual_values["1pctCO2"].shape[0],
    )
    raw_combined[:available_ramp_up_years, :] = stage_annual_values[
        "1pctCO2"
    ][:available_ramp_up_years, :]
    raw_combined[
        ramp_up_years : ramp_up_years + ramp_down_years,
        :,
    ] = stage_annual_values["1pctCO2-cdr"][:ramp_down_years, :]
    simulation_index = np.arange(raw_combined.shape[0], dtype=float)
    nominal_years = nominal_start_year + simulation_index.astype(int)
    experiment_labels = np.where(
        simulation_index < ramp_up_years,
        "1pctCO2",
        "1pctCO2-cdr",
    )

    picontrol_elapsed_years = stage_elapsed_years["piControl"]
    picontrol_values = stage_annual_values["piControl"]
    picontrol_slopes = np.zeros(len(region_names), dtype=float)
    for region_index in range(len(region_names)):
        picontrol_slopes[region_index] = np.polyfit(
            picontrol_elapsed_years,
            picontrol_values[:, region_index],
            1,
        )[0]

    # 初始10年窗口中心设为共同参考点，使其均值保持原值；
    # CDR末10年窗口与之相隔270年，严格复现旧空间图的线性漂移订正。
    drift_offset_years = (
        simulation_index - pi_reference_initial_window_centre
    )[:, None]
    corrected_combined = raw_combined - drift_offset_years * picontrol_slopes[None, :]
    corrected_smoothed = (
        pd.DataFrame(corrected_combined)
        .rolling(window=11, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=float)
    )
    corrected_smoothed[~np.isfinite(corrected_combined)] = np.nan

    north_atlantic_share = (
        corrected_combined[:, 0] / corrected_combined[:, 3] * 100.0
    )

    for time_index in range(corrected_combined.shape[0]):
        for region_index, region_name in enumerate(region_names):
            annual_rows.append(
                {
                    "model": model_name,
                    "simulation_year": int(time_index + 1),
                    "nominal_year": int(nominal_years[time_index]),
                    "experiment": str(experiment_labels[time_index]),
                    "region": region_name,
                    "region_abbreviation": region_abbreviations[region_name],
                    "raw_sia_million_km2": float(
                        raw_combined[time_index, region_index]
                    ),
                    "picontrol_trend_million_km2_per_year": float(
                        picontrol_slopes[region_index]
                    ),
                    "picontrol_corrected_sia_million_km2": float(
                        corrected_combined[time_index, region_index]
                    ),
                    "picontrol_corrected_sia_11yr_mean_million_km2": float(
                        corrected_smoothed[time_index, region_index]
                    ),
                    "north_atlantic_share_of_arctic_percent": float(
                        north_atlantic_share[time_index]
                    ),
                }
            )

    initial_means = np.nanmean(
        corrected_combined[:endpoint_window_years, :],
        axis=0,
    )
    final_means = np.nanmean(
        corrected_combined[-endpoint_window_years:, :],
        axis=0,
    )
    endpoint_changes = final_means - initial_means

    regional_change_indices = np.arange(3)
    dominant_region_index = int(
        regional_change_indices[np.nanargmax(np.abs(endpoint_changes[:3]))]
    )
    dominant_region = region_names[dominant_region_index]
    absolute_regional_change_sum = float(np.nansum(np.abs(endpoint_changes[:3])))
    north_atlantic_absolute_change_share = (
        abs(float(endpoint_changes[0])) / absolute_regional_change_sum * 100.0
        if absolute_regional_change_sum > 0.0
        else np.nan
    )
    north_atlantic_net_change_contribution = (
        float(endpoint_changes[0]) / float(endpoint_changes[3]) * 100.0
        if not np.isclose(endpoint_changes[3], 0.0)
        else np.nan
    )

    for region_index, region_name in enumerate(region_names):
        change_value = float(endpoint_changes[region_index])
        endpoint_rows.append(
            {
                "model": model_name,
                "region": region_name,
                "region_abbreviation": region_abbreviations[region_name],
                "initial_10yr_mean_sia_million_km2": float(
                    initial_means[region_index]
                ),
                "final_10yr_mean_sia_million_km2": float(
                    final_means[region_index]
                ),
                "change_final_minus_initial_million_km2": change_value,
                "change_percent_of_initial": (
                    change_value / float(initial_means[region_index]) * 100.0
                    if not np.isclose(initial_means[region_index], 0.0)
                    else np.nan
                ),
                "change_direction": (
                    "increase" if change_value > 0.0 else "decrease"
                ),
            }
        )

    model_rows.append(
        {
            "model": model_name,
            "north_atlantic_initial_share_percent": float(
                initial_means[0] / initial_means[3] * 100.0
            ),
            "north_atlantic_final_share_percent": float(
                final_means[0] / final_means[3] * 100.0
            ),
            "north_atlantic_share_change_percentage_points": float(
                final_means[0] / final_means[3] * 100.0
                - initial_means[0] / initial_means[3] * 100.0
            ),
            "north_atlantic_change_million_km2": float(endpoint_changes[0]),
            "central_arctic_change_million_km2": float(endpoint_changes[1]),
            "north_pacific_change_million_km2": float(endpoint_changes[2]),
            "arctic_total_change_million_km2": float(endpoint_changes[3]),
            "northern_hemisphere_total_change_million_km2": float(
                endpoint_changes[4]
            ),
            "north_atlantic_net_change_contribution_percent": (
                north_atlantic_net_change_contribution
            ),
            "north_atlantic_share_of_absolute_regional_change_percent": (
                north_atlantic_absolute_change_share
            ),
            "dominant_region_by_absolute_change": dominant_region,
            "north_atlantic_is_dominant": dominant_region == "North Atlantic",
            "arctic_total_change_direction": (
                "increase" if endpoint_changes[3] > 0.0 else "decrease"
            ),
            "available_1pctco2_years": int(available_ramp_up_years),
            "missing_1pctco2_years_before_cdr": int(
                ramp_up_years - available_ramp_up_years
            ),
        }
    )


# %% 5. 汇总多模式统计

annual_dataframe = pd.DataFrame(annual_rows)
endpoint_dataframe = pd.DataFrame(endpoint_rows)
model_dataframe = pd.DataFrame(model_rows)
quality_dataframe = pd.DataFrame(quality_rows)

multimodel_group_columns = [
    "simulation_year",
    "nominal_year",
    "experiment",
    "region",
    "region_abbreviation",
]
multimodel_groups = annual_dataframe.groupby(
    multimodel_group_columns,
    as_index=False,
)
multimodel_mean = multimodel_groups[
    [
        "picontrol_corrected_sia_million_km2",
        "picontrol_corrected_sia_11yr_mean_million_km2",
        "north_atlantic_share_of_arctic_percent",
    ]
].mean()
multimodel_p17 = multimodel_groups[
    [
        "picontrol_corrected_sia_million_km2",
        "picontrol_corrected_sia_11yr_mean_million_km2",
        "north_atlantic_share_of_arctic_percent",
    ]
].quantile(0.17)
multimodel_p83 = multimodel_groups[
    [
        "picontrol_corrected_sia_million_km2",
        "picontrol_corrected_sia_11yr_mean_million_km2",
        "north_atlantic_share_of_arctic_percent",
    ]
].quantile(0.83)

multimodel_dataframe = multimodel_mean.merge(
    multimodel_p17,
    on=multimodel_group_columns,
    suffixes=("_mean", "_p17"),
).merge(
    multimodel_p83,
    on=multimodel_group_columns,
)
multimodel_dataframe = multimodel_dataframe.rename(
    columns={
        "picontrol_corrected_sia_million_km2": (
            "picontrol_corrected_sia_million_km2_p83"
        ),
        "picontrol_corrected_sia_11yr_mean_million_km2": (
            "picontrol_corrected_sia_11yr_mean_million_km2_p83"
        ),
        "north_atlantic_share_of_arctic_percent": (
            "north_atlantic_share_of_arctic_percent_p83"
        ),
    }
)

# %% 6. 保存结果

annual_dataframe.to_csv(annual_output_path, index=False, encoding="utf-8-sig")
multimodel_dataframe.to_csv(
    multimodel_output_path,
    index=False,
    encoding="utf-8-sig",
)
endpoint_dataframe.to_csv(endpoint_output_path, index=False, encoding="utf-8-sig")
model_dataframe.to_csv(model_output_path, index=False, encoding="utf-8-sig")
quality_dataframe.to_csv(quality_output_path, index=False, encoding="utf-8-sig")

print(f"\n区域年序列：{annual_output_path}")
print(f"多模式统计：{multimodel_output_path}")
print(f"端点变化：{endpoint_output_path}")
print(f"模式汇总：{model_output_path}")
print(f"区域定义：{region_output_path}")
print(f"质量检查：{quality_output_path}")
