# -*- coding: utf-8 -*-
"""
计算北极年平均海冰面积的bias-corrected later−early pattern。

科学口径
1. historical模式和NSIDC观测采用1979–2014年，逐月、逐格点拟合bias correction参数。
2. 固定参数分别应用于1pctCO2和1pctCO2-cdr，不使用piControl，不做漂移订正。
3. early为1pctCO2实验年1–10，later为CDR实验年131–140。
4. 每个日历月先订正，再将12个月和10年共同平均为年平均SIC端点。
5. pattern = bias-corrected later − bias-corrected early。
6. CA、NA、NP沿用最终分区；Total仅为CA+NA+NP之和。
"""


# %% 1. 导入依赖

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# %% 2. 参数与路径设置

project_root = Path(r"F:\sea\Antic sea ice constraint")
data_root = Path(r"F:\sea")
legacy_source_directory = (
    project_root / "archive" / "202608221141_code_cleanup" / "src"
)
arctic_module_file = (
    legacy_source_directory
    / "arctic_sic_bias_correction_blue_spatial_comparison_v6_lower_colorbar.py"
)
shared_module_file = (
    legacy_source_directory / "antarctic_cdr_bias_corrected_pipeline.py"
)

output_directory = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p16_bias_corrected_pattern"
    / "n023_arctic_sia_pattern"
)
cache_directory = output_directory / "regridding_cache"
output_directory.mkdir(parents=True, exist_ok=True)
cache_directory.mkdir(parents=True, exist_ok=True)

cell_output = output_directory / "arctic_bias_corrected_sia_pattern_cells.csv"
summary_output = output_directory / "arctic_bias_corrected_sia_pattern_summary.csv"
edge_output = output_directory / "arctic_bias_corrected_sia_pattern_edges.npz"
display_qc_output = output_directory / "arctic_bias_corrected_sia_pattern_display_qc.csv"
timeseries_output = output_directory / "arctic_bias_corrected_annual_sia_280yr.csv"
consistency_qc_output = (
    output_directory
    / "arctic_bias_corrected_annual_sia_pattern_consistency_qc.csv"
)
metadata_output = output_directory / "arctic_bias_corrected_sia_pattern_metadata.json"

model_order = [
    "ACCESS-ESM1-5",
    "CanESM5",
    "CESM2",
    "CNRM-ESM2-1",
    "GFDL-ESM4",
    "MIROC-ES2L",
    "UKESM1-0-LL",
    "CAS-ESM2-0",
    "NorESM2-LM",
]
calendar_months = np.arange(1, 13, dtype=int)
historical_years = np.arange(1979, 2015, dtype=int)
ramp_years = 140
endpoint_years = 10
cell_area_km2 = 625.0
color_limit_thousand_km2 = 1.5
earth_radius_km = 6371.0088
display_idw_neighbours = 8
display_idw_power = 2.0
display_max_nearest_distance_km = 300.0
# 允许不同求和顺序产生的浮点舍入差；1e-6 million km² = 1 km²。
pattern_timeseries_tolerance_million_km2 = 1.0e-6

target_latitude = np.arange(45.5, 90.0, 1.0)
target_longitude = np.arange(-179.5, 180.0, 1.0)
target_longitude_grid, target_latitude_grid = np.meshgrid(
    target_longitude,
    target_latitude,
)
target_latitude_radian = np.deg2rad(target_latitude_grid.ravel())
target_longitude_radian = np.deg2rad(target_longitude_grid.ravel())
target_xyz = np.column_stack(
    [
        np.cos(target_latitude_radian) * np.cos(target_longitude_radian),
        np.cos(target_latitude_radian) * np.sin(target_longitude_radian),
        np.sin(target_latitude_radian),
    ]
)
target_south = np.clip(target_latitude_grid - 0.5, -90.0, 90.0)
target_north = np.clip(target_latitude_grid + 0.5, -90.0, 90.0)
target_area_m2 = (
    (earth_radius_km * 1000.0) ** 2
    * np.deg2rad(1.0)
    * (
        np.sin(np.deg2rad(target_north))
        - np.sin(np.deg2rad(target_south))
    )
)
simulation_years = np.arange(1, 2 * ramp_years + 1, dtype=int)
co2_ppm = 284.7 * np.where(
    simulation_years <= ramp_years,
    1.01 ** simulation_years,
    1.01 ** (2 * ramp_years - simulation_years),
)
phase_labels = np.where(
    simulation_years <= ramp_years,
    "ramp-up",
    "ramp-down",
)

for required_file in [arctic_module_file, shared_module_file]:
    if not required_file.is_file():
        raise FileNotFoundError(required_file)


# %% 3. 载入既有bias correction与重网格方法

arctic_spec = importlib.util.spec_from_file_location(
    "arctic_historical_bias_correction_pattern",
    arctic_module_file,
)
if arctic_spec is None or arctic_spec.loader is None:
    raise ImportError(arctic_module_file)
arctic_module = importlib.util.module_from_spec(arctic_spec)
sys.modules[arctic_spec.name] = arctic_module
arctic_spec.loader.exec_module(arctic_module)

shared_spec = importlib.util.spec_from_file_location(
    "polar_bias_correction_shared_pattern",
    shared_module_file,
)
if shared_spec is None or shared_spec.loader is None:
    raise ImportError(shared_module_file)
shared_module = importlib.util.module_from_spec(shared_spec)
sys.modules[shared_spec.name] = shared_module
shared_spec.loader.exec_module(shared_module)

arctic_module.BASE_DIR = data_root
arctic_module.HISTORICAL_DIR = data_root / "sic-historical"
arctic_module.OBSERVATION_DIR = data_root / "seaice-obs-north-monthly"
arctic_module.CACHE_DIR = cache_directory

shared_module.SEA_ROOT = data_root
shared_module.HIST_DIR = data_root / "sic-historical"
shared_module.UP_DIR = data_root / "cmipseaicead"
shared_module.CDR_DIR = data_root / "cmipseaice"

for required_directory in [
    arctic_module.HISTORICAL_DIR,
    arctic_module.OBSERVATION_DIR,
    shared_module.UP_DIR,
    shared_module.CDR_DIR,
]:
    if not required_directory.is_dir():
        raise FileNotFoundError(f"缺少数据目录：{required_directory}")


# %% 4. 读取NSIDC观测网格并建立最终分区

observation = arctic_module.load_observations()
domain_mask = np.asarray(observation.domain_mask, dtype=bool)
latitude = np.asarray(observation.latitude, dtype=float)
longitude = np.asarray(observation.longitude, dtype=float)
longitude_180 = (longitude + 180.0) % 360.0 - 180.0

central_arctic_mask = domain_mask & (latitude >= 70.0) & (latitude <= 90.0)
north_atlantic_mask = (
    domain_mask
    & (latitude >= 45.0)
    & (latitude < 70.0)
    & (longitude_180 >= -60.0)
    & (longitude_180 < 30.0)
)
north_pacific_mask = (
    domain_mask
    & (latitude >= 45.0)
    & (latitude < 70.0)
    & ((longitude_180 >= 120.0) | (longitude_180 < -150.0))
)
region_order = ["Central Arctic", "North Atlantic", "North Pacific"]
annual_region_order = [*region_order, "Total (CA+NA+NP)"]
region_masks = {
    "Central Arctic": central_arctic_mask,
    "North Atlantic": north_atlantic_mask,
    "North Pacific": north_pacific_mask,
}

overlap_count = int(
    np.sum(
        central_arctic_mask.astype(int)
        + north_atlantic_mask.astype(int)
        + north_pacific_mask.astype(int)
        > 1
    )
)
if overlap_count != 0:
    raise ValueError(f"三个最终分区存在{overlap_count}个重叠网格")


# %% 5. 逐模式逐月订正并计算两个年平均端点

cell_frames = []
summary_records = []
timeseries_records = []
edge_early_models = []
edge_later_models = []
display_qc_records = []
failure_records = []

for model in model_order:
    print("\n" + "=" * 100)
    print(model)
    print("=" * 100)

    historical_dataset = None
    try:
        (
            historical_dataset,
            historical_data,
            historical_latitude,
            historical_longitude,
            historical_metadata,
        ) = arctic_module.open_historical_model(model)

        historical_mapping = arctic_module.build_or_load_regrid_map(
            model,
            str(historical_metadata["variant"]),
            str(historical_metadata["grid"]),
            historical_latitude,
            historical_longitude,
            observation,
        )
        ramp_up_files = shared_module.experiment_files(
            shared_module.UP_DIR,
            model,
            "1pctCO2",
        )
        ramp_down_files = shared_module.experiment_files(
            shared_module.CDR_DIR,
            model,
            "1pctCO2-cdr",
        )

        annual_sic_sum = np.zeros(
            (simulation_years.size, *domain_mask.shape),
            dtype=np.float32,
        )
        annual_sic_month_count = np.zeros(
            (simulation_years.size, *domain_mask.shape),
            dtype=np.uint8,
        )

        for month_number in calendar_months:
            ramp_up_dataset = None
            ramp_down_dataset = None
            try:
                historical_indices = np.flatnonzero(
                    np.asarray(historical_metadata["months"], dtype=int)
                    == month_number
                )
                model_historical_years = np.asarray(
                    historical_metadata["years"],
                    dtype=int,
                )[historical_indices]
                if not np.array_equal(model_historical_years, historical_years):
                    raise ValueError(
                        f"{model} month={month_number} historical年份不完整"
                    )

                observation_selection = (
                    np.asarray(observation.months, dtype=int) == month_number
                )
                observation_month = np.asarray(
                    observation.values[observation_selection],
                    dtype=np.float32,
                )
                observation_month_years = np.asarray(
                    observation.years[observation_selection],
                    dtype=int,
                )
                if not np.array_equal(observation_month_years, historical_years):
                    raise ValueError(
                        f"month={month_number} NSIDC观测年份不完整"
                    )

                historical_native = shared_module.to_fraction(
                    arctic_module,
                    np.asarray(
                        historical_data.isel(time=historical_indices).load().values,
                        dtype=np.float32,
                    ),
                    dict(historical_metadata["sic_attrs"]),
                    f"Arctic/{model}/historical/{month_number}",
                )
                historical_regridded = arctic_module.regrid_sic_stack(
                    historical_native,
                    historical_mapping,
                )
                historical_regridded[:, ~domain_mask] = np.nan

                mean_ratio, standard_deviation_ratio, stable_mask = (
                    shared_module.fit_factors(
                        historical_regridded,
                        observation_month,
                        domain_mask,
                    )
                )

                (
                    ramp_up_dataset,
                    ramp_up_native,
                    ramp_up_latitude,
                    ramp_up_longitude,
                ) = shared_module.open_selected(
                    arctic_module,
                    ramp_up_files,
                    int(month_number),
                    f"Arctic/{model}/1pctCO2/{month_number}",
                )
                (
                    ramp_down_dataset,
                    ramp_down_native,
                    ramp_down_latitude,
                    ramp_down_longitude,
                ) = shared_module.open_selected(
                    arctic_module,
                    ramp_down_files,
                    int(month_number),
                    f"Arctic/{model}/1pctCO2-cdr/{month_number}",
                )

                ramp_up_mapping = arctic_module.build_or_load_regrid_map(
                    model,
                    str(historical_metadata["variant"]),
                    f"{historical_metadata['grid']}_up",
                    ramp_up_latitude,
                    ramp_up_longitude,
                    observation,
                )
                ramp_down_mapping = arctic_module.build_or_load_regrid_map(
                    model,
                    str(historical_metadata["variant"]),
                    f"{historical_metadata['grid']}_cdr",
                    ramp_down_latitude,
                    ramp_down_longitude,
                    observation,
                )
                ramp_up_regridded = arctic_module.regrid_sic_stack(
                    ramp_up_native[:ramp_years],
                    ramp_up_mapping,
                )
                ramp_down_regridded = arctic_module.regrid_sic_stack(
                    ramp_down_native[:ramp_years],
                    ramp_down_mapping,
                )
                ramp_up_regridded[:, ~domain_mask] = np.nan
                ramp_down_regridded[:, ~domain_mask] = np.nan

                corrected_ramp_up = shared_module.apply_correction(
                    ramp_up_regridded,
                    mean_ratio,
                    standard_deviation_ratio,
                    stable_mask,
                )
                corrected_ramp_down = shared_module.apply_correction(
                    ramp_down_regridded,
                    mean_ratio,
                    standard_deviation_ratio,
                    stable_mask,
                )

                ramp_up_finite = np.isfinite(corrected_ramp_up)
                ramp_down_finite = np.isfinite(corrected_ramp_down)
                annual_sic_sum[:ramp_years] += np.where(
                    ramp_up_finite,
                    corrected_ramp_up,
                    0.0,
                )
                annual_sic_sum[ramp_years:] += np.where(
                    ramp_down_finite,
                    corrected_ramp_down,
                    0.0,
                )
                annual_sic_month_count[:ramp_years] += ramp_up_finite
                annual_sic_month_count[ramp_years:] += ramp_down_finite

                print(
                    f"[完成] month={month_number:02d} | "
                    f"stable={np.sum(stable_mask)}/{np.sum(domain_mask)}"
                )

            finally:
                for dataset in [ramp_up_dataset, ramp_down_dataset]:
                    if dataset is not None:
                        dataset.close()

        common_annual_mask = domain_mask & np.all(
            annual_sic_month_count > 0,
            axis=0,
        )
        if np.sum(common_annual_mask) < 100:
            raise ValueError(f"{model} 280年共同有效订正格点少于100")

        annual_region_sia = {
            region_name: np.zeros(simulation_years.size, dtype=np.float64)
            for region_name in region_order
        }
        early_mean_sic = np.zeros(domain_mask.shape, dtype=np.float64)
        later_mean_sic = np.zeros(domain_mask.shape, dtype=np.float64)

        for year_index, simulation_year in enumerate(simulation_years):
            annual_mean_sic = annual_sic_sum[year_index] / 12.0
            annual_mean_sic[~common_annual_mask] = np.nan

            for region_name in region_order:
                region_mask = region_masks[region_name] & common_annual_mask
                annual_region_sia[region_name][year_index] = (
                    np.nansum(
                        annual_mean_sic[region_mask],
                        dtype=np.float64,
                    )
                    * cell_area_km2
                    / 1.0e6
                )

            if simulation_year <= endpoint_years:
                early_mean_sic += np.nan_to_num(annual_mean_sic) / endpoint_years
            if simulation_year > simulation_years.size - endpoint_years:
                later_mean_sic += np.nan_to_num(annual_mean_sic) / endpoint_years

        early_mean_sic[~common_annual_mask] = np.nan
        later_mean_sic[~common_annual_mask] = np.nan
        common_endpoint = common_annual_mask.copy()
        delta_sic = later_mean_sic - early_mean_sic

        annual_total_sia = (
            annual_region_sia["Central Arctic"]
            + annual_region_sia["North Atlantic"]
            + annual_region_sia["North Pacific"]
        )
        annual_region_sia["Total (CA+NA+NP)"] = annual_total_sia

        for region_name in annual_region_order:
            for year_index, simulation_year in enumerate(simulation_years):
                timeseries_records.append(
                    {
                        "model": model,
                        "region": region_name,
                        "simulation_year": int(simulation_year),
                        "phase": str(phase_labels[year_index]),
                        "co2_ppm": float(co2_ppm[year_index]),
                        "bias_corrected_annual_mean_sia_million_km2": float(
                            annual_region_sia[region_name][year_index]
                        ),
                    }
                )

        del annual_sic_sum
        del annual_sic_month_count

        # 25 km格点直接求和的结果用于定量统计。
        delta_sia_thousand_km2 = delta_sic * cell_area_km2 / 1000.0
        raw_binned_delta = np.full(target_latitude_grid.shape, np.nan, dtype=float)
        binned_sum = np.zeros(target_latitude_grid.shape, dtype=np.float64)
        binned_count = np.zeros(target_latitude_grid.shape, dtype=np.int32)
        valid_cells = (
            domain_mask
            & np.isfinite(delta_sia_thousand_km2)
            & (latitude >= 45.0)
            & (latitude <= 90.0)
        )
        row_indices = np.clip(
            np.floor(latitude[valid_cells] - 45.0).astype(int),
            0,
            target_latitude.size - 1,
        )
        column_indices = np.mod(
            np.floor(longitude_180[valid_cells] + 180.0).astype(int),
            target_longitude.size,
        )
        np.add.at(
            binned_sum,
            (row_indices, column_indices),
            delta_sia_thousand_km2[valid_cells],
        )
        np.add.at(binned_count, (row_indices, column_indices), 1)
        raw_binned_delta[binned_count > 0] = binned_sum[binned_count > 0]

        # 显示场沿用原pattern方法：端点SIC球面IDW到完整1°网格。
        source_valid = (
            common_endpoint
            & np.isfinite(latitude)
            & np.isfinite(longitude_180)
            & (latitude >= 45.0)
            & (latitude <= 90.0)
        )
        if np.sum(source_valid) < 100:
            raise ValueError(f"{model} 可用于display IDW的订正SIC格点少于100")

        source_latitude_radian = np.deg2rad(latitude[source_valid])
        source_longitude_radian = np.deg2rad(longitude_180[source_valid])
        source_xyz = np.column_stack(
            [
                np.cos(source_latitude_radian)
                * np.cos(source_longitude_radian),
                np.cos(source_latitude_radian)
                * np.sin(source_longitude_radian),
                np.sin(source_latitude_radian),
            ]
        )
        source_tree = cKDTree(source_xyz)
        neighbour_count = min(display_idw_neighbours, source_xyz.shape[0])
        chord_distance, source_indices = source_tree.query(
            target_xyz,
            k=neighbour_count,
        )
        if neighbour_count == 1:
            chord_distance = chord_distance[:, None]
            source_indices = source_indices[:, None]

        chord_distance = np.clip(np.asarray(chord_distance, dtype=float), 0.0, 2.0)
        great_circle_distance_km = (
            2.0
            * earth_radius_km
            * np.arcsin(chord_distance / 2.0)
        )
        nearest_distance_km = great_circle_distance_km[:, 0]
        idw_weights = 1.0 / np.maximum(
            great_circle_distance_km,
            1.0e-6,
        ) ** display_idw_power

        early_neighbours = early_mean_sic[source_valid][source_indices]
        later_neighbours = later_mean_sic[source_valid][source_indices]
        early_display_flat = np.sum(
            idw_weights * early_neighbours,
            axis=1,
        ) / np.sum(idw_weights, axis=1)
        later_display_flat = np.sum(
            idw_weights * later_neighbours,
            axis=1,
        ) / np.sum(idw_weights, axis=1)

        exact_target = nearest_distance_km <= 1.0e-6
        early_display_flat[exact_target] = early_neighbours[exact_target, 0]
        later_display_flat[exact_target] = later_neighbours[exact_target, 0]
        outside_display_range = (
            nearest_distance_km > display_max_nearest_distance_km
        )
        early_display_flat[outside_display_range] = np.nan
        later_display_flat[outside_display_range] = np.nan

        early_display_sic = early_display_flat.reshape(target_latitude_grid.shape)
        later_display_sic = later_display_flat.reshape(target_latitude_grid.shape)
        common_display = (
            np.isfinite(early_display_sic)
            & np.isfinite(later_display_sic)
        )
        display_delta_sic = np.where(
            common_display,
            later_display_sic - early_display_sic,
            np.nan,
        )
        provisional_display_delta = np.where(
            common_display,
            display_delta_sic * target_area_m2 / 1.0e9,
            0.0,
        )

        # 正、负贡献分别闭合到25 km直接求和结果，插值只改善显示连续性。
        raw_finite = np.where(
            np.isfinite(raw_binned_delta),
            raw_binned_delta,
            0.0,
        )
        raw_positive = np.clip(raw_finite, 0.0, None)
        raw_negative = np.clip(-raw_finite, 0.0, None)
        provisional_positive = np.clip(provisional_display_delta, 0.0, None)
        provisional_negative = np.clip(-provisional_display_delta, 0.0, None)

        raw_positive_sum = float(np.sum(raw_positive, dtype=np.float64))
        raw_negative_sum = float(np.sum(raw_negative, dtype=np.float64))
        provisional_positive_sum = float(
            np.sum(provisional_positive, dtype=np.float64)
        )
        provisional_negative_sum = float(
            np.sum(provisional_negative, dtype=np.float64)
        )
        if raw_positive_sum > 0.0 and provisional_positive_sum <= 0.0:
            raise ValueError(f"{model} display正贡献无法闭合")
        if raw_negative_sum > 0.0 and provisional_negative_sum <= 0.0:
            raise ValueError(f"{model} display负贡献无法闭合")

        positive_scale = 0.0
        negative_scale = 0.0
        if raw_positive_sum > 0.0:
            positive_scale = raw_positive_sum / provisional_positive_sum
        if raw_negative_sum > 0.0:
            negative_scale = raw_negative_sum / provisional_negative_sum

        display_delta = (
            provisional_positive * positive_scale
            - provisional_negative * negative_scale
        )
        display_delta[~common_display] = np.nan
        early_display_sic[~common_display] = np.nan
        later_display_sic[~common_display] = np.nan

        edge_early_models.append(early_display_sic.astype(np.float32))
        edge_later_models.append(later_display_sic.astype(np.float32))

        display_positive_sum = float(
            np.nansum(np.clip(display_delta, 0.0, None), dtype=np.float64)
        )
        display_negative_sum = float(
            np.nansum(np.clip(-display_delta, 0.0, None), dtype=np.float64)
        )
        raw_net_sum = raw_positive_sum - raw_negative_sum
        display_net_sum = display_positive_sum - display_negative_sum
        accepted_distance = nearest_distance_km[
            nearest_distance_km <= display_max_nearest_distance_km
        ]
        display_qc_records.append(
            {
                "model": model,
                "raw_positive_sum_thousand_km2": raw_positive_sum,
                "display_positive_sum_thousand_km2": display_positive_sum,
                "positive_closure_error_thousand_km2": (
                    display_positive_sum - raw_positive_sum
                ),
                "raw_negative_sum_thousand_km2": raw_negative_sum,
                "display_negative_sum_thousand_km2": display_negative_sum,
                "negative_closure_error_thousand_km2": (
                    display_negative_sum - raw_negative_sum
                ),
                "raw_net_sum_thousand_km2": raw_net_sum,
                "display_net_sum_thousand_km2": display_net_sum,
                "net_closure_error_thousand_km2": (
                    display_net_sum - raw_net_sum
                ),
                "display_finite_bin_count": int(np.sum(common_display)),
                "display_total_bin_count": int(common_display.size),
                "nearest_distance_p95_km": float(
                    np.nanpercentile(accepted_distance, 95.0)
                ),
                "nearest_distance_max_km": float(np.nanmax(accepted_distance)),
                "positive_scale_factor": positive_scale,
                "negative_scale_factor": negative_scale,
            }
        )

        cell_frames.append(
            pd.DataFrame(
                {
                    "model": model,
                    "latitude_deg_north": target_latitude_grid.ravel(),
                    "longitude_deg_east": target_longitude_grid.ravel(),
                    "bias_corrected_raw_delta_sia_thousand_km2": (
                        raw_binned_delta.ravel()
                    ),
                    "bias_corrected_display_delta_sia_thousand_km2": (
                        display_delta.ravel()
                    ),
                }
            )
        )

        model_region_values = {}
        for region_name in region_order:
            region_mask = region_masks[region_name] & common_endpoint
            early_sia = float(
                np.nansum(early_mean_sic[region_mask], dtype=np.float64)
                * cell_area_km2
                / 1.0e6
            )
            later_sia = float(
                np.nansum(later_mean_sic[region_mask], dtype=np.float64)
                * cell_area_km2
                / 1.0e6
            )
            delta_sia = later_sia - early_sia
            model_region_values[region_name] = {
                "early": early_sia,
                "later": later_sia,
                "delta": delta_sia,
            }
            summary_records.append(
                {
                    "model": model,
                    "region": region_name,
                    "bias_corrected_early_sia_million_km2": early_sia,
                    "bias_corrected_later_sia_million_km2": later_sia,
                    "bias_corrected_delta_sia_million_km2": delta_sia,
                }
            )

        total_early_sia = (
            model_region_values["Central Arctic"]["early"]
            + model_region_values["North Atlantic"]["early"]
            + model_region_values["North Pacific"]["early"]
        )
        total_later_sia = (
            model_region_values["Central Arctic"]["later"]
            + model_region_values["North Atlantic"]["later"]
            + model_region_values["North Pacific"]["later"]
        )
        total_delta_sia = total_later_sia - total_early_sia

        summary_records.append(
            {
                "model": model,
                "region": "Total (CA+NA+NP)",
                "bias_corrected_early_sia_million_km2": total_early_sia,
                "bias_corrected_later_sia_million_km2": total_later_sia,
                "bias_corrected_delta_sia_million_km2": total_delta_sia,
            }
        )

    except Exception as error:
        failure_records.append(
            {
                "model": model,
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        print(f"[失败] {model}: {type(error).__name__}: {error}")

    finally:
        if historical_dataset is not None:
            historical_dataset.close()

if failure_records:
    raise RuntimeError(
        "九模式bias-corrected pattern没有全部完成：\n"
        + pd.DataFrame(failure_records).to_string(index=False)
    )


# %% 6. 计算年平均序列、平滑和模式集合统计

summary = pd.DataFrame(summary_records)
timeseries = pd.DataFrame(timeseries_records)
expected_individual_rows = (
    len(model_order) * len(annual_region_order) * simulation_years.size
)
if len(timeseries) != expected_individual_rows:
    raise ValueError(
        f"年平均序列行数错误：{len(timeseries)} != {expected_individual_rows}"
    )

timeseries = timeseries.sort_values(
    ["region", "model", "simulation_year"]
).reset_index(drop=True)
timeseries["bias_corrected_annual_mean_sia_11yr_mean_million_km2"] = np.nan
timeseries["intermodel_sd_11yr_mean_million_km2"] = np.nan
timeseries["series_type"] = "individual model"

for region_name in annual_region_order:
    for model in model_order:
        selection = (
            (timeseries["region"] == region_name)
            & (timeseries["model"] == model)
        )
        annual_values = timeseries.loc[
            selection,
            "bias_corrected_annual_mean_sia_million_km2",
        ]
        timeseries.loc[
            selection,
            "bias_corrected_annual_mean_sia_11yr_mean_million_km2",
        ] = annual_values.rolling(
            11,
            center=True,
            min_periods=6,
        ).mean().to_numpy(float)

mme_timeseries = (
    timeseries.groupby(
        ["region", "simulation_year", "phase"],
        as_index=False,
    )
    .agg(
        co2_ppm=("co2_ppm", "first"),
        bias_corrected_annual_mean_sia_million_km2=(
            "bias_corrected_annual_mean_sia_million_km2",
            "mean",
        ),
        bias_corrected_annual_mean_sia_11yr_mean_million_km2=(
            "bias_corrected_annual_mean_sia_11yr_mean_million_km2",
            "mean",
        ),
        intermodel_sd_11yr_mean_million_km2=(
            "bias_corrected_annual_mean_sia_11yr_mean_million_km2",
            "std",
        ),
    )
)
mme_timeseries["model"] = "9-model MME"
mme_timeseries["series_type"] = "multi-model mean"
mme_timeseries = mme_timeseries[timeseries.columns]
timeseries = pd.concat([timeseries, mme_timeseries], ignore_index=True)
timeseries = timeseries.sort_values(
    ["region", "series_type", "model", "simulation_year"]
).reset_index(drop=True)


# %% 7. 检查pattern端点与年平均序列的一致性

individual_timeseries = timeseries.loc[
    timeseries["series_type"] == "individual model"
].copy()
early_timeseries = (
    individual_timeseries.loc[individual_timeseries["simulation_year"] <= 10]
    .groupby(["model", "region"], as_index=False)[
        "bias_corrected_annual_mean_sia_million_km2"
    ]
    .mean()
    .rename(
        columns={
            "bias_corrected_annual_mean_sia_million_km2": (
                "timeseries_early_sia_million_km2"
            )
        }
    )
)
later_timeseries = (
    individual_timeseries.loc[individual_timeseries["simulation_year"] >= 271]
    .groupby(["model", "region"], as_index=False)[
        "bias_corrected_annual_mean_sia_million_km2"
    ]
    .mean()
    .rename(
        columns={
            "bias_corrected_annual_mean_sia_million_km2": (
                "timeseries_later_sia_million_km2"
            )
        }
    )
)
consistency_qc = summary.merge(
    early_timeseries,
    on=["model", "region"],
    how="left",
    validate="one_to_one",
).merge(
    later_timeseries,
    on=["model", "region"],
    how="left",
    validate="one_to_one",
)
consistency_qc["timeseries_delta_sia_million_km2"] = (
    consistency_qc["timeseries_later_sia_million_km2"]
    - consistency_qc["timeseries_early_sia_million_km2"]
)
consistency_qc["timeseries_minus_pattern_delta_million_km2"] = (
    consistency_qc["timeseries_delta_sia_million_km2"]
    - consistency_qc["bias_corrected_delta_sia_million_km2"]
)
maximum_pattern_timeseries_error = float(
    consistency_qc["timeseries_minus_pattern_delta_million_km2"].abs().max()
)
if maximum_pattern_timeseries_error > pattern_timeseries_tolerance_million_km2:
    raise ValueError(
        "年平均序列端点与pattern数值不一致："
        f"{maximum_pattern_timeseries_error:.3e} million km²；"
        "允许误差="
        f"{pattern_timeseries_tolerance_million_km2:.1e} million km²"
    )


# %% 8. 检查并保存正式pattern和序列结果

cells = pd.concat(cell_frames, ignore_index=True)
display_qc = pd.DataFrame(display_qc_records)
edge_early = np.stack(edge_early_models, axis=0)
edge_later = np.stack(edge_later_models, axis=0)

expected_cell_rows = len(model_order) * target_latitude.size * target_longitude.size
if len(cells) != expected_cell_rows:
    raise ValueError(f"pattern cells行数错误：{len(cells)} != {expected_cell_rows}")
expected_edge_shape = (len(model_order), *target_latitude_grid.shape)
if edge_early.shape != expected_edge_shape:
    raise ValueError(f"early edge数组维度错误：{edge_early.shape}")
if edge_later.shape != expected_edge_shape:
    raise ValueError(f"later edge数组维度错误：{edge_later.shape}")
if len(display_qc) != len(model_order):
    raise ValueError(f"display QC模式数错误：{len(display_qc)}")

maximum_closure_error = float(
    display_qc[
        [
            "positive_closure_error_thousand_km2",
            "negative_closure_error_thousand_km2",
            "net_closure_error_thousand_km2",
        ]
    ].abs().to_numpy().max()
)
if maximum_closure_error > 1.0e-8:
    raise ValueError(
        f"display与25 km直接求和结果未闭合：{maximum_closure_error:.3e}"
    )

cells.to_csv(
    cell_output,
    index=False,
    encoding="utf-8-sig",
    float_format="%.10g",
)
summary.to_csv(
    summary_output,
    index=False,
    encoding="utf-8-sig",
    float_format="%.10g",
)
display_qc.to_csv(
    display_qc_output,
    index=False,
    encoding="utf-8-sig",
    float_format="%.10g",
)
timeseries.to_csv(
    timeseries_output,
    index=False,
    encoding="utf-8-sig",
    float_format="%.10g",
)
consistency_qc.to_csv(
    consistency_qc_output,
    index=False,
    encoding="utf-8-sig",
    float_format="%.10g",
)
np.savez_compressed(
    edge_output,
    models=np.asarray(model_order, dtype="U32"),
    latitude_deg_north=target_latitude.astype(np.float32),
    longitude_deg_east=target_longitude.astype(np.float32),
    early_bias_corrected_sic_fraction=edge_early,
    later_bias_corrected_sic_fraction=edge_later,
)

metadata = {
    "analysis": "Arctic annual-mean bias-corrected SIA later-minus-early pattern",
    "models": model_order,
    "historical_training_period": "1979-2014",
    "early": "1pctCO2 experiment years 1-10, all 12 calendar months",
    "later": "1pctCO2-cdr experiment years 131-140, all 12 calendar months",
    "difference": "later minus early",
    "annual_timeseries": (
        "simulation years 1-140 are 1pctCO2 years 1-140; simulation years "
        "141-280 are 1pctCO2-cdr years 1-140; each year is the equal mean "
        "of 12 monthly bias-corrected regional SIA values; cells without a "
        "stable correction in a month remain excluded from that monthly sum"
    ),
    "timeseries_smoothing": "11-year centered running mean, minimum 6 years",
    "pattern_timeseries_consistency_tolerance_million_km2": (
        pattern_timeseries_tolerance_million_km2
    ),
    "bias_correction": (
        "Monthly grid-cell M*=(M-Mbar_11yr)*sigma_obs/sigma_model+"
        "Mbar_11yr*mean_obs/mean_model"
    ),
    "piControl_drift_correction": False,
    "grid": "NOAA/NSIDC G02202 V6 25 km before conservative 1-degree binning",
    "cell_area_km2": cell_area_km2,
    "color_limit_thousand_km2": color_limit_thousand_km2,
    "display_method": {
        "role": "visualization only; regional values use direct 25 km sums",
        "method": (
            "spherical IDW endpoint SIC to full 1-degree grid, delta SIC times "
            "exact target-cell area, positive/negative sign-conservative scaling"
        ),
        "idw_neighbours": display_idw_neighbours,
        "idw_power": display_idw_power,
        "max_nearest_distance_km": display_max_nearest_distance_km,
    },
    "regions": {
        "Central Arctic": "70-90N, all longitudes",
        "North Atlantic": "45-70N, 60W-30E",
        "North Pacific": "45-70N, 120E-150W",
        "Total (CA+NA+NP)": "arithmetic sum of the three regions",
    },
}
metadata_output.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


# %% 9. 输出运行摘要

print("=" * 96)
print("北极bias-corrected年平均SIA pattern：later − early")
print("未使用piControl，未做漂移订正")
print("=" * 96)
print(f"[pattern cells] {cell_output}")
print(f"[region summary] {summary_output}")
print(f"[corrected edges] {edge_output}")
print(f"[display QC] {display_qc_output}")
print(f"[annual timeseries] {timeseries_output}")
print(f"[pattern consistency QC] {consistency_qc_output}")
print(f"[metadata] {metadata_output}")
