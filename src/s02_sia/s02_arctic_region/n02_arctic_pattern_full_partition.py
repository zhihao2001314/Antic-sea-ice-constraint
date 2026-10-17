# %% 1. 导入依赖

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter


# %% 2. 参数与路径设置

project_root = Path(r"F:\sea\Antic sea ice constraint")
output_directory = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p02_pattern_region"
    / "n02"
)

pattern_input_path = (
    project_root
    / "data"
    / "s02_sia"
    / "s02_arctic_region"
    / "arctic_sia_pattern_values_1deg.csv"
)
cell_output_path = output_directory / "arctic_pattern_full_partition_cells.csv"
summary_output_path = (
    output_directory / "arctic_pattern_full_partition_summary.csv"
)
quality_output_path = output_directory / "arctic_pattern_full_partition_qc.csv"
dominance_output_path = output_directory / "arctic_pattern_na_arctic_sign_check.csv"
temperature_output_path = (
    output_directory / "arctic_pattern_region_temperature_summary.csv"
)
observed_edge_output_path = (
    output_directory / "observed_sic_1988_2020_climatology_edge.npz"
)

tas_field_input_directory = (
    project_root / "output" / "s03tem_n01"
)
tas_summary_input_path = (
    tas_field_input_directory / "arctic_tas_pattern_summary.csv"
)
thetao_field_input_path = (
    project_root
    / "output"
    / "s03_n02"
    / "arctic_thetao_pattern_fields.npz"
)
observed_sic_directory = Path(r"F:\sea\seaice-obs")

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

outer_boundary_latitude = 45.0
north_atlantic_latitude_north = 80.0
north_pacific_latitude_north = 65.0
north_atlantic_longitude_west = -120.0
north_atlantic_longitude_east = 90.0
north_pacific_longitude_east = 90.0
north_pacific_longitude_west = -120.0

sea_ice_concentration_threshold = 0.15

observed_start_year = 1988
observed_end_year = 2020
observed_samples_per_year = 183
observed_valid_sample_fraction = 0.50
observed_valid_year_fraction = 0.80
observed_read_chunk_size = 8
observed_smoothing_sigma_grid_cells = 0.80
observed_minimum_smoothing_weight = 0.02
observed_grid_shape = (448, 304)
observed_grid_cell_size_m = 25_000.0
observed_lower_left_corner_m = (-3_850_000.0, -5_350_000.0)


# %% 3. 读取已有pattern格点结果

pattern_input_columns = [
    "model",
    "latitude_deg_north",
    "longitude_deg_east",
    "corrected_delta_sia_raw_thousand_km2",
    "corrected_delta_sia_display_thousand_km2",
    "edge_picontrol_adjusted_co2_sic_fraction",
    "edge_picontrol_adjusted_cdr_sic_fraction",
]
pattern_dataframe = pd.read_csv(
    pattern_input_path,
    usecols=pattern_input_columns,
)
pattern_dataframe = pattern_dataframe.loc[
    (pattern_dataframe["latitude_deg_north"] >= outer_boundary_latitude)
    & (pattern_dataframe["latitude_deg_north"] <= 90.0)
].copy()
pattern_dataframe["longitude_deg_east"] = (
    (pattern_dataframe["longitude_deg_east"] + 180.0) % 360.0
) - 180.0

full_grid_latitude = np.arange(45.5, 90.0, 1.0)
full_grid_longitude = np.arange(-179.5, 180.0, 1.0)
full_grid_index = pd.MultiIndex.from_product(
    [full_grid_latitude, full_grid_longitude],
    names=["latitude_deg_north", "longitude_deg_east"],
)

complete_model_dataframes = []

for model_name in model_order:
    model_dataframe = pattern_dataframe.loc[
        pattern_dataframe["model"] == model_name
    ].set_index(["latitude_deg_north", "longitude_deg_east"])
    model_dataframe = model_dataframe.reindex(full_grid_index).reset_index()
    model_dataframe["model"] = model_name
    model_dataframe[
        [
            "corrected_delta_sia_raw_thousand_km2",
            "corrected_delta_sia_display_thousand_km2",
            "edge_picontrol_adjusted_co2_sic_fraction",
            "edge_picontrol_adjusted_cdr_sic_fraction",
        ]
    ] = model_dataframe[
        [
            "corrected_delta_sia_raw_thousand_km2",
            "corrected_delta_sia_display_thousand_km2",
            "edge_picontrol_adjusted_co2_sic_fraction",
            "edge_picontrol_adjusted_cdr_sic_fraction",
        ]
    ].fillna(0.0)
    complete_model_dataframes.append(model_dataframe)

pattern_dataframe = pd.concat(
    complete_model_dataframes,
    ignore_index=True,
)
pattern_dataframe = pattern_dataframe.sort_values(
    ["model", "latitude_deg_north", "longitude_deg_east"]
).reset_index(drop=True)


# %% 4. 建立完整覆盖45–90°N的三个互斥区域

latitude = pattern_dataframe["latitude_deg_north"]
longitude = pattern_dataframe["longitude_deg_east"]

north_atlantic_mask = (
    (latitude >= outer_boundary_latitude)
    & (latitude < north_atlantic_latitude_north)
    & (longitude >= north_atlantic_longitude_west)
    & (longitude < north_atlantic_longitude_east)
)
north_pacific_mask = (
    (latitude >= outer_boundary_latitude)
    & (latitude < north_pacific_latitude_north)
    & (
        (longitude >= north_pacific_longitude_east)
        | (longitude < north_pacific_longitude_west)
    )
)
central_arctic_mask = ~(north_atlantic_mask | north_pacific_mask)

pattern_dataframe["partition_region"] = ""
pattern_dataframe.loc[central_arctic_mask, "partition_region"] = (
    "Central Arctic"
)
pattern_dataframe.loc[north_atlantic_mask, "partition_region"] = (
    "North Atlantic"
)
pattern_dataframe.loc[north_pacific_mask, "partition_region"] = (
    "North Pacific"
)

pattern_dataframe["partition_region_abbreviation"] = ""
pattern_dataframe.loc[central_arctic_mask, "partition_region_abbreviation"] = "CA"
pattern_dataframe.loc[north_atlantic_mask, "partition_region_abbreviation"] = "NA"
pattern_dataframe.loc[north_pacific_mask, "partition_region_abbreviation"] = "NP"


# %% 5. 识别三区海冰格点

initial_sea_ice_mask = (
    pattern_dataframe["edge_picontrol_adjusted_co2_sic_fraction"]
    >= sea_ice_concentration_threshold
)
final_sea_ice_mask = (
    pattern_dataframe["edge_picontrol_adjusted_cdr_sic_fraction"]
    >= sea_ice_concentration_threshold
)
pattern_dataframe["initial_sea_ice_mask"] = initial_sea_ice_mask
pattern_dataframe["final_sea_ice_mask"] = final_sea_ice_mask
pattern_dataframe["sea_ice_union_mask"] = (
    initial_sea_ice_mask | final_sea_ice_mask
)
pattern_dataframe["partition_sea_ice_grid_point"] = pattern_dataframe[
    "sea_ice_union_mask"
]


# %% 6. 计算三区变化量和完整北极之和

region_order = ["North Atlantic", "Central Arctic", "North Pacific"]
summary_rows = []
quality_rows = []

for model_name in model_order:
    model_dataframe = pattern_dataframe.loc[
        pattern_dataframe["model"] == model_name
    ]

    regional_delta_values = []
    regional_absolute_values = []

    for region_name in region_order:
        region_dataframe = model_dataframe.loc[
            model_dataframe["partition_region"] == region_name
        ]
        delta_sia_million_km2 = (
            region_dataframe[
                "corrected_delta_sia_raw_thousand_km2"
            ].sum()
            / 1000.0
        )
        regional_delta_values.append(delta_sia_million_km2)
        regional_absolute_values.append(abs(delta_sia_million_km2))

        summary_rows.append(
            {
                "model": model_name,
                "region": region_name,
                "delta_sia_million_km2": delta_sia_million_km2,
                "region_grid_cell_count": int(region_dataframe.shape[0]),
                "sea_ice_union_grid_cell_count": int(
                    region_dataframe["sea_ice_union_mask"].sum()
                ),
            }
        )

    arctic_total_delta = float(np.sum(regional_delta_values))
    full_arctic_delta = (
        model_dataframe["corrected_delta_sia_raw_thousand_km2"].sum()
        / 1000.0
    )
    absolute_regional_sum = float(np.sum(regional_absolute_values))

    for summary_row in summary_rows[-3:]:
        summary_row["absolute_change_share_of_three_regions_percent"] = (
            abs(summary_row["delta_sia_million_km2"])
            / absolute_regional_sum
            * 100.0
        )

    summary_rows.append(
        {
            "model": model_name,
            "region": "Arctic total (NA+CA+NP)",
            "delta_sia_million_km2": arctic_total_delta,
            "region_grid_cell_count": int(model_dataframe.shape[0]),
            "sea_ice_union_grid_cell_count": int(
                model_dataframe["partition_sea_ice_grid_point"].sum()
            ),
            "absolute_change_share_of_three_regions_percent": 100.0,
        }
    )
    region_membership_count = (
        central_arctic_mask.loc[model_dataframe.index].astype(int)
        + north_atlantic_mask.loc[model_dataframe.index].astype(int)
        + north_pacific_mask.loc[model_dataframe.index].astype(int)
    )
    quality_rows.append(
        {
            "model": model_name,
            "grid_shape": "45 x 360",
            "grid_cell_count": int(model_dataframe.shape[0]),
            "regional_overlap_cell_count": int(
                (region_membership_count > 1).sum()
            ),
            "unassigned_grid_cell_count": int(
                (model_dataframe["partition_region"] == "").sum()
            ),
            "three_region_sum_million_km2": arctic_total_delta,
            "full_arctic_delta_million_km2": full_arctic_delta,
            "closure_error_million_km2": (
                arctic_total_delta - full_arctic_delta
            ),
        }
    )

summary_dataframe = pd.DataFrame(summary_rows)
quality_dataframe = pd.DataFrame(quality_rows)


# %% 7. 汇总北大西洋与北极总变化方向

dominance_rows = []

for model_name in model_order:
    model_summary = summary_dataframe.loc[
        summary_dataframe["model"] == model_name
    ]
    north_atlantic_change = float(
        model_summary.loc[
            model_summary["region"] == "North Atlantic",
            "delta_sia_million_km2",
        ].iloc[0]
    )
    central_arctic_change = float(
        model_summary.loc[
            model_summary["region"] == "Central Arctic",
            "delta_sia_million_km2",
        ].iloc[0]
    )
    north_pacific_change = float(
        model_summary.loc[
            model_summary["region"] == "North Pacific",
            "delta_sia_million_km2",
        ].iloc[0]
    )
    arctic_total_change = float(
        model_summary.loc[
            model_summary["region"] == "Arctic total (NA+CA+NP)",
            "delta_sia_million_km2",
        ].iloc[0]
    )

    regional_absolute_change = {
        "North Atlantic": abs(north_atlantic_change),
        "Central Arctic": abs(central_arctic_change),
        "North Pacific": abs(north_pacific_change),
    }
    dominant_region = max(
        regional_absolute_change,
        key=regional_absolute_change.get,
    )
    same_sign = bool(
        np.sign(north_atlantic_change) == np.sign(arctic_total_change)
    )

    dominance_rows.append(
        {
            "model": model_name,
            "na_delta_sia_million_km2": north_atlantic_change,
            "ca_delta_sia_million_km2": central_arctic_change,
            "np_delta_sia_million_km2": north_pacific_change,
            "arctic_total_delta_sia_million_km2": arctic_total_change,
            "dominant_region": dominant_region,
            "na_is_dominant": dominant_region == "North Atlantic",
            "na_arctic_same_sign": same_sign,
            "na_dominant_and_same_sign": (
                dominant_region == "North Atlantic" and same_sign
            ),
            "quantitative_sia_field": (
                "corrected_delta_sia_raw_thousand_km2"
            ),
        }
    )

dominance_dataframe = pd.DataFrame(dominance_rows)


# %% 8. 计算三区海冰域的表层海温和近地面气温变化

thetao_fields = np.load(thetao_field_input_path)
tas_summary_dataframe = pd.read_csv(tas_summary_input_path)

temperature_latitude = np.asarray(thetao_fields["latitude"], dtype=float)
temperature_longitude = np.asarray(thetao_fields["longitude"], dtype=float)

if temperature_latitude.shape != (45, 360):
    raise ValueError(
        f"温度输出应为45×360，实际={temperature_latitude.shape}"
    )

temperature_rows = []
temperature_area_weight = np.cos(np.deg2rad(temperature_latitude))

for model_name in model_order:
    model_key = model_name.replace("-", "_")
    tas_model_input_path = (
        tas_field_input_directory
        / f"arctic_tas_{model_name}_pattern_1deg.npz"
    )
    tas_model_fields = np.load(tas_model_input_path)
    tas_latitude = np.asarray(tas_model_fields["lat"], dtype=float)
    tas_longitude = np.asarray(tas_model_fields["lon"], dtype=float)
    tas_change = np.asarray(
        tas_model_fields["corrected_difference"],
        dtype=float,
    )
    thetao_change = np.asarray(
        thetao_fields[f"corrected__{model_key}"],
        dtype=float,
    )

    if not np.array_equal(tas_latitude, temperature_latitude):
        raise ValueError(f"{model_name} tas与thetao纬度网格不一致")

    if not np.array_equal(tas_longitude, temperature_longitude):
        raise ValueError(f"{model_name} tas与thetao经度网格不一致")

    tas_model_summary = tas_summary_dataframe.loc[
        tas_summary_dataframe["model"] == model_name
    ].iloc[0]

    model_dataframe = pattern_dataframe.loc[
        pattern_dataframe["model"] == model_name
    ].sort_values(["latitude_deg_north", "longitude_deg_east"])
    model_latitude = model_dataframe["latitude_deg_north"].to_numpy().reshape(45, 360)
    model_longitude = model_dataframe["longitude_deg_east"].to_numpy().reshape(45, 360)

    if not np.array_equal(model_latitude, temperature_latitude):
        raise ValueError(f"{model_name} SIA与温度纬度网格不一致")

    if not np.array_equal(model_longitude, temperature_longitude):
        raise ValueError(f"{model_name} SIA与温度经度网格不一致")

    sea_ice_union_grid = model_dataframe[
        "sea_ice_union_mask"
    ].to_numpy(dtype=bool).reshape(45, 360)
    region_grid = model_dataframe[
        "partition_region"
    ].to_numpy().reshape(45, 360)

    for region_name in region_order:
        region_sea_ice_mask = (
            (region_grid == region_name) & sea_ice_union_grid
        )
        tas_valid = region_sea_ice_mask & np.isfinite(tas_change)
        thetao_valid = region_sea_ice_mask & np.isfinite(thetao_change)

        delta_tas_deg_c = float(
            np.sum(tas_change[tas_valid] * temperature_area_weight[tas_valid])
            / np.sum(temperature_area_weight[tas_valid])
        )
        delta_surface_thetao_deg_c = float(
            np.sum(
                thetao_change[thetao_valid]
                * temperature_area_weight[thetao_valid]
            )
            / np.sum(temperature_area_weight[thetao_valid])
        )

        temperature_rows.append(
            {
                "model": model_name,
                "region": region_name,
                "delta_tas_degC": delta_tas_deg_c,
                "delta_surface_layer_thetao_degC": (
                    delta_surface_thetao_deg_c
                ),
                "tas_valid_sea_ice_grid_cell_count": int(tas_valid.sum()),
                "thetao_valid_sea_ice_grid_cell_count": int(
                    thetao_valid.sum()
                ),
                "temperature_spatial_mask": (
                    "initial_or_final_SIC_ge_15pct_within_region"
                ),
                "tas_cdr_window_year_index": tas_model_summary[
                    "cdr_window_year_index"
                ],
                "tas_vertical_definition": tas_model_summary[
                    "tas_vertical_definition"
                ],
                "tas_level_name": tas_model_summary["tas_level_name"],
                "tas_level_value": tas_model_summary["tas_level_value"],
                "tas_level_units": tas_model_summary["tas_level_units"],
            }
        )

temperature_dataframe = pd.DataFrame(temperature_rows)
summary_dataframe = summary_dataframe.merge(
    temperature_dataframe[
        [
            "model",
            "region",
            "delta_tas_degC",
            "delta_surface_layer_thetao_degC",
            "tas_cdr_window_year_index",
            "tas_vertical_definition",
            "tas_level_name",
            "tas_level_value",
            "tas_level_units",
        ]
    ],
    how="left",
    on=["model", "region"],
)


# %% 9. 计算1988–2020观测海冰气候态边缘场

observed_years = list(range(observed_start_year, observed_end_year + 1))
observed_climatology_sum = np.zeros(observed_grid_shape, dtype=np.float64)
observed_climatology_count = np.zeros(observed_grid_shape, dtype=np.int16)

for observed_year in observed_years:
    observed_year_path = (
        observed_sic_directory / f"SIC_{observed_year}.npy"
    )
    observed_year_array = np.load(
        observed_year_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    if observed_year_array.shape != (
        observed_samples_per_year,
        *observed_grid_shape,
    ):
        raise ValueError(
            f"{observed_year_path.name}形状异常：{observed_year_array.shape}"
        )

    if str(observed_year_array.dtype) != "int32":
        raise ValueError(
            f"{observed_year_path.name}应为int32，实际={observed_year_array.dtype}"
        )

    observed_value_sum = np.zeros(observed_grid_shape, dtype=np.float64)
    observed_valid_count = np.zeros(observed_grid_shape, dtype=np.int32)

    for chunk_start in range(
        0,
        observed_samples_per_year,
        observed_read_chunk_size,
    ):
        chunk_stop = min(
            chunk_start + observed_read_chunk_size,
            observed_samples_per_year,
        )
        observed_raw = np.asarray(
            observed_year_array[chunk_start:chunk_stop],
            dtype=np.float64,
        )
        observed_valid = (
            np.isfinite(observed_raw)
            & (observed_raw >= 0.0)
            & (observed_raw <= 250.5)
        )
        observed_fraction = observed_raw / 250.0
        observed_value_sum += np.where(
            observed_valid,
            observed_fraction,
            0.0,
        ).sum(axis=0)
        observed_valid_count += observed_valid.sum(axis=0, dtype=np.int32)

    observed_minimum_samples = int(
        np.ceil(
            observed_samples_per_year * observed_valid_sample_fraction
        )
    )
    observed_annual_mean = np.full(
        observed_grid_shape,
        np.nan,
        dtype=float,
    )
    observed_annual_valid = (
        observed_valid_count >= observed_minimum_samples
    )
    observed_annual_mean[observed_annual_valid] = (
        observed_value_sum[observed_annual_valid]
        / observed_valid_count[observed_annual_valid]
    )

    observed_climatology_sum += np.where(
        observed_annual_valid,
        observed_annual_mean,
        0.0,
    )
    observed_climatology_count += observed_annual_valid.astype(np.int16)

observed_minimum_years = int(
    np.ceil(len(observed_years) * observed_valid_year_fraction)
)
observed_climatology = np.full(
    observed_grid_shape,
    np.nan,
    dtype=float,
)
observed_climatology_valid = (
    observed_climatology_count >= observed_minimum_years
)
observed_climatology[observed_climatology_valid] = (
    observed_climatology_sum[observed_climatology_valid]
    / observed_climatology_count[observed_climatology_valid]
)

observed_smooth_values = gaussian_filter(
    np.where(observed_climatology_valid, observed_climatology, 0.0),
    sigma=observed_smoothing_sigma_grid_cells,
    mode="constant",
    cval=0.0,
)
observed_smooth_weights = gaussian_filter(
    observed_climatology_valid.astype(float),
    sigma=observed_smoothing_sigma_grid_cells,
    mode="constant",
    cval=0.0,
)
observed_smoothed_climatology = np.full(
    observed_grid_shape,
    np.nan,
    dtype=float,
)
observed_smoothed_valid = (
    observed_smooth_weights >= observed_minimum_smoothing_weight
)
observed_smoothed_climatology[observed_smoothed_valid] = np.clip(
    observed_smooth_values[observed_smoothed_valid]
    / observed_smooth_weights[observed_smoothed_valid],
    0.0,
    1.0,
)

observed_x_minimum_m, observed_y_minimum_m = (
    observed_lower_left_corner_m
)
observed_x_centers_m = (
    observed_x_minimum_m
    + 0.5 * observed_grid_cell_size_m
    + np.arange(observed_grid_shape[1]) * observed_grid_cell_size_m
)
observed_y_maximum_m = (
    observed_y_minimum_m
    + observed_grid_shape[0] * observed_grid_cell_size_m
)
observed_y_centers_m = (
    observed_y_maximum_m
    - 0.5 * observed_grid_cell_size_m
    - np.arange(observed_grid_shape[0]) * observed_grid_cell_size_m
)
observed_x_grid_m, observed_y_grid_m = np.meshgrid(
    observed_x_centers_m,
    observed_y_centers_m,
)


# %% 10. 保存计算结果和QC

cell_output_columns = [
    "model",
    "latitude_deg_north",
    "longitude_deg_east",
    "partition_region",
    "partition_region_abbreviation",
    "corrected_delta_sia_raw_thousand_km2",
    "corrected_delta_sia_display_thousand_km2",
    "edge_picontrol_adjusted_co2_sic_fraction",
    "edge_picontrol_adjusted_cdr_sic_fraction",
    "initial_sea_ice_mask",
    "final_sea_ice_mask",
    "sea_ice_union_mask",
    "partition_sea_ice_grid_point",
]

pattern_dataframe[cell_output_columns].to_csv(
    cell_output_path,
    index=False,
    encoding="utf-8-sig",
)
summary_dataframe.to_csv(
    summary_output_path,
    index=False,
    encoding="utf-8-sig",
)
quality_dataframe.to_csv(
    quality_output_path,
    index=False,
    encoding="utf-8-sig",
)
dominance_dataframe.to_csv(
    dominance_output_path,
    index=False,
    encoding="utf-8-sig",
)
temperature_dataframe.to_csv(
    temperature_output_path,
    index=False,
    encoding="utf-8-sig",
)
np.savez_compressed(
    observed_edge_output_path,
    projected_x_m=observed_x_grid_m,
    projected_y_m=observed_y_grid_m,
    sic_climatology_fraction=observed_climatology,
    sic_smoothed_for_edge_fraction=observed_smoothed_climatology,
    valid_year_count=observed_climatology_count,
)

print(f"框选区域格点：{cell_output_path}")
print(f"三区变化量：{summary_output_path}")
print(f"质量检查：{quality_output_path}")
print(f"北大西洋主导诊断：{dominance_output_path}")
print(f"三区温度变化：{temperature_output_path}")
print(f"观测海冰气候态边缘场：{observed_edge_output_path}")
