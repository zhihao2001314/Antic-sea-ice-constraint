# -*- coding: utf-8 -*-
"""
生成未经 piControl 漂移订正的年度 ΔSIA、ΔTo 和 ΔTAS pattern 数据。

三区定义：
    CA：70–90°N，完整纬圈；
    NA：45–70°N，60°W–60°E；
    NP：45–70°N的其余经度。

本脚本只计算并保存 output，不绘图。
"""


# %% 1. 导入依赖

from pathlib import Path
import json

import numpy as np
import pandas as pd


# %% 2. 参数与路径设置

project_root = Path(r"F:\sea\Antic sea ice constraint")

sia_input_path = (
    project_root
    / "data"
    / "s02_sia"
    / "s02_arctic_region"
    / "arctic_sia_pattern_values_1deg.csv"
)
thetao_input_path = (
    project_root
    / "output"
    / "s03_n02"
    / "arctic_thetao_pattern_fields.npz"
)
tas_input_directory = project_root / "output" / "s03tem_n01"

output_directory = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p04_pattern_unadjusted"
    / "n05"
)
output_directory.mkdir(parents=True, exist_ok=True)

cell_output_path = output_directory / "unadjusted_three_pattern_cells.csv"
summary_output_path = output_directory / "unadjusted_three_pattern_summary.csv"
quality_output_path = output_directory / "unadjusted_three_pattern_qc.csv"
metadata_output_path = output_directory / "unadjusted_three_pattern_metadata.json"

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

region_order = ["North Atlantic", "Central Arctic", "North Pacific"]
region_abbreviation = {
    "North Atlantic": "NA",
    "Central Arctic": "CA",
    "North Pacific": "NP",
}

analysis_latitude_minimum = 45.0
central_arctic_latitude_south = 70.0
north_atlantic_longitude_west = -60.0
north_atlantic_longitude_east = 60.0
sea_ice_concentration_threshold = 0.15

target_latitude = np.arange(45.5, 90.0, 1.0)
target_longitude = np.arange(-179.5, 180.0, 1.0)
target_longitude_grid, target_latitude_grid = np.meshgrid(
    target_longitude,
    target_latitude,
)
target_area_weight = np.cos(np.deg2rad(target_latitude_grid))


# %% 3. 读取并补齐未经订正的SIA格点

sia_input_columns = [
    "model",
    "latitude_deg_north",
    "longitude_deg_east",
    "raw_delta_sia_thousand_km2",
    "edge_raw_co2_sic_fraction",
    "edge_raw_cdr_sic_fraction",
]
pattern_dataframe = pd.read_csv(
    sia_input_path,
    usecols=sia_input_columns,
)
pattern_dataframe = pattern_dataframe.loc[
    pattern_dataframe["latitude_deg_north"] >= analysis_latitude_minimum
].copy()
pattern_dataframe["longitude_deg_east"] = (
    (pattern_dataframe["longitude_deg_east"] + 180.0) % 360.0
) - 180.0

full_grid_index = pd.MultiIndex.from_product(
    [target_latitude, target_longitude],
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
            "raw_delta_sia_thousand_km2",
            "edge_raw_co2_sic_fraction",
            "edge_raw_cdr_sic_fraction",
        ]
    ] = model_dataframe[
        [
            "raw_delta_sia_thousand_km2",
            "edge_raw_co2_sic_fraction",
            "edge_raw_cdr_sic_fraction",
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


# %% 4. 建立新的完整三区掩膜

latitude = pattern_dataframe["latitude_deg_north"]
longitude = pattern_dataframe["longitude_deg_east"]

central_arctic_mask = latitude >= central_arctic_latitude_south
north_atlantic_mask = (
    (latitude < central_arctic_latitude_south)
    & (longitude >= north_atlantic_longitude_west)
    & (longitude < north_atlantic_longitude_east)
)
north_pacific_mask = (
    (latitude < central_arctic_latitude_south)
    & ~(
        (longitude >= north_atlantic_longitude_west)
        & (longitude < north_atlantic_longitude_east)
    )
)

pattern_dataframe["partition_region"] = ""
pattern_dataframe.loc[
    central_arctic_mask,
    "partition_region",
] = "Central Arctic"
pattern_dataframe.loc[
    north_atlantic_mask,
    "partition_region",
] = "North Atlantic"
pattern_dataframe.loc[
    north_pacific_mask,
    "partition_region",
] = "North Pacific"
pattern_dataframe["partition_region_abbreviation"] = pattern_dataframe[
    "partition_region"
].map(region_abbreviation)

initial_sea_ice_mask = (
    pattern_dataframe["edge_raw_co2_sic_fraction"]
    >= sea_ice_concentration_threshold
)
final_sea_ice_mask = (
    pattern_dataframe["edge_raw_cdr_sic_fraction"]
    >= sea_ice_concentration_threshold
)
pattern_dataframe["raw_sea_ice_union_mask"] = (
    initial_sea_ice_mask | final_sea_ice_mask
)


# %% 5. 读取未经订正的thetao和tas pattern

thetao_fields = np.load(thetao_input_path)
thetao_latitude = np.asarray(thetao_fields["latitude"], dtype=float)
thetao_longitude = np.asarray(thetao_fields["longitude"], dtype=float)

if not np.array_equal(thetao_latitude, target_latitude_grid):
    raise ValueError("thetao纬度网格与目标1°网格不一致")

if not np.array_equal(thetao_longitude, target_longitude_grid):
    raise ValueError("thetao经度网格与目标1°网格不一致")

pattern_dataframe["raw_delta_thetao_degC"] = np.nan
pattern_dataframe["raw_delta_tas_degC"] = np.nan

for model_name in model_order:
    model_key = model_name.replace("-", "_")
    thetao_raw = np.asarray(
        thetao_fields[f"raw__{model_key}"],
        dtype=float,
    )

    tas_input_path = (
        tas_input_directory
        / f"arctic_tas_{model_name}_pattern_1deg.npz"
    )
    tas_fields = np.load(tas_input_path)
    tas_latitude = np.asarray(tas_fields["lat"], dtype=float)
    tas_longitude = np.asarray(tas_fields["lon"], dtype=float)
    tas_raw = np.asarray(tas_fields["raw_difference"], dtype=float)

    if not np.array_equal(tas_latitude, target_latitude_grid):
        raise ValueError(f"{model_name} tas纬度网格不一致")

    if not np.array_equal(tas_longitude, target_longitude_grid):
        raise ValueError(f"{model_name} tas经度网格不一致")

    model_indices = pattern_dataframe.index[
        pattern_dataframe["model"] == model_name
    ]
    pattern_dataframe.loc[
        model_indices,
        "raw_delta_thetao_degC",
    ] = thetao_raw.ravel()
    pattern_dataframe.loc[
        model_indices,
        "raw_delta_tas_degC",
    ] = tas_raw.ravel()


# %% 6. 计算三区变化量与北极总量

summary_rows = []
quality_rows = []

for model_name in model_order:
    model_dataframe = pattern_dataframe.loc[
        pattern_dataframe["model"] == model_name
    ].sort_values(["latitude_deg_north", "longitude_deg_east"])

    region_membership_count = (
        (model_dataframe["partition_region"] == "North Atlantic").astype(int)
        + (model_dataframe["partition_region"] == "Central Arctic").astype(int)
        + (model_dataframe["partition_region"] == "North Pacific").astype(int)
    )
    regional_sia_sum = 0.0

    for region_name in region_order:
        region_dataframe = model_dataframe.loc[
            model_dataframe["partition_region"] == region_name
        ]
        region_sia = float(
            region_dataframe["raw_delta_sia_thousand_km2"].sum()
            / 1000.0
        )
        regional_sia_sum += region_sia

        region_ice_mask = region_dataframe[
            "raw_sea_ice_union_mask"
        ].to_numpy(dtype=bool)
        region_latitude = region_dataframe[
            "latitude_deg_north"
        ].to_numpy(dtype=float)
        region_weight = np.cos(np.deg2rad(region_latitude))

        region_thetao = region_dataframe[
            "raw_delta_thetao_degC"
        ].to_numpy(dtype=float)
        thetao_valid = region_ice_mask & np.isfinite(region_thetao)
        region_thetao_mean = float(
            np.sum(region_thetao[thetao_valid] * region_weight[thetao_valid])
            / np.sum(region_weight[thetao_valid])
        )

        region_tas = region_dataframe[
            "raw_delta_tas_degC"
        ].to_numpy(dtype=float)
        tas_valid = region_ice_mask & np.isfinite(region_tas)
        region_tas_mean = float(
            np.sum(region_tas[tas_valid] * region_weight[tas_valid])
            / np.sum(region_weight[tas_valid])
        )

        summary_rows.append(
            {
                "model": model_name,
                "region": region_name,
                "region_abbreviation": region_abbreviation[region_name],
                "raw_delta_sia_million_km2": region_sia,
                "raw_delta_thetao_degC": region_thetao_mean,
                "raw_delta_tas_degC": region_tas_mean,
                "raw_sea_ice_union_grid_cell_count": int(
                    region_ice_mask.sum()
                ),
                "thetao_valid_grid_cell_count": int(thetao_valid.sum()),
                "tas_valid_grid_cell_count": int(tas_valid.sum()),
            }
        )

    full_ice_mask = model_dataframe[
        "raw_sea_ice_union_mask"
    ].to_numpy(dtype=bool)
    full_latitude = model_dataframe[
        "latitude_deg_north"
    ].to_numpy(dtype=float)
    full_weight = np.cos(np.deg2rad(full_latitude))
    full_thetao = model_dataframe[
        "raw_delta_thetao_degC"
    ].to_numpy(dtype=float)
    full_tas = model_dataframe[
        "raw_delta_tas_degC"
    ].to_numpy(dtype=float)
    full_thetao_valid = full_ice_mask & np.isfinite(full_thetao)
    full_tas_valid = full_ice_mask & np.isfinite(full_tas)

    summary_rows.append(
        {
            "model": model_name,
            "region": "Arctic total (NA+CA+NP)",
            "region_abbreviation": "ARC",
            "raw_delta_sia_million_km2": regional_sia_sum,
            "raw_delta_thetao_degC": float(
                np.sum(
                    full_thetao[full_thetao_valid]
                    * full_weight[full_thetao_valid]
                )
                / np.sum(full_weight[full_thetao_valid])
            ),
            "raw_delta_tas_degC": float(
                np.sum(
                    full_tas[full_tas_valid]
                    * full_weight[full_tas_valid]
                )
                / np.sum(full_weight[full_tas_valid])
            ),
            "raw_sea_ice_union_grid_cell_count": int(full_ice_mask.sum()),
            "thetao_valid_grid_cell_count": int(full_thetao_valid.sum()),
            "tas_valid_grid_cell_count": int(full_tas_valid.sum()),
        }
    )

    direct_sia_sum = float(
        model_dataframe["raw_delta_sia_thousand_km2"].sum() / 1000.0
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
                (region_membership_count == 0).sum()
            ),
            "three_region_raw_sia_million_km2": regional_sia_sum,
            "direct_raw_sia_million_km2": direct_sia_sum,
            "closure_error_million_km2": regional_sia_sum - direct_sia_sum,
        }
    )


# %% 7. 保存结果和绘图元数据

summary_dataframe = pd.DataFrame(summary_rows)
quality_dataframe = pd.DataFrame(quality_rows)

sia_absolute = np.abs(
    pattern_dataframe["raw_delta_sia_thousand_km2"].to_numpy(dtype=float)
)
thetao_absolute = np.abs(
    pattern_dataframe["raw_delta_thetao_degC"].to_numpy(dtype=float)
)
tas_absolute = np.abs(
    pattern_dataframe["raw_delta_tas_degC"].to_numpy(dtype=float)
)

sia_color_limit = max(
    1.0,
    float(np.ceil(np.nanpercentile(sia_absolute, 99.5) * 2.0) / 2.0),
)
thetao_color_limit = max(
    1.0,
    float(np.ceil(np.nanpercentile(thetao_absolute, 98.0))),
)
tas_color_limit = max(
    1.0,
    float(np.ceil(np.nanpercentile(tas_absolute, 98.0))),
)

cell_output_columns = [
    "model",
    "latitude_deg_north",
    "longitude_deg_east",
    "partition_region",
    "partition_region_abbreviation",
    "raw_delta_sia_thousand_km2",
    "raw_delta_thetao_degC",
    "raw_delta_tas_degC",
    "edge_raw_co2_sic_fraction",
    "edge_raw_cdr_sic_fraction",
    "raw_sea_ice_union_mask",
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

metadata = {
    "adjustment": "none",
    "definition": "CDR final 10 yr minus 1pctCO2 initial 10 yr",
    "central_arctic": "70-90N, all longitudes",
    "north_atlantic": "45-70N, 60W-60E",
    "north_pacific": "45-70N, remaining longitudes",
    "sia_color_limit_thousand_km2": sia_color_limit,
    "thetao_color_limit_degC": thetao_color_limit,
    "tas_color_limit_degC": tas_color_limit,
    "temperature_region_mask": (
        "region AND (raw initial SIC >= 15% OR raw final SIC >= 15%)"
    ),
}
metadata_output_path.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"未订正三变量格点：{cell_output_path}")
print(f"未订正三区汇总：{summary_output_path}")
print(f"质量检查：{quality_output_path}")
print(f"绘图元数据：{metadata_output_path}")
