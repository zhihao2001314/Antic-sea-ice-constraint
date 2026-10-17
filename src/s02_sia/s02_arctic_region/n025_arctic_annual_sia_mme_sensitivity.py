# -*- coding: utf-8 -*-
"""
计算北大西洋年平均订正海冰序列的模式剔除敏感性集合统计。

输入是n023已经保存的逐模式280年年平均序列和pattern端点QC。
本脚本不读取原始NetCDF，不重新执行bias correction，也不使用piControl。
"""


# %% 1. 导入依赖

from pathlib import Path
import json

import pandas as pd


# %% 2. 参数与路径设置

project_root = Path(r"F:\sea\Antic sea ice constraint")
input_directory = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p16_bias_corrected_pattern"
    / "n023_arctic_sia_pattern"
)
output_directory = (
    project_root
    / "output"
    / "s02sia"
    / "s02_arctic_region"
    / "p16_bias_corrected_pattern"
    / "n025_annual_sia_mme_sensitivity"
)
output_directory.mkdir(parents=True, exist_ok=True)

timeseries_file = input_directory / "arctic_bias_corrected_annual_sia_280yr.csv"
consistency_qc_file = (
    input_directory
    / "arctic_bias_corrected_annual_sia_pattern_consistency_qc.csv"
)
sensitivity_timeseries_output = (
    output_directory / "north_atlantic_mme_exclusion_sensitivity_280yr.csv"
)
sensitivity_endpoints_output = (
    output_directory / "north_atlantic_mme_exclusion_sensitivity_endpoints.csv"
)
metadata_output = output_directory / "north_atlantic_mme_exclusion_metadata.json"

region_name = "North Atlantic"
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
exclusion_cases = {
    "without_cesm2": ["CESM2"],
    "without_cnrm_esm2_1": ["CNRM-ESM2-1"],
    "without_canesm5": ["CanESM5"],
    "without_noresm2_lm": ["NorESM2-LM"],
    "without_cesm2_and_canesm5": ["CESM2", "CanESM5"],
    "without_cesm2_and_noresm2_lm": ["CESM2", "NorESM2-LM"],
}


# %% 3. 读取并检查n023结果

for required_file in [timeseries_file, consistency_qc_file]:
    if not required_file.is_file():
        raise FileNotFoundError(f"缺少n023计算结果：{required_file}")

timeseries = pd.read_csv(timeseries_file)
consistency_qc = pd.read_csv(consistency_qc_file)

required_timeseries_columns = {
    "model",
    "region",
    "simulation_year",
    "phase",
    "co2_ppm",
    "bias_corrected_annual_mean_sia_million_km2",
    "bias_corrected_annual_mean_sia_11yr_mean_million_km2",
    "series_type",
}
required_endpoint_columns = {
    "model",
    "region",
    "timeseries_early_sia_million_km2",
    "timeseries_later_sia_million_km2",
    "timeseries_delta_sia_million_km2",
}
if not required_timeseries_columns.issubset(timeseries.columns):
    raise KeyError(
        "n023年平均序列缺少字段："
        f"{sorted(required_timeseries_columns - set(timeseries.columns))}"
    )
if not required_endpoint_columns.issubset(consistency_qc.columns):
    raise KeyError(
        "n023端点QC缺少字段："
        f"{sorted(required_endpoint_columns - set(consistency_qc.columns))}"
    )

individual_timeseries = timeseries.loc[
    (timeseries["region"] == region_name)
    & (timeseries["series_type"] == "individual model")
].copy()
individual_endpoints = consistency_qc.loc[
    consistency_qc["region"] == region_name
].copy()

available_models = set(individual_timeseries["model"].unique())
if available_models != set(model_order):
    raise ValueError(
        f"模式集合不一致：缺少={sorted(set(model_order) - available_models)}，"
        f"额外={sorted(available_models - set(model_order))}"
    )
if len(individual_timeseries) != len(model_order) * 280:
    raise ValueError(
        "北大西洋逐模式序列行数错误："
        f"{len(individual_timeseries)} != {len(model_order) * 280}"
    )
if individual_timeseries.duplicated(["model", "simulation_year"]).any():
    raise ValueError("北大西洋逐模式序列存在重复的模式—年份记录")


# %% 4. 逐方案重新计算集合平均和模式间离散度

sensitivity_timeseries_frames = []
sensitivity_endpoint_frames = []

for case_name, excluded_models in exclusion_cases.items():
    retained_models = [
        model for model in model_order if model not in excluded_models
    ]
    retained_timeseries = individual_timeseries.loc[
        individual_timeseries["model"].isin(retained_models)
    ].copy()
    retained_endpoints = individual_endpoints.loc[
        individual_endpoints["model"].isin(retained_models),
        [
            "model",
            "region",
            "timeseries_early_sia_million_km2",
            "timeseries_later_sia_million_km2",
            "timeseries_delta_sia_million_km2",
        ],
    ].copy()

    sensitivity_statistics = (
        retained_timeseries.groupby(
            ["simulation_year", "phase"],
            as_index=False,
            sort=True,
        )
        .agg(
            co2_ppm=("co2_ppm", "first"),
            ensemble_mean_raw_annual_sia_million_km2=(
                "bias_corrected_annual_mean_sia_million_km2",
                "mean",
            ),
            ensemble_mean_11yr_sia_million_km2=(
                "bias_corrected_annual_mean_sia_11yr_mean_million_km2",
                "mean",
            ),
            intermodel_sd_11yr_sia_million_km2=(
                "bias_corrected_annual_mean_sia_11yr_mean_million_km2",
                "std",
            ),
        )
    )
    sensitivity_statistics.insert(0, "case", case_name)
    sensitivity_statistics.insert(
        1,
        "excluded_models",
        ";".join(excluded_models),
    )
    sensitivity_statistics.insert(
        2,
        "retained_model_count",
        len(retained_models),
    )
    sensitivity_timeseries_frames.append(sensitivity_statistics)

    retained_endpoints.insert(0, "case", case_name)
    retained_endpoints.insert(
        1,
        "excluded_models",
        ";".join(excluded_models),
    )
    retained_endpoints.insert(
        2,
        "retained_model_count",
        len(retained_models),
    )
    sensitivity_endpoint_frames.append(retained_endpoints)

sensitivity_timeseries = pd.concat(
    sensitivity_timeseries_frames,
    ignore_index=True,
)
sensitivity_endpoints = pd.concat(
    sensitivity_endpoint_frames,
    ignore_index=True,
)


# %% 5. 检查并保存结果

expected_timeseries_rows = len(exclusion_cases) * 280
if len(sensitivity_timeseries) != expected_timeseries_rows:
    raise ValueError(
        "敏感性集合序列行数错误："
        f"{len(sensitivity_timeseries)} != {expected_timeseries_rows}"
    )

expected_endpoint_rows = sum(
    len(model_order) - len(excluded_models)
    for excluded_models in exclusion_cases.values()
)
if len(sensitivity_endpoints) != expected_endpoint_rows:
    raise ValueError(
        "敏感性端点行数错误："
        f"{len(sensitivity_endpoints)} != {expected_endpoint_rows}"
    )

sensitivity_timeseries.to_csv(
    sensitivity_timeseries_output,
    index=False,
    encoding="utf-8-sig",
    float_format="%.10g",
)
sensitivity_endpoints.to_csv(
    sensitivity_endpoints_output,
    index=False,
    encoding="utf-8-sig",
    float_format="%.10g",
)

metadata = {
    "analysis": "North Atlantic annual-mean bias-corrected SIA MME sensitivity",
    "source": str(timeseries_file),
    "region": "North Atlantic: 45-70N, 60W-30E",
    "input_series": "annual mean of 12 monthly bias-corrected SIA values",
    "smoothing": "11-year centered running mean inherited from n023",
    "intermodel_sd": "sample standard deviation across retained models",
    "piControl_drift_correction": False,
    "cases": exclusion_cases,
}
metadata_output.write_text(
    json.dumps(metadata, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print(f"[序列] {sensitivity_timeseries_output}")
print(f"[端点] {sensitivity_endpoints_output}")
print(f"[元数据] {metadata_output}")
