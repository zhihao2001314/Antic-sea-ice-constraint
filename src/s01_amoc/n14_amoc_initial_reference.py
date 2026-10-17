# %% 1. 导入依赖

from pathlib import Path
import json
import pandas as pd


# %% 2. 参数与路径设置

root = Path(r"F:\sea\Antic sea ice constraint")
directory = root / "output/amoc_s01_n01/l_40"
initial_years = 10
smoothing_years = 11
models = ["ACCESS-ESM1-5", "CanESM5", "CESM2", "CNRM-ESM2-1", "GFDL-ESM4",
          "MIROC-ES2L", "NorESM2-LM", "UKESM1-0-LL", "CAS-ESM2-0"]
series_path = directory / "amoc_initial_reference_plot_data.csv"
endpoint_path = directory / "amoc_initial_reference_endpoints.csv"
metadata_path = directory / "amoc_initial_reference_metadata.json"
assert not any(path.exists() for path in [series_path, endpoint_path, metadata_path])


# %% 3. 读取既有原值年序列

annual = pd.read_csv(directory / "CMIP6_AMOC_9models_annual_series.csv")
strength = annual.pivot(index="Simulation year", columns="Model", values="Forced AMOC strength (Sv)")
strength = strength.reindex(columns=models).sort_index()
co2 = annual.loc[annual.Model == models[0]].set_index("Simulation year")["Atmospheric CO2 (ppm)"]
assert strength.notna().all().all()


# %% 4. 初始参照与平滑

# 原值减自身初始10年均值，零线表示初始状态；不使用控制趋势。
initial = strength.loc[1:initial_years].mean()
change = strength - initial
smoothed = change.rolling(smoothing_years, center=True, min_periods=1).mean()
result = smoothed.copy()
result.index.name = "simulation_year"
result["co2_ppm"] = co2.reindex(result.index)
result["mmm"] = smoothed.mean(axis=1)
result["std"] = smoothed.std(axis=1, ddof=0)
result["lower"] = result.mmm - result["std"]
result["upper"] = result.mmm + result["std"]
endpoint = pd.DataFrame({"initial_mean_sv": initial,
                         "late_mean_sv": strength.loc[271:280].mean(),
                         "late_minus_initial_sv": change.loc[271:280].mean()})
endpoint.index.name = "model"


# %% 5. 保存结果与口径

result.to_csv(series_path, encoding="utf-8-sig")
endpoint.to_csv(endpoint_path, encoding="utf-8-sig")
metadata = {"source": "existing Forced AMOC strength (Sv) column; no raw NetCDF read",
            "definition": "40N nearest latitude; maximum Atlantic-Arctic overturning below500m",
            "reference": "each model initial years1-10 mean",
            "drift_correction": "none; default unless explicitly requested",
            "dimensions": "CSV pivot: Simulation year, Model; raw field unit Sv",
            "smoothing": "11-year centered, min_periods1; endpoints from unsmoothed values",
            "spread": "population SD across models, not confidence interval"}
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(endpoint.round(3).to_string())
