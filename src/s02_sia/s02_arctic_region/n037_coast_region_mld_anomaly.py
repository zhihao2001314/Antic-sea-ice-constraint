# %% 1. 导入依赖

from pathlib import Path
import json
import numpy as np
import pandas as pd


# %% 2. 参数与路径设置

root = Path(r"F:\sea\Antic sea ice constraint")
directory = root / "output/s02sia/s02_arctic_region/p14_mld_fixed_regions_raw"
source = directory / "coast_region_20261012_plot_data.csv"
destination = directory / "coast_region_20261012_anomaly_plot_data.csv"
summary_path = directory / "coast_region_20261012_anomaly_endpoints.csv"
metadata_path = directory / "coast_region_20261012_anomaly_metadata.json"
assert not any(path.exists() for path in [destination, summary_path, metadata_path])
column = "march_mld_mean_crossing_only_m"


# %% 3. 读取新区域原值结果

data = pd.read_csv(source)
metadata = json.loads((directory / "coast_region_20261012_plot_metadata.json").read_text(encoding="utf-8"))
rows = []
summaries = []


# %% 4. 分别计算初始基准及异常

for model in metadata["models"]:
    for region in metadata["regions"]:
        sub = data.loc[(data.model == model) & (data.region == region)].sort_values("simulation_year").copy()
        assert sub.simulation_year.tolist() == list(range(1, 281))
        initial = sub.loc[sub.simulation_year.between(1, 10), column]
        assert len(initial) == 10 and initial.notna().all()
        baseline = float(initial.mean())
        sub["initial_mean_m"] = baseline
        sub["march_mld_anomaly_m"] = sub[column] - baseline
        sub["march_mld_anomaly_m_11yr"] = sub.march_mld_anomaly_m.rolling(11, center=True, min_periods=11).mean()
        assert np.isclose(sub.loc[sub.simulation_year.between(1, 10), "march_mld_anomaly_m"].mean(), 0, atol=1e-8)
        assert np.allclose(sub.march_mld_anomaly_m_11yr,
                           sub[column + "_11yr"] - baseline, equal_nan=True)
        late = float(sub.loc[sub.simulation_year.between(271, 280), column].mean())
        summaries.append({"model": model, "region": region, "initial_mean_m": baseline,
                          "late_mean_m": late, "late_minus_initial_m": late - baseline})
        rows.append(sub)
result = pd.concat(rows, ignore_index=True)
assert len(result) == 4480
limits = result.march_mld_anomaly_m_11yr
metadata["plot_y_min_m"] = float(np.floor(limits.min() / 100) * 100)
metadata["plot_y_max_m"] = float(np.ceil(limits.max() / 100) * 100)
metadata["input"] = str(source)
metadata["anomaly_definition"] = "Each model and each fixed region: annual resolved area-weighted March MLD minus its own simulation years 1-10 mean."
metadata["smoothing"] = "11-year centered mean of anomalies, min_periods=11; endpoint statistics use unsmoothed years 271-280."


# %% 5. 保存计算结果和核对表

result.to_csv(destination, index=False, encoding="utf-8-sig")
pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(pd.DataFrame(summaries).round(2).to_string(index=False))
