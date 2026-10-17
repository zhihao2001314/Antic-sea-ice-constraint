# %% 1. 导入依赖

from pathlib import Path
import json
import numpy as np
import pandas as pd


# %% 2. 参数与路径设置

root = Path(r"F:\sea\Antic sea ice constraint")
directory = root / "output/s02sia/s02_arctic_region/p14_mld_fixed_regions_raw"
destination = directory / "coast_region_20261012_plot_data.csv"
summary_path = directory / "coast_region_20261012_endpoints.csv"
metadata_path = directory / "coast_region_20261012_plot_metadata.json"
assert not any(path.exists() for path in [destination,summary_path,metadata_path])


# %% 3. 读取重算后的原生面积统计

annual = pd.read_csv(directory / "na_march_mld_coast_regions_20261012_regions_annual.csv")
metadata = json.loads((directory / "na_march_mld_coast_regions_20261012_regions_metadata.json").read_text(encoding="utf-8"))
metadata["source_attachment"] = "codex-clipboard-31604a86-3ff1-4954-a21f-e19956d581cc.png"
metadata["digitization"] = "inverse LambertConformal; displayed image 1728x1468, map x137..1590 y172..1164; offshore strokes approximate; Natural Earth 110m continental boundaries"
verified_metadata_path = directory / "coast_region_20261012_verified_metadata.json"
assert not verified_metadata_path.exists()
metadata["provenance_note"] = "This verified record supersedes the inherited source_attachment/digitization descriptions in the calculation metadata; numerical results and masks are unchanged."
verified_metadata_path.write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
columns = ["march_mld_mean_crossing_only_m","march_mld_p90_crossing_only_m",
           "resolved_mld_ge_1000m_fraction","no_cross_weight_fraction"]
rows = []
endpoint_rows = []


# %% 4. 保存深度分布和覆盖诊断

for model in metadata["models"]:
    for region in metadata["regions"]:
        sub = annual.loc[(annual.model == model) & (annual.region == region)].sort_values("simulation_year").copy()
        assert sub.simulation_year.tolist() == list(range(1,281))
        for column in columns:
            sub[column + "_11yr"] = sub[column].rolling(11,center=True,min_periods=11).mean()
        rows.append(sub)
        for window,first,last in [("initial",1,10),("late",271,280)]:
            values = sub.loc[sub.simulation_year.between(first,last)]
            row = {"model":model,"region":region,"window":window}
            for column in columns:
                row[column] = float(values[column].mean())
            endpoint_rows.append(row)
plot_data = pd.concat(rows,ignore_index=True)
plot_data.to_csv(destination,index=False,encoding="utf-8-sig")
endpoints = pd.DataFrame(endpoint_rows)
endpoints.to_csv(summary_path,index=False,encoding="utf-8-sig")
record = {"models":metadata["models"],"regions":metadata["regions"],
    "input":str(directory / "na_march_mld_coast_regions_20261012_regions_annual.csv"),
    "selection":"fixed hand-drawn offshore limits clipped to the coastline; no outcome-sign threshold; no moving MLD threshold mask",
    "weighting":"native areacello m2",
    "plot_y_min_m":0,
    "plot_y_max_m":float(np.ceil(plot_data.march_mld_mean_crossing_only_m_11yr.max()*1.1/100)*100),
    "p90_plot_y_max_m":float(np.ceil(plot_data.march_mld_p90_crossing_only_m_11yr.max()*1.1/500)*500),
    "co2_plot_y_min_ppm":250,
    "co2_plot_y_max_ppm":1200,
    "co2_plot_ticks_ppm":[284.7,700,1147],
    "depth":"resolved crossing-only regional mean and spatial P90; no-cross separately quantified",
    "fraction":"resolved MLD>=1000m area / area of all valid MLD cells",
    "limitations":"P90 is a spatial distribution statistic, not a fixed-core mean; no-cross is not evidence of absent convection; offshore boundaries are approximate hand-drawn lines; continental boundaries clipped to Natural Earth 110m land",
    "smoothing":"11-year centered means, min_periods11; endpoints computed before smoothing"}
metadata_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding="utf-8")
print(endpoints.round(2).to_string(index=False))
