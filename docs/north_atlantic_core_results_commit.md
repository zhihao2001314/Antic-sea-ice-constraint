# 北大西洋／北极主结果提交清单

此文档记录首次提交快照。海冰选图现已按 PPT 修正，当前以 `docs/sea_ice_ppt_submission.md` 为准；下列海冰历史条目不再代表当前提交范围。其他 AMOC、MLD、N² 条目不变。

本次仅保存AMOC序列、北极海冰空间图与序列、北大西洋MLD空间图与序列、沿岸分区初始N²及IAP观测剖面，以及对应计算／绘图代码。包含四模式稳定阶段MLD／翻转流序列，不包含滞回图。

不提交南极、急流、降水、水汽收支、温度分析、EOF、散点相关及其他试验结果。原始数据和output仍按现有.gitignore规则保留在本机，不强制加入Git；本清单不是包含原始数据的独立可运行数据包。

这是既有结果与源代码的版本快照，不重新计算、不改科研方法、不借提交重构旧脚本。AMOC保留既有异常版本及原始强度版本，二者不能互相替代。北极海冰选择现有非漂移订正主图，不带入bias-corrected敏感性图。MLD采用既有8模式三阶段空间图及沿岸分区序列，N²采用沿岸分区两张clean剖面图。

旧文件说明头中个别历史目录名与实际保存位置不同；以下以当前真实文件路径为准。Python文件仅作语法解析检查，不执行耗时计算，图件不重绘。其他暂存改动不混入本次提交；此操作仅本地commit，不推送远端。

## 文件白名单

- `src/s01_amoc/n02_amoc_nine_model_timeseries.py`
- `src/s01_amoc/n05_amoc4065deta.py`
- `src/s01_amoc/n08_mldraw.py`
- `src/s01_amoc/n14_amoc_initial_reference.py`
- `plot/s01_amoc/n02_amocdeta9model.py`
- `plot/s01_amoc/n05_mldraw.py`
- `plot/s01_amoc/n14_amoc_initial_reference.py`
- `src/s02_sia/s02_arctic_region/n01_arctic_regional_sia_timeseries.py`
- `src/s02_sia/s02_arctic_region/n02_arctic_pattern_full_partition.py`
- `src/s02_sia/s02_arctic_region/n05_unadjusted_sia_thetao_tas_patterns.py`
- `plot/S02_sia/s02_arctic_region/n01_arctic_regional_sia_timeseries.py`
- `plot/S02_sia/s02_arctic_region/n06_sia_only.py`
- `plot/S02_sia/s02_arctic_region/n033_arctic_sia_multimodel_cas_rectangle.py`
- `src/s02_sia/s02_arctic_region/n10_mldcalculate.py`
- `src/s02_sia/s02_arctic_region/n035_coast_regions_mld.py`
- `src/s02_sia/s02_arctic_region/n036_coast_region_plot_data.py`
- `src/s02_sia/s02_arctic_region/n037_coast_region_mld_anomaly.py`
- `plot/S02_sia/s02_arctic_region/n08_85n8modelmld.py`
- `plot/S02_sia/s02_arctic_region/n035_coast_regions_map.py`
- `plot/S02_sia/s02_arctic_region/n036_coast_region_mean.py`
- `plot/S02_sia/s02_arctic_region/n037_coast_region_mld_anomaly.py`
- `src/s02_sia/s02_arctic_region/n039_coast_initial_n2.py`
- `src/s02_sia/s02_arctic_region/n043_coast_iapv4_n2.py`
- `plot/S02_sia/s02_arctic_region/n043_coast_n2_profiles.py`
- `figure/amoc_s01_n01/l_40/AMOC_change_40N_reference_style.png`
- `figure/amoc_s01_n01/l_40/AMOC_change_40N_reference_style.pdf`
- `figure/amoc_s01_n01/l_40/AMOC_change_40N_reference_style.svg`
- `figure/amoc_s01_n01/n12_deepwater_regions_300m/amoc_deepwater_regions_300m_9models_anomaly.png`
- `figure/amoc_s01_n01/n12_deepwater_regions_300m/amoc_deepwater_regions_300m_9models_anomaly.pdf`
- `figure/amoc_s01_n01/n12_deepwater_regions_300m/amoc_deepwater_regions_300m_9models_anomaly.svg`
- `figure/amoc_s01_n01/n13_deepwater_regions_raw_strength/north_atlantic_overturning_raw_strength_9models.png`
- `figure/amoc_s01_n01/n13_deepwater_regions_raw_strength/north_atlantic_overturning_raw_strength_9models.pdf`
- `figure/amoc_s01_n01/n13_deepwater_regions_raw_strength/north_atlantic_overturning_raw_strength_9models.svg`
- `figure/s02_sia/s02_arctic_region/p17_cas_focus/n033_sia_multimodel_cas_rectangle/arctic_sia_partition_pattern_cas_rectangle.png`
- `figure/s02_sia/s02_arctic_region/p17_cas_focus/n033_sia_multimodel_cas_rectangle/arctic_sia_partition_pattern_cas_rectangle.pdf`
- `figure/s02_sia/s02_arctic_region/p17_cas_focus/n033_sia_multimodel_cas_rectangle/arctic_sia_partition_pattern_cas_rectangle.svg`
- `figure/s02_sia/s02_arctic_region/p06_mld/n02_namld_three_phase/na_march_mld_ramp_up_8models.png`
- `figure/s02_sia/s02_arctic_region/p06_mld/n02_namld_three_phase/na_march_mld_ramp_up_8models.pdf`
- `figure/s02_sia/s02_arctic_region/p06_mld/n02_namld_three_phase/na_march_mld_ramp_up_8models.svg`
- `figure/s02_sia/s02_arctic_region/p06_mld/n02_namld_three_phase/na_march_mld_ramp_down_8models.png`
- `figure/s02_sia/s02_arctic_region/p06_mld/n02_namld_three_phase/na_march_mld_ramp_down_8models.pdf`
- `figure/s02_sia/s02_arctic_region/p06_mld/n02_namld_three_phase/na_march_mld_ramp_down_8models.svg`
- `figure/s02_sia/s02_arctic_region/p06_mld/n02_namld_three_phase/na_march_mld_later_early_8models.png`
- `figure/s02_sia/s02_arctic_region/p06_mld/n02_namld_three_phase/na_march_mld_later_early_8models.pdf`
- `figure/s02_sia/s02_arctic_region/p06_mld/n02_namld_three_phase/na_march_mld_later_early_8models.svg`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_mean_timeseries_final.png`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_mean_timeseries_final.pdf`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_mean_timeseries_final.svg`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_anomaly_timeseries.png`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_anomaly_timeseries.pdf`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_anomaly_timeseries.svg`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/MLD_coast_sampling_regions_20261012.png`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/MLD_coast_sampling_regions_20261012.pdf`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/MLD_coast_sampling_regions_20261012.svg`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_li_n2_profiles_3x3_clean.png`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_li_n2_profiles_3x3_clean.pdf`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_li_n2_profiles_3x3_clean.svg`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_nordic_n2_profiles_3x3_clean.png`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_nordic_n2_profiles_3x3_clean.pdf`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/coast_region_20261012_nordic_n2_profiles_3x3_clean.svg`
- `figure/s02_sia/n02_arctic_region/arctic_regional_sia_nine_models.png`
- `figure/s02_sia/n02_arctic_region/arctic_regional_sia_nine_models.pdf`
- `src/s02_sia/s02_arctic_region/n054_cesm2_stabilization.py`
- `plot/S02_sia/s02_arctic_region/n054_cesm2_stabilization.py`
- `src/s02_sia/s02_arctic_region/n055_gfdl_stabilization.py`
- `plot/S02_sia/s02_arctic_region/n055_gfdl_stabilization.py`
- `src/s02_sia/s02_arctic_region/n056_cnrm_stabilization.py`
- `plot/S02_sia/s02_arctic_region/n056_cnrm_stabilization.py`
- `src/s02_sia/s02_arctic_region/n057_noresm_stabilization.py`
- `plot/S02_sia/s02_arctic_region/n057_noresm_stabilization.py`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/cesm2_coast_stabilization_20261013_timeseries.png`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/cesm2_coast_stabilization_20261013_timeseries.pdf`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/cesm2_coast_stabilization_20261013_timeseries.svg`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/gfdl_coast_stabilization_20261013_timeseries.png`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/gfdl_coast_stabilization_20261013_timeseries.pdf`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/gfdl_coast_stabilization_20261013_timeseries.svg`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/cnrm_coast_stabilization_20261013_timeseries.png`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/cnrm_coast_stabilization_20261013_timeseries.pdf`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/cnrm_coast_stabilization_20261013_timeseries.svg`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/noresm_coast_stabilization_20261013_timeseries.png`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/noresm_coast_stabilization_20261013_timeseries.pdf`
- `figure/s02_sia/s02_arctic_region/p14_mld_fixed_regions_raw/noresm_coast_stabilization_20261013_timeseries.svg`
