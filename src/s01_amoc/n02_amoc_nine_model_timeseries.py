# -*- coding: utf-8 -*-
"""
CMIP6 1pctCO2 / 1pctCO2-cdr：AMOC 单变量分析与位置中性主图

用途
----
只处理 AMOC，不依赖 TAS 或海冰数据。程序会自动读取：
    F:\\sea\\cmip6amocad          -> 1pctCO2
    F:\\sea\\cmip6amoc            -> 1pctCO2-cdr
    F:\\sea\\cmip6amoc-picontrol  -> piControl

支持两类输入：
1. 已经预处理好的一维 AMOC 指数；
2. CMIP6 overturning streamfunction（如 msftmz / msftyz）；CNRM式j-mean索引会优先用已有海冰文件中的二维nav_lat/nav_lon映射到地理纬度，不强制要求areacello。

多维流函数的内部计算定义（仅写入方法与处理元数据，不写进主图）：
    大西洋盆地，固定 26.5°N（选取模式网格上最邻近纬度），在 500 m 以下全部有效深度上取最大正向翻转流函数，单位统一为 Sv。

    可选诊断：对 CESM2 额外计算固定 40°N 指标，用于检查纬度敏感性。
    对 CNRM-ESM2-1 自动输出初始异常基线诊断，比较 piControl 均值参考
    与分支线性参考，检查初始正异常是否来自控制态漂移或分支对齐。

输出
----
Figure 1：位置中性的单面板 AMOC 绝对异常时间序列（与参考图统一风格）
可选 Figure 2：AMOC–CO2 滞回环
可选 Figure 3：逐模式最大减弱、滞后、恢复率和末期残余异常
CSV：年度序列、指标、处理元数据、质量控制报告

说明
----
- 数据不全的模式会自动跳过，不要求九个模式全部存在。
- 只有同时具备 1pctCO2、1pctCO2-cdr 和 piControl 的模式才参与分析。
- NorESM2-LM 按 CDR 的 parent_variant_label 严格使用 1pctCO2/p1 → CDR/p1；
  只有 p4 时不再强行拼接为连续轨迹；
  两段均使用其完整数据，不做补齐、外推或按较短阶段误判排除。
- 主图异常采用与分支时刻对齐的 piControl 线性漂移订正；同时输出控制态均值订正作为敏感性诊断。
- 为降低年际噪声，绘图、最低值和滞回指标采用11年中心滑动平均；
  原始年度异常仍完整输出到 annual_series.csv。
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import warnings
from pathlib import Path
from typing import Any


# =============================================================================
# 0. Windows 下自动切换到用户的 viper 环境
# =============================================================================

VIPER_PYTHON = Path(r"D:\Download\Anaconda\envs\viper\python.exe")


def _normal_path(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def ensure_viper() -> None:
    """Windows 下若当前解释器不是 viper，则用 viper 重新运行本脚本。"""
    if os.name != "nt":
        return
    if _normal_path(sys.executable) == _normal_path(VIPER_PYTHON):
        return
    if not VIPER_PYTHON.exists():
        print("[提示] 未找到 viper Python，继续使用当前解释器：")
        print(f"       当前：{sys.executable}")
        print(f"       目标：{VIPER_PYTHON}")
        return

    script = Path(__file__).resolve()
    print("[Python] 正在切换到 viper 环境……")
    result = subprocess.run(
        [str(VIPER_PYTHON), str(script), *sys.argv[1:]],
        cwd=str(script.parent),
        check=False,
    )
    raise SystemExit(result.returncode)


ensure_viper()

import matplotlib

SHOW_FIGURE = False
if not SHOW_FIGURE:
    matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
xr.set_options(keep_attrs=True)


# =============================================================================
# 1. 用户配置
# =============================================================================

BASE_DIR = Path(r"F:\sea")

DIR_1PCTCO2 = BASE_DIR / "cmip6amocad"
DIR_CDR = BASE_DIR / "cmip6amoc"
DIR_PI = BASE_DIR / "cmip6amoc-picontrol"
DIR_AREACELLO = BASE_DIR / "areacello"

# CNRM 的 msftyz 只有 j-mean 网格行，没有地理纬度。程序会依次在这些
# 已有目录中寻找同模式、同海洋网格的二维纬度/经度。areacello不存在也没关系；
# 海冰 siconc/siconca 文件通常带 nav_lat/nav_lon，可直接用于 j 行映射。
CNRM_GRID_SEARCH_DIRS = [
    DIR_AREACELLO,
    BASE_DIR / "cmipseaicead",
    BASE_DIR / "cmipseaice",
    BASE_DIR / "cmipseaice-picontrol",

    # 某些文件会把辅助经纬度保存在AMOC文件本身或同级目录。
    DIR_1PCTCO2,
    DIR_CDR,
    DIR_PI,
]

# CNRM完整j行纬度映射的持久缓存。首次成功后，后续运行不再依赖
# 海冰文件是否移动或文件排序是否发生变化。
CNRM_GRID_CACHE_DIR = (
    BASE_DIR / "publication_figures" / "amoc_only" / "cnrm_grid_cache"
)
CNRM_GRID_CACHE_VERSION = "verified_annualprofile_v2"
ALLOW_CNRM_HISTORICAL_MAPPING_FALLBACK = False

OUTPUT_DIR = BASE_DIR / "publication_figures" / "amoc_only"
OUTPUT_PREFIX = "CMIP6_AMOC_1pctCO2_CDR_linearPi_annualProfile_11yrSmooth_strictLineage"

RAMP_UP_YEARS = 140
PREFERRED_TOTAL_YEARS = 301
MIN_CDR_YEARS = 140       # 至少覆盖 CO2 从峰值降回 PI 的阶段
SMOOTH_WINDOW = 11
FINAL_MEAN_YEARS = 10
# 正文主结果采用分支时刻对齐的piControl线性漂移订正；同时保留mean敏感性结果。
PI_CORRECTION = "linear"

# 主指标统一使用固定纬度上的正向上层翻转最大值。
# 不能再对20–60°N同时取最大值，否则最大位置会随时间移动并低估CESM2等模式的衰弱。
AMOC_EXTREMUM_MODE = "max"
CO2_BASELINE = 284.7
CALENDAR_START_YEAR = 1850
FIGURE1_USE_STD_BAND = True   # True: ±1标准差；False: 17th–83rd分位数

# 默认不排除任何模式；是否纳入由三阶段文件完整性和严格QA共同决定。
EXCLUDED_MODELS: set[str] = set()

# AMOC主指标采用固定26.5°N；固定40°N作为纬度敏感性诊断。
MAIN_AMOC_LATITUDE = 26.5
DIAGNOSTIC_AMOC_LATITUDE = 40.0
AMOC_MIN_DEPTH_M = 500.0
MAX_LATITUDE_OFFSET_DEG = 2.5
ALLOW_PRECOMPUTED_1D_INDEX = False
STRICT_UNIT_VALIDATION = True

# 本次交付只默认生成用户指定的位置中性单面板主图。
# 其他科学诊断仍保留在完整脚本中，需要时可单独开启。
MAKE_HYSTERESIS_FIGURE = False
MAKE_METRICS_FIGURE = False

# 只对CESM2额外计算40°N诊断，避免所有模式重复读取造成过长运行时间。
RUN_CESM2_LATITUDE_DIAGNOSTIC = False

# 自动输出CNRM初始异常诊断（前40年）。
RUN_CNRM_INITIAL_ANOMALY_DIAGNOSTIC = False
INITIAL_DIAGNOSTIC_YEARS = 40

# 若某模式盆地坐标没有标签，可在这里人工指定 Atlantic basin 的
# Python位置索引（从0开始）。CESM2的CMIP6标准化msftmz三盆地依次为：
# 0=Atlantic-Arctic、1=Indian-Pacific、2=Global。原始POP transport_reg
# 会在函数中先按其自己的标准选择index 1，不受这里的CESM2覆盖影响。
BASIN_INDEX_OVERRIDE: dict[str, int] = {
    "CESM2": 0,
}

# 准确性优先：盆地标签缺失时不再按“最强盆地”猜测。
# CESM2原始POP输出的transport_reg=1是有官方定义的Atlantic-Arctic区域，
# 由程序单独识别；其他无标签模式必须提供BASIN_INDEX_OVERRIDE。
ALLOW_AUTOMATIC_BASIN_FALLBACK = False

CO2_AXIS_MIN = 100.0
CO2_AXIS_MAX = 1250.0


# =============================================================================
# 2. 模式名称、别名和颜色
# =============================================================================

MODEL_INFO = {
    "ACCESS-ESM1-5": {
        "aliases": ["ACCESS-ESM1-5", "ACCESSESM5", "ACCESSM5"],
        "short": "ACCESS",
        "color": "#CC79A7",
    },
    "CanESM5": {
        "aliases": ["CanESM5", "CANESM5", "CanESM5-1"],
        "short": "CanESM5",
        "color": "#009E73",
    },
    "CESM2": {
        "aliases": ["CESM2"],
        "short": "CESM2",
        "color": "#E69F00",
    },
    "CNRM-ESM2-1": {
        "aliases": ["CNRM-ESM2-1", "CNRM-ESM2"],
        "short": "CNRM",
        "color": "#0072B2",
    },
    "GFDL-ESM4": {
        "aliases": ["GFDL-ESM4", "GFDL"],
        "short": "GFDL",
        "color": "#8C564B",
    },
    "MIROC-ES2L": {
        "aliases": ["MIROC-ES2L", "MIROC"],
        "short": "MIROC",
        "color": "#56B4E9",
    },
    "NorESM2-LM": {
        "aliases": ["NorESM2-LM", "NORESM2-LM", "NORESM2-LM2"],
        "short": "NorESM",
        "color": "#D55E00",
    },
    "UKESM1-0-LL": {
        "aliases": ["UKESM1-0-LL", "UKESM"],
        "short": "UKESM",
        "color": "#A6761D",
    },
    "CAS-ESM2-0": {
        "aliases": ["CAS-ESM2-0", "CAS-ESM2", "CAS"],
        "short": "CAS",
        "color": "#6A5ACD",
    },
}


# =============================================================================
# 3. 绘图风格
# =============================================================================


def set_publication_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 10.5,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 8.0,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


# =============================================================================
# 4. NetCDF 引擎与文件查找
# =============================================================================


def get_netcdf_engine() -> str:
    engines = xr.backends.list_engines()
    if "netcdf4" in engines:
        return "netcdf4"
    if "h5netcdf" in engines:
        return "h5netcdf"
    raise RuntimeError(
        "\n当前 Python 环境没有 NetCDF4 读取引擎。\n"
        "请安装：netCDF4 h5netcdf h5py cftime dask\n"
        f"当前可用引擎：{list(engines)}"
    )


NETCDF_ENGINE = get_netcdf_engine()

# 缓存由固定网格文件构造的 j-index -> geographic latitude 映射，避免重复读取。
J_INDEX_LATITUDE_CACHE: dict[tuple[str, int], xr.DataArray] = {}


def normalize_name(text: Any) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper())


def _cnrm_aliases(model: str) -> list[str]:
    configuration = MODEL_INFO.get(model, {})
    aliases = configuration.get("aliases", [model])
    return [normalize_name(alias) for alias in aliases]


def _path_matches_model(path: Path, model: str) -> bool:
    normalized_path = normalize_name(str(path))
    return any(alias in normalized_path for alias in _cnrm_aliases(model))


def _dataset_matches_model(dataset: xr.Dataset, model: str) -> bool:
    """用CMIP6全局属性识别模式，避免只依赖文件名或文件夹名。"""
    values: list[str] = []
    for key in (
        "source_id",
        "model_id",
        "source",
        "parent_source_id",
        "title",
        "institution_id",
    ):
        value = dataset.attrs.get(key)
        if value is not None:
            values.append(str(value))

    normalized = normalize_name(" ".join(values))
    if not normalized:
        return False

    return any(alias in normalized for alias in _cnrm_aliases(model))


def _all_netcdf_files(directory: Path) -> list[Path]:
    """递归列出NetCDF，不限于.nc扩展名。"""
    if not directory.exists():
        return []

    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".nc", ".nc4", ".cdf"}
        ],
        key=lambda path: str(path).lower(),
    )


def _discover_cnrm_grid_files(
    model: str,
    directories: list[Path],
) -> tuple[list[Path], list[str]]:
    """发现所有CNRM网格候选，取消旧版“首个目录命中即返回”。

    第一层：路径中包含CNRM别名；
    第二层：打开NetCDF头部，用source_id/model_id识别；
    第三层：文件名像siconc/areacello/nav_lat时也进入候选，但后续仍会
            再次验证模式属性，避免误用其他模式网格。
    """
    searched: list[str] = []
    all_files: list[Path] = []
    seen_dirs: set[str] = set()

    for directory in directories:
        directory = Path(directory)
        key = str(directory).lower()
        if key in seen_dirs:
            continue
        seen_dirs.add(key)
        searched.append(str(directory))
        all_files.extend(_all_netcdf_files(directory))

    unique_files: list[Path] = []
    seen_files: set[str] = set()
    for path in all_files:
        try:
            key = str(path.resolve()).lower()
        except Exception:
            key = str(path).lower()
        if key not in seen_files:
            seen_files.add(key)
            unique_files.append(path)

    path_matches = [
        path for path in unique_files
        if _path_matches_model(path, model)
    ]

    likely_tokens = (
        "SICONC",
        "SICONCA",
        "AREACELLO",
        "NAVLAT",
        "GEOLAT",
        "OCEAN",
        "MSFTYZ",
        "MSFTMZ",
    )
    likely_files = [
        path for path in unique_files
        if any(token in normalize_name(path.name) for token in likely_tokens)
    ]

    candidate_pool: list[Path] = []
    candidate_seen: set[str] = set()

    for path in path_matches + likely_files:
        try:
            key = str(path.resolve()).lower()
        except Exception:
            key = str(path).lower()
        if key not in candidate_seen:
            candidate_seen.add(key)
            candidate_pool.append(path)

    # 文件名完全不含模式名时，用全局属性继续识别。
    attribute_matches: list[Path] = []
    for path in unique_files:
        if path in candidate_pool:
            continue

        dataset: xr.Dataset | None = None
        try:
            dataset = xr.open_dataset(
                path,
                decode_times=False,
                engine=NETCDF_ENGINE,
                cache=False,
            )
            if _dataset_matches_model(dataset, model):
                attribute_matches.append(path)
        except Exception:
            continue
        finally:
            if dataset is not None:
                dataset.close()

    for path in attribute_matches:
        try:
            key = str(path.resolve()).lower()
        except Exception:
            key = str(path).lower()
        if key not in candidate_seen:
            candidate_seen.add(key)
            candidate_pool.append(path)

    return candidate_pool, searched


def _cnrm_cache_path(model: str, j_size: int) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9_-]+", "_", model)
    return CNRM_GRID_CACHE_DIR / (
        f"{safe_model}_j{j_size}_latitude_mapping_"
        f"{CNRM_GRID_CACHE_VERSION}.npz"
    )


def _load_cnrm_mapping_cache(
    model: str,
    data_array: xr.DataArray,
    j_dim: str,
    j_size: int,
) -> xr.DataArray | None:
    """读取CNRM完整j行纬度映射缓存。

    缓存保存的是完整row_latitude数组，与本次目标纬度无关。因此可以同时
    支持26.5°N主分析和40°N敏感性诊断。对CNRM 294行网格额外用已验证的
    40°N位置(j=207, 39.970472°N)检查缓存方向，拒绝早期错误的j=39缓存。
    """
    cache_path = _cnrm_cache_path(model, j_size)
    if not cache_path.exists():
        return None

    try:
        payload = np.load(cache_path, allow_pickle=False)
        row_latitude = np.asarray(
            payload["row_latitude"],
            dtype=float,
        ).reshape(-1)

        if row_latitude.size != j_size:
            return None
        if np.isfinite(row_latitude).sum() < 20:
            return None

        if model == "CNRM-ESM2-1" and j_size == 294:
            nearest_40 = int(
                np.nanargmin(
                    np.abs(row_latitude - 40.0)
                )
            )
            latitude_40 = float(row_latitude[nearest_40])

            if (
                nearest_40 != 207
                or not np.isfinite(latitude_40)
                or abs(latitude_40 - 39.970472) > 0.25
            ):
                print(
                    "      忽略旧的CNRM错误缓存："
                    f"缓存40°N最近行为j={nearest_40}、"
                    f"纬度={latitude_40:.3f}°N；"
                    "已验证正确行为j=207、39.970472°N。"
                )
                return None

        nearest_target = int(
            np.nanargmin(
                np.abs(row_latitude - MAIN_AMOC_LATITUDE)
            )
        )
        actual_target = float(row_latitude[nearest_target])

        if (
            not np.isfinite(actual_target)
            or abs(actual_target - MAIN_AMOC_LATITUDE)
            > MAX_LATITUDE_OFFSET_DEG
        ):
            return None

        coordinate = (
            np.asarray(data_array[j_dim].values)
            if j_dim in data_array.coords
            else np.arange(1, j_size + 1, dtype=float)
        )

        mapped = xr.DataArray(
            row_latitude,
            dims=(j_dim,),
            coords={j_dim: coordinate},
            name=f"{j_dim}_geographic_latitude",
            attrs={
                "standard_name": "latitude",
                "units": "degrees_north",
                "long_name": f"Geographic latitude mapping for {j_dim}",
                "mapping_method": "persistent full-row latitude cache",
                "mapping_source": str(cache_path),
            },
        )

        print(
            f"      {model} 的{j_dim}映射从持久缓存恢复：\n"
            f"      {cache_path}\n"
            f"      {MAIN_AMOC_LATITUDE:.1f}°N最近行："
            f"0-based index={nearest_target}；"
            f"实际纬度={actual_target:.3f}°N"
        )
        return mapped

    except Exception as error:
        print(f"      CNRM持久缓存读取失败，重新搜索网格：{error}")
        return None


def _save_cnrm_mapping_cache(
    model: str,
    row_latitude: np.ndarray,
) -> Path:
    values = np.asarray(row_latitude, dtype=float).reshape(-1)
    CNRM_GRID_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cnrm_cache_path(model, values.size)
    np.savez_compressed(cache_path, row_latitude=values)
    return cache_path


def _normalized_column_map(frame: pd.DataFrame) -> dict[str, str]:
    return {
        normalize_name(column): str(column)
        for column in frame.columns
    }


def matches_alias(text: Any, aliases: list[str]) -> bool:
    normalized = normalize_name(text)
    return any(normalize_name(alias) in normalized for alias in aliases)


def find_model_files(base_dir: Path, aliases: list[str]) -> list[Path]:
    """优先查找模式子目录；若没有，则在文件名中匹配模式别名。"""
    if not base_dir.exists():
        return []

    folders = sorted(path for path in base_dir.iterdir() if path.is_dir())

    for folder in folders:
        if any(
            normalize_name(folder.name) == normalize_name(alias)
            for alias in aliases
        ):
            files = sorted(folder.rglob("*.nc"))
            if files:
                return files

    for folder in folders:
        if matches_alias(folder.name, aliases):
            files = sorted(folder.rglob("*.nc"))
            if files:
                return files

    return [
        path
        for path in sorted(base_dir.rglob("*.nc"))
        if matches_alias(path.name, aliases)
    ]


def parse_cmip6_variant_label(path: Path) -> tuple[int, int, int, int] | None:
    """从文件名或父目录中读取 CMIP6 的 r<i>i<i>p<i>f<i> 变体标签。"""
    match = re.search(
        r"r(?P<r>\d+)i(?P<i>\d+)p(?P<p>\d+)f(?P<f>\d+)",
        path.as_posix(),
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    return tuple(int(match.group(name)) for name in ("r", "i", "p", "f"))


def group_files_by_expected_physics(
    files: list[Path],
    expected_p: int,
    experiment: str,
) -> dict[tuple[int, int, int], list[Path]]:
    """按 (r, i, f) 分组，并严格保留指定 physics index。"""
    parsed = [(path, parse_cmip6_variant_label(path)) for path in files]
    labelled = [(path, label) for path, label in parsed if label is not None]
    if not labelled:
        raise ValueError(
            f"NorESM2-LM 的 {experiment} 文件名/目录中没有标准 "
            "r<i>i<i>p<i>f<i> 标签，无法安全核实阶段物理版本。"
        )

    available_p = sorted({int(label[2]) for _, label in labelled})
    matching = [
        (path, label)
        for path, label in labelled
        if int(label[2]) == int(expected_p)
    ]
    if not matching:
        lineage_note = ""
        if experiment == "1pctCO2" and expected_p == 1:
            lineage_note = (
                " 该CDR成员的parent_variant_label为r1i1p1f1；"
                "本地p4即使数据完整也不是它的父实验，不能用于连续轨迹或MMM。"
            )
        raise ValueError(
            f"NorESM2-LM 的 {experiment} 未找到 p{expected_p}；"
            f"可用 physics index 为 {available_p}。{lineage_note}"
        )

    groups: dict[tuple[int, int, int], list[Path]] = {}
    for path, label in matching:
        r_index, i_index, _, f_index = label
        groups.setdefault((r_index, i_index, f_index), []).append(path)

    return {
        key: sorted(group_files)
        for key, group_files in groups.items()
    }


def select_noresm_stage_files(
    up_files: list[Path],
    cdr_files: list[Path],
    pi_files: list[Path],
) -> tuple[list[Path], list[Path], list[Path], str]:
    """选择 NorESM2-LM 的同源 p1→p1 阶段，并保持完整r/i/f一致。

    NorESM2-LM的1pctCO2-cdr元数据把parent_variant_label声明为
    r1i1p1f1。即便本地1pctCO2/p4数据完整，它也不是该CDR成员的父实验，
    因此不能用来构造连续的CO2上升—移除轨迹。
    """
    up_groups = group_files_by_expected_physics(
        up_files,
        expected_p=1,
        experiment="1pctCO2",
    )
    cdr_groups = group_files_by_expected_physics(
        cdr_files,
        expected_p=1,
        experiment="1pctCO2-cdr",
    )

    common_members = set(up_groups).intersection(cdr_groups)
    if not common_members:
        raise ValueError(
            "NorESM2-LM 虽分别找到 1pctCO2/p1 与 1pctCO2-cdr/p1，"
            "但没有共同的 r/i/f 成员，不能安全拼接。"
        )

    # piControl通常为p1；若带标准标签，则优先要求同一r/i/f。
    labelled_pi = [
        (path, parse_cmip6_variant_label(path))
        for path in pi_files
    ]
    labelled_pi = [(path, label) for path, label in labelled_pi if label is not None]
    pi_groups: dict[tuple[int, int, int], list[Path]] = {}
    for path, label in labelled_pi:
        r_index, i_index, p_index, f_index = label
        if p_index == 1:
            pi_groups.setdefault((r_index, i_index, f_index), []).append(path)

    if labelled_pi and not pi_groups:
        available_pi_p = sorted({int(label[2]) for _, label in labelled_pi})
        raise ValueError(
            "NorESM2-LM 的 piControl 带有标准变体标签，但未找到 p1；"
            f"可用 physics index 为 {available_pi_p}。"
        )

    if pi_groups:
        common_with_pi = common_members.intersection(pi_groups)
        if not common_with_pi:
            raise ValueError(
                "NorESM2-LM 强迫阶段存在共同 r/i/f 成员，但 piControl/p1 "
                "没有同一成员，不能安全建立控制态基准。"
            )
        common_members = common_with_pi

    # 优先CMIP6最常用的r1i1f1；否则稳定选择编号最小的共同成员。
    selected_member = sorted(
        common_members,
        key=lambda item: (item != (1, 1, 1), item),
    )[0]
    selected_pi = (
        sorted(pi_groups[selected_member])
        if pi_groups
        else sorted(pi_files)
    )

    r_index, i_index, f_index = selected_member
    pairing = (
        f"1pctCO2=r{r_index}i{i_index}p1f{f_index}; "
        f"1pctCO2-cdr=r{r_index}i{i_index}p1f{f_index}; "
        + (
            f"piControl=r{r_index}i{i_index}p1f{f_index}"
            if pi_groups
            else "piControl=未标注变体（保留全部文件）"
        )
    )
    return (
        up_groups[selected_member],
        cdr_groups[selected_member],
        selected_pi,
        pairing,
    )


def group_files_by_variant(
    files: list[Path],
) -> dict[tuple[int, int, int, int], list[Path]]:
    groups: dict[tuple[int, int, int, int], list[Path]] = {}
    for path in files:
        label = parse_cmip6_variant_label(path)
        if label is not None:
            groups.setdefault(label, []).append(path)
    return {key: sorted(value) for key, value in groups.items()}


def select_common_variant_files(
    up_files: list[Path],
    cdr_files: list[Path],
    pi_files: list[Path],
    model: str,
) -> tuple[list[Path], list[Path], list[Path], str]:
    """非Nor模式只保留三个实验共同的同一完整r/i/p/f，禁止混合成员。"""
    file_sets = {
        "1pctCO2": up_files,
        "1pctCO2-cdr": cdr_files,
        "piControl": pi_files,
    }
    grouped = {
        experiment: group_files_by_variant(files)
        for experiment, files in file_sets.items()
    }
    labelled_experiments = {
        experiment: groups
        for experiment, groups in grouped.items()
        if groups
    }

    if not labelled_experiments:
        return (
            sorted(up_files),
            sorted(cdr_files),
            sorted(pi_files),
            "三个实验均无标准variant标签；保留各目录全部文件并依靠元数据QA",
        )

    common_variants = set.intersection(
        *(set(groups) for groups in labelled_experiments.values())
    )
    if not common_variants:
        available = {
            experiment: [
                f"r{r}i{i}p{p}f{f}"
                for r, i, p, f in sorted(groups)
            ]
            for experiment, groups in labelled_experiments.items()
        }
        raise ValueError(
            f"{model} 的三个实验没有共同r/i/p/f成员；可用变体={available}。"
        )

    selected = sorted(
        common_variants,
        key=lambda item: (
            item[0] != 1,
            item[1] != 1,
            item[2] != 1,
            item[3] != 1,
            item,
        ),
    )[0]
    r_index, i_index, p_index, f_index = selected

    selected_files = {
        experiment: (
            groups[selected]
            if groups
            else sorted(file_sets[experiment])
        )
        for experiment, groups in grouped.items()
    }
    label = f"r{r_index}i{i_index}p{p_index}f{f_index}"
    return (
        selected_files["1pctCO2"],
        selected_files["1pctCO2-cdr"],
        selected_files["piControl"],
        f"共同variant={label}",
    )


def open_multi_dataset(files: list[Path]) -> xr.Dataset:
    if not files:
        raise FileNotFoundError("未找到 NetCDF 文件。")

    file_names = [str(path) for path in files]
    common_kwargs = {
        "engine": NETCDF_ENGINE,
        "parallel": False,
        "data_vars": "minimal",
        "coords": "minimal",
        "compat": "override",
        "chunks": {"time": 120},
    }

    last_error: Exception | None = None

    # 优先解码时间，以便使用真实calendar逐月天数形成年度平均场；若文件时间
    # 不规范，再退回数值时间，并由annual_mean_dataarray按calendar保守处理。
    for decode_times in (True, False):
        try:
            return xr.open_mfdataset(
                file_names,
                combine="by_coords",
                decode_times=decode_times,
                **common_kwargs,
            )
        except Exception as error:
            last_error = error

    for decode_times in (True, False):
        try:
            return xr.open_mfdataset(
                file_names,
                combine="nested",
                concat_dim="time",
                decode_times=decode_times,
                **common_kwargs,
            )
        except Exception as error:
            last_error = error

    raise RuntimeError(f"无法打开数据集：{last_error}")


# =============================================================================
# 5. 变量、维度和坐标识别
# =============================================================================


def find_time_dimension(data_array: xr.DataArray) -> str:
    for dim in data_array.dims:
        if dim.lower() == "time":
            return dim
    for dim in data_array.dims:
        if dim in data_array.coords:
            if str(data_array[dim].attrs.get("axis", "")).upper() == "T":
                return dim
    raise KeyError(f"无法识别时间维：{data_array.dims}")


def extract_numeric_time_metadata(
    coordinate: xr.DataArray,
) -> tuple[float, float, str, str]:
    """无论时间是否已解码，都恢复branch_time所需的原数值时间和calendar。"""
    units = str(
        coordinate.attrs.get("units", "")
        or coordinate.encoding.get("units", "")
    )
    calendar = str(
        coordinate.attrs.get("calendar", "")
        or coordinate.encoding.get("calendar", "")
        or "standard"
    )
    values = np.asarray(coordinate.values).reshape(-1)
    if values.size == 0:
        return np.nan, np.nan, units, calendar

    if np.issubdtype(values.dtype, np.number):
        numeric = np.asarray(values, dtype=float)
    else:
        numeric = np.asarray([], dtype=float)
        if units:
            try:
                import cftime

                numeric = np.asarray(
                    cftime.date2num(
                        values.tolist(),
                        units=units,
                        calendar=calendar,
                    ),
                    dtype=float,
                )
            except Exception:
                numeric = np.asarray([], dtype=float)

    finite = numeric[np.isfinite(numeric)]
    start = float(finite[0]) if finite.size else np.nan
    end = float(finite[-1]) if finite.size else np.nan
    return start, end, units, calendar


def find_amoc_variable(dataset: xr.Dataset) -> xr.DataArray:
    priority = [
        "msftmz",
        "msftyz",
        "amoc",
        "AMOC",
        "amoc_index",
        "moc",
        "MOC",
        "atlantic_moc",
    ]
    for name in priority:
        if name in dataset.data_vars:
            return dataset[name]

    for name, variable in dataset.data_vars.items():
        standard_name = str(variable.attrs.get("standard_name", "")).lower()
        long_name = str(variable.attrs.get("long_name", "")).lower()
        combined = f"{name} {standard_name} {long_name}".lower()
        if "overturning" in combined or "streamfunction" in combined:
            return variable

    raise KeyError("未找到 AMOC、msftmz 或 msftyz 变量。")



def _geographic_coordinate_score(
    name: str,
    variable: xr.DataArray,
    kind: str,
) -> int:
    """给经纬度候选变量评分；兼容一维、二维及带单例维的坐标。"""
    standard_name = str(variable.attrs.get("standard_name", "")).lower()
    units = str(variable.attrs.get("units", "")).lower()
    long_name = str(variable.attrs.get("long_name", "")).lower()
    name_lower = name.lower()
    score = 0

    if kind == "latitude":
        if standard_name == "latitude":
            score += 20
        if name_lower in {
            "lat", "latitude", "nav_lat", "geolat", "latitude_t",
            "nav_lat_grid_t", "gphit", "gphiv",
        }:
            score += 14
        if "degree_north" in units or "degrees_north" in units:
            score += 12
        if "latitude" in long_name:
            score += 5
    elif kind == "longitude":
        if standard_name == "longitude":
            score += 20
        if name_lower in {
            "lon", "longitude", "nav_lon", "geolon", "longitude_t",
            "nav_lon_grid_t", "glamt", "glamv",
        }:
            score += 14
        if "degree_east" in units or "degrees_east" in units:
            score += 12
        if "longitude" in long_name:
            score += 5
    else:
        raise ValueError(f"未知地理坐标类型：{kind}")

    # 真正的经纬度通常是一维或二维；三维仅允许通过squeeze去掉单例维。
    squeezed = variable.squeeze()
    if squeezed.ndim in (1, 2):
        score += 2
    else:
        score = 0
    return score


def _find_geographic_coordinate_candidates(
    dataset: xr.Dataset,
    kind: str,
) -> list[xr.DataArray]:
    """返回按可信度排序的经纬度候选。

    关键兼容：
    ``dataset.variables`` 中的对象可能是 ``xarray.Variable``；
    这里始终通过 ``dataset[name]`` 取得 ``DataArray``，保留变量名、
    维度和属性，并避免不同xarray版本下 ``Variable.squeeze`` 的差异。
    """
    candidates: list[tuple[int, xr.DataArray]] = []

    for name in dataset.variables:
        variable = dataset[name].squeeze()

        score = _geographic_coordinate_score(
            name,
            variable,
            kind,
        )

        if score > 0 and variable.ndim in (1, 2):
            candidates.append(
                (
                    score,
                    variable,
                )
            )

    candidates.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        variable
        for _, variable in candidates
    ]


def _find_2d_geographic_coordinate(
    dataset: xr.Dataset,
    kind: str,
) -> xr.DataArray | None:
    """保留旧接口：返回评分最高的二维经度或纬度。"""
    for variable in _find_geographic_coordinate_candidates(dataset, kind):
        if variable.ndim == 2:
            return variable
    return None


def _make_sparse_j_latitude_mapping(
    data_array: xr.DataArray,
    j_dim: str,
    j_size: int,
    latitude_index: int,
    actual_latitude: float,
    source: str,
    method: str,
) -> xr.DataArray:
    """构造只在目标j行有有效纬度的映射；足以稳定选择固定40°N。"""
    values = np.full(j_size, np.nan, dtype=float)
    values[int(latitude_index)] = float(actual_latitude)
    if j_dim in data_array.coords:
        coordinate = np.asarray(data_array[j_dim].values)
    else:
        coordinate = np.arange(1, j_size + 1, dtype=float)
    return xr.DataArray(
        values,
        dims=(j_dim,),
        coords={j_dim: coordinate},
        name=f"{j_dim}_geographic_latitude",
        attrs={
            "standard_name": "latitude",
            "units": "degrees_north",
            "long_name": f"Geographic latitude mapping for {j_dim}",
            "mapping_method": method,
            "mapping_source": source,
        },
    )


def _load_previous_cnrm_j_mapping(
    model: str,
    data_array: xr.DataArray,
    j_dim: str,
    j_size: int,
) -> xr.DataArray | None:
    """从历史processing_metadata中复用当前目标纬度的CNRM网格行。

    只有实际纬度接近MAIN_AMOC_LATITUDE，且可用时Requested latitude也与
    当前目标一致，才允许复用。该方法仅作为外部网格搜索失败后的最后兜底。
    """
    roots = [
        OUTPUT_DIR,
        BASE_DIR / "publication_figures",
    ]

    metadata_files: list[Path] = []
    seen: set[str] = set()

    for root in roots:
        if not root.exists():
            continue

        for path in root.rglob("*.csv"):
            if "METADATA" not in normalize_name(path.name):
                continue

            try:
                key = str(path.resolve()).lower()
            except Exception:
                key = str(path).lower()

            if key not in seen:
                seen.add(key)
                metadata_files.append(path)

    metadata_files.sort(
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )

    for metadata_file in metadata_files:
        try:
            frame = pd.read_csv(metadata_file)
        except Exception:
            continue

        columns = _normalized_column_map(frame)

        model_column = columns.get("MODEL")
        latitude_column = (
            columns.get("ACTUALLATITUDEDEGN")
            or columns.get("ACTUALLATITUDE")
        )
        index_column = (
            columns.get("LATITUDEINDEX")
            or columns.get("LATINDEX")
        )
        requested_column = (
            columns.get("REQUESTEDLATITUDEDEGN")
            or columns.get("REQUESTEDLATITUDE")
        )
        definition_column = columns.get("AMOCDEFINITION")

        if (
            model_column is None
            or latitude_column is None
            or index_column is None
        ):
            continue

        rows = frame[
            frame[model_column].astype(str).map(normalize_name)
            == normalize_name(model)
        ]

        for _, row in rows.iterrows():
            try:
                index = int(round(float(row[index_column])))
                actual_latitude = float(row[latitude_column])
            except Exception:
                continue

            if not (0 <= index < j_size):
                continue
            if not np.isfinite(actual_latitude):
                continue
            if (
                abs(actual_latitude - MAIN_AMOC_LATITUDE)
                > MAX_LATITUDE_OFFSET_DEG
            ):
                continue

            if requested_column is not None:
                try:
                    requested_latitude = float(row[requested_column])
                    if (
                        np.isfinite(requested_latitude)
                        and abs(
                            requested_latitude - MAIN_AMOC_LATITUDE
                        ) > 0.25
                    ):
                        continue
                except Exception:
                    pass

            if definition_column is not None:
                definition = normalize_name(row.get(definition_column, ""))
                if definition and "FIXED" not in definition:
                    continue

            mapped = _make_sparse_j_latitude_mapping(
                data_array=data_array,
                j_dim=j_dim,
                j_size=j_size,
                latitude_index=index,
                actual_latitude=actual_latitude,
                source=str(metadata_file),
                method=(
                    "reused from historical metadata validated "
                    f"for fixed {MAIN_AMOC_LATITUDE:.1f} degN"
                ),
            )

            print(
                f"      {model} 的{j_dim}映射从历史metadata复用：\n"
                f"      文件：{metadata_file}\n"
                f"      0-based索引：{index}；"
                f"实际纬度：{actual_latitude:.3f}°N"
            )
            return mapped

    return None


def _resample_row_latitude(
    row_latitude: np.ndarray,
    target_size: int,
) -> np.ndarray:
    """仅用于源j维与目标j维相差很小（通常292/294）的固定网格。"""
    source = np.asarray(row_latitude, dtype=float).reshape(-1)
    if source.size == target_size:
        return source
    finite = np.isfinite(source)
    if finite.sum() < 10:
        return np.full(target_size, np.nan, dtype=float)
    source_x = np.linspace(0.0, 1.0, source.size)
    target_x = np.linspace(0.0, 1.0, target_size)
    return np.interp(target_x, source_x[finite], source[finite], left=np.nan, right=np.nan)


def build_j_index_latitude_mapping(
    model: str,
    data_array: xr.DataArray,
) -> xr.DataArray | None:
    """把CNRM式j-mean映射到地理纬度。

    两层处理：
    1. 从已有海冰、海洋和AMOC文件识别一维或二维经纬度；
    2. 兼容292/294等少量网格行差异；

    准确性模式默认禁止从历史processing_metadata复用单个j索引，避免旧图的
    错误纬度映射继续污染CNRM曲线。缓存文件带算法版本号，旧缓存不会复用。
    """
    j_dim_candidates = [
        dim
        for dim in data_array.dims
        if dim.lower() in {"j-mean", "j_mean", "jmean"}
    ]
    if not j_dim_candidates:
        return None

    j_dim = j_dim_candidates[0]
    j_size = int(data_array.sizes[j_dim])
    cache_key = (model, j_size)

    # 完整行纬度映射同时服务26.5°N主分析和40°N诊断。

    if cache_key in J_INDEX_LATITUDE_CACHE:
        return J_INDEX_LATITUDE_CACHE[cache_key].copy(deep=False)

    # 第0层：读取此前成功保存的完整294行映射。
    persistent = _load_cnrm_mapping_cache(
        model=model,
        data_array=data_array,
        j_dim=j_dim,
        j_size=j_size,
    )
    if persistent is not None:
        J_INDEX_LATITUDE_CACHE[cache_key] = persistent
        return persistent.copy(deep=False)

    # 搜索显式目录，并扩大到E:\\sea下可能包含海洋/海冰网格的目录。
    search_directories: list[Path] = list(CNRM_GRID_SEARCH_DIRS)

    if BASE_DIR.exists():
        for path in BASE_DIR.iterdir():
            if not path.is_dir():
                continue

            normalized = normalize_name(path.name)
            if "PUBLICATIONFIGURES" in normalized:
                continue

            if any(
                token in normalized
                for token in (
                    "SEAICE",
                    "SICONC",
                    "SICONCA",
                    "OCEAN",
                    "THETAO",
                    "TOS",
                    "AREACELLO",
                    "SO",
                    "UO",
                    "VO",
                    "AMOC",
                )
            ):
                search_directories.append(path)

    unique_files, searched_directories = _discover_cnrm_grid_files(
        model=model,
        directories=search_directories,
    )

    def file_priority(path: Path) -> tuple[int, int, int, str]:
        normalized_path = normalize_name(str(path))
        model_rank = 0 if _path_matches_model(path, model) else 1
        areacello_rank = 0 if "AREACELLO" in normalized_path else 1

        try:
            file_size = int(path.stat().st_size)
        except OSError:
            file_size = 10**18

        return (
            model_rank,
            areacello_rank,
            file_size,
            str(path).lower(),
        )

    unique_files.sort(key=file_priority)
    inspection_rows: list[dict[str, str]] = []

    for grid_file in unique_files:
        grid: xr.Dataset | None = None

        try:
            grid = xr.open_dataset(
                grid_file,
                decode_times=False,
                engine=NETCDF_ENGINE,
                cache=False,
            )

            # 候选池中可能包含变量名看起来像网格、但实际属于其他模式的文件。
            # 路径或全局属性至少一个必须匹配CNRM。
            if (
                not _path_matches_model(grid_file, model)
                and not _dataset_matches_model(grid, model)
            ):
                inspection_rows.append(
                    {
                        "file": str(grid_file),
                        "status": "SKIP_OTHER_MODEL",
                        "detail": "路径和source_id/model_id均不匹配CNRM",
                    }
                )
                continue

            latitude_candidates = _find_geographic_coordinate_candidates(
                grid,
                "latitude",
            )
            longitude_candidates = _find_geographic_coordinate_candidates(
                grid,
                "longitude",
            )

            if not latitude_candidates:
                inspection_rows.append(
                    {
                        "file": str(grid_file),
                        "status": "NO_LATITUDE",
                        "detail": "未找到一维或二维地理纬度变量",
                    }
                )
                continue

            for latitude in latitude_candidates:
                latitude = latitude.squeeze()

                # ---------------------------------------------------------
                # A. 一维纬度
                # ---------------------------------------------------------
                if latitude.ndim == 1:
                    source_size = int(latitude.size)
                    if abs(source_size - j_size) > 6:
                        continue

                    row_latitude = np.asarray(
                        latitude.values,
                        dtype=float,
                    ).reshape(-1)

                    if source_size != j_size:
                        row_latitude = _resample_row_latitude(
                            row_latitude,
                            j_size,
                        )
                        method = (
                            f"1D geographic latitude resampled "
                            f"{source_size}->{j_size}"
                        )
                    else:
                        method = "1D geographic latitude"

                    if np.isfinite(row_latitude).sum() < 20:
                        continue

                    nearest = int(
                        np.nanargmin(
                            np.abs(row_latitude - MAIN_AMOC_LATITUDE)
                        )
                    )
                    actual_latitude = float(row_latitude[nearest])

                    if (
                        not np.isfinite(actual_latitude)
                        or abs(actual_latitude - MAIN_AMOC_LATITUDE)
                        > MAX_LATITUDE_OFFSET_DEG
                    ):
                        continue

                    coordinate = (
                        np.asarray(data_array[j_dim].values)
                        if j_dim in data_array.coords
                        else np.arange(1, j_size + 1, dtype=float)
                    )

                    mapped = xr.DataArray(
                        row_latitude,
                        dims=(j_dim,),
                        coords={j_dim: coordinate},
                        name=f"{j_dim}_geographic_latitude",
                        attrs={
                            "standard_name": "latitude",
                            "units": "degrees_north",
                            "mapping_method": method,
                            "mapping_source": str(grid_file),
                            "latitude_variable": str(latitude.name),
                            "longitude_variable": "not required for 1D latitude",
                        },
                    )

                    cache_path = _save_cnrm_mapping_cache(
                        model,
                        row_latitude,
                    )
                    J_INDEX_LATITUDE_CACHE[cache_key] = mapped

                    print(
                        f"      {model} 的{j_dim}地理映射来源：{grid_file}\\n"
                        f"      纬度变量：{latitude.name}{latitude.dims}；"
                        f"方法：{method}\\n"
                        f"      {MAIN_AMOC_LATITUDE:.1f}°N最近行：0-based index={nearest}；"
                        f"实际纬度={actual_latitude:.3f}°N\\n"
                        f"      持久缓存：{cache_path}"
                    )
                    return mapped.copy(deep=False)

                # ---------------------------------------------------------
                # B. 二维纬度
                # ---------------------------------------------------------
                if latitude.ndim != 2:
                    continue

                possible_j_dims = sorted(
                    latitude.dims,
                    key=lambda dim: abs(int(latitude.sizes[dim]) - j_size),
                )
                grid_j_dim = possible_j_dims[0]
                source_j_size = int(latitude.sizes[grid_j_dim])

                # 允许292/294等halo或边界差异。
                if abs(source_j_size - j_size) > 6:
                    continue

                other_dims = [
                    dim
                    for dim in latitude.dims
                    if dim != grid_j_dim
                ]
                if len(other_dims) != 1:
                    continue
                grid_i_dim = other_dims[0]

                lat_values = np.asarray(
                    latitude.transpose(
                        grid_j_dim,
                        grid_i_dim,
                    ).values,
                    dtype=float,
                )

                valid_mask = np.isfinite(lat_values)
                mapping_method = "global-row median latitude"

                longitude = None
                for candidate in longitude_candidates:
                    candidate = candidate.squeeze()
                    if (
                        candidate.ndim == 2
                        and set(candidate.dims) == set(latitude.dims)
                    ):
                        longitude = candidate
                        break

                if longitude is not None:
                    lon_values = np.asarray(
                        longitude.transpose(
                            grid_j_dim,
                            grid_i_dim,
                        ).values,
                        dtype=float,
                    )
                    lon180 = ((lon_values + 180.0) % 360.0) - 180.0

                    atlantic_mask = (
                        np.isfinite(lon180)
                        & (lon180 >= -100.0)
                        & (lon180 <= 20.0)
                        & np.isfinite(lat_values)
                        & (lat_values >= -20.0)
                        & (lat_values <= 80.0)
                    )

                    if (
                        np.count_nonzero(
                            np.sum(atlantic_mask, axis=1) >= 3
                        )
                        >= 20
                    ):
                        valid_mask = atlantic_mask
                        mapping_method = (
                            "Atlantic-sector row median latitude"
                        )

                row_latitude_source = np.full(
                    source_j_size,
                    np.nan,
                    dtype=float,
                )

                for row_index in range(source_j_size):
                    values = lat_values[
                        row_index,
                        valid_mask[row_index],
                    ]
                    values = values[np.isfinite(values)]
                    if values.size:
                        row_latitude_source[row_index] = float(
                            np.nanmedian(values)
                        )

                if np.isfinite(row_latitude_source).sum() < 20:
                    continue

                row_latitude = _resample_row_latitude(
                    row_latitude_source,
                    j_size,
                )

                if source_j_size != j_size:
                    mapping_method += (
                        f"; normalized j resampling "
                        f"{source_j_size}->{j_size}"
                    )

                nearest = int(
                    np.nanargmin(
                        np.abs(row_latitude - MAIN_AMOC_LATITUDE)
                    )
                )
                actual_latitude = float(row_latitude[nearest])

                if (
                    not np.isfinite(actual_latitude)
                    or abs(actual_latitude - MAIN_AMOC_LATITUDE)
                    > MAX_LATITUDE_OFFSET_DEG
                ):
                    continue

                coordinate = (
                    np.asarray(data_array[j_dim].values)
                    if j_dim in data_array.coords
                    else np.arange(1, j_size + 1, dtype=float)
                )

                mapped = xr.DataArray(
                    row_latitude,
                    dims=(j_dim,),
                    coords={j_dim: coordinate},
                    name=f"{j_dim}_geographic_latitude",
                    attrs={
                        "standard_name": "latitude",
                        "units": "degrees_north",
                        "mapping_method": mapping_method,
                        "mapping_source": str(grid_file),
                        "latitude_variable": str(latitude.name),
                        "longitude_variable": (
                            str(longitude.name)
                            if longitude is not None
                            else "not found"
                        ),
                    },
                )

                cache_path = _save_cnrm_mapping_cache(
                    model,
                    row_latitude,
                )
                J_INDEX_LATITUDE_CACHE[cache_key] = mapped

                print(
                    f"      {model} 的{j_dim}地理映射来源：{grid_file}\\n"
                    f"      纬度：{latitude.name}{latitude.shape}；"
                    f"经度："
                    f"{longitude.name if longitude is not None else '未找到'}；"
                    f"方法：{mapping_method}\\n"
                    f"      {MAIN_AMOC_LATITUDE:.1f}°N最近行：0-based index={nearest}；"
                    f"实际纬度={actual_latitude:.3f}°N\\n"
                    f"      持久缓存：{cache_path}"
                )
                return mapped.copy(deep=False)

            inspection_rows.append(
                {
                    "file": str(grid_file),
                    "status": "INCOMPATIBLE_GRID",
                    "detail": (
                        f"有经纬度，但没有与j-size={j_size}"
                        "兼容的一维或二维纬度"
                    ),
                }
            )

        except Exception as error:
            inspection_rows.append(
                {
                    "file": str(grid_file),
                    "status": "OPEN_OR_READ_ERROR",
                    "detail": str(error),
                }
            )
        finally:
            if grid is not None:
                grid.close()

    if ALLOW_CNRM_HISTORICAL_MAPPING_FALLBACK:
        previous = _load_previous_cnrm_j_mapping(
            model,
            data_array,
            j_dim,
            j_size,
        )
        if previous is not None:
            J_INDEX_LATITUDE_CACHE[cache_key] = previous
            return previous.copy(deep=False)

    # 写出详细报告，便于识别究竟是没有候选文件，还是坐标维不兼容。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_{model}_jmean_grid_search_report.csv"
    )

    if inspection_rows:
        report_frame = pd.DataFrame(inspection_rows)
    else:
        report_frame = pd.DataFrame(
            [
                {
                    "file": "",
                    "status": "NO_CANDIDATE_FILES",
                    "detail": "没有发现可验证的CNRM NetCDF候选文件",
                }
            ]
        )

    report_frame.to_csv(
        report_path,
        index=False,
        encoding="utf-8-sig",
    )

    preview = "\\n".join(
        (
            f"  - {row['status']}: "
            f"{Path(row['file']).name} | "
            f"{row['detail']}"
        )
        for row in inspection_rows[:16]
    )

    raise ValueError(
        f"{model} 的{j_dim}仍无法建立地理纬度映射。\\n"
        f"已搜索目录：\\n  - "
        + "\\n  - ".join(searched_directories)
        + f"\\n候选NetCDF数量：{len(unique_files)}。\\n"
        f"详细报告：{report_path}\\n"
        f"部分检查结果：\\n{preview}\\n"
        "本版已支持一维纬度、二维nav_lat/nav_lon、292/294小差异、"
        "带版本的持久缓存；准确性模式未启用历史metadata索引复用。"
    )


def find_latitude(
    dataset: xr.Dataset,
    data_array: xr.DataArray,
) -> xr.DataArray | None:
    candidates: list[tuple[int, xr.DataArray]] = []
    variables = dict(dataset.variables)
    variables.update({name: data_array.coords[name] for name in data_array.coords})

    for name, variable in variables.items():
        standard_name = str(variable.attrs.get("standard_name", "")).lower()
        units = str(variable.attrs.get("units", "")).lower()
        axis = str(variable.attrs.get("axis", "")).upper()
        name_lower = name.lower()

        score = 0
        if standard_name == "latitude":
            score += 10
        if axis == "Y":
            score += 6
        if name_lower in {
            "lat",
            "latitude",
            "nav_lat",
            "geolat",
            "rlat",
            "lat_aux_grid",
            "y",
        }:
            score += 5
        if "degree_north" in units or "degrees_north" in units:
            score += 4
        if score > 0 and variable.ndim in (1, 2):
            candidates.append((score, variable))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def find_vertical_coordinate(
    dataset: xr.Dataset,
    data_array: xr.DataArray,
) -> xr.DataArray | None:
    candidates: list[tuple[int, xr.DataArray]] = []
    variables = dict(dataset.variables)
    variables.update({name: data_array.coords[name] for name in data_array.coords})

    for name, variable in variables.items():
        if variable.ndim != 1:
            continue

        standard_name = str(variable.attrs.get("standard_name", "")).lower()
        axis = str(variable.attrs.get("axis", "")).upper()
        positive = str(variable.attrs.get("positive", "")).lower()
        name_lower = name.lower()

        score = 0
        if standard_name in {"depth", "ocean_depth", "depth_below_geoid"}:
            score += 10
        if axis == "Z":
            score += 6
        if positive in {"up", "down"}:
            score += 2
        if name_lower in {
            "lev",
            "olevel",
            "depth",
            "depth_coord",
            "moc_z",
            "z_t",
            "z",
        }:
            score += 5
        if score > 0 and any(dim in data_array.dims for dim in variable.dims):
            candidates.append((score, variable))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


# =============================================================================
# 6. 时间平均、单位和空间选择
# =============================================================================


def infer_time_sampling(series: xr.DataArray, time_dim: str) -> str:
    """保守识别 monthly/annual；无法可靠判断时返回 unknown。"""
    coordinate = series[time_dim]

    try:
        months = np.asarray(coordinate.dt.month.values, dtype=int)
        years = np.asarray(coordinate.dt.year.values, dtype=int)
        if months.size >= 2 and years.size == months.size:
            if np.unique(years).size < years.size:
                return "monthly"
            return "annual"
    except Exception:
        pass

    values = np.asarray(coordinate.values)
    units = str(coordinate.attrs.get("units", "")).lower()
    if values.size < 2 or not np.issubdtype(values.dtype, np.number):
        return "unknown"

    numeric = np.asarray(values, dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size < 2:
        return "unknown"
    median_step = float(np.nanmedian(np.diff(numeric)))

    if "day" in units:
        if 20.0 <= median_step <= 40.0:
            return "monthly"
        if 300.0 <= median_step <= 370.0:
            return "annual"
    elif "month" in units:
        if 0.5 <= median_step <= 1.5:
            return "monthly"
        if 10.0 <= median_step <= 13.0:
            return "annual"
    elif "year" in units:
        if 0.8 <= median_step <= 1.2:
            return "annual"

    # CMIP数值时间常为days since；即使units属性丢失，步长仍可保守判断。
    if 20.0 <= median_step <= 40.0:
        return "monthly"
    if 300.0 <= median_step <= 370.0:
        return "annual"
    return "unknown"


def annual_mean_dataarray(
    series: xr.DataArray,
    time_dim: str,
) -> tuple[xr.DataArray, str, str]:
    """先形成年度平均场，再做AMOC空间极值，符合常用CMIP诊断顺序。"""
    try:
        days = series[time_dim].dt.days_in_month
        numerator = (series * days).groupby(f"{time_dim}.year").sum(
            dim=time_dim,
            skipna=True,
        )
        denominator = days.where(series.notnull()).groupby(
            f"{time_dim}.year"
        ).sum(
            dim=time_dim,
            skipna=True,
        )
        annual = numerator / denominator
        return annual, "year", "calendar-day-weighted annual mean field"
    except Exception:
        pass

    sampling = infer_time_sampling(series, time_dim)
    if sampling == "annual":
        return series, time_dim, "input already annual"
    if sampling != "monthly":
        raise ValueError(
            f"无法可靠判断{time_dim}是月数据还是年数据；"
            f"units={series[time_dim].attrs.get('units', '')!r}，"
            f"长度={series.sizes[time_dim]}。为避免错误年平均，已停止。"
        )

    number_of_months = int(series.sizes[time_dim])
    if number_of_months < 12:
        raise ValueError("月序列不足一年。")
    if number_of_months % 12 != 0:
        raise ValueError(
            f"月序列长度{number_of_months}不能被12整除，"
            "存在不完整年份，不能静默裁剪。"
        )

    number_of_years = number_of_months // 12
    calendar = str(series[time_dim].attrs.get("calendar", "")).lower()
    if "360" in calendar:
        month_days = np.full(12, 30.0, dtype=float)
    else:
        # CMIP的noleap/365_day最常见；数值时间无法可靠恢复闰年时采用noleap。
        month_days = np.asarray(
            [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
            dtype=float,
        )

    year_index = xr.DataArray(
        np.repeat(np.arange(number_of_years, dtype=int), 12),
        dims=(time_dim,),
        coords={time_dim: series[time_dim]},
        name="analysis_year",
    )
    weights = xr.DataArray(
        np.tile(month_days, number_of_years),
        dims=(time_dim,),
        coords={time_dim: series[time_dim]},
    )
    numerator = (series * weights).groupby(year_index).sum(
        dim=time_dim,
        skipna=True,
    )
    denominator = weights.where(series.notnull()).groupby(year_index).sum(
        dim=time_dim,
        skipna=True,
    )
    annual = numerator / denominator
    return annual, "analysis_year", "numeric-time day-weighted annual mean field"


def annual_weighted_mean(series: xr.DataArray, time_dim: str) -> np.ndarray:
    """一维兼容包装；主AMOC路径使用annual_mean_dataarray先做年度平均场。"""
    annual, _, _ = annual_mean_dataarray(series, time_dim)
    values = np.asarray(annual.compute().values, dtype=float).squeeze()
    if values.ndim != 1:
        raise ValueError(f"年平均后序列不是一维：shape={values.shape}")
    return values


def depth_values_in_meters(depth: xr.DataArray) -> np.ndarray:
    values = np.asarray(depth.values, dtype=float)
    units = str(depth.attrs.get("units", "")).lower().replace(" ", "")

    if "cm" in units and "m" not in units.replace("cm", ""):
        values = values / 100.0
    elif "km" in units:
        values = values * 1000.0

    return np.abs(values)


def convert_amoc_to_sv(
    values: np.ndarray,
    units: str,
) -> tuple[np.ndarray, str]:
    values = np.asarray(values, dtype=float)
    units_lower = units.lower().replace(" ", "")

    if "sv" in units_lower or "sverdrup" in units_lower:
        return values, "already Sv"

    if "kg" in units_lower and ("s-1" in units_lower or "/s" in units_lower):
        return values / 1.0e9, "kg s-1 divided by 1e9"

    if ("m3" in units_lower or "m^3" in units_lower) and (
        "s-1" in units_lower or "/s" in units_lower
    ):
        return values / 1.0e6, "m3 s-1 divided by 1e6"

    if STRICT_UNIT_VALIDATION:
        raise ValueError(
            f"AMOC单位{units!r}不能可靠转换为Sv。"
            "准确性模式禁止仅凭数值量级猜测单位。"
        )

    magnitude = float(np.nanmedian(np.abs(values)))
    if magnitude > 1.0e7:
        return values / 1.0e9, "magnitude-based divided by 1e9"
    if magnitude > 1.0e4:
        return values / 1.0e6, "magnitude-based divided by 1e6"

    return values, "unknown units; assumed Sv"


def subset_1d_coordinate(
    data_array: xr.DataArray,
    coordinate: xr.DataArray,
    lower: float,
    upper: float,
) -> tuple[xr.DataArray, bool]:
    coordinate = coordinate.squeeze()
    if coordinate.ndim != 1:
        return data_array, False

    dim = coordinate.dims[0]
    if dim not in data_array.dims:
        return data_array, False

    values = np.asarray(coordinate.values, dtype=float)
    indices = np.where(
        np.isfinite(values) & (values >= lower) & (values <= upper)
    )[0]
    if indices.size == 0:
        return data_array, False

    return data_array.isel({dim: indices}), True



def select_nearest_latitude(
    data_array: xr.DataArray,
    latitude: xr.DataArray,
    target_latitude: float,
    max_offset: float = MAX_LATITUDE_OFFSET_DEG,
) -> tuple[xr.DataArray, float, int, str]:
    """选择最接近目标纬度的单个网格点。

    返回：选取后的DataArray、实际纬度、索引、纬度维名称。
    固定纬度是本版最重要的定义约束；不允许在20–60°N之间逐年追踪移动最大值。
    """
    latitude = latitude.squeeze()
    if latitude.ndim != 1:
        raise ValueError(
            f"固定纬度AMOC要求一维纬度坐标，但得到shape={latitude.shape}。"
        )

    latitude_dim = latitude.dims[0]
    if latitude_dim not in data_array.dims:
        raise ValueError(
            f"纬度维{latitude_dim}不在AMOC变量维度{data_array.dims}中。"
        )

    values = np.asarray(latitude.values, dtype=float).reshape(-1)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        raise ValueError("纬度坐标全部无效。")

    local = int(np.argmin(np.abs(values[finite_indices] - target_latitude)))
    index = int(finite_indices[local])
    actual_latitude = float(values[index])
    offset = abs(actual_latitude - target_latitude)

    if offset > max_offset:
        raise ValueError(
            f"目标纬度{target_latitude:.1f}°N附近没有足够接近的网格点；"
            f"最近为{actual_latitude:.2f}°，偏差{offset:.2f}°。"
        )

    return (
        data_array.isel({latitude_dim: index}),
        actual_latitude,
        index,
        latitude_dim,
    )


def find_basin_dimension(data_array: xr.DataArray) -> str | None:
    for dim in data_array.dims:
        name = dim.lower()
        if (
            "basin" in name
            or "region" in name
            or "sector" in name
            or name in {"transport_reg", "transport_region"}
        ):
            return dim
    return None


def _decode_basin_label(value: Any) -> str:
    """将 bytes / numpy.bytes_ / 普通字符串统一解码。"""
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8", errors="ignore").strip().lower()
    return str(value).strip().lower()


def _candidate_basin_coordinates(
    dataset: xr.Dataset,
    data_array: xr.DataArray,
    basin_dim: str,
) -> list[tuple[str, xr.DataArray]]:
    """寻找所有沿 basin_dim 定义的一维盆地标签变量。

    关键兼容：CAS 原始文件的维度叫 ``basin``，但标签坐标叫
    ``sector(basin)``，不能只查找名为 basin 的坐标。
    """
    candidates: list[tuple[str, xr.DataArray]] = []
    seen: set[str] = set()

    preferred_names = [
        "transport_regions",
        "transport_region",
        "sector",
        "basin",
        "region",
        basin_dim,
    ]
    for name in preferred_names:
        if name in dataset.variables:
            variable = dataset[name]
            if (
                variable.ndim in (1, 2)
                and variable.dims[0] == basin_dim
            ):
                candidates.append((name, variable))
                seen.add(name)

    for name, variable in dataset.variables.items():
        if name in seen:
            continue
        if variable.ndim in (1, 2) and variable.dims[0] == basin_dim:
            candidates.append((name, variable))
            seen.add(name)

    for name in data_array.coords:
        if name in seen:
            continue
        variable = data_array.coords[name]
        if variable.ndim in (1, 2) and variable.dims[0] == basin_dim:
            candidates.append((name, variable))
            seen.add(name)

    return candidates


def _atlantic_position_from_requested(
    coordinate: xr.DataArray,
    size: int,
) -> int | None:
    """从CMOR ``requested`` 属性恢复Atlantic在basin维上的位置。

    CESM2的部分CMIP6文件把basin保存为整数坐标，同时仅在属性中写出
    ``atlantic_arctic_ocean=0, indian_pacific_ocean=1, global_ocean=2``。
    这仍是可验证的盆地定义，不应被当作“无标签”数据。
    """
    requested = str(coordinate.attrs.get("requested", "")).strip().lower()
    if not requested:
        return None

    match = re.search(
        r"(?:atlantic_arctic_ocean|atlantic_ocean)\s*=\s*(-?\d+)",
        requested,
    )
    if match:
        requested_value = int(match.group(1))
        values = np.asarray(coordinate.values).reshape(-1)
        if values.size == size and np.issubdtype(values.dtype, np.number):
            positions = np.where(
                np.isclose(values.astype(float), float(requested_value))
            )[0]
            if positions.size:
                return int(positions[0])
        if 0 <= requested_value < size:
            return requested_value

    # 兼容requested被保存为字符列表而不是“名称=值”的情况。
    region_names = re.findall(
        r"(?:atlantic_arctic_ocean|atlantic_ocean|"
        r"indian_pacific_ocean|global_ocean)",
        requested,
    )
    for index, name in enumerate(region_names[:size]):
        if name in {"atlantic_arctic_ocean", "atlantic_ocean"}:
            return index
    return None


def select_atlantic_basin(
    data_array: xr.DataArray,
    dataset: xr.Dataset,
    time_dim: str,
    model: str,
    forced_index: int | None = None,
) -> tuple[xr.DataArray, str, int | None]:
    basin_dim = find_basin_dimension(data_array)
    if basin_dim is None:
        # CAS 老师处理后的 CDR 文件是 amoc(time, lev, lat)，已经去掉盆地维。
        return data_array, "no basin dimension; already basin-selected", None

    size = data_array.sizes[basin_dim]

    if forced_index is not None:
        if forced_index >= size:
            raise IndexError(
                f"指定盆地索引 {forced_index} 超出 {basin_dim} 长度 {size}。"
            )
        return (
            data_array.isel({basin_dim: forced_index}),
            f"forced basin index {forced_index}",
            forced_index,
        )

    # CESM/POP的标准MOC结构：transport_reg=0是全球，1是Atlantic-Arctic。
    # 必须先于CESM2的CMIP6 basin覆盖执行，因为两个坐标体系的索引不同。
    if (
        model == "CESM2"
        and basin_dim.lower() in {"transport_reg", "transport_region"}
        and size == 2
    ):
        return (
            data_array.isel({basin_dim: 1}),
            "CESM/POP transport_reg index 1: Atlantic-Arctic",
            1,
        )

    # 先从 sector/basin/region 等沿 basin 维定义的标签中识别 Atlantic。
    for coordinate_name, coordinate in _candidate_basin_coordinates(
        dataset,
        data_array,
        basin_dim,
    ):
        values = np.asarray(coordinate.values)

        requested_position = _atlantic_position_from_requested(
            coordinate,
            size,
        )
        if requested_position is not None:
            return (
                data_array.isel({basin_dim: requested_position}),
                f"{coordinate_name}:requested Atlantic-Arctic",
                requested_position,
            )

        labels: list[str] = []
        if values.dtype.kind in {"U", "S", "O"}:
            if values.ndim == 1 and values.size == size:
                labels = [_decode_basin_label(value) for value in values]
            elif values.ndim == 2 and values.shape[0] == size:
                labels = [
                    "".join(_decode_basin_label(value) for value in row).strip()
                    for row in values
                ]

        if labels:
            print(f"      {model} 盆地标签 {coordinate_name}: {labels}")
            for index, label in enumerate(labels):
                if "atlantic" in label or label.startswith("atl") or "atl_" in label:
                    return (
                        data_array.isel({basin_dim: index}),
                        f"{coordinate_name}:{label}",
                        index,
                    )

        flag_meanings = str(coordinate.attrs.get("flag_meanings", ""))
        flag_values = coordinate.attrs.get("flag_values", None)
        if flag_meanings and flag_values is not None and values.ndim == 1:
            meanings = flag_meanings.split()
            flags = np.atleast_1d(flag_values)
            for meaning, flag in zip(meanings, flags):
                if "atl" in meaning.lower():
                    matches = np.where(values == flag)[0]
                    if matches.size:
                        index = int(matches[0])
                        return (
                            data_array.isel({basin_dim: index}),
                            f"{coordinate_name}-flag:{meaning}",
                            index,
                        )

    # 最后才使用显式覆盖。这样CESM2原始POP仍选择transport_reg index 1；
    # 仅当CMIP6标准化文件的字符标签和requested属性都被预处理剥离时，
    # 才对basin=3使用已核实的Atlantic-Arctic位置0。
    if model in BASIN_INDEX_OVERRIDE:
        index = int(BASIN_INDEX_OVERRIDE[model])
        if index < 0 or index >= size:
            raise IndexError(
                f"{model} 的 BASIN_INDEX_OVERRIDE={index} 超出范围。"
            )
        return (
            data_array.isel({basin_dim: index}),
            f"verified manual override index {index}",
            index,
        )

    if size == 1:
        return data_array.isel({basin_dim: 0}), "single basin", 0

    if not ALLOW_AUTOMATIC_BASIN_FALLBACK:
        raise ValueError(
            f"{model} 的盆地维{basin_dim}有{size}项，但没有可验证的Atlantic标签。"
            "准确性模式禁止按幅度猜测盆地；请检查transport_regions/sector标签，"
            "或在BASIN_INDEX_OVERRIDE中明确给出Atlantic索引。"
        )

    # 标签缺失时，不再只看最前24个月。某些CAS文件开头存在连续缺测，
    # 因此在完整记录中均匀抽取最多24个时刻评估各盆地的有效翻转幅度。
    scores: list[float] = []
    total_time = int(data_array.sizes[time_dim])
    sample_count = min(24, total_time)
    sample_indices = np.unique(
        np.linspace(0, total_time - 1, sample_count, dtype=int)
    )

    for index in range(size):
        candidate = data_array.isel({basin_dim: index})
        sample = candidate.isel({time_dim: sample_indices})
        reduce_dims = [dim for dim in sample.dims if dim != time_dim]

        try:
            maximum = sample.max(dim=reduce_dims, skipna=True).compute().values
            minimum = sample.min(dim=reduce_dims, skipna=True).compute().values
            finite = np.concatenate(
                [
                    np.asarray(maximum, dtype=float).ravel(),
                    np.asarray(minimum, dtype=float).ravel(),
                ]
            )
            finite = finite[np.isfinite(finite)]
            score = float(np.nanmedian(np.abs(finite))) if finite.size else -np.inf
        except Exception:
            score = -np.inf

        if not np.isfinite(score):
            score = -np.inf
        scores.append(score)

    finite_indices = [i for i, score in enumerate(scores) if np.isfinite(score)]
    if not finite_indices:
        raise ValueError(
            f"{model} 无法识别 Atlantic basin：所有 {basin_dim} 候选均无有效值。"
        )

    index = max(finite_indices, key=lambda i: scores[i])
    return (
        data_array.isel({basin_dim: index}),
        f"auto strongest basin index {index}",
        index,
    )


def find_moc_component_dimension(data_array: xr.DataArray) -> str | None:
    for dim in data_array.dims:
        normalized = normalize_name(dim)
        if normalized in {
            "MOCCOMP",
            "MOCCOMPONENT",
            "MOCCOMPONENTS",
        }:
            return dim
    return None


def combine_moc_components(
    data_array: xr.DataArray,
    dataset: xr.Dataset,
    model: str,
) -> tuple[xr.DataArray, str]:
    """CESM/POP中总MOC为Eulerian、bolus和submeso分量之和。"""
    component_dim = find_moc_component_dimension(data_array)
    if component_dim is None:
        return data_array, "no MOC component dimension"

    size = int(data_array.sizes[component_dim])
    if size == 1:
        return (
            data_array.isel({component_dim: 0}),
            f"single {component_dim} component",
        )

    if model != "CESM2":
        raise ValueError(
            f"{model} 含有{component_dim}={size}个MOC分量，"
            "但没有经过验证的合并规则，已停止而不是取错分量。"
        )

    label_text = ""
    for name in ("moc_components", component_dim):
        if name in dataset.variables:
            values = np.asarray(dataset[name].values).reshape(-1)
            label_text = " | ".join(_decode_basin_label(value) for value in values)
            break

    return (
        data_array.sum(dim=component_dim, skipna=True),
        f"sum over {component_dim} ({label_text or size}) for total residual MOC",
    )


# =============================================================================
# 7. AMOC 年序列提取
# =============================================================================


def extract_annual_amoc(
    files: list[Path],
    model: str,
    basin_index: int | None = None,
    extremum_mode: str = "max",
    target_latitude: float = MAIN_AMOC_LATITUDE,
    expected_experiment: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """提取固定纬度AMOC年序列。

    多维流函数定义：
        Atlantic basin -> 最近目标纬度的单一纬度格点
        -> 500 m以下全部有效深度 -> 先形成年度平均流函数剖面
        -> 对每个年度剖面取上层翻转单元强度。

    ``extremum_mode`` 支持：
    - ``"max"``：沿深度取正向最大值；
    - ``"negative_min"``：沿深度取最小值后乘以-1，用于符号约定相反的模式；
    - ``"auto"``：优先采用正向maximum；仅当其控制态接近0，而 ``-minimum``
      给出合理正值时，自动切换为 ``negative_min``。

    这一自动切换主要用于CNRM。它不是按绝对值选择任意深层单元，而是只在
    正向maximum明显失效（接近0）时纠正流函数符号约定。
    """
    dataset = open_multi_dataset(files)

    try:
        source_id = str(dataset.attrs.get("source_id", "")).strip()
        experiment_id = str(dataset.attrs.get("experiment_id", "")).strip()

        if source_id:
            valid_sources = {
                normalize_name(model),
                *{
                    normalize_name(alias)
                    for alias in MODEL_INFO.get(model, {}).get("aliases", [])
                },
            }
            if normalize_name(source_id) not in valid_sources:
                raise ValueError(
                    f"文件source_id={source_id!r}与目标模式{model!r}不一致。"
                    "已停止，防止目录/别名误匹配造成模式曲线串线。"
                )

        if (
            expected_experiment
            and experiment_id
            and normalize_name(experiment_id)
            != normalize_name(expected_experiment)
        ):
            raise ValueError(
                f"文件experiment_id={experiment_id!r}，"
                f"但当前应为{expected_experiment!r}。"
            )

        variable = find_amoc_variable(dataset).squeeze()
        time_dim = find_time_dimension(variable)
        units = str(variable.attrs.get("units", ""))

        variable = variable.astype("float64")
        variable = variable.where(
            np.isfinite(variable) & (np.abs(variable) < 1.0e20)
        )

        valid_count = int(variable.count().compute().item())
        if valid_count == 0:
            raise ValueError(
                f"{model} 的 {variable.name} 清理缺测值后全部为NaN。"
            )

        time_coordinate = dataset[time_dim]
        time_start, time_end, time_units, time_calendar = (
            extract_numeric_time_metadata(time_coordinate)
        )

        metadata: dict[str, Any] = {
            "variable": variable.name or "unknown",
            "original_units": units,
            "original_dims": str(variable.dims),
            "original_shape": str(tuple(variable.shape)),
            "source_id": source_id,
            "experiment_id": experiment_id,
            "variant_label": str(dataset.attrs.get("variant_label", "")).strip(),
            "variant_info": str(dataset.attrs.get("variant_info", "")).strip(),
            "physics_index": dataset.attrs.get("physics_index", np.nan),
            "parent_experiment_id": str(
                dataset.attrs.get("parent_experiment_id", "")
            ).strip(),
            "parent_variant_label": str(
                dataset.attrs.get("parent_variant_label", "")
            ).strip(),
            "branch_method": str(dataset.attrs.get("branch_method", "")).strip(),
            "valid_value_count_after_cleaning": valid_count,
            "branch_time_in_parent": dataset.attrs.get(
                "branch_time_in_parent",
                np.nan,
            ),
            "branch_time_in_child": dataset.attrs.get(
                "branch_time_in_child",
                np.nan,
            ),
            "parent_time_units": str(
                dataset.attrs.get("parent_time_units", "")
            ).strip(),
            "time_units": time_units,
            "time_calendar": time_calendar,
            "time_start_numeric": time_start,
            "time_end_numeric": time_end,
            "requested_latitude_degN": float(target_latitude),
        }

        non_time_dims = [
            dim for dim in variable.dims if dim != time_dim
        ]

        # 已经是一维指数时，无法验证它是否为固定纬度定义。
        if not non_time_dims:
            if not ALLOW_PRECOMPUTED_1D_INDEX:
                raise ValueError(
                    f"{model} 的{variable.name}是一维预计算指数，无法确认是否采用"
                    f"固定{target_latitude:.1f}°N定义。若确认定义一致，可将"
                    "ALLOW_PRECOMPUTED_1D_INDEX设为True。"
                )

            annual = annual_weighted_mean(variable, time_dim)
            annual_sv, conversion = convert_amoc_to_sv(annual, units)
            annual_sv = np.asarray(annual_sv, dtype=float).squeeze()

            if annual_sv.ndim != 1 or np.isfinite(annual_sv).sum() == 0:
                raise ValueError(
                    f"{model} 的一维AMOC指数年平均后没有有效值，"
                    f"shape={annual_sv.shape}。"
                )

            metadata.update(
                {
                    "definition": "precomputed 1-D AMOC index",
                    "basin_selection": "not applicable",
                    "basin_index": None,
                    "actual_latitude_degN": np.nan,
                    "latitude_offset_deg": np.nan,
                    "latitude_index": None,
                    "latitude_dimension": "not applicable",
                    "latitude_mapping_method": "not applicable",
                    "latitude_mapping_source": "not applicable",
                    "depth_subset": "not applicable",
                    "piControl_maximum_depth_m": np.nan,
                    "extremum_mode": "precomputed",
                    "streamfunction_orientation_mode": "precomputed",
                    "unit_conversion": conversion,
                }
            )
            return annual_sv, metadata

        # 1. Atlantic basin。
        subset, basin_description, selected_basin_index = select_atlantic_basin(
            variable,
            dataset,
            time_dim,
            model,
            forced_index=basin_index,
        )

        # CESM/POP原始MOC包含Eulerian、bolus和submeso分量；AMOC强度应使用总和。
        subset, component_description = combine_moc_components(
            subset,
            dataset,
            model,
        )

        # 2. 固定纬度。
        latitude = find_latitude(dataset, subset)
        latitude_mapping_source = "native latitude coordinate"
        latitude_mapping_method = "direct"

        if latitude is None:
            latitude = build_j_index_latitude_mapping(model, subset)
            if latitude is None:
                raise ValueError(
                    f"{model} 无法识别纬度坐标，且不存在可映射的j-index维。"
                )

            latitude_mapping_source = str(
                latitude.attrs.get(
                    "mapping_source",
                    "fixed ocean grid",
                )
            )
            latitude_mapping_method = str(
                latitude.attrs.get(
                    "mapping_method",
                    "j-index mapping",
                )
            )

        (
            subset,
            actual_latitude,
            latitude_index,
            latitude_dim,
        ) = select_nearest_latitude(
            subset,
            latitude,
            target_latitude,
        )

        # 3. 500 m以下全部有效深度。
        depth = find_vertical_coordinate(dataset, subset)
        if depth is None or depth.ndim != 1:
            raise ValueError(f"{model} 无法识别一维深度坐标。")

        depth_dim = depth.dims[0]
        if depth_dim not in subset.dims:
            raise ValueError(
                f"深度维{depth_dim}不在固定纬度后的AMOC变量维度"
                f"{subset.dims}中。"
            )

        depth_m_all = depth_values_in_meters(depth)
        depth_indices = np.where(
            np.isfinite(depth_m_all)
            & (depth_m_all >= AMOC_MIN_DEPTH_M)
        )[0]

        if depth_indices.size == 0:
            raise ValueError(
                f"{model} 在{AMOC_MIN_DEPTH_M:.0f} m以下没有有效深度层。"
            )

        subset = subset.isel({depth_dim: depth_indices})
        selected_depth_m = depth_m_all[depth_indices]

        subset_valid_count = int(subset.count().compute().item())
        if subset_valid_count == 0:
            raise ValueError(
                f"{model} 在Atlantic、实际纬度{actual_latitude:.2f}°N、"
                f"{AMOC_MIN_DEPTH_M:.0f} m以下区域内全部为NaN。"
            )

        reduce_dims = [
            dim for dim in subset.dims if dim != time_dim
        ]
        if depth_dim not in reduce_dims:
            raise ValueError(
                f"{model} 固定纬度后缺少深度维，无法定义AMOC最大值。"
            )

        if extremum_mode not in {
            "max",
            "negative_min",
            "auto",
        }:
            raise ValueError(
                "extremum_mode只能为'max'、'negative_min'或'auto'。"
            )

        # 标准顺序：先形成年度平均流函数剖面，再沿深度取最大值。
        # max(annual mean profile) 与 annual mean(monthly max) 并不等价；
        # 前者更符合CMIP AMOC诊断文献的定义，也避免季节性最大深度跳动。
        annual_profile, annual_dim, annual_method = annual_mean_dataarray(
            subset,
            time_dim,
        )

        # 两种符号约定下的候选：A. 正向maximum；B. minimum乘以-1。
        annual_max_da = annual_profile.max(
            dim=reduce_dims,
            skipna=True,
        )
        annual_negative_min_da = -annual_profile.min(
            dim=reduce_dims,
            skipna=True,
        )

        annual_max_sv, conversion_max = convert_amoc_to_sv(
            np.asarray(annual_max_da.compute().values, dtype=float).squeeze(),
            units,
        )
        annual_max_sv = np.asarray(
            annual_max_sv,
            dtype=float,
        ).squeeze()

        (
            annual_negative_min_sv,
            conversion_negative_min,
        ) = convert_amoc_to_sv(
            np.asarray(
                annual_negative_min_da.compute().values,
                dtype=float,
            ).squeeze(),
            units,
        )
        annual_negative_min_sv = np.asarray(
            annual_negative_min_sv,
            dtype=float,
        ).squeeze()

        max_mean = float(np.nanmean(annual_max_sv))
        negative_min_mean = float(
            np.nanmean(annual_negative_min_sv)
        )

        selected_orientation = extremum_mode

        if extremum_mode == "auto":
            # 始终优先正向maximum。只有maximum接近0且 -minimum 明显为正，
            # 才认为该模式采用了相反符号约定。
            max_is_implausibly_small = (
                not np.isfinite(max_mean)
                or max_mean < 1.0
            )
            negative_min_is_plausible = (
                np.isfinite(negative_min_mean)
                and negative_min_mean >= 2.0
                and negative_min_mean <= 80.0
            )

            if (
                max_is_implausibly_small
                and negative_min_is_plausible
            ):
                selected_orientation = "negative_min"
            else:
                selected_orientation = "max"

        if selected_orientation == "negative_min":
            annual_sv = annual_negative_min_sv
            conversion = conversion_negative_min
            extremum_description = (
                "negative of minimum over depth "
                "(streamfunction sign corrected)"
            )
        else:
            annual_sv = annual_max_sv
            conversion = conversion_max
            extremum_description = "maximum over depth only"

        if annual_sv.ndim != 1 or np.isfinite(annual_sv).sum() == 0:
            raise ValueError(
                f"{model} 的固定纬度AMOC年平均后没有有效值，"
                f"shape={annual_sv.shape}。"
            )

        selected_mean = float(np.nanmean(annual_sv))
        if (
            not np.isfinite(selected_mean)
            or abs(selected_mean) < 1.0e-8
        ):
            raise ValueError(
                f"{model} 固定{target_latitude:.1f}°N提取到的AMOC控制强度"
                "接近0 Sv，无法作为piControl基准。"
                f" 正向maximum均值={max_mean:.6g} Sv；"
                f"-minimum均值={negative_min_mean:.6g} Sv；"
                f"实际纬度={actual_latitude:.3f}°N；"
                f"纬度索引={latitude_index}。"
                "这通常表示流函数符号或j行映射仍不正确。"
            )

        # 控制态平均剖面上的代表深度，用于质量检查。
        try:
            profile = subset.mean(
                dim=time_dim,
                skipna=True,
            )
            other_profile_dims = [
                dim for dim in profile.dims if dim != depth_dim
            ]

            if other_profile_dims:
                if selected_orientation == "negative_min":
                    profile = profile.min(
                        dim=other_profile_dims,
                        skipna=True,
                    )
                else:
                    profile = profile.max(
                        dim=other_profile_dims,
                        skipna=True,
                    )

            profile_values = np.asarray(
                profile.compute().values,
                dtype=float,
            ).reshape(-1)

            if np.isfinite(profile_values).any():
                if selected_orientation == "negative_min":
                    depth_index = int(np.nanargmin(profile_values))
                else:
                    depth_index = int(np.nanargmax(profile_values))
                maximum_depth_m = float(selected_depth_m[depth_index])
            else:
                maximum_depth_m = np.nan
        except Exception:
            maximum_depth_m = np.nan

        print(
            f"      {model} 流函数方向诊断："
            f"max均值={max_mean:.3f} Sv；"
            f"-min均值={negative_min_mean:.3f} Sv；"
            f"采用={selected_orientation}"
        )

        metadata.update(
            {
                "definition": (
                    f"Atlantic AMOC at fixed {target_latitude:.1f} degN "
                    f"(actual {actual_latitude:.2f} degN), "
                    f"depth >= {AMOC_MIN_DEPTH_M:.0f} m; "
                    f"{extremum_description}"
                ),
                "basin_selection": basin_description,
                "basin_index": selected_basin_index,
                "MOC component treatment": component_description,
                "annualization method": annual_method,
                "annual dimension": annual_dim,
                "actual_latitude_degN": actual_latitude,
                "latitude_offset_deg": abs(
                    actual_latitude - target_latitude
                ),
                "latitude_index": latitude_index,
                "latitude_dimension": latitude_dim,
                "latitude_mapping_method": latitude_mapping_method,
                "latitude_mapping_source": latitude_mapping_source,
                "depth_subset": f">={AMOC_MIN_DEPTH_M:.0f} m",
                "piControl_maximum_depth_m": maximum_depth_m,
                "extremum_mode": extremum_description,
                "streamfunction_orientation_mode": selected_orientation,
                "candidate_max_mean_sv": max_mean,
                "candidate_negative_min_mean_sv": negative_min_mean,
                "unit_conversion": conversion,
            }
        )

        return annual_sv, metadata

    finally:
        dataset.close()


# =============================================================================
# 8. piControl 订正、CO2 路径和指标
# =============================================================================


def infer_branch_start_index(
    forced_metadata: dict[str, Any],
    pi_metadata: dict[str, Any],
    pi_length: int,
) -> int:
    """根据 branch_time_in_parent 将强迫试验与 piControl 分支时刻对齐。

    CMIP6 的 branch_time_in_parent 通常与父试验 time 使用相同单位。
    对数值时间坐标，直接用分支时间与 piControl 起始时间之差换算成年。
    无法可靠解析时返回0，并由后续初始异常QA提示。
    """
    branch = forced_metadata.get("branch_time_in_parent", np.nan)
    pi_start = pi_metadata.get("time_start_numeric", np.nan)
    units = str(pi_metadata.get("time_units", "")).lower()
    calendar = str(pi_metadata.get("time_calendar", "")).lower()

    try:
        branch = float(branch)
        pi_start = float(pi_start)
    except Exception:
        return 0

    if not np.isfinite(branch):
        return 0
    if not np.isfinite(pi_start):
        pi_start = 0.0

    delta = branch - pi_start
    if "day" in units:
        if "360" in calendar:
            days_per_year = 360.0
        elif calendar in {"gregorian", "standard", "proleptic_gregorian"}:
            days_per_year = 365.2425
        else:
            days_per_year = 365.0
        years = delta / days_per_year
    elif "month" in units:
        years = delta / 12.0
    elif "year" in units:
        years = delta
    else:
        # 常见CMIP6数值时间若量级大于1000，多半为days since ...
        if "360" in calendar:
            inferred_days_per_year = 360.0
        else:
            inferred_days_per_year = 365.0
        years = delta / inferred_days_per_year if abs(delta) > 1000.0 else delta

    index = int(np.floor(years + 1.0e-6))
    return int(np.clip(index, 0, max(0, pi_length - 1)))


def build_pi_reference(
    pi_values: np.ndarray,
    forced_length: int,
    method: str = "linear",
    start_index: int = 0,
) -> tuple[np.ndarray, float, float, int]:
    """构造与强迫试验分支时刻对齐的piControl参考序列。

    旧版错误地用“整个piControl均值 + 斜率×时间”作为年0参考，
    相当于把控制试验中段均值放到分支时刻，会制造初始偏差。
    本版从branch对应的控制段开始拟合，并使用拟合截距作为年0参考。
    """
    pi_values = np.asarray(pi_values, dtype=float).reshape(-1)
    if np.isfinite(pi_values).sum() < 20:
        raise ValueError("piControl 有效年数少于20年。")

    start_index = int(np.clip(start_index, 0, max(0, pi_values.size - 1)))
    segment = pi_values[start_index:]
    valid_segment = np.isfinite(segment)

    # 分支后的控制段太短或缺测过多时，退回整个piControl，但不再使用全段均值作截距。
    if valid_segment.sum() < 20:
        start_index = 0
        segment = pi_values
        valid_segment = np.isfinite(segment)

    time = np.arange(segment.size, dtype=float)
    pi_mean = float(np.nanmean(segment))

    if method == "mean":
        return np.full(forced_length, pi_mean), pi_mean, 0.0, start_index

    if method != "linear":
        raise ValueError("PI_CORRECTION 只能为 linear 或 mean。")

    slope, intercept = np.polyfit(
        time[valid_segment],
        segment[valid_segment],
        1,
    )
    forced_time = np.arange(forced_length, dtype=float)
    reference = intercept + slope * forced_time

    return reference, pi_mean, float(slope), start_index


def running_mean(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1:
        return values.copy()

    return (
        pd.Series(values)
        .rolling(
            window=window,
            center=True,
            min_periods=max(1, window // 2),
        )
        .mean()
        .to_numpy()
    )


def create_co2_series(
    total_years: int,
    baseline: float = CO2_BASELINE,
    ramp_up_years: int = RAMP_UP_YEARS,
) -> np.ndarray:
    exponent = np.zeros(total_years, dtype=float)

    for index in range(total_years):
        if index < ramp_up_years:
            exponent[index] = index + 1
        else:
            exponent[index] = max(2 * ramp_up_years - 1 - index, 0)

    return baseline * (1.01 ** exponent)


def calculate_hysteresis_area(
    percent_anomaly: np.ndarray,
    co2_ratio: np.ndarray,
    peak_index: int,
    return_index: int,
) -> float:
    values = np.asarray(percent_anomaly, dtype=float)
    log_co2 = np.log(np.asarray(co2_ratio, dtype=float))

    ramp_x = log_co2[: peak_index + 1]
    ramp_y = values[: peak_index + 1]

    removal_x = log_co2[peak_index : return_index + 1][::-1]
    removal_y = values[peak_index : return_index + 1][::-1]

    valid_ramp = np.isfinite(ramp_x) & np.isfinite(ramp_y)
    valid_removal = np.isfinite(removal_x) & np.isfinite(removal_y)

    if valid_ramp.sum() < 5 or valid_removal.sum() < 5:
        return np.nan

    ramp_x = ramp_x[valid_ramp]
    ramp_y = ramp_y[valid_ramp]
    removal_x = removal_x[valid_removal]
    removal_y = removal_y[valid_removal]

    overlap_min = max(np.nanmin(ramp_x), np.nanmin(removal_x))
    overlap_max = min(np.nanmax(ramp_x), np.nanmax(removal_x))

    if overlap_max <= overlap_min:
        return np.nan

    common_x = np.linspace(overlap_min, overlap_max, 300)
    ramp_interp = np.interp(common_x, ramp_x, ramp_y)
    removal_interp = np.interp(common_x, removal_x, removal_y)

    return float(np.trapz(np.abs(removal_interp - ramp_interp), common_x))


def calculate_metrics(
    model: str,
    anomaly: np.ndarray,
    pi_mean: float,
    simulation_years: np.ndarray,
    co2_ratio: np.ndarray,
) -> dict[str, float | str]:
    anomaly = np.asarray(anomaly, dtype=float)

    if not np.isfinite(pi_mean) or abs(float(pi_mean)) < 1.0e-8:
        raise ValueError(
            f"{model} 的piControl AMOC均值为{pi_mean!r} Sv，"
            "无法计算百分比异常。"
        )

    smooth = running_mean(anomaly)
    percent = anomaly / pi_mean * 100.0
    percent_smooth = running_mean(percent)

    peak_index = min(RAMP_UP_YEARS - 1, len(anomaly) - 1)
    return_index = min(2 * RAMP_UP_YEARS - 1, len(anomaly) - 1)

    post_peak = np.asarray(smooth[peak_index:], dtype=float)
    finite_post_peak = np.where(np.isfinite(post_peak))[0]
    if finite_post_peak.size == 0:
        raise ValueError(f"{model} 在CO2峰值之后没有有效的平滑AMOC值。")
    local_index = finite_post_peak[int(np.argmin(post_peak[finite_post_peak]))]
    minimum_index = peak_index + int(local_index)
    minimum_anomaly = float(smooth[minimum_index])
    minimum_percent = float(percent_smooth[minimum_index])
    lag = int(simulation_years[minimum_index] - simulation_years[peak_index])

    final_years = min(FINAL_MEAN_YEARS, len(anomaly))
    final_anomaly = float(np.nanmean(anomaly[-final_years:]))
    final_percent = float(np.nanmean(percent[-final_years:]))

    if np.isclose(minimum_anomaly, 0.0):
        recovery = np.nan
    else:
        recovery = (final_anomaly - minimum_anomaly) / (0.0 - minimum_anomaly) * 100.0

    if return_index < len(anomaly) - 4:
        stable_values = anomaly[return_index:]
        stable_time = np.arange(len(stable_values), dtype=float)
        valid = np.isfinite(stable_values)
        if valid.sum() >= 5:
            stable_slope = float(
                np.polyfit(stable_time[valid], stable_values[valid], 1)[0]
            )
        else:
            stable_slope = np.nan
    else:
        stable_slope = np.nan

    hysteresis_area = calculate_hysteresis_area(
        percent_smooth,
        co2_ratio,
        peak_index,
        return_index,
    )

    initial_10yr = float(np.nanmean(anomaly[: min(10, len(anomaly))]))
    initial_10yr_percent = initial_10yr / pi_mean * 100.0

    return {
        "Model": model,
        "piControl climatology (Sv)": pi_mean,
        "Initial 10-yr anomaly (Sv)": initial_10yr,
        "Initial 10-yr anomaly (%)": initial_10yr_percent,
        "Minimum anomaly (Sv)": minimum_anomaly,
        "Maximum weakening (Sv)": -minimum_anomaly,
        "Minimum AMOC strength relative to piControl (Sv)": pi_mean + minimum_anomaly,
        "Maximum weakening (%)": -minimum_percent,
        "Minimum simulation year": int(simulation_years[minimum_index]),
        "Lag after CO2 peak (yr)": lag,
        f"Final {final_years}-yr anomaly (Sv)": final_anomaly,
        f"Final {final_years}-yr anomaly (%)": final_percent,
        "End-of-run recovery (%)": recovery,
        "Normalized hysteresis area (% × ln(C/C0))": hysteresis_area,
        "Post-removal stabilization trend (Sv yr-1)": stable_slope,
    }


# =============================================================================
# 9. 单模式处理
# =============================================================================


def process_model(
    model: str,
    configuration: dict[str, Any],
    target_latitude: float = MAIN_AMOC_LATITUDE,
    strict_qa: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    aliases = configuration["aliases"]

    up_files = find_model_files(DIR_1PCTCO2, aliases)
    cdr_files = find_model_files(DIR_CDR, aliases)
    pi_files = find_model_files(DIR_PI, aliases)

    missing = []
    if not up_files:
        missing.append("1pctCO2")
    if not cdr_files:
        missing.append("1pctCO2-cdr")
    if not pi_files:
        missing.append("piControl")
    if missing:
        raise FileNotFoundError("缺少：" + ", ".join(missing))

    variant_pairing = "按目录中唯一/全部可用文件读取"
    if model == "NorESM2-LM":
        up_files, cdr_files, pi_files, variant_pairing = (
            select_noresm_stage_files(up_files, cdr_files, pi_files)
        )
    else:
        up_files, cdr_files, pi_files, variant_pairing = (
            select_common_variant_files(
                up_files,
                cdr_files,
                pi_files,
                model,
            )
        )

    if verbose:
        print(f"\n处理模式：{model}（固定{target_latitude:.1f}°N）")
        print(f"  1pctCO2 文件数：{len(up_files)}")
        print(f"  CDR 文件数：{len(cdr_files)}")
        print(f"  piControl 文件数：{len(pi_files)}")
        if model == "NorESM2-LM":
            print(f"  NorESM阶段配对：{variant_pairing}")

    # CNRM的msftyz可能采用与其他模式相反的流函数符号。
    # 先在piControl中诊断方向，再把同一方向固定应用到1pctCO2和CDR。
    pi_extremum_mode = (
        "auto"
        if model == "CNRM-ESM2-1"
        else AMOC_EXTREMUM_MODE
    )

    try:
        pi_raw, pi_meta = extract_annual_amoc(
            pi_files,
            model,
            basin_index=None,
            extremum_mode=pi_extremum_mode,
            target_latitude=target_latitude,
            expected_experiment="piControl",
        )
    except Exception as error:
        raise RuntimeError(f"piControl 提取失败：{error}") from error

    selected_basin_index = pi_meta.get("basin_index")
    selected_orientation = str(
        pi_meta.get(
            "streamfunction_orientation_mode",
            AMOC_EXTREMUM_MODE,
        )
    )

    if selected_orientation not in {"max", "negative_min"}:
        selected_orientation = AMOC_EXTREMUM_MODE

    try:
        up_raw, up_meta = extract_annual_amoc(
            up_files,
            model,
            basin_index=selected_basin_index,
            extremum_mode=selected_orientation,
            target_latitude=target_latitude,
            expected_experiment="1pctCO2",
        )
    except Exception as error:
        raise RuntimeError(f"1pctCO2 提取失败：{error}") from error

    try:
        cdr_raw, cdr_meta = extract_annual_amoc(
            cdr_files,
            model,
            basin_index=selected_basin_index,
            extremum_mode=selected_orientation,
            target_latitude=target_latitude,
            expected_experiment="1pctCO2-cdr",
        )
    except Exception as error:
        raise RuntimeError(f"1pctCO2-cdr 提取失败：{error}") from error

    if len(up_raw) < RAMP_UP_YEARS:
        raise ValueError(f"1pctCO2 只有 {len(up_raw)} 年。")
    if len(cdr_raw) < MIN_CDR_YEARS:
        raise ValueError(
            f"1pctCO2-cdr 只有 {len(cdr_raw)} 年，少于 {MIN_CDR_YEARS} 年。"
        )

    if model != "NorESM2-LM":
        metadata_variants = [
            str(meta.get("variant_label", "")).strip()
            for meta in (pi_meta, up_meta, cdr_meta)
        ]
        metadata_variants = [label for label in metadata_variants if label]
        if len(set(metadata_variants)) > 1:
            raise ValueError(
                f"{model} 的piControl/1pctCO2/CDR全局属性variant_label不一致："
                f"{metadata_variants}。"
            )

    if model == "NorESM2-LM":
        up_variant = str(up_meta.get("variant_label", "")).strip()
        cdr_variant = str(cdr_meta.get("variant_label", "")).strip()
        parent_experiment = str(
            cdr_meta.get("parent_experiment_id", "")
        ).strip()
        parent_variant = str(
            cdr_meta.get("parent_variant_label", "")
        ).strip()

        if up_variant and "p1" not in up_variant.lower():
            raise ValueError(
                f"NorESM2-LM 1pctCO2全局属性variant_label={up_variant!r}，"
                "与CDR父实验所要求的p1不一致。"
            )
        if cdr_variant and "p1" not in cdr_variant.lower():
            raise ValueError(
                f"NorESM2-LM CDR全局属性variant_label={cdr_variant!r}，"
                "与文件选择所要求的p1不一致。"
            )
        if parent_experiment and normalize_name(parent_experiment) != "1PCTCO2":
            raise ValueError(
                "NorESM2-LM CDR的parent_experiment_id不是1pctCO2："
                f"{parent_experiment!r}。"
            )
        if parent_variant and up_variant and parent_variant != up_variant:
            raise ValueError(
                "NorESM2-LM CDR并非从所用1pctCO2成员分支："
                f"parent_variant_label={parent_variant!r}，"
                f"1pctCO2 variant_label={up_variant!r}。"
            )

        # NorESM2-LM 的 CDR 文件把分支时刻写成
        # branch_time_in_parent=0，时间原点则是 0141-01-01。
        # 这里的 0 表示从 1pctCO2 第141年开始，而不是第1年开始。
        cdr_branch_time = str(
            cdr_meta.get("branch_time_in_parent", "")
        ).strip()
        cdr_parent_time_units = str(
            cdr_meta.get("parent_time_units", "")
        ).lower()

        if (
            cdr_branch_time in {"0", "0.0"}
            and "since 0141-01-01" in cdr_parent_time_units
        ):
            cdr_branch_index = RAMP_UP_YEARS
        else:
            cdr_branch_index = infer_branch_start_index(
                cdr_meta,
                up_meta,
                pi_length=len(up_raw),
            )
        if abs(cdr_branch_index - RAMP_UP_YEARS) > 1:
            raise ValueError(
                "NorESM2-LM CDR分支时刻与1pctCO2第140年不一致："
                f"推断year-index={cdr_branch_index}。"
            )

    # 三个实验实际选取纬度应基本一致。
    actual_lats = np.asarray(
        [
            pi_meta.get("actual_latitude_degN", np.nan),
            up_meta.get("actual_latitude_degN", np.nan),
            cdr_meta.get("actual_latitude_degN", np.nan),
        ],
        dtype=float,
    )
    finite_lats = actual_lats[np.isfinite(actual_lats)]
    if finite_lats.size and np.nanmax(finite_lats) - np.nanmin(finite_lats) > 1.0:
        raise ValueError(
            f"piControl/1pctCO2/CDR实际纬度不一致：{actual_lats.tolist()}。"
        )

    # 固定纬度正向maximum通常应为正。仅在控制态均值为负时统一翻转符号。
    sign_flipped = float(np.nanmean(pi_raw)) < 0.0
    if sign_flipped:
        pi_raw = -pi_raw
        up_raw = -up_raw
        cdr_raw = -cdr_raw

    available_total = RAMP_UP_YEARS + len(cdr_raw)
    total_length = min(PREFERRED_TOTAL_YEARS, available_total)
    cdr_length = total_length - RAMP_UP_YEARS

    combined = np.concatenate(
        [
            np.asarray(up_raw[:RAMP_UP_YEARS], dtype=float),
            np.asarray(cdr_raw[:cdr_length], dtype=float),
        ]
    )

    branch_start_index = infer_branch_start_index(
        up_meta,
        pi_meta,
        pi_length=len(pi_raw),
    )

    reference_mean, pi_mean, _, reference_start_index = build_pi_reference(
        pi_raw,
        forced_length=total_length,
        method="mean",
        start_index=branch_start_index,
    )
    reference_linear, _, pi_slope, _ = build_pi_reference(
        pi_raw,
        forced_length=total_length,
        method="linear",
        start_index=branch_start_index,
    )

    if PI_CORRECTION == "mean":
        reference = reference_mean
    elif PI_CORRECTION == "linear":
        reference = reference_linear
    else:
        raise ValueError("PI_CORRECTION 只能为 mean 或 linear。")

    # 阻止后续百分比计算出现float division by zero，并给出可诊断信息。
    if not np.isfinite(pi_mean) or abs(pi_mean) < 1.0e-8:
        raise ValueError(
            f"piControl AMOC均值为{pi_mean!r} Sv，不能作为异常百分比基准。"
            f" 模式={model}；"
            f"流函数方向={pi_meta.get('streamfunction_orientation_mode')}；"
            f"max候选均值={pi_meta.get('candidate_max_mean_sv')} Sv；"
            f"-min候选均值={pi_meta.get('candidate_negative_min_mean_sv')} Sv；"
            f"实际纬度={pi_meta.get('actual_latitude_degN')}°N；"
            f"纬度索引={pi_meta.get('latitude_index')}。"
        )

    anomaly_mean_reference = combined - reference_mean
    anomaly_linear_reference = combined - reference_linear
    anomaly = combined - reference

    if np.isfinite(combined).sum() == 0:
        raise ValueError("拼接后的强迫实验AMOC序列全部为NaN。")
    if np.isfinite(anomaly).sum() == 0:
        raise ValueError("相对于piControl订正后的AMOC异常全部为NaN。")

    initial_mean = float(np.nanmean(anomaly[:10]))
    transition_jump = float(
        np.nanmean(cdr_raw[:5])
        - np.nanmean(up_raw[RAMP_UP_YEARS - 5:RAMP_UP_YEARS])
    )
    smoothed_anomaly = running_mean(anomaly)
    minimum_anomaly = float(np.nanmin(smoothed_anomaly))
    maximum_weakening_percent = float(-minimum_anomaly / pi_mean * 100.0)
    minimum_forced_strength = float(np.nanmin(running_mean(combined)))

    if verbose:
        print(
            f"  变量：{up_meta['variable']} | 盆地：{pi_meta['basin_selection']} | "
            f"流函数方向：{pi_meta.get('streamfunction_orientation_mode', 'max')} | "
            f"单位：{up_meta['unit_conversion']}"
        )
        print(
            f"  请求纬度：{target_latitude:.1f}°N | "
            f"实际纬度(pi/up/cdr)："
            f"{pi_meta.get('actual_latitude_degN', np.nan):.2f}/"
            f"{up_meta.get('actual_latitude_degN', np.nan):.2f}/"
            f"{cdr_meta.get('actual_latitude_degN', np.nan):.2f}°N | "
            f"控制态最大深度：{pi_meta.get('piControl_maximum_depth_m', np.nan):.0f} m"
        )
        print(
            f"  纬度定位：{pi_meta.get('latitude_mapping_method', 'direct')} | "
            f"纬度维：{pi_meta.get('latitude_dimension', 'unknown')} | "
            f"索引：{pi_meta.get('latitude_index', 'unknown')}"
        )
        print(
            f"  piControl：{pi_mean:.2f} Sv | 漂移：{pi_slope * 100:+.2f} Sv/century | "
            f"控制段起点：year-index {reference_start_index} | 分析长度：{total_length} yr"
        )
        print(
            f"  最低强迫AMOC：{minimum_forced_strength:.2f} Sv | "
            f"最大减弱：{minimum_anomaly:+.2f} Sv ({maximum_weakening_percent:.1f}%) | "
            f"前10年均值：{initial_mean:+.2f} Sv | up→CDR接缝：{transition_jump:+.2f} Sv"
        )

    qa_warnings: list[str] = []
    qa_failures: list[str] = []

    if abs(initial_mean) > 1.5:
        qa_warnings.append(
            f"前10年平均异常为{initial_mean:+.2f} Sv，超过1.5 Sv"
        )
    if abs(initial_mean) > 3.0:
        qa_failures.append(
            f"前10年平均异常为{initial_mean:+.2f} Sv，超过硬阈值3 Sv；"
            "强迫分支与piControl很可能没有正确对齐"
        )

    if abs(transition_jump) > 2.0:
        qa_warnings.append(
            f"1pctCO2与CDR接缝跳变为{transition_jump:+.2f} Sv，超过2 Sv"
        )
    if abs(transition_jump) > 4.0:
        qa_failures.append(
            f"1pctCO2与CDR接缝跳变为{transition_jump:+.2f} Sv，超过硬阈值4 Sv；"
            "CDR可能不是从所用1pctCO2成员正确分支"
        )

    if not (2.0 <= pi_mean <= 50.0):
        qa_failures.append(
            f"piControl AMOC为{pi_mean:.2f} Sv，超出宽松2–50 Sv范围"
        )

    if qa_warnings and verbose:
        print("  诊断警告：" + "；".join(qa_warnings))

    if qa_failures and strict_qa:
        raise ValueError("；".join(qa_failures) + "，已停止纳入主多模式平均。")

    return {
        "model": model,
        "target_latitude": target_latitude,
        "anomaly": anomaly,
        "anomaly_mean_reference": anomaly_mean_reference,
        "anomaly_linear_reference": anomaly_linear_reference,
        "combined_strength": combined,
        "reference": reference,
        "reference_mean": reference_mean,
        "reference_linear": reference_linear,
        "pi_mean": pi_mean,
        "pi_slope": pi_slope,
        "pi_reference_start_index": reference_start_index,
        "branch_start_index": branch_start_index,
        "initial_10yr_anomaly_mean": initial_mean,
        "transition_jump_sv": transition_jump,
        "minimum_forced_strength_sv": minimum_forced_strength,
        "maximum_weakening_percent_quick": maximum_weakening_percent,
        "sign_flipped": sign_flipped,
        "variant_pairing": variant_pairing,
        "up_file_count": len(up_files),
        "cdr_file_count": len(cdr_files),
        "pi_file_count": len(pi_files),
        "metadata": pi_meta,
        "up_metadata": up_meta,
        "cdr_metadata": cdr_meta,
        "length": total_length,
    }



def make_cnrm_initial_anomaly_diagnostic(
    result: dict[str, Any],
) -> None:
    """检查CNRM初始异常对piControl基线选择的敏感性。

    本函数输出年度CSV、摘要CSV和一张基线诊断图，但不会为了让初始异常
    归零而人为平移AMOC序列。
    """
    if not RUN_CNRM_INITIAL_ANOMALY_DIAGNOSTIC:
        return

    model = str(result.get("model", "CNRM-ESM2-1"))
    target_latitude = float(
        result.get("target_latitude", MAIN_AMOC_LATITUDE)
    )

    combined = np.asarray(
        result["combined_strength"],
        dtype=float,
    )
    reference_mean = np.asarray(
        result["reference_mean"],
        dtype=float,
    )
    reference_linear = np.asarray(
        result["reference_linear"],
        dtype=float,
    )
    anomaly_primary = np.asarray(
        result["anomaly"],
        dtype=float,
    )
    anomaly_mean = np.asarray(
        result["anomaly_mean_reference"],
        dtype=float,
    )
    anomaly_linear = np.asarray(
        result["anomaly_linear_reference"],
        dtype=float,
    )

    length = min(
        INITIAL_DIAGNOSTIC_YEARS,
        combined.size,
        reference_mean.size,
        reference_linear.size,
    )
    if length < 10:
        print("CNRM初始异常诊断跳过：可用年数不足10年。")
        return

    simulation_year = np.arange(1, length + 1)
    calendar_year = CALENDAR_START_YEAR + simulation_year - 1

    annual = pd.DataFrame(
        {
            "Model": model,
            "Requested latitude (degN)": target_latitude,
            "Simulation year": simulation_year,
            "Calendar year": calendar_year,
            "Forced AMOC strength (Sv)": combined[:length],
            "piControl mean reference (Sv)": reference_mean[:length],
            "piControl branch-linear reference (Sv)": reference_linear[:length],
            "Primary AMOC anomaly (Sv)": anomaly_primary[:length],
            "Anomaly using piControl mean (Sv)": anomaly_mean[:length],
            "Anomaly using branch-linear piControl (Sv)": anomaly_linear[:length],
        }
    )

    def period_mean(values: np.ndarray, years: int) -> float:
        n = min(years, values.size)
        return float(np.nanmean(values[:n]))

    mean_10 = period_mean(anomaly_mean, 10)
    linear_10 = period_mean(anomaly_linear, 10)
    mean_20 = period_mean(anomaly_mean, 20)
    linear_20 = period_mean(anomaly_linear, 20)
    primary_10 = period_mean(anomaly_primary, 10)

    reduction = abs(mean_10) - abs(linear_10)

    if abs(mean_10) <= 2.0 and abs(linear_10) <= 2.0:
        interpretation = (
            "Initial anomaly is small under both references; "
            "no strong baseline-offset signal."
        )
        status = "PASS"
    elif abs(linear_10) <= 2.0 and reduction >= 1.0:
        interpretation = (
            "The branch-linear piControl reference substantially reduces "
            "the initial offset; control drift or low-frequency baseline "
            "choice is the leading explanation."
        )
        status = "BASELINE_SENSITIVE"
    elif abs(mean_10) > 2.0 and abs(linear_10) > 2.0:
        interpretation = (
            "The initial offset persists under both references; inspect "
            "branch_time_in_parent, ensemble/member consistency, and "
            "piControl low-frequency variability. Do not remove it by "
            "arbitrary demeaning."
        )
        status = "PERSISTENT_OFFSET"
    else:
        interpretation = (
            "The initial offset is moderately sensitive to the reference; "
            "report both mean and branch-linear sensitivity results."
        )
        status = "MODERATE_SENSITIVITY"

    pi_meta = result.get("metadata", {})
    up_meta = result.get("up_metadata", {})

    summary = pd.DataFrame(
        [
            {
                "Model": model,
                "Requested latitude (degN)": target_latitude,
                "Actual latitude (degN)": pi_meta.get(
                    "actual_latitude_degN"
                ),
                "Latitude index": pi_meta.get("latitude_index"),
                "piControl climatology (Sv)": result.get("pi_mean"),
                "piControl trend (Sv century-1)": (
                    float(result.get("pi_slope", np.nan)) * 100.0
                ),
                "branch_time_in_parent": up_meta.get(
                    "branch_time_in_parent"
                ),
                "piControl time start": pi_meta.get(
                    "time_start_numeric"
                ),
                "piControl time units": pi_meta.get("time_units"),
                "Inferred branch index": result.get(
                    "branch_start_index"
                ),
                "Reference start index": result.get(
                    "pi_reference_start_index"
                ),
                "Primary first-10-yr anomaly (Sv)": primary_10,
                "Mean-reference first-10-yr anomaly (Sv)": mean_10,
                "Linear-reference first-10-yr anomaly (Sv)": linear_10,
                "Mean-reference first-20-yr anomaly (Sv)": mean_20,
                "Linear-reference first-20-yr anomaly (Sv)": linear_20,
                "Absolute-offset reduction by linear reference (Sv)": reduction,
                "Diagnostic status": status,
                "Interpretation": interpretation,
            }
        ]
    )

    lat_tag = str(target_latitude).replace(".", "p")
    annual_path = (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_CNRM_initial_{lat_tag}N_annual.csv"
    )
    summary_path = (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_CNRM_initial_{lat_tag}N_summary.csv"
    )

    annual.to_csv(
        annual_path,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    set_publication_style()
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(10.5, 8.0),
        sharex=True,
    )

    axes[0].plot(
        calendar_year,
        combined[:length],
        color=MODEL_INFO["CNRM-ESM2-1"]["color"],
        linewidth=2.2,
        label="Forced AMOC",
    )
    axes[0].plot(
        calendar_year,
        reference_mean[:length],
        color="black",
        linewidth=1.8,
        linestyle="--",
        label="piControl mean reference",
    )
    axes[0].plot(
        calendar_year,
        reference_linear[:length],
        color="0.45",
        linewidth=1.8,
        linestyle=":",
        label="Branch-linear piControl reference",
    )
    axes[0].set_ylabel("AMOC strength (Sv)")
    axes[0].set_title(
        f"CNRM-ESM2-1 initial-baseline diagnostic at "
        f"{target_latitude:.1f}°N"
    )
    axes[0].legend(
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
    )
    axes[0].grid(color="0.90", linewidth=0.6)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    axes[1].plot(
        calendar_year,
        anomaly_mean[:length],
        color="#0072B2",
        linewidth=2.0,
        label="Anomaly vs piControl mean",
    )
    axes[1].plot(
        calendar_year,
        anomaly_linear[:length],
        color="#D55E00",
        linewidth=2.0,
        linestyle="--",
        label="Anomaly vs branch-linear piControl",
    )
    axes[1].axhline(
        0.0,
        color="black",
        linewidth=1.0,
        linestyle=":",
    )
    axes[1].set_xlabel("Nominal calendar year")
    axes[1].set_ylabel("AMOC anomaly (Sv)")
    axes[1].legend(
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
    )
    axes[1].grid(color="0.90", linewidth=0.6)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    figure.tight_layout()
    save_figure(
        figure,
        f"Figure5_CNRM_initial_baseline_{lat_tag}N",
    )
    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)

    print("\n" + "=" * 82)
    print(
        f"CNRM-ESM2-1 初始异常诊断（固定{target_latitude:.1f}°N）"
    )
    print("=" * 82)
    print(f"主结果前10年异常：{primary_10:+.3f} Sv")
    print(f"均值参考前10年异常：{mean_10:+.3f} Sv")
    print(f"线性参考前10年异常：{linear_10:+.3f} Sv")
    print(f"均值参考前20年异常：{mean_20:+.3f} Sv")
    print(f"线性参考前20年异常：{linear_20:+.3f} Sv")
    print(f"诊断状态：{status}")
    print(f"判读：{interpretation}")
    print(f"年度诊断：{annual_path}")
    print(f"摘要诊断：{summary_path}")
    print("=" * 82)


# =============================================================================
# 10. 绿图辅助函数
# =============================================================================


def add_model_legend(
    figure: plt.Figure,
    models: list[str],
    y_anchor: float,
    ncol: int = 3,
) -> None:
    handles = []
    for model in models:
        style = MODEL_INFO[model]
        handles.append(
            Line2D(
                [],
                [],
                color=style["color"],
                linewidth=1.7,
                label=model,
            )
        )

    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, y_anchor),
        ncol=ncol,
        frameon=False,
        fontsize=8.0,
        handlelength=2.6,
        columnspacing=1.4,
    )


def save_figure(figure: plt.Figure, stem: str) -> None:
    for extension in ("png", "pdf", "svg"):
        path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.{extension}"
        if extension == "png":
            figure.savefig(
                path,
                dpi=600,
                bbox_inches="tight",
                pad_inches=0.10,
            )
        else:
            figure.savefig(
                path,
                bbox_inches="tight",
                pad_inches=0.10,
            )
        print(f"  {extension.upper()}：{path}")


# =============================================================================
# 11. Figure 1：AMOC 时间序列
# =============================================================================


def make_figure_1(
    model_results: dict[str, dict[str, Any]],
    common_length: int,
) -> None:
    """绘制单面板 AMOC 主图。

    风格与海冰主图保持一致：
    - 彩色细线：各模式；
    - 灰色带：模式间离散度；
    - 黑色粗线：多模式平均；
    - 紫色虚线：大气 CO2；
    - 左轴：AMOC 绝对异常（Sv）；
    - 右轴：大气 CO2（ppm）；
    - 横轴：日历年份；
    - 所有模式曲线、离散范围和MMM均采用11年中心滑动平均。
    """
    set_publication_style()

    models = [model for model in MODEL_INFO if model in model_results]
    if len(models) < 2:
        raise RuntimeError("有效模式少于2个，无法绘制AMOC多模式主图。")

    simulation_years = np.arange(1, common_length + 1)
    calendar_years = CALENDAR_START_YEAR + simulation_years - 1
    co2 = create_co2_series(common_length)

    anomaly_stack = np.vstack(
        [
            np.asarray(model_results[model]["anomaly"][:common_length], dtype=float)
            for model in models
        ]
    )
    anomaly_smooth = np.vstack([running_mean(row) for row in anomaly_stack])
    mmm_anomaly = np.nanmean(anomaly_smooth, axis=0)

    if FIGURE1_USE_STD_BAND:
        spread = np.nanstd(anomaly_smooth, axis=0, ddof=0)
        anomaly_low = mmm_anomaly - spread
        anomaly_high = mmm_anomaly + spread
        spread_label = r"$\pm$1 Std Dev"
    else:
        anomaly_low = np.nanquantile(anomaly_smooth, 0.17, axis=0)
        anomaly_high = np.nanquantile(anomaly_smooth, 0.83, axis=0)
        spread_label = "17th–83rd percentiles"

    figure, axis = plt.subplots(figsize=(13.8, 7.8))
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")

    # 模式间离散范围先画，避免遮盖模式曲线。
    axis.fill_between(
        calendar_years,
        anomaly_low,
        anomaly_high,
        facecolor="0.82",
        edgecolor="none",
        alpha=0.78,
        zorder=1,
    )

    # 各模式11年中心滑动平均曲线。
    for model_index, model in enumerate(models):
        axis.plot(
            calendar_years,
            anomaly_smooth[model_index],
            color=MODEL_INFO[model]["color"],
            linewidth=1.9,
            alpha=0.93,
            zorder=3,
        )

    # 多模式平均。
    axis.plot(
        calendar_years,
        mmm_anomaly,
        color="black",
        linewidth=4.0,
        zorder=5,
    )

    # piControl基准。
    axis.axhline(
        0.0,
        color="black",
        linewidth=1.8,
        linestyle=(0, (5, 3)),
        zorder=2,
    )

    axis.set_title(
        "AMOC Anomaly",
        fontsize=22,
        fontweight="bold",
        pad=22,
    )
    axis.set_xlabel("Calendar Year", fontsize=18, labelpad=13)
    axis.set_ylabel(
        "AMOC Anomaly (Sv)",
        fontsize=19,
        fontweight="bold",
        labelpad=14,
    )

    axis.set_xlim(calendar_years[0], calendar_years[-1])
    first_tick = int(np.ceil(calendar_years[0] / 50.0) * 50)
    last_tick = int(np.floor(calendar_years[-1] / 50.0) * 50)
    if first_tick <= last_tick:
        axis.set_xticks(np.arange(first_tick, last_tick + 1, 50))

    axis.minorticks_on()
    axis.tick_params(
        axis="both",
        which="major",
        direction="in",
        length=11,
        width=2.2,
        labelsize=15,
    )
    axis.tick_params(
        axis="both",
        which="minor",
        direction="in",
        length=5,
        width=1.5,
    )

    axis.spines["left"].set_linewidth(2.6)
    axis.spines["bottom"].set_linewidth(2.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(False)

    # 右轴CO2路径。
    co2_axis = axis.twinx()
    co2_axis.plot(
        calendar_years,
        co2,
        color="#B000C8",
        linewidth=3.5,
        linestyle=(0, (7, 4)),
        zorder=4,
    )
    co2_axis.set_ylim(CO2_AXIS_MIN, CO2_AXIS_MAX)
    co2_axis.set_yticks([200, 400, 600, 800, 1000, 1200])
    co2_axis.set_ylabel(
        "Atmospheric CO$_2$ (ppm)",
        fontsize=18,
        fontweight="bold",
        color="black",
        labelpad=18,
    )
    co2_axis.minorticks_on()
    co2_axis.tick_params(
        axis="y",
        which="major",
        direction="in",
        length=11,
        width=2.2,
        labelsize=15,
        colors="black",
    )
    co2_axis.tick_params(
        axis="y",
        which="minor",
        direction="in",
        length=5,
        width=1.5,
        colors="black",
    )
    co2_axis.spines["right"].set_linewidth(2.6)
    co2_axis.spines["right"].set_color("black")
    co2_axis.spines["top"].set_visible(False)
    co2_axis.spines["left"].set_visible(False)

    # 右侧统一图例。
    legend_handles: list[Any] = [
        Line2D(
            [],
            [],
            color=MODEL_INFO[model]["color"],
            linewidth=2.5,
            label=model,
        )
        for model in models
    ]
    legend_handles.extend(
        [
            Patch(
                facecolor="0.82",
                edgecolor="none",
                alpha=0.78,
                label=spread_label,
            ),
            Line2D(
                [],
                [],
                color="black",
                linewidth=4.0,
                label=f"{len(models)}-model MMM",
            ),
            Line2D(
                [],
                [],
                color="#B000C8",
                linewidth=3.5,
                linestyle=(0, (7, 4)),
                label="Atmospheric CO$_2$",
            ),
        ]
    )

    legend = axis.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(1.14, 0.50),
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        edgecolor="black",
        fontsize=11.4,
        handlelength=2.8,
        borderpad=0.65,
        labelspacing=0.55,
    )
    legend.get_frame().set_linewidth(1.4)

    figure.subplots_adjust(
        left=0.095,
        right=0.72,
        bottom=0.14,
        top=0.88,
    )

    print("\n保存 Figure 1（单面板海冰风格）：")
    save_figure(figure, "Figure1_AMOC_anomaly_timeseries")

    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)


# =============================================================================
# 12. Figure 2：AMOC–CO2 滞回环
# =============================================================================


def make_figure_2(
    model_results: dict[str, dict[str, Any]],
    common_length: int,
) -> None:
    set_publication_style()

    models = [model for model in MODEL_INFO if model in model_results]
    co2 = create_co2_series(common_length)
    co2_ratio = co2 / CO2_BASELINE

    peak_index = min(RAMP_UP_YEARS - 1, common_length - 1)
    return_index = min(2 * RAMP_UP_YEARS - 1, common_length - 1)

    percent_stack = np.vstack(
        [
            running_mean(
                model_results[model]["anomaly"][:common_length]
                / model_results[model]["pi_mean"]
                * 100.0
            )
            for model in models
        ]
    )
    mmm_percent = np.nanmean(percent_stack, axis=0)

    figure, axis = plt.subplots(figsize=(8.5, 6.6))

    ramp_slice = slice(0, peak_index + 1)
    removal_slice = slice(peak_index, return_index + 1)
    stabilization_slice = slice(return_index, common_length)

    # 个体模式轨迹
    for model_index, model in enumerate(models):
        color = MODEL_INFO[model]["color"]
        values = percent_stack[model_index]

        axis.plot(
            co2_ratio[ramp_slice],
            values[ramp_slice],
            color=color,
            linewidth=0.9,
            alpha=0.20,
            zorder=1,
        )
        axis.plot(
            co2_ratio[removal_slice],
            values[removal_slice],
            color=color,
            linewidth=0.9,
            alpha=0.20,
            zorder=1,
        )
        if return_index < common_length - 1:
            axis.plot(
                co2_ratio[stabilization_slice],
                values[stabilization_slice],
                color=color,
                linewidth=0.8,
                alpha=0.16,
                linestyle="--",
                zorder=1,
            )

    ramp_color = "#D55E00"
    removal_color = "#0072B2"

    axis.plot(
        co2_ratio[ramp_slice],
        mmm_percent[ramp_slice],
        color=ramp_color,
        linewidth=3.0,
        label="MMM: CO$_2$ ramp-up",
        zorder=5,
    )
    axis.plot(
        co2_ratio[removal_slice],
        mmm_percent[removal_slice],
        color=removal_color,
        linewidth=3.0,
        label="MMM: CO$_2$ removal",
        zorder=5,
    )
    if return_index < common_length - 1:
        axis.plot(
            co2_ratio[stabilization_slice],
            mmm_percent[stabilization_slice],
            color="0.20",
            linewidth=2.0,
            linestyle="--",
            label="MMM: post-removal stabilization",
            zorder=5,
        )

    axis.scatter(
        co2_ratio[peak_index],
        mmm_percent[peak_index],
        s=40,
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        label="CO$_2$ peak",
        zorder=7,
    )

    axis.scatter(
        co2_ratio[return_index],
        mmm_percent[return_index],
        s=42,
        marker="s",
        facecolor="white",
        edgecolor="black",
        linewidth=1.0,
        label="CO$_2$ returned to PI",
        zorder=7,
    )

    axis.axhline(0.0, color="0.45", linewidth=0.8, linestyle=":")
    axis.axvline(1.0, color="0.45", linewidth=0.8, linestyle=":")

    axis.set_xlim(0.95, max(4.1, float(np.nanmax(co2_ratio)) + 0.05))
    axis.set_xlabel("Atmospheric CO$_2$ / CO$_{2,PI}$")
    axis.set_ylabel("AMOC anomaly relative to piControl (%)")
    axis.set_title("AMOC path dependence during CO$_2$ ramp-up and removal")
    axis.grid(color="0.90", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.legend(
        loc="best",
        frameon=True,
        facecolor="white",
        edgecolor="0.80",
        framealpha=0.95,
        fontsize=8.0,
    )

    figure.tight_layout()

    print("\n保存 Figure 2：")
    save_figure(figure, "Figure2_AMOC_CO2_hysteresis")

    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)


# =============================================================================
# 13. Figure 3：逐模式 AMOC 指标
# =============================================================================


def make_figure_3(metrics: pd.DataFrame) -> None:
    set_publication_style()

    model_order = [model for model in MODEL_INFO if model in set(metrics["Model"])]
    metrics = metrics.set_index("Model").loc[model_order].reset_index()

    final_anomaly_column = [
        column
        for column in metrics.columns
        if column.startswith("Final") and column.endswith("anomaly (%)")
    ][0]

    panels = [
        (
            "Maximum weakening (%)",
            "Maximum AMOC weakening (%)",
            "(a)",
        ),
        (
            "Lag after CO2 peak (yr)",
            "Lag of maximum weakening (yr)",
            "(b)",
        ),
        (
            "End-of-run recovery (%)",
            "End-of-run AMOC recovery (%)",
            "(c)",
        ),
        (
            final_anomaly_column,
            "Final AMOC anomaly relative to piControl (%)",
            "(d)",
        ),
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 8.6))
    y_positions = np.arange(len(metrics))

    for axis, (column, title, panel) in zip(axes.flat, panels):
        values = metrics[column].to_numpy(dtype=float)

        for index, row in metrics.iterrows():
            model = str(row["Model"])
            axis.scatter(
                float(row[column]),
                index,
                s=62,
                facecolor=MODEL_INFO[model]["color"],
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )

        axis.axvline(0.0, color="0.55", linewidth=0.8, linestyle=":")
        if column == "End-of-run recovery (%)":
            axis.axvline(100.0, color="0.55", linewidth=0.8, linestyle="--")

        median = float(np.nanmedian(values))
        axis.axvline(
            median,
            color="black",
            linewidth=1.7,
            linestyle="-",
            alpha=0.75,
        )

        axis.set_yticks(y_positions)
        axis.set_yticklabels(metrics["Model"])
        axis.invert_yaxis()
        axis.set_xlabel(title)
        axis.grid(axis="x", color="0.90", linewidth=0.6)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.text(
            0.015,
            0.97,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11.5,
            fontweight="bold",
        )
        axis.text(
            0.98,
            0.05,
            f"Median = {median:+.1f}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.8,
            color="0.25",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "0.82",
                "alpha": 0.90,
            },
        )

    figure.subplots_adjust(
        left=0.15,
        right=0.98,
        bottom=0.09,
        top=0.97,
        wspace=0.30,
        hspace=0.30,
    )

    print("\n保存 Figure 3：")
    save_figure(figure, "Figure3_AMOC_model_metrics")

    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)


# =============================================================================
# 14. 质量控制
# =============================================================================


def build_qa_report(
    model_results: dict[str, dict[str, Any]],
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for model, result in model_results.items():
        pi_mean = float(result["pi_mean"])
        drift_century = float(result["pi_slope"] * 100.0)

        metric_row = metrics[metrics["Model"] == model].iloc[0]
        initial_percent = float(metric_row["Initial 10-yr anomaly (%)"])

        rows.append(
            {
                "Model": model,
                "Check": "piControl climatology",
                "Value": pi_mean,
                "Status": "PASS" if 2.0 <= pi_mean <= 50.0 else "WARN",
                "Comment": "Expected broad AMOC range: 2–50 Sv",
            }
        )
        rows.append(
            {
                "Model": model,
                "Check": "piControl drift",
                "Value": drift_century,
                "Status": "PASS" if abs(drift_century) <= 3.0 else "WARN",
                "Comment": "Sv per century",
            }
        )
        rows.append(
            {
                "Model": model,
                "Check": "Initial 10-yr anomaly",
                "Value": float(result["initial_10yr_anomaly_mean"]),
                "Status": (
                    "PASS"
                    if abs(float(result["initial_10yr_anomaly_mean"])) <= 1.5
                    else "WARN"
                ),
                "Comment": (
                    f"Sv; {initial_percent:+.1f}% of piControl; "
                    "large values indicate branch/control mismatch"
                ),
            }
        )
        transition = float(result["transition_jump_sv"])
        rows.append(
            {
                "Model": model,
                "Check": "1pctCO2-to-CDR continuity",
                "Value": transition,
                "Status": "PASS" if abs(transition) <= 2.0 else "WARN",
                "Comment": "first 5 yr CDR minus final 5 yr 1pctCO2 (Sv)",
            }
        )
        actual_lat = float(result["metadata"].get("actual_latitude_degN", np.nan))
        lat_offset = abs(actual_lat - MAIN_AMOC_LATITUDE) if np.isfinite(actual_lat) else np.nan
        rows.append(
            {
                "Model": model,
                "Check": "Fixed latitude selection",
                "Value": actual_lat,
                "Status": "PASS" if np.isfinite(lat_offset) and lat_offset <= MAX_LATITUDE_OFFSET_DEG else "WARN",
                "Comment": f"Requested {MAIN_AMOC_LATITUDE:.1f} degN; offset={lat_offset:.2f} deg",
            }
        )
        rows.append(
            {
                "Model": model,
                "Check": "Atlantic basin selection",
                "Value": result["metadata"].get("basin_index"),
                "Status": (
                    "PASS"
                    if "auto strongest" not in str(
                        result["metadata"].get("basin_selection", "")
                    ).lower()
                    else "WARN"
                ),
                "Comment": str(result["metadata"].get("basin_selection", "")),
            }
        )
        rows.append(
            {
                "Model": model,
                "Check": "MOC component treatment",
                "Value": "",
                "Status": "PASS",
                "Comment": str(
                    result["metadata"].get("MOC component treatment", "")
                ),
            }
        )
        rows.append(
            {
                "Model": model,
                "Check": "Annualization order",
                "Value": "",
                "Status": "PASS",
                "Comment": (
                    str(result["metadata"].get("annualization method", ""))
                    + "; annual mean profile before depth extremum"
                ),
            }
        )

    return pd.DataFrame(rows)


def make_cesm2_latitude_diagnostic(
    main_result: dict[str, Any],
    diagnostic_result: dict[str, Any],
) -> None:
    """比较CESM2固定26.5°N主结果与40°N敏感性结果。"""
    common_length = min(
        main_result["length"],
        diagnostic_result["length"],
    )
    years = CALENDAR_START_YEAR + np.arange(common_length)

    main_anomaly = running_mean(
        main_result["anomaly"][:common_length]
    )
    diagnostic_anomaly = running_mean(
        diagnostic_result["anomaly"][:common_length]
    )

    figure, axis = plt.subplots(figsize=(10.2, 5.8))
    axis.plot(
        years,
        main_anomaly,
        color="#E69F00",
        linewidth=2.8,
        label=f"CESM2 at {MAIN_AMOC_LATITUDE:.1f}°N",
    )
    axis.plot(
        years,
        diagnostic_anomaly,
        color="#0072B2",
        linewidth=2.4,
        linestyle="--",
        label=f"CESM2 at {DIAGNOSTIC_AMOC_LATITUDE:.1f}°N",
    )
    axis.axhline(
        0.0,
        color="black",
        linewidth=1.2,
        linestyle=(0, (5, 3)),
    )
    axis.set_xlabel("Nominal calendar year")
    axis.set_ylabel("AMOC anomaly (Sv)")
    axis.set_title("CESM2 fixed-latitude AMOC sensitivity")
    axis.grid(color="0.90", linewidth=0.6)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        frameon=True,
        facecolor="white",
        edgecolor="0.8",
    )
    figure.tight_layout()

    main_tag = str(MAIN_AMOC_LATITUDE).replace(".", "p")
    diagnostic_tag = str(
        DIAGNOSTIC_AMOC_LATITUDE
    ).replace(".", "p")
    save_figure(
        figure,
        f"Figure4_CESM2_{main_tag}N_vs_{diagnostic_tag}N",
    )

    if SHOW_FIGURE:
        plt.show()
    plt.close(figure)


def run_cesm2_latitude_diagnostic(
    main_result: dict[str, Any],
) -> None:
    if not RUN_CESM2_LATITUDE_DIAGNOSTIC:
        return
    if "CESM2" not in MODEL_INFO:
        return

    print("\n" + "-" * 82)
    print(
        "CESM2固定纬度敏感性诊断："
        f"{MAIN_AMOC_LATITUDE:.1f}°N vs "
        f"{DIAGNOSTIC_AMOC_LATITUDE:.1f}°N"
    )

    try:
        diagnostic = process_model(
            "CESM2",
            MODEL_INFO["CESM2"],
            target_latitude=DIAGNOSTIC_AMOC_LATITUDE,
            strict_qa=False,
            verbose=True,
        )
    except Exception as error:
        print(
            f"CESM2 {DIAGNOSTIC_AMOC_LATITUDE:.1f}°N诊断失败："
            f"{error}"
        )
        return

    common_length = min(
        main_result["length"],
        diagnostic["length"],
    )
    years = np.arange(1, common_length + 1)
    co2_ratio = (
        create_co2_series(common_length)
        / CO2_BASELINE
    )

    main_label = (
        f"CESM2_{str(MAIN_AMOC_LATITUDE).replace('.', 'p')}N"
    )
    diagnostic_label = (
        f"CESM2_{str(DIAGNOSTIC_AMOC_LATITUDE).replace('.', 'p')}N"
    )

    main_metrics = calculate_metrics(
        main_label,
        main_result["anomaly"][:common_length],
        float(main_result["pi_mean"]),
        years,
        co2_ratio,
    )
    diagnostic_metrics = calculate_metrics(
        diagnostic_label,
        diagnostic["anomaly"][:common_length],
        float(diagnostic["pi_mean"]),
        years,
        co2_ratio,
    )

    def final_percent(metrics: dict[str, Any]) -> float:
        return float(
            next(
                value
                for key, value in metrics.items()
                if key.startswith("Final")
                and key.endswith("anomaly (%)")
            )
        )

    main_definition = f"{MAIN_AMOC_LATITUDE:.1f}N main"
    diagnostic_definition = (
        f"{DIAGNOSTIC_AMOC_LATITUDE:.1f}N diagnostic"
    )

    summary = pd.DataFrame(
        [
            {
                "Definition": main_definition,
                "Requested latitude (degN)": MAIN_AMOC_LATITUDE,
                "Actual latitude (degN)": (
                    main_result["metadata"].get(
                        "actual_latitude_degN"
                    )
                ),
                "piControl AMOC (Sv)": main_result["pi_mean"],
                "Maximum weakening (Sv)": -float(
                    main_metrics["Minimum anomaly (Sv)"]
                ),
                "Maximum weakening (%)": (
                    main_metrics["Maximum weakening (%)"]
                ),
                "Lag after CO2 peak (yr)": (
                    main_metrics["Lag after CO2 peak (yr)"]
                ),
                "Final anomaly (%)": final_percent(
                    main_metrics
                ),
            },
            {
                "Definition": diagnostic_definition,
                "Requested latitude (degN)": (
                    DIAGNOSTIC_AMOC_LATITUDE
                ),
                "Actual latitude (degN)": (
                    diagnostic["metadata"].get(
                        "actual_latitude_degN"
                    )
                ),
                "piControl AMOC (Sv)": diagnostic["pi_mean"],
                "Maximum weakening (Sv)": -float(
                    diagnostic_metrics["Minimum anomaly (Sv)"]
                ),
                "Maximum weakening (%)": (
                    diagnostic_metrics["Maximum weakening (%)"]
                ),
                "Lag after CO2 peak (yr)": (
                    diagnostic_metrics["Lag after CO2 peak (yr)"]
                ),
                "Final anomaly (%)": final_percent(
                    diagnostic_metrics
                ),
            },
        ]
    )

    main_tag = str(MAIN_AMOC_LATITUDE).replace(".", "p")
    diagnostic_tag = str(
        DIAGNOSTIC_AMOC_LATITUDE
    ).replace(".", "p")

    summary_path = (
        OUTPUT_DIR
        / (
            f"{OUTPUT_PREFIX}_CESM2_"
            f"{main_tag}N_vs_{diagnostic_tag}N_summary.csv"
        )
    )
    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    annual_rows: list[dict[str, Any]] = []
    for definition, result in [
        (main_definition, main_result),
        (diagnostic_definition, diagnostic),
    ]:
        anomaly = np.asarray(
            result["anomaly"][:common_length],
            dtype=float,
        )
        smooth = running_mean(anomaly)

        for index in range(common_length):
            annual_rows.append(
                {
                    "Definition": definition,
                    "Simulation year": index + 1,
                    "Calendar year": CALENDAR_START_YEAR + index,
                    "AMOC anomaly (Sv)": anomaly[index],
                    "AMOC 11-yr smoothed anomaly (Sv)": smooth[index],
                }
            )

    annual_path = (
        OUTPUT_DIR
        / (
            f"{OUTPUT_PREFIX}_CESM2_"
            f"{main_tag}N_vs_{diagnostic_tag}N_annual.csv"
        )
    )
    pd.DataFrame(annual_rows).to_csv(
        annual_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"CESM2纬度敏感性摘要：{summary_path}")
    print(summary.to_string(index=False))

    make_cesm2_latitude_diagnostic(
        main_result,
        diagnostic,
    )


# =============================================================================
# 15. 主程序
# =============================================================================


def main() -> None:
    print("=" * 82)
    print("CMIP6 1pctCO2 / 1pctCO2-cdr：AMOC 单变量分析")
    print("=" * 82)
    print(f"Python：{sys.executable}")
    print(f"NetCDF engine：{NETCDF_ENGINE}")
    print(f"1pctCO2：{DIR_1PCTCO2}")
    print(f"CDR：{DIR_CDR}")
    print(f"piControl：{DIR_PI}")
    print("CNRM地理网格搜索目录：")
    for directory in CNRM_GRID_SEARCH_DIRS:
        print(f"  - {directory} (存在={directory.exists()})")
    print(f"降噪方法：{SMOOTH_WINDOW}年中心滑动平均（原始年序列仍写入CSV）")
    print(
        "AMOC主定义：Atlantic basin, fixed "
        f"{MAIN_AMOC_LATITUDE:.1f}°N (nearest model grid), "
        f"maximum over all valid depths >= {AMOC_MIN_DEPTH_M:.0f} m"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 检查CNRM持久缓存。完整正确映射在40°N处应为j=207；
    # 仅删除早期错误的j=39缓存，正确缓存保留给26.5°N和40°N共同使用。
    cnrm_cache = (
        CNRM_GRID_CACHE_DIR
        / "CNRM-ESM2-1_j294_latitude_mapping.npz"
    )
    if cnrm_cache.exists():
        remove_cache = False
        cache_reason = ""

        try:
            payload = np.load(
                cnrm_cache,
                allow_pickle=False,
            )
            cached_latitude = np.asarray(
                payload["row_latitude"],
                dtype=float,
            ).reshape(-1)

            if (
                cached_latitude.size != 294
                or np.isfinite(cached_latitude).sum() < 20
            ):
                remove_cache = True
                cache_reason = "纬度数组长度或有效值数量不正确"
            else:
                nearest_40 = int(
                    np.nanargmin(
                        np.abs(cached_latitude - 40.0)
                    )
                )
                actual_40 = float(
                    cached_latitude[nearest_40]
                )

                if (
                    nearest_40 != 207
                    or abs(actual_40 - 39.970472) > 0.25
                ):
                    remove_cache = True
                    cache_reason = (
                        f"40°N最近行为j={nearest_40}、"
                        f"纬度={actual_40:.3f}°N，"
                        "与已验证的j=207不一致"
                    )

        except Exception as error:
            remove_cache = True
            cache_reason = f"缓存无法读取：{error}"

        if remove_cache:
            try:
                cnrm_cache.unlink()
                print(
                    f"已删除旧CNRM错误缓存：{cnrm_cache}\n"
                    f"原因：{cache_reason}"
                )
            except Exception as error:
                print(
                    "旧CNRM错误缓存删除失败；程序会在读取时忽略："
                    f"{error}"
                )
        else:
            print(f"CNRM网格缓存检查通过：{cnrm_cache}")

    model_results: dict[str, dict[str, Any]] = {}
    failed_models: dict[str, str] = {}

    for model, configuration in MODEL_INFO.items():
        try:
            result = process_model(
                model,
                configuration,
                target_latitude=MAIN_AMOC_LATITUDE,
                strict_qa=True,
                verbose=True,
            )
            model_results[model] = result

            if model == "CNRM-ESM2-1":
                try:
                    make_cnrm_initial_anomaly_diagnostic(
                        result
                    )
                except Exception as diagnostic_error:
                    print(
                        "CNRM初始异常诊断输出失败，但不影响主分析："
                        f"{diagnostic_error}"
                    )

        except Exception as error:
            failed_models[model] = str(error)
            print(f"\n跳过模式 {model}：{error}")

    if len(model_results) < 2:
        raise RuntimeError(
            "\n成功模式少于2个，无法进行多模式 AMOC 分析。\n"
            "请检查三个 AMOC 文件夹和变量结构。"
        )

    common_length = min(result["length"] for result in model_results.values())
    models = [model for model in MODEL_INFO if model in model_results]

    print("\n" + "-" * 82)
    print(f"成功模式数：{len(models)}")
    print("成功模式：" + ", ".join(models))
    print(f"共同分析长度：{common_length} 年")

    if failed_models:
        print("\n跳过的模式：")
        for model, reason in failed_models.items():
            print(f"  {model}：{reason}")

    years = np.arange(1, common_length + 1)
    co2 = create_co2_series(common_length)
    co2_ratio = co2 / CO2_BASELINE

    metrics_rows: list[dict[str, Any]] = []
    annual_rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []

    for model in models:
        result = model_results[model]
        anomaly = result["anomaly"][:common_length]

        metrics_rows.append(
            calculate_metrics(
                model,
                anomaly,
                float(result["pi_mean"]),
                years,
                co2_ratio,
            )
        )

        percent = anomaly / float(result["pi_mean"]) * 100.0
        smooth = running_mean(anomaly)
        percent_smooth = running_mean(percent)

        for index, year in enumerate(years):
            annual_rows.append(
                {
                    "Model": model,
                    "Simulation year": int(year),
                    "Atmospheric CO2 (ppm)": float(co2[index]),
                    "Atmospheric CO2 / CO2_PI": float(co2_ratio[index]),
                    "Forced AMOC strength (Sv)": float(
                        result["combined_strength"][index]
                    ),
                    "Primary piControl reference (Sv)": float(
                        result["reference"][index]
                    ),
                    "piControl mean reference (Sv)": float(
                        result["reference_mean"][index]
                    ),
                    "piControl linear reference (Sv)": float(
                        result["reference_linear"][index]
                    ),
                    "AMOC anomaly (Sv)": float(anomaly[index]),
                    "AMOC smoothed anomaly (Sv)": float(smooth[index]),
                    "AMOC anomaly (%)": float(percent[index]),
                    "AMOC smoothed anomaly (%)": float(percent_smooth[index]),
                    "AMOC anomaly using piControl mean (Sv)": float(
                        result["anomaly_mean_reference"][index]
                    ),
                    "AMOC anomaly using branch-linear piControl (Sv)": float(
                        result["anomaly_linear_reference"][index]
                    ),
                }
            )

        meta = result["metadata"]
        metadata_rows.append(
            {
                "Model": model,
                "Variable": meta.get("variable"),
                "Original units": meta.get("original_units"),
                "Original dims": meta.get("original_dims"),
                "Original shape": meta.get("original_shape"),
                "AMOC definition": meta.get("definition"),
                "Basin selection": meta.get("basin_selection"),
                "Basin index": meta.get("basin_index"),
                "MOC component treatment": meta.get("MOC component treatment"),
                "Annualization method": meta.get("annualization method"),
                "Source ID": meta.get("source_id"),
                "piControl variant label": meta.get("variant_label"),
                "1pctCO2 variant label": result["up_metadata"].get("variant_label"),
                "1pctCO2-cdr variant label": result["cdr_metadata"].get("variant_label"),
                "CDR parent experiment": result["cdr_metadata"].get(
                    "parent_experiment_id"
                ),
                "CDR parent variant": result["cdr_metadata"].get(
                    "parent_variant_label"
                ),
                "CDR branch time in parent": result["cdr_metadata"].get(
                    "branch_time_in_parent"
                ),
                "CDR variant info": result["cdr_metadata"].get("variant_info"),
                "Requested latitude (degN)": meta.get("requested_latitude_degN"),
                "Actual latitude (degN)": meta.get("actual_latitude_degN"),
                "Latitude offset (deg)": meta.get("latitude_offset_deg"),
                "Latitude index": meta.get("latitude_index"),
                "Latitude dimension": meta.get("latitude_dimension"),
                "Latitude mapping method": meta.get("latitude_mapping_method"),
                "Latitude mapping source": meta.get("latitude_mapping_source"),
                "Depth subset applied": meta.get("depth_subset"),
                "piControl maximum depth (m)": meta.get("piControl_maximum_depth_m"),
                "Extremum mode": meta.get("extremum_mode"),
                "Streamfunction orientation mode": meta.get(
                    "streamfunction_orientation_mode"
                ),
                "Candidate max mean (Sv)": meta.get(
                    "candidate_max_mean_sv"
                ),
                "Candidate -minimum mean (Sv)": meta.get(
                    "candidate_negative_min_mean_sv"
                ),
                "Unit conversion": meta.get("unit_conversion"),
                "Sign flipped": result["sign_flipped"],
                "Variant/stage pairing": result.get("variant_pairing"),
                "1pctCO2 selected file count": result.get("up_file_count"),
                "1pctCO2-cdr selected file count": result.get("cdr_file_count"),
                "piControl selected file count": result.get("pi_file_count"),
                "Primary piControl correction": PI_CORRECTION,
                "piControl climatology (Sv)": result["pi_mean"],
                "piControl trend (Sv yr-1)": result["pi_slope"],
                "piControl reference start index": result.get("pi_reference_start_index"),
                "Initial 10-yr anomaly mean (Sv)": result.get("initial_10yr_anomaly_mean"),
                "1pctCO2-to-CDR transition jump (Sv)": result.get("transition_jump_sv"),
                "Analysis length (yr)": common_length,
            }
        )

    metrics_dataframe = pd.DataFrame(metrics_rows)
    annual_dataframe = pd.DataFrame(annual_rows)
    metadata_dataframe = pd.DataFrame(metadata_rows)
    qa_dataframe = build_qa_report(model_results, metrics_dataframe)

    metrics_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_metrics.csv"
    annual_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_annual_series.csv"
    metadata_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_processing_metadata.csv"
    qa_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_QA_report.csv"
    failed_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_failed_models.csv"

    metrics_dataframe.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    annual_dataframe.to_csv(annual_path, index=False, encoding="utf-8-sig")
    metadata_dataframe.to_csv(metadata_path, index=False, encoding="utf-8-sig")
    qa_dataframe.to_csv(qa_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {"Model": model, "Failure reason": reason}
            for model, reason in failed_models.items()
        ],
        columns=["Model", "Failure reason"],
    ).to_csv(failed_path, index=False, encoding="utf-8-sig")

    make_figure_1(model_results, common_length)
    if MAKE_HYSTERESIS_FIGURE:
        make_figure_2(model_results, common_length)
    if MAKE_METRICS_FIGURE:
        make_figure_3(metrics_dataframe)

    if "CESM2" in model_results:
        run_cesm2_latitude_diagnostic(model_results["CESM2"])

    print("\n" + "=" * 82)
    print("AMOC分析完成。")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"指标：{metrics_path}")
    print(f"年度序列：{annual_path}")
    print(f"处理元数据：{metadata_path}")
    print(f"QA：{qa_path}")
    print(f"未纳入模式诊断：{failed_path}")
    print("\n主要指标摘要：")

    summary_columns = [
        "Model",
        "piControl climatology (Sv)",
        "Maximum weakening (%)",
        "Lag after CO2 peak (yr)",
        "End-of-run recovery (%)",
    ]
    print(metrics_dataframe[summary_columns].to_string(index=False))
    print("=" * 82)


if __name__ == "__main__":
    main()
