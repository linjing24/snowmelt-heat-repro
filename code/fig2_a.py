# -*- coding: utf-8 -*-
import os
import re
import math
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

try:
    from scipy.stats import kendalltau
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


# =========================
# ✅ 你只需要改这里的路径
# =========================
ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（不要改）

FOLDER = ROOT / "data" / "1980-2019气象要素-站点-年尺度（7.9）"   # <<< 需要你改：指向你要读取的“包含很多站点文件”的文件夹
OUTPUT_DIR = ROOT / "results" / "fig_3_1_fourvars"     # <<< 需要你改：输出目录（建议放仓库内，复现友好）
AGG_METHOD = "mean"  # <<< 可选：'mean' 或 'sum'（水文变量遵从该设置，气温始终 mean）


# =========================
# 字体（可选）
# =========================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({"font.size": 16})
plt.rcParams.update({'axes.titlesize': 14})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'


# =========================
# 列名别名映射（你可按需要继续加）
# =========================
ALIASES = {
    "year": ["year", "yr", "年份", "年"],
    "total_precipitation": ["total_precipitation", "precipitation", "total_prcp", "prcp_total", "precip", "ppt", "降水", "总降水"],
    "total_snow": ["total_snow", "snowfall", "snow", "swe", "snow_water", "snow_depth", "积雪", "降雪", "雪深"],
    "total_rain": ["total_rain", "rainfall", "rain", "降雨"],
    "avg_temperature": ["avg_temperature", "temperature", "tmean", "年均温", "年平均气温"],
}


# =========================
# 读取 + 预处理工具函数
# =========================
def read_any(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df2.columns]

    mapping = {}
    for std, cands in ALIASES.items():
        if std in df2.columns:
            continue
        for col in df2.columns:
            if col in [c.lower() for c in cands]:
                mapping[col] = std
                break
    return df2.rename(columns=mapping)


def load_and_stack_all(folder: Path,
                       x_min: int = 1980,
                       x_max: int = 2019) -> pd.DataFrame:
    """
    读取 folder 目录下所有 csv/xlsx/xls（只读当前目录，不递归子目录），
    拼接成长表：station, year, total_precipitation, total_rain, total_snow, avg_temperature
    """
    if not folder.exists():
        raise FileNotFoundError(f"Input folder not found: {folder}")

    records: List[pd.DataFrame] = []

    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".csv", ".xlsx", ".xls"]:
            continue

        try:
            df = read_any(p)
            df = normalize_columns(df)

            need_cols = ["year", "total_precipitation", "total_rain", "total_snow", "avg_temperature"]
            for need in need_cols:
                if need not in df.columns:
                    raise ValueError(f"Missing column: {need}; found columns: {df.columns.tolist()}")

            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df.dropna(subset=["year"])
            df["year"] = df["year"].astype(int)

            df = df[(df["year"] >= x_min) & (df["year"] <= x_max)]
            df = df.sort_values("year").drop_duplicates(subset=["year"], keep="last")

            df["station"] = p.stem
            records.append(df[["station", "year",
                               "total_precipitation", "total_rain", "total_snow",
                               "avg_temperature"]])

        except Exception as e:
            print(f"[WARN] {p.name}: {e}")

    if not records:
        raise RuntimeError(
            "No valid station files found in the folder.\n"
            f"Folder: {folder}\n"
            "Tip: make sure files are directly in this folder (not only inside subfolders), "
            "and each file contains year + the 4 variables."
        )

    return pd.concat(records, ignore_index=True)


# =========================
# 趋势：Theil–Sen + MK
# =========================
def theil_sen_per_year(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    n = len(x)
    if n < 3:
        return np.nan

    slopes = []
    for i in range(n - 1):
        dx = x[i + 1:] - x[i]
        dy = y[i + 1:] - y[i]
        v = dx != 0
        if v.any():
            slopes.extend((dy[v] / dx[v]).tolist())

    return float(np.median(slopes)) if slopes else np.nan


def mk_tau_p(series: np.ndarray) -> Tuple[float, float]:
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n < 3:
        return (np.nan, np.nan)

    if _HAVE_SCIPY:
        x = np.arange(n)
        tau, p = kendalltau(x, s, nan_policy="omit")
        return float(tau), float(p)

    # 纯 Python 近似
    S = 0
    for i in range(n - 1):
        S += np.sign(s[i + 1:] - s[i]).sum()

    varS = n * (n - 1) * (2 * n + 5) / 18.0
    if varS == 0:
        return (np.nan, np.nan)

    z = (S - 1) / math.sqrt(varS) if S > 0 else (S + 1) / math.sqrt(varS) if S < 0 else 0.0
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    tau = (2 * S) / (n * (n - 1))
    return float(tau), float(p)


def theil_sen_line_with_mk(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if len(x) < 3:
        return (np.nan, np.nan, np.nan)

    slope = theil_sen_per_year(x, y)
    if not np.isfinite(slope):
        return (np.nan, np.nan, np.nan)

    intercept = float(np.median(y - slope * x))
    _, p = mk_tau_p(y)
    return (slope, intercept, p)


def piecewise_trend_theil_sen(x: np.ndarray, y: np.ndarray, breakpoint: int):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask1 = x < breakpoint
    mask2 = x >= breakpoint
    seg1 = theil_sen_line_with_mk(x[mask1], y[mask1])
    seg2 = theil_sen_line_with_mk(x[mask2], y[mask2])
    return seg1, seg2


def fit_full_period(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    return theil_sen_line_with_mk(x, y)


def star_annotation(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


# =========================
# 年序列统计（均值 & 95%CI）
# =========================
def read_folder_stats_vars(folder: Path,
                           columns=('total_precipitation', 'total_rain', 'total_snow', 'avg_temperature'),
                           x_min=1980, x_max=2019):
    all_df = load_and_stack_all(folder, x_min=x_min, x_max=x_max)
    years = np.arange(x_min, x_max + 1)

    mean_dict: Dict[str, np.ndarray] = {}
    ci_half_dict: Dict[str, np.ndarray] = {}

    for col in columns:
        means = []
        ci_halves = []
        is_hydro = col in ["total_precipitation", "total_rain", "total_snow"]

        for yr in years:
            vals = all_df.loc[all_df["year"] == yr, col].values.astype(float)
            vals = vals[np.isfinite(vals)]
            n = len(vals)
            if n == 0:
                means.append(np.nan)
                ci_halves.append(np.nan)
                continue

            std_x = float(np.nanstd(vals, ddof=1)) if n > 1 else 0.0

            # 水文变量：根据 AGG_METHOD 选择 mean 或 sum；温度永远 mean
            if (AGG_METHOD.lower() == "sum") and is_hydro:
                agg = float(np.nansum(vals))
                # 对 sum 的 CI 严格来说要考虑协方差，这里给近似：用 std 表示离散度，不除 sqrt(n)
                sem = std_x
            else:
                agg = float(np.nanmean(vals))
                sem = std_x / math.sqrt(n) if n > 1 else np.nan

            means.append(agg)
            ci_halves.append(1.96 * sem if np.isfinite(sem) else np.nan)

        mean_dict[col] = np.array(means, dtype=float)
        ci_half_dict[col] = np.array(ci_halves, dtype=float)

    return years, mean_dict, ci_half_dict


# =========================
# 绘图：四联图（95%CI + 分段趋势）
# =========================
def plot_four_vars_from_folder(folder: Path,
                               var_defs,
                               breakpoint=2000,
                               x_min=1980, x_max=2019,
                               y_mins=(0, 0, 0, -10),
                               output_dir: Path = Path("output")):

    output_dir.mkdir(parents=True, exist_ok=True)

    columns = [v[0] for v in var_defs]
    years, mean_dict, ci_half_dict = read_folder_stats_vars(folder, columns=columns, x_min=x_min, x_max=x_max)

    fig, axes = plt.subplots(4, 1, figsize=(6.5, 8), sharex=True)

    # 注释位置（你也可以自己调）
    text_pos = [
        {"x": 1985, "y_factor": 0.5, "x_offset": 8, "y_offset": 5},
        {"x": 1985, "y_factor": 0.5, "x_offset": 8, "y_offset": 20},
        {"x": 1985, "y_factor": 0.2, "x_offset": 8, "y_offset": 25},
        {"x": 1985, "y_factor": 0.1, "x_offset": 8, "y_offset": 4},
    ]

    for idx, (col, panel_title, color, y_unit, y_scale, y_step) in enumerate(var_defs):
        mean_series = np.asarray(mean_dict[col], dtype=float)
        ci_half = np.asarray(ci_half_dict[col], dtype=float)
        ax = axes[idx]

        y_min = y_mins[idx]
        finite_vals = mean_series[np.isfinite(mean_series)]
        if finite_vals.size == 0:
            y_max = y_min + 1.0
        else:
            top = np.nanmax(finite_vals)
            margin = 0.05 * (abs(top) if np.isfinite(top) else 1.0) + (1.0 if y_unit == '°C' else 0.0)
            y_max = max(top * y_scale + margin, y_min + 1.0)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

        ax.plot(years, mean_series, color=color, lw=1.8)
        ax.fill_between(years,
                        mean_series - ci_half,
                        mean_series + ci_half,
                        color=color, alpha=0.2, edgecolor='none')

        ax.axvspan(breakpoint, x_max, facecolor='lightgray', alpha=0.4, zorder=0)

        (s1, i1, p1), (s2, i2, p2) = piecewise_trend_theil_sen(years, mean_series, breakpoint)
        mask1 = years < breakpoint
        mask2 = years >= breakpoint
        if np.isfinite(s1) and np.isfinite(i1):
            ax.plot(years[mask1], s1 * years[mask1] + i1, '--', color=color, lw=1)
        if np.isfinite(s2) and np.isfinite(i2):
            ax.plot(years[mask2], s2 * years[mask2] + i2, '--', color=color, lw=1)

        ax.set_title(panel_title, loc='left', y=0.8)

        # 坐标轴样式
        if idx == 0:
            ax.xaxis.set_label_position('top')
            ax.xaxis.tick_top()
            ax.spines['top'].set_visible(True)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_position(('outward', 10))
            ax.spines['left'].set_position(('outward', 10))
            ax.set_ylabel(f'{panel_title} ({y_unit})')
        elif idx == 3:
            ax.spines['bottom'].set_visible(True)
            ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_position(('outward', 10))
            ax.spines['right'].set_position(('outward', 10))
            ax.set_xlabel('Year')
            ax.yaxis.set_label_position('right')
            ax.yaxis.tick_right()
            ax.set_ylabel(f'{panel_title} ({y_unit})')
        else:
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            if idx % 2 == 1:
                ax.spines['left'].set_visible(False)
                ax.spines['right'].set_visible(True)
                ax.spines['right'].set_position(('outward', 10))
                ax.yaxis.set_label_position('right')
                ax.yaxis.tick_right()
            else:
                ax.spines['left'].set_visible(True)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_position(('outward', 10))
                ax.yaxis.set_label_position('left')
                ax.yaxis.tick_left()
            ax.set_ylabel(f'{panel_title} ({y_unit})')
            ax.tick_params(axis='x', length=0)

        ax.yaxis.set_major_locator(MultipleLocator(y_step))
        ax.set_xticks([1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2019])
        ax.tick_params(axis='both', which='major', direction='in')

        s_full, _, p_full = fit_full_period(years, mean_series)
        star1, star2, star_full = star_annotation(p1), star_annotation(p2), star_annotation(p_full)

        tp = text_pos[idx]
        x0 = tp["x"] + tp["x_offset"]
        y0 = y_min + (y_max - y_min) * tp["y_factor"] + tp["y_offset"]

        def fmt(v):
            return "nan" if (not np.isfinite(v)) else f"{v:.3f}"

        text_str = (
            f"$s_1^{{{x_min}-{breakpoint-1}}}$ = {fmt(s1)}{star1}\n"
            f"$s_2^{{{breakpoint}-{x_max}}}$ = {fmt(s2)}{star2}\n"
            f"$s^{{{x_min}-{x_max}}}$ = {fmt(s_full)}{star_full}"
        )

        ax.text(
            x0, y0,
            text_str,
            color=color,
            fontsize=12,
            linespacing=1,
            fontfamily='Times New Roman',
            ha='left', va='bottom',
            fontweight='bold', fontstyle='italic'
        )

    plt.tight_layout(h_pad=0.3)
    plt.subplots_adjust(top=0.85, bottom=0.10, hspace=0.05)

    out_path = output_dir / "four_variables_from_folder.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Figure saved to: {out_path}")

    plt.show()


def main():
    # 四个面板定义：列名、标题、颜色、单位、y上限比例、y主刻度步长
    var_defs = [
        ('total_precipitation', 'Precipitation', 'orange', 'mm', 1.8, 300),
        ('total_rain',          'Rain',          'red',    'mm', 1.4, 200),
        ('total_snow',          'Snow',          'blue',   'mm', 1.5, 30),
        ('avg_temperature',     'Temperature',   'green',  '°C', 1.5, 2),
    ]

    # 你可以按图形实际范围调整 y_mins
    plot_four_vars_from_folder(
        folder=FOLDER,
        var_defs=var_defs,
        breakpoint=2000,
        x_min=1980,
        x_max=2019,
        y_mins=(100, 150, 5, 4),
        output_dir=OUTPUT_DIR
    )


if __name__ == "__main__":
    main()

