import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator
import re, math
from typing import Tuple
try:
    from scipy.stats import kendalltau
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

# ——— 全局字体设置 ———
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({"font.size": 16})
plt.rcParams.update({'axes.titlesize': 14})
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm']      = 'Times New Roman'
plt.rcParams['mathtext.it']      = 'Times New Roman:italic'
plt.rcParams['mathtext.bf']      = 'Times New Roman:bold'

# 与第二段脚本保持一致的聚合方式设置
AGG_METHOD = "mean"  # 'mean' 或 'sum'，水文变量遵从该设置，气温始终用 mean

# 与第二段脚本一致的列名别名映射
ALIASES = {
    "year": ["year", "yr", "年份", "年"],
    "total_precipitation": ["total_precipitation", "precipitation", "total_prcp", "prcp_total", "precip", "ppt", "降水", "总降水"],
    "total_snow": ["total_snow", "snowfall", "snow", "swe", "snow_water", "snow_depth", "积雪", "降雪", "雪深"],
    "total_rain": ["total_rain", "rainfall", "rain", "降雨"],
    "avg_temperature": ["avg_temperature", "temperature", "tmean", "年均温", "年平均气温"],
}

# —— 与第二段相同：读任意格式（csv/xlsx/xls） ——
def read_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        return pd.read_csv(path)
    elif ext in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

# —— 与第二段相同：标准化列名并做别名映射 ——
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df2.columns]
    mapping = {}
    for std, cands in ALIASES.items():
        if std in df2.columns:
            mapping[std] = std
            continue
        for col in df2.columns:
            if col in cands:
                mapping[col] = std
                break
    return df2.rename(columns=mapping)

# —— 与第二段相同：读取整个文件夹为“长表” ——
def load_and_stack_all(folder: str) -> pd.DataFrame:
    """
    读取文件夹所有站点，统一列名，拼接为长表：station, year, total_precipitation, total_rain, total_snow, avg_temperature
    年份限制 1980–2019，与第二段脚本保持一致。
    """
    records = []
    for fn in sorted(os.listdir(folder)):
        p = os.path.join(folder, fn)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(fn)[1].lower()
        if ext not in [".csv", ".xlsx", ".xls"]:
            continue
        try:
            df = read_any(p)
            df = normalize_columns(df)
            for need in ["year", "total_precipitation", "total_rain", "total_snow", "avg_temperature"]:
                if need not in df.columns:
                    raise ValueError(f"Missing column: {need}; found: {df.columns.tolist()}")
            df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
            df = df.dropna(subset=["year"])
            df["year"] = df["year"].astype(int)
            df = df[(df["year"] >= 1980) & (df["year"] <= 2019)]
            df = df.sort_values("year").drop_duplicates(subset=["year"], keep="last")
            df["station"] = os.path.splitext(os.path.basename(p))[0]
            records.append(df[["station", "year",
                               "total_precipitation", "total_rain", "total_snow",
                               "avg_temperature"]])
        except Exception as e:
            print(f"[WARN] {fn}: {e}")
    if not records:
        raise RuntimeError("No valid station files found.")
    all_df = pd.concat(records, ignore_index=True)
    return all_df

# —— Theil–Sen 与 MK，按“每年”变化量 ——
def theil_sen_per_year(x: np.ndarray, series: np.ndarray) -> float:
    """
    Theil–Sen 斜率，单位为“每年”，与第二段相同方法，
    只是这里不再乘以 10（不化为每十年）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(series, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    n = len(x)
    if n < 3:
        return np.nan
    slopes = []
    for i in range(n - 1):
        dx = x[i+1:] - x[i]
        dy = y[i+1:] - y[i]
        v = dx != 0
        if v.any():
            slopes.extend((dy[v] / dx[v]).tolist())
    return float(np.median(slopes)) if slopes else np.nan  # per year

def mk_tau_p(series: np.ndarray) -> Tuple[float, float]:
    """
    与第二段相同的 Mann–Kendall τ 与 p 值实现。
    """
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
        S += np.sign(s[i+1:] - s[i]).sum()
    varS = n * (n - 1) * (2 * n + 5) / 18.0
    if varS == 0:
        return (np.nan, np.nan)
    z = (S - 1) / math.sqrt(varS) if S > 0 else (S + 1) / math.sqrt(varS) if S < 0 else 0.0
    from math import erf, sqrt
    p = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    tau = (2 * S) / (n * (n - 1))
    return float(tau), float(p)

def theil_sen_line_with_mk(x: np.ndarray, y: np.ndarray):
    """
    计算给定时间序列的 Theil–Sen 直线（斜率/截距）以及 Mann–Kendall p 值。
    斜率单位：每年。
    截距：median(y - slope * x) 的鲁棒估计。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    slope = theil_sen_per_year(x, y)
    if not np.isfinite(slope):
        return np.nan, np.nan, np.nan
    intercept = float(np.median(y - slope * x))
    _, p = mk_tau_p(y)
    return slope, intercept, p

# —— 分段趋势：与第二段同样“1980–1999 & 2000–2019”方式 ——
def piecewise_trend_theil_sen(x, y, breakpoint):
    """
    使用 Theil–Sen + MK 做分段趋势：
      段1：x < breakpoint  （例如 1980–1999）
      段2：x >= breakpoint （例如 2000–2019）
    返回：(s1, i1, p1), (s2, i2, p2)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask1 = x < breakpoint
    mask2 = x >= breakpoint
    s1, i1, p1 = theil_sen_line_with_mk(x[mask1], y[mask1])
    s2, i2, p2 = theil_sen_line_with_mk(x[mask2], y[mask2])
    return (s1, i1, p1), (s2, i2, p2)

def fit_full_period(x, y):
    """
    全时段 Theil–Sen + MK：
    返回 slope_per_year, intercept, pvalue
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope = theil_sen_per_year(x, y)
    if not np.isfinite(slope):
        return np.nan, np.nan, np.nan
    intercept = float(np.median(y - slope * x))
    _, p = mk_tau_p(y)
    return slope, intercept, p

def star_annotation(p):
    """根据 p 值返回星号"""
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''

# ===========================
# 读取“单个文件夹，多个文件”，四变量统计（均值 & 95%CI）
# —— 与第二段数据读取/聚合逻辑统一 ——
# ===========================
def read_folder_stats_vars(folder,
                           columns=('total_precipitation','total_rain','total_snow','avg_temperature'),
                           x_min=1980, x_max=2019):
    """
    使用与第二段相同的方式读取整个文件夹（支持 csv/xlsx/xls + 别名），
    得到所有站点的长表，再按年和变量计算：
      - 区域年平均/年总量（依据 AGG_METHOD）
      - 站点间 95% 置信区间（1.96 * std / sqrt(N)，若 AGG_METHOD 为 sum，则近似 std 形式）
    返回：
      years (ndarray),
      mean_dict {col: ndarray},
      ci_half_dict {col: ndarray}
    """
    all_df = load_and_stack_all(folder)
    all_df = all_df[(all_df["year"] >= x_min) & (all_df["year"] <= x_max)]
    years = np.arange(x_min, x_max + 1)

    mean_dict = {}
    ci_half_dict = {}

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

            if n == 1:
                std_x = 0.0
            else:
                std_x = float(np.nanstd(vals, ddof=1))

            if AGG_METHOD.lower() == "sum" and is_hydro:
                agg = float(np.nansum(vals))
                sem = std_x  # 近似
            else:
                agg = float(np.nanmean(vals))
                sem = std_x / math.sqrt(n) if n > 1 else np.nan

            means.append(agg)
            ci_halves.append(1.96 * sem if np.isfinite(sem) else np.nan)

        mean_dict[col] = np.array(means, dtype=float)
        ci_half_dict[col] = np.array(ci_halves, dtype=float)

    return years, mean_dict, ci_half_dict

# ===========================
# 绘制四联图（含95%CI & Theil–Sen 趋势）
# ===========================
def plot_four_vars_from_folder(folder,
                               var_defs,
                               breakpoint=2000,
                               x_min=1980, x_max=2019,
                               y_mins=(0, 0, 0, -10),
                               output_dir="output"):
    """
    var_defs: 列表（长度=4），每项为：
      (col_name, panel_title, color, y_unit, y_scale_factor, y_major_step)
    """
    os.makedirs(output_dir, exist_ok=True)

    columns = [v[0] for v in var_defs]
    years, mean_dict, ci_half_dict = read_folder_stats_vars(
        folder, columns=columns, x_min=x_min, x_max=x_max
    )

    fig, axes = plt.subplots(4, 1, figsize=(6.5, 8), sharex=True)

    # 注释位置（可根据数据微调）
    text_pos = [
        {"x":1985,"y_factor":0.5, "x_offset":8,  "y_offset":5},
        {"x":1985,"y_factor":0.5, "x_offset":8,  "y_offset":20},
        {"x":1985,"y_factor":0.2, "x_offset":8,  "y_offset":25},
        {"x":1985,"y_factor":0.1, "x_offset":8,  "y_offset":4},
    ]

    for idx, (col, panel_title, color, y_unit, y_scale, y_step) in enumerate(var_defs):
        mean_series = np.asarray(mean_dict[col], dtype=float)
        ci_half     = np.asarray(ci_half_dict[col], dtype=float)

        ax = axes[idx]

        # y 轴范围
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

        # 均值曲线 + 95% CI
        ax.plot(years, mean_series, color=color, lw=1.8)
        ax.fill_between(years,
                        mean_series - ci_half,
                        mean_series + ci_half,
                        color=color, alpha=0.2, edgecolor='none')

        # 灰色背景（breakpoint 之后）
        ax.axvspan(breakpoint, x_max, facecolor='lightgray', alpha=0.4, zorder=0)

        # 分段趋势线（Theil–Sen + MK，1980–1999 & 2000–2019）
        (s1,i1,p1), (s2,i2,p2) = piecewise_trend_theil_sen(years, mean_series, breakpoint)
        mask1 = years < breakpoint
        mask2 = years >= breakpoint
        ax.plot(years[mask1], s1*years[mask1]+i1, '--', color=color, lw=1)
        ax.plot(years[mask2], s2*years[mask2]+i2, '--', color=color, lw=1)

        # 面板标题
        ax.set_title(panel_title, loc='left', y=0.8)

        # 坐标轴造型
        if idx == 0:
            ax.xaxis.set_label_position('top'); ax.xaxis.tick_top()
            ax.spines['top'].set_visible(True);    ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(True);   ax.spines['right'].set_visible(False)
            ax.spines['top'].set_position(('outward',10))
            ax.spines['left'].set_position(('outward',10))
            ax.set_ylabel(f'{panel_title} ({y_unit})')
        elif idx == 3:
            ax.spines['bottom'].set_visible(True); ax.spines['top'].set_visible(False)
            ax.spines['left'].set_visible(False);  ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_position(('outward',10))
            ax.spines['right'].set_position(('outward',10))
            ax.set_xlabel('Year')
            ax.yaxis.set_label_position('right'); ax.yaxis.tick_right()
            ax.set_ylabel(f'{panel_title} ({y_unit})')
        else:
            ax.spines['top'].set_visible(False);  ax.spines['bottom'].set_visible(False)
            if idx%2==1:
                ax.spines['left'].set_visible(False); ax.spines['right'].set_visible(True)
                ax.spines['right'].set_position(('outward',10))
                ax.yaxis.set_label_position('right'); ax.yaxis.tick_right()
            else:
                ax.spines['left'].set_visible(True);  ax.spines['right'].set_visible(False)
                ax.spines['left'].set_position(('outward',10))
                ax.yaxis.set_label_position('left');  ax.yaxis.tick_left()
            ax.set_ylabel(f'{panel_title} ({y_unit})')
            ax.tick_params(axis='x', length=0)

        # 刻度
        ax.yaxis.set_major_locator(MultipleLocator(y_step))
        ax.set_xticks([1980,1985,1990,1995,2000,2005,2010,2015,2019])
        ax.tick_params(axis='both', which='major', direction='in')

        # 全时段趋势（Theil–Sen + MK）
        s_full, _, p_full = fit_full_period(years, mean_series)
        star1, star2, star_full = star_annotation(p1), star_annotation(p2), star_annotation(p_full)

        # 注释文本
        tp = text_pos[idx]
        x0 = tp["x"] + tp["x_offset"]
        y0 = y_min + (y_max - y_min)*tp["y_factor"] + tp["y_offset"]

        text_str = (
            f"$s_1^{{{x_min}-{breakpoint-1}}}$ = {s1:.3f}{star1}\n"
            f"$s_2^{{{breakpoint}-{x_max}}}$ = {s2:.3f}{star2}\n"
            f"$s^{{{x_min}-{x_max}}}$ = {s_full:.3f}{star_full}"
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
    out_path = os.path.join(output_dir, "four_variables_from_folder.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {out_path}")
    plt.show()

def main():
    # —— 单个“区域/类别”的文件夹（内部包含多个CSV/XLSX，每个文件至少含 year + 四变量列） ——
    ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "derived" / "threshold_input.csv")

    # 四个面板的定义：列名、标题、颜色、单位、上限比例、y主刻度步长
    var_defs = [
        ('total_precipitation',  'Precipitation', 'orange',  'mm', 1.8, 300),
        ('total_rain',           'Rain',          'red',     'mm', 1.4, 200),
        ('total_snow',           'Snow',          'blue',    'mm', 1.5, 30),
        ('avg_temperature',      'Temperature',   'green',   '°C', 1.5,   2),
    ]

    plot_four_vars_from_folder(
        folder=folder,
        var_defs=var_defs,
        breakpoint=2000,          # 分段：1980–1999 和 2000–2019
        x_min=1980, x_max=2019,
        y_mins=(100, 150, 5, 4),  # 可按数据调整
        output_dir=r"C:\Users\A\Desktop\3.1-整个北方"
    )

if __name__ == "__main__":
    main()
