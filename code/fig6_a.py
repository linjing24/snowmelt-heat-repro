import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ================= 1. 配置参数 =================
ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（code/ 的上一层）

INPUT_DIR = str(ROOT / "data" / "41.SDI指数计算结果")
OUTPUT_PNG = str(ROOT / "results" / "Figure_5yr_Boxplot_SDI_DecadeBG_Final.png")
os.makedirs(os.path.dirname(OUTPUT_PNG), exist_ok=True)

BOX_WIDTH = 0.25
START_YEAR = 1980
END_YEAR = 2019

# ✅ 箱体颜色：每10年一个颜色（两箱同色）——完全沿用你 PSI 的设置
decade_box_colors = {
    1980: "#8FAADC",
    1990: "#FFD966",
    2000: "#A9D18E",
    2010: "#DC5A79",
}

# ✅ 背景颜色：非常浅（与箱体区分明显）——完全沿用你 PSI 的设置
decade_bg_colors = {
    1980: "#F4F7FF",
    1990: "#FFFDF0",
    2000: "#F4FFF5",
    2010: "#FFF4F0",
}
BG_ALPHA = 1

POINT_SIZE = 14
JITTER_SCALE = 0.03

# ================= 2. 读取数据 =================
all_dfs = []
files = glob.glob(os.path.join(INPUT_DIR, "*"))

for fn in files:
    try:
        if fn.lower().endswith(".csv"):
            df = pd.read_csv(fn)
        elif fn.lower().endswith((".xls", ".xlsx")):
            df = pd.read_excel(fn)
        else:
            continue

        # 智能识别列名
        col_year = next((c for c in df.columns if 'year' in str(c).lower()), df.columns[0])
        col_sdi = next((c for c in df.columns if 'sdi' in str(c).lower()), None)
        if col_sdi is None and df.shape[1] > 5:
            col_sdi = df.columns[5]

        if col_sdi:
            df = df.rename(columns={col_year: "year", col_sdi: "SDI"})
            clean_df = df[["year", "SDI"]].dropna()
            clean_df["year"] = pd.to_numeric(clean_df["year"], errors='coerce')
            clean_df["SDI"] = pd.to_numeric(clean_df["SDI"], errors='coerce')
            all_dfs.append(clean_df.dropna())
    except:
        pass

if not all_dfs:
    print("❌ 没有数据")
    raise SystemExit

big_df = pd.concat(all_dfs, ignore_index=True)
big_df = big_df[(big_df.year >= START_YEAR) & (big_df.year <= END_YEAR)]

# 如果你想保留原 SDI 的分位裁剪逻辑，可以打开下面两行（与 PSI 的 -200~2000 不同）
low, high = big_df["SDI"].quantile([0.01, 0.98])
big_df = big_df[(big_df["SDI"] >= low) & (big_df["SDI"] <= high)]

# ================= 3. 生成每5年区间 =================
periods = []
for s in range(START_YEAR, END_YEAR + 1, 5):
    e = min(s + 4, END_YEAR)
    periods.append((s, e))

labels = [f"{s}-{e}" for s, e in periods]

plot_data = []
group_decades = []
all_values = []

for (s, e) in periods:
    subset = big_df[(big_df["year"] >= s) & (big_df["year"] <= e)]["SDI"].values
    plot_data.append(subset)
    all_values.extend(subset)
    decade = (s // 10) * 10
    group_decades.append(decade)

# ================= 4. 绘图 =================
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 20})
fig, ax = plt.subplots(figsize=(13, 6))

positions = np.arange(len(plot_data))

# --- 4.1 背景（每十年一块，非常浅色） ---
unique_decades = sorted(set(group_decades))
for dec in unique_decades:
    idxs = [i for i, d in enumerate(group_decades) if d == dec]
    if not idxs:
        continue
    left = min(idxs) - 0.5
    right = max(idxs) + 0.5
    bgc = decade_bg_colors.get(dec, "#F7F7F7")
    ax.axvspan(left, right, facecolor=bgc, edgecolor="none", alpha=BG_ALPHA, zorder=-5)

# --- 4.2 箱线图（箱体不透明，按十年同色） ---
bp = ax.boxplot(
    plot_data,
    positions=positions,
    widths=BOX_WIDTH,
    patch_artist=True,
    showfliers=False,
    medianprops={'color': 'red', 'linewidth': 0.6},
    whiskerprops={'color': 'black', 'linewidth': 0.6, 'linestyle': '-'},
    capprops={'color': 'black', 'linewidth': 0.6},
    boxprops={'linewidth': 0.8, 'edgecolor': 'black'}
)

for i in range(len(plot_data)):
    dec = group_decades[i]
    c = decade_box_colors.get(dec, "#CCCCCC")
    bp['boxes'][i].set_facecolor(c)
    bp['boxes'][i].set_alpha(1.0)

# --- 4.2.1 帽子更短（缩短帽子横向长度） ---
CAP_SHRINK = 0.8
for cap in bp['caps']:
    x = cap.get_xdata()
    x_center = np.mean(x)
    half = (x[1] - x[0]) / 2 * CAP_SHRINK
    cap.set_xdata([x_center - half, x_center + half])

# --- 4.3 十年之间分隔虚线（两箱一组） ---
for i in range(1, len(plot_data)):
    if group_decades[i] != group_decades[i - 1]:
        ax.axvline(i - 0.5, color="#444444", linestyle="--", linewidth=1, zorder=1)

# --- 4.4 只画“须线外(outliers)”散点：白心深灰边 ---
rng = np.random.default_rng(42)

for i, data_points in enumerate(plot_data):
    if len(data_points) == 0:
        continue

    q1 = np.percentile(data_points, 25)
    q3 = np.percentile(data_points, 75)
    iqr = q3 - q1
    low_w = q1 - 2.5 * iqr
    high_w = q3 + 2.5 * iqr

    out_mask = (data_points < low_w) | (data_points > high_w)
    out_vals = data_points[out_mask]
    if len(out_vals) == 0:
        continue

    x_jitter = rng.normal(loc=positions[i], scale=JITTER_SCALE, size=len(out_vals))
    ax.scatter(
        x_jitter,
        out_vals,
        facecolors="white",
        edgecolors="#333333",
        linewidths=0.5,
        s=POINT_SIZE,
        zorder=6
    )

# ================= 5. Y轴自动适应 =================
y_all = np.array(all_values)
if len(y_all) > 0:
    y_lower = np.nanmin(y_all)
    y_upper = np.nanpercentile(y_all, 99.5)
    margin = (y_upper - y_lower) * 0.1
    ax.set_ylim(y_lower - margin, y_upper + margin)

# ================= 6. 美化 =================
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=35, ha='right')

ax.set_ylabel("SDI", fontsize=24, fontweight='bold')
ax.set_xlabel("Period", fontsize=24, fontweight='bold')

# ✅ 不画横向虚线网格线（按你要求删除）
ax.yaxis.grid(False)

ax.set_axisbelow(True)

for spine in ax.spines.values():
    spine.set_linewidth(1)
    spine.set_color('black')

plt.tight_layout()
fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ SDI绘图完成（已完全同步 PSI 风格参数）。\n保存至: {OUTPUT_PNG}")

