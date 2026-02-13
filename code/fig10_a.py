import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from scipy.interpolate import make_interp_spline

# ================= 1. 配置路径 =================
melt_folder = r"D:\BNU-Artical2\数据处理过程\35-1. 融雪极值-划分3大子区"
heat_folder = r"D:\BNU-Artical2\数据处理过程\36-1.高温极值-划分3大子区"
output_folder = r"D:\BNU-Artical2\数据处理过程\非线性\子区域_全时段-相关拟合-V4_Fixed"
os.makedirs(output_folder, exist_ok=True)

# ================= 2. 数据读取与合并 =================
print("正在读取数据...")
all_dfs = []
if not os.path.exists(melt_folder):
    print("❌ 错误：融雪文件夹不存在")
    exit()

regions = [d for d in os.listdir(melt_folder) if os.path.isdir(os.path.join(melt_folder, d))]
for region in regions:
    mdir = os.path.join(melt_folder, region)
    hdir = os.path.join(heat_folder, region)

    if not os.path.exists(hdir): continue

    for mf in glob.glob(os.path.join(mdir, "*.csv")):
        try:
            sid = os.path.basename(mf).split("_")[2]
            dfm = pd.read_csv(mf, usecols=["year", "melt_peak"])
            hf_cand = glob.glob(os.path.join(hdir, f"*站点_{sid}_*heat_extremes*.csv"))
            if not hf_cand: continue
            dfh = pd.read_csv(hf_cand[0], usecols=["year", "hot_days"])
            df = pd.merge(dfm, dfh, on="year", how="inner")
            all_dfs.append(df)
        except Exception as e:
            continue

if not all_dfs:
    print("❌ 未找到任何匹配的数据。")
    exit()

data = pd.concat(all_dfs, ignore_index=True)
data = data.dropna()
print(f"数据读取完成，共 {len(data)} 条有效记录。")


# ================= 3. 关键修改：剔除 0 值 =================
# 只保留有融雪发生的年份，这样分析"融雪调节"才更有物理意义
data_valid = data[data['melt_peak'] > 0].copy()
print(f"剔除 0 值后，剩余有效数据: {len(data_valid)} 条")

# ================= 4. 重新计算阈值 (基于有效数据) =================
X = data_valid["melt_peak"].values.reshape(-1, 1)
y = data_valid["hot_days"].values

tree = DecisionTreeRegressor(
    max_depth=1,
    min_samples_leaf=max(10, int(0.10 * len(X))),
    random_state=42
)
tree.fit(X, y)
threshold = tree.tree_.threshold[0]
print(f"基于有效数据的新阈值: {threshold}")

# ================= 5. 绘图：纯净版分箱折线图 =================
plt.rcParams["font.family"] = "Times New Roman"
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# --- A. 不画散点，只画分箱统计线 ---
# 分箱步长设为 0.5 或 1，看数据量决定
bin_step = 0.5
bins = np.arange(0, 15 + bin_step, bin_step)
data_valid['bin'] = pd.cut(data_valid['melt_peak'], bins)

# 画带置信区间的折线图
sns.lineplot(
    data=data_valid,
    x='melt_peak',
    y='hot_days',
    ax=ax,
    color='#2F5597',
    linewidth=1.5,
    marker='o',       # 加点，显示每个箱子的位置
    markersize=5,
    markeredgecolor='black',  # 【修改这里】标记点边缘颜色
    markeredgewidth=0.5,  # 【修改这里】标记点边缘粗细
    errorbar=('ci', 95), # 显示95%置信区间
    label='Mean Trend (95% CI)'
)

# --- B. 阈值线 ---
if threshold > 0:
    ax.axvline(x=threshold, color='#D0021B', linestyle='--', linewidth=5, zorder=10)
    # 标注文本
    ax.text(threshold + 0.2, ax.get_ylim()[1]*0.9,
            f'{threshold:.2f} mm/d',
            color='#D0021B', fontsize=22, fontweight='bold')

# --- C. 美化 ---
ax.set_xlim(0, 10) # 聚焦在 0-15 这个核心区间
ax.set_ylim(-3, None) # Y轴从0开始

ax.set_xlabel(r"$M_{peak}$ (mm d$^{-1}$)", fontsize=26, weight='bold')
ax.set_ylabel(r"$N_{hot}$ (days)", fontsize=26, weight='bold')
ax.set_title("Nonlinear Response (Excluding Zero-Melt)", fontsize=24, pad=15)

ax.tick_params(axis="both", labelsize=26)
ax.legend(frameon=True, fontsize=20)
ax.grid(True, linestyle='--', linewidth=1.5) # 加一点淡网格，增加专业感

plt.tight_layout()

# 保存
save_path = os.path.join(output_folder, "clean_line_plot_no_zero-V2.png")
plt.savefig(save_path)
plt.show()
