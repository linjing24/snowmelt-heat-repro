import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from scipy.interpolate import make_interp_spline
from pathlib import Path

# ================= 1. 配置路径 (Tmax 分时段版) =================
ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（code/ 的上一层）

melt_folder = str(ROOT / "data" / "35-1. 融雪极值-划分3大子区")
heat_folder = str(ROOT / "data" / "36-1.高温极值-划分3大子区")
output_folder = str(ROOT / "results" / "非线性" / "子区域_分时段-相关拟合_txx-Advanced")
os.makedirs(output_folder, exist_ok=True)

# ================= 2. 定义分析时段 =================
periods = {
    "1980_1999": (1980, 1999),
    "2000_2019": (2000, 2019),
}

# ================= 3. 数据读取与合并 =================
print("正在读取数据...")
all_dfs = []
regions = [d for d in os.listdir(melt_folder) if os.path.isdir(os.path.join(melt_folder, d))]
for region in regions:
    mdir = os.path.join(melt_folder, region)
    hdir = os.path.join(heat_folder, region)
    for mf in glob.glob(os.path.join(mdir, "*.csv")):
        sid = os.path.basename(mf).split("_")[2]
        dfm = pd.read_csv(mf, usecols=["year", "melt_peak"])
        hf_cand = glob.glob(os.path.join(hdir, f"*站点_{sid}_*heat_extremes*.csv"))
        if not hf_cand: continue
        # 注意：读取 txx
        dfh = pd.read_csv(hf_cand[0], usecols=["year", "txx"])
        df = pd.merge(dfm, dfh, on="year", how="inner")
        all_dfs.append(df)

if not all_dfs: raise RuntimeError("未找到任何数据")
full_data = pd.concat(all_dfs, ignore_index=True)
print(f"数据读取完成，共 {len(full_data)} 条记录。")

# ================= 4. 循环处理每个时段 =================
plt.rcParams["font.family"] = "Times New Roman"

for pname, (start, end) in periods.items():
    print(f"\n正在处理时段: {pname} ({start}-{end})...")

    # --- A. 筛选数据 & 剔除 0 值 ---
    # 先按年份筛选
    data = full_data[(full_data["year"] >= start) & (full_data["year"] <= end)].copy()

    # 再剔除无融雪年份 (Mpeak = 0)
    data = data[data['melt_peak'] > 0]

    if len(data) < 20:
        print(f"⚠️ {pname} 时段有效样本不足 (<20)，跳过。")
        continue

    print(f"  - 有效样本数: {len(data)}")

    # --- B. 计算 CART 阈值 ---
    X = data["melt_peak"].values.reshape(-1, 1)
    y = data["txx"].values  # Y轴改为 txx

    tree = DecisionTreeRegressor(
        max_depth=1,
        min_samples_leaf=max(5, int(0.10 * len(X))),
        random_state=42
    )
    tree.fit(X, y)
    threshold = tree.tree_.threshold[0]
    if threshold <= 0: threshold = None
    print(f"  - 检测到的 Tmax 阈值: {threshold}")

    # --- C. 高级绘图 ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 1. 绘制折线图 (带点 + 置信区间)
    # 使用极细分箱或原始数据聚合
    bin_step = 0.1
    bins = np.arange(0, data['melt_peak'].max() + bin_step, bin_step)

    sns.lineplot(
        data=data,
        x='melt_peak',
        y='txx',  # Y轴是 txx
        ax=ax,
        color='#C00000',  # 深红色 (代表高温强度)
        linewidth=1.5,
        marker='o',
        markersize=5,
        markeredgecolor='black',
        markeredgewidth=0.5,
        errorbar=('ci', 95),  # 浅红色阴影
        label='Mean Trend (95% CI)'
    )

    # 2. 绘制阈值线 (黑色虚线)
    if threshold is not None:
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=5, zorder=10)

        # 标注文字
        y_range = data['txx'].max() - data['txx'].min()
        y_pos = data['txx'].max() - y_range * 0.2  # 放在顶部

        ax.text(threshold - 2, y_pos,
                f'{threshold:.2f} mm/d',
                color='black', fontsize=22, fontweight='bold')

    # 3. 坐标轴与美化
    ax.set_xlim(0, 12)  # 截断在 15mm
    # Tmax 的 Y 轴范围通常在 20-45 之间
    # ax.set_ylim(25, 45)

    ax.set_xlabel(r"$M_{peak}$ (mm d$^{-1}$)", fontsize=26, weight='bold')
    ax.set_ylabel(r"$T_{max}$ (°C)", fontsize=26, weight='bold')

    period_str = pname.replace('_', '–')
    ax.set_title(f"Nonlinear Response ($T_{{max}}$, {period_str}) - Excluding Zero-Melt", fontsize=24, pad=15)

    ax.tick_params(axis="both", labelsize=26)
    ax.legend(frameon=True, fontsize=20, loc='upper right')

    # 添加网格
    ax.grid(True, linestyle='--', linewidth=1.5)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(output_folder, f"Clean_LinePlot_Txx_{pname}.png")
    plt.savefig(save_path)
    print(f"✅ 图片已保存: {save_path}")
    plt.close()

print("\n🎉 所有时段（Txx）绘图完成！")

