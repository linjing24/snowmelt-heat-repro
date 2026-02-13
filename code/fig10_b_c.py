import os, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

# ================= 1. 配置路径 (保持不变) =================
melt_folder = r"D:\BNU-Artical2\数据处理过程\35-1. 融雪极值-划分3大子区"
heat_folder = r"D:\BNU-Artical2\数据处理过程\36-1.高温极值-划分3大子区"
output_folder = r"D:\BNU-Artical2\数据处理过程\非线性\子区域_分时段-CleanStyle"
os.makedirs(output_folder, exist_ok=True)

# ================= 2. 定义分析时段 =================
periods = {
    "1980_1999": (1980, 1999),
    "2000_2019": (2000, 2019),
}

# ================= 3. 数据读取与合并 (保持不变) =================
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
        dfh = pd.read_csv(hf_cand[0], usecols=["year", "hot_days"])
        df = pd.merge(dfm, dfh, on="year", how="inner")
        all_dfs.append(df)

if not all_dfs: raise RuntimeError("未找到任何数据")
full_data = pd.concat(all_dfs, ignore_index=True)
print(f"数据读取完成，共 {len(full_data)} 条记录。")

# ================= 4. 循环处理每个时段 (核心修改) =================
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
    y = data["hot_days"].values

    tree = DecisionTreeRegressor(
        max_depth=1,
        min_samples_leaf=max(5, int(0.10 * len(X))),
        random_state=42
    )
    tree.fit(X, y)
    threshold = tree.tree_.threshold[0]
    if threshold <= 0: threshold = None
    print(f"  - 检测到的阈值: {threshold}")

    # --- C. 绘图：模仿您提供的折线图风格 ---
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # 1. 绘制折线图 (带点 + 置信区间)
    # 这里的关键是把连续的 X 轴画成折线，Seaborn 会自动处理重复的 X 值
    # 如果 X 值太稀疏，我们可以先做一个细微的分箱，或者直接画（如果数据够密）

    # 为了复刻那张图的密集波动感，我们这里不分箱，直接画原始数据的聚合线
    # 或者用极细的分箱 (比如 0.1mm)
    bin_step = 0.1
    bins = np.arange(0, data['melt_peak'].max() + bin_step, bin_step)
    data['bin_fine'] = pd.cut(data['melt_peak'], bins)

    # 计算每个细分箱的均值，作为折线点 (这样线会更平滑一点，但保留波动)
    # 如果想完全复刻那种"毛刺感"，直接用原始数据画也可以，但 sns.lineplot 默认会聚合

    sns.lineplot(
        data=data,
        x='melt_peak',
        y='hot_days',
        ax=ax,
        color='#2F5597',  # 深蓝色
        linewidth=1.5,
        marker='o',  # 圆点
        markersize=5,
        markeredgecolor='black',  # 点的黑边
        markeredgewidth=0.5,
        errorbar=('ci', 95),  # 浅蓝色阴影
        label='Mean Trend (95% CI)'
    )

    # 2. 绘制阈值线 (红色竖虚线)
    if threshold is not None:
        ax.axvline(x=threshold, color='#C00000', linestyle='--', linewidth=5, zorder=10)

        # 标注文字 (红色)
        y_max = ax.get_ylim()[1]
        ax.text(threshold - 1, y_max * 0.65,
                f'{threshold:.2f} mm/d',
                color='#C00000', fontsize=22, fontweight='bold')

    # 3. 坐标轴与美化
    ax.set_xlim(0, 10)  # 截断在 10mm (参考那张图的范围)
    ax.set_ylim(-2, None)

    ax.set_xlabel(r"$M_{peak}$ (mm d$^{-1}$)", fontsize=26, weight='bold')
    ax.set_ylabel(r"$N_{hot}$ (days)", fontsize=26, weight='bold')

    period_str = pname.replace('_', '–')
    ax.set_title(f"Nonlinear Response ({period_str}) - Excluding Zero-Melt", fontsize=24, pad=15)

    ax.tick_params(axis="both", labelsize=26)
    ax.legend(frameon=True, fontsize=20, loc='upper right')

    # 添加网格
    ax.grid(True, linestyle='--', linewidth=1.5)

    plt.tight_layout()

    # 保存图片
    save_path = os.path.join(output_folder, f"Clean_LinePlot_{pname}.png")
    plt.savefig(save_path)
    print(f"✅ 图片已保存: {save_path}")
    plt.close()

print("\n🎉 所有时段绘图完成！")
