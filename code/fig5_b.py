import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ========== 1. 配置区 ==========
# ✅ 按你 GitHub 仓库结构（见截图：snowmelt-heat-repro/data/...）做“相对路径”，只要仓库位置变了也能跑
ROOT = Path(__file__).resolve().parents[1]  # 项目根目录（code/ 的上一层）

files = {
    'Northern China': str(ROOT / "data" / "31. 冬季降雪-春季气温（SPI）" / "PSI.csv"),
    'Northeast':      str(ROOT / "data" / "31. 冬季降雪-春季气温（SPI）" / "PSI指数-东北数据.csv"),
    'North China':    str(ROOT / "data" / "31. 冬季降雪-春季气温（SPI）" / "PSI指数-华北数据.csv"),
    'Northwest':      str(ROOT / "data" / "31. 冬季降雪-春季气温（SPI）" / "PSI指数-西北数据.csv"),
}

# 自定义颜色
region_colors = {
    'Northern China': '#8FAADC',
    'Northeast': '#B0D597',
    'North China': '#FFE389',
    'Northwest': '#FFA3A3',
}
# 自定义标记形状 (拐点符号)
region_markers = {
    'Northern China': 'o',  # 圆点
    'Northeast': '^',  # 上三角
    'North China': 's',  # 正方形
    'Northwest': 'D',  # 菱形
}

out_png = r"D:\BNU-Artical2\图5-结果图\Psi_Combined_LineChart-V2.png"
os.makedirs(os.path.dirname(out_png), exist_ok=True)

# ========== 2. 读取并聚合 (保持智能列识别) ==========
agg = {}
print("正在读取 PSI 数据...")

for region, path in files.items():
    if not os.path.exists(path):
        print(f"❌ 跳过: 找不到文件 {path}")
        continue

    try:
        df = pd.read_csv(path, encoding='gbk')
    except Exception as e:
        print(f"❌ 读取错误 {region}: {e}")
        continue

    # --- 智能寻找列 ---
    col_year = None
    for c in df.columns:
        if 'year' in str(c).lower() or '年' in str(c):
            col_year = c
            break
    if col_year is None: col_year = df.columns[0]

    col_psi = None
    for c in df.columns:
        if 'psi' in str(c).lower():
            col_psi = c
            break

    if col_psi is None:
        if len(df.columns) > 6:
            col_psi = df.columns[6]
            print(f"⚠️ {region}: 未找到列名含PSI的列，尝试使用第7列 [{col_psi}]")
        else:
            print(f"❌ {region}: 列数不足且未找到PSI列，跳过。")
            continue

    # 提取数据
    tmp = df[[col_year, col_psi]].dropna()
    tmp.columns = ['year', 'PSI']
    tmp['year'] = pd.to_numeric(tmp['year'], errors='coerce')
    tmp['PSI'] = pd.to_numeric(tmp['PSI'], errors='coerce')
    tmp = tmp.dropna()
    tmp['year'] = tmp['year'].astype(int)

    # 聚合
    grp = tmp.groupby('year')['PSI']
    mean_vals = grp.mean()
    count = grp.count()
    std = grp.std(ddof=1)
    ci_vals = 1.96 * (std / np.sqrt(count))

    agg[region] = pd.DataFrame({
        'year': mean_vals.index,
        'mean': mean_vals.values,
        'ci': ci_vals.values
    })

if not agg:
    print("❌ 没有有效数据，程序终止。")
    exit()

# ========== 3. 绘图 (合并为单张图) ==========
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 26})

# 创建画布
fig, ax = plt.subplots(figsize=(10, 7))

regions = list(files.keys())

# ---- 统一的CI样式（浅灰+透明度）----
ci_color = '#B0B0B0'   # 浅灰
ci_alpha = 0.3        # 透明度（你可以调：0.15~0.35）

for region in regions:
    if region not in agg:
        continue

    df_r = agg[region]
    yrs = df_r['year']
    m = df_r['mean']
    ci = df_r['ci']

    color = region_colors.get(region, 'black')
    marker = region_markers.get(region, 'o')

    # 1. 计算 P 值 (用于图例)
    slope, intercept, r_val, p_val, std_err = linregress(yrs, m)

    if p_val < 0.001:
        stars = "***"
        p_str = "$P$ < 0.001"
    elif p_val < 0.01:
        stars = "**"
        p_str = f"$P$ = {p_val:.3f}"
    elif p_val < 0.05:
        stars = "*"
        p_str = f"$P$ = {p_val:.3f}"
    else:
        stars = ""
        p_str = f"$P$ = {p_val:.2f}"

    label_text = f"{region} ({p_str}{stars})"

    # 2. 绘制折线（实线 + Marker）
    ax.plot(
        yrs, m,
        linestyle='-', linewidth=4, alpha=0.6,
        marker=marker, markersize=10,
        color=color,
        label=label_text
    )

    # 3. 绘制置信区间阴影（统一浅灰 + 透明度）
    ax.fill_between(
        yrs, m - ci, m + ci,
        color=ci_color, alpha=ci_alpha,
        linewidth=0, # 去掉边缘线，视觉更干净
    )

    # 4. 绘制趋势线（虚线）
    xs = np.array([yrs.min(), yrs.max()])
    ax.plot(xs, slope * xs + intercept, linestyle='--', color=color, linewidth=3)

# ========== 4. 装饰与保存 ==========
ax.set_xlabel('Year')
ax.set_xticks([1980, 1990, 2000, 2010, 2019])
ax.set_ylabel('PSI')

# 自动缩放Y轴
ax.autoscale(enable=True, axis='y')

# ---- 背景网格线：横纵 + 虚线 + 透明度 ----
ax.set_axisbelow(True)  # 网格线放在图形元素下面
ax.grid(
    True, which='major', axis='both',
    linestyle='--', linewidth=2, alpha=0.95
)

# 图例：加粗
leg = ax.legend(frameon=False, fontsize=18, loc='upper center', bbox_to_anchor=(0.5, 1.02))
for t in leg.get_texts():
    t.set_fontweight('bold')

plt.tight_layout()
fig.savefig(out_png, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"✅ PSI 合并绘图完成！图片保存至: {out_png}")
