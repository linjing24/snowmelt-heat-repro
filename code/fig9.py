# -*- coding: utf-8 -*-
import os, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import shap

# ========= 路径 =========
DATA_DIR = r"D:\BNU-Artical2\数据处理过程\47.结果3.4变量汇总（最终版）-标准化"
OUT_DIR  = r"D:\BNU-Artical2\结果图_SHAP_yearscale_70_30_RF"
os.makedirs(OUT_DIR, exist_ok=True)

# 统一字体
plt.rcParams["font.family"] = "Times New Roman"

# ========= 变量清单 =========
PREDICTOR_CANDIDATES = [
    "melt_peak", "swe", "pre", "SDI", "PSI",
    "tem_avg_y", "spei_1", "spei_3", "spei_6", "spei_12",
    "alt", "precip", "PET"
]
RESPONSES = ["hot_days", "txx"]
EXCLUDE_IN_SHAP = {"lat", "lon"}      # 若后续加入经纬度，会在解释中屏蔽
KEY_DEP_FEATURE = "melt_peak"
TRAIN_FRAC = 0.7                      # 站点级 7:3 划分

# ========= 月→年聚合规则 =========
SUM_VARS  = {"precip", "PET", "pre", "hot_days"}   # 年总量
MAX_VARS  = {"melt_peak", "txx"}                   # 年最大
MEAN_VARS = {"swe", "SDI", "PSI", "tem_avg_y",     # 年均（含 SPEI、地形）
             "spei_1", "spei_3", "spei_6", "spei_12",
             "lat", "lon", "alt"}

# ========= 图上显示用的重命名 =========
FEATURE_RENAME = {
    "melt_peak": r"$M_{\text{peak}}$",
    "txx":       r"$T_{\max}$", "hot_days":  r"$N_{\text{hot}}$",
    "swe": "SWE", "pre": "Pre",
    "SDI": "SDI", "PSI": "PSI", "tem_avg_y": "T",
    "spei_1": "SPEI-1", "spei_3": "SPEI-3", "spei_6": "SPEI-6", "spei_12": "SPEI-12",
    "alt": "Alt", "PET": "PET"
}
def pretty(name: str) -> str:
    return FEATURE_RENAME.get(name, name)

# ========= 读数 & 聚合 =========
def load_monthly_dataframe(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*.csv"))
    dfs = []
    for fp in files:
        df = pd.read_csv(fp)
        if "station" not in df.columns:
            df["station"] = os.path.basename(fp).replace(".csv", "")
        if "year" not in df.columns:
            if "date" in df.columns:
                d = pd.to_datetime(df["date"], errors="coerce")
                df["year"] = d.dt.year
            else:
                raise ValueError(f"{fp} 缺少 year 列且无法从 date 推断。")
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def to_yearly(df_monthly: pd.DataFrame):
    cols = set(df_monthly.columns)
    predictors = [c for c in PREDICTOR_CANDIDATES if c in cols]
    responses  = [c for c in RESPONSES if c in cols]

    agg = {}
    for v in predictors + responses:
        if v in SUM_VARS and v in cols:   agg[v] = "sum"
        elif v in MAX_VARS and v in cols: agg[v] = "max"
        else:                              agg[v] = "mean"
    yearly = df_monthly.groupby(["station", "year"], as_index=False).agg(agg)
    return yearly, predictors, responses

# ========= 随机森林 =========
def make_model_rf():
    return RandomForestRegressor(
        n_estimators=800,
        max_depth=8,
        min_samples_leaf=20,
        min_samples_split=40,
        max_features='sqrt',
        bootstrap=True,
        n_jobs=-1,
        random_state=42
    )

# ========= SHAP 上下组合图（上 Bar / 下 Beeswarm） =========
def plot_combo_beeswarm_with_bar(
    shap_values_expl: shap._explanation.Explanation,
    features_to_show: list,
    title: str,
    outfile: str,
    figsize=(7, 8)   # <<< 可控尺寸
):
    # 1) 计算 mean(|SHAP|)
    shap_sub  = shap_values_expl[:, features_to_show]
    mean_abs  = np.abs(shap_sub.values).mean(axis=0)

    order_desc   = np.argsort(mean_abs)[::-1]
    feats_sorted = [features_to_show[i] for i in order_desc]
    means_sorted = mean_abs[order_desc]

    feats_plot  = feats_sorted[::-1]
    means_plot  = means_sorted[::-1]
    labels_plot = [pretty(f) for f in feats_plot]

    # 2) 先建图
    plt.close('all')
    fig, ax_bee = plt.subplots(figsize=figsize)
    plt.sca(ax_bee)

    # 3) 绘制 SHAP beeswarm
    shap.plots.beeswarm(
        shap_sub[:, feats_plot],
        max_display=len(feats_plot),
        show=False
    )

    # 4) 关键：强制回写尺寸，避免被 SHAP 改掉
    fig.set_size_inches(*figsize, forward=True)

    # 5) 美化
    ax_bee.set_yticklabels(labels_plot, fontsize=24, color='black')
    ax_bee.tick_params(axis='x', labelsize=24)
    ax_bee.set_xlabel("SHAP value", fontsize=24, labelpad=8)
    ax_bee.set_title(title, pad=14, fontsize=18)
    ax_bee.spines['right'].set_visible(False)
    ax_bee.set_zorder(10)
    ax_bee.patch.set_alpha(0)

    # 6) 顶部 bar 图
    ylocs = ax_bee.get_yticks()
    ax_bar = ax_bee.twiny()
    ax_bar.set_zorder(-10)
    ax_bar.patch.set_alpha(0)
    ax_bar.barh(
        ylocs, means_plot, height=0.6,
        color="#DAE3F3", alpha=0.9, edgecolor="none",
        zorder=-20
    )
    ax_bar.set_ylim(ax_bee.get_ylim())
    ax_bar.xaxis.set_label_position('top')
    ax_bar.xaxis.tick_top()
    ax_bar.tick_params(axis='x', labelsize=24)
    ax_bar.set_xlabel("Mean(|SHAP|)", fontsize=24, labelpad=8)
    xmax = float(means_plot.max()) * 1.12 if means_plot.size else 1.0
    ax_bar.set_xlim(0, xmax)
    ax_bar.spines['right'].set_visible(False)

    # 7) 调整 colorbar 字体
    for a in fig.axes:
        if a not in (ax_bee, ax_bar):
            a.tick_params(labelsize=22)
            if hasattr(a, "yaxis") and hasattr(a.yaxis, "label"):
                a.yaxis.label.set_size(22)

    plt.tight_layout()
    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

# ------------------ 主流程：按站点 7:3 划分 + 分期 ------------------
if __name__ == "__main__":
    FIGSIZE_BEESWARM = (7, 8)
    FIGSIZE_DEP = (7, 8)

    monthly = load_monthly_dataframe(DATA_DIR)
    yearly, predictors, responses = to_yearly(monthly)

    print(f"年尺度数据形状: {yearly.shape}")
    print(f"用于建模的预测因子: {predictors}")
    print(f"用于建模的响应变量: {responses}")

    # 两个时期
    periods = {
        "1980_1999": (1980, 1999),
        "2000_2019": (2000, 2019)
    }

    all_metrics = []

    for pname, (start, end) in periods.items():
        print(f"\n====== 分析时期 {pname} ({start}-{end}) ======")
        df_period = yearly[(yearly["year"] >= start) & (yearly["year"] <= end)].copy()

        # 站点级 7:3 划分
        rng = np.random.RandomState(2025)
        stations = pd.unique(df_period["station"])
        rng.shuffle(stations)
        n_train = int(len(stations) * TRAIN_FRAC)
        train_stations = set(stations[:n_train])
        test_stations  = set(stations[n_train:])

        df_train = df_period[df_period["station"].isin(train_stations)].copy()
        df_test  = df_period[df_period["station"].isin(test_stations)].copy()

        print(f"训练站点数: {len(train_stations)}, 测试站点数: {len(test_stations)}")
        print(f"训练样本: {len(df_train)}, 测试样本: {len(df_test)}")

        for target in responses:
            print(f"\n=== RandomForest 70/30 split modeling for {target} ({pname}) ===")
            cols_need = predictors + [target, "station"]
            dtr = df_train.dropna(subset=cols_need).copy()
            dte = df_test.dropna(subset=cols_need).copy()

            X_tr = dtr[predictors].values
            y_tr = dtr[target].values
            X_te = dte[predictors].values
            y_te = dte[target].values

            model = make_model_rf()
            model.fit(X_tr, y_tr)

            # 评价
            yhat_tr = model.predict(X_tr)
            yhat_te = model.predict(X_te)
            r2_train = r2_score(y_tr, yhat_tr)
            r2_test  = r2_score(y_te, yhat_te)
            mae_test = mean_absolute_error(y_te, yhat_te)
            print(f"Train R²={r2_train:.3f} | Test R²={r2_test:.3f}, Test MAE={mae_test:.3f}")

            all_metrics.append({
                "period": pname, "response": target,
                "R2_train": r2_train, "R2_test": r2_test, "MAE_test": mae_test
            })

            # SHAP 解释
            explainer = shap.TreeExplainer(model)
            X_te_df = pd.DataFrame(dte[predictors], columns=predictors)
            shap_values_test = explainer(X_te_df)

            features_to_show = [f for f in predictors if f not in EXCLUDE_IN_SHAP]

            # beeswarm+bar
            ttl = f"SHAP  — {target}  — {pname}"
            outp = os.path.join(OUT_DIR, f"RF_SHAP_combo_exLatLon_{target}_test_{pname}.png")
            plot_combo_beeswarm_with_bar(shap_values_test, features_to_show, ttl, outp, figsize=FIGSIZE_BEESWARM)

            # 依存图
            if KEY_DEP_FEATURE in features_to_show:
                plt.close('all')
                shap.plots.scatter(
                    shap_values_test[:, KEY_DEP_FEATURE],
                    color=shap_values_test,
                    show=False
                )
                fig = plt.gcf()
                fig.set_size_inches(*FIGSIZE_DEP, forward=True)
                ax = plt.gca()
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='x', labelsize=24)
                ax.set_xlabel(ax.get_xlabel(), fontsize=24)
                plt.title(
                    f"SHAP Dependence (RF): {pretty(KEY_DEP_FEATURE)} -> {target} (test set, {pname})",
                    fontsize=20
                )
                plt.tight_layout()
                plt.savefig(
                    os.path.join(OUT_DIR, f"RF_SHAP_depend_{KEY_DEP_FEATURE}_{target}_test_{pname}.png"),
                    dpi=300, bbox_inches="tight"
                )
                plt.close()

    # 保存 metrics
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(OUT_DIR, "RF_metrics_70_30_split_by_period.csv"),
        index=False, encoding="utf-8-sig"
    )
    print(f"\n完成：图件和 metrics 已保存到 {OUT_DIR}")
