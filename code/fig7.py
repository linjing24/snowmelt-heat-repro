import os
import glob
import pandas as pd

# 定义代表性年份
rep_years = [1980, 1985, 1990, 1995, 2000, 2005, 2010, 2015, 2019]

def process_station_file(file_path, rep_years):
    """
    处理单个站点的 CSV 文件：
      - 解析日期并提取年份；
      - 筛选出代表性年份的数据；
      - 筛选 model_precip_type 为 "Snow" 的记录；
      - 按年份聚合：计算降雪总量（total_snow）和降水频次（snow_frequency，即发生降雪的天数）；
      - 同时提取经纬度信息（假设存在 "lon" 和 "lat" 列，若无则输出空值）；
      - 返回一个 DataFrame，列：station_id, year, total_snow, snow_frequency, lon, lat
    """
    try:
        df = pd.read_csv(file_path, parse_dates=["date"])
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return None

    # 提取年份
    if "year" not in df.columns:
        df["year"] = df["date"].dt.year

    # 筛选代表性年份
    df_rep = df[df["year"].isin(rep_years)].copy()
    if df_rep.empty:
        return None

    # 筛选 model_precip_type 为 "Snow" 的记录
    df_snow = df_rep[df_rep["model_precip_type"] == "Snow"]

    # 按年份聚合：计算降雪总量和降水频次
    agg = df_snow.groupby("year")["pre"].agg(
        total_snow="sum",
        snow_frequency="count"
    ).reset_index()

    # 提取站点信息：站点ID、经纬度。若没有则赋空值
    if "station_id" in df.columns:
        station_id = df["station_id"].iloc[0]
    else:
        station_id = os.path.splitext(os.path.basename(file_path))[0]
    # 经度和纬度
    lon = df["lon"].iloc[0] if "lon" in df.columns else None
    lat = df["lat"].iloc[0] if "lat" in df.columns else None

    agg["station_id"] = station_id
    agg["lon"] = lon
    agg["lat"] = lat

    # 确保每个代表年份都有记录（若无则填0）
    agg_full = pd.DataFrame({"year": rep_years})
    agg = pd.merge(agg_full, agg, on="year", how="left")
    agg["total_snow"] = agg["total_snow"].fillna(0)
    agg["snow_frequency"] = agg["snow_frequency"].fillna(0).astype(int)
    agg["station_id"] = station_id
    agg["lon"] = lon
    agg["lat"] = lat

    return agg

def process_all_files(input_folder, rep_years):
    """
    遍历 input_folder 中所有 CSV 文件，
    对每个站点进行处理，返回所有站点数据的列表（每个元素为该站点的代表性年份数据 DataFrame）。
    """
    all_files = glob.glob(os.path.join(input_folder, "*.csv"))
    results = []
    for file_path in all_files:
        agg = process_station_file(file_path, rep_years)
        if agg is not None and not agg.empty:
            results.append(agg)
    if results:
        combined = pd.concat(results, ignore_index=True)
        return combined
    else:
        return None

def save_by_year(data, output_folder):
    """
    将所有站点的数据按照年份拆分，每个代表年份生成一个 CSV 文件。
    文件列包括：station_id, year, total_snow, snow_frequency, lon, lat。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for yr in rep_years:
        df_year = data[data["year"] == yr].copy()
        output_file = os.path.join(output_folder, f"{yr}_snow_data.csv")
        df_year.to_csv(output_file, index=False, encoding="utf-8-sig")
        print(f"已保存 {yr} 年数据到：{output_file}")

def main():
    # 修改为实际存放 348 个站点 CSV 文件的文件夹路径
    input_folder = r"D:\BNU-Artical2\数据处理过程\14-1.1980-2019 降雪识别-填充完整"
    # 修改为你希望保存输出文件的文件夹路径
    output_folder = r"D:\BNU-Artical2\数据处理过程\29. 气泡图数据"

    data = process_all_files(input_folder, rep_years)
    if data is not None:
        save_by_year(data, output_folder)
        print(f"共处理 {len(data['station_id'].unique())} 个站点的数据。")
    else:
        print("未能处理出有效数据。")

if __name__ == "__main__":
    main()
