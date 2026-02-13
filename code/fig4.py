import os
import numpy as np
import pandas as pd
import calendar
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm


# Helper function: average daylight hours for the 15th of each month
def day_length(lat, month):
    d = datetime(2001, month, 15).timetuple().tm_yday
    phi = np.deg2rad(lat)
    delta = np.deg2rad(23.45 * np.sin(2 * np.pi * (284 + d) / 365))
    omega = np.arccos(-np.tan(phi) * np.tan(delta))
    return 24 * omega / np.pi


# 1. Read and aggregate monthly data
def read_and_clean_data(file_path):
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])  # 转换为日期类型
    data = data.set_index('date')

    # 清理缺失值
    data = data.fillna(0)  # 如果需要填补NaN值，可以选择合适的填充方法
    return data


# 2. Calculate PET via Thornthwaite per station-year
def compute_pet(group):
    Tm = group['tem_avg'].values
    Tm_pos = np.where(Tm > 0, Tm, 0)
    I = np.sum((Tm_pos / 5) ** 1.514)
    a = 6.75e-7 * I ** 3 - 7.71e-5 * I ** 2 + 1.792e-2 * I + 0.49239
    pets = []
    for T, lat, month, year in zip(Tm_pos, group['lat'], group['month'], group['year']):
        ratio = (10 * T / I) if I > 0 else 0
        base = ratio ** a if ratio > 0 else 0
        L = day_length(lat, month)  # 日照时数
        N = calendar.monthrange(int(year), int(month))[1]  # 每月天数
        pets.append(16 * base * (L / 12) * (N / 30))
    group = group.copy()
    group['PET'] = pets
    return group


# 3. Water deficit D
def calculate_water_deficit(group):
    group['D'] = group['pre'] - group['PET']
    return group


# 4. Rolling sums for 1,3,6,12 months
def rolling_sums(group, window_sizes=[1, 3, 6, 12]):
    for k in window_sizes:
        group[f'D_{k}'] = (
            group
            .groupby('station_id')['D']
            .rolling(window=k, min_periods=k)
            .sum()
            .reset_index(level=0, drop=True)
        )
    return group


# 5. Empirical standardization to compute SPEI
def calculate_spei(group, window_sizes=[1, 3, 6, 12]):
    spei_list = []
    for k in window_sizes:
        Dk = group[f'D_{k}']
        p = Dk.rank(method='average') / (len(Dk) + 1)
        spei_list.append(pd.Series(norm.ppf(p), index=group.index))
    return spei_list


# 6. Process all files
def process_all_files(input_folder, output_folder):
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    for file in files:
        file_path = os.path.join(input_folder, file)
        output_file_path = os.path.join(output_folder, f"SPEI_{file}")

        # 读取数据
        data = read_and_clean_data(file_path)

        # 计算PET
        data = data.groupby(['station_id', 'year'], group_keys=False).apply(compute_pet)

        # 计算水分亏缺量
        data = calculate_water_deficit(data)

        # 滚动计算1, 3, 6, 12个月
        data = rolling_sums(data)

        # 计算SPEI
        spei_list = calculate_spei(data)

        # 将结果添加到DataFrame
        for i, spei in enumerate(spei_list):
            data[f'spei_{[1, 3, 6, 12][i]}'] = spei

        # 保存结果
        data.to_csv(output_file_path, index=False)
        print(f"Processed {file} and saved to {output_file_path}")


# 输入输出文件夹路径
input_folder = r"D:\BNU-Artical2\数据处理过程\1980-2019 降雪识别-填充完整（5.24）"
output_folder = r"D:\BNU-Artical2\数据处理过程\42-1. SPEI指数计算结果"

# 创建输出文件夹，如果不存在的话
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 处理所有站点的数据
process_all_files(input_folder, output_folder)

