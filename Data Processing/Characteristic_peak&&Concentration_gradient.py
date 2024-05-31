import pandas as pd
import matplotlib.pyplot as plt
import Pretreatment as pre

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取CSV文件
df_AAA = pd.read_csv('Data/AAA.csv',header=None)
df_BBB = pd.read_csv('Data/BBB.csv',header=None)
df_CCC = pd.read_csv('Data/CCC.csv',header=None)

# 获取x轴数据
x_AAA = df_AAA.iloc[0, 1:].values
x_BBB = df_BBB.iloc[0, 1:].values
x_CCC = df_CCC.iloc[0, 1:].values

# 获取组名
group_names_AAA_Peak = df_AAA.iloc[[64,65], 0].values
group_names_BBB_Peak = df_BBB.iloc[[64,65], 0].values
group_names_CCC_Peak = df_CCC.iloc[[6, 52], 0].values

group_names_AAA_gradient = df_AAA.iloc[4:65, 0].values
group_names_BBB_gradient = df_BBB.iloc[4:65, 0].values
group_names_CCC_gradient = df_CCC.iloc[6:51, 0].values

# 获取y轴数据
y_data_AAA_Peak = df_AAA.iloc[[64,65], 1:].values
y_data_BBB_Peak = df_BBB.iloc[[64,65], 1:].values
y_data_CCC_Peak = df_CCC.iloc[[6, 52], 1:].values

y_data_AAA_gradient = df_AAA.iloc[4:65, 1:].values
y_data_BBB_gradient = df_BBB.iloc[4:65, 1:].values
y_data_CCC_gradient = df_CCC.iloc[6:51, 1:].values

# 对每一组数据进行移动平均平滑处理
smoothed_y_data_AAA_Peak = pre.smooth_data(y_data_AAA_Peak, window_size=10)
smoothed_y_data_BBB_Peak = pre.smooth_data(y_data_BBB_Peak, window_size=10)
smoothed_y_data_CCC_Peak = pre.smooth_data(y_data_CCC_Peak, window_size=10)
# smoothed_y_data_CCC_Peak = y_data_CCC_Peak
# smoothed_y_data_CCC_Peak = pre.msc(y_data_CCC_Peak)

smoothed_y_data_AAA_gradient = pre.smooth_data(y_data_AAA_gradient, window_size=10)
smoothed_y_data_BBB_gradient = pre.smooth_data(y_data_BBB_gradient, window_size=10)
smoothed_y_data_CCC_gradient = pre.smooth_data(y_data_CCC_gradient, window_size=10)

# 绘制平滑后的曲线
pre.plot_smoothed_data(x_AAA, smoothed_y_data_AAA_Peak, group_names_AAA_Peak, 'AAA样品光谱图', 'saved_files/AAA样品光谱特征峰图.jpg', dpi=600, a=4, b=3.6)
pre.plot_smoothed_data(x_BBB, smoothed_y_data_BBB_Peak, group_names_BBB_Peak, 'BBB样品光谱图', 'saved_files/BBB样品光谱特征峰图.jpg', dpi=600, a=4, b=3.6)
pre.plot_smoothed_data(x_CCC, smoothed_y_data_CCC_Peak, group_names_CCC_Peak, 'CCC样品光谱图', 'saved_files/CCC样品光谱特征峰图.jpg', dpi=600, a=4, b=3.6)

pre.plot_smoothed_data(x_AAA, smoothed_y_data_AAA_gradient, group_names_AAA_gradient, 'AAA样品各浓度光谱图', 'saved_files/AAA样品光谱浓度梯度图.jpg', dpi=600, a=6, b=4)
pre.plot_smoothed_data(x_BBB, smoothed_y_data_BBB_gradient, group_names_BBB_gradient, 'BBB样品各浓度光谱图', 'saved_files/BBB样品浓度梯度图.jpg', dpi=600, a=6, b=4)
pre.plot_smoothed_data(x_CCC, smoothed_y_data_CCC_gradient, group_names_CCC_gradient, 'CCC样品各浓度光谱图', 'saved_files/CCC样品浓度梯度图.jpg', dpi=600, a=6, b=4)