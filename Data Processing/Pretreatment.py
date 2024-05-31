import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn import preprocessing
from scipy.signal import savgol_filter
from copy import deepcopy
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
import pywt
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from scipy.signal import detrend
from sklearn.metrics import r2_score, mean_squared_error
import csv

# 对每一组数据进行移动平均平滑处理
def smooth_data(data, window_size=10):
    """
    对每一组数据进行移动平均平滑处理。

    参数:
    - data: 要处理的数据，一个二维数组，每行为一组数据。
    - window_size: 移动平均窗口的大小，默认为10。

    返回:
    - smoothed_data: 平滑处理后的数据，与输入数据维度相同。
    """
    smoothed_data = []
    for y in data:
        smoothed_y = pd.Series(y).rolling(window=window_size).mean()
        smoothed_data.append(smoothed_y)
    return smoothed_data



# 绘制曲线并保存
def plot_smoothed_data(x_data, smoothed_data, group_names, title, filename, dpi=600, a=4, b=3.5):
    """
    绘制平滑后的曲线。

    参数:
    - x_data: x轴数据。
    - smoothed_data: 平滑后的y轴数据，一个二维数组。
    - group_names: 每组数据的名称，一个列表。
    - title: 图表标题。
    - filename: 保存文件的路径和名称。
    - dpi: 图片分辨率，默认为300。
    """
    plt.figure(figsize=(a,b))
    color_dict = {}  # 用于存储每个标签对应的颜色
    for i, corrected_y in enumerate(smoothed_data):
        label = group_names[i]
        if label not in color_dict:
            color_dict[label] = np.random.rand(3,)  # 生成一个随机颜色并存储在字典中
        color = color_dict[label]
        # 检查是否已经添加过该标签名，如果是则不再添加
        handles, labels = plt.gca().get_legend_handles_labels()
        if label not in labels:
            plt.plot(x_data, corrected_y, label=label, color=color)
        else:
            # 如果标签已存在，则只绘制曲线，不添加到图例中
            plt.plot(x_data, corrected_y, color=color)

    plt.xlabel(r'Frequency/THz')
    plt.ylabel(r'Transmission')
    plt.title(title)
    plt.legend(fontsize=6)
    plt.savefig(filename, dpi=dpi)

def plot_VSpredictions(y_true, y_pred_rbf, y_pred_lin, y_pred_poly, title, filename, dpi=300, a=4, b=3.5):
    """
    绘制真实值与三种模型的预测值对比图。

    参数:
    - y_true: 真实值。
    - y_pred_rbf: 使用RBF核的SVM模型预测的值。
    - y_pred_lin: 使用线性核的SVM模型预测的值。
    - y_pred_poly: 使用多项式核的SVM模型预测的值。
    """
    plt.figure(figsize=(a,b))
    plt.plot(y_true, label='True',linestyle='--')
    plt.plot(y_pred_rbf, label='rbf_Predicted',linestyle='--')
    plt.plot(y_pred_lin, label='lin_Predicted',linestyle='-.')
    plt.plot(y_pred_poly, label='poly_Predicted',linestyle=':')
    plt.legend()
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(title)
    plt.savefig(filename, dpi=dpi)

def SG(data, w=7, p=3, d=0):
    """
    SG平滑  可以在这里调整参数  w代表窗口 p代表多项式阶数  d代表导数
    多项式的阶数决定了用于拟合的多项式的程度。更高的多项式阶可以 捕获更复杂的趋势 ，但也可能引入不必要的振荡。
    d为0代表不求导  d为1代表一阶导    过程即先平滑  后求导

    """
    data_copy = deepcopy(data)
    if isinstance(data_copy, pd.DataFrame):
        data_copy = data_copy.values

    data_copy = savgol_filter(data_copy, w, polyorder=p, deriv=d)
    return data_copy




def StandardScaler(data):

    scaler = StandardScaler()

    # 使用 StandardScaler 对数据进行标准化
    standardized_data = scaler.fit_transform(data)
    return standardized_data




def msc(input_data, reference=None):
    """
    多元散射校正（MSC）的实现。

    参数:
    input_data: numpy array，形状为 (样本数, 波长数) 的光谱数据。
    reference: 可选，用于校正的参考光谱。如果为 None，则使用输入数据的平均光谱。

    返回:
    校正后的数据。
    """
    # 如果没有提供参考，则使用所有样本的平均光谱

    if reference is None:
        reference = np.mean(input_data, axis=0)

    # 初始化校正后的数据数组
    corrected_data = np.zeros_like(input_data)

    # 对每个样本进行处理
    for i in range(input_data.shape[0]):
        # 获取当前样本
        sample = input_data[i, :]

        # 计算回归系数
        fit = np.polyfit(reference, sample, 1, full=True)

        # 应用校正
        corrected_data[i, :] = (sample - fit[0][1]) / fit[0][0]

    return corrected_data




def snv(input_data):
    # 对每一行应用SNV转换
    snv_transformed = (input_data - input_data.mean(axis=1, keepdims=True)) / input_data.std(axis=1, keepdims=True)
    return snv_transformed

def pca(input_data, n_components=None):
    """
    主成分分析（PCA）的实现。

    参数:
    input_data: numpy array，形状为 (样本数, 特征数) 的数据。
    n_components: 可选，要保留的主成分数量。默认为 None，即保留所有主成分。

    返回:
    降维后的数据。
    """
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(input_data)
    return pca_data

def detrend_signal(signal):
    """
    去除信号的基线漂移

    参数:
    signal: 一维数组，表示输入信号。

    返回:
    detrended_signal: 去除基线漂移后的信号。
    """
    return detrend(signal)

def empirical_mode_decomposition(signal):
    """
    经验模态分解（EMD）

    参数:
    signal: 一维数组，表示输入信号。

    返回:
    IMFs: 按照IMF顺序排列的数组（IMF = Intrinsic Mode Function）。
    """
    # EMD的实现可以使用第三方库，比如PyEMD
    from PyEMD import EMD
    emd = EMD()
    IMFs = emd(signal)
    return IMFs



def evaluate_model(Y_true, Y_pred, sample_name):
    """
    计算并打印模型评估指标。

    参数:
    - Y_true: 真实值。
    - Y_pred: 预测值。
    - ee: 次数。
    - kk: loss值。
    """
    r2 = r2_score(Y_true, Y_pred)
    r2_str = '{:.2f}'.format(r2)
    mse = mean_squared_error(Y_true, Y_pred)
    rmse = np.sqrt(mse)

    print(sample_name,"R方指标为：", r2_str)
    print(sample_name,"MSE  : ", round(mse, 4))
    print(sample_name,"RMSE : ", round(rmse, 4))

    return r2


def evaluate_SVM_model(Y_true, Y_pred,sample_name):

    cm = confusion_matrix(Y_true, Y_pred)   # 1. 混淆矩阵

    accuracy_rbf = accuracy_score(Y_true, Y_pred)  # 2. 准确率

    # recall = recall_score(Y_true, Y_pred)   # 3. 召回率
    #
    # f1 = f1_score(Y_true, Y_pred)   # 4. F1 分数
    #
    # auc = roc_auc_score(Y_true, Y_pred) # 5. ROC 曲线和 AUC

    print("{:<30}混淆矩阵：\n{}".format(sample_name, cm))
    print("{:<30}准确率 : {:.2f}".format(sample_name, accuracy_rbf))
    # print("{:<40}召回率 : {:.2f}".format(sample_name, recall))
    # print("{:<40}f1分数 : {:.2f}".format(sample_name, f1))
    # print("{:<40}AUC  : {:.2f}".format(sample_name, auc))


def convert_percentage_to_float(x):
    if isinstance(x, str) and '%' in x:
        return float(x.strip('%'))/ 100.0
    else:
        return x  # 如果不是字符串或者没有百分号，直接返回原始值

def augment_data(original_data, labels):
    """
    对数据进行扩充，计算相同标签值的数据两两之间的平均值，并创建新的数据组。

    参数:
    - original_data: 原始数据，每行表示一个样本。
    - labels: 标签，表示每个样本的类别。

    返回:
    - augmented_data: 扩充后的数据，包含新的数据组。
    - augmented_labels: 扩充后的标签，与扩充后的数据相对应。
    """
    unique_labels = np.unique(labels)  # 获取唯一的标签值
    augmented_data = []
    augmented_labels = []

    # 对每个标签值进行循环
    for label in unique_labels:
        # 找到具有相同标签值的数据
        label_indices = np.where(labels == label)[0]
        label_data = original_data[label_indices]

        # 计算两两数据之间的平均值
        for i in range(len(label_data)):
            for j in range(i + 1, len(label_data)):
                avg_data = np.mean([label_data[i], label_data[j]], axis=0)
                augmented_data.append(avg_data)
                augmented_labels.append(label)

    return np.array(augmented_data), np.array(augmented_labels)

def augment_and_merge_data(input_csv_file, output_csv_file, terminating_row):
    """
    从输入的CSV文件中读取数据，对数据进行扩充，然后将原始数据和扩充后的数据合并，并将结果写入到输出的CSV文件中。

    参数:
    - input_csv_file: 输入的CSV文件名，包含原始数据。
    - output_csv_file: 输出的CSV文件名，包含合并后的数据。
    """
    # 读取CSV文件
    df = pd.read_csv(input_csv_file)

    # 提取特征数据，转换为float类型
    x_data = df.iloc[0:terminating_row, 1:].values.astype(float)

    # 提取标签数据，转换为float类型
    y = df.iloc[0:terminating_row, 0].apply(convert_percentage_to_float).values.astype(float)

    # 创建新的DataFrame对象
    new_df = pd.DataFrame(x_data, columns=df.columns[1:])  # 特征从第1列开始
    new_df.insert(0, 'Label', y)  # 将标签插入到DataFrame的第0列

    # 对数据进行扩充
    augmented_data, augmented_labels = augment_data(x_data, y)

    # 将扩充后的数据和标签转换为 DataFrame
    augmented_df = pd.DataFrame(augmented_data, columns=new_df.columns[1:])  # 特征从第1列开始
    augmented_df.insert(0, 'Label', augmented_labels)  # 将标签插入到DataFrame的第0列

    # 合并原始数据和扩充后的数据
    merged_df = pd.concat([new_df, augmented_df])

    # 将合并后的 DataFrame 写入 CSV 文件
    merged_df.to_csv(output_csv_file, index=False)

def Data_Sort(input_csv_file1, input_csv_file2, input_csv_file3, output_csv_file, terminating_row1, terminating_row2):
    # 读取CSV文件
    df1 = pd.read_csv(input_csv_file1)
    df2 = pd.read_csv(input_csv_file2)
    df3 = pd.read_csv(input_csv_file3)

    # 提取特征数据，转换为float类型
    x_data1 = df1.iloc[3:terminating_row1, 1:].values.astype(float)
    x_data2 = df2.iloc[3:terminating_row1, 1:].values.astype(float)
    x_data3 = df3.iloc[6:terminating_row2, 1:].values.astype(float)

    x_data4 = df1.iloc[0:3, 1:].values.astype(float)
    x_data5 = df2.iloc[0:3, 1:].values.astype(float)
    x_data6 = df3.iloc[1:6, 1:].values.astype(float)

    # 合并特征数据
    x_data0 = np.concatenate((x_data4, x_data5, x_data6), axis=0)

    # 提取标签数据，统一为统一的标签值
    unified_y0 = np.full(11, 'Background')  # 使用 NumPy 创建由 'TNT' 组成的数组
    unified_y1 = np.full(terminating_row1-3, 'KNO3')  # 使用 NumPy 创建由 'KNO3' 组成的数组
    unified_y2 = np.full(terminating_row1-3, 'NH4NO3')  # 使用 NumPy 创建由 'NH4NO3' 组成的数组
    unified_y3 = np.full(terminating_row2-6, 'TNT')  # 使用 NumPy 创建由 'TNT' 组成的数组

    # 创建新的DataFrame对象
    new_df0 = pd.DataFrame(x_data0, columns=df1.columns[1:])  # 特征从第1列开始
    new_df0.insert(0, 'Label', unified_y0)  # 将标签插入到DataFrame的第0列

    new_df1 = pd.DataFrame(x_data1, columns=df1.columns[1:])  # 特征从第1列开始
    new_df1.insert(0, 'Label', unified_y1)  # 将标签插入到DataFrame的第0列

    new_df2 = pd.DataFrame(x_data2, columns=df2.columns[1:])  # 特征从第1列开始
    new_df2.insert(0, 'Label', unified_y2)  # 将标签插入到DataFrame的第0列

    new_df3 = pd.DataFrame(x_data3, columns=df3.columns[1:])  # 特征从第1列开始
    new_df3.insert(0, 'Label', unified_y3)  # 将标签插入到DataFrame的第0列

    # 合并原始数据和扩充后的数据
    merged_df = pd.concat([new_df0, new_df1, new_df2, new_df3])

    # 将合并后的 DataFrame 写入 CSV 文件
    merged_df.to_csv(output_csv_file, index=False)
