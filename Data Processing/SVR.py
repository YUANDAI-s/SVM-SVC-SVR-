import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVR
import Pretreatment as pre
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取CSV文件
pre.augment_and_merge_data('Data/AAA.csv', 'Data/AAA_augment_data.csv', 63)
pre.augment_and_merge_data('Data/BBB.csv', 'Data/BBB_augment_data.csv', 63)

df_AAA = pd.read_csv('Data/AAA_augment_data.csv',header=None)
df_BBB = pd.read_csv('Data/BBB_augment_data.csv',header=None)
df_CCC = pd.read_csv('Data/CCC.csv',header=None)


# 提取除第一行外的数据作为x轴数据，转换为float类型(透射率)
x_data_AAA = df_AAA.iloc[1:126, 1:].values.astype(float)
x_data_BBB = df_BBB.iloc[1:126, 1:].values.astype(float)
x_data_CCC = df_CCC.drop(df_CCC.index[6]).iloc[1:51, 1:].values.astype(float)

# 提取第一列数据作为组名,转换为float类型（标签）
y_AAA = df_AAA.iloc[1:126, 0].apply(pre.convert_percentage_to_float).values.astype(float)
y_BBB = df_BBB.iloc[1:126, 0].apply(pre.convert_percentage_to_float).values.astype(float)
y_CCC = df_CCC.drop(df_CCC.index[6]).iloc[1:51, 0].apply(pre.convert_percentage_to_float).values.astype(float)


# sg平滑处理
# print('使用sg就解除注释即可，每次只用一种预处理')
x_data_scaled_AAA = pre.SG(x_data_AAA)
x_data_scaled_BBB = pre.SG(x_data_BBB)
x_data_scaled_CCC = pre.SG(x_data_CCC)

# 划分数据集
X_train_AAA, X_test_AAA, Y_train_AAA, Y_test_AAA = train_test_split(x_data_scaled_AAA, y_AAA, train_size=0.8, random_state=42)
X_train_BBB, X_test_BBB, Y_train_BBB, Y_test_BBB = train_test_split(x_data_scaled_BBB, y_BBB, train_size=0.8, random_state=42)
X_train_CCC, X_test_CCC, Y_train_CCC, Y_test_CCC = train_test_split(x_data_scaled_CCC, y_CCC, train_size=0.8, random_state=42)

# PCA主成分分析
# x_data_scaled=pre.pca(x_data_scaled,50)

# 扣除背景
# background_corrected_data = []
#
# for x in x_data_scaled:
#     corrected_x = (x - x_data_scaled[0]) / x_data_scaled[0]
#     background_corrected_data.append(corrected_x)

# MSC（多元散射校正）×
# print('使用msc就解除注释即可，每次只用一种预处理')
# x_data_scaled=pre.msc(x_data_scaled_sg)   #msc就执行这行代码

# SNV（标准正态变量）×
# print('使用snv就解除注释即可，每次只用一种预处理')
# x_data_scaled=pre.snv(x_data_scaled_sg)    #就执行这行代码    可以在pre的.py程序里设置具体参数

# 标准化处理,将数据按特征列进行均值为0，方差为1的缩放
# print('使用标准化就解除注释即可，每次只用一种预处理')
# x_data_scaled=x_data_scaled_sg.transpose()                 #这个要改转置  转置的.后面要加括号 #执行标准化操作 标准化操作要进行一次反射率转置
# x_data_scaled=pre.StandardScaler(x_data_scaled)           #标准化操作    一直出错是因为 在定义def函数的时候，需要定义两个输入 一个self  一个 传参 没有定义self  所以一直报错   传入的数据是198*310
# x_data_scaled=x_data_scaled.transpose()                 #za处理完毕了 再转置回来

# 标准化
# scaler = MinMaxScaler()
# x_data_scaled = scaler.fit_transform(x_data)

# 归一化
# scaler = MinMaxScaler()
# x_data_scaled = scaler.fit_transform(x_data)

def train_svr(x_data_scaled, y, C=1,gamma=0.1,epsilon=0.01,degree=2):
    """
    使用SVR训练三种核的模型。

    参数:
    - x_data_scaled_AAA: 经过缩放的AAA数据。
    - y_AAA: AAA的标签数据。
    - x_data_scaled_BBB: 经过缩放的BBB数据。
    - y_BBB: BBB的标签数据。
    - x_data_scaled_CCC: 经过缩放的CCC数据。
    - y_CCC: CCC的标签数据。

    返回:
    - svr_rbf: RBF核的SVR模型。
    - svr_lin: 线性核的SVR模型。
    - svr_poly: 多项式核的SVR模型。
    """
    # 使用RBF核训练模型
    svr_rbf = SVR(kernel='rbf', C=C, gamma=gamma, epsilon=epsilon)
    svr_rbf.fit(x_data_scaled, y)

    # 使用线性核训练模型
    svr_lin = SVR(kernel='linear', C=C, gamma=gamma, epsilon=epsilon)
    svr_lin.fit(x_data_scaled, y)

    # 使用多项式核训练模型
    svr_poly = SVR(kernel='poly', C=C, gamma=gamma, epsilon=epsilon, degree=degree)
    svr_poly.fit(x_data_scaled, y)

    return svr_rbf, svr_lin, svr_poly
# 训练数据
svr_rbf_AAA, svr_lin_AAA, svr_poly_AAA = train_svr(x_data_scaled_AAA, y_AAA)
svr_rbf_BBB, svr_lin_BBB, svr_poly_BBB = train_svr(x_data_scaled_BBB, y_BBB)
svr_rbf_CCC, svr_lin_CCC, svr_poly_CCC = train_svr(x_data_scaled_CCC, y_CCC)

svr_rbf_AAA_split, svr_lin_AAA_split, svr_poly_AAA_split = train_svr(X_train_AAA, Y_train_AAA)
svr_rbf_BBB_split, svr_lin_BBB_split, svr_poly_BBB_split = train_svr(X_train_BBB, Y_train_BBB)
svr_rbf_CCC_split, svr_lin_CCC_split, svr_poly_CCC_split = train_svr(X_train_CCC, Y_train_CCC)


# 预测结果
def predict_svr(x_data_scaled, svr_rbf, svr_lin, svr_poly):
    # 使用RBF核训练模型
    y_rbf_pred = svr_rbf.predict(x_data_scaled)

    # 使用线性核训练模型
    y_lin_pred = svr_lin.predict(x_data_scaled)

    # 使用多项式核训练模型
    y_poly_pred = svr_poly.predict(x_data_scaled)

    return y_rbf_pred, y_lin_pred, y_poly_pred

y_rbf_pred_AAA, y_lin_pred_AAA, y_poly_pred_AAA = predict_svr(x_data_scaled_AAA, svr_rbf_AAA, svr_lin_AAA, svr_poly_AAA)
y_rbf_pred_BBB, y_lin_pred_BBB, y_poly_pred_BBB = predict_svr(x_data_scaled_BBB, svr_rbf_BBB, svr_lin_BBB, svr_poly_BBB)
y_rbf_pred_CCC, y_lin_pred_CCC, y_poly_pred_CCC = predict_svr(x_data_scaled_CCC, svr_rbf_CCC, svr_lin_CCC, svr_poly_CCC)

y_rbf_pred_AAA_test, y_lin_pred_AAA_test, y_poly_pred_AAA_test = predict_svr(X_test_AAA, svr_rbf_AAA_split, svr_lin_AAA_split, svr_poly_AAA_split)
y_rbf_pred_BBB_test, y_lin_pred_BBB_test, y_poly_pred_BBB_test = predict_svr(X_test_BBB, svr_rbf_BBB_split, svr_lin_BBB_split, svr_poly_BBB_split)
y_rbf_pred_CCC_test, y_lin_pred_CCC_test, y_poly_pred_CCC_test = predict_svr(X_test_CCC, svr_rbf_CCC_split, svr_lin_CCC_split, svr_poly_CCC_split)

# 计算R-squared值,MSE,RMSE
print('rbf:')
pre.evaluate_model(y_AAA, y_rbf_pred_AAA,'AAA')
pre.evaluate_model(y_BBB, y_rbf_pred_BBB,'BBB')
pre.evaluate_model(y_CCC, y_rbf_pred_CCC,'CCC')

print('split_rbf:')
pre.evaluate_model(Y_test_AAA, y_rbf_pred_AAA_test,'AAA')
pre.evaluate_model(Y_test_BBB, y_rbf_pred_BBB_test,'BBB')
pre.evaluate_model(Y_test_CCC, y_rbf_pred_CCC_test,'CCC')

print('lin:')
pre.evaluate_model(y_AAA, y_lin_pred_AAA,'AAA')
pre.evaluate_model(y_BBB, y_lin_pred_BBB,'BBB')
pre.evaluate_model(y_CCC, y_lin_pred_CCC,'CCC')

print('split_lin:')
pre.evaluate_model(Y_test_AAA, y_lin_pred_AAA_test,'AAA')
pre.evaluate_model(Y_test_BBB, y_lin_pred_BBB_test,'BBB')
pre.evaluate_model(Y_test_CCC, y_lin_pred_CCC_test,'CCC')

print('poly:')
pre.evaluate_model(y_AAA, y_poly_pred_AAA,'AAA')
pre.evaluate_model(y_BBB, y_poly_pred_BBB,'BBB')
pre.evaluate_model(y_CCC, y_poly_pred_CCC,'CCC')

print('split_poly:')
pre.evaluate_model(Y_test_AAA, y_poly_pred_AAA_test,'AAA')
pre.evaluate_model(Y_test_BBB, y_poly_pred_BBB_test,'BBB')
pre.evaluate_model(Y_test_CCC, y_poly_pred_CCC_test,'CCC')

# 绘制真实值与预测值的对比图
pre.plot_VSpredictions(y_AAA, y_rbf_pred_AAA, y_lin_pred_AAA, y_poly_pred_AAA, '未划分数据集', 'saved_files/(AAA—SVR)True vs Predicted Values.jpg',dpi=600, a=6.5, b=4)
pre.plot_VSpredictions(y_BBB, y_rbf_pred_BBB, y_lin_pred_BBB, y_poly_pred_BBB, '未划分数据集', 'saved_files/(BBB—SVR)True vs Predicted Values.jpg',dpi=600, a=6.5, b=4)
pre.plot_VSpredictions(y_CCC, y_rbf_pred_CCC, y_lin_pred_CCC, y_poly_pred_CCC, '未划分数据集', 'saved_files/(CCC—SVR)True vs Predicted Values.jpg',dpi=600, a=6.5, b=4)

pre.plot_VSpredictions(Y_test_AAA, y_rbf_pred_AAA_test, y_lin_pred_AAA_test, y_poly_pred_AAA_test, 'AAA样品支持向量机回归效果', 'saved_files/(split_AAA—SVR)True vs Predicted Values.jpg',dpi=600, a=6.5, b=4)
pre.plot_VSpredictions(Y_test_BBB, y_rbf_pred_BBB_test, y_lin_pred_BBB_test, y_poly_pred_BBB_test, 'BBB样品支持向量机回归效果', 'saved_files/(split_BBB—SVR)True vs Predicted Values.jpg',dpi=600, a=6.5, b=4)
pre.plot_VSpredictions(Y_test_CCC, y_rbf_pred_CCC_test, y_lin_pred_CCC_test, y_poly_pred_CCC_test, 'CCC样品支持向量机回归效果', 'saved_files/(split_CCC—SVR)True vs Predicted Values.jpg',dpi=600, a=6.5, b=4)
