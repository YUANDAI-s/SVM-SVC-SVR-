import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
import Pretreatment as pre
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

pre.Data_Sort('Data/AAA.csv', 'Data/BBB.csv', 'Data/CCC.csv', 'Data/Sort.csv', 63, 51)

df_Sort = pd.read_csv('Data/Sort.csv',header=None)

# 提取除第一行外的数据作为x轴数据，转换为float类型(透射率)
x_data_Sort = df_Sort.iloc[1:, 1:].values.astype(float)

# 提取第一列数据作为组名,转换为float类型（标签）
y_Sort = df_Sort.iloc[1:, 0].values.astype(str)

# sg平滑处理
# print('使用sg就解除注释即可，每次只用一种预处理')
x_data_scaled_Sort = pre.SG(x_data_Sort)

# 划分数据集
X_train_Sort, X_test_Sort, Y_train_Sort, Y_test_Sort = train_test_split(x_data_scaled_Sort, y_Sort, train_size=0.8, random_state=42)

# 创建一个 SVM 分类器，并指定核函数
clf_linear = svm.SVC(kernel='linear', C=1, gamma=0.1)  # 线性核函数
clf_poly = svm.SVC(kernel='poly', degree=1, C=1, gamma=0.1)  # 多项式核函数，次数为1
clf_rbf = svm.SVC(kernel='rbf', gamma=0.008, C=1)  # 高斯径向基核函数

clf_linear_split = svm.SVC(kernel='linear', C=1, gamma=0.1)  # 线性核函数
clf_poly_split = svm.SVC(kernel='poly', degree=1, C=1, gamma=0.1)  # 多项式核函数，次数为1
clf_rbf_split = svm.SVC(kernel='rbf', gamma=0.008, C=1)  # 高斯径向基核函数

# 在训练集上训练各个分类器
clf_linear_split.fit(x_data_scaled_Sort, y_Sort)
clf_poly_split.fit(x_data_scaled_Sort, y_Sort)
clf_rbf_split.fit(x_data_scaled_Sort, y_Sort)

clf_linear_split.fit(X_train_Sort, Y_train_Sort)
clf_poly_split.fit(X_train_Sort, Y_train_Sort)
clf_rbf_split.fit(X_train_Sort, Y_train_Sort)

# 在测试集上进行预测，并计算准确率
y_pred_linear = clf_linear_split.predict(x_data_scaled_Sort)
y_pred_poly = clf_poly_split.predict(x_data_scaled_Sort)
y_pred_rbf = clf_rbf_split.predict(x_data_scaled_Sort)

y_pred_linear_split = clf_linear_split.predict(X_test_Sort)
y_pred_poly_split = clf_poly_split.predict(X_test_Sort)
y_pred_rbf_split = clf_rbf_split.predict(X_test_Sort)

pre.evaluate_SVM_model(y_Sort, y_pred_linear,'Linear Kernel')
pre.evaluate_SVM_model(y_Sort, y_pred_poly,'Polynomial Kernel')
pre.evaluate_SVM_model(y_Sort, y_pred_rbf,'RBF Kernel')

pre.evaluate_SVM_model(Y_test_Sort, y_pred_linear_split,'Linear Kernel split')
pre.evaluate_SVM_model(Y_test_Sort, y_pred_poly_split,'Polynomial Kernel split')
pre.evaluate_SVM_model(Y_test_Sort, y_pred_rbf_split,'RBF Kernel split')

pre.plot_VSpredictions(Y_test_Sort, y_pred_rbf, y_pred_linear, y_pred_poly, '未划分数据集', 'saved_files/(Sort)True vs Predicted Values.jpg',dpi=600, a=6.5, b=4)

pre.plot_VSpredictions(Y_test_Sort, y_pred_rbf_split, y_pred_linear_split, y_pred_poly_split, '支持向量机分类效果', 'saved_files/(split_Sort)True vs Predicted Values.jpg',dpi=600, a=6.5, b=4)

