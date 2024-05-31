import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score

# 定义CARS函数...
def CARS(X, y, iteration=50, n_comps=20, cv=10):
    #iteration：算法的迭代次数     n_comps：PLSR模型中使用的主成分数量。   cv：交叉验证的折数
    #每次生成的波段数量不一

    N, D = X.shape
    prob = 0.8
    a = np.power((D / 2), (1 / (iteration - 1)))
    k = (np.log(D / 2)) / (iteration - 1)
    r = [round(a * np.exp(-(k * i)) * D) for i in range(1, iteration + 1)]

    weights = np.ones(D) / D
    RMSECV = []
    idWs = []

    for i in range(iteration):
        idCal = np.random.choice(np.arange(N), size=int(prob * N), replace=False)
        idW = np.random.choice(np.arange(D), size=r[i], p=weights / weights.sum(), replace=False)
        idWs.append(idW)

        X_cal = X[idCal[:, np.newaxis], idW]
        Y_cal = y[idCal]
        comp = min(n_comps, len(idW))
        pls = PLSRegression(n_components=comp)
        pls.fit(X_cal, Y_cal)

        absolute = np.abs(pls.coef_).reshape(-1)
        weights[idW] = absolute / sum(absolute)
        MSE = -cross_val_score(pls, X_cal, Y_cal, cv=cv, scoring="neg_mean_squared_error")
        RMSE = np.mean(np.sqrt(MSE))
        RMSECV.append(RMSE)

    best_index = np.argmin(RMSECV)
    W_best = idWs[best_index]
    return W_best

def convert_percentage_to_float(x):
    if '%' in x:
        return float(x.strip('%')) / 100
    else:
        return float(x)

# 读取CSV文件
data = pd.read_csv('Data/AAA.csv', header=None)

X = data.iloc[1:63, 1:].values  # 第一列是标签，其余的是特征
y = data.iloc[1:63, 0].apply(convert_percentage_to_float).values.astype(float)

# 应用CARS算法选择特征
selected_features_indices = CARS(X, y)

# 使用选定的特征，并将标签列添加回来
X_selected = X[:, selected_features_indices]
selected_data_with_labels = np.column_stack((y, X_selected))

# 将选定的特征及其标签保存到Excel，按照最优特征顺序
selected_data_df_optimal_order = pd.DataFrame(selected_data_with_labels, columns=['Label'] + [f'Feature_{i}' for i in selected_features_indices])
selected_data_df_optimal_order.to_csv('selected_features_optimal_order.csv', index=False)
print('selected_features_optimal_order.csv   为波段重要性排序')



# 对选定的特征索引按波段顺序进行排序
sorted_selected_features_indices = np.sort(selected_features_indices)

# 使用排序后的选定特征，并将标签列添加回来
X_selected_sorted = X[:, sorted_selected_features_indices]
selected_data_with_labels_sorted = np.column_stack((y, X_selected_sorted))

# 将选定的特征及其标签保存到Excel，按照波段顺序排序
selected_data_df_band_order = pd.DataFrame(selected_data_with_labels_sorted, columns=['Label'] + [f'Feature_{i}' for i in sorted_selected_features_indices])
selected_data_df_band_order.to_csv('selected_features_band_order.csv', index=False)
print('selected_features_band_order.csv   为波段顺序排序')


print('进程处理完毕，文件已保存')