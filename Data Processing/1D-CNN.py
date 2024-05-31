import torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import Pretreatment as pre
import datetime
import os
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimSun']  # 设置字体为宋体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

folder_path = 'saved_files'
# 如果文件夹不存在，则创建它
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

num_epochs = 8000
batch_size = 5


now = datetime.datetime.now()
formatted_time = now.strftime("%m-%d_%H-%M") #在这里记录好此次训练的起始时间

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = torch.utils.data.TensorDataset(*data_arrays)
    print(DataLoader(dataset, batch_size, shuffle=is_train))
    print("****************************************************************")
    return DataLoader(dataset, batch_size, shuffle=is_train)     #返回的就是是传过去的数据集的加载器，是数据集的加载器

# 读取CSV文件
df_AAA = pd.read_csv('Data/AAA_augment_data.csv', header=None)
df_BBB = pd.read_csv('Data/BBB_augment_data.csv', header=None)
df_CCC = pd.read_csv('Data/CCC.csv', header=None)

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  # 如果输出为 'cuda'，说明你正在使用 GPU


def Data_loading(X_train, X_test, Y_train, Y_test):
    # 以下为改成张量格原来不是深度学习的格式，式，经过操作之后改为深度学习格式。
    # 经过torch.tensor操作变成了tensor的模式,用法为torch.tensor（目标变量，torch类型）
    # 已经进行了两个数据（x系列的）的转换tensor模式，原模式不是tensor格式，无法使用。
    X_train = torch.tensor(X_train, dtype=torch.float32).to("cuda")
    X_test = torch.tensor(X_test, dtype=torch.float32).to("cuda")
    Y_train = torch.tensor(Y_train, dtype=torch.float32).to("cuda")
    Y_test = torch.tensor(Y_test, dtype=torch.float32).to("cuda")

    a = X_train.shape[0]    #tongj统计个数用的  shape0 代表统计的是行数
    print('训练集样本数量为：')
    print(a)

    b = X_test.shape[0] #测试集样本数量为b
    print('测试集样本数量为：')
    print(b)

    Y_train = Y_train.reshape(a, 1)
    Y_test = Y_test.reshape(b, 1)

    return X_train, X_test, Y_train, Y_test

X_train_AAA, X_test_AAA, Y_train_AAA, Y_test_AAA = Data_loading(X_train_AAA, X_test_AAA, Y_train_AAA, Y_test_AAA)
X_train_BBB, X_test_BBB, Y_train_BBB, Y_test_BBB = Data_loading(X_train_BBB, X_test_BBB, Y_train_BBB, Y_test_BBB)
X_train_CCC, X_test_CCC, Y_train_CCC, Y_test_CCC = Data_loading(X_train_CCC, X_test_CCC, Y_train_CCC, Y_test_CCC)

#数据加载的工作完成，接下来对网络进行声明
def Network_declaration(Data_in_features = 5120, Data_out_features = 256):
    net = nn.Sequential(
        nn.Conv1d(1,16, 13, stride=2),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Conv1d(16, 32, 11, stride=2),
        nn.BatchNorm1d(32),
        nn.ReLU(),
        nn.MaxPool1d(2, stride=1),

        nn.Conv1d(32, 64,9, stride=1),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Conv1d(64, 64,7, stride=2),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.MaxPool1d(2, stride=1),

        nn.Conv1d(64, 128, 5, stride=1),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Conv1d(128, 256, 3, stride=2),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.MaxPool1d(2, stride=2),

        nn.Flatten(),
        nn.Linear(Data_in_features, Data_out_features),#改动地方  两个矩阵如果出现差异，就改这里，让第二个数和第三个数一致即可，不需要其他操作
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(0.2),

        nn.Linear(64,32),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(32, 1)
    ).to("cuda")

    return net

net1 = Network_declaration(5120, 256)
net2 = Network_declaration(5120, 256)
net3 = Network_declaration(5120, 256)


def Data_training(net, X_train, Y_train, loss_data_title, loss_data_filename, sample_name, folder_path = 'saved_files', lr=0.001):
    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    data_iter = load_array((X_train, Y_train), batch_size, is_train=True)
    ee = 0
    best_loss = 1000
    loss_history = []  # 用于记录每个 epoch 的 loss 值

    print(f'下面开始训练{sample_name}数据')

    for epoch in range(num_epochs):
        for X_, y_ in data_iter:
            X_ = X_.unsqueeze(1)
            yy = net(X_)
            l = loss(yy, y_)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(X_train.unsqueeze(1)), Y_train)
        loss_history.append(l.item())  # 记录当前 epoch 的 loss 值
        print("进行迭代中，", f'epoch的次数为 {epoch + 1}, loss {l:.6f}，最佳loss值为{best_loss:.6f},最佳模型更替在{ee + 1}次')

        if l < best_loss:
            best_loss = l
            file_name = f"best_model--Temporary_{formatted_time}.pth"
            file_path = os.path.join(folder_path, file_name)
            torch.save(net.state_dict(), file_path)
            print('更新了最佳模型')
            ee = epoch
            kk = best_loss
            kk = float(kk)
            kk = round(kk, 4)
        else:
            pass

    # 绘制 loss 值随着训练次数的变化图
    plt.figure(figsize=(4.5, 3.5))
    plt.plot(loss_history, label=f'{sample_name} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title(loss_data_title)
    plt.legend()
    plt.savefig(loss_data_filename, dpi=600)

    return ee, kk, file_path


def Conversion_format_evaluation(Y_test, X_test, net, ee, kk, sample_name, file_path):
    # 将测试集标签 Y_test 转换为 numpy 数组，并将其移动到 CPU 上
    Y_true = Y_test
    Y_true = Y_true.cpu().numpy()

    # 使用神经网络 net 对测试集 X_test 进行预测，并将预测结果转换为 numpy 数组
    Y_pred = net(X_test.unsqueeze(1))
    Y_pred = Y_pred.cpu().detach().numpy()

    # 计算模型的 R2 分数
    r2 = pre.evaluate_model(Y_true, Y_pred, sample_name)
    print('在第', ee, '次时，获取到最佳模型，此次loss值为', kk)

    # 构建新的模型文件名，并重命名模型文件
    cc = f"model__{formatted_time}__R2={r2}_迭代{num_epochs}次({sample_name})"
    new_name = f'{cc}.pth'
    newfile_path = os.path.join(folder_path, new_name)
    os.rename(file_path, newfile_path)
    print(f"模型参数文件已保存为：{new_name}")

    # 绘制散点图并保存
    plt.figure(figsize=(4, 3.5))
    tt1 = min(Y_true)
    tt2 = max(Y_true)
    plt.clf()  # 清空之前的记录
    plt.scatter(Y_true, Y_pred)  # 绘制散点图
    plt.xlim(0.7 * tt1, 1.1 * tt2)
    plt.ylim(0.7 * tt1, 1.1 * tt2)
    plt.text(0.45 * tt2, 0.9 * tt2, 'R\u00b2 =' + str(r2), fontsize=12, ha='center')
    x = np.arange(0, 100, 0.1)  # 均衡线
    y = x
    plt.plot(x, y, '-r')  # 绘制红色线条
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(sample_name)  # 设置标题
    new_name = f'{cc}.jpg'
    file_path = os.path.join(folder_path, new_name)
    plt.savefig(file_path, dpi=600)
    print(f"模型R2图文件已保存为：{new_name}")

ee_AAA, kk_AAA, file_path_AAA = Data_training(net1, X_train_AAA, Y_train_AAA, 'Training Process(AAA)', 'saved_files/(AAA)Training Process.jpg', 'AAA')
plt.clf()  # 清空当前图形
Conversion_format_evaluation(Y_test_AAA, X_test_AAA, net1, ee_AAA, kk_AAA, 'AAA', file_path_AAA)
plt.clf()  # 清空当前图形
ee_BBB, kk_BBB, file_path_BBB = Data_training(net2, X_train_BBB, Y_train_BBB, 'Training Process(BBB)', 'saved_files/(BBB)Training Process.jpg', 'BBB')
plt.clf()  # 清空当前图形
Conversion_format_evaluation(Y_test_BBB, X_test_BBB, net2, ee_BBB, kk_BBB, 'BBB', file_path_BBB)
plt.clf()  # 清空当前图形
ee_CCC, kk_CCC, file_path_CCC = Data_training(net3, X_train_CCC, Y_train_CCC, 'Training Process(CCC)', 'saved_files/(CCC)Training Process.jpg', 'CCC')
plt.clf()  # 清空当前图形
Conversion_format_evaluation(Y_test_CCC, X_test_CCC, net3, ee_CCC, kk_CCC, 'CCC', file_path_CCC)
plt.clf()  # 清空当前图形