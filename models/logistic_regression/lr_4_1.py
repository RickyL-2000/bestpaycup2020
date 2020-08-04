# %%
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from sklearn.metrics import roc_auc_score

# %%
base_dir = os.getcwd()
# base_dir = '/Users/jason/bestpaycup2020'
x_df = pd.read_csv(base_dir + '/dataset/dataset4/trainset/train_main.csv')
y_df = pd.read_csv(base_dir + '/dataset/raw_dataset/trainset/train_label.csv')
data_x = np.array(x_df)
# train_x = np.delete(train_x, 0, axis=1)
data_y = np.array(y_df)

# %%
# 将x与y对应，并做预处理
data_x = data_x[data_x[:, 0].argsort()]
data_y = data_y[data_y[:, 0].argsort()]
data_x = data_x[:, 1:].astype(float)
data_y = data_y[:, 1:].astype(float).reshape(1, -1)[0]

# %%
# 归一化
n, l = data_x.shape
for j in range(l):
    meanVal = np.mean(data_x[:, j])
    stdVal = np.std(data_x[:, j])
    if stdVal != 0 and not np.all(meanVal == 0.0):
        data_x[:, j] = (data_x[:, j] - meanVal) / stdVal

# %%
# 打乱数据
state = np.random.get_state()
np.random.shuffle(data_x)
np.random.set_state(state)
np.random.shuffle(data_y)


# %%
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


# %%
def get_score(pred, lab):
    pred = F.softmax(pred, dim=1).detach().numpy()
    lab = lab.detach().numpy()
    try:
        return roc_auc_score(lab, pred[:, 1])
    except:
        print(lab)
        print(pred[:, 1])


# %%
net = LR()

# %%
criterion = nn.CrossEntropyLoss()
optm = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
epochs = 30000
group_n = 10
group_size = epochs // group_n

# %%
# train
for j in range(group_n):
    # 交叉验证
    train_x = np.concatenate((data_x[:j*group_size, :], data_x[(j+1)*group_size:, :]), axis=0)
    train_y = np.append(data_y[:j*group_size], data_y[(j+1)*group_size:])
    val_x = data_x[j*group_size: (j+1)*group_size, :]
    val_y = data_y[j*group_size: (j+1)*group_size]
    for i in range(j * group_size, (j+1) * group_size):
        net.train()
        x = torch.from_numpy(train_x).float()
        y = torch.from_numpy(train_y).long()
        y_hat = net(x)
        loss = criterion(y_hat, y)
        optm.zero_grad()
        loss.backward()
        optm.step()
        if (i+1) % 100 == 0:
            net.eval()
            val_in = torch.from_numpy(val_x).float()
            val_lab = torch.from_numpy(val_y).long()
            val_out = net(val_in)
            score = get_score(val_out, val_lab)
            print("Epoch:{}, Loss:{:.4f}, Score：{:.2f}".format(i + 1, loss.item(), score))

# %%
# 保存模型 （可跳过）
torch.save(net, base_dir + '/models/logistic_regression/lr_4_1.pt')

# %%
# 载入模型（可跳过）
net = torch.load(base_dir + '/models/logistic_regression/lr_4_1.pt')
net.eval()

# %%
# 测试集
test_x = np.array(pd.read_csv(base_dir + '/dataset/dataset4/testset/test_a_main.csv'))
y_df = pd.read_csv(base_dir + '/dataset/raw_dataset/testset/submit_example.csv')

# %%
# lgb_1_2 的输出
out_1_2_1 = pd.read_csv(base_dir + '/models/treemodel/output_1_2_1.csv')

# %%
test_x = test_x[test_x[:, 0].argsort()]
test_x = test_x[:, 1:].astype(float)
n_t, l_t = test_x.shape
for j in range(l_t):
    meanVal = np.mean(test_x[:, j])
    stdVal = np.std(test_x[:, j])
    test_x[:, j] = (test_x[:, j] - meanVal) / stdVal

# %%
test_in = torch.from_numpy(test_x).float()

# %%
test_out = net(test_in)
pred = F.softmax(test_out, dim=1).detach().numpy()[:, 1]
y_df.loc[:, 'prob'] = pred

# %%
# 将无op或无trans记录的预测值换成out_1_2_1的
test_df = pd.read_csv(base_dir + '/dataset/dataset4/testset/test_a_main.csv')
for i in range(len(test_df)):
    if test_df['n_op'].loc[i] == 0 or test_df['n_trans'].loc[i] == 0:
        y_df['prob'].loc[i] = out_1_2_1['prob'].loc[i]

# %%
# 保存
y_df.to_csv(base_dir + '/models/logistic_regression/output_4_1_1.csv', index=False)

