# %%
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.metrics import roc_auc_score

# %%
base_dir = os.getcwd()
x_df = pd.read_csv(base_dir + '/dataset/dataset1/trainset/train_base2.csv')
y_df = pd.read_csv(base_dir + '/dataset/raw_dataset/trainset/train_label.csv')
data_x = np.array(x_df)
# train_x = np.delete(train_x, 0, axis=1)
data_y = np.array(y_df)

# %%
# 将x与y对应
data_x = data_x[data_x[:, 0].argsort()]
data_y = data_y[data_y[:, 0].argsort()]

# %%
# 归一化
n, l = data_x.shape
for j in range(l):
    meanVal = np.mean(data_x[:, j])
    stdVal = np.std(data_x[:, j])
    data_x[:, j] = (data_x[:, j] - meanVal) / stdVal

# %%
# 打乱数据
state = np.random.get_state()
np.random.shuffle(data_x)
np.random.set_state(state)
np.random.shuffle(data_y)

# %%
# 抽取验证集
train_x = data_x[:int(0.9 * n), 1:]
train_y = data_y[:int(0.9 * n), 1:]
val_x = data_x[int(0.9 * n):, 1:]
val_y = data_y[int(0.9 * n):, 1:]
train_x = train_x.astype(float)
train_y = train_y.astype(float).reshape(1, -1)[0]
val_x = val_x.astype(float)
val_y = val_y.astype(float).reshape(1, -1)[0]


# %%
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(123, 2)

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


# %%
def test(pred, lab):
    t = pred.max(-1)[1] == lab
    return torch.mean(t.float())


# %%
net = LR()
criterion = nn.CrossEntropyLoss()
optm = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
epochs = 2000

# %%
# train
for i in range(epochs):
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
        accur = test(val_out, val_lab)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i + 1, loss.item(), accur))

# %%
# 测试集
test_x = np.array(pd.read_csv(base_dir + '/dataset/dataset1/testset/test_a_base2.csv'))
y_df = pd.read_csv(base_dir + '/dataset/raw_dataset/testset/submit_example.csv')

# %%
test_x = test_x[test_x[:, 0].argsort()]
n_t, l_t = test_x.shape
for j in range(l_t):
    meanVal = np.mean(test_x[:, j])
    stdVal = np.std(test_x[:, j])
    test_x[:, j] = (test_x[:, j] - meanVal) / stdVal

# %%
test_in = torch.from_numpy(test_x).float()
test_out = net(test_in)


# %%
"""---------------------以下为试水部分-------------------------"""

arr = np.array([[1, 2, 3],
                [3, 2, 1],
                [2, 1, 3]])
arr = arr[arr[:, 0].argsort()]
print(arr)

# %%
arr = np.array([[1], [2], [3]])
arr = arr.reshape(1, -1)
print(arr[0])

# %%
print(y_hat.reshape(1, -1)[0])
print(y)
