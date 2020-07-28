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
train_y = train_y.astype(float).reshape(1, -1)
val_x = val_x.astype(float)
val_y = val_y.astype(float).reshape(1, -1)


# %%
class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.fc = nn.Linear(123, 1)

    def forward(self, x):
        out = self.fc(x)
        out = torch.sigmoid(out)
        return out


# %%
def get_score(pred, lab):
    return roc_auc_score(pred, lab)


# %%
net = LR()
criterion = nn.CrossEntropyLoss()
optm = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
epochs = 1000

# %%
# train
for i in range(epochs):
    net.train()
    x = torch.from_numpy(train_x).float()
    y = torch.from_numpy(train_y).long()
    y_hat = net(x)
    loss = criterion(y_hat.reshape(1, -1), y)
    optm.zero_grad()
    loss.backward()
    optm.step()
    if (i+1) % 100 == 0:
        net.eval()
        test_in = torch.from_numpy(val_x).float()
        test_l = torch.from_numpy(val_y).long()
        test_out = net(test_in)
        score = get_score(test_out.detach().numpy(), test_in.detach().numpy())
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(i + 1, loss.item(), score))


# %%
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
