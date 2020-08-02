# %%
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import numpy as np
import os
import pandas as pd

# %%
base_dir = os.getcwd()
# base_dir = '/Users/jason/bestpaycup2020'
x_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_main.csv')
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
# n, l = data_x.shape
# for j in range(l):
#     meanVal = np.mean(data_x[:, j])
#     stdVal = np.std(data_x[:, j])
#     if stdVal != 0 and not np.all(meanVal == 0.0):
#         data_x[:, j] = (data_x[:, j] - meanVal) / stdVal

# %%
# 打乱数据
state = np.random.get_state()
np.random.shuffle(data_x)
np.random.set_state(state)
np.random.shuffle(data_y)

# %%
X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.1)

# %%
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)
params = {
    'learning_rate': 0.1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.2,
    'max_depth': 4,
    'objective': 'multiclass',
    'num_class': 2
}

# %%
# train
clf = lgb.train(params, train_data, valid_sets=[val_data])


# %%
def get_score(pred, lab):
    return roc_auc_score(lab, pred[:, 1])


# %%
# prediction validation
y_pred = clf.predict(X_val)
y_pred = np.array([list(x).index(max(x)) for x in y_pred])
y_val = np.array(y_val)
score = roc_auc_score(y_val, y_pred)
print(score)

