# %%
import lightgbm as lgb
from sklearn import datasets
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
from xgboost import XGBClassifier

base_dir='/Users/jason/bestpaycup2020'
x_df = pd.read_csv(base_dir + '/dataset/dataset1/trainset/train_base.csv')
y_df = pd.read_csv(base_dir + '/dataset/raw_dataset/trainset/train_label.csv')
data_x = np.array(x_df)
# train_x = np.delete(train_x, 0, axis=1)
data_y = np.array(y_df)

data_x = data_x[data_x[:, 0].argsort()]
data_y = data_y[data_y[:, 0].argsort()]
data_x = data_x[:, 1:].astype(float)
data_y = data_y[:, 1:].astype(float).reshape(1, -1)[0]

# 归一化
n, l = data_x.shape
for j in range(l):
    meanVal = np.mean(data_x[:, j])
    stdVal = np.std(data_x[:, j])
    data_x[:, j] = (data_x[:, j] - meanVal) / stdVal

# 打乱数据
state = np.random.get_state()
np.random.shuffle(data_x)
np.random.set_state(state)
np.random.shuffle(data_y)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)

# %%
# lgb

lgbm_model=LGBMClassifier(boosting_type='gbdt', num_leaves=300, max_depth=-1, learning_rate=0.1, n_estimators=50, subsample_for_bin=200000, objective='binary')
lgbm_model.fit(x_train,y_train)
#用建立好的lightbm模型运用到训练集和测试集上，进行预测
y_train_pred = lgbm_model.predict(x_train)
y_test_pred = lgbm_model.predict(x_test)

print('训练集：{:.4f}'.format(roc_auc_score(y_train, y_train_pred)))
print('测试集：{:.4f}'.format(roc_auc_score(y_test, y_test_pred)))

# %%
clf = RandomForestClassifier(n_estimators=300, max_features=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False, class_weight=None)
s=clf.fit(x_train, y_train)

y_train_pred = lgbm_model.predict(x_train)
y_rd_pred = s.predict(x_test)
print('训练集：{:.4f}'.format(roc_auc_score(y_train, y_train_pred)))
print('测试集：{:.4f}'.format(roc_auc_score(y_test, y_rd_pred)))





# %%
