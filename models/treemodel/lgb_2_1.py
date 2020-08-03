# %%
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
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
# train_data = lgb.Dataset(X_train, label=y_train)
# val_data = lgb.Dataset(X_val, label=y_val)
# params = {
#     'learning_rate': 0.1,
#     'max_depth': 10,
#     'num_leaves': 1000,
#     'objective': 'binary',
#     'subsample': 0.8,
#     'colsample_bytree': 0.8,
#     'metric': 'auc',
#     'n_estimators': 63
#     # 'is_training_metric': True,
# }

# %%
# train
# clf = lgb.train(params, train_data, valid_sets=[val_data])

# %%
# 调参，找出最佳 n_estimators
clf = lgb.cv(params, train_data, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
             early_stopping_rounds=50, seed=0)

print('best n_estimators:', len(clf['auc-mean']))
print('best cv score:', pd.Series(clf['auc-mean']).max())

# %%
# 调参，确定max_depth和num_leaves
params_test1 = {
    'max_depth': range(3, 8, 1),
    'num_leaves': range(5, 100, 5)
}

gsearch1 = GridSearchCV(estimator=lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metrics='auc',
    learning_rate=0.1,
    n_estimators=63,
    max_depth=10,
    bagging_fraction=0.8,
    feature_fraction=0.8
),
    param_grid=params_test1,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1)

gsearch1.fit(X_train, y_train)

print(gsearch1.best_params_)
print(gsearch1.best_score_)
# output:
# {'max_depth': 6, 'num_leaves': 30}
# 0.7111428352044761

# %%
# 确定 min_data_in_leaf 和 max_bin
params_test2 = {
    'max_bin': range(5, 256, 10),
    'min_data_in_leaf': range(1, 102, 10)
}

gsearch2 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=63,
                                 max_depth=6,
                                 num_leaves=30,
                                 bagging_fraction=0.8,
                                 feature_fraction=0.8),
    param_grid=params_test2, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch2.fit(X_train, y_train)

print(gsearch2.best_params_)
print(gsearch2.best_score_)

# output:
# {'max_bin': 15, 'min_data_in_leaf': 71}
# 0.7130982903950965

# %%
# 确定 feature_fraction, bagging_fraction, bagging_freq
params_test3 = {
    'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'bagging_freq': range(0, 81, 10)
}

gsearch3 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=63,
                                 max_depth=6,
                                 num_leaves=30,
                                 max_bin=15,
                                 min_data_in_leaf=71),
    param_grid=params_test3, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch3.fit(X_train, y_train)

print(gsearch3.best_params_)
print(gsearch3.best_score_)

# output
# {'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 0.8}
# 0.7130982903950965

# %%
# 确定 lambda_l1 和 lambda_l2
params_test4 = {'lambda_l1': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
                'lambda_l2': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}

gsearch4 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=63,
                                 max_depth=6,
                                 num_leaves=30,
                                 max_bin=15,
                                 min_data_in_leaf=71,
                                 bagging_fraction=0.6,
                                 bagging_freq=0,
                                 feature_fraction=0.8),
    param_grid=params_test4, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch4.fit(X_train, y_train)

print(gsearch4.best_params_)
print(gsearch4.best_score_)

# output
# {'lambda_l1': 1.0, 'lambda_l2': 0.7}
# 0.7132416453983882

# %%
# 确定 min_split_gain
params_test5 = {'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

gsearch5 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=63,
                                 max_depth=6,
                                 num_leaves=30,
                                 max_bin=15,
                                 min_data_in_leaf=71,
                                 bagging_fraction=0.6,
                                 bagging_freq=0,
                                 feature_fraction=0.8,
                                 lambda_l1=1.0,
                                 lambda_l2=0.7),
    param_grid=params_test5, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch5.fit(X_train, y_train)

print(gsearch5.best_params_)
print(gsearch5.best_score_)

# output
# {'min_split_gain': 0.1}
# 0.7117714487623986

# %%
# 训练
model = lgb.LGBMClassifier(boosting_type='gbdt',
                           objective='binary',
                           metrics='auc',
                           learning_rate=0.01,
                           n_estimators=1000,
                           max_depth=6,
                           num_leaves=30,
                           max_bin=15,
                           min_data_in_leaf=71,
                           bagging_fraction=0.6,
                           bagging_freq=0,
                           feature_fraction=0.8,
                           lambda_l1=1.0,
                           lambda_l2=0.7,
                           min_split_gain=0.1)

model.fit(X_train, y_train)

y_hat = model.predict(X_val)

print('auc: ', roc_auc_score(y_val, y_hat))


# %%
def get_score(pred, lab):
    return roc_auc_score(lab, pred[:, 1])


# %%
# prediction validation
# y_pred = clf.predict(X_val)
# y_pred = np.array([list(x).index(max(x)) for x in y_pred])
# y_val = np.array(y_val)
# score = roc_auc_score(y_val, y_pred)
# print(score)
