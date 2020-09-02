# %%
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
import pandas as pd

# %%
# base_dir = os.getcwd()
base_dir = '/Users/jason/bestpaycup2020'
x_df = pd.read_csv(base_dir + '/dataset/dataset1/trainset/train_base.csv')
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
    'max_depth': 10,
    'num_leaves': 1000,
    'objective': 'binary',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'metric': 'auc',
    'n_estimators': 63
    # 'is_training_metric': True,
}

# %%
# train
# clf = lgb.train(params, train_data, valid_sets=[val_data])

# %%
# 调参，找出最佳 n_estimators
# n_estimators: 47
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
    n_estimators=47,
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
# 
# 调参，确定max_depth和num_leaves...
# {'max_depth': 7, 'num_leaves': 20}
# 0.6860003095778059


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
                                 n_estimators=47,
                                 max_depth=7,
                                 num_leaves=20,
                                 bagging_fraction=0.8,
                                 feature_fraction=0.8),
    param_grid=params_test2, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch2.fit(X_train, y_train)

print(gsearch2.best_params_)
print(gsearch2.best_score_)

# output:
# {'max_bin': 5, 'min_data_in_leaf': 51}
# 0.6849877309744751

# %%
# 确定 feature_fraction, bagging_fraction, bagging_freq
params_test3 = {
    'feature_fraction': [0.3, 0.2, 0.5, 0.4],
    'bagging_fraction': [0.3, 0.4, 0.5, 0.6, 0.7],
    'bagging_freq': range(0, 40, 10)
}

gsearch3 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=47,
                                 max_depth=7,
                                 num_leaves=20,
                                 max_bin=5,
                                 min_data_in_leaf=51),
    param_grid=params_test3, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch3.fit(X_train, y_train)

print(gsearch3.best_params_)
print(gsearch3.best_score_)

# output
# {'bagging_fraction': 0.3, 'bagging_freq': 0, 'feature_fraction': 0.5}
# 0.6856456152180354

# %%
# 确定 lambda_l1 和 lambda_l2
params_test4 = {'lambda_l1': [0.7, 0.8, 0.9, 1.0, 1.2 , 1.3],
                'lambda_l2': [0.4, 0.5, 0.6]}

gsearch4 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=47,
                                 max_depth=7,
                                 num_leaves=20,
                                 max_bin=5,
                                 min_data_in_leaf=51,
                                 bagging_fraction=0.3,
                                 bagging_freq=0,
                                 feature_fraction=0.5),
    param_grid=params_test4, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch4.fit(X_train, y_train)

print(gsearch4.best_params_)
print(gsearch4.best_score_)

# output
# {'lambda_l1': 0.9, 'lambda_l2': 0.5}
# 0.6869625897803501

# %%
# 确定 min_split_gain
params_test5 = {'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

gsearch5 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=47,
                                 max_depth=7,
                                 num_leaves=20,
                                 max_bin=5,
                                 min_data_in_leaf=51,
                                 bagging_fraction=0.3,
                                 bagging_freq=0,
                                 feature_fraction=0.5,
                                 lambda_l1=0.9,
                                 lambda_l2=0.5),
    param_grid=params_test5, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch5.fit(X_train, y_train)

print(gsearch5.best_params_)
print(gsearch5.best_score_)

# output
# {'min_split_gain': 0.0}
# 0.6869625897803501

# %%
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'auc',
    'learning_rate': 0.05,
    'n_estimators': 10000,
    'max_depth': 7,
    'num_leaves': 20,
    'max_bin': 5,
    'min_data_in_leaf': 51,
    'bagging_fraction': 0.3,
    'bagging_freq': 0,
    'feature_fraction': 0.5,
    'lambda_l1': 0.9,
    'lambda_l2': 0.5,
    'min_split_gain': 0.0
}

model2 = lgb.train(params, train_data, num_boost_round=1000, valid_sets=val_data, early_stopping_rounds=500)

y_hat = model2.predict(X_val)

print('auc: ', roc_auc_score(y_val, y_hat))

# %%
# 查看权重
headers = x_df.columns.tolist()
headers.pop(0)
pd.set_option('display.max_rows', None)
print(pd.DataFrame({
    'column': headers,
    'importance': model2.feature_importance().tolist()
}).sort_values(by='importance'))

importance = pd.DataFrame({
    'column': headers,
    'importance': model2.feature_importance().tolist()
}).sort_values(by='importance')

importance.to_csv(base_dir + '/models/treemodel/lgb_2_1_weight2.csv', index=False)

# %%
# model prediction
# 测试集
test_x = np.array(pd.read_csv(base_dir + '/dataset/dataset1/testset/test_b_base.csv'))
y_df = pd.read_csv(base_dir + '/dataset/raw_dataset/testset/submit_example_b.csv')

test_x = test_x[test_x[:, 0].argsort()]
test_x = test_x[:, 1:].astype(float)

# %%
pred = model2.predict(test_x, num_iteration=model2.best_iteration)
y_df.loc[:, 'prob'] = pred

# %%
# 将无op或无trans记录的预测值换成out_1_1的
test_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_b_main.csv')
for i in range(len(test_df)):
    if test_df['n_op'].loc[i] == 0 or test_df['n_trans'].loc[i] == 0:
        y_df['prob'].loc[i] = out_1_1['prob'].loc[i]

# %%

y_df.to_csv(base_dir + '/models/treemodel/output_1_3_1.csv', index=False)

# %%
# prediction validation
# y_pred = clf.predict(X_val)
# y_pred = np.array([list(x).index(max(x)) for x in y_pred])
# y_val = np.array(y_val)
# score = roc_auc_score(y_val, y_pred)
# print(score)
