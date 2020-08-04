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
x_df = pd.read_csv(base_dir + '/dataset/dataset4/trainset/train_base.csv')
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
clf = lgb.cv(params, train_data, num_boost_round=1000, nfold=5, stratified=False, shuffle=True, metrics='auc',
             early_stopping_rounds=50, seed=0)

print('best n_estimators:', len(clf['auc-mean']))
print('best cv score:', pd.Series(clf['auc-mean']).max())

# best:54
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
    n_estimators=54,
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
# {'max_depth': 7, 'num_leaves': 35}
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
                                 n_estimators=54,
                                 max_depth=7,
                                 num_leaves=35,
                                 bagging_fraction=0.8,
                                 feature_fraction=0.8),
    param_grid=params_test2, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch2.fit(X_train, y_train)

print(gsearch2.best_params_)
print(gsearch2.best_score_)

# output:
# {'max_bin': 45, 'min_data_in_leaf': 81}
# 0.7130982903950965

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
                                 n_estimators=54,
                                 max_depth=7,
                                 num_leaves=35,
                                 max_bin=45,
                                 min_data_in_leaf=81),
    param_grid=params_test3, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch3.fit(X_train, y_train)

print(gsearch3.best_params_)
print(gsearch3.best_score_)

# output
# {'bagging_fraction': 0.3, 'bagging_freq': 0, 'feature_fraction': 0.4}
# 0.7130982903950965

# %%
# 确定 lambda_l1 和 lambda_l2
params_test4 = {'lambda_l1': [0.9, 1.0, 1.2 , 1.3, 1.4,1.5,1.6],
                'lambda_l2': [0.4, 0.5, 0.6]}

gsearch4 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=54,
                                 max_depth=7,
                                 num_leaves=35,
                                 max_bin=45,
                                 min_data_in_leaf=81,
                                 bagging_fraction=0.3,
                                 bagging_freq=0,
                                 feature_fraction=0.4),
    param_grid=params_test4, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch4.fit(X_train, y_train)

print(gsearch4.best_params_)
print(gsearch4.best_score_)

# output
# {'lambda_l1': 1.0, 'lambda_l2': 0.5}
# 0.7132416453983882

# %%
# 确定 min_split_gain
params_test5 = {'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

gsearch5 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=54,
                                 max_depth=7,
                                 num_leaves=35,
                                 max_bin=45,
                                 min_data_in_leaf=81,
                                 bagging_fraction=0.3,
                                 bagging_freq=0,
                                 feature_fraction=0.4,
                                 lambda_l1=1.0,
                                 lambda_l2=0.5),
    param_grid=params_test5, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch5.fit(X_train, y_train)

print(gsearch5.best_params_)
print(gsearch5.best_score_)

# output
# {'min_split_gain': 0.0}
# 0.7117714487623986

# %%
# 训练 1
"""
这个训练的效果不好，只有0.53的分数
"""
model1 = lgb.LGBMClassifier(boosting_type='gbdt',
                            objective='binary',
                            metrics='auc',
                            learning_rate=0.01,
                            n_estimators=10000,
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

model1.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=500)

y_hat = model1.predict(X_val)

print('auc: ', roc_auc_score(y_val, y_hat))

# %%
# 查看权重
headers = x_df.columns.tolist()
headers.pop(0)
pd.set_option('display.max_rows', None)
print(pd.DataFrame({
    'column': headers,
    'importance': model.feature_importances_,
}).sort_values(by='importance'))

importance = pd.DataFrame({
    'column': headers,
    'importance': model.feature_importances_,
}).sort_values(by='importance')

importance.to_csv(base_dir + '/models/treemodel/lgb_2_1_weight1.csv', index=False)

# %%
# 训练 2
"""
这个训练效果很好，本地0.72，上传后0.67（添加output1_1_1后）
"""
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'auc',
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'max_depth': 7,
    'num_leaves': 35,
    'max_bin': 45,
    'min_data_in_leaf': 81,
    'bagging_fraction': 0.3,
    'bagging_freq': 0,
    'feature_fraction': 0.4,
    'lambda_l1': 1.0,
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
# model2 prediction
# 测试集
test_x = np.array(pd.read_csv(base_dir + '/dataset/dataset4/testset/test_a_base.csv'))
y_df = pd.read_csv(base_dir + '/dataset/raw_dataset/testset/submit_example.csv')

# lr_1_1 的输出
out_1_1 = pd.read_csv(base_dir + '/models/logistic_regression/output_1_1_1.csv')

test_x = test_x[test_x[:, 0].argsort()]
test_x = test_x[:, 1:].astype(float)

# %%
pred = model2.predict(test_x, num_iteration=model2.best_iteration)
y_df.loc[:, 'prob'] = pred

# %%
# 将无op或无trans记录的预测值换成out_1_1的
test_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_main.csv')
for i in range(len(test_df)):
    if test_df['n_op'].loc[i] == 0 or test_df['n_trans'].loc[i] == 0:
        y_df['prob'].loc[i] = out_1_1['prob'].loc[i]

# %%

y_df.to_csv(base_dir + '/models/treemodel/output_lgb_base.csv', index=False)

# %%
# prediction validation
# y_pred = clf.predict(X_val)
# y_pred = np.array([list(x).index(max(x)) for x in y_pred])
# y_val = np.array(y_val)
# score = roc_auc_score(y_val, y_pred)
# print(score)
