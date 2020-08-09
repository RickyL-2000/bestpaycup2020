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
x_df = pd.read_csv(base_dir + '/dataset/dataset4/trainset/train_main.csv')
y_df = pd.read_csv(base_dir + '/dataset/raw_dataset/trainset/train_label.csv')
data_x = np.array(x_df)
data_y = np.array(y_df)

# %%
data_x = data_x[data_x[:, 0].argsort()]
data_y = data_y[data_y[:, 0].argsort()]
data_x = data_x[:, 1:].astype(float)
data_y = data_y[:, 1:].astype(float).reshape(1, -1)[0]

state = np.random.get_state()
np.random.shuffle(data_x)
np.random.set_state(state)
np.random.shuffle(data_y)

# %%
X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size=0.1)

# %%
# 确定 n_estimators
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val)
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'nthread': 4,
    'learning_rate': 0.1,
    'num_leaves': 30,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}

data_train = lgb.Dataset(X_train, y_train)
cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, stratified=False, shuffle=True,
                    metrics='auc', early_stopping_rounds=50, seed=0)

print('best n_estimators:', len(cv_results['auc-mean']))
print('best cv score:', pd.Series(cv_results['auc-mean']).max())

# output:
# best n_estimators: 155
# best cv score: 0.7127048191154473

# %%
# 调参，确定max_depth和num_leaves
params_test1 = {
    'max_depth': range(1, 15, 1),
    'num_leaves': range(10, 150, 5)
}

gsearch1 = GridSearchCV(estimator=lgb.LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
    metrics='auc',
    learning_rate=0.1,
    n_estimators=155,
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

# output
# {'max_depth': 4, 'num_leaves': 30}
# 0.7131046044818581
# {'max_depth': 5, 'num_leaves': 15}
# 0.7141591621140838

# %%
# 确定 min_data_in_leaf 和 max_bin
params_test2 = {
    # 'max_bin': range(5, 256, 10),
    'max_bin': range(200, 300, 10),
    'min_data_in_leaf': range(1, 120, 10)
}

gsearch2 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=155,
                                 max_depth=5,
                                 num_leaves=15,
                                 bagging_fraction=0.8,
                                 feature_fraction=0.8),
    param_grid=params_test2, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch2.fit(X_train, y_train)

print(gsearch2.best_params_)
print(gsearch2.best_score_)

# output
# {'max_bin': 255, 'min_data_in_leaf': 81}
# 0.7161114571663181
# {'max_bin': 230, 'min_data_in_leaf': 61}
# 0.7160718359744372
# ????

# %%
# 确定 feature_fraction, bagging_fraction, bagging_freq
params_test3 = {
    # 'feature_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'feature_fraction': [0.4, 0.5, 0.6, 0.7, 0.8],
    'bagging_fraction': [0.6, 0.7, 0.8, 0.9, 1.0],
    'bagging_freq': range(0, 81, 10)
}

gsearch3 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=155,
                                 max_depth=5,
                                 num_leaves=15,
                                 max_bin=255,
                                 min_data_in_leaf=81),
    param_grid=params_test3, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch3.fit(X_train, y_train)

print(gsearch3.best_params_)
print(gsearch3.best_score_)

# output
# {'bagging_fraction': 0.6, 'bagging_freq': 0, 'feature_fraction': 0.8}
# 0.7161114571663181

# %%
# 确定 lambda_l1 和 lambda_l2
params_test4 = {
    # 'lambda_l1': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    'lambda_l1': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0],
    # 'lambda_l2': [1e-5, 1e-3, 1e-1, 0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    'lambda_l2': [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.0]
}

gsearch4 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=155,
                                 max_depth=5,
                                 num_leaves=15,
                                 max_bin=255,
                                 min_data_in_leaf=81,
                                 bagging_fraction=0.6,
                                 bagging_freq=0,
                                 feature_fraction=0.8),
    param_grid=params_test4, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch4.fit(X_train, y_train)

print(gsearch4.best_params_)
print(gsearch4.best_score_)

# output:
# {'lambda_l1': 1e-05, 'lambda_l2': 1e-05}
# 0.7161114717458197

# {'lambda_l1': 0.0001, 'lambda_l2': 0.0001}
# 0.7161116029568383

# %%
# 确定 min_split_gain
params_test5 = {'min_split_gain': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}

gsearch5 = GridSearchCV(
    estimator=lgb.LGBMClassifier(boosting_type='gbdt',
                                 objective='binary',
                                 metrics='auc',
                                 learning_rate=0.1,
                                 n_estimators=155,
                                 max_depth=5,
                                 num_leaves=15,
                                 max_bin=255,
                                 min_data_in_leaf=81,
                                 bagging_fraction=0.6,
                                 bagging_freq=0,
                                 feature_fraction=0.8,
                                 lambda_l1=0.0001,
                                 lambda_l2=0.0001),
    param_grid=params_test5, scoring='roc_auc', cv=5, n_jobs=-1
)

gsearch5.fit(X_train, y_train)

print(gsearch5.best_params_)
print(gsearch5.best_score_)

# output:
# {'min_split_gain': 0.0}
# 0.7161116029568383

# %%
# 开始正式训练
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'auc',
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'max_depth': 5,
    'num_leaves': 15,
    'max_bin': 255,
    'min_data_in_leaf': 81,
    'bagging_fraction': 0.6,
    'bagging_freq': 0,
    'feature_fraction': 0.8,
    'lambda_l1': 0.0001,
    'lambda_l2': 0.0001,
    'min_split_gain': 0.0
}

model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=val_data, early_stopping_rounds=500)

y_hat = model.predict(X_val)

print('auc: ', roc_auc_score(y_val, y_hat))

# output:
# auc:  0.7111066090506122

# %%
# 查看权重
headers = x_df.columns.tolist()
headers.pop(0)
pd.set_option('display.max_rows', None)
print(pd.DataFrame({
    'column': headers,
    'importance': model.feature_importance().tolist()
}).sort_values(by='importance'))

importance = pd.DataFrame({
    'column': headers,
    'importance': model.feature_importance().tolist()
}).sort_values(by='importance')

importance.to_csv(base_dir + '/models/treemodel/lgb_4_1_weight2.csv', index=False)

# %%
# prediction
test_x = np.array(pd.read_csv(base_dir + '/dataset/dataset4/testset/test_a_main.csv'))
y_df = pd.read_csv(base_dir + '/dataset/raw_dataset/testset/submit_example.csv')

# lgb_1_2 的输出
out_1_2_1 = pd.read_csv(base_dir + '/models/treemodel/output_1_2_1.csv')

test_x = test_x[test_x[:, 0].argsort()]
test_x = test_x[:, 1:].astype(float)

# %%
pred = model.predict(test_x, num_iteration=model.best_iteration)
y_df.loc[:, 'prob'] = pred

# %%
# 将无op或无trans记录的预测值换成out_1_1的
test_df = pd.read_csv(base_dir + '/dataset/dataset4/testset/test_a_main.csv')
for i in range(len(test_df)):
    if test_df['n_op'].loc[i] == 0 or test_df['n_trans'].loc[i] == 0:
        y_df['prob'].loc[i] = out_1_2_1['prob'].loc[i]

# %%
# save
y_df.to_csv(base_dir + '/models/treemodel/output_4_1_1.csv', index=False)
