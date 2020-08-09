# 树模型

## lightGBM

### lgb_1_1

第一个对lgb和rf进行尝试，白给

### lgb_1_2

仅对base进行自动化调参，结果0.66

### lgb_2_1_1

自动化调参后的第一次原生lgbm接口实验

参数：
```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'auc',
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'max_depth': 6,
    'num_leaves': 30,
    'max_bin': 15,
    'min_data_in_leaf': 71,
    'bagging_fraction': 0.6,
    'bagging_freq': 0,
    'feature_fraction': 0.8,
    'lambda_l1': 1.0,
    'lambda_l2': 0.7,
    'min_split_gain': 0.1
}

model2 = lgb.train(params, train_data, num_boost_round=1000, valid_sets=val_data, early_stopping_rounds=500)
```

输出：
```shell script
[1853]	valid_0's auc: 0.721527
Early stopping, best iteration is:
[1353]	valid_0's auc: 0.722401
auc:  0.7224009643762731
```

### lgb_2_1_2

把树稍微调深调大了些

参数
```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'auc',
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'max_depth': 9,
    'num_leaves': 150,
    'max_bin': 15,
    'min_data_in_leaf': 71,
    'bagging_fraction': 0.6,
    'bagging_freq': 0,
    'feature_fraction': 0.8,
    'lambda_l1': 1.0,
    'lambda_l2': 0.7,
    'min_split_gain': 0.1
}

model2 = lgb.train(params, train_data, num_boost_round=1000, valid_sets=val_data, early_stopping_rounds=500)
```

输出
```shell script
Early stopping, best iteration is:
[773]	valid_0's auc: 0.722797
auc:  0.7227970144979746
```

### lgb_2_1_2

把树调得更深更大了，效果更差了，感觉需要修改数据集了

```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'auc',
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'max_depth': 10,
    'num_leaves': 1000,
    'max_bin': 15,
    'min_data_in_leaf': 71,
    'bagging_fraction': 0.6,
    'bagging_freq': 0,
    'feature_fraction': 0.8,
    'lambda_l1': 1.0,
    'lambda_l2': 0.7,
    'min_split_gain': 0.1
}

model2 = lgb.train(params, train_data, num_boost_round=1000, valid_sets=val_data, early_stopping_rounds=500)
```

输出
```shell script
Early stopping, best iteration is:
[1033]	valid_0's auc: 0.719666
auc:  0.7196664865389145
```

### lgb_4_1_1

用了dataset4，是精简版的dataset2。跑分结果和lgb_2_1不相上下。没什么进展。

### lgb_4_2

将dataset4中的train_main中的op和trans为0的记录全部删除，做成train_main_part后训练的lgb（相当于分了两拨人分别训练）。

效果奇差，只有0.5+


## XGBoost

### xgb_4_1

对xgb进行尝试，初步感觉精度和lgb差不多
