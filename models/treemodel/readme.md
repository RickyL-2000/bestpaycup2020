# 树模型

## lightGBM

### lgb_1_1

第一个对lgb和rf进行尝试，白给

### lgb_1_2

仅对**dataset4**(zs!!!!)的base进行自动化调参，结果0.66

### lgb_1_3

仅对base进行自动化调参，但是是对test_b

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

### output_2_1_4.csv

将output_2_1_2进行0.3阈值放缩后的结果。

分数 与output_2_1_2一样

### output_2_1_5.csv

将output_2_1_2进行0.23阈值放缩后的结果。

分数与上一条一样

### output_2_1_6

在上一条的基础上将概率变成离散值

分数：0.61

**再也不许交离散值！！！！！！！！**

### lgb_2_2_1

尝试对test_b进行预测

### out_2_2_1

对test_b进行预测，分数只有0.63

### out_2_2_3

对out_2_2_1进行以0.23为界拉伸，分数几乎没有变化

### lgb_4_1_1

用了dataset4，是精简版的dataset2。跑分结果和lgb_2_1不相上下。没什么进展。

### lgb_4_1_3 & lgb_4_1_4

```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'auc',
    'learning_rate': 0.02,
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

model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=val_data, early_stopping_rounds=300)
```

稍微调了下早停和学习率，分数提升0.007

```shell script
[985]	valid_0's auc: 0.729851
auc:  0.729850676255246
```

### lgb_4_1_5

本地突破0.73！

```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'auc',
    'learning_rate': 0.02,
    'n_estimators': 10000,
    'max_depth': 7,
    'num_leaves': 15,
    'max_bin': 255,
    'min_data_in_leaf': 71,
    'bagging_fraction': 0.6,
    'bagging_freq': 0,
    'feature_fraction': 0.8,
    'lambda_l1': 0.0001,
    'lambda_l2': 0.0001,
    'min_split_gain': 0.0
}

model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=val_data, early_stopping_rounds=300)
```

```shell script
[719]	valid_0's auc: 0.730432
auc:  0.7304322519376283
```

### lgb_4_1_6

```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'auc',
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'max_depth': 7,
    'num_leaves': 31,
    'max_bin': 255,
    'min_data_in_leaf': 71,
    'bagging_fraction': 0.6,
    'bagging_freq': 0,
    'feature_fraction': 0.8,
    'lambda_l1': 0.0001,
    'lambda_l2': 0.0001,
    'min_split_gain': 0.0
}

model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=val_data, early_stopping_rounds=200)

```

```shell script
[1163]	valid_0's auc: 0.731604
auc:  0.7316037799068713
```

### lgb_4_7

```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metrics': 'auc',
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'max_depth': 7,
    'num_leaves': 61,
    'max_bin': 255,
    'min_data_in_leaf': 71,
    'bagging_fraction': 0.6,
    'bagging_freq': 0,
    'feature_fraction': 0.8,
    'lambda_l1': 0.0001,
    'lambda_l2': 0.0001,
    'min_split_gain': 0.0
}

model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=val_data, early_stopping_rounds=200)
```

```shell script
[849]	valid_0's auc: 0.731436
auc:  0.7314362478172962
```

### lgb_4_2

将dataset4中的train_main中的op和trans为0的记录全部删除，做成train_main_part后训练的lgb（相当于分了两拨人分别训练）。

效果奇差，只有0.5+


## XGBoost

### xgb_4_1

对xgb进行尝试，初步感觉精度和lgb差不多
