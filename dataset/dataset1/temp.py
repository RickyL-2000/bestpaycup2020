# %%
import pandas as pd
import os
base_dir = r'D:\RickyLi\Documents\浙江大学\学习\比赛\翼支付杯2020\bestpaycup2020'

# %%
test_df = pd.read_csv(base_dir + '/dataset/dataset1/testset/test_a_base.csv')
pd.set_option('display.max_rows', None)
print(test_df.isnull().sum())
# test_df['provider'].fillna(test_df['provider'].mode()[0], inplace=True)
# test_df.to_csv(base_dir + '/dataset/dataset1/testset/test_a_base.csv', index=False)

# %%
train_df = pd.read_csv(base_dir + '/dataset/dataset1/trainset/train_base.csv')
print(train_df.isnull().sum())

# %%
