# %%
import pandas as pd
import os

base_dir = os.getcwd()

# %%
train_df = pd.read_csv(base_dir + r'/dataset/dataset2/trainset/train_op.csv')
test_df = pd.read_csv(base_dir + r'/dataset/dataset2/testset/test_a_op.csv')

train_df = train_df.fillna({'ip': 'ip_nan'})
train_df = train_df.fillna({'ip_3': 'ip_3_nan'})

test_df = test_df.fillna({'ip': 'ip_nan'})
test_df = test_df.fillna({'ip_3': 'ip_3_nan'})

train_df.to_csv(base_dir + '/dataset/dataset2/trainset/train_op.csv', index=False)
test_df.to_csv(base_dir + '/dataset/dataset2/testset/test_a_op.csv', index=False)

