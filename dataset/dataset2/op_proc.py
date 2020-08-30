# %%
import pandas as pd
import os

base_dir = os.getcwd()

# %%
train_df = pd.read_csv(base_dir + r'/dataset/dataset2/trainset/train_op.csv')
test_a_df = pd.read_csv(base_dir + r'/dataset/dataset2/testset/test_a_op.csv')

train_df = train_df.fillna({'ip': 'ip_nan'})
train_df = train_df.fillna({'ip_3': 'ip_3_nan'})

test_a_df = test_a_df.fillna({'ip': 'ip_nan'})
test_a_df = test_a_df.fillna({'ip_3': 'ip_3_nan'})

train_df = train_df.fillna({'channel': train_df['channel'].mode()[0]})
test_a_df = test_a_df.fillna({'channel': test_a_df['channel'].mode()[0]})

train_df.to_csv(base_dir + '/dataset/dataset2/trainset/train_op.csv', index=False)
test_a_df.to_csv(base_dir + '/dataset/dataset2/testset/test_a_op.csv', index=False)

# %%
test_b_df = pd.read_csv(base_dir + r'/dataset/dataset2/testset/test_b_op.csv')

test_b_df = test_b_df.fillna({'ip': 'ip_nan'})
test_b_df = test_b_df.fillna({'ip_3': 'ip_3_nan'})
test_b_df = test_b_df.fillna({'channel': test_b_df['channel'].mode()[0]})

test_b_df.to_csv(base_dir + '/dataset/dataset2/testset/test_b_op.csv', index=False)

