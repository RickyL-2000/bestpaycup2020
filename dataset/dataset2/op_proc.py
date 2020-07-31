# %%
import pandas as pd
import os

base_dir = os.getcwd()

# %%
# 处理ip的哑变量
train_df = pd.read_csv(base_dir + r'/dataset/dataset2/trainset/train_op.csv')
test_df = pd.read_csv(base_dir + r'/dataset/dataset2/testset/test_a_op.csv')

train_df = train_df.fillna({'ip': 'ip_nan'})
train_df = train_df.fillna({'ip_3': 'ip_3_nan'})

test_df = test_df.fillna({'ip': 'ip_nan'})
test_df = test_df.fillna({'ip_3': 'ip_3_nan'})

train_df.to_csv(base_dir + '/dataset/dataset2/trainset/train_op.csv', index=False)
test_df.to_csv(base_dir + '/dataset/dataset2/testset/test_a_op.csv', index=False)

# %%
# 处理channel的众数赋值
train_df = pd.read_csv(base_dir + r'/dataset/dataset2/trainset/train_op.csv')
test_df = pd.read_csv(base_dir + r'/dataset/dataset2/testset/test_a_op.csv')

train_df = train_df.fillna({'channel': train_df['channel'].mode()[0]})
test_df = test_df.fillna({'channel': test_df['channel'].mode()[0]})

print(train_df['channel'].isnull().sum())
print(test_df['channel'].isnull().sum())

# %%
train_df.to_csv(base_dir + '/dataset/dataset2/trainset/train_op.csv', index=False)
test_df.to_csv(base_dir + '/dataset/dataset2/testset/test_a_op.csv', index=False)