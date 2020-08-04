# %%
import pandas as pd
import os

base_dir = os.getcwd()

# %%
train_main_df = pd.read_csv(base_dir + '/dataset/dataset4/trainset/train_main.csv')
test_main_df = pd.read_csv(base_dir + '/dataset/dataset4/testset/test_a_main.csv')
train_label_df = pd.read_csv(base_dir + '/dataset/raw_dataset/trainset/train_label.csv')

train_main_part_df = pd.DataFrame(columns=train_main_df.columns)
test_main_part_df = pd.DataFrame(columns=test_main_df.columns)
train_label_part_df = pd.DataFrame(columns=train_label_df.columns)

n_train = len(train_main_df)
n_test = len(test_main_df)

# %%
for i in range(n_train):
    if i % 1000 == 0:
        print(i)
    if train_main_df['n_op'].loc[i] != 0 and train_main_df['n_trans'].loc[i] != 0:
        train_main_part_df.loc[len(train_main_part_df)] = train_main_df.loc[i]

for i in range(n_test):
    if i % 1000 == 0:
        print(i)
    if test_main_df['n_op'].loc[i] != 0 and test_main_df['n_trans'].loc[i] != 0:
        test_main_part_df.loc[len(test_main_part_df)] = test_main_df.loc[i]

# %%
train_main_df = train_main_df.sort_values('user')
for i in range(n_train):
    if i % 1000 == 0:
        print(i)
    if train_main_df['n_op'].loc[i] != 0 and train_main_df['n_trans'].loc[i] != 0:
        train_label_part_df.loc[len(train_label_part_df)] = train_label_df.loc[i]

# %%
train_main_part_df.to_csv(base_dir + '/dataset/dataset4/trainset/train_main_part.csv', index=False)
test_main_part_df.to_csv(base_dir + '/dataset/dataset4/testset/test_a_main_part.csv', index=False)

# %%
train_label_part_df.to_csv(base_dir + '/dataset/dataset4/trainset/train_label_part.csv', index=False)
