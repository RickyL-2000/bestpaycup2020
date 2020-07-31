# %%
import pandas as pd
import os
import re

base_dir = os.getcwd()


# %%
def user2int(entry):
    level = re.search("^(Train_ /TestA_)([0-9]+)", entry)
    if level:
        return int(level.group(2))
    return entry


def tm2int(entry):
    days = int(entry[:entry.find('days')])
    flag = org.find(':')
    hours = int(entry[entry.find('s') + 2: flag])
    minutes = int(entry[flag + 1: entry.find(':', flag + 1)])
    seconds = float(entry[entry.find(':', flag + 1) + 1:])
    return ((24 * days + hours) * 60 + minutes) * 60 + seconds


# %%
train_df = pd.read_csv(base_dir + r'/dataset/raw_dataset/trainset/train_trans.csv')
n = len(train_df)
for i in range(n):
    if i % 1000 == 0:
        print(i)
    train_df['user'].loc[i] = train_df['user'].loc[i][6:]
train_df = train_df.fillna({'tunnel_in': 'tunnel_in_nan'})
t_out_mode = train_df['tunnel_out'].mode()[0]
train_df = train_df.fillna({'tunnel_out': t_out_mode})
train_df = train_df.fillna({'type2': 'type2_nan'})
train_df = train_df.fillna({'ip': 'ip_nan'})
train_df = train_df.fillna({'ip_3': 'ip_3_nan'})
for i in range(n):
    if i % 1000 == 0:
        print(i)
    org = train_df.loc[i, 'tm_diff']
    days = int(org[:org.find('days')])
    flag = org.find(':')
    hours = int(org[org.find('s') + 2: flag])
    minutes = int(org[flag + 1: org.find(':', flag + 1)])
    seconds = float(org[org.find(':', flag + 1) + 1:])
    train_df['tm_diff'].loc[i] = ((24 * days + hours) * 60 + minutes) * 60 + seconds

train_df.to_csv(base_dir + '/dataset/dataset2/trainset/train_trans.csv', index=False)

# %%
test_df = pd.read_csv(base_dir + r'/dataset/raw_dataset/testset/test_a_trans.csv')
n = len(test_df)
for i in range(n):
    if i % 1000 == 0:
        print(i)
    test_df['user'].loc[i] = test_df['user'].loc[i][6:]
test_df = test_df.fillna({'tunnel_in': 'tunnel_in_nan'})
t_out_mode = test_df['tunnel_out'].mode()[0]
test_df = test_df.fillna({'tunnel_out': t_out_mode})
test_df = test_df.fillna({'type2': 'type2_nan'})
test_df = test_df.fillna({'ip': 'ip_nan'})
test_df = test_df.fillna({'ip_3': 'ip_3_nan'})
for i in range(n):
    if i % 1000 == 0:
        print(i)
    org = test_df.loc[i, 'tm_diff']
    days = int(org[:org.find('days')])
    flag = org.find(':')
    hours = int(org[org.find('s') + 2: flag])
    minutes = int(org[flag + 1: org.find(':', flag + 1)])
    seconds = float(org[org.find(':', flag + 1) + 1:])
    test_df['tm_diff'].loc[i] = ((24 * days + hours) * 60 + minutes) * 60 + seconds

test_df.to_csv(base_dir + '/dataset/dataset2/testset/test_a_trans.csv', index=False)


