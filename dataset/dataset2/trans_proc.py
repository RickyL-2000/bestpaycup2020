# %%
import pandas as pd
import os
import re

base_dir = os.getcwd()


# %%
def trans_proc(df, isTrain=True):
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

    n = len(df)
    for i in range(n):
        df.loc[i, 'user'] = df.loc[i, 'user'][6:]
    df = df.fillna({'tunnel_in': 'tunnel_in_nan'}, inplace=True)
    df['tunnel_out'].fillna(df['tunnel_out'].mode()[0], inplace=True)
    df['type2'].fillna('type2_nan', inplace=True)
    for i in range(n):
        org = df.loc[i, 'tm_diff']
        days = int(org[:org.find('days')])
        flag = org.find(':')
        hours = int(org[org.find('s') + 2: flag])
        minutes = int(org[flag + 1: org.find(':', flag + 1)])
        seconds = float(org[org.find(':', flag + 1) + 1:])
        df.loc[i, 'tm_diff'] = ((24 * days + hours) * 60 + minutes) * 60 + seconds

    if isTrain:
        df.to_csv(base_dir + '/dataset/dataset2/trainset/train_trans.csv', index=False)
    else:
        df.to_csv(base_dir + '/dataset/dataset2/testset/test_a_trans.csv', index=False)


# %%
if __name__ == '__main__':
    train_df = pd.read_csv(base_dir + r'/dataset/raw_dataset/trainset/train_trans.csv')
    test_df = pd.read_csv(base_dir + r'/dataset/raw_dataset/testset/test_a_trans.csv')
    trans_proc(train_df, isTrain=True)
    trans_proc(test_df, isTrain=False)

# %%
'''
train_df = pd.read_csv(base_dir + r'/dataset/raw_dataset/trainset/train_trans.csv')
print(train_df['tunnel_out'])
'''