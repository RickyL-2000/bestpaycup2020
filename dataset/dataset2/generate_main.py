# %%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
base_dir = os.getcwd()

# %%
# 初始化表头
header = ['op_type_0', 'op_type_1', 'op_type_2', 'op_type_3', 'op_type_4', 'op_type_5', 'op_type_6', 'op_type_7',
             'op_type_8', 'op_type_9', 'op_type_perc', 'op_type_std', 'op_type_n', 'op_mode_0', 'op_mode_1',
             'op_mode_2', 'op_mode_3', 'op_mode_4', 'op_mode_5', 'op_mode_6', 'op_mode_7', 'op_mode_8', 'op_mode_9',
             'op_mode_perc', 'op_mode_std', 'op_mode_n', 'op_device_perc', 'op_device_std', 'op_device_nan_perc',
             'op_device_n', 'op_ip_perc', 'op_ip_std', 'op_ip_nan_perc', 'op_ip_n', 'op_net_type_0', 'op_net_type_1',
             'op_net_type_2', 'op_net_type_3', 'op_net_type_perc', 'op_net_type_std', 'op_net_type_nan_perc',
             'op_channel_0', 'op_channel_1', 'op_channel_2', 'op_channel_3', 'op_channel_4', 'op_channel_perc',
             'op_channel_std', 'op_channel_n', 'op_ip_3_perc', 'op_ip_3_std', 'op_ip_3_nan_perc', 'op_ip_3_n',
             'op_freq', 'op_ip_freq', 'op_ip_3_freq', 'op_ip_3_ch_freq', 'op_ip_48h_n', 'op_device_48h_n', 'op_48h_n',
             'trans_platform_0', 'trans_platform_1', 'trans_platform_2', 'trans_platform_3',
             'trans_platform_4', 'trans_platform_5', 'trans_platform_perc', 'trans_platform_std', 'trans_platform_n',
             'trans_tunnel_in_0', 'trans_tunnel_in_1', 'trans_tunnel_in_2', 'trans_tunnel_in_3', 'trans_tunnel_in_4',
             'trans_tunnel_in_5', 'trans_tunnel_in_perc', 'trans_tunnel_in_std', 'trans_tunnel_in_n',
             'trans_tunnel_in_nan_perc', 'trans_tunnel_out_0', 'trans_tunnel_out_1', 'trans_tunnel_out_2',
             'trans_tunnel_out_3', 'trans_tunnel_out_perc', 'trans_tunnel_out_std', 'trans_tunnel_n',
             'trans_amount_max', 'trans_amount_avg', 'trans_amount_std', 'trans_type1_0', 'trans_type1_1',
             'trans_type1_2', 'trans_type1_3', 'trans_type1_4', 'trans_type1_perc', 'trans_type1_std',
             'trans_ip_perc', 'trans_ip_std', 'trans_ip_nan_perc', 'trans_ip_n']


# %%
feature_train = pd.DataFrame(columns=header)
feature_test = pd.DataFrame(columns=header)
train_base_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_base.csv')
train_op_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_op.csv')
train_trans_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_trans.csv')
test_base_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_base.csv')
test_op_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_op.csv')
test_trans_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_trans.csv')
n_train = len(train_base_df)
n_test = len(test_base_df)

# %%
# load encoder
op_type = np.loadtxt(base_dir + '/dataset/dataset2/encoders/enc_op_type.csv', delimiter=',')
mp_op_type = {}
for i in range(op_type.shape[0]):
    mp_op_type[op_type[i, 0]] = op_type[i, 1:].tolist()

op_mode = np.loadtxt(base_dir + '/dataset/dataset2/encoders/enc_op_mode.csv', delimiter=',')
mp_op_mode = {}
for i in range(op_mode.shape[0]):
    mp_op_mode[op_mode[i, 0]] = op_mode[i, 1:].tolist()

net_type = np.loadtxt(base_dir + '/dataset/dataset2/encoders/enc_net_type.csv', delimiter=',')
mp_net_type = {}
for i in range(net_type.shape[0]):
    mp_net_type[net_type[i, 0]] = net_type[i, 1:].tolist()

channel = np.loadtxt(base_dir + '/dataset/dataset2/encoders/enc_channel.csv', delimiter=',')
mp_channel = {}
for i in range(channel.shape[0]):
    mp_channel[channel[i, 0]] = channel[i, 1:].tolist()

platform = np.loadtxt(base_dir + '/dataset/dataset2/encoders/enc_platform.csv', delimiter=',')
mp_platform = {}
for i in range(platform.shape[0]):
    mp_platform[platform[i, 0]] = platform[i, 1:].tolist()

tunnel_in = np.loadtxt(base_dir + '/dataset/dataset2/encoders/enc_tunnel_in.csv', delimiter=',')
mp_tunnel_in = {}
for i in range(tunnel_in.shape[0]):
    mp_tunnel_in[tunnel_in[i, 0]] = tunnel_in[i, 1:].tolist()

tunnel_out = np.loadtxt(base_dir + '/dataset/dataset2/encoders/enc_tunnel_out.csv', delimiter=',')
mp_tunnel_out = {}
for i in range(tunnel_out.shape[0]):
    mp_tunnel_out[tunnel_out[i, 0]] = tunnel_out[i, 1:].tolist()

type1 = np.loadtxt(base_dir + '/dataset/dataset2/encoders/enc_type1.csv', delimiter=',')
mp_type1 = {}
for i in range(type1.shape[0]):
    mp_type1[type1[i, 0]] = type1[i, 1:].tolist()

type2 = np.loadtxt(base_dir + '/dataset/dataset2/encoders/enc_type2.csv', delimiter=',')
mp_type2 = {}
for i in range(type2.shape[0]):
    mp_type2[type2[i, 0]] = type2[i, 1:].tolist()


# %%
def process(n, isTrain=True)
    for i in range(n):
        cur_user = train_base_df['user'].loc[i]
        if isTrain:
            tr_trans_user = train_trans_df[train_trans_df['user'] == cur_user]  # 该用户的trans记录
            tr_op_user = train_op_df[train_op_df['user'] == cur_user]           # 该用户的op记录
        else:
            tr_trans_user = test_trans_df[train_trans_df['user'] == cur_user]  # 该用户的trans记录
            tr_op_user = test_op_df[train_op_df['user'] == cur_user]           # 该用户的op记录
        line = []   # 一行，即当前用户的所有二次特征

        n_tr_trans_user = len(tr_trans_user)    # 该用户的trans记录条数
        n_tr_op_user = len(tr_op_user)          # 该用户的op记录条数

        ### op_type
        mode_op_type = tr_op_user['op_type'].mode()[0]
        code = mp_op_type[mode_op_type]
        line.extend(code)

        line.append(sum(tr_op_user['op_type'].apply(lambda x: 1 if x == mode_op_type else 0)) / n_tr_op_user)

        s = tr_op_user['op_type'].value_counts()
        line.append(np.std(s.values))

        line.append(len(s))

        ### op_mode
        mode_op_mode = tr_op_user['op_mode'].mode()[0]
        code = mp_op_mode[mode_op_mode]
        line.extend(code)

        line.append(sum(tr_op_user['op_mode'].apply(lambda x: 1 if x == mode_op_mode else 0)) / n_tr_op_user)

        s = tr_op_user['op_mode'].value_counts()
        line.append(np.std(s.values))

        line.append(len(s))

        ### op_device
        mode_op_device = tr_op_user['op_device'].mode()[0]
        line.append(sum(tr_op_user['op_device'].apply(lambda x: 1 if x == mode_op_device else 0)) / n_tr_op_user)

        s = tr_op_user['op_device'].value_counts()
        line.append(np.std(s.values))

        line.append(tr_op_user['op_device'].isnull().sum() / n_tr_op_user)

        line.append(len(s))

        ### op_ip
        mode_op_ip = tr_op_user['ip'].mode()[0]
        line.append(sum(tr_op_user['ip'].apply(lambda x: 1 if x == mode_op_ip else 0)) / n_tr_op_user)

        s = tr_op_user['ip'].value_counts()
        line.append(np.std(s.values))

        line.append(tr_op_user['ip'].isnull().sum() / n_tr_op_user)

        line.append(len(s))
        n_ip = len(s)

        ### op_net_type
        mode_op_net_type = tr_op_user['net_type'].mode()[0]
        code = mp_net_type[mode_op_net_type]
        line.extend(code)

        line.append(sum(tr_op_user['net_type'].apply(lambda x: 1 if x == mode_op_net_type else 0)) / n_tr_op_user)

        s = tr_op_user['net_type'].value_counts()
        line.append(np.std(s.values))

        line.append(tr_op_user['net_type'].isnull().sum() / n_tr_op_user)

        ### channel
        mode_op_channel = tr_op_user['channel'].mode()[0]
        code = mp_channel[mode_op_channel]
        line.extend(code)

        line.append(sum(tr_op_user['channel'].apply(lambda x: 1 if x == mode_op_channel else 0)) / n_tr_op_user)

        s = tr_op_user['channel'].value_counts()
        line.append(np.std(s.values))

        line.append(len(s))

        ### ip_3
        mode_op_ip_3 = tr_op_user['ip_3'].mode()[0]
        line.append(sum(tr_op_user['ip_3'].apply(lambda x: 1 if x == mode_op_ip_3 else 0)) / n_tr_op_user)

        s = tr_op_user['ip_3'].value_counts()
        line.append(np.std(s.values))

        line.append(tr_op_user['ip_3'].isnull().sum() / n_tr_op_user)

        line.append(len(s))
        n_ip_3 = len(s)

        ### tm_diff
        tr_op_tm_max = tr_op_user['tm_diff'].values.max()
        tr_op_tm_min = tr_op_user['tm_diff'].values.min()
        coef = 3600 * 24 / (tr_op_tm_max - tr_op_tm_min)
        line.append(n_tr_op_user * coef)

        line.append(n_ip * coef)

        line.append(n_ip_3 * coef)

        ### 对tm_diff排序
        tr_op_user.sort_values('tm_diff', inplace=True)
        cnt = 0
        s = tr_op_user['ip_3']
        pre = s.loc[0]
        for j in range(1, n_tr_op_user):
            if s.loc[j] != pre:
                pre = s.loc[j]
                cnt += 1
        line.append(cnt)

        ### 48h最高ip种类数量、最高的op_device种类数量、最高的op记录次数
        gap = 48 * 3600
        start = tr_op_tm_min
        end = start + gap
        max_48h_ip_n = 0
        max_48h_op_device_n = 0
        max_48h_op_n = 0
        while start <= tr_op_tm_max:
            gap_df = tr_op_user[start <= tr_op_user['tm_diff'] < end]
            max_48h_ip_n = max(max_48h_ip_n, gap_df['ip'].nunique())
            max_48h_op_device_n = max(max_48h_op_device_n, gap_df['op_device'].nunique())
            max_48h_op_n = max(max_48h_op_n, len(gap_df))
            start = end
            end += gap

        line.extend([max_48h_ip_n, max_48h_op_device_n, max_48h_op_n])

        ### platform
        mode_trans_platform = tr_trans_user['platform'].mode()[0]
        code = mp_platform[mode_trans_platform]
        line.extend(code)

        line.append(sum(tr_trans_user['platform'].apply(lambda x: 1 if x == mode_trans_platform else 0)) / n_tr_trans_user)

        s = tr_trans_user['platform'].value_counts()
        line.append(np.std(s.values))

        line.append(len(s))

        ### tunnel_in
        mode_trans_tunnel_in = tr_trans_user['tunnel_in'].mode()[0]
        code = mp_tunnel_in[mode_trans_tunnel_in]
        line.extend(code)

        line.append(sum(tr_trans_user['tunnel_in'].apply(lambda x: 1 if x == mode_trans_tunnel_in else 0)) / n_tr_trans_user)

        s = tr_trans_user['tunnel_in'].value_counts()
        line.append(np.std(s.values))

        line.append(len(s))

        line.append(tr_trans_user['tunnel_in'].isnull().sum() / n_tr_trans_user)

        ### tunnel_out
        mode_trans_tunnel_out = tr_trans_user['tunnel_out'].mode()[0]
        code = mp_tunnel_out[mode_trans_tunnel_out]
        line.extend(code)

        line.append(sum(tr_trans_user['tunnel_out'].apply(lambda x: 1 if x == mode_trans_tunnel_out else 0)) / n_tr_trans_user)

        s = tr_trans_user['tunnel_out'].value_counts()
        line.append(np.std(s.values))

        line.append(len(s))

        ### amount
        s = tr_trans_user['amount']
        line.append(s.values.max())
        line.append(s.values.mean())
        line.append(s.values.std())
        sum_amount = (s.values.sum())

        ### type1
        mode_trans_type1 = tr_trans_user['type1'].mode()[0]
        code = mp_type1[mode_trans_type1]
        line.extend(code)

        line.append(sum(tr_trans_user['type1'].apply(lambda x: 1 if x == mode_trans_type1 else 0)) / n_tr_trans_user)

        s = tr_trans_user['type1'].value_counts()
        line.append(np.std(s.values))

        ### trans_ip
        mode_trans_ip = tr_trans_user['ip'].mode()[0]
        line.append(sum(tr_trans_user['ip'].apply(lambda x: 1 if x == mode_trans_ip else 0)) / n_tr_trans_user)

        s = tr_trans_user['ip'].value_counts()
        line.append(np.std(s.values))

        line.append(tr_trans_user['ip'].isnull().sum() / n_tr_trans_user)

        line.append(len(s))
        n_ip = len(s)

        ### type2
        mode_trans_type2 = tr_trans_user['type2'].mode()[0]
        code = mp_type2[mode_trans_type2]
        line.extend(code)

        line.append(sum(tr_trans_user['type2'].apply(lambda x: 1 if x == mode_trans_type2 else 0)) / n_tr_trans_user)

        s = tr_trans_user['type2'].value_counts()
        line.append(np.std(s.values))

        ### trans_ip_3
        mode_trans_ip_3 = tr_trans_user['ip_3'].mode()[0]
        line.append(sum(tr_trans_user['ip_3'].apply(lambda x: 1 if x == mode_trans_ip_3 else 0)) / n_tr_trans_user)

        s = tr_trans_user['ip'].value_counts()
        line.append(np.std(s.values))

        line.append(tr_trans_user['ip'].isnull().sum() / n_tr_trans_user)

        line.append(len(s))
        n_ip_3 = len(s)

        ### tm_diff
        tr_trans_tm_max = tr_trans_user['tm_diff'].values.max()
        tr_trans_tm_min = tr_trans_user['tm_diff'].values.min()
        coef = 3600 * 24 / (tr_trans_tm_max - tr_trans_tm_min)
        line.append(n_tr_trans_user * coef)

        line.append(sum_amount * coef)

        line.append(n_ip * coef)

        line.append(n_ip_3 * coef)

        ### 对tm_diff排序
        tr_trans_user.sort_values('tm_diff', inplace=True)
        cnt = 0
        s = tr_trans_user['ip_3']
        pre = s.loc[0]
        for j in range(1, n_tr_trans_user):
            if s.loc[j] != pre:
                pre = s.loc[j]
                cnt += 1
        line.append(cnt)

        ### 48h最高amount总量、最高的trans数量、最高的platform种类数量、最高的ip种类数量
        gap = 48 * 3600
        start = tr_trans_tm_min
        end = start + gap
        max_48h_sum_amount = 0
        max_48h_trans_n = 0
        max_48h_platform_n = 0
        max_48h_ip_n = 0
        while start <= tr_trans_tm_max:
            gap_df = tr_trans_user[start <= tr_trans_user['tm_diff'] < end]
            max_48h_sum_amount = max(max_48h_sum_amount, gap_df['amount'].values.sum())
            max_48h_trans_n = max(max_48h_trans_n, len(gap_df))
            max_48h_platform_n = max(max_48h_platform_n, gap_df['platform'].nunique())
            max_48h_ip_n = max(max_48h_ip_n, gap_df['ip'].nunique())
            start = end
            end += gap

        line.extend([max_48h_sum_amount, max_48h_trans_n, max_48h_platform_n, max_48h_ip_n])

        ### 填入feature矩阵
        if isTrain:
            feature_train.loc[len(feature_train)] = line
        else:
            feature_test.loc[len(feature_test)] = line

    # 存
    if isTrain:
        feature_train.to_csv(base_dir + '/dataset/dataset2/trainset/feature_train.csv', index=False)
    else:
        feature_test.to_csv(base_dir + '/dataset/dataset2/testset/feature_test.csv', index=False)


# %%
process(n_train, isTrain=True)
process(n_test, isTrain=False)

# %%
