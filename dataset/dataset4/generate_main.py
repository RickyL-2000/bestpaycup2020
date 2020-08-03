# %%
import os
import pandas as pd
import numpy as np

base_dir = os.getcwd()

# %%
# 原来的需要保留的表头
old_headers_op = ['user', 'n_op', 'op_type_perc', 'op_type_std', 'op_type_n', 'op_mode_perc', 'op_mode_std',
                  'op_mode_n', 'op_device_perc', 'op_device_std', 'op_device_nan_perc', 'op_ip_perc', 'op_ip_std',
                  'op_ip_nan_perc', 'op_ip_n', 'op_net_type_perc', 'op_net_type_std', 'op_net_type_nan_perc',
                  'op_channel_perc', 'op_channel_std', 'op_channel_n', 'op_ip_3_perc', 'op_ip_3_std',
                  'op_ip_3_nan_perc', 'op_ip_3_n', 'op_ip_3_ch_freq', 'op_ip_48h_n', 'op_device_48h_n', 'op_48h_n']
old_headers_trans = ['user', 'n_trans', 'trans_platform_perc', 'trans_platform_std', 'trans_tunnel_in_perc',
                     'trans_tunnel_in_std', 'trans_tunnel_in_nan_perc', 'trans_tunnel_out_std', 'trans_amount_max',
                     'trans_amount_avg', 'trans_amount_std', 'trans_type1_0', 'trans_type1_1', 'trans_type1_2',
                     'trans_type1_3', 'trans_type1_4', 'trans_type1_perc', 'trans_type1_std', 'trans_ip_perc',
                     'trans_ip_std', 'trans_ip_nan_perc', 'trans_ip_n', 'trans_type2_perc', 'trans_type2_std',
                     'trans_ip_3_std', 'trans_ip_3_nan_perc', 'trans_ip_3_ch_freq', 'trans_amount_48h_n',
                     'trans_48h_n', 'trans_platform_48h_n', 'trans_ip_48h_n']

headers_op = ['n_op', 'op_type_perc', 'op_type_std', 'op_type_n', 'op_mode_perc', 'op_mode_std', 'op_mode_n',
              'op_device_perc', 'op_device_std', 'op_device_nan_perc', 'op_ip_perc', 'op_ip_std', 'op_ip_nan_perc',
              'op_ip_n', 'op_net_type_perc', 'op_net_type_std', 'op_net_type_nan_perc', 'op_channel_perc',
              'op_channel_std', 'op_channel_n', 'op_ip_3_perc', 'op_ip_3_std', 'op_ip_3_nan_perc', 'op_ip_3_n',
              'op_ip_3_ch_freq', 'op_ip_ch_freq', 'op_device_ch_freq', 'op_net_type_ch_freq', 'op_channel_ch_freq',
              'op_ip_48h_n', 'op_ip_3_48h_n', 'op_device_48h_n', 'op_48h_n', ]
headers_trans = ['n_trans', 'trans_platform_perc', 'trans_platform_std', 'trans_tunnel_in_perc', 'trans_tunnel_in_std',
                 'trans_tunnel_in_nan_perc', 'trans_tunnel_out_std', 'trans_amount_max', 'trans_amount_avg',
                 'trans_amount_std', 'trans_type1_0', 'trans_type1_1', 'trans_type1_2', 'trans_type1_3',
                 'trans_type1_4', 'trans_type1_perc', 'trans_type1_std', 'trans_ip_perc', 'trans_ip_std',
                 'trans_ip_nan_perc', 'trans_ip_n', 'trans_type2_perc', 'trans_type2_std', 'trans_ip_3_std',
                 'trans_ip_3_nan_perc', 'trans_ip_3_ch_freq', 'trans_ip_ch_freq', 'trans_platform_ch_freq',
                 'trans_amount_48h_n', 'trans_48h_n', 'trans_platform_48h_n', 'trans_ip_48h_n', 'trans_ip_3_48h_n']

# %%
# 新的特征矩阵
feature_train_op = pd.DataFrame(columns=headers_op)
feature_test_op = pd.DataFrame(columns=headers_op)
feature_train_trans = pd.DataFrame(columns=headers_trans)
feature_test_trans = pd.DataFrame(columns=headers_trans)

# 旧的特征矩阵
old_feature_train = pd.read_csv(base_dir + '/dataset/dataset2/trainset/feature_train.csv')
old_feature_test = pd.read_csv(base_dir + '/dataset/dataset2/testset/feature_test.csv')

# %%
# 将需要留下的特征搬运到新的矩阵中
for feature in old_headers_op:
    feature_train_op[feature] = old_feature_train[feature]
    feature_test_op[feature] = old_feature_train[feature]
for feature in old_headers_trans:
    feature_train_trans[feature] = old_feature_test[feature]
    feature_test_trans[feature] = old_feature_test[feature]

# %%
# 载入辅助矩阵，准备写入新特征
train_op_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_op.csv')
train_trans_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_trans.csv')
test_op_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_op.csv')
test_trans_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_trans.csv')


# %%
def process(n, isTrain=True):
    for i in range(n):
        if i % 1000 == 0:
            print(i)

        if isTrain:
            cur_user = feature_train_op['user'].loc[i]
            trans_user = train_trans_df[train_trans_df['user'] == cur_user]
            op_user = train_op_df[train_op_df['user'] == cur_user]
        else:
            cur_user = feature_test_op['user'].loc[i]
            trans_user = test_trans_df[test_trans_df['user'] == cur_user]
            op_user = test_op_df[test_op_df['user'] == cur_user]

        n_trans_user = len(trans_user)
        n_op_user = len(op_user)

        if n_op_user > 0:
            # 对 tm_diff 排序
            op_user.sort_values('tm_diff', inplace=True)

            # 统计各种频率
            cnt_op_ip = 0
            cnt_op_device = 0
            cnt_op_net_type = 0
            cnt_op_channel = 0
            pre_op_ip = op_user['ip'].loc[0]
            pre_op_device = op_user['op_device'].loc[0]
            pre_op_net_type = op_user['net_type'].loc[0]
            pre_op_channel = op_user['channel'].loc[0]
            for j in range(1, n_op_user):
                if pre_op_ip != op_user['ip'].loc[j]:
                    pre_op_ip = op_user['ip'].loc[j]
                    cnt_op_ip += 1
                if pre_op_device != op_user['op_device'].loc[j]:
                    pre_op_device = op_user['op_device'].loc[j]
                    cnt_op_device += 1
                if pre_op_net_type != op_user['net_type'].loc[j]:
                    pre_op_net_type = op_user['net_type'].loc[j]
                    cnt_op_net_type += 1
                if pre_op_channel != op_user['channel'].loc[j]:
                    pre_op_channel = op_user['channel'].loc[j]
                    cnt_op_channel += 1

        if n_trans_user > 0:
            # 对 tm_diff 排序
            trans_user.sort_values('tm_diff', inplace=True)

            # 统计各种频率
            cnt_trans_ip = 0
            cnt_trans_platform = 0
            pre_trans_ip = trans_user['ip'].loc[0]
            pre_trans_platform = trans_user['platform'].loc[0]
            for j in range(1, n_trans_user):
                if pre_trans_ip != trans_user['ip'].loc[j]:
                    pre_trans_ip = trans_user['ip'].loc[j]
                    cnt_trans_ip += 1
                if pre_trans_platform != trans_user['platform'].loc[j]:
                    pre_trans_platform = trans_user['platform'].loc[j]
                    cnt_trans_platform += 1


# %%
temp = pd.read_csv(r'C:\Users\RickyLi\Desktop\main_dataset.csv')
print(temp['维度'].tolist())
