# %%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
base_dir = os.getcwd()

# %%
feature_train_trans = pd.DataFrame()
feature_test_trans = pd.DataFrame()
feature_train_op = pd.DataFrame()
feature_test_op = pd.DataFrame()
train_base_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_base.csv')
train_op_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_op.csv')
train_trans_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_trans.csv')
test_base_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_base.csv')
test_op_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_op.csv')
test_trans_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_trans.csv')
train_n = len(train_base_df)
test_n = len(test_base_df)

# %%
# TODO: 要先把encoder计算好

# %%
# op_type onehot+pca
op_type = pd.concat([train_op_df['op_type'], test_op_df['op_type']])
dim_op_type = 10

values_op_type_org = op_type.unique().tolist()      # 原来shape的values
values_op_type = np.array(values_op_type_org).reshape(len(values_op_type_org), -1)
enc_op_type = OneHotEncoder()
enc_op_type.fit(values_op_type)
onehot_op_type = enc_op_type.transform(values_op_type).toarray()

pca_op_type = PCA(n_components=dim_op_type)
pca_op_type.fit(onehot_op_type)
result_op_type = pca_op_type.transform(onehot_op_type)
mp_op_type = dict(zip(values_op_type_org, [code for code in result_op_type]))

# %%
# op_mode onehot+pca
op_mode = pd.concat([train_op_df['op_mode'], test_op_df['op_mode']])
dim_op_mode = 10

values_op_mode_org = op_mode.unique().tolist()      # 原来shape的values
values_op_mode = np.array(values_op_mode_org).reshape(len(values_op_mode_org), -1)
enc_op_mode = OneHotEncoder()
enc_op_mode.fit(values_op_mode)
onehot_op_mode = enc_op_mode.transform(values_op_mode).toarray()

pca_op_mode = PCA(n_components=dim_op_mode)
pca_op_mode.fit(onehot_op_mode)
result_op_mode = pca_op_mode.transform(onehot_op_mode)
mp_op_mode = dict(zip(values_op_mode_org, [code for code in result_op_mode]))

# %%
# net_type onehot
net_type = pd.concat([train_op_df['net_type'], test_op_df['net_type']])

values_net_type_org = net_type.unique().tolist()
values_net_type = np.array(values_net_type_org).reshape(len(values_net_type_org), -1)
enc_net_type = OneHotEncoder()
enc_net_type.fit(values_net_type)
onehot_net_type = enc_net_type.transform(values_net_type)
mp_net_type = dict(zip(values_net_type_org, [code for code in onehot_net_type]))

# %%


# %%
# train set
for i in range(train_n):
    tr_trans_user = train_trans_df[train_trans_df['user'] == train_base_df.loc[i, 'user']]
    tr_op_user = train_op_df[train_op_df['user'] == train_base_df.loc[i, 'user']]

