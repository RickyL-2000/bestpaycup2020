# %%
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
base_dir = os.getcwd()

# %%
train_op_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_op.csv')
train_trans_df = pd.read_csv(base_dir + '/dataset/dataset2/trainset/train_trans.csv')
test_op_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_op.csv')
test_trans_df = pd.read_csv(base_dir + '/dataset/dataset2/testset/test_a_trans.csv')

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

pd.DataFrame.from_dict(data=mp_op_type, orient='columns')\
    .to_csv(base_dir + '/dataset/dataset2/encoders/enc_op_type.csv')

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

pd.DataFrame.from_dict(data=mp_op_mode, orient='columns')\
    .to_csv(base_dir + '/dataset/dataset2/encoders/enc_op_mode.csv')

# %%
# net_type onehot
net_type = pd.concat([train_op_df['net_type'], test_op_df['net_type']])

values_net_type_org = net_type.unique().tolist()
values_net_type = np.array(values_net_type_org).reshape(len(values_net_type_org), -1)
enc_net_type = OneHotEncoder()
enc_net_type.fit(values_net_type)
onehot_net_type = enc_net_type.transform(values_net_type)
mp_op_net_type = dict(zip(values_net_type_org, [code for code in onehot_net_type]))

pd.DataFrame.from_dict(data=mp_op_net_type, orient='columns')\
    .to_csv(base_dir + '/dataset/dataset2/encoders/enc_op_net_type.csv')

# %%
# op_channel onehot+pca
channel = pd.concat([train_op_df['channel'], test_op_df['channel']])
dim_channel = 5

values_channel_org = channel.unique().tolist()      # 原来shape的values
values_channel = np.array(values_channel_org).reshape(len(values_channel_org), -1)
enc_channel = OneHotEncoder()
enc_channel.fit(values_channel)
onehot_channel = enc_channel.transform(values_channel).toarray()

pca_channel = PCA(n_components=dim_channel)
pca_channel.fit(onehot_channel)
result_channel = pca_channel.transform(onehot_channel)
mp_op_channel = dict(zip(values_channel_org, [code for code in result_channel]))

pd.DataFrame.from_dict(data=mp_op_channel, orient='columns')\
    .to_csv(base_dir + '/dataset/dataset2/encoders/enc_op_channel.csv')

# %%
# tran_platform onehot
platform = pd.concat([train_trans_df['platform'], test_trans_df['platform']])

values_platform_org = platform.unique().tolist()
values_platform = np.array(values_platform_org).reshape(len(values_platform_org), -1)
enc_platform = OneHotEncoder()
enc_platform.fit(values_platform)
onehot_platform = enc_platform.transform(values_platform)
mp_trans_platform = dict(zip(values_platform_org, [code for code in onehot_platform]))

pd.DataFrame.from_dict(data=mp_trans_platform, orient='columns')\
    .to_csv(base_dir + '/dataset/dataset2/encoders/enc_trans_platform.csv')

# %%
# tunnel_in onehot
tunnel_in = pd.concat([train_trans_df['tunnel_in'], test_trans_df['tunnel_in']])

values_tunnel_in_org = tunnel_in.unique().tolist()
values_tunnel_in = np.array(values_tunnel_in_org).reshape(len(values_tunnel_in_org), -1)
enc_tunnel_in = OneHotEncoder()
enc_tunnel_in.fit(values_tunnel_in)
onehot_tunnel_in = enc_tunnel_in.transform(values_tunnel_in)
mp_trans_tunnel_in = dict(zip(values_tunnel_in_org, [code for code in onehot_tunnel_in]))

pd.DataFrame.from_dict(data=mp_trans_tunnel_in, orient='columns')\
    .to_csv(base_dir + '/dataset/dataset2/encoders/enc_trans_tunnel_in.csv')

# %%
# tunnel_out onehot
tunnel_out = pd.concat([train_trans_df['tunnel_out'], test_trans_df['tunnel_out']])

values_tunnel_out_org = tunnel_out.unique().tolist()
values_tunnel_out = np.array(values_tunnel_out_org).reshape(len(values_tunnel_out_org), -1)
enc_tunnel_out = OneHotEncoder()
enc_tunnel_out.fit(values_tunnel_out)
onehot_tunnel_out = enc_tunnel_out.transform(values_tunnel_out)
mp_trans_tunnel_out = dict(zip(values_tunnel_out_org, [code for code in onehot_tunnel_out]))

pd.DataFrame.from_dict(data=mp_trans_tunnel_out, orient='columns')\
    .to_csv(base_dir + '/dataset/dataset2/encoders/enc_trans_tunnel_out.csv')

# %%
# trans_type1 onehot+pca
type1 = pd.concat([train_trans_df['type1'], test_trans_df['type1']])
dim_type1 = 5

values_type1_org = type1.unique().tolist()      # 原来shape的values
values_type1 = np.array(values_type1_org).reshape(len(values_type1_org), -1)
enc_type1 = OneHotEncoder()
enc_type1.fit(values_type1)
onehot_type1 = enc_type1.transform(values_type1).toarray()

pca_type1 = PCA(n_components=dim_type1)
pca_type1.fit(onehot_type1)
result_type1 = pca_type1.transform(onehot_type1)
mp_trans_type1 = dict(zip(values_type1_org, [code for code in result_type1]))

pd.DataFrame.from_dict(data=mp_trans_type1, orient='columns')\
    .to_csv(base_dir + '/dataset/dataset2/encoders/enc_trans_type1.csv')

# %%
type2 = pd.concat([train_trans_df['type2'], test_trans_df['type2']])
dim_type2 = 5

values_type2_org = type2.unique().tolist()      # 原来shape的values
values_type2 = np.array(values_type2_org).reshape(len(values_type2_org), -1)
enc_type2 = OneHotEncoder()
enc_type2.fit(values_type2)
onehot_type2 = enc_type2.transform(values_type2).toarray()

pca_type2 = PCA(n_components=dim_type2)
pca_type2.fit(onehot_type2)
result_type2 = pca_type2.transform(onehot_type2)
mp_trans_type2 = dict(zip(values_type2_org, [code for code in result_type2]))

pd.DataFrame.from_dict(data=mp_trans_type2, orient='columns')\
    .to_csv(base_dir + '/dataset/dataset2/encoders/enc_trans_type2.csv')
