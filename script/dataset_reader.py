# %%
import pandas as pd
import matplotlib.pyplot as plt
import os,sys

#%% vscode user
os.chdir(sys.path[0]) # 把运行目录强制转移到当前脚本所在文件夹
print(f"工作路径切换到当前脚本路径: {os.getcwd()}")
TRAIN_BASE_PATH = './../dataset/raw_dataset/trainset/train_base.csv'
TRAIN_OP_PATH = './../dataset/raw_dataset/trainset/train_op.csv'
TRAIN_TRANS_PATH = "./../dataset/raw_dataset/trainset/train_trans.csv"

SAMPLE_TRANS_PATH = "./../dataset/raw_dataset/sample_trainset/sample_trans.csv"

#%% pycharm user
# 运行目录在“工作区”
TRAIN_BASE_PATH = './dataset/raw_dataset/trainset/train_base.csv'
TRAIN_OP_PATH = './dataset/raw_dataset/trainset/train_op.csv'
TRAIN_TRANS_PATH = "./dataset/raw_dataset/trainset/train_trans.csv"

SAMPLE_TRANS_PATH = "./dataset/raw_dataset/sample_trainset/sample_trans.csv"


# %% 
base = pd.read_csv(TRAIN_BASE_PATH)

index = ['user', 'sex', 'age', 'provider', 'level', 'verified', 'using_time',
       'regist_type', 'card_a_cnt', 'card_b_cnt', 'card_c_cnt', 'agreement1',
       'op1_cnt', 'op2_cnt', 'card_d_cnt', 'agreement_total', 'service1_cnt',
       'service1_amt', 'service2_cnt', 'agreement2', 'agreement3',
       'agreement4', 'acc_count', 'login_cnt_period1', 'login_cnt_period2',
       'ip_cnt', 'login_cnt_avg', 'login_days_cnt', 'province', 'city',
       'balance', 'balance_avg', 'balance1', 'balance1_avg', 'balance2',
       'balance2_avg', 'service3', 'service3_level', 'product1_amount',
       'product2_amount', 'product3_amount', 'product4_amount',
       'product5_amount', 'product6_amount', 'product7_cnt',
       'product7_fail_cnt']

print("base表条目总数：",len(base[:]['user'].values)) # 总量
print("属性".ljust(16),"缺失量".ljust(8),"种类数".ljust(10))  # 值的种类
for e in index:
    print(f"{e:<20}{base[e].isnull().sum():<8}{len(base[e].value_counts()):<10}")  # 值的种类

# %%
csv1 = pd.read_csv(TRAIN_OP_PATH)


# %%
print(len(csv1[:]['user'].values)) # 总量
print(csv1['user'].isnull().sum()) # 缺失值的数量
print(csv1['op_type'].isnull().sum())     
print(csv1['op_mode'].isnull().sum())  
print(csv1['op_device'].isnull().sum())
print(csv1['ip'].isnull().sum())
print(csv1['net_type'].isnull().sum())
print(csv1['channel'].isnull().sum())
print(csv1['ip_3'].isnull().sum())
print(csv1['tm_diff'].isnull().sum())
# %% 
# 统计种类及频率信息
print(csv1['user'].value_counts())
print(csv1['op_type'].value_counts())
print(csv1['op_mode'].value_counts())
print(csv1['op_device'].value_counts())
print(csv1['ip'].value_counts())
print(csv1['net_type'].value_counts())
print(csv1['channel'].value_counts())
print(csv1['ip_3'].value_counts())
print(csv1['tm_diff'].value_counts())
# %%
trans = pd.read_csv(TRAIN_TRANS_PATH)
sample_trans = pd.read_csv(SAMPLE_TRANS_PATH)

print(sample_trans.head(10))


# %%
def feature_plot(feature):
    _x = list(feature.index)
    _y = list(feature.values)

    plt.figure()
    plt.bar(range(len(_x)), _y)

    for xx, yy in zip(range(len(_x)), _y):
        plt.text(xx, yy+5, str(yy), ha='ljust')

    plt.xticks(range(len(_x)), _x)
    plt.show()


# %%
"""统计user"""
user = trans.groupby(by='user').count().sort_values(by='platform', ascending=False)['platform']

# feature_plot(user)

print(len(user.index))  # 统计user的种类数量

# %%
"""统计platform"""
platform = trans.groupby(by='platform').count().sort_values(by='user', ascending=False)['user']

# feature_plot(platform)

print(len(platform.index))  # platform的种类数量
print(trans['platform'].isnull().sum())     # 缺失值的数量
print(platform.index)

# %%
"""tunnel_in"""
tunnel_in = trans.groupby(by='tunnel_in').count().sort_values(by='user', ascending=False)['user']

print(len(tunnel_in.index))
print(trans['tunnel_in'].isnull().sum())
print(len(trans['tunnel_in']))
print(tunnel_in.index)

# %%
"""tunnel_out"""
tunnel_out = trans.groupby(by='tunnel_out').count().sort_values(by='user', ascending=False)['user']

print(len(tunnel_out.index))
print(trans['tunnel_out'].isnull().sum())
print(len(trans['tunnel_out']))
print(tunnel_in.index)

# %%
"""amount"""
print(trans['amount'].isnull().sum())

# %%
"""type1"""
type1 = trans.groupby(by='type1').count().sort_values(by='user', ascending=False)['user']

print(len(type1.index))
print(trans['type1'].isnull().sum())
print(len(trans['type1']))
print(type1.index)

# %%
"""ip"""
ip = trans.groupby(by='ip').count().sort_values(by='user', ascending=False)['user']

print(len(ip.index))
print(trans['ip'].isnull().sum())
print(len(trans['ip']))
print(ip.index)

# %%
"""type2"""
type2 = trans.groupby(by='type2').count().sort_values(by='user', ascending=False)['user']

print(len(type2.index))
print(trans['type2'].isnull().sum())
print(len(trans['type2']))
print(type2.index)

# %%
"""ip_3"""
ip_3 = trans.groupby(by='ip_3').count().sort_values(by='user', ascending=False)['user']

print(len(ip_3.index))
print(trans['ip_3'].isnull().sum())
print(len(trans['ip_3']))
print(ip_3.index)

# %%
"""tm_diff"""
tm_diff = trans.groupby(by='tm_diff').count().sort_values(by='user', ascending=False)['user']

print(len(tm_diff.index))
print(trans['tm_diff'].isnull().sum())
print(len(trans['tm_diff']))
print(tm_diff.index)