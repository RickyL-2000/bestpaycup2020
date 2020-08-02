# %%
import pandas as pd
import matplotlib.pyplot as plt
import os,sys

#%% initial path
cur_path = sys.path[0].split(os.path.sep)
workspace_path = os.path.sep.join(cur_path[:cur_path.index("bestpaycup2020")+1])
os.chdir(workspace_path) # 把运行目录强制转移到【工作区】
print(f"把运行目录强制转移到【工作区】{os.getcwd()}")

TRAIN_BASE_PATH     = './dataset/raw_dataset/trainset/train_base.csv'
TRAIN_OP_PATH       = './dataset/raw_dataset/trainset/train_op.csv'
TRAIN_TRANS_PATH    = "./dataset/raw_dataset/trainset/train_trans.csv"

TEST_BASE_PATH      = './dataset/raw_dataset/testset/test_a_base.csv'
TEST_OP_PATH        = './dataset/raw_dataset/testset/test_a_op.csv'
TEST_TRANS_PATH     = "./dataset/raw_dataset/testset/test_a_trans.csv"

SAMPLE_BASE_PATH    = "./dataset/raw_dataset/sample_trainset/sample_base.csv"
SAMPLE_OP_PATH      = './dataset/raw_dataset/sample_trainset/sample_op.csv'
SAMPLE_TRANS_PATH   = "./dataset/raw_dataset/sample_trainset/sample_trans.csv"

# %% 
# index = ['user', 'sex', 'age', 'provider', 'level', 'verified', 'using_time',
#        'regist_type', 'card_a_cnt', 'card_b_cnt', 'card_c_cnt', 'agreement1',
#        'op1_cnt', 'op2_cnt', 'card_d_cnt', 'agreement_total', 'service1_cnt',
#        'service1_amt', 'service2_cnt', 'agreement2', 'agreement3',
#        'agreement4', 'acc_count', 'login_cnt_period1', 'login_cnt_period2',
#        'ip_cnt', 'login_cnt_avg', 'login_days_cnt', 'province', 'city',
#        'balance', 'balance_avg', 'balance1', 'balance1_avg', 'balance2',
#        'balance2_avg', 'service3', 'service3_level', 'product1_amount',
#        'product2_amount', 'product3_amount', 'product4_amount',
#        'product5_amount', 'product6_amount', 'product7_cnt',
#        'product7_fail_cnt']
base = pd.read_csv(TRAIN_BASE_PATH)
index = base.columns.tolist()
print("base表条目总数：",len(base[:]['user'].values)) # 总量
print("属性".ljust(16),"缺失量".ljust(8),"种类数".ljust(10))  # 值的种类
for e in index:
    print(f"{e:<20}{base[e].isnull().sum():<8}{len(base[e].value_counts()):<10}")  # 值的种类

# %%
for e in index:
    if e in ['using_time','card_a_cnt', 'card_b_cnt', 'card_c_cnt',
       'op1_cnt', 'op2_cnt', 'card_d_cnt', 'agreement_total', 'service1_cnt',
       'service1_amt', 'service2_cnt', 'acc_count', 'login_cnt_period1', 'login_cnt_period2',
       'ip_cnt', 'login_cnt_avg', 'login_days_cnt', 'product7_cnt',
       'product7_fail_cnt']:
        base[e].plot(kind='box',title=e)
        # plt.legend(loc="upper left")
        plt.show()


# %%
op = pd.read_csv(TRAIN_OP_PATH)
index = op.columns.tolist()
print("op表条目总数：",len(op[:]['user'].values)) # 总量
print("属性".ljust(16),"缺失量".ljust(8),"种类数".ljust(10))  # 值的种类
for e in index:
    print(f"{e:<20}{op[e].isnull().sum():<8}{len(op[e].value_counts()):<10}")  # 值的种类

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

# %%
base_dir = os.getcwd()
test_base_df = pd.read_csv(base_dir + '/dataset/raw_dataset/testset/test_a_base.csv')
print(test_base_df['province'].value_counts())
print(len(test_base_df.groupby(by='province').count().sort_values(by='user', ascending=False)['user'].index))

# %%
test_base_new = pd.read_csv(base_dir + '/dataset/dataset1/testset/test_a_base.csv')
print(test_base_df['province'].value_counts())
print(test_base_new['0'].value_counts())
print(test_base_new['1'].value_counts())
print(test_base_new['2'].value_counts())
print(test_base_new['3'].value_counts())
print(test_base_new['4'].value_counts())
print(test_base_new['5'].value_counts())
print(test_base_new['6'].value_counts())
print(test_base_new['7'].value_counts())
print(test_base_new['8'].value_counts())
print(test_base_new['9'].value_counts())
print(test_base_new['10'].value_counts())
print(test_base_new['11'].value_counts())
print(test_base_new['12'].value_counts())
print(test_base_new['13'].value_counts())
print(test_base_new['14'].value_counts())
print(test_base_new['15'].value_counts())
print(test_base_new['16'].value_counts())
print(test_base_new['17'].value_counts())
print(test_base_new['18'].value_counts())
print(test_base_new['19'].value_counts())
print(test_base_new['20'].value_counts())
print(test_base_new['21'].value_counts())
print(test_base_new['22'].value_counts())
print(test_base_new['23'].value_counts())
print(test_base_new['24'].value_counts())
print(test_base_new['25'].value_counts())
print(test_base_new['26'].value_counts())
print(test_base_new['27'].value_counts())
print(test_base_new['28'].value_counts())
print(test_base_new['29'].value_counts())
print(test_base_new['30'].value_counts())
print(test_base_new['31'].value_counts())
print(len(test_base_new))