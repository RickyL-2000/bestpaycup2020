# %%
import pandas as pd
import matplotlib.pyplot as plt
import os,sys
os.chdir(sys.path[0]) # 把运行目录强制转移到当前脚本所在文件夹
print(f"工作路径切换到当前脚本路径: {os.getcwd()}")

# %%
csv1 = pd.read_csv('./../dataset/raw_dataset/trainset/train_op.csv')


# %%
print(csv1[0:1]['user'].values)

# %%
"""以下为trans文件的地盘"""

# %%

trans = pd.read_csv(r"./../dataset/raw_dataset/trainset/train_trans.csv")
sample_trans = pd.read_csv(r"./../dataset/raw_dataset/sample_trainset/sample_trans.csv")

print(sample_trans.head(10))

# %%
"""基本信息"""
print(trans.info())
print(trans.describe())


# %%
def feature_plot(feature):
    _x = list(feature.index)
    _y = list(feature.values)

    plt.figure()
    plt.bar(range(len(_x)), _y)

    for xx, yy in zip(range(len(_x)), _y):
        plt.text(xx, yy+5, str(yy), ha='center')

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