# %%
import os
import pandas as pd
import numpy as np
import os,sys
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

# %%
# dataframe写入文件
# frame.to_csv("emm.csv", index=False)


# %%
class OpPreProc:
    def __init__(self, base_dir, isTrain=True, isSample=False):
        self.base_dir = base_dir
        self.isTrain = isTrain
        self.isSample = isSample

        self.df = None
        self.df_org = None
        self.n = 0          # df 的行数
        self.p = 0          # df 的列数（算上user列）

    def readDF(self):
        if self.isSample:
            if self.isTrain:
                self.df = pd.read_csv(self.base_dir + r'/dataset/raw_dataset/sample_trainset/sample_op.csv')
                self.n, self.p = self.df.shape[0], self.df.shape[1]
            else:
                raise RuntimeError("Test set has no sample")
        else:
            if self.isTrain:
                self.df = pd.read_csv(self.base_dir + r'/dataset/raw_dataset/trainset/train_op.csv')
            else:
                self.df = pd.read_csv(self.base_dir + r'/dataset/raw_dataset/testset/test_a_op.csv')
            self.n, self.p = self.df.shape[0], self.df.shape[1]
        self.df_org = self.df.copy(deep=True)

    def user_proc(self):
        for i in range(self.n):
            self.df.loc[i, 'user'] = self.df.loc[i, 'user'][6:]

    def op_type_proc(self):
        """对op_type进行整数赋值，方便之后的随机森林"""
        values = self.df['op_type'].unique().tolist()
        m = dict(zip(values, range(len(values))))
        self.df['op_type'] = self.df['op_type'].map(lambda x: m[x])

    def op_mode_proc(self):
        """对op_mode进行整数赋值，方便之后的随机森林"""
        values = self.df['op_mode'].unique().tolist()
        m = dict(zip(values, range(len(values))))
        self.df['op_mode'] = self.df['op_mode'].map(lambda x: m[x])

    def op_device_proc(self):
        pass

    def net_type_proc(self):
        """先哑变量填充，再one-hot编码"""
        # 将缺失值全部替换成'net_type_nan'
        self.df = self.df.fillna({'net_type': 'net_type_nan'})
        # 进行独热编码
        self.df = self.df.join(pd.get_dummies(self.df.net_type))
        self.df = self.df.drop(labels='net_type_proc', axis=1)

    def channel_proc(self):
        mode = self.df['channel'].mode()    # 众数赋值
        self.df = self.df.fillna({'channel': mode})
        # 整数赋值，方便随机森林
        values = self.df['channel'].unique().tolist()
        m = dict(zip(values, range(len(values))))
        self.df['channel'] = self.df['channel'].map(lambda x: m[x])

    def tm_diff_proc(self):
        for i in range(self.n):
            org = self.df.loc[i, 'tm_diff']
            days = int(org[:org.find('days')])
            hours = int(org[])

    def main(self):
        self.readDF()
        pass

    def test(self):
        self.readDF()
        self.user_proc()
        pass


# %%
if __name__ == "__main__":
    base_dir = os.getcwd()  # 工作区路径
    trainset_main = OpPreProc(base_dir, isTrain=True, isSample=True)
    trainset_main.test()

# %%
"""以下为试水专用"""

# %%
df = pd.read_csv(r"./dataset/raw_dataset/sample_trainset/sample_op.csv")
key = df['op_type'].unique().tolist()
m = dict(zip(key, range(len(key))))
df['op_type'] = df['op_type'].map(lambda x: m[x])
print(df)

# %%
df = pd.read_csv(r"./dataset/raw_dataset/sample_trainset/sample_op.csv")
df = df.fillna({'net_type': 0})
df = df.join(pd.get_dummies(df.net_type))
print(df)

# %%
df = pd.read_csv(r"./dataset/raw_dataset/sample_trainset/sample_op.csv")
print(max(df.groupby('channel').count()['user'].values))    # 众数的数量
print(df['channel'].mode())

# %%
df = pd.read_csv(r"./dataset/raw_dataset/sample_trainset/sample_op.csv")
op_type_onehot = pd.get_dummies(df.op_type)
print(op_type_onehot)