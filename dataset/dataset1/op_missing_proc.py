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

    def op_device_proc(self, dim=50):
        """
        对op_device进行one-hot+降维编码，并通过随机森林进行缺失值填补
        :param dim: 降到多少维度
        :return:
        """
        # one-hot + 降维
        values = self.df['op_device'].unique().tolist()
        values = np.array(values).reshape(len(values), -1)  # 转成列向量
        # one-hot encoder
        enc = OneHotEncoder()
        enc.fit(values)
        onehot = enc.transform(values).toarray()
        # pca encoder
        pca = PCA(n_components=dim) # 从1036维降到50维
        pca.fit(onehot)
        result = pca.transform(onehot)  # 降维后的one-hot
        mp = dict(zip(values, [code for code in result]))
        mp[np.nan] = [np.nan for i in range(dim)]    # 补充缺失值
        # 填入df
        newdf = pd.DataFrame(columns=["op_device_" + str(i) for i in range(dim)])     # 添加表头
        for i in range(self.n):
            code = mp[self.df.loc[i, 'op_device']]
            newdf.loc[len(newdf)] = code
        self.df = self.df.join(newdf)
        self.df = self.df.drop(labels='op_device')
        # 选择随机森林的输入数据
        X = self.df[['op_type', 'op_mode', 'net_type_0', 'net_type_1', 'net_type_2', 'net_type_3', 'channel',
                     'tm_dff']].loc[(self.df['op_device_0'].notnull())].values
        isnull_x = self.df[['op_type', 'op_mode', 'net_type_0', 'net_type_1', 'net_type_2', 'net_type_3', 'channel',
                     'tm_dff']].loc[(self.df['op_device_0'].isnull())].values
        Y_list = [self.df['op_device_' + str(i)].loc[(self.df['op_device'].notnull())].values for i in range(dim)]
        # x_df = self.df[['op_type', 'op_mode', 'net_type_0',
        #                 'net_type_1', 'net_type_2', 'net_type_3',
        #                 'channel', 'tm_dff'].extend(['op_device_{}'.format(i) for i in range(dim)])]
        # x_df_notnull_list = [x_df.loc[(self.df['op_device_{}'.format(i)].notnull())] for i in range(dim)]
        rfr = RandomForestRegressor(n_estimators=200)
        for i in range(dim):
            rfr.fit(X, Y_list[i])
            prediction = rfr.predict(isnull_x)
            self.df.loc[(self.df['op_device_' + str(i)].isnull()), 'op_device_'+str(i)] = prediction
            # TODO: 计算与PCA降维后最近的向量并以该类别赋值

    def net_type_proc(self):
        """先哑变量填充，再one-hot编码"""
        # 将缺失值全部替换成'net_type_nan'
        self.df = self.df.fillna({'net_type': 'net_type_nan'})
        # 进行独热编码
        # 首先进行整数赋值，方便重命名列名，如此一来dummy列的名称就变成了net_type_x
        values = self.df['net_type'].unique().tolist()
        m = dict(zip(values, range(len(values))))
        self.df['channel'] = self.df['channel'].map(lambda x: m[x])
        # 再进行编码
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
            flag = org.find(':')
            hours = int(org[org.find('s')+2: flag])
            minutes = int(org[flag+1: org.find(':', flag+1)])
            seconds = float(org[org.find(':', flag+1)+1:])
            self.df.loc[i, 'tm_diff'] = ((24 * days + hours) * 60 + minutes) * 60 + seconds

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
s = pd.get_dummies(df.net_type)     # join新表的时候，即时新表有行号也没有关系
print(s)
df = df.join(s)
print(df)

# %%
df = pd.read_csv(r"./dataset/raw_dataset/sample_trainset/sample_op.csv")
print(max(df.groupby('channel').count()['user'].values))    # 众数的数量
print(df['channel'].mode())

# %%
df = pd.read_csv(r"./dataset/raw_dataset/sample_trainset/sample_op.csv")
op_type_onehot = pd.get_dummies(df.op_type)
print(op_type_onehot.loc[3])

# %%
df = pd.DataFrame(columns=['a', 'b', 'c'])
# df.columns = [1, 2, 3]
# df.append(pd.Series([1, 2, 3], index=df.columns), ignore_index=True)
df.loc[len(df)] = [1, 2, 3]
print(df)
print(len(df))

# %%
df = pd.read_csv(r"./dataset/raw_dataset/sample_trainset/sample_op.csv")
print(df['op_device'].notnull())
