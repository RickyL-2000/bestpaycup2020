# %%
import pandas as pd
import matplotlib.pyplot as plt
import os,sys
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
#%% 
TRAIN_TRANS_PATH = "./../raw_dataset/trainset/train_trans.csv"
trans = pd.read_csv(TRAIN_TRANS_PATH)
# 首先，对platform做onehot
encoder = LabelEncoder()  
platform = encoder.fit_transform(trans['platform'].values)  
platform = np.array([platform]).T
enc = OneHotEncoder()
a=enc.fit_transform(platform)
a=a.toarray()
a=pd.DataFrame(a)
a.columns=['platform1','platform2','platform3','platform4','platform5','platform6']
trans = pd.concat([trans,a],axis=1)
trans = trans.drop(['platform'],axis=1)
trans.head()
# 对tunnel_in 做onehot并把缺失值做成新的一类
p = trans['tunnel_in'].fillna('a')
trans['tunnel_in'] = p 
tunnel_in = encoder.fit_transform(trans['tunnel_in'].values)  
tunnel_in = np.array([tunnel_in]).T
enc = OneHotEncoder()
a=enc.fit_transform(tunnel_in)
a=a.toarray()
a=pd.DataFrame(a)
a.columns=['tunnel_in1','tunnel_in2','tunnel_in3','tunnel_in4','tunnel_in5','tunnel_in6']
trans = pd.concat([trans,a],axis=1)
trans = trans.drop(['tunnel_in'],axis=1)
trans.head()
# 将ip和ip3两列去掉
trans = trans.drop(['ip'],axis=1)
trans = trans.drop(['ip_3'],axis=1)
trans.head()
# 对tunnel_out进行onehot加上众数填补
trans['tunnel_out'].value_counts()
p = trans['tunnel_out'].fillna('6ee790756007e69a')
trans['tunnel_out'] = p 
tunnel_out = encoder.fit_transform(trans['tunnel_out'].values)  
tunnel_out = np.array([tunnel_out]).T
enc = OneHotEncoder()
a=enc.fit_transform(tunnel_out)
a=a.toarray()
a=pd.DataFrame(a)
a.columns=['tunnel_out1','tunnel_out2','tunnel_out3','tunnel_out4']
trans = pd.concat([trans,a],axis=1)
trans = trans.drop(['tunnel_out'],axis=1)
trans.head()
# type1做onehot
encoder = LabelEncoder()  
type1 = encoder.fit_transform(trans['type1'].values)  
type1 = np.array([type1]).T
enc = OneHotEncoder()
a=enc.fit_transform(type1)
a=a.toarray()
a=pd.DataFrame(a)
a.columns=['type1_1','type1_2','type1_3','type1_4','type1_5','type1_6','type1_7','type1_8','type1_9','type1_10',
'type1_11','type1_12','type1_13','type1_14','type1_15','type1_16','type1_17','type1_18','type1_19','type1_20']
trans = pd.concat([trans,a],axis=1)
trans = trans.drop(['type1'],axis=1)
trans.head()

# 对type2 缺失值单独做成一类然后onehot
p = trans['type2'].fillna('a')
trans['type2'] = p 
type2 = encoder.fit_transform(trans['type2'].values)  
type2 = np.array([type2]).T
enc = OneHotEncoder()
a=enc.fit_transform(type2)
a=a.toarray()
a=pd.DataFrame(a)
a.columns=['type2_1','type2_2','type2_3','type2_4','type2_5','type2_6','type2_7','type2_8','type2_9','type2_10',
'type2_11','type2_12','type2_13']
trans = pd.concat([trans,a],axis=1)
trans = trans.drop(['type2'],axis=1)
trans.head()
trans.to_csv(path_or_buf='./../dataset1/train_trans.csv')
# %%
