# %%
import pandas as pd
import numpy as np
import os,sys
import re
import matplotlib.pyplot as plt

#%% vscode user
os.chdir(sys.path[0]) # 把运行目录强制转移到当前脚本所在文件夹
print(f"工作路径切换到当前脚本路径: {os.getcwd()}")
TRAIN_BASE_PATH = './../../dataset/raw_dataset/trainset/train_base.csv'
TRAIN_OP_PATH = './../../dataset/raw_dataset/trainset/train_op.csv'
TRAIN_TRANS_PATH = "./../../dataset/raw_dataset/trainset/train_trans.csv"

SAMPLE_BASE_PATH = "./../../dataset/raw_dataset/sample_trainset/sample_base.csv"
SAMPLE_TRANS_PATH = "./../../dataset/raw_dataset/sample_trainset/sample_trans.csv"

PROCESSED_TRAIN_PATH = "./../../dataset/dataset1/trainset/train_trans.csv"

#%% pycharm user
# 运行目录在“工作区”
TRAIN_BASE_PATH = './dataset/raw_dataset/trainset/train_base.csv'
TRAIN_OP_PATH = './dataset/raw_dataset/trainset/train_op.csv'
TRAIN_TRANS_PATH = "./dataset/raw_dataset/trainset/train_trans.csv"

SAMPLE_BASE_PATH = "./dataset/raw_dataset/sample_trainset/sample_base.csv"
SAMPLE_TRANS_PATH = "./dataset/raw_dataset/sample_trainset/sample_trans.csv"

PROCESSED_TRAIN_PATH = "./dataset/dataset1/trainset/train_trans.csv"


# %%
def process_base(base_path,verbose=False):
    def to_int(entry):
        if type(entry) is str:
            level = re.search("^(category|level) ([0-9]+)",entry)
            if level:
                return int(level.group(2))
        return entry

    # 读取数据，制作备份，数据转型
    base = pd.read_csv(base_path)
    base2 = base.copy()
    for e in base2.columns:
        base2[e] = base[e].apply(to_int)

    # 处理缺失值
    base2["sex"][base2["sex"].isna()] = 3
    base2["balance_avg"].fillna(base2["balance_avg"].mode()[0],inplace=True)
    base2["balance1_avg"].fillna(base2["balance1_avg"].mode()[0],inplace=True)
    # base2["balance1_avg"][base2["balance1_avg"].isna()]=base2["balance1_avg"].mode()[0]

    # 合并service3项
    base2["service3"][base2["service3"]==0] = -1
    base2["service3"][base2["service3"] != -1] = base2["service3_level"][base2["service3_level"].notna()]
    base2.drop("service3_level",axis=1,inplace=True)    # 删除service3_level列

    if verbose:
        print(base2.info())
        print(base2.discribe())
    return base2

# %%
def process_op():
    pass


# %%
def process_trans():
    pass


# %%
if __name__ == "__main__":
    # base_path = SAMPLE_BASE_PATH
    base_path = TRAIN_BASE_PATH
    base2 = process_base(base_path)
    with open(PROCESSED_TRAIN_PATH,'w') as f:
        base2.to_csv(f,index=False)



# %%
