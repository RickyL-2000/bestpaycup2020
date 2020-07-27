# %%
import pandas as pd
import numpy as np
import os,sys

#%% vscode user
os.chdir(sys.path[0]) # 把运行目录强制转移到当前脚本所在文件夹
print(f"工作路径切换到当前脚本路径: {os.getcwd()}")
TRAIN_BASE_PATH = './../../dataset/raw_dataset/trainset/train_base.csv'
TRAIN_OP_PATH = './../../dataset/raw_dataset/trainset/train_op.csv'
TRAIN_TRANS_PATH = "./../../dataset/raw_dataset/trainset/train_trans.csv"

SAMPLE_TRANS_PATH = "./../../dataset/raw_dataset/sample_trainset/sample_trans.csv"

#%% pycharm user
# 运行目录在“工作区”
TRAIN_BASE_PATH = './dataset/raw_dataset/trainset/train_base.csv'
TRAIN_OP_PATH = './dataset/raw_dataset/trainset/train_op.csv'
TRAIN_TRANS_PATH = "./dataset/raw_dataset/trainset/train_trans.csv"

SAMPLE_TRANS_PATH = "./dataset/raw_dataset/sample_trainset/sample_trans.csv"


# %%
def process_base():
    pass


# %%
def process_op():
    pass


# %%
def process_trans():
    pass


# %%
if __name__ == "__main__":
    pass
