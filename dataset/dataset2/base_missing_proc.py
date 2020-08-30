#%% initial path
import os,sys
cur_path = sys.path[0].split(os.path.sep)
workspace_path = os.path.sep.join(cur_path[:cur_path.index("bestpaycup2020")+1])
base_dir = workspace_path
os.chdir(workspace_path) # 把运行目录强制转移到【工作区】
print(f"把运行目录强制转移到【工作区】{os.getcwd()}")

# %%
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
#%%
TRAIN_BASE_PATH     = "./dataset/raw_dataset/trainset/train_base.csv"
TRAIN_OP_PATH       = "./dataset/raw_dataset/trainset/train_op.csv"
TRAIN_TRANS_PATH    = "./dataset/raw_dataset/trainset/train_trans.csv"

TEST_BASE_PATH      = "./dataset/raw_dataset/testset/test_a_base.csv"
TEST_OP_PATH        = "./dataset/raw_dataset/testset/test_a_op.csv"
TEST_TRANS_PATH     = "./dataset/raw_dataset/testset/test_a_trans.csv"

SAMPLE_BASE_PATH    = "./dataset/raw_dataset/sample_trainset/sample_base.csv"
SAMPLE_OP_PATH      = "./dataset/raw_dataset/sample_trainset/sample_op.csv"
SAMPLE_TRANS_PATH   = "./dataset/raw_dataset/sample_trainset/sample_trans.csv"

PROCESSED_TRAIN_BASE_PATH = "./dataset/dataset2/trainset/train_base.csv"
PROCESSED_TEST_BASE_PATH  = "./dataset/dataset2/testset/test_a_base.csv"


# %%
def process_base(base_path):
    # TODO: provider, province和city都各有一个缺失值，需要众数填补
    def to_int(entry):
        if type(entry) is str:
            level = re.search("^(category |level |Train_|TestA_|TestB_)([0-9]+)",entry)
            if level:
                return int(level.group(2))
        return entry

    # 读取数据，制作备份，数据转型
    base_df = pd.read_csv(base_path)
    for e in base_df.columns:
        base_df[e] = base_df[e].apply(to_int)

    # 显式处理缺失值
    base_df["sex"].fillna(2,inplace=True)

    # 合并service3项
    base_df["service3"][base_df["service3"]==0] = -1
    base_df["service3"][base_df["service3"] != -1] = base_df["service3_level"][base_df["service3_level"].notna()]
    base_df.drop("service3_level",axis=1,inplace=True)    # 删除service3_level列

    # 隐式处理其余缺失值
    for e in base_df.columns:
        base_df[e].fillna(base_df[e].mode()[0],inplace=True)            

    return base_df

#%%
for base_path,processed_base_path in [(TRAIN_BASE_PATH,PROCESSED_TRAIN_BASE_PATH),(TEST_BASE_PATH,PROCESSED_TEST_BASE_PATH)]:
    base_df = process_base(base_path)
    if not os.path.exists(os.path.split(processed_base_path)[0]):
        os.makedirs(os.path.split(processed_base_path)[0])
    with open(processed_base_path,"w") as f:
        base_df.to_csv(f,index=False)


def process_base_onehot(base_dir, dim):
    train_df = pd.read_csv(PROCESSED_TRAIN_BASE_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_BASE_PATH)
    province = pd.concat([train_df['province'], test_df['province']])
    city = pd.concat([train_df['city'], test_df['city']])

    values_pro = province.unique().tolist()
    m_pro = dict(zip(values_pro, range(len(values_pro))))
    train_df['province'] = train_df['province'].map(lambda x: m_pro[x])
    train_df = train_df.join(pd.get_dummies(train_df['province']))
    train_df = train_df.drop(labels='province', axis=1)

    test_df['province'] = test_df['province'].map(lambda x: m_pro[x])
    test_df = test_df.join(pd.get_dummies(test_df['province']))
    test_df = test_df.drop(labels='province', axis=1)

    values_ct_org = city.unique().tolist()
    values_ct = np.array(values_ct_org).reshape(len(values_ct_org), -1)
    enc = OneHotEncoder()
    enc.fit(values_ct)
    onehot = enc.transform(values_ct).toarray()

    pca = PCA(n_components=dim)
    pca.fit(onehot)
    result = pca.transform(onehot)
    mp = dict(zip(values_ct_org, [code for code in result]))

    newdf_train = pd.DataFrame(columns=['city_'+str(i) for i in range(dim)])
    for i in range(len(train_df)):
        code = mp[train_df.loc[i, 'city']]
        newdf_train.loc[len(newdf_train)] = code
    train_df = train_df.join(newdf_train)
    train_df = train_df.drop(labels='city', axis=1)

    newdf_test = pd.DataFrame(columns=['city_'+str(i) for i in range(dim)])
    for i in range(len(test_df)):
        code = mp[test_df.loc[i, 'city']]
        newdf_test.loc[len(newdf_test)] = code
    test_df = test_df.join(newdf_test)
    test_df = test_df.drop(labels='city', axis=1)

    train_df.to_csv(PROCESSED_TRAIN_BASE_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_BASE_PATH, index=False)

# %%
process_base_onehot(os.getcwd(), dim=50)

# %%
def delete_pro_31(base_dir):
    test_df = pd.read_csv(PROCESSED_TEST_BASE_PATH)
    test_df = test_df.drop(labels='31', axis=1)
    test_df.to_csv(PROCESSED_TEST_BASE_PATH, index=False)

# %%
delete_pro_31(base_dir=os.getcwd())

# %% final check
def show_missing(path):
    if type(path) is str:
        base = pd.read_csv(path)
    elif type(path) is pd.DataFrame:
        base = path
    else:
        raise Exception("invalid input type")
    index = sorted(base.columns.tolist())
    print("\n=======条目总数：",len(base[:]['user'].values)) # 总量
    print("属性".ljust(16),"缺失量".ljust(8),"种类数".ljust(10))  # 值的种类
    for e in index:
        print(f"{e:<20}{base[e].isnull().sum():<8}{len(base[e].value_counts()):<10}")  # 值的种类

processed_train_base = pd.read_csv(PROCESSED_TRAIN_BASE_PATH)
processed_test_base = pd.read_csv(PROCESSED_TEST_BASE_PATH)
show_missing(processed_test_base)
show_missing(processed_train_base)

# %%
