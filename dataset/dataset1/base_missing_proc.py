# %%
import pandas as pd
import numpy as np
import os,sys
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA

#%% initial path
cur_path = sys.path[0].split(os.path.sep)
workspace_path = os.path.sep.join(cur_path[:cur_path.index("bestpaycup2020")+1])
os.chdir(workspace_path) # 把运行目录强制转移到【工作区】
print(f"把运行目录强制转移到【工作区】{os.getcwd()}")

#%%
TRAIN_BASE_PATH     = "./dataset/raw_dataset/trainset/train_base.csv"
TRAIN_OP_PATH       = "./dataset/raw_dataset/trainset/train_op.csv"
TRAIN_TRANS_PATH    = "./dataset/raw_dataset/trainset/train_trans.csv"

TEST_A_BASE_PATH      = "./dataset/raw_dataset/testset/test_a_base.csv"
TEST_A_OP_PATH        = "./dataset/raw_dataset/testset/test_a_op.csv"
TEST_A_TRANS_PATH     = "./dataset/raw_dataset/testset/test_a_trans.csv"
TEST_B_BASE_PATH      = "./dataset/raw_dataset/testset/test_b_base.csv"
TEST_B_OP_PATH        = "./dataset/raw_dataset/testset/test_b_op.csv"
TEST_B_TRANS_PATH     = "./dataset/raw_dataset/testset/test_b_trans.csv"

SAMPLE_BASE_PATH    = "./dataset/raw_dataset/sample_trainset/sample_base.csv"
SAMPLE_OP_PATH      = "./dataset/raw_dataset/sample_trainset/sample_op.csv"
SAMPLE_TRANS_PATH   = "./dataset/raw_dataset/sample_trainset/sample_trans.csv"

PROCESSED_TRAIN_BASE_PATH = "./dataset/dataset1/trainset/train_base.csv"
PROCESSED_TEST_A_BASE_PATH  = "./dataset/dataset1/testset/test_a_base.csv"
PROCESSED_TEST_B_BASE_PATH  = "./dataset/dataset1/testset/test_b_base.csv"


# %%
def process_base(base_path,verbose=False):
    # TODO: provider, province和city都各有一个缺失值，需要众数填补
    def to_int(entry):
        if type(entry) is str:
            level = re.search("^(category |level |Train_|TestA_|TestB_)([0-9]+)",entry)
            if level:
                return int(level.group(2))
        return entry

    # 读取数据，制作备份，数据转型
    base = pd.read_csv(base_path)
    base2 = base.copy()
    for e in base2.columns:
        base2[e] = base[e].apply(to_int)

    # 处理缺失值
    # base2["sex"][base2["sex"].isna()] = 3
    base2["sex"].fillna(base2["sex"].mode()[0], inplace=True)
    base2["balance_avg"].fillna(base2["balance_avg"].mode()[0],inplace=True)
    base2["balance1_avg"].fillna(base2["balance1_avg"].mode()[0],inplace=True)
    # base2["balance1_avg"][base2["balance1_avg"].isna()]=base2["balance1_avg"].mode()[0]

    # 合并service3项
    base2["service3"][base2["service3"]==0] = -1
    base2["service3"][base2["service3"] != -1] = base2["service3_level"][base2["service3_level"].notna()]
    base2.drop("service3_level",axis=1,inplace=True)    # 删除service3_level列

    print(f"{base_path} has shape {base2.shape} after processing")

    if verbose:
        print(base2.info())
        print(base2.discribe())
    return base2


#%%
for base_path,processed_base_path in [(TRAIN_BASE_PATH,PROCESSED_TRAIN_BASE_PATH),
                                      (TEST_A_BASE_PATH,PROCESSED_TEST_A_BASE_PATH),
                                      (TEST_B_BASE_PATH,PROCESSED_TEST_B_BASE_PATH)]:
    base2 = process_base(base_path)
    if not os.path.exists(os.path.split(processed_base_path)[0]):
        os.makedirs(os.path.split(processed_base_path)[0])
    with open(processed_base_path,"w") as f:
        base2.to_csv(f,index=False,line_terminator='\n')

# %%
# base_path = TEST_B_BASE_PATH
# processed_base_path = PROCESSED_TEST_B_BASE_PATH
# base_df = process_base(base_path)
# if not os.path.exists(os.path.split(processed_base_path)[0]):
#     os.makedirs(os.path.split(processed_base_path)[0])
# with open(processed_base_path, "w") as f:
#     base_df.to_csv(f, index=False, line_terminator='\n')


# %%
def process_base_onehot(base_dir, dim):
    train_df = pd.read_csv(base_dir + '/dataset/dataset1/trainset/train_base.csv')
    test_a_df = pd.read_csv(base_dir + '/dataset/dataset1/testset/test_a_base.csv')
    test_b_df = pd.read_csv(base_dir + '/dataset/dataset1/testset/test_b_base.csv')
    province = pd.concat([train_df['province'], test_a_df['province'], test_b_df['province']])
    city = pd.concat([train_df['city'], test_a_df['city'], test_b_df['city']])

    values_pro = province.unique().tolist()
    # m_pro = dict(zip(values_pro, range(len(values_pro))))
    m_pro = dict(zip(values_pro, ['province_' + str(i) for i in range(len(values_pro))]))
    train_df['province'] = train_df['province'].map(lambda x: m_pro[x])
    train_df = train_df.join(pd.get_dummies(train_df['province']))
    train_df = train_df.drop(labels='province', axis=1)

    test_a_df['province'] = test_a_df['province'].map(lambda x: m_pro[x])
    test_a_df = test_a_df.join(pd.get_dummies(test_a_df['province']))
    test_a_df = test_a_df.drop(labels='province', axis=1)

    test_b_df['province'] = test_b_df['province'].map(lambda x: m_pro[x])
    test_b_df = test_b_df.join(pd.get_dummies(test_b_df['province']))
    test_b_df = test_b_df.drop(labels='province', axis=1)

    values_ct_org = city.unique().tolist()
    values_ct = np.array(values_ct_org).reshape(len(values_ct_org), -1)
    enc = OneHotEncoder()
    enc.fit(values_ct)
    onehot = enc.transform(values_ct).toarray()

    pca = PCA(n_components=dim)
    pca.fit(onehot)
    result = pca.transform(onehot)
    mp = dict(zip(values_ct_org, [code for code in result]))

    # 存encoders
    pd.DataFrame.from_dict(data=mp, orient='columns')\
        .to_csv(base_dir + '/dataset/dataset1/encoders/enc_base_city.csv', index=False)

    newdf_train = pd.DataFrame(columns=['city_'+str(i) for i in range(dim)])
    for i in range(len(train_df)):
        code = mp[train_df.loc[i, 'city']]
        newdf_train.loc[len(newdf_train)] = code
    train_df = train_df.join(newdf_train)
    train_df = train_df.drop(labels='city', axis=1)

    newdf_test_a = pd.DataFrame(columns=['city_'+str(i) for i in range(dim)])
    for i in range(len(test_a_df)):
        code = mp[test_a_df.loc[i, 'city']]
        newdf_test_a.loc[len(newdf_test_a)] = code
    test_a_df = test_a_df.join(newdf_test_a)
    test_a_df = test_a_df.drop(labels='city', axis=1)

    newdf_b_test = pd.DataFrame(columns=['city_' + str(i) for i in range(dim)])
    for i in range(len(test_b_df)):
        code = mp[test_b_df.loc[i, 'city']]
        newdf_b_test.loc[len(newdf_b_test)] = code
    test_b_df = test_b_df.join(newdf_b_test)
    test_b_df = test_b_df.drop(labels='city', axis=1)

    train_df.to_csv(base_dir + '/dataset/dataset1/trainset/train_base.csv', index=False)
    test_a_df.to_csv(base_dir + '/dataset/dataset1/testset/test_a_base.csv', index=False)
    test_b_df.to_csv(base_dir + '/dataset/dataset1/testset/test_b_base.csv', index=False)


# %%
process_base_onehot(os.getcwd(), dim=50)


# %%
def delete_pro_31(base_dir):
    test_df = pd.read_csv(base_dir + '/dataset/dataset1/testset/test_a_base.csv')
    test_df = test_df.drop(labels='31', axis=1)
    test_df.to_csv(base_dir + '/dataset/dataset1/testset/test_a_base.csv', index=False)


# %%
delete_pro_31(base_dir=os.getcwd())
