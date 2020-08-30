#%% initial path
import os,sys
cur_path = sys.path[0].split(os.path.sep)
workspace_path = os.path.sep.join(cur_path[:cur_path.index("bestpaycup2020")+1])
base_dir = workspace_path
os.chdir(workspace_path) # 把运行目录强制转移到【工作区】
print(f"把运行目录强制转移到【工作区】{os.getcwd()}")

# %% 导入模块
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
#%% 路径
TRAIN_BASE_PATH     = "./dataset/raw_dataset/trainset/train_base.csv"
TRAIN_OP_PATH       = "./dataset/raw_dataset/trainset/train_op.csv"
TRAIN_TRANS_PATH    = "./dataset/raw_dataset/trainset/train_trans.csv"

TEST_A_BASE_PATH      = "./dataset/raw_dataset/testset/test_a_base.csv"
TEST_A_OP_PATH        = "./dataset/raw_dataset/testset/test_a_op.csv"
TEST_A_TRANS_PATH     = "./dataset/raw_dataset/testset/test_a_trans.csv"
TEST_B_BASE_PATH      = "./dataset/raw_dataset/testset/test_a_base.csv"
TEST_B_OP_PATH        = "./dataset/raw_dataset/testset/test_a_op.csv"
TEST_B_TRANS_PATH     = "./dataset/raw_dataset/testset/test_a_trans.csv"

SAMPLE_BASE_PATH    = "./dataset/raw_dataset/sample_trainset/sample_base.csv"
SAMPLE_OP_PATH      = "./dataset/raw_dataset/sample_trainset/sample_op.csv"
SAMPLE_TRANS_PATH   = "./dataset/raw_dataset/sample_trainset/sample_trans.csv"

PROCESSED_TRAIN_BASE_PATH = "./dataset/dataset3/trainset/train_base.csv"
PROCESSED_TEST_A_BASE_PATH  = "./dataset/dataset3/testset/test_a_base.csv"
PROCESSED_TEST_B_BASE_PATH  = "./dataset/dataset3/testset/test_b_base.csv"


# %% 数据类型预处理函数：整型，缺失值，去零
def process_base(base_path):
    # TODO: provider, province和city都各有一个缺失值，需要众数填补
    def to_int(entry):
        if type(entry) is str:
            level = re.search("^(category |level )([0-9]+)",entry)
            if level:
                return int(level.group(2))
        return entry

    # 处理表顺序, 删除'user'列
    base_df = pd.read_csv(base_path)
    user = base_df['user'].astype('category')
    base_df = base_df.sort_values(by='user',axis=0)  # 每列都根据user顺序重排，index行号即为user编号
    # base_df = base_df.sort_index(axis=0)
    base_df = base_df.sort_index(axis=1)            # 每行的属性根据属性名的字典顺序重排
    base_df.drop('user',axis=1,inplace=True)
    base_df.insert(loc=0,column='user',value=user)

    # 数据转化为整形
    for e in base_df.columns:
        if e == 'user':
            continue    # 保留'user'列的string格式
        base_df[e] = base_df[e].apply(to_int)

    # 显式处理缺失值(默认隐式众数填补)
    # # sex={-1,0,1}
    # base_df["sex"][base_df["sex"]==0] = -1
    # base_df["sex"].fillna(0,inplace=True)

    # 处理转化为数字后的0项
    for e in ['agreement1','agreement2','agreement3','agreement4']:
        base_df[e][base_df[e]==0] = -1
    for e in ['provider','level','regist_type']:
        base_df[e] = base_df[e]+1

    # 合并service3项
    base_df["service3"][base_df["service3"]==0] = -1
    base_df["service3"][base_df["service3"] != -1] = base_df["service3_level"][base_df["service3_level"].notna()]
    base_df.drop("service3_level",axis=1,inplace=True)    # 删除service3_level列

    # 省市独热编码太费时且比重很小, 故直接删除省市信息
    base_df.drop('city',axis=1,inplace=True)
    base_df.drop('province',axis=1,inplace=True)


    # 隐式处理其余缺失值
    for e in base_df.columns:
        base_df[e].fillna(base_df[e].mode()[0],inplace=True)    

    print(f"{base_path} has shape {base_df.shape} after processing")
    
    return base_df



#%% 数据类型预处理
for base_path,processed_base_path in [
    (TRAIN_BASE_PATH,PROCESSED_TRAIN_BASE_PATH),
    ( TEST_A_BASE_PATH, PROCESSED_TEST_A_BASE_PATH),
    ( TEST_B_BASE_PATH, PROCESSED_TEST_B_BASE_PATH),
    ]:
    
    base_df = process_base(base_path)

    if not os.path.exists(os.path.split(processed_base_path)[0]):
        os.makedirs(os.path.split(processed_base_path)[0])
    with open(processed_base_path,"w") as f:
        base_df.to_csv(f,index=False,line_terminator='\n')

#%% 省市独热编码函数
def process_base_onehot(base_dir, dim1,dim2):
    train_df = pd.read_csv(PROCESSED_TRAIN_BASE_PATH)
    test_df = pd.read_csv(PROCESSED_TEST_BASE_PATH)

    # province
    province = pd.concat([train_df['province'], test_df['province']])
    values_ct_org = province.unique().tolist()
    values_ct = np.array(values_ct_org).reshape(len(values_ct_org), -1)
    enc = OneHotEncoder()
    enc.fit(values_ct)
    onehot = enc.transform(values_ct).toarray()

    pca = PCA(n_components=dim1)
    pca.fit(onehot)
    result = pca.transform(onehot)
    mp = dict(zip(values_ct_org, [code for code in result]))

    newdf_train = pd.DataFrame(columns=['province_'+str(i) for i in range(dim1)])
    for i in range(len(train_df)):
        code = mp[train_df.loc[i, 'province']]
        newdf_train.loc[len(newdf_train)] = code
    train_df = train_df.join(newdf_train)
    train_df = train_df.drop(labels='province', axis=1)

    newdf_test = pd.DataFrame(columns=['province_'+str(i) for i in range(dim1)])
    for i in range(len(test_df)):
        code = mp[test_df.loc[i, 'province']]
        newdf_test.loc[len(newdf_test)] = code
    test_df = test_df.join(newdf_test)
    test_df = test_df.drop(labels='province', axis=1)

    # city
    city = pd.concat([train_df['city'], test_df['city']])
    values_ct_org = city.unique().tolist()
    values_ct = np.array(values_ct_org).reshape(len(values_ct_org), -1)
    enc = OneHotEncoder()
    enc.fit(values_ct)
    onehot = enc.transform(values_ct).toarray()

    pca = PCA(n_components=dim2)
    pca.fit(onehot)
    result = pca.transform(onehot)
    mp = dict(zip(values_ct_org, [code for code in result]))

    newdf_train = pd.DataFrame(columns=['city_'+str(i) for i in range(dim2)])
    for i in range(len(train_df)):
        code = mp[train_df.loc[i, 'city']]
        newdf_train.loc[len(newdf_train)] = code
    train_df = train_df.join(newdf_train)
    train_df = train_df.drop(labels='city', axis=1)

    newdf_test = pd.DataFrame(columns=['city_'+str(i) for i in range(dim2)])
    for i in range(len(test_df)):
        code = mp[test_df.loc[i, 'city']]
        newdf_test.loc[len(newdf_test)] = code
    test_df = test_df.join(newdf_test)
    test_df = test_df.drop(labels='city', axis=1)

    train_df.to_csv(PROCESSED_TRAIN_BASE_PATH, index=False)
    test_df.to_csv(PROCESSED_TEST_BASE_PATH, index=False)

# %% 省市独热编码
# process_base_onehot(os.getcwd(), dim1=8,dim2=78) # set to 8 and 64 for faster training



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
