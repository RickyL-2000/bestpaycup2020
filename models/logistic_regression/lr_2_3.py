#%% 切换运行路径
import os,sys
cur_path = sys.path[0].split(os.path.sep)
workspace_path = os.path.sep.join(cur_path[:cur_path.index("bestpaycup2020")+1])
base_dir = workspace_path
os.chdir(workspace_path) # 把运行目录强制转移到【工作区】
print(f"把运行目录强制转移到【工作区】{os.getcwd()}")
#%% 导入模块
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
#%% 定义路径

LABEL_PATH          = "./dataset/raw_dataset/trainset/train_label.csv"

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

# num_data        = 47782
# NUM_REMAIN_TEST = int(0.2 * num_data)
test_rate = 0.2
#%% 建立数据集
class BaseFactory():
    class Base(Dataset):
        def __init__(self,base,label) -> None:
            assert len(base)==len(label)
            self.base =   base.reset_index(drop=True)
            self.label = label.reset_index(drop=True)

        def __len__(self):
            return len(self.base)
        
        def __getitem__(self, index: int):
            base_x = self.base.iloc[index].to_numpy()
            base_y = self.label.iloc[index].loc['label']
            return base_x, base_y


    def __init__(self,base_csv_file,label_file,test_rate):
        self.base = pd.read_csv(base_csv_file).sort_values(by='user').reset_index(drop=True)
        self.label = pd.read_csv(label_file).sort_values(by='user').reset_index(drop=True)
        self.balance()
        self.base_train,self.base_test, self.label_train,self.label_test = train_test_split(
            self.balance_base,self.balance_label,
            stratify=self.balance_label['label'],
            test_size=test_rate,
            shuffle=True,
            random_state = 42
        )

    def balance(self):
        pos_label = self.label[self.label['label']==1]
        pos_index = pos_label.index    # 正样本index
        pos_base = self.base.iloc[pos_index]

        neg_label = self.label[self.label['label']==0]
        neg_label = neg_label.sample(n=len(pos_index))
        neg_index = neg_label.index
        neg_base = self.base.iloc[neg_index]

        self.balance_base = pos_base.append(neg_base)
        self.balance_label = pos_label.append(neg_label)

        self.balance_base =  pd.concat([pos_base,neg_base],axis=0,ignore_index=True)
        self.balance_label = pd.concat([pos_label,neg_label],axis=0,ignore_index=True)

    def build_dataset(self):
        return  self.Base(self.base_train,self.label_train),self.Base(self.base_test,self.label_test)

    # def __len__(self):
    #     return len(self.base)
    
    # def __getitem__(self, index: int):
    #     base_x = self.balance_base.iloc[index].to_numpy()
    #     base_y = self.balance_label.iloc[index].loc['label']
    #     return base_x, base_y


base_factory = BaseFactory(PROCESSED_TRAIN_BASE_PATH,LABEL_PATH,test_rate)
train_base,test_base = base_factory.build_dataset()

# test_base = Base(PROCESSED_TRAIN_BASE_PATH,LABEL_PATH,train=False)


# %% 建立网络模型

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()

        # self.fc1 = nn.Linear(42,10)         # 暂时弃用

        self.bn0 = nn.BatchNorm1d(124)
        self.province_fc1 = nn.Linear(31,2)
        self.city_fc1 = nn.Linear(50,4)
        
        self.bn1 = nn.BatchNorm1d(48)
        self.all_fc1 = nn.Linear(48,256)

        self.bn2 = nn.BatchNorm1d(256)
        # self.all_fc2 = nn.Linear(256,128)

        # self.bn3 = nn.BatchNorm1d(128)
        # self.all_fc3 = nn.Linear(128,32)   

        # self.bn4 = nn.BatchNorm1d(32)
        # self.all_fc4 = nn.Linear(32,2)   # 输出两个量，高信用得分和低信用得分(意义不明)

        self.all_fc5 = nn.Linear(256,2)

    def forward(self,x):
        # x[user,base_info,province_onehot,city_pca]
        x = self.bn0(x)
        _base = x[:,1:43]         # 42 columns
        _province = x[:,43:43+31] # 30 columns
        _city = x[:,43+31:]       # 50 columns
        
        # 不能强制填零来提升速度，模型会混乱
            # _zero = torch.zeros_like(_base,requires_grad=False)[:,:,:64-_base.shape[2]]
            # _base = torch.cat((_base,_zero),dim=2)
            # _zero = torch.zeros_like(_province,requires_grad=False)[:,:,:256-_province.shape[2]]
            # _province = torch.cat((_province,_zero),dim=2)
            # _zero = torch.zeros_like(_city,requires_grad=False)[:,:,:64-_city.shape[2]]
            # _city = torch.cat((_city,_zero),dim=2)

        ####_base = self.fc1(_base)
        _province =torch.relu(self.province_fc1(_province))
        _city =    torch.relu(self.city_fc1(_city))

        x = torch.cat((_base,_city,_province),dim=1)

        # x = self.bn1(x)
        x = self.all_fc1(x)

        x = torch.relu(x)
        # x = self.bn2(x)
        # x = self.all_fc2(x)

        # x = torch.relu(x)
        # x = self.bn3(x)
        # x = self.all_fc3(x)

        # x = torch.relu(x)
        # x = self.bn4(x)
        # x = self.all_fc4(x) # 两个得分

        x = self.all_fc5(x)

        return x

net=Net()
print(net)
# %% 测试模型的数据通路
rand_input = torch.randn(10,124)
out=net(rand_input)
print(out.shape)
net.zero_grad()
out.backward(torch.ones_like(out))

# %% 超参
num_data        = 47782
NUM_EPOCH       = 5
BATCH_SIZE      = 50
PRINT_PER_BATCH = 20

LR              = 1e-3
# SGD
MOMENTUM        = 0.9
# Adam
BETAS           = (0.9, 0.999)
EPS             = 1e-08
WEIGHT_DECAY    = 0
# balance_rate
BALANCE_RATE    = 0.5

#%% 数据加载类，损失函数和优化器
train_base_dl = DataLoader(train_base,batch_size=BATCH_SIZE,shuffle=True,drop_last=True)

# criterion = nn.MSELoss()     # 输出一个值时，用MSE平方收敛更快
criterion = nn.CrossEntropyLoss()   # 输出两个得分时，用交叉熵函数

# optimizer = optim.SGD(net.parameters(),lr=LR,momentum=MOMENTUM)
optimizer = optim.Adam(net.parameters(),lr=LR, betas=BETAS, eps=EPS, weight_decay=WEIGHT_DECAY)

#%% 模型初始化，准备记录变量
net.train() # fix BatchNorm and Dropout layer
for params in net.parameters():
    init.normal_(params, mean=0, std=0.1)
net.zero_grad()
loss_list=[[],[]]
roc_auc_list=[]

#%% 训练模型
for epoch in range(NUM_EPOCH):
    running_loss = [0,0]
    for i,data in enumerate(train_base_dl,0):
        # 处理输入数据的数据类型
        inputs,labels = data
        inputs = inputs.float()
        labels = labels.long()

        # 正向经过网络
        outputs=net(inputs)

        # 反向传播和损失函数
        optimizer.zero_grad()

        loss_classify = criterion(outputs,labels)
        loss_classify.backward()

        # loss_classify.backward(retain_graph=True)
        # pred = torch.softmax(outputs, dim=1)[:,1].round()
        # loss_balance = (0.5-pred.mean()).abs()*BALANCE_RATE/0.5
        #     # 这一项为了保证输出结果尽可能均值0.2465363（label的均值），以此提高roc_auc
        #     # loss = loss_classify+loss_balance
        #     # loss = loss_classify
        # outputs.backward(torch.randn_like(outputs)*loss_balance)    # 给随机扰动

        optimizer.step()

        # 记录状态信息
        running_loss[0] += loss_classify.item()
        # running_loss[1] += loss_balance.item()
        loss_list[0].append(loss_classify.item())
        # loss_list[1].append(loss_balance.item())
        
        # 打印状态信息
        if i % PRINT_PER_BATCH == PRINT_PER_BATCH-1:    # 每一定批次打印一次
            pred = torch.softmax(outputs.detach(), dim=1)[:,1].round()
            try:
                roc_auc_list.append(
                    roc_auc_score(
                        y_true  = labels.detach().numpy(),
                        y_score =   pred.detach().numpy(),
                        ))
            except:
                roc_auc_list.append(0.5)
            print(  f'[{epoch + 1}, {i + 1:5d}] '
                    f'loss_classify: {running_loss[0] / PRINT_PER_BATCH:.4f} '
                    f'loss_balance: {running_loss[1] / PRINT_PER_BATCH:.4f} '
                    f'roc_auc: {roc_auc_list[-1]:.4f}')
            running_loss = [0 for x in range(len(running_loss))]


# %% 损失函数，准确率可视化
plt.plot(loss_list[0],'-r',label='classify')
plt.plot(loss_list[1],'-y',label='balance')
plt.plot(roc_auc_list,'-b',label='roc_auc')
plt.legend()
plt.title(f'{__file__.split(os.sep)[-1]}')
plt.show()


#%% 测试集校验
test_base_dl = DataLoader(test_base,batch_size=BATCH_SIZE,drop_last=True)

net.eval() # fix BatchNorm and Dropout layer
running_loss=0
with torch.no_grad():
    for i,data in enumerate(test_base_dl,0):
        
        inputs,labels = data
        inputs = inputs.float()
        labels = labels.long()

        # 正向经过网络
        outputs=net(inputs)

        # 损失函数
        loss = criterion(outputs,labels)
        running_loss+=loss
        # roc_auc
        pred = torch.softmax(outputs.detach(), dim=1).numpy()[:,1].round()   # 输出的第二个看作高信用得分
        roc_auc = roc_auc_score(
                y_true  =  labels.detach().view(-1).tolist(),
                y_score = pred,
                )
    print(f'loss: {running_loss*BATCH_SIZE/len(test_base):.9f}  roc_auc:{roc_auc:.9f}')
    print(pred.mean())

# %%
