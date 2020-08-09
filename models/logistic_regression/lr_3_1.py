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

PROCESSED_TRAIN_BASE_PATH = "./dataset/dataset3/trainset/train_base.csv"
PROCESSED_TEST_BASE_PATH  = "./dataset/dataset3/testset/test_a_base.csv"

test_rate = 0.2
#%% 建立数据集
class BaseFactory():
    class Base(Dataset):
        def __init__(self,base,label) -> None:
            assert len(base)==len(label)
            self.base =   base.reset_index(drop=True).drop('user',axis=1)
            self.label = label.reset_index(drop=True).drop('user',axis=1)

        def __len__(self):
            return len(self.base)
        
        def __getitem__(self, index: int):
            base_x = self.base.iloc[index].to_numpy()
            base_y = self.label.iloc[index].loc['label']
            return base_x, base_y


    def __init__(self,base_csv_file,label_file,test_rate):
        self.base   = pd.read_csv(base_csv_file).sort_values(by='user').reset_index(drop=True)
        self.label  =    pd.read_csv(label_file).sort_values(by='user').reset_index(drop=True)
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

base_factory = BaseFactory(PROCESSED_TRAIN_BASE_PATH,LABEL_PATH,test_rate)
train_base,test_base = base_factory.build_dataset()

# %% 建立网络模型

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.bn0      = nn.BatchNorm1d(128)
        self.all_fc1  = nn.Linear(128,256)
        self.bn1      = nn.BatchNorm1d(256)
        self.all_fc2  = nn.Linear(256,2)

    def forward(self,x):
        x = self.bn0(x)
        x = self.all_fc1(x)
        x = torch.relu(x)
        x = self.bn1(x)
        x = self.all_fc2(x)
        return x

net=Net()
print(net)
# %% 测试模型的数据通路
rand_input = torch.randn(10,128)
out=net(rand_input)
print(out.shape)
net.zero_grad()
out.backward(torch.ones_like(out))

# %% 超参
NUM_EPOCH       = 5
BATCH_SIZE      = 10
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

        # 损失函数
        loss_classify = criterion(outputs,labels)

        # 反向传播
        optimizer.zero_grad()
        loss_classify.backward()
        optimizer.step()

        # 记录状态信息
        running_loss[0] += loss_classify.item()
        loss_list[0].append(loss_classify.item())
        
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
# plt.plot(loss_list[1],'-y',label='balance')
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
