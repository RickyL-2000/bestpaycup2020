# bestpaycup2020
第二届翼支付杯大数据建模大赛-信用风险用户识别

## 文件架构

.
├── LICENSE
├── README.md
└── dataset
    ├── dataset1
    │   ├── readme.md
    │   ├── testset
    │   └── trainset
    └── raw_dataset
        ├── testset
        │   ├── submit_example.csv
        │   ├── test_a_base.csv
        │   ├── test_a_op.csv
        │   └── test_a_trans.csv
        └── trainset
            ├── train_base.csv
            ├── train_label.csv
            ├── train_op.csv
            ├── train_trans.csv
            └── 数据字典.doc

## Todo List

[ ] 完善todo list



## 任务

信用风险用户识别

## 数据

备注：报名参赛或加入队伍后，可获取数据下载权限。

数据集共分为训练数据集、初赛测试数据集。

训练数据集中的文件包含：

 	 黑白样本标签(0/1)：train_label.csv，共47782条数据。

 	 脱敏后的用户基础信息：train_base.csv，共47782条数据。

​	  脱敏后的用户操作信息：train_op.csv，共2774988条数据。

 	 脱敏后的用户交易信息：train_trans.csv，共591266条数据。

初赛测试数据集包含：

​	  脱敏后的用户基础信息：test_a_base.csv，共24315条数据。

​	  脱敏后的用户操作信息：test_a_op.csv，共1109432条数据。

​	  脱敏后的用户交易信息：test_a_trans.csv，共147870条数据

详见数据字典

| 字段名            | 字段说明（数据经过脱敏处理）                                 |
| ----------------- | ------------------------------------------------------------ |
| user              | 样本编号，e.g., Train_00000、Train_00001...                  |
| sex               | 性别，编码后取值为：category 0、category1                    |
| age               | 年龄，处理后仅保留大小关系，为某一区间的整数                 |
| provider          | 运营商类型，编码后取值为：category 0、category 1...          |
| level             | 用户等级，编码后取值为：category 0、category 1...            |
| verified          | 是否实名，编码后取值为：category 0、category1                |
| using_time        | 使用时长，处理后仅保留大小关系，为某一区间的整数             |
| regist_type       | 注册类型，编码后取值为：category 0、category 1...            |
| card_a_cnt        | a类型卡的数量，处理后仅保留大小关系，为某一区间的整数        |
| card_b_cnt        | b类型卡的数量，处理后仅保留大小关系，为某一区间的整数        |
| card_c_cnt        | c类型卡的数量，处理后仅保留大小关系，为某一区间的整数        |
| card_d_cnt        | d类型卡的数量，处理后仅保留大小关系，为某一区间的整数        |
| op1_cnt           | 某类型1操作数量，处理后仅保留大小关系，为某一区间的整数      |
| op2_cnt           | 某类型2操作数量，处理后仅保留大小关系，为某一区间的整数      |
| service1_cnt      | 某业务1产生数量，处理后仅保留大小关系，为某一区间的整数      |
| service1_amt      | 某业务1产生金额，处理后仅保留大小关系，为某一区间的整数      |
| service2_cnt      | 某业务2产生数量，处理后仅保留大小关系，为某一区间的整数      |
| agreement_total   | 开通协议数量，处理后仅保留大小关系，为某一区间的整数         |
| agreement1        | 是否开通协议1，编码后取值为：category 0、category1           |
| agreement2        | 是否开通协议2，编码后取值为：category 0、category1           |
| agreement3        | 是否开通协议3，编码后取值为：category 0、category1           |
| agreement4        | 是否开通协议4，编码后取值为：category 0、category1           |
| acc_count         | 账号数量，处理后仅保留大小关系，为某一区间的整数             |
| login_cnt_period1 | 某段时期1的登录次数，处理后仅保留大小关系，为某一区间的整数  |
| login_cnt_period2 | 某段时期2的登录次数，处理后仅保留大小关系，为某一区间的整数  |
| ip_cnt            | 某段时期登录ip个数，处理后仅保留大小关系，为某一区间的整数   |
| login_cnt_avg     | 某段时期登录次数均值，处理后仅保留大小关系，为某一区间的整数 |
| login_days_cnt    | 某段时期登录天数，处理后仅保留大小关系，为某一区间的整数     |
| province          | 省份，处理成类别编码                                         |
| city              | 城市，处理成类别编码                                         |
| balance           | 余额等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| balance_avg       | 近某段时期余额均值等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| balance1          | 类型1余额等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| balance1_avg      | 近某段时期类型1余额均值等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| balance2          | 类型2余额等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| balance2_avg      | 近某段时期类型2余额均值等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| service3          | 是否服务3用户，编码后取值为：category 0、category1           |
| service3_level    | 服务3等级，编码后取值为：category 0、category1...            |
| product1_amount   | 产品1金额等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| product2_amount   | 产品2金额等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| product3_amount   | 产品3金额等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| product4_amount   | 产品4金额等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| product5_amount   | 产品5金额等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| product6_amount   | 产品6金额等级，处理成保留大小关系的类别编码：level 1、level2... 例如：level 2 > level 1 |
| product7_cnt      | 产品7申请次数，处理后仅保留大小关系，为某一区间的整数        |
| product7_fail_cnt | 产品7申请失败次数，处理后仅保留大小关系，为某一区间的整数    |

**操作信息**

| 字段名    | 字段说明（数据经过脱敏处理）                                 |
| --------- | ------------------------------------------------------------ |
| user      | 样本编号，e.g., Train_00000、Train_00001...                  |
| op_type   | 操作类型编码，处理成类别编码                                 |
| op_mode   | 操作模式编码，处理成类别编码                                 |
| op_device | 操作设备编码，处理成类别编码                                 |
| ip        | 设备ip编码，处理成类别编码                                   |
| net_type  | 网络类型编码，处理成类别编码                                 |
| channel   | 渠道类型编码，处理成类别编码                                 |
| ip_3      | 设备ip前三位编码，处理成类别编码                             |
| tm_diff   | 距离某起始时间点的时间间隔，处理成如下格式。例如： 9 days 09:02:45.000000000，表示距离某起始时间点9天9小时2分钟45秒 |

**交易信息**

| 字段名     | 字段说明（数据经过脱敏处理）                                 |
| ---------- | ------------------------------------------------------------ |
| user       | 样本编号，e.g., Train_00000、Train_00001...                  |
| platform   | 平台类型编码，处理成类别编码                                 |
| tunnel_in  | 来源类型编码，处理成类别编码                                 |
| tunnel_out | 去向类型编码，处理成类别编码                                 |
| amount     | 交易金额，处理后仅保留大小关系，为某一区间的整数             |
| type1      | 交易类型1编码，处理成类别编码                                |
| type2      | 交易类型2编码，处理成类别编码                                |
| ip         | 设备ip编码，处理成类别编码                                   |
| ip_3       | 设备ip前三位编码，处理成类别编码                             |
| tm_diff    | 距离某起始时间点的时间间隔，处理成如下格式。例如： 9 days 09:02:45.000000000，表示距离某起始时间点9天9小时2分钟45秒 |



## 评分标准

本次比赛采用auc对结果进行评分，评分代码示例：

``` python
from sklearn.metrics import roc_auc_score
y_true = [1, 0, 1, 1]
y_pred = [1,0, 1, 0]
score = roc_auc_score(y_true, y_pred)
```





