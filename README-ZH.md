# 疾病风险预测自动化拟合框架

---

## 依赖要求
- **python>=3.8**
- pip

## 安装
- 克隆仓库
```bash
git clone -b dev https://codeup.aliyun.com/gupo/gpra/dzs_rsk_pred_automl.git 
```
- 安装相关依赖

简单运行以下脚本即可从`setup.py`安装相关的依赖
```bash
cd dzs_rsk_pred_automl
pip install .
```
## Config
创建一个`.ini`的配置文件，该文件包含了一些必要的配置信息包括：数据库、预处理、模型参数等。详见例子`cfg_sample.ini`。
```ini
[db] # database
host =  'your host'
user = 'your username'
password = 'your passwd'
port = 'your port'
db = 'your db'
use_unicode = True
charset = utf8mb4
[source]  # origin data readed from table 
table = 'your reading table'
cols  = 'features and label,split by ',''
id = 'unique identifier such as id_no'
tgt = 'label'

[target] # prediction result write to result
table = 'your writing table'
user_id = 'unique identifier such as id_no'
pd_cols = 'writing columns in csv file,split with ',''
sql_cols = 'writing colunmes in table relatived with above pd_cols,'split with ',''

[monotonic_constraint] # Tree-based model parameter, see detail in lightgbm documentation
pos= 'positive constraint features in tree-based model,split by',''
neg= 'negtive constraint features in tree-based model,split by',''

#  user's proc_func.py absolute path    see example in db_rsk_pred/preprocess/Proc_demo.py
[preprocess]  
proc_func_path = 'your proc_func.py absolute path'

[rsk_factor] # columns in risk factor(features) with the value of True or False
factor = 'some colums,split by ',''
```
## 预处理
创建`preprpcess.py`文件，在改文件中写入预处理相关函数，该文件可以创建新的字段名称并在旧特征上应用其中的用户自定义函数来生成处理后的数据，如下所示，具体细节请详见本例`lung_preproc.py`
```python
import pandas as pd
class MyProc:
    def __init__(self):
        # 需要更改的字段
        self.column = ['old_column_name', 'new_column_name']

        # 字段应用的转换函数

    def smk_qty_m(self, x):
        if pd.isna(x):
            return 0
```
然后，将该文件的绝对路径添加到配置文件`.ini`中的`[preprocess] proc_func_path`
## 训练
运行以下脚本将会产生两个阶段。第一阶段，根据配置文件中的信息从数据库中获取原始数据并自动划分为训练集和测试集并存储为csv文件；第二阶段，使用上述的训练数据集自动训练并且调优一个lightgbm的clasiffier模型并保存与本地
```bash
python main.py train -c 'your .ini path'
```
训练后的模型自动保存在相对路径`./model.json`，用户也可以用如下方式自定义存储路径：
```bash
python main.py train -c 'your .ini path' --save 'your dir root'
```
训练好的模型可以自动被记录和存储在**mlflow**平台中。第一步，先修改`main.py`中的如下部分:
```python
mlflow.set_tracking_uri("your server addr")  
mlflow.set_experiment("experiment name")
```
然后增加`-ml`参数:
```bash
python main.py train -c 'your .ini path' -ml
```
进入mlflow的地址，你可以看到参数、指标、模型信息等结果

## 预测
对默认的测试集数据(./data/test_data.csv) 使用训练好的模型进行疾病风险度预测，并且输出可解释性的shap value：
```bash
python main.py pred
```
结果保存于`./data/full_result.csv`，同时会写入到用户在`.ini  [target]`自定义的数据表中

如果你想使用自己的数据集进行预测，只需要在运行脚本中加入`-pd` 参数：
```bash
python main.py pred -pd 'your test_data.csv root'
```
