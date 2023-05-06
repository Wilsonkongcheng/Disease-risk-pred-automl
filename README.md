# Disease Risk Pred with Auto-ML

---

## Requirement
- **python>=3.8**
- pip

## Install
- Clone Repo
```bash
git clone -b dev https://codeup.aliyun.com/gupo/gpra/dzs_rsk_pred_automl.git 
```
- Install Packages

  Install all required packages from `setup.py` with following simply run:
```bash
cd dzs_rsk_pred_automl
pip install .
```
## Config
Create a configuration file(`.ini`)which contains necessary information including Database,
Pre-process,Model params etc.See the example in `cfg_sample.ini`.
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
## Pre-process
Create `preprpcess.py` to write some pre-process function,it can create  new columns and apply manually functions to the the old columns to genetate processed data as following,more
detail see the demo in `lung_preproc.py`.
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
After this,be sure to add this file's absolute path into `[preprocess] proc_func_path` in the confing file`.ini`  .

## Train
Running following scrips will happen two stage.Firstly, Fetch origin data from remote database configured in `.ini` and then split into train and test dataset to store with csv files.
Secondly,train and optimized a lightgbm classifier model automatically with above dataset  and then save to local disk.
```bash
python main.py train -c 'your .ini path'
```
The trained model default saved in relative path './model.json'. You can specified manually path as folloing:
```bash
python main.py train -c 'your .ini path' --save 'your dir root'
```
The trained model can also be logged and store into your **mlflow** platform. Firstly configuration in `main.py`as following:
```python
mlflow.set_tracking_uri("your server addr")  
mlflow.set_experiment("experiment name")
```
And then add `-ml` as following:
```bash
python main.py train -c 'your .ini path' -ml
```
Go to your provided server addr, you can see the results of params, metrics, model info etc.

## Prediction
Use your trained model to predict risk probability and output explanation with shap value from default test data(./data/test_data.csv) as folloing:
```bash
python main.py pred
```
The result  saved in `./data/full_result.csv`,and it's also write to the database with your manually configuration in `.ini  [target]`

If you want to use your own dataset to prediction,just add `-pd` as following:
```bash
python main.py pred -pd 'your test_data.csv root'
```
