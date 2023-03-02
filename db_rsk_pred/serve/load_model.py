
import argparse

import pandas as pd
import xgboost as xgb
# from params import paramsP
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
# import eli5
from db_rsk_pred.reader.db import *
from db_rsk_pred.reader.db import DB
# from db_rsk_pred.preprocess.preprocess import *
from db_rsk_pred.util.util import init_logger
import joblib

LOGGER = init_logger()

# TypeError
class Model:

    def __init__(self, model_path) -> None:
        # self.model = joblib.load('lgb.pkl')
        try:
            self.model = joblib.load(model_path)
        except:
            raise FileNotFoundError(''' the model file doesn't exist''' )
        self.feature_names = self.model.booster_.feature_name
        print(self.feature_names)
    
    def predict(self,data,feature_importance=False):
        if isinstance(data,list) and isinstance(data[0],dict):
            features = list(data[0].keys())
            if not all([True for f in self.feature_names if f in features]):
                raise ValueError('some features to run the model are missing in the data')
            else:
                data = pd.DataFrame(data)[self.feature_names]
        elif  isinstance(data,pd.DataFrame): 
            try :
                data = data[self.feature_names]
            except:
                 raise ValueError('some features to run the model are missing in the data')       
        else:
            raise TypeError('data type has to be list of dict or pandas dataframe')
        preds =self.model.predict_proba(data)
        return preds
        # if feature_importance:
        #     explan=eli5.explain_prediction(self.model,X.iloc[idx],targets=[1]) ##对预测结果进行解释
        #     explan=eli5.format_as_dict(explan)## 把解释转换成dataframe
        #     return preds,explan
        # else:
        #     return preds
        
        
        



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("-d", "--data",default='test_data.csv')
#     parser.add_argument("-c","--cfg",default='cfg.ini')
#     parser.add_argument("-m","--model",default='model.json')
#     args = parser.parse_args()
#     cp = args.cfg
#     cfg = config_from_ini(open(cp,'rt',encoding='utf-8'),read_from_file=True)
#     cols = cfg.source.cols 

#     cols = [c.strip()for c in cols.split(',') if len(c.strip())>0]
#     #
#     tgt = cfg.source.tgt

#     data = pd.read_csv(f'./data/{args.data}')
#     processor = PreProcessor()

    
#     data,col_mapping = processor.process(data)
#     cols = [col_mapping[c] for c in cols if c!=cfg.source.id]
    
#     xgb_model = xgb.XGBClassifier()
#     xgb_model.load_model(f'./model/{args.model}')
    
#     preds =  xgb_model.predict_proba(data[cols])[:,1]
#     # LOGGER.info('test accuracy:%s',accuracy_score(data[tgt],preds))
    
#     data['Pred'] = preds
#     data.to_csv('test_result.csv')
#     LOGGER.info('test results saved to local disk')
#     db =DB(cfg.db.host,cfg.db.user,cfg.db.password,cfg.source.table,cfg.source.cols,cfg.source.tgt)
#     data = db.write_result(data)
#     LOGGER.info('test results saved to db_rsk_preds')

    

    

    