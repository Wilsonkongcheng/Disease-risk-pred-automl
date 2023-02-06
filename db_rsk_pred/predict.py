import argparse

import pandas as pd
import xgboost as xgb
# from params import paramsP
from sklearn.metrics import accuracy_score
from lightgbm import LGBMClassifier
import eli5
from db_rsk_pred.reader.db import *
from db_rsk_pred.reader.db import DB
from db_rsk_pred.preprocess.preprocess import *
from db_rsk_pred.util.util import init_logger
import joblib
from db_rsk_pred.serve.load_model import *

LOGGER = init_logger()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data",default='./data/test_data.csv')
    parser.add_argument("-c","--cfg",default='cfg.ini')
    parser.add_argument("-m","--model",default='model.pkl')
    args = parser.parse_args()
    cp = args.cfg
    cfg = config_from_ini(open(cp,'rt',encoding='utf-8'),read_from_file=True)
    cols = cfg.source.cols 

    cols = [c.strip()for c in cols.split(',') if len(c.strip())>0]
    #
    # tgt = cfg.source.tgt

    data = pd.read_csv(f'{args.data}')
    processor = PreProcessor()

    
    data,col_mapping = processor.process(data)
    cols = [col_mapping[c] for c in cols if c!=cfg.source.id]
    
    model = Model(args.model)
    
    preds =  model.model.predict_proba(data[cols])[:,1]
    # LOGGER.info('test accuracy:%s',accuracy_score(data[tgt],preds))
    
    data['Pred'] = preds
    LOGGER.info('sample of prediction results')
    print(data.head())
    data.to_csv('test_result.csv')
    LOGGER.info('test results saved to local disk')
    # db =DB(cfg.db.host,cfg.db.user,cfg.db.password,cfg.source.table,cfg.source.cols,cfg.source.tgt)
    # data = db.write_result(data)
    # LOGGER.info('test results saved to db_rsk_preds')
