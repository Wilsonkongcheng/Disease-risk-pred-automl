
import argparse
from functools import partial

import optuna
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# from .model import param_setting
from db_rsk_pred.reader.db import *
from db_rsk_pred.preprocess.preprocess import *
from db_rsk_pred.preprocess.preprocess import PreProcessor
from db_rsk_pred.util.util import init_logger
from db_rsk_pred.reader.read_data_from_db import read_db
import joblib
import os 



logger = init_logger()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default='../data/train_data.csv')
    parser.add_argument("-c", "--cfg", default='cfg.ini')
    parser.add_argument('-s','--source',default='csv')
    parser.add_argument('--save',default='./')
    
    args = parser.parse_args()
    # print(args.data)
    cp = args.cfg
    cfg = config_from_ini(
        open(cp, 'rt', encoding='utf-8'), read_from_file=True)
    
    cols = cfg.source.cols
    cols = [c.strip()for c in cols.split(',') if len(c.strip()) > 0]
    tgt = cfg.source.tgt

    pos_constraints = cfg.monotonic_constraint.pos
    # if not all([c for c in pos_constraints if c not in cols]):
    #     raise ValueError('constraint features must be a subset of the features required to train the model')
    pos_constraints = [c.strip()for c in pos_constraints.split(',') if len(c.strip()) > 0]
    # monotone_constraints = [ 1 for _ in pos_constraints]

    if args.source =='csv':
        data = pd.read_csv(f'{args.data}')
    else:
        data = read_db(cfg)

   
    
    processor = PreProcessor()
    data, col_mapping = processor.process(data)
    cols = [col_mapping[c] for c in cols if c != cfg.source.id]  # remove user_id then col_name mapping
    
    print(cols) # ['年龄_年', '性别', '运动', '吸烟', '饮酒', 'BMI', '收缩压', '糖调节受损', '高血压', '动脉粥样硬化性', '多囊卵巢综合征', '类固醇糖尿病', '二型糖尿病家族史', '高密度脂蛋白', '甘油三酯']
    
#     train_data = data.sample(frac=0.8)
#     eval_data = data[~data.index.isin(train_data.index)]
#     objective = partial(param_setting.optuna_objective, train_data,
#                         eval_data, cols, tgt)
#     study = optuna.create_study(direction='maximize')
#     study.optimize(objective, n_trials=2,timeout=2,n_jobs=-1,show_progress_bar =True)
#     best_trial = study.best_trial
#     params =best_trial.params
#     # params = best_trial['params']
#     logger.info('trail ends, now retrain lightgbm with the optimal hyper-parameters')
#
#     best_model = LGBMClassifier(
#         **params)
#
#     best_model.fit(data[cols], data[tgt])
#     joblib.dump(best_model,os.path.join(args.save,'model.pkl'))
#     # best_model.booster.save_model(args.save)
# # save model
#
#     # best_model = best_trial.user_attrs['model']
#     # best_model.save(args.save)
#     logger.info('training finished ! model has been saved to %s',args.save)